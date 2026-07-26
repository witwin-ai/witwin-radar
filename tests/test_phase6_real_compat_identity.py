"""The analytic bridge that makes "the real path is the complex special case"
checkable without a GPU.

``dirichlet.cu::path_response`` and ``fmcw_beat.cu`` look like two different
models. They are not: the Dirichlet spectrum is algebraically the exact
zero-padded ``N_fft``-point DFT of the beat sample sequence, at
``carrier_hz = fc``, ``carrier_rate_hz = 0``, ``tau_rate = 0``. Pinning that
identity in float64 on the CPU is what lets the Phase-6 refactor widen the
Dirichlet family to complex weights without the real baseline becoming
unfalsifiable: the reference is a closed form, not the kernel's own output.

Everything here is float64 and CPU-only by construction. The identity is
algebra, not a kernel property, and a test that needed CUDA would not run in
the default suite - which is exactly where the compatibility criterion has to
be checked, because it is the criterion most likely to be quietly broken by an
unrelated change.

The closed forms, verbatim:

    dirichlet_kernel(x, n) = [sin((n + 0.5) x) / sin(x/2)] * exp(-j n x)
    path_response(d)[bin]  = dirichlet_kernel(x, n) * exp(+j phi0)
      tau  = 2 d / c0,  n = (N - 1)/2
      phi0 = 2 pi (fc tau + S tau (t0 - tau/2))
      k0   = d * k0_per_meter = f_beat * N_fft / fs
      x    = 2 pi (bin - k0) / N_fft

    beat[m] = exp(+j 2 pi [fc tau + S tau (t0 - tau/2) + S tau m T_s])
    DFT_{N_fft}(zero-padded beat)[bin] == path_response(d)[bin]

The ``n = (N - 1)/2`` in the Dirichlet kernel is not a convention: it is
exactly the half-sample group delay of an ``N``-point rectangular window, which
is why the geometric sum closes. A pad factor changes ``N_fft`` and therefore
``x``, but never ``n``.
"""

from __future__ import annotations

import math

import pytest
import torch


C0 = 299792458.0

# The probe geometry from the Phase-6 physics survey, kept so the measured
# errors quoted in the stage report are reproducible.
FC_HZ = 77.0e9
SLOPE_HZ_PER_S = 60.0e6 / 1.0e-6
SAMPLE_RATE_HZ = 5.0e6
NUM_SAMPLES = 256
PAD_FACTOR = 16
T_START_S = 6.0e-6
DISTANCE_M = 3.7


def _dirichlet_kernel(x: torch.Tensor, n: float) -> torch.Tensor:
    """``[sin((n + 0.5) x) / sin(x/2)] * exp(-j n x)``, float64.

    The removable singularity at ``x -> 0`` is handled by the limit ``2n + 1``,
    exactly as the kernel does. The test geometries below never sit on it, but
    leaving it out would make the reference disagree with the kernel at a bin
    that a future test might land on.
    """

    half = 0.5 * x
    scale = torch.where(
        half.abs() < 1e-12,
        torch.full_like(x, 2.0 * n + 1.0),
        torch.sin((n + 0.5) * x) / torch.sin(half),
    )
    return scale.to(torch.complex128) * torch.exp(-1j * n * x.to(torch.complex128))


def _path_response(
    distance_m: float,
    *,
    num_samples: int,
    n_fft: int,
    sample_rate_hz: float,
    slope_hz_per_s: float,
    fc_hz: float,
    t_start_s: float,
) -> torch.Tensor:
    n = (num_samples - 1) / 2
    k0_per_meter = (slope_hz_per_s * 2.0 / C0) * n_fft / sample_rate_hz
    tau = 2.0 * distance_m / C0
    phi0 = 2.0 * math.pi * (
        fc_hz * tau + slope_hz_per_s * tau * (t_start_s - 0.5 * tau)
    )
    k0 = distance_m * k0_per_meter
    bins = torch.arange(n_fft, dtype=torch.float64)
    x = 2.0 * math.pi * (bins - k0) / n_fft
    return _dirichlet_kernel(x, n) * torch.exp(torch.tensor(1j * phi0, dtype=torch.complex128))


def _beat_samples(
    distance_m: float,
    *,
    num_samples: int,
    sample_rate_hz: float,
    slope_hz_per_s: float,
    fc_hz: float,
    t_start_s: float,
) -> torch.Tensor:
    """One chirp's fast-time samples, ``tau_rate = 0``, kernel-owned carrier."""

    tau = 2.0 * distance_m / C0
    sample_period_s = 1.0 / sample_rate_hz
    t_m = torch.arange(num_samples, dtype=torch.float64) * sample_period_s
    cycles = (
        fc_hz * tau
        + slope_hz_per_s * tau * (t_start_s - 0.5 * tau)
        + slope_hz_per_s * tau * t_m
    )
    return torch.exp(2j * math.pi * cycles.to(torch.complex128))


def _padded_dft(samples: torch.Tensor, n_fft: int) -> torch.Tensor:
    padded = torch.zeros(n_fft, dtype=torch.complex128)
    padded[: samples.shape[0]] = samples
    return torch.fft.fft(padded)


def _max_relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    scale = reference.abs().max()
    return float(((actual - reference).abs() / scale).max())


# ---------------------------------------------------------------------------
# T0.1 - the DFT identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pad_factor", [PAD_FACTOR, 1], ids=["padded", "unpadded"])
def test_the_dirichlet_spectrum_is_the_dft_of_the_beat_samples(pad_factor):
    n_fft = NUM_SAMPLES * pad_factor
    response = _path_response(
        DISTANCE_M,
        num_samples=NUM_SAMPLES,
        n_fft=n_fft,
        sample_rate_hz=SAMPLE_RATE_HZ,
        slope_hz_per_s=SLOPE_HZ_PER_S,
        fc_hz=FC_HZ,
        t_start_s=T_START_S,
    )
    samples = _beat_samples(
        DISTANCE_M,
        num_samples=NUM_SAMPLES,
        sample_rate_hz=SAMPLE_RATE_HZ,
        slope_hz_per_s=SLOPE_HZ_PER_S,
        fc_hz=FC_HZ,
        t_start_s=T_START_S,
    )
    transformed = _padded_dft(samples, n_fft)

    error = _max_relative_error(transformed, response)
    assert error < 1e-11, error


def test_the_identity_survives_a_superposition_of_targets():
    """Both sides are linear in the weight, so a multi-target check is not a
    stronger statement about the phase - it is a statement that neither side
    smuggled in a per-target normalisation."""

    n_fft = NUM_SAMPLES * PAD_FACTOR
    distances = [1.0, 3.7, 11.25]
    amplitudes = [0.5, -0.25, 1.0]

    response = sum(
        amplitude
        * _path_response(
            distance,
            num_samples=NUM_SAMPLES,
            n_fft=n_fft,
            sample_rate_hz=SAMPLE_RATE_HZ,
            slope_hz_per_s=SLOPE_HZ_PER_S,
            fc_hz=FC_HZ,
            t_start_s=T_START_S,
        )
        for distance, amplitude in zip(distances, amplitudes)
    )
    samples = sum(
        amplitude
        * _beat_samples(
            distance,
            num_samples=NUM_SAMPLES,
            sample_rate_hz=SAMPLE_RATE_HZ,
            slope_hz_per_s=SLOPE_HZ_PER_S,
            fc_hz=FC_HZ,
            t_start_s=T_START_S,
        )
        for distance, amplitude in zip(distances, amplitudes)
    )

    error = _max_relative_error(_padded_dft(samples, n_fft), response)
    assert error < 1e-11, error


def test_range_recovered_from_the_beat_frequency_is_the_authored_range():
    """``f_beat = S * tau_rt`` has no factor of two; the two lives in ``tau``.

    Writing the round trip into the delay and not into the slope is the whole
    reason a bistatic path works at all, and it is the single easiest place to
    be self-consistently 2x wrong.
    """

    for distance in (0.25, 3.7, 42.0):
        tau = 2.0 * distance / C0
        f_beat = SLOPE_HZ_PER_S * tau
        recovered = C0 * f_beat / (2.0 * SLOPE_HZ_PER_S)
        assert recovered == pytest.approx(distance, rel=1e-12)


def test_the_beat_bin_matches_the_dirichlet_k0():
    """``k0 = d * k0_per_meter`` and ``b = f_beat * N_fft / fs`` are one number.

    Two independent expressions for the same quantity, in two different files,
    is how the Dirichlet and beat families drift apart.
    """

    n_fft = NUM_SAMPLES * PAD_FACTOR
    k0_per_meter = (SLOPE_HZ_PER_S * 2.0 / C0) * n_fft / SAMPLE_RATE_HZ
    for distance in (0.25, 3.7, 42.0):
        f_beat = SLOPE_HZ_PER_S * (2.0 * distance / C0)
        assert distance * k0_per_meter == pytest.approx(
            f_beat * n_fft / SAMPLE_RATE_HZ, rel=1e-12
        )


# ---------------------------------------------------------------------------
# T0.2 - the MIMO time bridge
# ---------------------------------------------------------------------------


def test_the_mimo_ifft_is_the_beat_sample_sequence():
    """``DirichletSolver.mimo(..., freq_domain=False)`` IS the beat cube.

    The MIMO configuration sets ``n_fft = num_bins = adc_samples``, so the
    forward transform is a plain DFT and its inverse returns the samples
    exactly, with no window and no scale factor. That is the statement the
    Phase-6 FMCW owner has to preserve when it takes over the real path.

    Recorded non-analogue, and the reason this runs with the range-loss update
    OFF: the MIMO kernel optionally rescales ``amp *= dist0/dist`` across
    chirps, and ``fmcw_beat.cu`` holds the weight constant across chirps. There
    is no counterpart, so a compatibility test either disables it or the new
    owner reproduces it. It is not negligible and must not be assumed away.
    """

    n_fft = NUM_SAMPLES
    distances = [1.0, 3.7, 11.25]
    amplitudes = [0.5, -0.25, 1.0]

    spectrum = sum(
        amplitude
        * _path_response(
            distance,
            num_samples=NUM_SAMPLES,
            n_fft=n_fft,
            sample_rate_hz=SAMPLE_RATE_HZ,
            slope_hz_per_s=SLOPE_HZ_PER_S,
            fc_hz=FC_HZ,
            t_start_s=T_START_S,
        )
        for distance, amplitude in zip(distances, amplitudes)
    )
    samples = sum(
        amplitude
        * _beat_samples(
            distance,
            num_samples=NUM_SAMPLES,
            sample_rate_hz=SAMPLE_RATE_HZ,
            slope_hz_per_s=SLOPE_HZ_PER_S,
            fc_hz=FC_HZ,
            t_start_s=T_START_S,
        )
        for distance, amplitude in zip(distances, amplitudes)
    )

    error = _max_relative_error(torch.fft.ifft(spectrum), samples)
    assert error < 1e-10, error


def test_the_two_carrier_homes_give_the_same_slow_time_slope():
    """The kernel-owned and weight-owned carriers differ by a constant only.

    ``d(cycles)/d(t_c) = tau_rate * (fc + S*(t0 - tau + t_m))`` for both
    ``(carrier_hz, carrier_rate_hz) = (fc, 0)`` and ``(0, fc)``. This is why
    rule R3 in the synthesis contract chooses WHICH parameter must equal
    ``f_ref`` from the weight's provenance instead of demanding a particular
    one: a kernel-owned carrier multiplies the full ``tau(t)`` and therefore
    already walks.
    """

    tau_rt = 2.0 * DISTANCE_M / C0
    tau_rate = 2.0 * 12.0 / C0  # 12 m/s receding, monostatic
    chirp_period_s = 60.0e-6
    t_m = 137.0 / SAMPLE_RATE_HZ

    def cycles(carrier_hz: float, carrier_rate_hz: float, chirp: int) -> float:
        drift = tau_rate * chirp * chirp_period_s
        tau = tau_rt + drift
        return (
            carrier_hz * tau
            + carrier_rate_hz * drift
            + SLOPE_HZ_PER_S * tau * (T_START_S - 0.5 * tau)
            + SLOPE_HZ_PER_S * tau * t_m
        )

    kernel_step = cycles(FC_HZ, 0.0, 1) - cycles(FC_HZ, 0.0, 0)
    weight_step = cycles(0.0, FC_HZ, 1) - cycles(0.0, FC_HZ, 0)
    assert kernel_step == pytest.approx(weight_step, rel=1e-12)

    # And the one that drops the rate term is wrong by orders of magnitude,
    # asserted two-sided so a future "simplification" cannot pass by accident.
    ramp_only_step = cycles(0.0, 0.0, 1) - cycles(0.0, 0.0, 0)
    assert abs(kernel_step / ramp_only_step) > 20.0
