"""float64 pure-Torch CPU oracle for the whole Phase-4 chain.

TEST-ONLY. CLAUDE.md permits a CPU/Torch reference implementation only under
``tests/``; a production module that imported this would be introducing a Torch
numerical backend, and ``tests/test_phase4_import_boundary.py`` rejects it.

Why an oracle and not finite differences alone: the production loss is float32
and its magnitude is dominated by terms that are almost unrelated to the
parameter under test, so a naive central difference on the production chain
subtracts two nearly equal float32 numbers and can return an exactly zero
derivative that looks like a real answer. The oracle is float64 and is itself
FD-validated in float64, where the conditioning is fine; production AD is then
compared against the oracle.

The oracle reimplements, independently:

* free-space leg transfer  -  distance, delay, and the Channel-convention
  coefficient ``sqrt(P) * lambda / (4 pi d) * exp(-j 2 pi f tau)``;
* the two-way composition and the scalar target response;
* the Channel-to-beat conjugation;
* the FMCW beat sample sum.
"""

from __future__ import annotations

import math

import torch

from . import phase4_geometry as geo


def leg_transfer(
    source: torch.Tensor,
    sink: torch.Tensor,
    power_w: float,
    frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Delay and complex free-space transfer of one line-of-sight leg.

    Channel phasor convention: ``exp(-j k d)`` under ``exp(+j 2 pi f t)``.
    Endpoint polarizations are transverse to every leg in the fixture plane, so
    the polarization projection is exactly one and does not appear.
    """

    distance = torch.linalg.norm(sink - source)
    delay = distance / geo.C0_M_PER_S
    wavelength = geo.C0_M_PER_S / frequency_hz
    amplitude = math.sqrt(power_w) * wavelength / (4.0 * math.pi * distance)
    phase = -2.0 * math.pi * frequency_hz * delay
    return delay, amplitude * torch.exp(1j * phase.to(torch.complex128))


def target_response(
    amplitude: torch.Tensor, phase_rad: torch.Tensor
) -> torch.Tensor:
    """Scalar target response, authored in the Channel convention."""

    return amplitude.to(torch.complex128) * torch.exp(
        -1j * phase_rad.to(torch.complex128)
    )


def round_trip(
    tx: torch.Tensor,
    site: torch.Tensor,
    rx: torch.Tensor,
    amplitude: torch.Tensor,
    phase_rad: torch.Tensor,
    *,
    power_w: float = geo.TX_POWER_W,
    frequency_hz: float = geo.REFERENCE_FREQUENCY_HZ,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Composed round-trip delay and transfer for one scatter site.

    Each leg independently applies its own ``sqrt(P) * lambda / (4 pi d)``, so
    with unit power on both the site is a 1 W isotropic re-radiator. That is a
    declared spike simplification, not the radar equation; R-ADR-002 records it
    and the magnitude test asserts it verbatim so it cannot change silently.
    """

    tau_in, c_in = leg_transfer(tx, site, power_w, frequency_hz)
    tau_out, c_out = leg_transfer(site, rx, power_w, frequency_hz)
    transfer = c_out * target_response(amplitude, phase_rad) * c_in
    return tau_in + tau_out, transfer


def round_trip_delay_rate(
    tx: torch.Tensor,
    site: torch.Tensor,
    rx: torch.Tensor,
    site_velocity: torch.Tensor,
) -> torch.Tensor:
    """Two-way ``d(tau)/dt`` from the site velocity projected on both legs."""

    unit_in = (site - tx) / torch.linalg.norm(site - tx)
    unit_out = (site - rx) / torch.linalg.norm(site - rx)
    return (
        torch.dot(unit_in, site_velocity) + torch.dot(unit_out, site_velocity)
    ) / geo.C0_M_PER_S


def beat_samples(
    total_delay_s: torch.Tensor,
    delay_rate: torch.Tensor,
    beat_weight: torch.Tensor,
    pair_offsets: torch.Tensor,
    spec,
    segment_tx_index=None,
) -> torch.Tensor:
    """The beat sum, in the BEAT convention, evaluated in float64.

    Mirrors ``fmcw_beat.cu`` exactly:

        cycles = carrier * tau + carrier_rate * (tau - tau_rt)
               + slope * tau * (t_start - 0.5 * tau)
               + slope * tau * t_m,   tau = tau_rt + tau_rate * t_slot
        t_slot(c, p) = (c * num_tx + tx[p]) * chirp_period

    ``carrier_rate`` applies the carrier to the intra-frame delay CHANGE only.
    On the production path the absolute carrier phase lives in the weight,
    frozen at ``tau_rt``, and this term is the Doppler the weight cannot carry.

    ``t_slot`` is TDM slot time, not chirp time: with several transmitters
    sharing the frame, the sensor pairs driven by transmitter ``tx`` sit a whole
    chirp period later in slow time than those driven by ``tx - 1``.
    ``segment_tx_index`` defaults to all-zero, which with ``spec.num_tx == 1``
    is the single-transmitter case ``t_slot == c * T_chirp``.
    """

    offsets = [int(v) for v in pair_offsets.tolist()]
    num_segments = len(offsets) - 1
    num_tx = int(getattr(spec, "num_tx", 1))
    if segment_tx_index is None:
        tx_of_segment = [0] * num_segments
    else:
        tx_of_segment = [int(v) for v in segment_tx_index]
    chirps = torch.arange(spec.num_chirps, dtype=torch.float64)
    samples = torch.arange(spec.num_samples, dtype=torch.float64)
    t_m = samples * spec.sample_period_s

    out = torch.zeros(
        (spec.num_chirps, num_segments, spec.num_samples), dtype=torch.complex128
    )
    for segment in range(num_segments):
        t_c = (chirps * num_tx + tx_of_segment[segment]) * spec.chirp_period_s
        for row in range(offsets[segment], offsets[segment + 1]):
            drift = delay_rate[row].to(torch.float64) * t_c.reshape(-1, 1)
            tau = total_delay_s[row].to(torch.float64) + drift
            cycles = (
                spec.carrier_hz * tau
                + spec.carrier_rate_hz * drift
                + spec.slope_hz_per_s * tau * (spec.t_start_s - 0.5 * tau)
                + spec.slope_hz_per_s * tau * t_m.reshape(1, -1)
            )
            phasor = torch.exp(2j * math.pi * cycles.to(torch.complex128))
            out[:, segment, :] = out[:, segment, :] + beat_weight[row].to(
                torch.complex128
            ) * phasor
    return out


def channel_to_beat(transfer_ref: torch.Tensor) -> torch.Tensor:
    """The single conjugation, reimplemented independently."""

    return torch.conj(transfer_ref)


def synthesize(
    total_delay_s: torch.Tensor,
    delay_rate: torch.Tensor,
    transfer_ref: torch.Tensor,
    pair_offsets: torch.Tensor,
    spec,
    segment_tx_index=None,
) -> torch.Tensor:
    return beat_samples(
        total_delay_s,
        delay_rate,
        channel_to_beat(transfer_ref),
        pair_offsets,
        spec,
        segment_tx_index,
    )


def radar_loss(iq: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Phase-sensitive squared-error loss, accumulated in float64.

    A magnitude-only loss would pass even with the conjugation inverted, and
    ``.abs()`` would put a kink at zero exactly where a finite difference wants
    smoothness. This has neither problem.
    """

    delta = iq.to(torch.complex128) - reference.to(torch.complex128)
    return (delta.real**2 + delta.imag**2).sum()


def full_chain_loss(
    tx: torch.Tensor,
    site: torch.Tensor,
    rx: torch.Tensor,
    amplitude: torch.Tensor,
    phase_rad: torch.Tensor,
    reference_iq: torch.Tensor,
    spec,
    *,
    delay_rate: torch.Tensor | None = None,
) -> torch.Tensor:
    """The whole chain, end to end, in float64: the loss-level oracle."""

    total_delay, transfer = round_trip(tx, site, rx, amplitude, phase_rad)
    rate = torch.zeros(1, dtype=torch.float64) if delay_rate is None else delay_rate
    iq = synthesize(
        total_delay.reshape(1),
        rate.reshape(1),
        transfer.reshape(1),
        torch.tensor([0, 1], dtype=torch.int64),
        spec,
    )
    return radar_loss(iq, reference_iq)


__all__ = [
    "beat_samples",
    "channel_to_beat",
    "full_chain_loss",
    "leg_transfer",
    "radar_loss",
    "round_trip",
    "round_trip_delay_rate",
    "synthesize",
    "target_response",
]
