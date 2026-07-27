"""One live differentiable call per registered autograd boundary, by name.

Phase 9 asks the same question of every ``torch.autograd.Function`` in the
package - does a second-order request fail loudly, before any partial
second-order result, naming the owner - and that question needs a working
FIRST-order call at each boundary to ask it through. The boundaries do not
share a fixture: a two-way join needs frozen legs, a waveform needs rows and a
spec, a sensor weight needs a geometry and a plan, and a frontend needs a noise
realisation. Building all six inside the test that consumes them would have put
two hundred lines of setup in front of six one-line assertions and would have
made the next consumer - a backward launch budget, a tape ledger - build them a
second time.

Each entry is a :class:`Boundary`: an owner name, a leaf, and a callable that
turns the leaf into a real scalar loss through the production entry point. The
callable takes the leaf so a caller can hand it a dual, a ``requires_grad``
clone, or a plain tensor and get the same computation.

Everything here is deliberately small - two rows, two chirps, thirty-two
samples. The question is a contract question, and a large frame answers it no
better while costing every consumer the time.

The numerical settings are copied from the operator-level AD tests that own
them rather than re-derived, and the two that matter are called out at their
definitions: the pulse must be an LFM, and the carrier rate must be non-zero.
"""

from __future__ import annotations

import torch

from . import join_fixture as fx


REFERENCE_FREQUENCY_HZ = 77.0e9

#: Two rows, one pair segment. The delays and rates are the operator tests'
#: own numbers so a failure here and a failure there describe the same point.
DELAYS = (2.4683743e-8, 3.1712819e-8)
RATES = (8.0055e-9, -3.1e-9)
WEIGHTS = (0.6 - 0.3j, -0.2 + 0.45j)


class Boundary:
    """One autograd boundary: a name, a leaf, and a loss over it."""

    __slots__ = ("name", "owner", "leaf", "loss")

    def __init__(self, name: str, owner: str, leaf: torch.Tensor, loss) -> None:
        self.name = name
        self.owner = owner
        self.leaf = leaf
        self.loss = loss


def _rows(device: str = "cuda"):
    tau = torch.tensor(DELAYS, dtype=torch.float32, device=device)
    rate = torch.tensor(RATES, dtype=torch.float32, device=device)
    weight = torch.tensor(WEIGHTS, dtype=torch.complex64, device=device)
    offsets = torch.tensor([0, len(DELAYS)], dtype=torch.int64, device=device)
    return tau, rate, weight, offsets


def _fmcw(device: str = "cuda") -> Boundary:
    from witwin.radar.synthesis.contracts import FmcwBeatSpec
    from witwin.radar.synthesis.fmcw_beat import synthesize_beat_rows

    # carrier_rate_hz is non-zero on purpose: it is what makes the delay-rate
    # derivative differ from the delay derivative times the chirp time, and a
    # spec with it zeroed would exercise a simpler backward than production's.
    spec = FmcwBeatSpec(
        num_samples=32,
        num_chirps=2,
        sample_period_s=1.0 / 4.4e6,
        chirp_period_s=65.0e-6,
        slope_hz_per_s=60.012e12,
        t_start_s=6.0e-6,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
        carrier_hz=0.0,
        carrier_rate_hz=REFERENCE_FREQUENCY_HZ,
    )
    tau, rate, weight, offsets = _rows(device)

    def loss(leaf: torch.Tensor) -> torch.Tensor:
        return synthesize_beat_rows(leaf, rate, weight, offsets, spec).abs().square().sum()

    return Boundary("fmcw", "synthesis.fmcw_beat", tau, loss)


def _ofdm(device: str = "cuda") -> Boundary:
    from witwin.radar.synthesis.contracts import OfdmCfrSpec
    from witwin.radar.synthesis.ofdm_cfr import synthesize_cfr_rows

    spec = OfdmCfrSpec(
        num_subcarriers=8,
        num_symbols=2,
        subcarrier_spacing_hz=120.0e3,
        cyclic_prefix_s=2.0e-6,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
        max_expected_delay_s=1.0e-6,
        carrier_hz=0.0,
        carrier_rate_hz=REFERENCE_FREQUENCY_HZ,
    )
    tau, rate, weight, offsets = _rows(device)

    def loss(leaf: torch.Tensor) -> torch.Tensor:
        return synthesize_cfr_rows(leaf, rate, weight, offsets, spec).abs().square().sum()

    return Boundary("ofdm", "synthesis.ofdm_cfr", tau, loss)


def _pulsed(device: str = "cuda") -> Boundary:
    from witwin.radar.synthesis.contracts import PulsedEchoSpec
    from witwin.radar.synthesis.pulsed_echo import synthesize_echo_rows

    # LFM, not rectangular. A rectangular pulse's dependence on the delay is
    # entirely through its support test, so its almost-everywhere delay
    # derivative is EXACTLY zero - a real property of the model, and a
    # higher-order test built on it would be asserting nothing.
    spec = PulsedEchoSpec(
        num_pulses=2,
        num_samples=32,
        sample_period_s=2.0e-9,
        pri_s=1.0e-6,
        range_gate_start_s=0.0,
        pulse_kind="lfm",
        pulse_width_s=2.0e-8,
        bandwidth_hz=5.0e8,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
        max_expected_delay_rate=0.0,
        carrier_hz=0.0,
        carrier_rate_hz=REFERENCE_FREQUENCY_HZ,
    )
    tau, rate, weight, offsets = _rows(device)
    zero_rate = torch.zeros_like(rate)

    def loss(leaf: torch.Tensor) -> torch.Tensor:
        return (
            synthesize_echo_rows(leaf, zero_rate, weight, offsets, spec)
            .abs()
            .square()
            .sum()
        )

    return Boundary("pulsed", "synthesis.pulsed_echo", tau, loss)


def _two_way(device: str = "cuda") -> Boundary:
    from witwin.radar.paths.two_way import TwoWayComposer
    from witwin.radar.scattering import ScalarRcsResponse

    sources = (0,)
    sites = (10, 11)
    sinks = (20,)
    inbound = fx.frozen_leg(fx.leg_rows(sources, sites, (0,)), device=device)
    outbound = fx.frozen_leg(fx.leg_rows(sites, sinks, (0,)), device=device)
    composer = TwoWayComposer.freeze(
        inbound,
        outbound,
        torch.tensor(sites, dtype=torch.int64, device=device),
        radar_source_ids=sources,
        radar_sink_ids=sinks,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
    )
    tau_in, rate_in, c_in = fx.payload(composer.inbound_row_count, seed=101, device=device)
    tau_out, rate_out, c_out = fx.payload(
        composer.outbound_row_count, seed=102, device=device
    )

    def leg(tau, coefficient, rate):
        return fx.leg_batch(
            tau.to(torch.float32),
            coefficient.to(torch.complex64),
            rate=rate.to(torch.float32),
        )

    response = ScalarRcsResponse.from_values(
        1.4, 0.35, device=device, requires_grad=False
    )

    def loss(leaf: torch.Tensor) -> torch.Tensor:
        composed = composer.compose(
            leg(leaf, c_in, rate_in), leg(tau_out, c_out, rate_out), response
        )
        return (
            composed.complex_transfer_ref.abs().square().sum()
            + (composed.total_delay_s.to(torch.float64) * 1.0e8).square().sum()
        )

    return Boundary("two_way", "paths.two_way", tau_in.to(torch.float32), loss)


def _sensor_weight(device: str = "cuda") -> Boundary:
    from witwin.radar.sensors import ROW_KIND_VIA, evaluate_sensor_weights
    from witwin.radar.sensors.contracts import AntennaPatternSpec
    from witwin.radar.sensors.weights import (
        SensorWeightGeometry,
        SensorWeightModes,
        SensorWeightPlan,
    )

    num_tx, num_rx, rows = 1, 1, 2
    geometry = SensorWeightGeometry(
        num_tx=num_tx,
        num_rx=num_rx,
        tx_velocity=torch.zeros((num_tx, 3), dtype=torch.float32, device=device),
        rx_velocity=torch.zeros((num_rx, 3), dtype=torch.float32, device=device),
        site_velocity=torch.zeros((rows, 3), dtype=torch.float32, device=device),
        fixed_length_m=torch.ones(rows, dtype=torch.float32, device=device),
        tx_index=torch.zeros(rows, dtype=torch.int64, device=device),
        rx_index=torch.zeros(rows, dtype=torch.int64, device=device),
        row_kind=torch.full((rows,), ROW_KIND_VIA, dtype=torch.int32, device=device),
        normals=torch.tensor(
            [[0.0, 0.0, 1.0]] * rows, dtype=torch.float32, device=device
        ),
        # x polarized, which is transverse to a boresight along -z. A
        # polarization ALONG the propagation direction projects to zero and
        # would give a zero weight that looks exactly like a working fixture.
        pol_tx=torch.tensor([[1.0, 0.0, 0.0]] * num_tx, dtype=torch.float32, device=device),
        pol_rx=torch.tensor([[1.0, 0.0, 0.0]] * num_rx, dtype=torch.float32, device=device),
        local_axes=torch.eye(3, dtype=torch.float32, device=device),
    )
    # A real half-wave dipole table rather than zeros: a zero pattern gives a
    # zero weight, and a higher-order test on a zero is a test of nothing.
    plan = SensorWeightPlan.build(
        AntennaPatternSpec.half_wave_dipole(),
        modes=SensorWeightModes(
            spreading=True, tx_power=False, legacy_real_polarization=False
        ),
        wavelength_m=3.894e-3,
        device=device,
    )
    tx_pos = torch.zeros((num_tx, 3), dtype=torch.float32, device=device)
    rx_pos = torch.zeros((num_rx, 3), dtype=torch.float32, device=device)
    # Boresight is -z: the pattern angles are ``atan2(v_x, -v_z)`` and
    # ``atan2(v_y, -v_z)``, and outside the tabulated support the dipole gain is
    # exactly zero by design. A site placed off boresight therefore produces a
    # zero weight, a zero loss and a zero gradient - a fixture that runs and
    # measures nothing.
    site_out = torch.tensor(
        [[0.4, 0.1, -5.0], [-0.3, 0.2, -6.0]], dtype=torch.float32, device=device
    )
    intensity = torch.ones(rows, dtype=torch.float32, device=device)
    weight = torch.ones(rows, dtype=torch.complex64, device=device)
    site_in = site_out.clone()

    def loss(leaf: torch.Tensor) -> torch.Tensor:
        result = evaluate_sensor_weights(
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            site_in=leaf,
            site_out=site_out,
            intensity=intensity,
            weight=weight,
            geometry=geometry,
            plan=plan,
        )
        return result.weight.abs().square().sum()

    return Boundary("sensor_weight", "sensors.weights", site_in, loss)


def _frontend(device: str = "cuda") -> Boundary:
    from witwin.radar.frontend import (
        AgcSpec,
        FrontendChain,
        FrontendSpec,
        LnaSpec,
        NoiseSpec,
        PortSpec,
        SeedSpec,
    )

    # No ADC. The quantizer is on the far side of the wall and refuses a
    # derivative outright, so a higher-order question about the frontend has to
    # be asked of the stages that DO publish one.
    chain = FrontendChain(
        FrontendSpec(
            port=PortSpec(50.0),
            noise=NoiseSpec(
                noise_figure_db=3.0,
                bandwidth_hz=1e6,
                phase_noise_dbc_per_hz=-80.0,
                phase_offset_hz=1e5,
                phase_sample_rate_hz=1e6,
            ),
            lna=LnaSpec(gain_db=10.0),
            agc=AgcSpec(target_rms=1e-3, mode="global"),
            seed=SeedSpec(5),
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(31)
    signal = (
        torch.complex(
            torch.randn(256, generator=generator), torch.randn(256, generator=generator)
        )
        .to(torch.complex64)
        .to(device)
        * 1e-4
    )

    def loss(leaf: torch.Tensor) -> torch.Tensor:
        return chain.apply(leaf).signal.abs().square().sum()

    return Boundary("frontend", "frontend.chain", signal, loss)


def _aspect(device: str = "cuda") -> Boundary:
    """The aspect response, driven from its INBOUND direction table.

    The leaf is ``dir_in`` rather than a response parameter because that is the
    edge both the tape ledger and the higher-order rule are about: the two
    direction tables are the legs' own aliased tensors, so this boundary is the
    one place a geometry gradient enters the scatter response.
    """

    from witwin.radar.paths.two_way import TwoWayComposer
    from witwin.radar.scattering import AspectScatterResponse

    sources, sites, sinks = (0,), (10, 11), (20,)
    inbound = fx.frozen_leg(fx.leg_rows(sources, sites, (0,)), device=device)
    outbound = fx.frozen_leg(fx.leg_rows(sites, sinks, (0,)), device=device)
    composer = TwoWayComposer.freeze(
        inbound,
        outbound,
        torch.tensor(sites, dtype=torch.int64, device=device),
        radar_source_ids=sources,
        radar_sink_ids=sinks,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
    )

    def unit(rows: int, seed: int, sign: float) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        raw = torch.rand(rows, 3, generator=generator, dtype=torch.float32)
        # Well inside the illuminated cone. A direction on the clamp boundary
        # has an exactly zero lobe derivative on one side, which would make a
        # first-order assertion here pass on a number that means nothing.
        vectors = torch.stack(
            [0.7 + 0.3 * raw[:, 0], 0.5 * (raw[:, 1] - 0.5), 0.5 * (raw[:, 2] - 0.5)],
            dim=1,
        )
        vectors = sign * vectors / torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        return vectors.to(device)

    rows_in = composer.inbound_row_count
    rows_out = composer.outbound_row_count
    dir_in = unit(rows_in, 71, -1.0)
    dir_out = unit(rows_out, 72, 1.0)
    site_count = composer.site_count
    response = AspectScatterResponse(
        axis=unit(site_count, 73, 1.0),
        amplitude=torch.tensor(
            [1.3 + 0.4 * index for index in range(site_count)],
            dtype=torch.float32,
            device=device,
        ),
        phase_rad=torch.tensor(
            [0.35 + 0.25 * index for index in range(site_count)],
            dtype=torch.float32,
            device=device,
        ),
        exponent=2.0,
        coherent_interval_s=1.0e-3,
    )
    tau_in, _, c_in = fx.payload(rows_in, seed=201, device=device)
    tau_out, _, c_out = fx.payload(rows_out, seed=202, device=device)
    row_valid = torch.ones(composer.path_count, dtype=torch.int32, device=device)

    def loss(leaf: torch.Tensor) -> torch.Tensor:
        inbound_batch = fx.leg_batch(
            tau_in.to(torch.float32), c_in.to(torch.complex64), direction=leaf
        )
        outbound_batch = fx.leg_batch(
            tau_out.to(torch.float32), c_out.to(torch.complex64), direction=dir_out
        )
        s_re, s_im = response.evaluate_rows(
            composer, inbound_batch, outbound_batch, row_valid
        )
        return (s_re.square() + s_im.square()).sum()

    return Boundary("aspect", "scattering.aspect", dir_in, loss)


#: The seven boundaries the Phase-9 higher-order rejection and the tape/budget
#: ledger are both asserted at, by name. Built on demand rather than eagerly:
#: each one costs a CUDA allocation and a consumer usually wants one.
#:
#: There were nine until Phase 11 deleted the ``dirichlet`` and ``mimo_linear``
#: contexts with their route. There are seven names and EIGHT tape owners:
#: ``frontend`` runs two contexts - the noise phase and the AGC gain - in one
#: call. Every
#: ``torch.autograd.Function`` in the package is reachable from this table,
#: which is what lets ``test_phase9_backward_budget.py`` assert that the ledger
#: enumerates all of them rather than the ones someone remembered.
BUILDERS = {
    "two_way": _two_way,
    "aspect": _aspect,
    "fmcw": _fmcw,
    "ofdm": _ofdm,
    "pulsed": _pulsed,
    "sensor_weight": _sensor_weight,
    "frontend": _frontend,
}

BOUNDARY_NAMES = tuple(BUILDERS)


def boundary(name: str, *, device: str = "cuda") -> Boundary:
    """The named boundary, with no default.

    A ``dict.get`` with a fallback here would turn a typo in a parametrize list
    into a second copy of another boundary's test, silently.
    """

    return BUILDERS[name](device)


__all__ = ["BOUNDARY_NAMES", "BUILDERS", "Boundary", "boundary"]
