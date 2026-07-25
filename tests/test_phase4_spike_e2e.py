"""The Phase-4 stop/go gate.

One differentiable scatter site, two propagation legs through the Channel
consumer, a scalar target response, native FMCW synthesis, and a scalar loss.
Every gradient is checked AT THE LOSS, not per stage, because a per-stage check
cannot see a tape that was severed between stages.

The oracle is the float64 pure-Torch chain in ``tests/support/reference_chain``.
It is validated by float64 finite differences here, and production AD is then
compared against it. A float32 finite difference on the production loss is not
a usable oracle: the loss is dominated by terms that are almost independent of
the parameter under test, so the difference of two nearly equal float32 numbers
can return a confident zero.

Provisional dependency note (R-ADR-008): Channel and Core are consumed from
source checkouts, not pinned release wheels.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import fd  # noqa: E402
from support import phase4_geometry as geo  # noqa: E402
from support import reference_chain as ref  # noqa: E402
from support import spike_driver as drv  # noqa: E402


pytestmark = pytest.mark.gpu


POSITION_STEP_M = 1.0e-5
PARAMETER_STEP = {"amplitude": 1.0e-1, "phase_rad": 1.0e-5}
# Below this magnitude a gradient component is structurally zero rather than a
# small number whose ratio means anything. The z components of a planar,
# z-polarized fixture are exactly this case.
ZERO_FLOOR = 1.0e-6


@pytest.fixture(scope="module")
def spike():
    return drv.Phase4Spike()


@pytest.fixture(scope="module")
def spec():
    return drv.make_spec(num_chirps=4)


@pytest.fixture(scope="module")
def reference_iq(spec):
    return drv.make_reference_iq(spec)


def _oracle_gradients(reference_iq, spec):
    tx, site, rx = drv.oracle_positions()
    tx = tx.clone().requires_grad_(True)
    site = site.clone().requires_grad_(True)
    rx = rx.clone().requires_grad_(True)
    amplitude = torch.tensor(
        drv.SPIKE_AMPLITUDE, dtype=torch.float64, requires_grad=True
    )
    phase = torch.tensor(
        drv.SPIKE_PHASE_RAD, dtype=torch.float64, requires_grad=True
    )
    loss = ref.full_chain_loss(tx, site, rx, amplitude, phase, reference_iq, spec)
    loss.backward()
    return {
        "tx": tx.grad,
        "site": site.grad,
        "rx": rx.grad,
        "amplitude": amplitude.grad,
        "phase_rad": phase.grad,
        "loss": float(loss.detach()),
    }


def _oracle_loss(reference_iq, spec, **overrides):
    tx, site, rx = drv.oracle_positions()
    values = {
        "tx": tx,
        "site": site,
        "rx": rx,
        "amplitude": torch.tensor(drv.SPIKE_AMPLITUDE, dtype=torch.float64),
        "phase_rad": torch.tensor(drv.SPIKE_PHASE_RAD, dtype=torch.float64),
    }
    values.update(overrides)
    return ref.full_chain_loss(
        values["tx"],
        values["site"],
        values["rx"],
        values["amplitude"],
        values["phase_rad"],
        reference_iq,
        spec,
    )


# --------------------------------------------------------------------------
# Criterion 1: the oracle itself
# --------------------------------------------------------------------------


def test_oracle_matches_float64_finite_differences(reference_iq, spec):
    """Validate the oracle before anything is compared against it."""

    oracle = _oracle_gradients(reference_iq, spec)
    for name in ("tx", "site", "rx"):
        for axis in range(3):
            measured = fd.central_difference(
                lambda value, key=name: _oracle_loss(
                    reference_iq, spec, **{key: value}
                ),
                drv.oracle_positions()[("tx", "site", "rx").index(name)],
                axis,
                POSITION_STEP_M,
            )
            assert (
                fd.relative_error(
                    measured, float(oracle[name][axis]), floor=ZERO_FLOOR
                )
                < 1e-3
            ), (name, axis, measured, float(oracle[name][axis]))

    for name in ("amplitude", "phase_rad"):
        base = torch.tensor(
            drv.SPIKE_AMPLITUDE if name == "amplitude" else drv.SPIKE_PHASE_RAD,
            dtype=torch.float64,
        ).reshape(1)
        measured = fd.central_difference(
            lambda value, key=name: _oracle_loss(
                reference_iq, spec, **{key: value.reshape(())}
            ),
            base,
            0,
            PARAMETER_STEP[name],
        )
        assert (
            fd.relative_error(measured, float(oracle[name]), floor=ZERO_FLOOR) < 1e-3
        ), (name, measured, float(oracle[name]))


# --------------------------------------------------------------------------
# Criterion 1a / 1b: reverse mode at the loss
# --------------------------------------------------------------------------


def test_reverse_mode_loss_gradients_match_the_oracle(spike, spec, reference_iq):
    tx, site, rx = drv.positions(requires_grad=True)
    response = drv.make_response(requires_grad=True)
    loss = spike.loss(
        tx,
        site,
        rx,
        response,
        spec,
        reference_iq,
        ad_mode="vjp",
        include_delay_rate=False,
    )
    loss.backward()

    oracle = _oracle_gradients(reference_iq, spec)
    assert (
        fd.relative_error(float(loss.detach()), oracle["loss"], floor=ZERO_FLOOR)
        < 1e-4
    )

    measured = {
        "tx": tx.grad.reshape(3),
        "site": site.grad.reshape(3),
        "rx": rx.grad.reshape(3),
    }
    for name, gradient in measured.items():
        assert gradient is not None
        for axis in range(3):
            assert (
                fd.relative_error(
                    float(gradient[axis]), float(oracle[name][axis]), floor=ZERO_FLOOR
                )
                < 1e-3
            ), (name, axis)
        # In-plane components carry real signal; the out-of-plane component is
        # exactly zero for this planar, z-polarized fixture, which is a
        # structural fact and is asserted rather than hidden.
        assert abs(float(gradient[0])) > ZERO_FLOOR
        assert abs(float(gradient[1])) > ZERO_FLOOR
        assert abs(float(gradient[2])) <= ZERO_FLOOR

    assert (
        fd.relative_error(
            float(response.amplitude.grad), float(oracle["amplitude"]), floor=ZERO_FLOOR
        )
        < 1e-3
    )
    # The phase gradient is the conjugation witness: inverting the
    # Channel-to-beat conjugation flips its sign.
    assert (
        fd.relative_error(
            float(response.phase_rad.grad), float(oracle["phase_rad"]), floor=1e-12
        )
        < 1e-3
    )
    assert abs(float(response.phase_rad.grad)) > 1e-12


def test_site_gradient_is_the_negated_sum_of_the_endpoint_gradients(
    spike, spec, reference_iq
):
    """A structural identity the two legs must satisfy jointly.

    Translating the whole scene changes nothing, so the three position
    gradients must sum to zero. This catches a leg whose gradient never
    reached the site at all, which a per-parameter tolerance check can miss.
    """

    tx, site, rx = drv.positions(requires_grad=True)
    spike.loss(
        tx,
        site,
        rx,
        drv.make_response(),
        spec,
        reference_iq,
        ad_mode="vjp",
        include_delay_rate=False,
    ).backward()
    torch.testing.assert_close(
        site.grad, -(tx.grad + rx.grad), rtol=1e-4, atol=1e-7
    )


# --------------------------------------------------------------------------
# Criterion 1c: forward mode at the loss
# --------------------------------------------------------------------------


def test_forward_mode_loss_tangent_matches_the_oracle(spike, reference_iq):
    """Directional derivative through the native jvp companion.

    Single chirp and no delay rate, so exactly one dual meaning is live: the
    duals are position perturbations, not velocities.
    """

    spec = drv.make_spec(num_chirps=1)
    reference = drv.make_reference_iq(spec)
    direction = {
        "tx": torch.tensor([[0.3, -0.2, 0.0]], dtype=torch.float32, device="cuda"),
        "site": torch.tensor([[-0.5, 0.4, 0.0]], dtype=torch.float32, device="cuda"),
        "rx": torch.tensor([[0.1, 0.25, 0.0]], dtype=torch.float32, device="cuda"),
    }
    tx, site, rx = drv.positions()
    response = drv.make_response()

    with forward_ad.dual_level():
        loss = spike.loss(
            forward_ad.make_dual(tx, direction["tx"]),
            forward_ad.make_dual(site, direction["site"]),
            forward_ad.make_dual(rx, direction["rx"]),
            response,
            spec,
            reference,
            ad_mode="jvp",
            include_delay_rate=False,
        )
        tangent = forward_ad.unpack_dual(loss).tangent
        assert tangent is not None, "the forward tape did not reach the loss"
        measured = float(tangent)

    oracle = _oracle_gradients(reference, spec)
    expected = sum(
        float(
            torch.dot(
                oracle[name], direction[name].reshape(3).double().cpu()
            )
        )
        for name in ("tx", "site", "rx")
    )
    assert fd.relative_error(measured, expected, floor=ZERO_FLOOR) < 1e-3

    # And the same directional derivative by float64 central difference.
    o_tx, o_site, o_rx = drv.oracle_positions()
    directional = fd.directional_derivative(
        lambda a, b, c: _oracle_loss(reference, spec, tx=a, site=b, rx=c),
        (o_tx, o_site, o_rx),
        tuple(
            direction[name].reshape(3).double().cpu()
            for name in ("tx", "site", "rx")
        ),
        POSITION_STEP_M,
    )
    assert fd.relative_error(measured, directional, floor=ZERO_FLOOR) < 1e-3


# --------------------------------------------------------------------------
# Criterion 1d: Doppler through the ADR-038 forward-only dual
# --------------------------------------------------------------------------


def test_doppler_delay_rate_matches_the_analytic_two_way_projection(spike):
    spec = drv.make_spec(num_chirps=8, carrier_hz=geo.REFERENCE_FREQUENCY_HZ)
    velocity = torch.tensor([[0.0, 12.0, 0.0]], dtype=torch.float32, device="cuda")
    tx, site, rx = drv.positions()
    response = drv.make_response()

    with forward_ad.dual_level():
        composed, inbound, outbound = spike.paths(
            tx,
            forward_ad.make_dual(site, velocity),
            rx,
            response,
            ad_mode="jvp",
        )
        assert inbound.delay_rate is not None
        assert outbound.delay_rate is not None
        rate = composed.delay_rate.clone()
        iq = None
    assert rate is not None

    o_tx, o_site, o_rx = drv.oracle_positions()
    analytic = float(
        ref.round_trip_delay_rate(
            o_tx, o_site, o_rx, velocity.reshape(3).double().cpu()
        )
    )
    measured = float(rate[0])
    # Eight significant digits: this is the Doppler primitive and it has to be
    # exact, not merely close.
    assert fd.relative_error(measured, analytic, floor=1e-18) < 1e-7, (
        measured,
        analytic,
    )

    doppler_hz = -geo.REFERENCE_FREQUENCY_HZ * measured
    assert doppler_hz == pytest.approx(
        -geo.REFERENCE_FREQUENCY_HZ * analytic, rel=1e-6
    )
    assert doppler_hz < 0.0  # a receding site

    # The rate reaches synthesis as a primal and shows up as a slow-time slope.
    from witwin.radar.synthesis.fmcw_beat import synthesize_fmcw_beat

    iq = synthesize_fmcw_beat(composed, spec).cpu().to(torch.complex128)
    steps = iq[1:, 0, 0] * torch.conj(iq[:-1, 0, 0])
    slope = float(torch.angle(steps).mean())
    dphi_dtau = 2.0 * math.pi * (
        geo.REFERENCE_FREQUENCY_HZ
        + spec.slope_hz_per_s * (spec.t_start_s - float(composed.total_delay_s[0]))
    )
    assert slope == pytest.approx(
        dphi_dtau * measured * spec.chirp_period_s, abs=1e-4
    )
    # Positive slow-time slope for a receding site: the documented beat
    # convention, asserted rather than flipped.
    assert slope > 0.0


# --------------------------------------------------------------------------
# Criterion 2: the tape is not detached anywhere
# --------------------------------------------------------------------------


def test_every_stage_output_stays_on_the_tape(spike, spec, reference_iq):
    tx, site, rx = drv.positions(requires_grad=True)
    response = drv.make_response(requires_grad=True)
    composed, inbound, outbound = spike.paths(
        tx, site, rx, response, ad_mode="vjp"
    )
    assert inbound.delay_s.requires_grad
    assert inbound.coefficient.requires_grad
    assert outbound.coefficient.requires_grad
    assert composed.total_delay_s.requires_grad
    assert composed.complex_transfer_ref.requires_grad

    from witwin.radar.synthesis.fmcw_beat import synthesize_fmcw_beat

    iq = synthesize_fmcw_beat(composed, spec)
    assert iq.requires_grad
    assert iq.grad_fn is not None
    loss = drv.radar_loss(iq, reference_iq)
    loss.backward()
    for tensor in (tx, site, rx, response.amplitude, response.phase_rad):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_a_dead_row_contributes_exactly_zero_to_loss_and_gradients(
    spike, spec, reference_iq
):
    """Dead-row semantics, injected.

    A frozen line-of-sight row is replayed as pure free-space transport and is
    never re-tested for visibility, so this route cannot produce a dead row by
    contract. The mask is therefore injected to exercise the semantics the
    contract promises: a dead row is a complete answer contributing zero, not
    an error, and it carries no gradient back to what it was built from.
    """

    from dataclasses import replace

    from witwin.radar.synthesis.fmcw_beat import synthesize_fmcw_beat

    tx, site, rx = drv.positions(requires_grad=True)
    response = drv.make_response(requires_grad=True)
    composed, _, _ = spike.paths(tx, site, rx, response, ad_mode="vjp")
    dead = replace(
        composed,
        row_valid=torch.zeros(
            composed.path_count, dtype=torch.bool, device=composed.device
        ),
    )
    iq = synthesize_fmcw_beat(dead, spec)
    assert torch.count_nonzero(iq) == 0

    loss = drv.radar_loss(iq, reference_iq)
    loss.backward()
    for tensor in (tx, site, rx, response.amplitude, response.phase_rad):
        assert tensor.grad is not None
        assert float(tensor.grad.abs().sum()) == 0.0


# --------------------------------------------------------------------------
# Analytic fixtures on the composed batch
# --------------------------------------------------------------------------


def test_composed_delay_and_magnitude_match_the_closed_form(spike):
    tx, site, rx = drv.positions()
    response = drv.make_response()
    composed, _, _ = spike.paths(tx, site, rx, response)

    assert composed.path_count == 1
    assert composed.sensor_pair_count == 1
    assert composed.pair_offsets.tolist() == [0, 1]
    assert float(composed.total_delay_s[0]) == pytest.approx(
        geo.round_trip_delay_s(), rel=1e-6
    )

    d_in, d_out = geo.leg_distances_m()
    wavelength = geo.C0_M_PER_S / geo.REFERENCE_FREQUENCY_HZ
    # Each leg independently applies sqrt(P) * lambda / (4 pi d), so with unit
    # power on both the site is a 1 W isotropic re-radiator. That is a declared
    # spike simplification, not the radar equation (R-ADR-002); asserting it
    # verbatim is what stops it changing silently.
    expected = (
        drv.SPIKE_AMPLITUDE
        * (wavelength / (4.0 * math.pi * d_in))
        * (wavelength / (4.0 * math.pi * d_out))
    )
    assert abs(complex(composed.complex_transfer_ref[0])) == pytest.approx(
        expected, rel=1e-5
    )
    assert composed.row_valid is not None and bool(composed.row_valid.all())
    assert composed.reference_frequency_hz == geo.REFERENCE_FREQUENCY_HZ


def test_per_frame_host_traffic_is_two_copies_and_two_synchronizations(spike):
    """R-ADR-006 budget: two legs, one validation copy each, nothing else."""

    tx, site, rx = drv.positions()
    _, inbound, outbound = spike.paths(tx, site, rx, drv.make_response())
    copies = 0
    syncs = 0
    for legs in (inbound, outbound):
        diagnostics = legs.diagnostics
        assert diagnostics.compact_count_d2h_copies == 0
        assert diagnostics.discovery_launch_count == 0
        copies += diagnostics.validation_d2h_copies
        syncs += diagnostics.validation_sync_count
    assert copies == 2
    assert syncs == 2
    # Freezing is paid once per topology, outside every loop, and reported
    # separately so it can never be mistaken for per-frame cost.
    assert spike.inbound.prepare_synchronizations == 3
    assert spike.outbound.prepare_synchronizations == 3
