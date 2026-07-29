"""TDM slot phase, end to end on the scene-driven route (Phase 11 migration).

This is the surviving half of ``tests/solvers/test_tdm_phase.py``, which drove
the same claim through ``Radar.mimo`` / ``Radar.mimo_from_trace`` and the
Dirichlet solver. The claim is unchanged and it is still worth having, because
it is the only place the two halves of TDM meet:

* ``synthesis/fmcw.py`` WRITES a per-transmitter slow-time phase, because
  a TDM front end fires its transmitters sequentially and transmitter ``m`` is
  sampled ``m T_chirp`` later than transmitter 0;
* ``processing/aoa.py::tdm_compensate`` REMOVES it, and ``point_cloud`` calls
  it unconditionally.

If the simulation stopped writing that phase, the compensation would ADD one and
every moving target's angle would be wrong while every static one stayed right.
``test_phase6_fmcw_tdm.py`` pins the slot table and the kernel's own arithmetic;
what is pinned HERE is that the production entry point still produces the phase
those two agree about, measured against the closed form
``theta = 4 pi v T_chirp / lambda``.

Two of the old file's three tests do not survive the route and are not replaced:
``mimo_from_trace``'s linear-rate path and the agreement between it and the
per-chirp interpolator loop were statements about two Dirichlet entry points,
and this route has one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytest.importorskip("witwin.channel")

from conftest import (  # noqa: E402
    STANDARD_CONFIG,
    make_scene_radar_or_skip,
    simulate_point_targets,
)

pytestmark = pytest.mark.gpu


_CONFIG = {**STANDARD_CONFIG, "adc_start_time": 6}

#: Closing speed in metres per second. Radial, so the answer is not a
#: near-zero a dead tangent could also produce.
_SPEED = 1.5

#: Boresight range. Far enough that the two-way spreading is well conditioned
#: and near enough to stay inside the unambiguous range.
_RANGE_M = 3.0


def _radar():
    return make_scene_radar_or_skip(_CONFIG)


def _frame(radar, closing_speed):
    return simulate_point_targets(
        radar, [((0.0, 0.0, -_RANGE_M), (0.0, 0.0, closing_speed))]
    )


def _rd(cube):
    """``[TX, RX, doppler, range]`` from ``[TX, RX, chirp, sample]``.

    The transform is spelled out rather than taken from ``processing`` on
    purpose: the quantity this file measures is a PHASE across transmitters at
    one cell, and the windowed, DC-handled processing chain is a different
    product whose own correctness is pinned in ``tests/processing/``.
    """

    profile = torch.fft.fft(cube, dim=-1)
    return torch.fft.fftshift(torch.fft.fft(profile, dim=2), dim=2)


def _peak_cell(rd):
    magnitude = rd.abs().sum(dim=(0, 1))
    magnitude[:, 0] = 0.0  # the DC range bin is the direct coupling, not a target
    flat = int(torch.argmax(magnitude))
    return flat // magnitude.shape[1], flat % magnitude.shape[1]


def _peak_vector(rd):
    """The ``[TX, RX]`` complex vector at the map's strongest cell."""

    doppler_bin, range_bin = _peak_cell(rd)
    return rd[:, :, doppler_bin, range_bin], doppler_bin, range_bin


def _relative_phase(moving_vector, static_vector):
    """Slow-time phase of each TX row relative to TX0, geometry cancelled.

    Referencing the static frame removes the transmitter's own POSITION phase,
    which is identical in both frames, and leaves exactly what the motion wrote.
    Averaging the unit phasors across the receivers rather than the angles
    avoids a wrap at +/- pi turning into an average of nothing.
    """

    difference = torch.angle(moving_vector * static_vector.conj())
    per_tx = torch.angle(
        torch.exp(1j * difference.to(torch.complex64)).mean(dim=1)
    )
    return (per_tx - per_tx[0]).cpu().numpy()


def _per_transmitter_phase(static_cube, moving_cube):
    static_vector, _, _ = _peak_vector(_rd(static_cube))
    moving_vector, _, _ = _peak_vector(_rd(moving_cube))
    return _relative_phase(moving_vector, static_vector)


def _theta(axes, array, closing_speed):
    """The per-slot phase step the kernel writes, with its sign.

    ``tdm_compensate`` REMOVES ``+ phase_sign * 4 pi v m T_chirp / lambda``, so
    what the synthesis must have written is its negative. Reading the sign off
    ``ArrayGeometry.phase_sign`` rather than choosing one here is what makes
    this an independent expectation: a convention flip anywhere in the chain
    moves both the measurement and this closed form, and the compensation test
    below is what would then catch it.
    """

    chirp_period_s = float(axes.slow_time_period_s) / int(axes.num_tx)
    return (
        -int(array.phase_sign)
        * 4.0
        * math.pi
        * float(closing_speed)
        * chirp_period_s
        / float(axes.wavelength_m)
    )


def test_the_slot_period_is_the_chirp_period_times_the_transmitter_count():
    """The record's own arithmetic, before anything is measured against it.

    Every expectation below divides ``slow_time_period_s`` by ``num_tx``.
    Confusing the TDM SLOT period with the raw chirp period costs a factor of
    ``num_tx`` in every compensated elevation, so the identity is asserted
    rather than assumed.
    """

    radar = _radar()
    frame = _frame(radar, 0.0)
    chirp_period_s = (
        radar.config.idle_time + radar.config.ramp_end_time
    ) * 1.0e-6
    assert float(frame.axes.slow_time_period_s) == pytest.approx(
        chirp_period_s * radar.config.num_tx, rel=1e-9
    )


def test_a_still_target_writes_no_per_transmitter_phase():
    """The falsifier. Nothing moved, so the frame cannot walk across slots."""

    radar = _radar()
    static = _frame(radar, 0.0)
    relative = _per_transmitter_phase(static.cube, static.cube)
    # Not exactly zero: the estimator averages unit phasors and takes an angle,
    # both in float32. Nine orders below the phase this file measures.
    assert np.abs(relative).max() < 1.0e-6, relative


@pytest.mark.parametrize("closing_speed", [_SPEED, -_SPEED])
def test_the_production_entry_writes_the_tdm_slot_phase(closing_speed):
    """``[0, theta, 2 theta]`` across the three transmitters, sign included.

    The measured ratio to the closed form is about 1.03 and that is physics
    rather than slack: the Doppler phase slope tracks the EFFECTIVE carrier
    ``fc + slope * t_mid``, which at this configuration is roughly 3 percent
    above ``fc``. The tolerance is the old file's, unchanged.
    """

    radar = _radar()
    static = _frame(radar, 0.0)
    moving = _frame(radar, closing_speed)

    relative = _per_transmitter_phase(static.cube, moving.cube)
    theta = _theta(static.axes, static.array, closing_speed)
    expected = np.array([0.0, theta, 2.0 * theta])
    assert np.abs(relative - expected).max() < 0.15 * abs(theta) + 0.02, (
        f"per-TX phase {relative} does not match the TDM expectation {expected}"
    )


def test_the_processing_compensation_removes_what_the_kernel_wrote():
    """The two halves, closed against each other on one cell.

    ``tdm_compensate`` is what ``point_cloud`` applies before it estimates an
    angle. Feeding it the measured Doppler velocity at the peak cell must leave
    the per-transmitter phase flat; a compensation with the wrong sign would
    DOUBLE it, and a compensation applied to a frame that never carried the
    phase would create one.
    """

    from witwin.radar.processing import tdm_compensate

    radar = _radar()
    static = _frame(radar, 0.0)
    moving = _frame(radar, _SPEED)

    static_vector, _, _ = _peak_vector(_rd(static.cube))
    moving_vector, _, _ = _peak_vector(_rd(moving.cube))

    # The velocity handed to the compensation is the one PRODUCTION hands it:
    # the peak of the processing chain's own Range-Doppler map, in the
    # closing-positive convention its axes record publishes. Reading it off the
    # bare transform above instead would introduce a second Doppler-axis
    # ordering, which is a question ``processing.range_doppler`` owns.
    combined = moving.combined_map().abs()
    flat = int(combined.argmax())
    velocity = moving.axes.velocity_mps[flat // combined.shape[1]].to(
        dtype=torch.float32, device=moving.cube.device
    )
    assert float(velocity) == pytest.approx(
        _SPEED, abs=moving.axes.velocity_bin_mps
    )

    compensated = tdm_compensate(
        moving_vector.reshape(-1, 1),
        velocity.reshape(1),
        moving.array,
        moving.axes,
    ).reshape(moving.array.num_tx, moving.array.num_rx)

    theta = abs(_theta(moving.axes, moving.array, _SPEED))
    before = np.abs(_relative_phase(moving_vector, static_vector)).max()
    after = np.abs(_relative_phase(compensated, static_vector)).max()
    assert before > 1.5 * theta, (before, theta)
    assert after < 0.25 * theta, (before, after, theta)
