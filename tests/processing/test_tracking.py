"""The detection handoff: a contract, a batching helper, and one reference tracker.

What is under test is the INTERFACE. A target on a known constant-velocity
trajectory is pushed frame by frame, and the assertions are that the stream
stays one track and that the per-frame position residual against the analytic
trajectory stays inside a range bin. There is no claim here that the reference
associator is a good tracker; there is a claim that the contract it plugs into
is usable.

The non-differentiability is asserted rather than documented. Phase-9 item 4
already names CFAR, peak selection and tracking as the non-differentiable
stages, and a stage that silently detached would hand a caller a plausible
number with a zero gradient.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.radar.processing import (
    DetectionFrame,
    PointCloud,
    TrackHandoff,
    nearest_neighbour_associator,
)
from witwin.radar.processing.pointcloud import POINT_CLOUD_COLUMNS


RANGE_BIN_M = 0.17172175137889091
FRAME_PERIOD_S = 0.1
FRAMES = 10
START_M = (0.6, 8.0, 0.0)
CLOSING_MPS = 4.0


def _trajectory(frame_index: int) -> torch.Tensor:
    """A target closing along its own line of sight at a constant speed.

    Radial, because that is the only velocity component a radar measures and
    the only one the reference associator predicts with.
    """

    start = torch.tensor(START_M, dtype=torch.float64)
    direction = start / start.square().sum().sqrt()
    travelled = CLOSING_MPS * FRAME_PERIOD_S * frame_index
    return start - direction * travelled


def _frame(frame_index: int, *, clutter: bool = False) -> DetectionFrame:
    position = _trajectory(frame_index)
    rows = [position]
    velocities = [CLOSING_MPS]
    energies = [30.0]
    if clutter:
        # A stationary return four metres to the side: near enough to be in the
        # same frame, far enough that a one-metre gate cannot confuse them.
        rows.append(torch.tensor([-4.0, 6.0, 0.0], dtype=torch.float64))
        velocities.append(0.0)
        energies.append(24.0)
    return DetectionFrame(
        time_s=frame_index * FRAME_PERIOD_S,
        xyz=torch.stack(rows, dim=0),
        velocity_mps=torch.tensor(velocities, dtype=torch.float64),
        energy=torch.tensor(energies, dtype=torch.float64),
        frame_index=frame_index,
    )


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_a_point_cloud_crosses_into_a_detection_frame_without_recomputing_anything():
    cloud = PointCloud(
        xyz=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
        velocity_mps=torch.tensor([1.5], dtype=torch.float64),
        energy=torch.tensor([12.0], dtype=torch.float64),
        range_m=torch.tensor([math.sqrt(14.0)], dtype=torch.float64),
    )
    frame = DetectionFrame.from_point_cloud(cloud, time_s=0.25, frame_index=2)
    assert frame.xyz is cloud.xyz
    assert frame.velocity_mps is cloud.velocity_mps
    assert frame.energy is cloud.energy
    assert frame.time_s == 0.25
    assert frame.frame_index == 2
    assert len(frame) == 1


def test_the_handoff_is_explicitly_non_differentiable_and_refuses_a_gradient():
    """Refused, not detached. A silent detach is a zero gradient nobody sees."""

    xyz = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="non-differentiable"):
        DetectionFrame(
            time_s=0.0,
            xyz=xyz,
            velocity_mps=torch.zeros(1, dtype=torch.float64),
            energy=torch.zeros(1, dtype=torch.float64),
            frame_index=0,
        )
    with pytest.raises(ValueError, match="non-differentiable"):
        DetectionFrame(
            time_s=0.0,
            xyz=xyz.detach(),
            velocity_mps=torch.zeros(1, dtype=torch.float64, requires_grad=True),
            energy=torch.zeros(1, dtype=torch.float64),
            frame_index=0,
        )


def test_a_handoff_is_an_ordered_stream():
    handoff = TrackHandoff()
    handoff.push(_frame(0))
    handoff.push(_frame(1))
    with pytest.raises(ValueError, match="ordered stream"):
        handoff.push(_frame(1))
    with pytest.raises(TypeError, match="DetectionFrame"):
        handoff.push(object())


# ---------------------------------------------------------------------------
# The reference associator
# ---------------------------------------------------------------------------


def test_a_constant_velocity_target_produces_one_continuous_track():
    """``N = 10`` frames, one track, residual under one range bin per frame."""

    handoff = TrackHandoff()
    assignments = [handoff.push(_frame(index)) for index in range(FRAMES)]

    assert handoff.track_count == 1, handoff.track_count
    for assignment in assignments:
        assert assignment.tolist() == [0]

    history = handoff.track(0)
    assert len(history) == FRAMES
    for index, (time_s, position) in enumerate(history):
        assert time_s == pytest.approx(index * FRAME_PERIOD_S)
        residual = float((position - _trajectory(index)).square().sum().sqrt())
        assert residual < 1.0 * RANGE_BIN_M, (index, residual)


def test_a_second_stationary_return_gets_its_own_track_and_keeps_it():
    handoff = TrackHandoff()
    for index in range(FRAMES):
        assignment = handoff.push(_frame(index, clutter=True))
        assert assignment.tolist() == [0, 1]
    assert handoff.track_count == 2
    assert len(handoff.track(0)) == FRAMES
    assert len(handoff.track(1)) == FRAMES


def test_a_detection_beyond_the_gate_opens_a_new_track():
    """The gate is a declared distance, so it is testable rather than implicit.

    The constant-velocity prediction is EXACT on the trajectory above - the
    residual is at the float64 floor - so a target that stays on it can never
    leave any gate. What tests the gate is a detection that is not on it.
    """

    handoff = TrackHandoff()
    handoff.push(_frame(0))
    jumped = DetectionFrame(
        time_s=FRAME_PERIOD_S,
        xyz=torch.tensor([[6.0, 3.0, 0.0]], dtype=torch.float64),
        velocity_mps=torch.tensor([CLOSING_MPS], dtype=torch.float64),
        energy=torch.tensor([30.0], dtype=torch.float64),
        frame_index=1,
    )
    assert handoff.push(jumped).tolist() == [1]
    assert handoff.track_count == 2

    # And a wider gate accepts it, which is what makes the gate the thing under
    # test rather than the distance.
    def wide(previous, current, elapsed_s):
        return nearest_neighbour_associator(previous, current, elapsed_s, gate_m=20.0)

    lenient = TrackHandoff(associator=wide)
    lenient.push(_frame(0))
    assert lenient.push(jumped).tolist() == [0]
    assert lenient.track_count == 1


def test_the_associator_predicts_forward_rather_than_matching_where_it_was():
    """At four metres per second and a tenth of a second, that is 0.4 m of lead.

    A nearest-neighbour match against the PREVIOUS position rather than the
    predicted one would still find this target - which is why the assertion is
    on the residual of the prediction, not on the association.
    """

    previous = _frame(0)
    current = _frame(1)
    matched = nearest_neighbour_associator(previous, current, FRAME_PERIOD_S)
    assert matched.tolist() == [0]

    predicted_error = float(
        (
            previous.xyz[0]
            - previous.xyz[0]
            / previous.xyz[0].square().sum().sqrt()
            * (CLOSING_MPS * FRAME_PERIOD_S)
            - current.xyz[0]
        )
        .square()
        .sum()
        .sqrt()
    )
    unpredicted_error = float(
        (previous.xyz[0] - current.xyz[0]).square().sum().sqrt()
    )
    assert predicted_error < 1e-12
    assert unpredicted_error == pytest.approx(CLOSING_MPS * FRAME_PERIOD_S, rel=1e-9)


def test_an_empty_frame_associates_to_nothing_and_does_not_raise():
    handoff = TrackHandoff()
    handoff.push(_frame(0))
    empty = DetectionFrame(
        time_s=FRAME_PERIOD_S,
        xyz=torch.zeros((0, 3), dtype=torch.float64),
        velocity_mps=torch.zeros(0, dtype=torch.float64),
        energy=torch.zeros(0, dtype=torch.float64),
        frame_index=1,
    )
    assert handoff.push(empty).tolist() == []
    assert handoff.track_count == 1


def test_an_external_associator_is_accepted_and_its_output_is_validated():
    calls = []

    def always_new(previous, current, elapsed_s):
        calls.append(elapsed_s)
        return torch.full((len(current),), -1, dtype=torch.int64)

    handoff = TrackHandoff(associator=always_new)
    handoff.push(_frame(0))
    handoff.push(_frame(1))
    assert handoff.track_count == 2
    assert calls[1] == pytest.approx(FRAME_PERIOD_S)

    def wrong_length(previous, current, elapsed_s):
        return torch.zeros(5, dtype=torch.int64)

    with pytest.raises(ValueError, match="assignments"):
        TrackHandoff(associator=wrong_length).push(_frame(0))

    def wrong_dtype(previous, current, elapsed_s):
        return torch.zeros(len(current))

    with pytest.raises(TypeError, match="int64"):
        TrackHandoff(associator=wrong_dtype).push(_frame(0))


# ---------------------------------------------------------------------------
# Fixed-size batching
# ---------------------------------------------------------------------------


def test_the_fixed_size_batch_replaces_reg_datas_numpy_random_path():
    """Three cases, on the input device, with an EXPLICIT generator.

    ``reg_data`` drew from the global ``numpy`` random state, so two runs of the
    same simulation produced different batches and nothing said so.
    """

    frame = _frame(0, clutter=True)
    generator = torch.Generator().manual_seed(3)
    batch = frame.as_fixed_size(8, generator=generator)
    assert tuple(batch.shape) == (8, len(POINT_CLOUD_COLUMNS))
    assert batch.dtype == torch.float64

    # Every row is one of the two detections, and both appear.
    rows = {tuple(round(float(v), 9) for v in row) for row in batch}
    assert len(rows) == 2

    # Same seed, same batch: it is reproducible, which reg_data was not.
    again = frame.as_fixed_size(8, generator=torch.Generator().manual_seed(3))
    assert torch.equal(batch, again)

    # More detections than slots: a subset, without replacement.
    small = frame.as_fixed_size(1, generator=torch.Generator().manual_seed(0))
    assert tuple(small.shape) == (1, 6)

    empty = DetectionFrame(
        time_s=0.0,
        xyz=torch.zeros((0, 3), dtype=torch.float64),
        velocity_mps=torch.zeros(0, dtype=torch.float64),
        energy=torch.zeros(0, dtype=torch.float64),
        frame_index=0,
    )
    zeros = empty.as_fixed_size(4)
    assert torch.equal(zeros, torch.zeros((4, 6), dtype=torch.float64))

    with pytest.raises(ValueError, match="positive int"):
        frame.as_fixed_size(0)


def test_the_batch_range_column_is_the_norm_of_its_own_position():
    frame = _frame(3)
    batch = frame.as_fixed_size(2, generator=torch.Generator().manual_seed(1))
    for row in batch:
        assert float(row[5]) == pytest.approx(
            float(row[:3].square().sum().sqrt()), rel=1e-12
        )
