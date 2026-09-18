"""Streamed frames against the stacked result of the identical session."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from support import multi_endpoint_driver as drv
from support.simulate_fixture import fixture_radar, static_scene

from witwin.radar import Motion, PointTargets

pytestmark = pytest.mark.gpu

FRAMES = 6

#: One frame's cube in bytes: 8 sensor pairs, 128 chirps, 256 range bins, complex64.
FRAME_CUBE_BYTES = 8 * 128 * 256 * 2 * 2

MOTIONS = [Motion.adc(), Motion.chirp(), Motion.adaptive()]
MOTION_IDS = ["adc", "chirp", "adaptive"]


def _session(radar, frames):
    origin = torch.tensor([[2.0, 0.2, 0.0]], device=radar.device)
    velocity = torch.tensor([[0.7, 0.0, 0.0]], device=radar.device)
    return {
        "targets": PointTargets(
            positions=origin,
            amplitude=drv.FIXTURE_AMPLITUDE,
            phase=drv.FIXTURE_PHASE_RAD,
            trajectory=lambda t: origin + velocity * t,
        ),
        "times": tuple(index * 2.0e-3 for index in range(frames)),
        "los": True,
        "reflections": 0,
    }


def _small(radar):
    """A new radar with a small cube; the record is frozen, so nothing is edited."""

    return radar.replace(waveform=replace(radar.waveform, samples_per_chirp=8, chirps_per_frame=4, output="beat"))


@pytest.mark.parametrize("motion", MOTIONS, ids=MOTION_IDS)
def test_streamed_frames_equal_the_stacked_cube_bit_for_bit(motion):
    stacked = _small(fixture_radar()).simulate(
        static_scene(), **_session(_small(fixture_radar()), FRAMES), motion=motion
    )

    radar = _small(fixture_radar())
    kwargs = _session(radar, FRAMES)
    streamed = list(radar.stream(static_scene(), **kwargs, motion=motion))

    assert len(streamed) == FRAMES
    for index, frame in enumerate(streamed):
        assert frame.frame_count == 1
        assert frame.times_s == (stacked.times_s[index],)
        assert np.array_equal(frame.sample_times_s[0], stacked.sample_times_s[index])
        assert frame.epochs == (stacked.epochs[index],)
        assert frame.rediscovery_reasons == (stacked.rediscovery_reasons[index],)
        assert frame.axis_names == stacked.axis_names
        assert frame.output_domain == stacked.output_domain
        assert frame.motion_sampling == stacked.motion_sampling
        torch.testing.assert_close(frame.cube[0], stacked.cube[index], rtol=0, atol=0)

    # The run-level statements are the conjunction over the frames.
    assert all(frame.path_set_complete for frame in streamed) == stacked.path_set_complete
    assert all(frame.motion_sampling_exhaustive for frame in streamed) == stacked.motion_sampling_exhaustive
    assert streamed[-1].discovery_count == stacked.discovery_count
    assert streamed[-1].compile_count == stacked.compile_count


def _large(radar):
    """A cube big enough that the stacked sequence dominates allocation."""

    return radar.replace(waveform=replace(radar.waveform, samples_per_chirp=256, chirps_per_frame=128, output="beat"))


def _still_session(radar, frames):
    """No trajectory anywhere, so each frame costs one scene evaluation."""

    return {
        "targets": PointTargets(
            positions=torch.tensor([[2.0, 0.2, 0.0]], device=radar.device),
            amplitude=drv.FIXTURE_AMPLITUDE,
            phase=drv.FIXTURE_PHASE_RAD,
        ),
        "times": tuple(index * 2.0e-3 for index in range(frames)),
        "los": True,
        "reflections": 0,
    }


def test_streaming_retains_one_frame_where_stacking_retains_the_sequence():
    """The point of the entry: peak allocation must not track the frame count."""

    sequence = 32

    def measure(run):
        radar = _large(fixture_radar())
        kwargs = _still_session(radar, sequence)
        run(radar, kwargs)  # warm the native route and the allocator
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated()
        run(radar, kwargs)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - resident

    def stacked(radar, kwargs):
        result = radar.simulate(static_scene(), **kwargs)
        assert result.frame_count == sequence
        del result

    def streamed(radar, kwargs):
        count = 0
        for frame in radar.stream(static_scene(), **kwargs):
            count += 1
            del frame
        assert count == sequence

    # One frame needs several live buffers at once - native output, assembled
    # cube, the one-frame stack - so the bound is a small multiple of a frame
    # rather than exactly one. What matters is that it does not scale with the
    # sequence, which the stacked route provably does.
    frame_bytes = FRAME_CUBE_BYTES
    stacked_peak, streamed_peak = measure(stacked), measure(streamed)
    assert stacked_peak > sequence * frame_bytes, (stacked_peak, frame_bytes)
    assert streamed_peak < 8 * frame_bytes, (streamed_peak, frame_bytes)
    assert streamed_peak < stacked_peak / 4, (streamed_peak, stacked_peak)


def test_streaming_holds_nothing_per_frame():
    """Resident allocation after a stream does not grow with the sequence length.

    The property is the SLOPE, not the value. The caching allocator's block reuse
    makes the absolute delta noisy in either direction - a short run can end below
    where it started - and it depends on which tests ran before this one, so an
    equality on the two deltas pins the allocator rather than the retention. A
    real per-frame retention would instead make the delta grow by about one cube
    per extra frame, which is what the bound below refuses.
    """

    def resident(frames):
        radar = _small(fixture_radar())
        kwargs = _session(radar, frames)
        for frame in radar.stream(static_scene(), **kwargs, motion=Motion.adc()):
            del frame
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for frame in radar.stream(static_scene(), **kwargs, motion=Motion.adc()):
            del frame
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() - before

    short, long_run = resident(2), resident(4 * FRAMES)
    assert long_run - short < FRAME_CUBE_BYTES, (short, long_run)


@pytest.mark.parametrize("motion", MOTIONS, ids=MOTION_IDS)
def test_retained_state_describes_the_observation_that_closed_the_frame(motion):
    """A sampled frame opens and closes at different world instants."""

    radar = _small(fixture_radar())
    kwargs = _session(radar, 2)
    result = radar.simulate(static_scene(), **kwargs, motion=motion)

    # The site moves, so the snapshot of the frame's last observation is not the
    # snapshot of its first. Publishing the opening one would silently describe
    # a world the simulation had already left.
    opening, closing = result.sample_times_s[-1][0], result.sample_times_s[-1][-1]
    assert closing > opening
    assert result.last_snapshot.time_s == pytest.approx(closing, abs=1e-12)
    assert result.last_snapshot.time_s != pytest.approx(opening, abs=1e-12)

    # The four diagnostics describe ONE observation, so they must agree.
    assert result.last_compiled_scene is not None
    assert result.last_propagation is not None and result.last_radar_paths is not None


def test_streaming_reports_the_frame_it_just_yielded():
    """Each frame reports itself, and the radar reports nothing.

    The diagnostics live on the yielded result and nowhere else, so a consumer
    reads the frame it is holding rather than a radar attribute that a later
    frame would already have overwritten.
    """

    radar = _small(fixture_radar())
    kwargs = _session(radar, FRAMES)
    seen = []
    for frame in radar.stream(static_scene(), **kwargs, motion=Motion.adc()):
        assert frame.last_snapshot is not None
        assert frame.last_snapshot.time_s == pytest.approx(frame.sample_times_s[0][-1], abs=1e-12)
        seen.append(frame.times_s[0])
    assert seen == list(kwargs["times"])
