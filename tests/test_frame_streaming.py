"""Streamed frames against the stacked result of the identical session."""

from dataclasses import replace

import pytest
import torch
from test_phase11_simulate_entry import _radar, _response, _static_scene

from witwin.radar.propagation import Kinematics
from witwin.radar.simulation import ScatterSitePolicy

pytestmark = pytest.mark.gpu

FRAMES = 6


def _session(radar, frames):
    origin = torch.tensor([[2.0, 0.2, 0.0]], device=radar.device)
    velocity = torch.tensor([[0.7, 0.0, 0.0]], device=radar.device)

    class Motion:
        def at(self, t):
            return Kinematics(origin + velocity * t, velocity)

    return {
        "times": tuple(index * 2.0e-3 for index in range(frames)),
        "response": _response(radar),
        "sites": ScatterSitePolicy.explicit(origin, trajectory=Motion()),
        "components": frozenset({"los"}),
        "max_depth": 0,
    }


def _small(radar):
    radar.system_config = replace(
        radar.system_config,
        waveform=replace(radar.system_config.waveform, adc_samples=8, chirp_per_frame=4, output_domain="beat"),
    )
    return radar


@pytest.mark.parametrize("sampling", ["adc", "chirp", "adaptive"])
def test_streamed_frames_equal_the_stacked_cube_bit_for_bit(sampling):
    stacked = _small(_radar()).simulate(_static_scene(), **_session(_small(_radar()), FRAMES), motion_sampling=sampling)

    radar = _small(_radar())
    kwargs = _session(radar, FRAMES)
    streamed = list(radar.stream(_static_scene(), **kwargs, motion_sampling=sampling))

    assert len(streamed) == FRAMES
    for index, frame in enumerate(streamed):
        assert frame.frame_count == 1
        assert frame.times_s == (stacked.times_s[index],)
        assert frame.sample_times_s == (stacked.sample_times_s[index],)
        assert frame.epochs == (stacked.epochs[index],)
        assert frame.rediscovery_reasons == (stacked.rediscovery_reasons[index],)
        assert frame.axes == stacked.axes
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

    radar.system_config = replace(
        radar.system_config,
        waveform=replace(radar.system_config.waveform, adc_samples=256, chirp_per_frame=128, output_domain="beat"),
    )
    return radar


def _still_session(radar, frames):
    """No trajectory anywhere, so each frame costs one scene evaluation."""

    return {
        "times": tuple(index * 2.0e-3 for index in range(frames)),
        "response": _response(radar),
        "sites": ScatterSitePolicy.explicit(torch.tensor([[2.0, 0.2, 0.0]], device=radar.device)),
        "components": frozenset({"los"}),
        "max_depth": 0,
    }


def test_streaming_retains_one_frame_where_stacking_retains_the_sequence():
    """The point of the entry: peak allocation must not track the frame count."""

    sequence = 32

    def measure(run):
        radar = _large(_radar())
        kwargs = _still_session(radar, sequence)
        run(radar, kwargs)  # warm the native route and the allocator
        radar._last_result = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated()
        run(radar, kwargs)
        radar._last_result = None
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - resident

    def stacked(radar, kwargs):
        result = radar.simulate(_static_scene(), **kwargs)
        assert result.frame_count == sequence
        del result

    def streamed(radar, kwargs):
        count = 0
        for frame in radar.stream(_static_scene(), **kwargs):
            count += 1
            del frame
        assert count == sequence

    # One frame needs several live buffers at once - native output, assembled
    # cube, the one-frame stack - so the bound is a small multiple of a frame
    # rather than exactly one. What matters is that it does not scale with the
    # sequence, which the stacked route provably does.
    frame_bytes = 8 * 128 * 256 * 2 * 2
    stacked_peak, streamed_peak = measure(stacked), measure(streamed)
    assert stacked_peak > sequence * frame_bytes, (stacked_peak, frame_bytes)
    assert streamed_peak < 8 * frame_bytes, (streamed_peak, frame_bytes)
    assert streamed_peak < stacked_peak / 4, (streamed_peak, stacked_peak)


def test_streaming_holds_nothing_per_frame():
    """Resident allocation after a stream is the same for any sequence length."""

    def resident(frames):
        radar = _small(_radar())
        kwargs = _session(radar, frames)
        for frame in radar.stream(_static_scene(), **kwargs, motion_sampling="adc"):
            del frame
        radar._last_result = None
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for frame in radar.stream(_static_scene(), **kwargs, motion_sampling="adc"):
            del frame
        radar._last_result = None
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() - before

    assert resident(2) == resident(4 * FRAMES)


@pytest.mark.parametrize("sampling", ["adc", "chirp", "adaptive"])
def test_retained_state_describes_the_observation_that_closed_the_frame(sampling):
    """A sampled frame opens and closes at different world instants."""

    radar = _small(_radar())
    kwargs = _session(radar, 2)
    result = radar.simulate(_static_scene(), **kwargs, motion_sampling=sampling)

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
    radar = _small(_radar())
    kwargs = _session(radar, FRAMES)
    seen = []
    for frame in radar.stream(_static_scene(), **kwargs, motion_sampling="adc"):
        assert radar.last_result is frame
        assert radar.last_snapshot is frame.last_snapshot
        seen.append(frame.times_s[0])
    assert seen == list(kwargs["times"])
