"""The streaming dataset writer (U-10, M5-8): scenes x seeds to disk, frame by frame, with labels."""

from __future__ import annotations

import json

import pytest
import torch

pytestmark = pytest.mark.gpu


def _radar():
    from witwin.radar import Fmcw, Noise, Radar

    waveform = Fmcw(
        slope=30e12,
        sample_rate=10e6,
        samples_per_chirp=32,
        chirps_per_frame=8,
        adc_start=1e-6,
        idle=5e-6,
        ramp_end=10e-6,
    )
    return Radar(
        reference_frequency_hz=77e9,
        waveform=waveform,
        tx=((0.0, 0.0, 0.0),),
        rx=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        power=12.0,
        antenna_unit="half_wavelength",
        noise=Noise(figure=10.0),
        seed=5,
    )


def _episode(start: float, *, seed, name=None):
    from witwin.core import Scene

    from witwin.radar import PointTargets
    from witwin.radar.dataset import Episode

    def trajectory(time_s: float) -> torch.Tensor:
        return torch.tensor([[0.5, 0.0, start + 2.0 * time_s]], dtype=torch.float32, device="cuda")

    targets = PointTargets(positions=trajectory(0.0), rcs=1.0, trajectory=trajectory)
    return Episode(scene=Scene(), targets=targets, times=(0.0, 0.05, 0.1), seed=seed, name=name)


def test_episodes_are_written_frame_by_frame_and_load_back(tmp_path, monkeypatch):
    from witwin.radar import Motion, Radar
    from witwin.radar.dataset import load_episode, read_dataset, write_dataset
    from witwin.radar.labels import ground_truth

    radar = _radar()
    episodes = [_episode(-6.0, seed=1, name="near"), _episode(-9.0, seed=None)]

    # Streaming: when the stream is asked for frame i, frame i - 1 is already on disk.
    written: list[bool] = []
    stream = Radar.stream

    def watched(self, *args, **kwargs):
        root = tmp_path / "set"
        for index, frame in enumerate(stream(self, *args, **kwargs)):
            if index:
                episode = "near" if self.seed == 1 else "episode_00001"
                written.append((root / episode / "frames" / f"{index - 1:06d}.pt").is_file())
            yield frame

    monkeypatch.setattr(Radar, "stream", watched)
    root = write_dataset(tmp_path / "set", radar, episodes, los=True, reflections=0, motion=Motion.chirp())
    monkeypatch.setattr(Radar, "stream", stream)
    assert written == [True] * 4

    index = read_dataset(root)
    assert [entry["name"] for entry in index["episodes"]] == ["near", "episode_00001"]
    assert [entry["frames"] for entry in index["episodes"]] == [3, 3]
    assert Radar.from_dict(index["radar"], device=radar.device) == radar

    for entry, episode in zip(index["episodes"], episodes, strict=True):
        result, labels = load_episode(root / entry["name"], device="cuda")
        run_radar = radar if episode.seed is None else radar.replace(seed=episode.seed)
        expected = run_radar.simulate(
            episode.scene, episode.targets, times=episode.times, los=True, reflections=0, motion=Motion.chirp()
        )
        assert torch.equal(result.cube, expected.cube)
        assert result.times == expected.times and result.path_set_complete == expected.path_set_complete
        truth = ground_truth(run_radar, episode.scene, episode.targets, times=episode.times)
        for name in ("range_m", "radial_velocity_mps", "azimuth_rad", "target_id", "site_id"):
            assert torch.equal(getattr(labels, name), getattr(truth, name)), name
    # Different seeds draw different receiver noise.
    near, _ = load_episode(root / "near")
    far, _ = load_episode(root / "episode_00001")
    assert not torch.equal(near.cube, far.cube)


def test_an_existing_dataset_needs_overwrite_and_a_foreign_directory_is_never_cleared(tmp_path):
    from witwin.radar.dataset import write_dataset

    radar = _radar()
    episodes = [_episode(-6.0, seed=1)]
    write_dataset(tmp_path / "set", radar, episodes, los=True, reflections=0, labels=False)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_dataset(tmp_path / "set", radar, episodes, los=True, reflections=0)
    write_dataset(tmp_path / "set", radar, episodes, los=True, reflections=0, overwrite=True)
    record = json.loads((tmp_path / "set" / "dataset.json").read_text())
    assert record["labels"] is True

    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="holds no dataset"):
        write_dataset(tmp_path / "other", radar, episodes, overwrite=True)
    assert (tmp_path / "other" / "keep.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="ad_mode must be 'none'"):
        write_dataset(tmp_path / "graded", radar, episodes, ad_mode="vjp")
