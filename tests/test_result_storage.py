"""Saved results, compact reprs and the package version (U-9, M5-7)."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re

import numpy as np
import pytest
import torch
from conftest import STANDARD_CONFIG

import witwin.radar
from witwin.radar import Fmcw, Pulsed, Radar, Result
from witwin.radar.processing import ProcessingAxes, ProcessingCube, range_doppler_map, range_profile
from witwin.radar.storage import RESULT_FORMAT_VERSION, read_metadata

REPO = pathlib.Path(__file__).resolve().parents[1]


def _result(radar: Radar, *, frames: int = 3, seed: int = 0) -> Result:
    """A result built the way the echo builds one, around a random cube, on the CPU."""

    axes = ProcessingAxes.from_radar(radar, device="cpu")
    generator = torch.Generator().manual_seed(seed)
    shape = (frames, axes.transmit_channels, axes.num_rx, axes.doppler_bin_count, axes.range_bin_count)
    cube = torch.randn(shape, generator=generator, dtype=torch.complex64)
    return Result(
        cube=cube,
        times=tuple(0.1 * index for index in range(frames)),
        kind=axes.waveform,
        axis_names=("frame", "tx", "rx", axes.slow_time_name, axes.fast_time_name),
        axes=axes,
        phasor=axes.phasor,
        time_dependence="exp(+j*2*pi*f*t)",
        reference_frequency_hz=axes.reference_frequency_hz,
        epochs=tuple(range(frames)),
        rediscovery_reasons=("initial",) + (None,) * (frames - 1),
        compile_count=1,
        discovery_count=frames,
        last_snapshot=None,
        last_compiled_scene=None,
        last_propagation=None,
        last_radar_paths=None,
        sample_times=tuple(np.linspace(index, index + 1e-3, 5) for index in range(frames)),
        path_set_complete=True,
        motion_sampling_exhaustive=False,
        motion_sampling="chirp",
        output_domain=axes.output_domain,
        adaptive_diagnostics=({"evaluations": 3, "max_tested_phase_error_rad": 0.01, "exhaustive": False},),
        observation_schedule="slow_time",
        discovery_complete=False,
        transmission_dropped_rows=2,
    )


def _assert_same(loaded: Result, saved: Result) -> None:
    assert torch.equal(loaded.cube, saved.cube)
    for field in dataclasses.fields(saved.axes):
        mine, theirs = getattr(loaded.axes, field.name), getattr(saved.axes, field.name)
        if isinstance(theirs, torch.Tensor):
            assert torch.equal(mine.cpu(), theirs.cpu()), field.name
        else:
            assert mine == theirs, field.name
    for field in dataclasses.fields(Result):
        name = field.name
        if name in ("cube", "axes", "sample_times", "events") or name.startswith("last_"):
            continue
        assert getattr(loaded, name) == getattr(saved, name), name
    for mine, theirs in zip(loaded.sample_times, saved.sample_times, strict=True):
        assert np.array_equal(mine, theirs)
    assert loaded.last_radar_paths is None and loaded.events == ()


def _radars() -> dict[str, Radar]:
    fmcw = Radar.from_dict(STANDARD_CONFIG, device="cpu")
    beat = Radar.from_dict({**STANDARD_CONFIG, "output_domain": "beat", "chirp_per_frame": 8}, device="cpu")
    bpm = beat.replace(
        tx=((0, 0, 0), (2, 0, 0)), waveform=Fmcw(**{**_fields(beat.waveform), "mimo": "bpm", "chirps_per_frame": 8})
    )
    pulsed = Radar(
        reference_frequency_hz=10e9,
        waveform=Pulsed.pmcw(Pulsed.m_sequence(4), chip_rate=1e7, repetitions=4, sample_rate=2e7),
        tx=[(0, 0, 0)],
        rx=[(0, 0, 0), (1, 0, 0)],
        antenna_unit="half_wavelength",
        power=30.0,
        device="cpu",
    )
    return {"fmcw": fmcw, "bpm": bpm, "pulsed": pulsed}


def _fields(record) -> dict:
    return {field.name: getattr(record, field.name) for field in dataclasses.fields(record) if field.init}


class TestSaveLoad:
    @pytest.mark.parametrize("name", ["fmcw", "bpm", "pulsed"])
    def test_a_saved_result_loads_back_equal(self, tmp_path, name):
        saved = _result(_radars()[name])
        saved.save(tmp_path / "run")
        _assert_same(Result.load(tmp_path / "run"), saved)

    def test_a_loaded_result_processes_identically(self, tmp_path):
        saved = _result(_radars()["fmcw"])
        saved.save(tmp_path / "run")
        loaded = Result.load(tmp_path / "run")
        assert torch.equal(loaded.frame(1).range_doppler().data, saved.frame(1).range_doppler().data)

    def test_every_tensor_file_loads_with_weights_only(self, tmp_path):
        _result(_radars()["fmcw"]).save(tmp_path / "run")
        for path in (tmp_path / "run").rglob("*.pt"):
            torch.load(path, weights_only=True)
        record = json.loads((tmp_path / "run" / "metadata.json").read_text())
        assert record["format"] == "witwin.radar.result" and record["version"] == RESULT_FORMAT_VERSION
        assert record["producer"]["version"] == witwin.radar.__version__
        assert set(record["not_saved"]) == {"last_snapshot", "last_compiled_scene", "last_propagation"} | {
            "last_radar_paths",
            "events",
        }

    def test_a_non_empty_directory_is_refused_without_overwrite(self, tmp_path):
        result = _result(_radars()["fmcw"])
        result.save(tmp_path / "run")
        with pytest.raises(FileExistsError, match="overwrite=True"):
            result.save(tmp_path / "run")
        other = _result(_radars()["fmcw"], frames=2, seed=1)
        other.save(tmp_path / "run", overwrite=True)
        assert Result.load(tmp_path / "run").frame_count == 2
        assert sorted(path.name for path in (tmp_path / "run" / "frames").iterdir()) == ["000000.pt", "000001.pt"]

    def test_overwrite_never_clears_a_directory_this_format_did_not_write(self, tmp_path):
        (tmp_path / "mine").mkdir()
        (tmp_path / "mine" / "notes.txt").write_text("keep")
        with pytest.raises(FileExistsError, match="holds no saved result"):
            _result(_radars()["fmcw"]).save(tmp_path / "mine", overwrite=True)
        assert (tmp_path / "mine" / "notes.txt").read_text() == "keep"

    def test_a_newer_format_version_and_an_unfinished_write_are_refused(self, tmp_path):
        _result(_radars()["fmcw"]).save(tmp_path / "run")
        metadata = tmp_path / "run" / "metadata.json"
        record = json.loads(metadata.read_text())
        metadata.write_text(json.dumps({**record, "version": RESULT_FORMAT_VERSION + 1}))
        with pytest.raises(ValueError, match="format version"):
            Result.load(tmp_path / "run")
        metadata.unlink()
        with pytest.raises(FileNotFoundError, match="did not finish"):
            read_metadata(tmp_path / "run")


class TestRepr:
    def test_result_and_frame_reprs_print_no_tensor_values(self):
        result = _result(_radars()["fmcw"])
        for text in (repr(result), repr(result.frame(0))):
            assert "tensor(" not in text and len(text) < 400, text
        assert "complex64[3, 3, 4, 128, 256] cpu" in repr(result)

    def test_processing_product_reprs_are_compact(self):
        result = _result(_radars()["fmcw"])
        cube = ProcessingCube(data=result.cube[0], axes=result.axes)
        products = (result.axes, cube, range_profile(cube), range_doppler_map(range_profile(cube)))
        for product in products:
            text = repr(product)
            assert "tensor(" not in text and len(text) < 1200, text
            assert re.match(rf"{type(product).__name__}\(", text)


def test_the_package_version_is_the_project_version():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text).group(1)
    assert witwin.radar.__version__ == declared


@pytest.mark.gpu
def test_a_simulated_result_round_trips_through_disk(tmp_path):
    """End to end on the GPU: simulate, save, load, and the cube and its RD map are bit-identical."""

    from witwin.core import Scene

    from witwin.radar import Motion, PointTargets

    radar = Radar(
        reference_frequency_hz=77e9,
        waveform=Fmcw(
            slope=30e12,
            sample_rate=10e6,
            samples_per_chirp=64,
            chirps_per_frame=8,
            adc_start=1e-6,
            idle=5e-6,
            ramp_end=10e-6,
        ),
        tx=((0.0, 0.0, 0.0),),
        rx=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        power=12.0,
        antenna_unit="half_wavelength",
    )
    trajectory = lambda t: torch.tensor([[0.0, 0.0, -5.0 + 2.0 * t]], device="cuda")  # noqa: E731
    targets = PointTargets(positions=trajectory(0.0), rcs=1.0, trajectory=trajectory)
    result = radar.simulate(Scene(), targets, times=[0.0, 0.02], los=True, reflections=0, motion=Motion.chirp())
    result.save(tmp_path / "run")
    loaded = Result.load(tmp_path / "run", device="cuda")
    assert loaded.cube.device.type == "cuda"
    assert torch.equal(loaded.cube, result.cube)
    assert torch.equal(loaded.frame(1).range_doppler().data, result.frame(1).range_doppler().data)
    assert loaded.motion_sampling == result.motion_sampling and loaded.epochs == result.epochs
    assert "tensor(" not in repr(result) and "tensor(" not in repr(radar.trace(Scene(), targets, times=[0.0]))
