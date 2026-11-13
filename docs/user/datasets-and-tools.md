# Datasets and tools

This guide covers everything that moves a radar, its output or its ground
truth in or out of a Python process: the installation check, the capability
record, saved results, the flat configuration format, TI mmWave profiles and
board presets, raw ADC captures in the DCA1000 layout, ground-truth labels,
and the dataset writer. The field-by-field reference of the configuration
format is [`docs/configuration.md`](../configuration.md).

Every example that writes files writes them into a temporary directory.

## The doctor

`witwin-radar-doctor` (or `python -m witwin.radar doctor`) checks an
installation: Python, PyTorch, CUDA and driver identity, the native library's
build identity and whether it carries code for the active GPU, the optional
Channel installation (and Channel's own doctor), and a one-target simulate on
the GPU. It prints a verdict with the fix for every problem and exits 1 when
Radar cannot run.

```bash
witwin-radar-doctor              # the report and the GPU smoke test
witwin-radar-doctor --no-smoke   # skip the simulate
python -m witwin.radar doctor --json
```

The same check from Python returns a plain record; `doctor_report` renders it
and gives the exit status:

```python
import json
import tempfile
from pathlib import Path

import torch
import witwin.radar as wr
from witwin.core import Scene
from witwin.radar import Adc, Agc, Fmcw, Motion, Noise, PointTargets, Radar, Result
from witwin.radar import config, dataset, labels, rawdata
from witwin.radar.doctor import doctor, doctor_report

record = doctor(smoke=False, channel_doctor=False)
text, status = doctor_report(record)
print("exit status", status, "|", next(line for line in text.splitlines() if line.startswith("Verdict")))
print("native library available:", record["cuda_available"], "| failures:", record["failures"])
```

## What this build supports: `capabilities()`

`witwin.radar.capabilities()` is a versioned, JSON-serialisable record of the
build: waveform kinds and their options, motion kinds, propagation
components, target records, pattern kinds, the AD leaves (see
[Gradients](gradients.md)), the processing vocabularies and the native
operators.

```python
caps = wr.capabilities()
print("schema", caps["schema_version"], "| sections:", ", ".join(caps))
print("FMCW MIMO:", caps["waveforms"]["fmcw"]["mimo"], "| CFAR:", caps["processing"]["cfar"])
json.dumps(caps)  # plain data
```

## Compact representations

Results, frames, paths and processing products print their structure, never
their values: a tensor is `dtype[shape] device`, a long tuple is shown by its
ends and length. Printing one never copies from the GPU.

```python
chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=256, chirps_per_frame=16,
             adc_start=6e-6, idle=7e-6, ramp_end=65e-6, output="beat")
radar = Radar(reference_frequency_hz=77e9, waveform=chirp, power=15.0, antenna_unit="half_wavelength",
              tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
scene = Scene()
targets = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0,
                       trajectory=lambda t: torch.tensor([[0.0, 0.0, -3.0 + t]], device="cuda"))
result = radar.simulate(scene, targets, times=(0.0, 0.05))
print(result)
print(result.frame(1))
print(result.frame(0).range_doppler())
```

## Saving and loading results

`result.save(path)` writes a directory and `Result.load(path, device="cpu")`
reads it back; the cube and the processing axes come back bit for bit.

```text
path/metadata.json        format "witwin.radar.result", version, producer, every scalar
path/axes.pt              the axes record's tensors
path/frames/000000.pt     {"cube": [tx, rx, slow, fast], "sample_times_s": float64[...]}
```

Each `.pt` file holds plain tensors and is read with
`torch.load(weights_only=True)`: no Radar, Channel or Core object is pickled.
The closing frame's world state (`last_snapshot`, `last_compiled_scene`,
`last_propagation`, `last_radar_paths`) and the path events are not saved; a
loaded result carries `None` and `()` there. `metadata.json` is written last,
so a directory without it is an interrupted write. Saving over an existing
result needs `overwrite=True`. `Paths` (the output of `Radar.trace`) cannot be
saved: it holds the live session.

```python
with tempfile.TemporaryDirectory() as tmp:
    result.save(Path(tmp) / "run")
    loaded = Result.load(Path(tmp) / "run", device="cuda")
    print(sorted(p.name for p in (Path(tmp) / "run").rglob("*") if p.is_file()))
print("cube identical:", torch.equal(loaded.cube, result.cube), "| times:", loaded.times,
      "| last_snapshot:", loaded.last_snapshot)
same_map = torch.equal(loaded.frame(1).range_doppler().data, result.frame(1).range_doppler().data)
print("range-Doppler map of the loaded result identical:", same_map)
assert torch.equal(loaded.cube, result.cube) and same_map
```

## The flat configuration: `to_dict`, `from_dict`, the schema

`radar.to_dict()` / `Radar.from_dict(mapping, **overrides)` and
`radar.to_json(path)` / `Radar.from_json(path, **overrides)` write and read
the flat configuration format; they round-trip exactly. Keyword overrides use
the `Radar` field names in SI units; `device` is always a keyword.
`witwin.radar.config.config_schema()` returns the format's JSON Schema
(draft 2020-12). The field table is in [`docs/configuration.md`](../configuration.md).

```python
noisy = radar.replace(noise=Noise(figure=10.0), agc=Agc(target_rms=0.05), adc=Adc(bits=12, full_scale=0.2))
flat = noisy.to_dict()
print(json.dumps(flat)[:160], "...")
assert Radar.from_dict(flat, device=noisy.device) == noisy
with tempfile.TemporaryDirectory() as tmp:
    noisy.to_json(Path(tmp) / "radar.json")
    louder = Radar.from_json(Path(tmp) / "radar.json", power=20.0)
print("from_json with an override: power", louder.power, "dBm")
schema = config.config_schema()
print("schema:", schema["$schema"], "|", len(schema["properties"]), "keys")
```

## TI mmWave profiles and board presets

`config.read_ti_cfg(path_or_text)` reads a TI mmWave SDK `.cfg` profile
(`channelCfg`, `adcCfg`, `profileCfg`, `chirpCfg`, `frameCfg`, `bpmCfg`; other
commands are processing or GUI settings and are ignored; an advanced-frame
profile is refused) into a `TiConfig`. The `.cfg` names channels, not
positions, so `TiConfig.to_radar(tx=..., rx=..., power=12.0, **overrides)`
takes the board's element positions indexed by channel number. Under TDM the
radar's transmitters are those in chirp-slot order; a `bpmCfg` loop becomes
`Fmcw(mimo="bpm")`; per-chirp variations or several profiles become an
`Fmcw.sequence`; `adcCfg` output format 0 is `sampling="real"`. The `.cfg`
gives no transmit power: `power` is in dBm. `TiConfig.frame_times()` gives the
`frameCfg` frame instants.

```python
profile = """
% a 3 TX x 4 RX TDM profile
sensorStop
channelCfg 15 7 0
adcCfg 2 1
profileCfg 0 77 7 6 60 0 0 60 1 256 5000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
chirpCfg 2 2 0 0 0 0 0 2
frameCfg 0 2 32 10 50 1 0
"""
ti = config.read_ti_cfg(profile)
# Element positions in half wavelengths, indexed by channel number.
board_tx = ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (4.0, 0.0, 0.0))
board_rx = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))
ti_radar = ti.to_radar(tx=board_tx, rx=board_rx, power=12.0)
print(f"{ti_radar.num_tx} TX x {ti_radar.num_rx} RX, slot order tx = {ti_radar.tx}")
print(f"slope {ti_radar.waveform.slope / 1e12:.0f} MHz/us, {ti_radar.waveform.samples_per_chirp} samples, "
      f"{ti_radar.waveform.chirps_per_frame} chirps per transmitter, frames at {ti.frame_times()[:3]} s")
```

The chirp configuration above fires TX0, TX2, TX1 in that order (masks 1, 4,
2), so the radar's second transmitter is the board's TX2.

`config.preset(name, **overrides)` builds an evaluation board: `"awr1843boost"`
(77 GHz) and `"iwr6843isk"` (60 GHz), both with a 3 TX x 4 RX layout
(receivers half a wavelength apart, TX0 and TX2 two wavelengths apart, TX1
midway and half a wavelength up: an 8-element azimuth row and a 4-element
elevated row) and a representative 256-sample, 64-loop TDM profile.

```python
board = config.preset("awr1843boost")
print(f"awr1843boost: {board.reference_frequency_hz / 1e9:.0f} GHz, {board.num_tx} TX x {board.num_rx} RX, "
      f"{board.waveform.samples_per_chirp} samples x {board.waveform.chirps_per_frame} chirps")
```

**Not validated against hardware.** The TI `.cfg` reading, the board
presets' element layouts and the DCA1000 layouts below were written from TI's
published documentation. They have not been checked against a real board or
a real capture; the presets' profiles are representative, not TI's demo
profiles. Check them against one capture of your own before relying on them.

## Raw ADC captures: DCA1000

`rawdata.write_dca1000(path, result, adc=radar.adc)` (or `scale=` codes per
unit) writes a beat-domain FMCW result as a post-processed DCA1000 capture:
little-endian int16, no packet headers, frames then chirps in transmission
order (TDM slot `c * num_tx + t`), then

* `layout="xwr16xx"` (two LVDS lanes; xWR16xx, 18xx, 68xx): complex samples
  receiver by receiver in groups `I[n], I[n+1], Q[n], Q[n+1]`; real samples
  receiver by receiver;
* `layout="xwr14xx"` (four lanes): complex samples sample by sample, the `I`
  of every receiver then the `Q`; real samples sample by sample.

`I + jQ` is the cube's beat phasor; `iq_swap=True` writes `Q` first. With
`adc=` the codes are exact; codes outside int16 are refused unless
`clip=True`. `rawdata.read_dca1000(path, num_tx=, num_rx=, chirps=, samples=,
layout=, real=, iq_swap=, scale=)` reads a capture into
`[frame, tx, rx, chirp, sample]`, divided by `scale`; feed a frame to
`ProcessingCube.from_tensor` (see [Processing](processing.md)).

```python
adc = Adc(bits=12, full_scale=0.2)
digital = radar.replace(noise=Noise(figure=10.0), lna_gain=40.0, agc=Agc(target_rms=0.05), adc=adc)
captured = digital.simulate(scene, targets, times=(0.0, 0.05))
with tempfile.TemporaryDirectory() as tmp:
    file = rawdata.write_dca1000(Path(tmp) / "adc_data.bin", captured, adc=adc)
    size = file.stat().st_size
    codes = rawdata.read_dca1000(file, num_tx=1, num_rx=4, chirps=16, samples=256)
print(f"{size} bytes = 2 frames x 16 chirps x 4 receivers x 256 samples x 2 (I, Q) x 2 bytes")
print("read back as codes", tuple(codes.shape), codes.dtype, "| codes x step == cube:",
      torch.allclose(codes * adc.step, captured.cube.cpu(), atol=1e-7))
assert torch.allclose(codes * adc.step, captured.cube.cpu(), atol=1e-7)
```

## Ground truth: `labels`

`labels.ground_truth(radar, scene, targets, times=, pose=None)` returns
`TargetLabels` with `[frame, target]` float64 tensors computed from the
simulated geometry at each frame instant: `world_position_m`, `position_m`
(radar frame: `x` right, `y` boresight, `z` up, the point cloud's frame),
`range_m`, `radial_velocity_mps` (closing positive), `azimuth_rad`,
`elevation_rad`, `rcs_m2` and `target_id`; `site_id`/`site_target` map a
composed row's site to its target. `labels.GroundTruth(...).at(t)` gives one
instant. The velocity is a central difference over `velocity_step_s` (1 ms)
and is good to about 5e-4 m/s. A `MeshTargets` structure reports its
area-weighted facet centroid, area and facet range extent, and NaN for the
cross section (the facets' coherent sum is the simulation).

```python
truth = labels.ground_truth(radar, scene, targets, times=(0.0, 0.05))
print("range", [round(v, 3) for v in truth.range_m[:, 0].tolist()], "m | radial velocity",
      [round(v, 3) for v in truth.radial_velocity_mps[:, 0].tolist()], "m/s | rcs", truth.rcs_m2[0, 0].item())
assert abs(truth.radial_velocity_mps[0, 0].item() - 1.0) < 1e-3
```

`labels.path_summary(paths, frame=0, observation=-1)` lists one traced
observation's composed rows: the Radar component class, the Channel leg
components, interaction depth, float64 delay, Doppler, amplitude and phase.
The Doppler is the frame mean from its first and last observations (NaN for
an adaptive frame, exactly zero for a static one):

```python
paths = radar.trace(scene, targets, times=(0.0,), motion=Motion.chirp())
summary = labels.path_summary(paths, frame=0)
print(f"rows {summary.delay_s.shape[0]}: delay {summary.delay_s[0].item() * 1e9:.3f} ns "
      f"(2R/c = {2 * 3.0 / 299_792_458.0 * 1e9:.3f} ns), Doppler {summary.doppler_hz[0].item():.1f} Hz "
      f"(2 v / lambda = {2 * 1.0 * 77e9 / 299_792_458.0:.1f} Hz), components {summary.component_names}")
```

## Datasets: `write_dataset`, `Episode`, `load_episode`

`dataset.write_dataset(root, radar, episodes, **stream_options)` streams every
`dataset.Episode(scene, targets, times, seed=None, name=None, pose=None,
options={})` frame by frame and writes one saved result per episode (each
frame file also carries its labels), then `root/dataset.json` (the radar's
`to_dict`, the options and the episode index). A seed is
`radar.replace(seed=...)` for that episode; `options` are `Radar.stream`
keywords, and an episode's own `options` override the shared ones.
`ad_mode` other than `"none"` is refused. `dataset.load_episode(root / name)`
returns `(Result, TargetLabels)` and `dataset.read_dataset(root)` the index.

```python
with tempfile.TemporaryDirectory() as tmp:
    root = dataset.write_dataset(Path(tmp) / "set", radar.replace(noise=Noise(figure=10.0)), [
        dataset.Episode(scene, targets, times=(0.0, 0.05), seed=1, name="walk-a"),
        dataset.Episode(scene, targets, times=(0.0, 0.05, 0.10), seed=2, name="walk-b"),
    ])
    index = dataset.read_dataset(root)
    episode, episode_labels = dataset.load_episode(root / "walk-b")
print("dataset keys:", sorted(index))
print(f"walk-b: cube {tuple(episode.cube.shape)}, labels {tuple(episode_labels.range_m.shape)} [frame, target]")
```
