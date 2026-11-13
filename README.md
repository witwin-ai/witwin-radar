# WiTwin Radar

WiTwin Radar is a GPU-accelerated, differentiable radar simulator. You describe a world with `witwin.core` (meshes,
materials, moving structures), say where the scatterers are and how a radar is built, and Radar returns the cube the
radar would record: FMCW, OFDM or pulsed, MIMO, through a modelled receive chain, with typed signal-processing
products (range-Doppler maps, detections, point clouds, tracks, images) on top. One-way propagation comes from
`witwin.channel`; Radar composes round trips, scatters off point, structure and physical-optics mesh targets, applies
antennas and the receiver, and synthesizes the waveform in native CUDA. Gradients flow back to target positions,
trajectories, cross sections, mesh vertices, antenna patterns, the radar's pose and gains.

## Install

```bash
pip install "witwin-radar[channel]"
witwin-radar-doctor
```

- Simulation needs an NVIDIA GPU with CUDA. The wheels ship one native library built for `torch==2.10.0` (CUDA 12.8);
  `witwin-radar-doctor` checks Torch, the native library and Channel and runs a GPU smoke test, with a fix for each
  problem it finds.
- The `channel` extra installs `witwin-channel` 0.6 on CPython 3.10-3.13, the versions Channel publishes. Radar
  installs on 3.10-3.14, but on 3.14 the extra installs nothing, so scene-driven simulation needs 3.10-3.13.
- Channel picks its ray-tracing backend when it compiles a scene: OptiX where the driver provides it, otherwise a
  pure-CUDA tracer. `witwin-channel-doctor` reports which one a machine gets.
- Linux and Windows are supported. Building from source: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quickstart

One FMCW radar, one target 3 m away walking toward it at 1 m/s, one Range-Doppler map:

```python
import torch
from witwin.core import Scene
from witwin.radar import Fmcw, PointTargets, Radar

chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=256, chirps_per_frame=64,
             adc_start=6e-6, idle=7e-6, ramp_end=65e-6)
radar = Radar(reference_frequency_hz=77e9, waveform=chirp, power=15.0, antenna_unit="half_wavelength",
              tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
# One 1 m^2 target 3 m down the boresight (-z), closing at 1 m/s.
targets = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0,
                       trajectory=lambda t: torch.tensor([[0.0, 0.0, -3.0 + t]], device="cuda"))
result = radar.simulate(Scene(), targets, times=(0.0,))
power = result.frame(0).range_doppler().data.abs().square().sum(dim=(0, 1))  # [Doppler, range]
doppler, rng = divmod(int(power.argmax()), power.shape[1])
axes = result.axes
print(f"range {axes.range_m[rng]:.2f} m, velocity {axes.velocity_mps[doppler]:+.2f} m/s")
```

It prints `range 3.01 m, velocity +0.84 m/s`: the nearest range bin, and the Doppler bin nearest 1 m/s (bins are
0.42 m/s wide for 64 chirps). Positive velocity means closing. The radar sits at the origin looking down `-z` with
`+y` up, its default pose. The [quickstart guide](docs/user/quickstart.md) walks through it line by line and adds a
wall, multipath and a point cloud.

## Where to go next

| Guide | What it covers |
|---|---|
| [Quickstart](docs/user/quickstart.md) | Install, the first map, a wall, the four verbs |
| [Conventions and units](docs/user/conventions.md) | Frames, the default pose against Channel's Z-up world, velocity sign, phasors, units |
| [Choosing `Motion`](docs/user/motion.md) | Adaptive (the default for moving FMCW), `adc`, `chirp`, discovery cadences, path events |
| [Targets](docs/user/targets.md) | Point, structure and mesh (physical optics) targets, materials, roughness, fluctuation, SMPL bodies |
| [Multipath and environment](docs/user/multipath.md) | Reflections, transmission, diffraction, clutter, scene authoring with Channel's importers and presets |
| [Waveforms and MIMO](docs/user/waveforms.md) | FMCW, chirp sequences, DDM/BPM, OFDM, pulsed and coded pulses |
| [Receive chain](docs/user/receive-chain.md) | Patterns, coupling, leakage, LNA, compression, noise, IQ, AGC, ADC |
| [Multi-radar networks](docs/user/networks.md) | Several radars, bistatic links, interference, emitters |
| [Processing cookbook](docs/user/processing.md) | Range-Doppler, CFAR, angles, point clouds, tracking, imaging, calibration |
| [Gradients and inverse design](docs/user/gradients.md) | `ad_mode`, the differentiable leaves, an inverse-design loop |
| [Datasets and tools](docs/user/datasets-and-tools.md) | Doctor, save/load, labels, the dataset writer, TI configs, DCA1000 |
| [Troubleshooting](docs/user/troubleshooting.md) | Common errors and what they mean |

Upgrading from 0.4: [`docs/migration-0.5.md`](docs/migration-0.5.md). What changed: [`CHANGELOG.md`](CHANGELOG.md).
What is supported, feature by feature: [`FEATURE_LIST.md`](FEATURE_LIST.md); at run time, `witwin.radar.capabilities()`.
The full pipeline contract: [`docs/pipeline_guide.md`](docs/pipeline_guide.md); the flat configuration format:
[`docs/configuration.md`](docs/configuration.md). Performance evidence and limits: [`PERFORMANCE.md`](PERFORMANCE.md).

## Examples

```bash
python -m examples.single_point          # a target and a wall: the radar equation and the multipath range, checked
python -m examples.moving_target         # moving targets, adaptive motion against the exhaustive reference
python -m examples.walking_human         # a walking body and its micro-Doppler spectrogram
python -m examples.music_imaging         # MUSIC angle imaging
python -m examples.rgbd_range_doppler --input path/to/depths.npz
```

All scene-driven examples need CUDA and Channel.

## The API in one screen

`Radar` is one flat immutable record in SI units (exceptions: `power` in dBm, gains in dB/dBi). A sub-record appears
only where a field has variants (the waveform, the antenna pattern, the motion sampling) or where several numbers are
coupled (`Noise`, `Agc`, `Adc`). `Radar.replace(...)` and `Radar.to(device)` return a new radar; `Radar.from_dict` /
`to_dict` read and write the flat configuration format, including TI-style FMCW keys.

The radar has four verbs, two halves and their fusion:

- `radar.trace(scene, targets, times=...)` runs the world half (world sampling, Channel discovery and replay,
  round-trip composition, scattering, antenna weighting) and returns `Paths`;
- `radar.echo(paths)` runs the instrument half (waveform synthesis, the receive chain, the output domain, the
  processing axes) and returns a `Result`;
- `radar.simulate(...)` fuses the two and stacks every frame; `radar.stream(...)` yields one frame at a time.

`radar.echo(radar.trace(...))` is bit-identical to `radar.simulate(...)`. The world-facing verbs take the
propagation request (`los=`, `reflections=`, `transmission=`, `diffraction=`, `environment=`), the in-frame sampling
(`motion=`) and the differentiation mode (`ad_mode=`, Channel's name). A `Result` holds the cube
`[frame, TX, RX, slow, fast]` with its `axis_names` and the `ProcessingAxes` every processing stage reads;
`result.frame(i)` is a `Frame` whose methods (`range_profile`, `range_doppler`, `range_angle`, `points`, `image`, ...)
are facades over `witwin.radar.processing`.

The root exports the whole happy path: `Adc`, `Agc`, `Aspect`, `Chirp`, `Compression`, `Coupling`, `Emitter`, `Fluctuation`, `Fmcw`, `Frame`, `Iq`, `Leakage`, `MeshTargets`, `Motion`, `Network`, `Noise`, `Ofdm`, `Paths`, `Pattern`, `PointTargets`, `Pulsed`, `Radar`, `RcsTable`, `Result`, `ScatteringMatrix`, `StructureTargets`, `capabilities` and `processing`. Advanced records stay importable from their owner modules.

## Architecture

```text
witwin.core Scene/DynamicScene
        |
        v
Radar session -> Channel one-way legs -> round-trip composition (native join)
        |                                     |
        +-> scattering -> antennas ----------+
                                              v
                      FMCW / OFDM / pulsed synthesis -> receive chain
                                              v
                                typed processing products
```

There is one native backend (`witwin/radar/cuda/`), with no CPU or Torch fallback for simulation; Channel is reached
only through `witwin/radar/channel.py`. Module ownership and the development rules are in [`AGENTS.md`](AGENTS.md);
how to build, test and contribute is in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License and citation

WiTwin Radar uses the WiTwin dual-license model. See the [WiTwin licensing page](https://witwin.ai/license). The
simulator is derived from [RF-Genesis](https://github.com/Asixa/RF-Genesis); cite the RF-Genesis SenSys 2023 paper
when that prior work is relevant.
