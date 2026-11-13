# Troubleshooting

The common failures, what the error says, and the fix. Start with
`witwin-radar-doctor` (see [Datasets and tools](datasets-and-tools.md)): it
checks PyTorch, CUDA, the native library and Channel, runs a GPU smoke test,
and prints a fix for every problem it finds.

The examples below share this setup:

```python
import dataclasses
import math

import torch
from witwin.core import Mesh, Scene
from witwin.radar import Fmcw, Motion, PointTargets, Radar, labels
from witwin.radar import processing as P

chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=256, chirps_per_frame=16,
             adc_start=6e-6, idle=7e-6, ramp_end=65e-6)
radar = Radar(reference_frequency_hz=77e9, waveform=chirp, power=15.0, antenna_unit="half_wavelength",
              tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
scene = Scene()
```

## Installation and runtime

### No CUDA, or a radar on the CPU

Simulation runs on CUDA only: Channel's propagation and Radar's synthesis are
native CUDA operators with no CPU route. A radar on the CPU is fine for
configuration and processing work, but its verbs refuse before any work:

```python
try:
    radar.to("cpu").simulate(scene, PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0), times=(0.0,))
except RuntimeError as error:
    print(str(error)[:120], "...")
```

Fix: `Radar(..., device="cuda")` (the default) or `radar.to("cuda")`. If
PyTorch sees no GPU at all, the runtime check says
`Radar runtime requirements failed: CUDA is unavailable; Radar has no CPU backend`:
check `nvidia-smi` and `CUDA_VISIBLE_DEVICES`, install a CUDA build of PyTorch
rather than a CPU-only one, and in a container run with `--gpus all`.

### The native library refuses to load

The native library records the exact PyTorch, CUDA, ABI and source identity
it was built with, and the loader refuses a mismatch rather than risk
undefined behaviour. The errors (`RadarExtensionABIError`) name the field:

```text
<lib> does not match the active runtime (torch_version: record '2.10.0' vs runtime '2.9.1')
<lib> was built for CUDA architectures ['8.6'], which cannot run on CUDA device 0 (<GPU>, sm_90).
    Rebuild it with TORCH_CUDA_ARCH_LIST including 9.0 (`python scripts/build_radar_cuda_prebuilt.py`),
    or install a release wheel, whose architecture list covers it.
<lib> was not built from the sources shipped beside it: source_fingerprint expected <sha>, found <sha>
the radar native library <lib> has no build record at <sidecar>; rebuild it with
    `python scripts/build_radar_cuda_prebuilt.py`
```

* **PyTorch version.** The release wheels are built for `torch==2.10.0`
  (CUDA 12.8). Install that PyTorch, or build the native library against the
  one you have from a source checkout.
* **GPU architecture.** The library carries code for a list of SM
  architectures; a GPU outside it is refused (the doctor reports
  `GPU SM <n> is outside the declared build SM values` or
  `the installed radar native library contains no sm_<n> image`). Rebuild with
  `TORCH_CUDA_ARCH_LIST=<major.minor> python scripts/build_radar_cuda_prebuilt.py`.
* **Stale fingerprint.** In a source checkout, any edit of a native source
  (`witwin/radar/cuda/`) changes the source fingerprint, and the old binary is
  refused until you rebuild (`scripts/dev_build.sh` on Linux).

### Channel missing, or the wrong version

Scene simulation (`trace`, `simulate`, `stream`) needs `witwin-channel` 0.6;
without it the first verb fails with `ModuleNotFoundError: No module named
'witwin.channel'`, and the doctor warns
`witwin-channel is not installed, so Radar cannot simulate a scene`. Install
it with `pip install 'witwin-radar[channel]'`. A Channel outside
`>=0.6.0,<0.7` is reported by the doctor as unsupported:
`pip install 'witwin-channel>=0.6.0,<0.7'`.

**Python 3.14.** Radar installs on Python 3.10-3.14, but Channel publishes
wheels for 3.10-3.13 only, so on 3.14 the `channel` extra installs nothing and
scene simulation is unavailable. Use Python 3.10-3.13 for simulation.

### OptiX or the pure-CUDA tracer

Channel picks its ray-tracing backend when it compiles a scene: OptiX where the
driver provides it (driver R555 or newer), otherwise a slower pure-CUDA
tracer. `witwin-channel-doctor` (or `python -m witwin.channel doctor`) reports
which one a machine gets, and warns
`OptiX is unusable, so scenes trace on the slower pure-CUDA backend (<reason>)`.
In a container, OptiX also needs
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`. The environment
variable `WITWIN_CHANNEL_TRACE_BACKEND` (`auto`, `optix` or `cuda`) forces a
choice; forcing `optix` where it is unusable is an error. The results are the
same either way; only the speed differs.

## Geometry and motion

### The radar looks at the floor in a Z-up scene

A radar's local frame is (right, up, -boresight) and its default pose is
`position=(0, 0, 0)`, `look_at=(0, 0, -1)`, `up=(0, 1, 0)`: a Y-up world
looking down `-z`. Channel and Core scenes are usually authored with `+z` up,
and there the default radar looks straight at the floor. The labels show
where a target really is relative to the boresight:

```python
ahead = PointTargets(positions=[(3.0, 0.0, 1.0)], rcs=1.0)  # 3 m along +x, 1 m up, in a Z-up scene
for name, r in (("default pose", radar),
                ("look_at=(1, 0, 0), up=(0, 0, 1)", radar.replace(look_at=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)))):
    truth = labels.ground_truth(r, scene, ahead, times=(0.0,))
    print(f"{name:32s} target at azimuth {math.degrees(truth.azimuth_rad[0, 0].item()):+6.1f} deg, "
          f"elevation {math.degrees(truth.elevation_rad[0, 0].item()):+6.1f} deg")
```

Fix: state the pose in a Z-up scene, e.g. `Radar(..., position=(0, 0, 1),
look_at=(1, 0, 1), up=(0, 0, 1))`. With isotropic elements a target behind or
beside the radar still returns an echo, so a wrong pose is easy to miss; it
shows up as wrong angles, and as missing targets once the elements are
directional.

### A mesh is not where it was authored

`witwin.core.Mesh` defaults to `recenter=True`, which silently moves the mesh
so its bounding-box centre is at the origin (then applies `position`). For
vertices authored in world coordinates, pass `recenter=False`:

```python
plate = torch.tensor([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
faces = torch.tensor([[0, 1, 2]])
for recenter in (True, False):
    mesh = Mesh(vertices=plate, faces=faces, recenter=recenter, fill_mode="surface", topology_diagnostics=False)
    print(f"recenter={recenter}: first vertex lands at {mesh.world_vertices[0].tolist()}")
```

### Zero Doppler: the target has no trajectory

A `PointTargets` without `trajectory=` is static: its positions hold for the
whole run and its Doppler is exactly zero. Motion comes from a callable
`trajectory(t) -> [S, 3]` tensor (world metres at time `t` seconds), from
`StructureTargets`/`MeshTargets` on a moving Core structure, or from the
radar's own `pose=`. Positive radial velocity means closing.

```python
static = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0)
walking = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0,
                       trajectory=lambda t: torch.tensor([[0.0, 0.0, -3.0 + t]], device="cuda"))
for name, targets in (("no trajectory", static), ("closing at 1 m/s", walking)):
    result = radar.simulate(scene, targets, times=(0.0,))
    power = result.frame(0).range_doppler().data.abs().square().sum(dim=(0, 1))
    doppler = int(power.max(dim=1).values.argmax())
    print(f"{name:17s}: strongest velocity bin {result.axes.velocity_mps[doppler]:+.2f} m/s "
          f"(bins of {result.axes.velocity_bin_mps:.2f} m/s; schedule {result.motion_sampling})")
```

## Processing

### CFAR finds nothing (or too much): the input is magnitude or dB

The CFAR detectors are square-law: a complex map is detected on `|x|^2`, and a
real map is read as power. Their thresholds are solved for exponentially
distributed cell power, so a magnitude or dB map breaks the false-alarm rate
by orders of magnitude. Pass the complex map, or power
(`P.combine_coherent(rd.data)` returns the boresight power):

```python
generator = torch.Generator().manual_seed(0)
noise = torch.complex(torch.randn(128, 256, generator=generator), torch.randn(128, 256, generator=generator))
for name, rd_map in (("complex", noise), ("power", noise.abs().square()), ("magnitude (wrong)", noise.abs())):
    found = P.ca_cfar(rd_map, guard_cells=(2, 4), training_cells=(4, 8), pfa=1e-2)
    print(f"{name:18s}: false-alarm rate {found.mask.float().mean().item():.4f} (asked for 0.0100)")
```

`frame.points()` does this for you; `P.point_cloud` groups peaks by default.

## Completeness and diagnostics

### Is the path set complete? `Result.semantics`

Two statements describe how far to trust a run:
`Result.path_set_complete` (no path birth within the requested bounds can
have been missed) and `Result.motion_sampling_exhaustive` (the world was
evaluated at every instant the waveform samples). `Result.semantics` gives
both in Channel's vocabulary (`complete`, `exhaustive`), with
`estimator="coherent"` and `combine="field"`. `Result.discovery_complete` is
Channel's own half: it is `False` when reflection candidates came from ray
discovery rather than the image method (`discovery="rays"`, or `"auto"` past
its cost budget).

The default `Motion.auto()` resolves to `Motion.adaptive()` for a moving FMCW
scene, which interpolates between probes: fast, but neither complete nor
exhaustive. `Motion.adc()` evaluates every ADC sample, the exhaustive
reference, and costs one world evaluation per ADC sample (here a reduced
4-chirp, 64-sample waveform keeps that small); `Motion.chirp()` evaluates once
per chirp.

```python
walking_result = radar.simulate(scene, walking, times=(0.0,))
small = radar.replace(waveform=dataclasses.replace(chirp, samples_per_chirp=64, chirps_per_frame=4))
print("auto (adaptive):", walking_result.semantics)
print("Motion.chirp(): ", radar.simulate(scene, walking, times=(0.0,), motion=Motion.chirp()).semantics)
print("Motion.adc():   ", small.simulate(scene, walking, times=(0.0,), motion=Motion.adc()).semantics)
```

If a result must be complete, use `Motion.adc()` (complete and exhaustive) or
`Motion.chirp()` (complete, one world evaluation per chirp), and check
`path_set_complete` before using the data.

### Adaptive diagnostics

An adaptive frame publishes a plain-dict record in
`Result.adaptive_diagnostics` (one per adaptive frame): how many world
evaluations it made, the accepted intervals, the largest tested phase and
amplitude errors against `Motion.adaptive(phase_error=,
relative_amplitude_error=)`, and its `complete`/`exhaustive` flags. A frame
whose `max_tested_phase_error_rad` sits near the tolerance is a frame to
re-run with `Motion.adc()` or a smaller `max_interval`.

```python
record = walking_result.adaptive_diagnostics[0]
print({key: record[key] for key in ("evaluations", "observation_count", "accepted_intervals",
                                    "max_tested_phase_error_rad", "complete")})
```

## Memory

* **`stream` instead of `simulate`.** `Radar.simulate` stacks every frame into
  one `[frame, tx, rx, slow, fast]` cube; `Radar.stream` yields one frame at a
  time with the same numbers, so a long sequence never has to fit at once.
* **`Paths` retains everything.** `Radar.trace` keeps every evaluated
  observation's rows so that `Radar.echo` can synthesize them later; for an
  `adc` schedule that is one row set per ADC sample. Use `simulate`/`stream`
  unless you need the paths, and trace few frames.
* **`batch_observations`.** The fused verbs hold one replay group at a time,
  at most `Motion.batch_observations` observations (256 by default). Lower it
  to cap peak memory on large scenes, e.g.
  `Motion.adaptive(batch_observations=64)` or, for the exhaustive schedule,
  `dataclasses.replace(Motion.adc(), batch_observations=64)`.
* A `Result` keeps the closing frame's world state (`last_snapshot`,
  `last_compiled_scene`, `last_propagation`, `last_radar_paths`) and, with
  `ad_mode` set, that frame's autograd graph; drop results you no longer need.

```python
for part in radar.stream(scene, walking, times=(0.0, 0.05, 0.10)):
    print(f"frame at {part.times[0]:.2f} s: cube {tuple(part.cube.shape)}")
```

## Old names raise after 0.5

0.5.0 renamed public names in one pass, with no aliases, so code written for
0.4 fails loudly (`TypeError` for a keyword, `AttributeError` for an
attribute) rather than doing something else:

```python
for attempt in (lambda: Radar(carrier=77e9, waveform=chirp, tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0),), power=10.0),
                lambda: radar.simulate(scene, static, times=(0.0,), grad="vjp"),
                lambda: walking_result.times_s):
    try:
        attempt()
    except (TypeError, AttributeError) as error:
        print(f"{type(error).__name__}: {error}")
```

`carrier` is now `reference_frequency_hz`, `grad=` is `ad_mode=`,
`times_s` is `times`, `Iq(phase_deg=)` is `Iq(phase=)` in radians, and so on;
the full table is in the [0.5 migration guide](../migration-0.5.md).
