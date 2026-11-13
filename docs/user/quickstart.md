# Quickstart

This guide installs WiTwin Radar, runs the smallest useful simulation, explains
it line by line, adds a wall to the world, and shows how to read the result.
Every Python block on this page runs as written, in order, on a CUDA machine.

## Install

```bash
pip install "witwin-radar[channel]"
witwin-radar-doctor
```

- **Simulation needs CUDA.** Propagation (Channel) and waveform synthesis
  (Radar's native library) are CUDA operators with no CPU route. Building a
  `Radar`, reading a configuration and the `witwin.radar.processing` functions
  work without a GPU. Linux and Windows are supported.
- **The `channel` extra** installs `witwin-channel` 0.6 and is limited to
  CPython 3.10-3.13, the versions Channel publishes. On Python 3.14 the extra
  installs nothing: Radar itself installs, but scene-driven simulation needs
  3.10-3.13.
- **`witwin-radar-doctor`** (also `python -m witwin.radar doctor`) prints the
  installed versions, the GPU and its compute capability, the native build's
  identity, the Channel version and Channel's own verdict, and then runs a
  one-target simulation. Run it first on a new machine; its last line is the
  verdict.
- The `smpl` extra adds the SMPL human-body helpers; the SMPL model files are
  not shipped (see [Targets](targets.md#smpl-bodies)).

## The smallest simulation

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
assert abs(float(axes.range_m[rng]) - 3.0) <= axes.range_bin_m
assert abs(float(axes.velocity_mps[doppler]) - 1.0) <= 2 * axes.velocity_bin_mps
```

It prints `range 3.01 m, velocity +0.84 m/s`. The velocity bin of this frame
is 0.42 m/s, so 1 m/s lands between two bins.

Line by line:

- `Fmcw(...)` is one FMCW ramp in SI units: slope 60 MHz/us written as
  `60e12` Hz/s, a 4.4 MHz ADC, 256 samples per chirp, 64 chirps per frame per
  transmitter, the first ADC sample 6 us after the ramp starts, 7 us of idle
  and a 65 us ramp. The chirp period is `idle + ramp_end` = 72 us.
  `Fmcw.from_ti(...)` builds the same record from a TI datasheet's units
  ([Waveforms](waveforms.md)).
- `Radar(...)` is one flat, immutable record. `reference_frequency_hz` is the
  carrier, `power` is the transmit power in dBm (one of the few non-SI fields,
  see [Conventions](conventions.md)), and `tx`/`rx` are element positions in
  the radar's own frame, here in half wavelengths (`antenna_unit`). Four
  receivers half a wavelength apart along the radar's right make a uniform
  linear array. The default pose sits at the origin and looks along world -z
  with +y up.
- `PointTargets(...)` says where the scatterers are and how strongly they
  scatter. `positions` is `(S, 3)` in metres; `rcs` is the radar cross section
  in m^2. `trajectory(t)` returns the same points at time `t` (seconds) as an
  `(S, 3)` tensor; here the target moves toward the radar at 1 m/s. A target
  without a trajectory is static.
- `radar.simulate(scene, targets, times=...)` runs the whole pipeline once per
  frame instant in `times`. `Scene()` is an empty `witwin.core` world: no
  geometry, so the only path is the line of sight to the target.
- `result.frame(0).range_doppler()` is the range-Doppler map of frame 0. Its
  `data` is `[tx, rx, Doppler, range]`; summing `|x|^2` over the array gives a
  power map whose peak is the target.
- `result.axes` holds the physical axes: `range_m` and `velocity_mps` turn a
  bin index into metres and metres per second. Positive velocity is closing.

Because the target moves, the run resamples the world inside the frame. The
default `motion=Motion.auto()` resolves here to error-controlled adaptive
sampling; [Motion](motion.md) explains the choices and their cost.

```python
print(result.motion_sampling, result.semantics)
```

## Add a wall

Geometry comes from `witwin.core`: a `Scene` of `Structure`s, each a geometry
and a material. A wall is a `Mesh` of two triangles. Its vertices are in world
coordinates, so the mesh must be built with `recenter=False`; the default
`recenter=True` silently moves a mesh to its bounding-box centre.

The wall below is a perfectly conducting plane `x = 1 m`, parallel to the
boresight and one metre to the radar's right. The target is static this time.

```python
from witwin.core import Mesh, PhysicalMaterial, Structure

corners = torch.tensor([[1.0, -2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, -8.0], [1.0, -2.0, -8.0]])
wall = Structure(geometry=Mesh(vertices=corners, faces=torch.tensor([[0, 1, 2], [0, 2, 3]]), recenter=False),
                 material=PhysicalMaterial.perfect_conductor())
scene = Scene(structures=(wall,))
still = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0)


def strongest_ranges(result, count):
    """The `count` strongest range bins of frame 0, in metres, nearest first."""
    profile = result.frame(0).range_profile().data.abs().square().sum(dim=(0, 1, 2))  # [range]
    bins = torch.topk(profile, count).indices
    return sorted(round(float(result.axes.range_m[b]), 2) for b in bins)


direct = radar.simulate(scene, still, times=(0.0,), reflections=0)
bounced = radar.simulate(scene, still, times=(0.0,), reflections=1)
print("reflections=0:", strongest_ranges(direct, 1))
print("reflections=1:", strongest_ranges(bounced, 3))
assert strongest_ranges(bounced, 3) == [3.01, 3.31, 3.61]
```

With `reflections=0` only the direct echo at 3 m is composed. With
`reflections=1` each leg may bounce once off the wall: the mirror image of the
target is at `(2, 0, -3)`, 3.61 m from the radar, so the two round trips that
bounce once (out direct, back via the wall, and the reverse) are
`(3 + 3.61) / 2 = 3.31 m` long and the one that bounces both ways is 3.61 m.
The default is `reflections=1`.

The wall itself returns nothing here, because by default only round trips
through a declared target are composed. `environment=True` adds the scene's
own returns; [Multipath](multipath.md) covers it, transmission through walls,
diffraction and importing scenes from files.

## Read the result

`simulate` returns a `Result`. Its `cube` is one complex tensor with axes
`[frame, tx, rx, slow, fast]`, named by `axis_names`; for this FMCW radar the
slow axis is the chirp and the fast axis is the range bin, because the default
FMCW output is the range spectrum.

```python
print(bounced.axis_names, tuple(bounced.cube.shape), bounced.cube.dtype)
print(f"range bin {bounced.axes.range_bin_m * 100:.2f} cm, max range {bounced.axes.max_unambiguous_range_m:.2f} m, "
      f"velocity bin {bounced.axes.velocity_bin_mps:.3f} m/s")
frame = bounced.frame(0)
print(frame.index, frame.time, tuple(frame.cube.shape))
```

- `axis_names` is the tuple of names, for example
  `('frame', 'tx', 'rx', 'chirp', 'range_bin')`.
- `axes` is the `ProcessingAxes` record every processing stage reads: the
  range and velocity axes (`range_m`, `velocity_mps`, float64), their bin
  sizes, the unambiguous range and speed, the wavelength, the phasor
  convention and the array layout. It is built once from the waveform and the
  array that produced the cube.
- `frame(i)` returns a `Frame`: frame `i`'s cube `[tx, rx, slow, fast]` (a
  view, not a copy) paired with `axes` and the frame's instant `time`. Its
  methods `range_profile()`, `range_doppler()`, `range_angle(...)`,
  `points()`, `virtual_cube()` and `image(...)` call `witwin.radar.processing`
  and compute nothing themselves. `range_doppler()` and `range_profile()` use
  rectangular windows by default; `points()` tapers the slow axis with Hann.
- `times`, `sample_times` (every instant the world was observed, per frame),
  `motion_sampling`, `path_set_complete`, `motion_sampling_exhaustive`,
  `semantics` and `events` describe how the cube was produced
  ([Motion](motion.md)). `Result.save(path)` and `Result.load(path)` write and
  read a result as plain tensors and JSON.

## The four verbs

**`simulate`** runs a session to completion and returns one `Result` holding
every frame. Peak device memory grows with the number of frames, because the
stacked cube is held at once.

**`stream`** runs the identical session and yields one one-frame `Result` per
instant in `times`. Its frames are bit-identical to `simulate`'s, and peak
memory does not grow with the frame count, provided you do not keep every
yielded result. Arguments are validated when iteration starts.

**`trace`** runs only the world half: sampling the world, propagation,
composing round trips, scattering and antenna weighting. It returns a `Paths`
that keeps every evaluated observation's rows, so a long, densely sampled trace
costs device memory. `Paths.rows()` gives the composed rows of one observation
(delays, complex transfers, identities).

**`echo`** runs the instrument half over a `Paths`: waveform synthesis, the
receive chain and the output transform. `radar.echo(radar.trace(...))` is
bit-identical to `radar.simulate(...)`. One trace can be echoed by radars that
differ in their receive chain, seed or FMCW output domain; a radar that changes
what the trace was sampled for (the carrier, the array, the waveform behind a
sampled schedule) is refused by name.

```python
from witwin.radar import Noise

paths = radar.trace(scene, still, times=(0.0,), reflections=1)
assert torch.equal(radar.echo(paths).cube, bounced.cube)
noisy = radar.replace(noise=Noise(figure=12.0), seed=7).echo(paths)  # same world, a noisy receiver
frames = list(radar.stream(scene, still, times=(0.0, 0.1), reflections=1))
print(paths.rows().path_count, "rows;", len(frames), "streamed frames;", noisy.cube.dtype)
assert torch.equal(frames[0].cube[0], bounced.cube[0])
```

## Where to go next

- [Conventions](conventions.md): frames, units, phase and sign conventions.
- [Motion](motion.md): how often the world is resampled, and what that costs.
- [Targets](targets.md): point targets, aspect, polarimetry, tables,
  fluctuation, structure anchors, mesh facets and SMPL bodies.
- [Multipath](multipath.md): reflections, transmission, diffraction, the
  environment's own returns, and importing scenes.
- [Waveforms](waveforms.md): FMCW, MIMO, OFDM and pulsed radars.
- [Pipeline guide](../pipeline_guide.md): the full contract.
