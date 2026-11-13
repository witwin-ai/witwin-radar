# Conventions: frames, units, signs and phases

This page states the coordinate frames, units and sign conventions every other
guide relies on. Each section has a runnable check on a small radar.

## Setup

```python
import cmath
import math

import torch
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure
from witwin.radar import Fmcw, PointTargets, Radar

chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=256, chirps_per_frame=64,
             adc_start=6e-6, idle=7e-6, ramp_end=65e-6)
radar = Radar(reference_frequency_hz=77e9, waveform=chirp, power=15.0, antenna_unit="half_wavelength",
              tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)))


def peak(result):
    """(range m, velocity m/s) of the strongest range-Doppler cell of frame 0."""
    power = result.frame(0).range_doppler().data.abs().square().sum(dim=(0, 1))
    doppler, rng = divmod(int(power.argmax()), power.shape[1])
    return float(result.axes.range_m[rng]), float(result.axes.velocity_mps[doppler])
```

## World frame and the radar's own frame

Everything in a scene - mesh vertices, target positions, trajectories,
`Radar.position` and `Radar.look_at` - is in the **world frame**, in metres.

The radar has its own **local frame**, `(right, up, -boresight)`: its x axis
points to the radar's right, y up, and z backwards (the boresight is local
-z). The pose fields build it: `position` is the origin, `look_at` the point
the boresight passes through, and `up` the world vector that completes the
frame (it need not be exactly perpendicular to the boresight, only not
parallel to it).

**Element positions** in `tx` and `rx` are offsets in this local frame. The
array above puts its receivers along local x, the radar's right.
`antenna_unit` says what they are measured in: `"m"` (the `Radar` default) or
`"half_wavelength"`, half the carrier wavelength (the default of the flat
configuration format, `Radar.from_dict`). Half-wavelength offsets describe the
same electrical array at any carrier.

### Y-up default, Z-up scenes

The default pose is `position=(0, 0, 0)`, `look_at=(0, 0, -1)`, `up=(0, 1, 0)`:
a **Y-up** world, looking along world -z. Channel and Core document **+Z up**.
In a Z-up scene world -z is down, so the default radar looks straight at the
floor. A radar in a Z-up scene states its pose:

```python
zup = radar.replace(look_at=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))  # boresight +x, up +z
result = zup.simulate(Scene(), PointTargets(positions=[(5.0, 0.0, 0.0)], rcs=1.0), times=(0.0,))
print(f"Z-up radar, target 5 m along +x: range {peak(result)[0]:.2f} m")
assert abs(peak(result)[0] - 5.0) <= result.axes.range_bin_m
```

`Radar.replace(...)` returns a new radar; a radar is never edited in place.

## The processing frame and `PointCloud.to_world`

Detections from `Frame.points()` (and `processing.point_cloud`) are in the
**processing frame**: origin at the radar, `x` to the radar's right, `y`
along the boresight, `z` up. This is not the local frame above (whose third
axis points backwards). `PointCloud.to_world(radar)` maps a cloud into the
world through the same pose the radar placed its elements with. Range and
velocity are scalars and do not change.

```python
target_world = (1.0, 0.0, -3.0)  # 1 m to the right of the boresight, 3 m ahead
result = radar.simulate(Scene(), PointTargets(positions=[target_world], rcs=1.0), times=(0.0,))
cloud = result.frame(0).points()
strongest = int(cloud.energy.argmax())
local_xyz = cloud.xyz[strongest].tolist()
world_xyz = cloud.to_world(radar).xyz[strongest].tolist()
print("processing frame:", [round(v, 2) for v in local_xyz], " world:", [round(v, 2) for v in world_xyz])
assert local_xyz[0] > 0.5 and local_xyz[1] > 2.5  # to the right, ahead
assert world_xyz[0] > 0.5 and world_xyz[2] < -2.5
```

A four-element array resolves angle coarsely, so the detection lands within a
few tenths of a metre of the target; the signs are what this check is about.

## Velocity: positive is closing

Every waveform publishes radial velocity with **positive meaning closing**
(the target approaching the radar), in `ProcessingAxes.velocity_mps`, in
range-Doppler maps and in point clouds. The cube's raw slow-time phase
progresses in opposite directions for FMCW and for OFDM/pulsed (see the next
section); processing reconciles the two exactly once, from the cube's
published phasor.

A target **without `trajectory=`** is static: it has zero Doppler however
fast anything else is declared to be. Motion is only ever positions that
change with time.

```python
def mover(speed):
    """A target 3 m ahead, moving toward the radar at `speed` m/s (negative is receding)."""
    return PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0,
                        trajectory=lambda t: torch.tensor([[0.0, 0.0, -3.0 + speed * t]], device="cuda"))


closing = peak(radar.simulate(Scene(), mover(2.0), times=(0.0,)))[1]
receding = peak(radar.simulate(Scene(), mover(-2.0), times=(0.0,)))[1]
static = peak(radar.simulate(Scene(), PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0), times=(0.0,)))[1]
print(f"closing {closing:+.2f} m/s, receding {receding:+.2f} m/s, static {static:+.2f} m/s")
assert closing > 0 > receding and static == 0.0
```

## Phasors and the complex baseband

The cube is complex baseband. Channel's conventions are a phasor
`exp(-j k d)` for a path of length `d` and a time dependence
`exp(+j 2 pi f t)`, and a result says which one its cube is in:

- **FMCW** cubes are de-chirped, which conjugates the echo:
  `Result.phasor == "conj(exp(-j*k*d))"`. A target at positive range is a
  positive beat frequency, and a path that lengthens by `dd` advances the
  bin's phase by `+2 k dd` (round trip).
- **OFDM** and **pulsed** cubes are in Channel's phasor unchanged:
  `"exp(-j*k*d)"`.

`Result.time_dependence` is `"exp(+j*2*pi*f*t)"` for every waveform. Processing
reads `phasor` as data and never infers it from the waveform's name.

Without a receive chain the cube is in sqrt(W) at the receive port; with one
it is in volts across `Radar.impedance` ([Receive chain](receive-chain.md)).
`Fmcw(sampling="real")` publishes the real part only
([Waveforms](waveforms.md)).

## Float64 positions

Target positions and trajectories may be float64 tensors, and the round-trip
path length, hence the carrier phase, is then computed from the float64
positions: sub-micrometre motion survives at any range. A float32 coordinate
near 10 m has a resolution of about 1 um, so smaller displacements vanish.
Positions given as Python lists become float32.

```python
def bin_phase(z, dtype):
    """Carrier phase of the strongest range bin, chirp 0, for a static target at (0, 0, -z)."""
    positions = torch.tensor([[0.0, 0.0, -z]], dtype=dtype, device="cuda")
    profile = radar.simulate(Scene(), PointTargets(positions=positions, rcs=1.0), times=(0.0,)).frame(0).range_profile()
    line = profile.data[0, 0, 0]
    return cmath.phase(complex(line[int(line.abs().argmax())]))


step = 0.2e-6  # 0.2 um
expected = 2.0 * (2.0 * math.pi / radar.wavelength) * step  # +2 k dd in the beat convention
for dtype in (torch.float32, torch.float64):
    moved = bin_phase(10.0 + step, dtype) - bin_phase(10.0, dtype)
    print(f"{dtype}: phase change {moved * 1e3:.3f} mrad (path-length term {expected * 1e3:.3f} mrad)")
    if dtype is torch.float64:
        assert abs(moved - expected) < 0.1 * expected
```

The float64 change matches `2 k dd` to within the small extra phase the
range-bin position itself contributes; the float32 one is zero, because
`10.0000002` and `10.0` are the same float32 number. A reflected path is
exact along the reflecting surface, but Channel's float32 interaction point
leaves an error across it of about half a float32 ulp of the point's
coordinate (about 0.5 um for a wall 4 m away).

## Meshes keep their authored coordinates only with `recenter=False`

`witwin.core.Mesh` defaults to `recenter=True`, which subtracts the bounding
box centre from the vertices: a wall authored at `z = -5` ends up through the
origin, with nothing raised. Build every mesh that is authored in world
coordinates with `recenter=False`. Channel's importers already do.

```python
vertices = torch.tensor([[-1.0, -1.0, -5.0], [1.0, -1.0, -5.0], [1.0, 1.0, -5.0], [-1.0, 1.0, -5.0]])
faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
print("recenter=True :", Mesh(vertices=vertices, faces=faces).world_vertices[0].tolist())
print("recenter=False:", Mesh(vertices=vertices, faces=faces, recenter=False).world_vertices[0].tolist())
assert float(Mesh(vertices=vertices, faces=faces, recenter=False).world_vertices[0, 2]) == -5.0
```

## Units

Every field is in SI units (m, s, Hz, W, rad, m^2, ohm, V, K), with these
documented exceptions:

| Field | Unit | Why |
|---|---|---|
| `Radar.power`, `Emitter.power` | dBm | how datasheets quote transmit power; `Radar.power_w` gives watts |
| `Pattern.gain` | dBi | absolute peak gain of the element |
| `Pattern` angles (`x_angles`, `y_angles`) | degrees | how measured cuts are tabulated |
| `Radar.lna_gain`, `Agc.min_gain`/`max_gain`, `Noise.figure` | dB | |
| `Leakage.isolation`, `Iq.gain` | dB | |
| `Iq.phase`, `Leakage.phase` | rad | |
| `Noise.phase_density`, `phase_flicker_density`, `phase_floor` | dBc/Hz | |
| `Compression.iip3`, `p1db` | dBm; `Compression.am_pm` in degrees | |
| `tx`, `rx` | `antenna_unit` (`"m"` or `"half_wavelength"`) | |
| `Fmcw.from_ti(...)`, `Radar.from_dict(...)` waveform keys | MHz/us, ksps, us | the vendor units of TI configuration files, read only there |

Field names carry no unit suffix, except where they name a quantity Channel
publishes under that name (`reference_frequency_hz`, `power_w`). The
processing package keeps unit-suffixed names (`range_m`, `velocity_mps`,
`range_bin_m`, `times_s`).

```python
print(f"power {radar.power} dBm = {radar.power_w * 1e3:.1f} mW; wavelength {radar.wavelength * 1e3:.3f} mm")
assert abs(radar.power_w - 10 ** (15.0 / 10) * 1e-3) < 1e-12
```

See also: [Quickstart](quickstart.md), [Motion](motion.md),
[Targets](targets.md), [Waveforms](waveforms.md).
