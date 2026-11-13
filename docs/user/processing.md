# Processing cookbook

Everything in this guide works the same on a simulated `Result` and on a
recorded cube. Processing reads the cube's metadata - waveform, output
domain, range and velocity axes, phasor convention, array layout - from a
`ProcessingAxes` record, never from the tensor's shape. The functions and
typed products live in `witwin.radar.processing`; `Result.frame(i)` returns a
`Frame` whose methods are thin facades over them.

Conventions used throughout:

* velocity is radial and **positive when closing**, for every waveform;
* the processing frame (point clouds, `ArrayGeometry`) is `x` right, `y`
  boresight, `z` up, with the origin at the radar; this is not the radar's
  pose-local frame `(right, up, -boresight)`, and `PointCloud.to_world(radar)`
  maps a cloud into the world;
* every transform is amplitude normalised (`1/N`), so an on-bin tone peaks at
  its own amplitude times the window's coherent gain;
* the detectors are square-law: pass complex IQ or power, never magnitude or
  dB.

## Setup: a scene to process

The quickstart radar with receiver noise and two targets in a Y-up world
(the default pose looks down `-z`, with `+x` to its right): one at 3 m on the
boresight, closing at 1 m/s, and one static at 5 m, 20 degrees to the right.
The radar publishes de-chirped beat samples (`Fmcw(output="beat")`) rather
than the default range spectrum, so that the range axis can be windowed and
zero padded below; everything else works on either output.

```python
import math

import torch
from witwin.core import Scene
from witwin.radar import Coupling, Fmcw, Leakage, Noise, PointTargets, Radar
from witwin.radar import processing as P

chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=256, chirps_per_frame=64,
             adc_start=6e-6, idle=7e-6, ramp_end=65e-6, output="beat")
radar = Radar(reference_frequency_hz=77e9, waveform=chirp, power=15.0, antenna_unit="half_wavelength",
              tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
              noise=Noise(figure=10.0))
scene = Scene()
side = (5.0 * math.sin(math.radians(20)), 0.0, -5.0 * math.cos(math.radians(20)))


def two_targets(t):
    return torch.tensor([[0.0, 0.0, -3.0 + t], side], device="cuda")


targets = PointTargets(positions=two_targets(0.0).tolist(), rcs=[1.0, 1.0], trajectory=two_targets)
result = radar.simulate(scene, targets, times=(0.0,))
frame = result.frame(0)
axes = result.axes
print(frame)
print(f"range bin {axes.range_bin_m * 100:.2f} cm over {axes.max_unambiguous_range_m:.1f} m; "
      f"velocity bin {axes.velocity_bin_mps:.3f} m/s, +/-{axes.max_unambiguous_speed_mps:.2f} m/s")
```

## The `Frame` facades

| Facade | Returns | Equivalent |
|---|---|---|
| `frame.processing_cube()` | `ProcessingCube` | the cube `[tx, rx, slow, fast]` with its axes |
| `frame.virtual_cube()` | `ProcessingCube` | `P.demultiplex(...)`: one channel per transmitter (DDM/BPM separated, TDM unchanged) |
| `frame.range_profile(window=, remove_dc=)` | `RangeProfile` | `P.range_profile(...)` |
| `frame.range_doppler(window=, range_window=, remove_dc=)` | `RangeDopplerMap` | `P.range_doppler_map(P.range_profile(...))` |
| `frame.range_angle(azimuth_rad, elevation_rad=, window=)` | `RangeAngleMap` | `P.range_angle_map(...)` of the virtual cube |
| `frame.points(pfa=, guard_cells=, training_cells=, route=, ...)` | `PointCloud` | range-Doppler (Hann), CA-CFAR, point cloud |
| `frame.aperture(radar, velocity_mps=, pose=)` | `Aperture` | `P.Aperture.from_radar(radar, time_s=frame.time, ...)` |
| `frame.image(aperture, pixels_m, range_bins=)` | `SarImage` | `P.backproject(...)` |
| `frame.array()` | `ArrayGeometry` | `P.ArrayGeometry.from_axes(axes)` |

`frame.range_doppler()` is **rectangular** in both axes by default, like every
processing stage; pass `window="hann"` for Doppler sidelobe control.
`frame.points()` is the one facade that tapers by default (`window="hann"`,
on the Doppler axis), because a detector over a rectangular Doppler transform
reports every strong target's sidelobes as targets. A fast-time (range)
window (`range_window=`) is refused on the FMCW spectrum output, which has
already been range-transformed; use `Fmcw(output="beat")` when you want to
taper or zero pad the range axis.

```python
rd = frame.range_doppler(window="hann")
power = P.combine_coherent(rd.data)  # [Doppler, range] power of the boresight sum
d, r = divmod(int(power.argmax()), power.shape[1])
print(f"strongest cell: range {axes.range_m[r]:.2f} m, velocity {axes.velocity_mps[d]:+.2f} m/s")
assert abs(axes.range_m[r] - 3.0) < 2 * axes.range_bin_m and abs(axes.velocity_mps[d] - 1.0) < axes.velocity_bin_mps
```

## A recorded cube

`P.ProcessingAxes.from_radar(radar)` builds the axes a radar's cube is
described by, without simulating; `P.ProcessingCube.from_tensor(cube, radar)`
pairs a recorded complex cube with them. The cube is `[tx, rx, slow, fast]`
(or `[slow, pair, fast]` with the pair axis receiver major,
`pair = rx * num_tx + tx`). It must follow this package's conventions: an
FMCW cube is in the de-chirped beat phasor (a target at positive range is a
positive beat frequency) and ADC samples are `Fmcw(output="beat")`; a device
whose IQ runs the other way is conjugated first.

Here the "recording" is the simulated beat cube, so the two routes must agree
exactly:

```python
recorded = result.cube[0].clone()  # [tx, rx, chirp, sample], as read from a capture
cube = P.ProcessingCube.from_tensor(recorded, radar)
rd_recorded = P.range_doppler_map(P.range_profile(cube))
print("from_radar axes == simulated axes:",
      torch.equal(P.ProcessingAxes.from_radar(radar).range_m.cpu(), axes.range_m.cpu()))
print("recorded-cube map == frame.range_doppler():", torch.equal(rd_recorded.data, frame.range_doppler().data))
assert torch.equal(rd_recorded.data, frame.range_doppler().data)
```

`witwin.radar.rawdata.read_dca1000` reads a DCA1000 capture into this layout;
see [Datasets and tools](datasets-and-tools.md).

## Zero padding

`range_bins=` on `range_profile` and `doppler_bins=` on `range_doppler_map`
zero pad the transforms (FMCW beat and OFDM cubes; a pulsed cube is refined
with `range_oversample` instead). Padding interpolates the spectrum: the bin
shrinks, the span and the resolution do not change, and the amplitude
normalisation is kept.

```python
padded = P.range_doppler_map(P.range_profile(cube, range_bins=1024), doppler_bins=256)
p = P.combine_coherent(padded.data)
d, r = divmod(int(p.argmax()), p.shape[1])
print(f"padded grid {tuple(p.shape)}: range {padded.axes.range_m[r]:.3f} m "
      f"(bin {padded.axes.range_bin_m * 100:.2f} cm), velocity {padded.axes.velocity_mps[d]:+.3f} m/s")
```

## Static clutter removal

`P.remove_static_clutter(profile, method=)` filters along slow time per range
bin: `"mean"` subtracts the slow-time mean (zero Doppler removed exactly,
length kept), `"mti2"` and `"mti3"` are the two- and three-pulse cancellers,
with responses `2|sin(pi f T)|` and `4 sin^2(pi f T)`, and one or two fewer
slow-time samples.

```python
profile = frame.range_profile()
for method in ("mean", "mti2"):
    moving_only = P.combine_coherent(P.range_doppler_map(P.remove_static_clutter(profile, method=method)).data)
    k = int(moving_only.max(dim=0).values.argmax())
    print(f"{method}: strongest return at {axes.range_m[k]:.2f} m")
static_bin = int((axes.range_m - 5.0).abs().argmin())
cleaned = P.combine_coherent(P.range_doppler_map(P.remove_static_clutter(profile)).data)
before = P.combine_coherent(P.range_doppler_map(profile).data)
print(f"static target at 5 m: {10 * math.log10(float(before[:, static_bin].max() / cleaned[:, static_bin].max())):.1f} dB lower")
```

## CFAR detection

`P.ca_cfar`, `P.go_cfar`, `P.so_cfar` and `P.os_cfar` threshold a
`[..., Doppler, range]` map. They are **square-law**: a complex map is
detected on `|x|^2`, and a real map is read as power. Pass complex IQ or
power (`P.combine_coherent` returns the boresight power), never a magnitude
or dB map: the threshold constants are solved for exponentially distributed
cell power, and on a magnitude map the false-alarm rate is wrong by orders of
magnitude. The Doppler axis wraps; at the range edges each cell uses the
training cells that exist and its own constant, so the rate holds there too.

`guard_cells` and `training_cells` are `(doppler, range)` half widths; `pfa`
is the false-alarm probability per cell. `P.cfar_false_alarm_probability`
evaluates the exact law, for checking a configuration:

A Hann taper on both axes keeps the strong targets' range and Doppler
sidelobes below the threshold:

```python
rd = frame.range_doppler(window="hann", range_window="hann")
power = P.combine_coherent(rd.data)
for name, detector in (("ca", P.ca_cfar), ("go", P.go_cfar), ("so", P.so_cfar), ("os", P.os_cfar)):
    found = detector(power, guard_cells=(2, 4), training_cells=(4, 8), pfa=1e-4)
    print(f"{name}-CFAR: {int(found.mask.sum())} cells over threshold")
alpha = torch.tensor(1e-4 ** (-1 / 48) * 48 - 48, dtype=torch.float64)
print("CA law at 48 training cells:",
      float(P.cfar_false_alarm_probability("ca", alpha, torch.tensor(24), torch.tensor(24))))
```

`P.ca_cfar_1d(profile_power, ...)` is the range-only detector.

## Peak grouping

One strong return lights up several cells (its main lobe and window
leakage). `P.group_peaks(detections, power, neighbourhood=(3, 3))` keeps a
detection only where it is the strongest cell of its neighbourhood:

```python
found = P.ca_cfar(power, guard_cells=(2, 4), training_cells=(4, 8), pfa=1e-6)
peaks = P.group_peaks(found, power)
print(f"{int(found.mask.sum())} detected cells -> {int(peaks.mask.sum())} peaks")
```

## Angles

`frame.array()` (or `P.ArrayGeometry.from_axes(axes)`) is the virtual array
in the processing frame. Three estimators take a virtual-antenna snapshot
`[P, N]` (P virtual elements, N snapshots) and return `[3, N]` direction
cosines `(u_x, u_y, u_z)`:

* `P.scan_aoa`: a steering-vector scan; reads any layout (holes, overlaps,
  non-planar). The default route.
* `P.fft2_aoa`: one zero-padded FFT over a planar lattice (ULA or rectangle);
  an array without a lattice is refused.
* `P.phase_comparison_aoa`: the TI three-transmitter layout (an azimuth line
  and an elevated line); anything else is refused by name.

A line array measures one cosine; the other is published as zero.

```python
array = frame.array()
pairs = rd.data.reshape(-1, *rd.data.shape[-2:])  # [virtual element, Doppler, range]
for want in (3.0, 5.0):
    r = int((axes.range_m - want).abs().argmin())
    d = int(power[:, r].argmax())
    column = pairs[:, d, r].reshape(-1, 1)  # one snapshot
    for name in ("scan", "fft2"):
        u = P.AOA_ROUTES[name](column, array)
        print(f"target at {want:.0f} m, {name}: azimuth {math.degrees(math.asin(float(u[0, 0]))):+.1f} deg")
```

`P.tdm_compensate` removes the TDM motion phase before an angle estimate, and
`P.music_spectrum`, `P.music_image`, `P.mvdr_weights`, `P.beam_cube` and
`P.conventional_steering` are the subspace and beamforming building blocks.

### Range-angle map

`frame.range_angle(azimuth_rad)` beamforms the virtual array over an azimuth
grid (radians, positive toward the radar's right) and sums power over
Doppler (`P.range_angle_map` on a `RangeDopplerMap`):

```python
azimuth = torch.linspace(-math.pi / 3, math.pi / 3, 121, dtype=torch.float64)
ra = frame.range_angle(azimuth)
k = int((axes.range_m - 5.0).abs().argmin())
print(f"5 m row peaks at {math.degrees(float(ra.azimuth_rad[int(ra.data[:, k].argmax())])):+.1f} deg")
```

## Point clouds

`P.point_cloud(detections, rd, array, route="scan", peak_neighbourhood=(3, 3),
...)` turns detections on a `RangeDopplerMap[TX, RX, D, R]` into a
`PointCloud`: `xyz` in metres in the processing frame, `velocity_mps`
(closing positive), `energy` (dB) and `range_m`. It groups peaks first by
default. `range_gate_m=(near, far)` gates in metres; `max_points` keeps the
strongest. `cloud.to_world(radar)` places the cloud in the world through the
radar's pose. `frame.points()` does a Doppler-tapered range-Doppler map,
CA-CFAR on the boresight power and the point cloud in one call; it does not
taper the range axis, so for the cleanest cloud from a beat cube spell the
chain out as `detect` does here.

```python
def detect(f, pfa=1e-6):
    rd = f.range_doppler(window="hann", range_window="hann")
    found = P.ca_cfar(P.combine_coherent(rd.data), guard_cells=(2, 4), training_cells=(4, 8), pfa=pfa)
    return P.point_cloud(found, rd, f.array())


cloud = detect(frame)
world = cloud.to_world(radar)
for xyz, v in zip(world.xyz.tolist(), cloud.velocity_mps.tolist()):
    print(f"world ({xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:+.2f}) m, {v:+.2f} m/s")
facade = frame.points(pfa=1e-6)
print(f"frame.points(): {facade.xyz.shape[0]} points in frame {facade.frame!r}")
```

### Clustering

`P.cluster_points(cloud, eps_m=, min_points=, velocity_scale_s=0.0)` is
DBSCAN over the points; with `velocity_scale_s > 0` a velocity difference
counts as the distance it covers in that time. `labels` is `-1` for noise.

```python
clusters = P.cluster_points(world, eps_m=0.5, min_points=1)
print("labels:", clusters.labels.tolist(), "| centroids:", [[round(c, 2) for c in row] for row in clusters.centroid.tolist()])
```

## Tracking

`P.KalmanTracker` is a multi-target constant-velocity Kalman filter over
`P.DetectionFrame` records (`time_s`, `xyz`, `velocity_mps`, `energy`,
`frame_index`); `push(frame)` returns each detection's track id. A frame
sequence from `Radar.stream`:

```python
tracker = P.KalmanTracker(measurement_noise_m=0.1, gate=4.0)
for i, part in enumerate(radar.stream(scene, targets, times=(0.0, 0.05, 0.10, 0.15))):
    f = part.frame(0)
    points = detect(f).to_world(radar)
    ids = tracker.push(P.DetectionFrame(time_s=f.time, xyz=points.xyz, velocity_mps=points.velocity_mps,
                                        energy=points.energy, frame_index=i))
    print(f"t={f.time:.2f} s:", [(int(n), round(float(z), 2)) for n, z in zip(ids, points.xyz[:, 2])])
```

Detection, peak grouping, angle estimation, clustering and tracking are
discrete choices and refuse a tensor that carries a gradient.

## Micro-Doppler

`P.microdoppler_spectrogram(signal, window_slots=, hop_slots=, window="hann")`
is a short-time Fourier transform of one slow-time sequence,
`P.SlowTimeSignal(samples, times_s, phasor, wavelength_m=None)`: one sensor
pair and one range gate, uniform absolute timestamps (s) and the producing
phasor (`result.phasor`). It returns a `MicroDopplerSpectrogram` record with
`data [..., W, F]`, `times_s` (window centres), `frequency_hz` (positive
closing) and, when a wavelength is given, `velocity_mps`.

```python
gate = int((axes.range_m - 3.0).abs().argmin())
slow = frame.range_profile().data[0, 0, :, gate]  # one pair, one range bin, [chirp]
stamps = tuple(frame.time + n * axes.slow_time_period_s for n in range(slow.shape[0]))
spectrogram = P.microdoppler_spectrogram(P.SlowTimeSignal(slow, stamps, result.phasor, wavelength_m=axes.wavelength_m),
                                         window_slots=32, hop_slots=16)
w, f = divmod(int(spectrogram.data.abs().argmax()), spectrogram.data.shape[-1])
print(f"{type(spectrogram).__name__}: {tuple(spectrogram.data.shape)}, peak at "
      f"{spectrogram.velocity_mps[f]:+.2f} m/s (window centre {spectrogram.times_s[w] * 1e3:.2f} ms)")
```

## Imaging by back-projection

`P.Aperture` records where every virtual element was at every slow-time
sample; `P.Aperture.from_radar(radar, time_s=, velocity_mps=, pose=)` builds it
from the radar (a straight track or the verbs' `pose(t)` ego-motion), and
`frame.aperture(radar)` does so at the frame's instant. `P.backproject(profile,
aperture, pixels_m)` (or `frame.image(aperture, pixels_m, range_bins=)`)
forms a complex image over any pixel grid `[*grid, 3]` in world metres; it is
native on CUDA and differentiable with respect to the samples. Zero pad the
range transform (8x the sample count) on an FMCW beat cube: the image reads
it by linear interpolation. The FMCW spectrum output cannot be padded, so
image a `Fmcw(output="beat")` radar, as here.

```python
aperture = frame.aperture(radar)
xs = torch.linspace(-1.0, 3.0, 81, dtype=torch.float64)
zs = torch.linspace(-6.0, -2.0, 161, dtype=torch.float64)
grid = torch.stack(torch.meshgrid(xs, torch.zeros(1, dtype=torch.float64), zs, indexing="ij"), dim=-1)[:, 0]
image = frame.image(aperture, grid.to("cuda"), range_bins=8 * chirp.samples_per_chirp)
brightest = grid.reshape(-1, 3)[int(image.data.abs().argmax())]
print(f"{tuple(image.data.shape)} image, brightest pixel at range "
      f"{float(brightest.norm()):.2f} m (frame {image.frame!r})")
assert abs(float(brightest.norm()) - 5.0) < 0.05
```

The brightest pixel is the static target at 5 m: back-projection assumes the
scene is still in the aperture's frame, so the closing target's Doppler phase
averages out over the 64 chirps and it images weaker than its range alone
would make it.

A four-element array resolves range finely and angle coarsely; a moving
platform (`velocity_mps=` or `pose=`) synthesises a long aperture (SAR).
`P.isar_aperture(aperture, center_m=, angular_velocity_rad_s=)` (or
`rotation=` per sample) re-expresses an aperture in the body frame of a
turning target, for ISAR; its cross-range resolution is `lambda / (2 dtheta)`
for a total turn `dtheta`. `P.element_responses(profile, aperture, point_m)`
returns each virtual element's back-projected response at one point. Chirp
sequences, DDM/BPM MIMO and stepped pulses are refused by
`Aperture.from_radar` (an explicit `Aperture(...)` still works), and the
aperture and pixel geometry carry no derivative.

## Calibration and background subtraction

* `P.estimate_array_calibration(profile, aperture, reflector_m)` measures each
  virtual channel's complex gain from a point reflector at a known position;
  the `ArrayCalibration` it returns divides a cube, range profile or
  range-Doppler map by those gains with `.apply(product)`.
* `P.estimate_range_calibration(profile, aperture, reflector_m, search_m=)`
  measures a round-trip delay bias; `RangeCalibration.apply(product)` moves
  the range axis.
* `P.subtract_background(product, background)` removes leakage and static
  clutter recorded without the target (`P.average_background([...])`
  averages several references).

Below, an array whose receive ports have gain and phase errors (a diagonal
`Coupling`) and board leakage is calibrated on a reflector. Subtract the
background first: the leakage tone's skirt would otherwise bias both
estimates.

```python
errors = torch.tensor([1.0, 0.8 * complex(math.cos(0.4), math.sin(0.4)), 1.2 * complex(math.cos(-0.7), math.sin(-0.7)),
                       0.9 * complex(math.cos(1.1), math.sin(1.1))], dtype=torch.complex64)
faulty = radar.replace(noise=None, coupling=Coupling(rx=torch.diag(errors).tolist()),
                            leakage=Leakage(isolation=30.0, delay=0.2e-9))
reflector = (0.0, 0.0, -2.0)
calibration_frame = faulty.simulate(scene, PointTargets(positions=[reflector], rcs=10.0), times=(0.0,)).frame(0)
background = faulty.simulate(scene, PointTargets(positions=[(0.0, 0.0, -40.0)], rcs=1e-9), times=(0.0,)).frame(0)
cleaned = P.subtract_background(calibration_frame.processing_cube(), background.processing_cube())
cal_profile = P.range_profile(cleaned, range_bins=8 * chirp.samples_per_chirp)
calibration = P.estimate_array_calibration(cal_profile, calibration_frame.aperture(faulty), reflector)
ratio = calibration.gains[0].cpu() / errors.to(torch.complex128)
ratio = ratio / ratio[0]
print("estimated / injected gains:", [f"{abs(v):.3f}<{math.degrees(math.atan2(v.imag, v.real)):+.1f}deg" for v in ratio.tolist()])
assert float((ratio - 1).abs().max()) < 0.05
bias = P.estimate_range_calibration(cal_profile, calibration_frame.aperture(faulty), reflector, search_m=0.15)
print(f"round-trip delay bias: {bias.delay_s * 1e12:+.2f} ps (an ideal radar has none)")
assert abs(bias.delay_s) < 5e-12
fixed = calibration.apply(cleaned)
print("calibrated, background-subtracted cube:", tuple(fixed.data.shape))
```

## What processing refuses

* A bare tensor where a typed product is expected (`range_profile(tensor)`):
  the fast axis of a bare tensor cannot say whether it is an FMCW spectrum or
  a beat signal.
* A fast-time window or `range_bins` on the FMCW spectrum output.
* `fft2_aoa` on an array without a lattice and `phase_comparison_aoa` on a
  layout it cannot read; `point_cloud`, `beam_cube` and `range_angle_map` on
  a DDM/BPM product that was not demultiplexed (`frame.virtual_cube()` and the
  facades do it for you).
* A gradient or forward tangent through CFAR, peak grouping, the
  angle estimators, point clouds, clustering, tracking and the range
  calibration estimate (all discrete). The transforms, clutter filters,
  range-angle map, micro-Doppler and back-projection are differentiable.
