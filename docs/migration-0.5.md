# Migrating from witwin-radar 0.4 to 0.5

Radar 0.5.0 is one breaking release with no compatibility aliases: an old keyword or attribute raises `TypeError` or
`AttributeError` rather than meaning something else. This guide lists every change that can break 0.4 code, with
the 0.4 spelling and the 0.5 spelling side by side, in the order you are likely to meet them. The complete list of
changes is in [`CHANGELOG.md`](../CHANGELOG.md); the naming rule and its full table are R-ADR-049 and R-ADR-050 in
`docs/dev/standards/`.

The fastest path through an upgrade:

1. Upgrade the environment (section 1).
2. Rename (section 2). Every rename fails loudly at the old name, except one unit change: **`Iq.phase` is radians**.
3. Read section 3 for the defaults that changed silently: the default `Motion`, the Range-Doppler window, the
   point-cloud route and peak grouping, CFAR input, and the numbers the physics fixes move.

## 1. Environment

| 0.4 | 0.5 |
|---|---|
| `witwin-channel>=0.5,<0.6` through the `channel` extra | `witwin-channel>=0.6.0,<0.7` (consumer contract 7). Radar 0.5 does not run on Channel 0.5. |
| `torch>=2.10` | `torch==2.10.0`, the version the packaged native library is built against. |
| the `channel` extra on every Python | the extra installs Channel on CPython 3.10-3.13 only (Channel's wheels); on 3.14 it installs Radar alone, and scene-driven simulation needs 3.10-3.13. |

```bash
pip install -U "witwin-radar[channel]"
witwin-radar-doctor          # checks Torch, the native library, Channel and runs a GPU smoke test
```

A different Torch needs a source build of the native library: `python scripts/build_radar_cuda_prebuilt.py` from a
checkout (see [`CONTRIBUTING.md`](../CONTRIBUTING.md)).

## 2. Renamed

A quantity Channel also publishes takes Channel's name (`reference_frequency_hz`, `ad_mode`, `power_w`); every
other public name carries no unit suffix, and its unit is on its docstring.

| 0.4 | 0.5 |
|---|---|
| `Radar(carrier=f)`, `radar.carrier`, `radar.replace(carrier=f)` | `Radar(reference_frequency_hz=f)`, `radar.reference_frequency_hz`, `radar.replace(reference_frequency_hz=f)` |
| `radar.transmit_power_watts` | `radar.power_w` (`Radar.power` stays, in dBm) |
| `radar.trace/simulate/stream(..., grad=m)` and `Network.trace/simulate/stream(..., grad=m)` | `ad_mode=m` (same values `"none"`, `"vjp"`, `"jvp"`) |
| `paths.grad`, `paths.carrier` | `paths.ad_mode`, `paths.reference_frequency_hz` |
| `result.times_s`, `result.sample_times_s` | `result.times`, `result.sample_times` |
| `result.birth_lag_bound_s`, `paths.birth_lag_bound_s` | `result.birth_lag_bound`, `paths.birth_lag_bound` |
| `frame.time_s` | `frame.time` |
| `Result(times_s=, sample_times_s=)`, `Result.from_frames(times_s=, ...)` | `times=`, `sample_times=` |
| `Leakage(isolation_db=x, delay_s=t)` | `Leakage(isolation=x, delay=t)` (dB, s) |
| `Iq(gain_db=g, phase_deg=p)` | `Iq(gain=g, phase=math.radians(p))` - **the phase is radians now** |
| `RcsTable(axis, rcs_m2=, frequencies_hz=, phase_rad=)` | `RcsTable(axis, rcs=, frequencies=, phase=)` |
| `MeshTargets(cluster_size_m=d)` | `MeshTargets(cluster_size=d)` (m) |
| `Fmcw/Ofdm/Pulsed.to_spec(carrier=f, ...)` | `.to_spec(reference_frequency_hz=f, ...)` |
| `Fmcw(...).bandwidth` (it returned the sample rate) | `Fmcw.sample_rate`; the swept bandwidth is `Fmcw.sweep_bandwidth` |
| `AdaptiveMotionSpec(...)` | `Motion.adaptive(...)` (same tolerances) |
| `from witwin.radar.simulation import SensorEndpointIds` | `from witwin.radar.binding import SensorEndpointIds` |

Before (0.4):

```python no-run
radar = Radar(carrier=77e9, waveform=chirp, tx=tx, rx=rx, power=12.0,
              leakage=Leakage(isolation_db=40.0, delay_s=1e-9), iq=Iq(gain_db=0.5, phase_deg=3.0))
result = radar.simulate(scene, targets, times=(0.0, 0.05), grad="vjp")
print(result.times_s, radar.transmit_power_watts, result.frame(1).time_s)
```

After (0.5), runnable as written:

```python
import math

import torch
from witwin.core import Scene
from witwin.radar import Fmcw, Iq, Leakage, Motion, PointTargets, Radar, processing

chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=64, chirps_per_frame=16,
             adc_start=6e-6, idle=7e-6, ramp_end=25e-6)
tx, rx = ((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
scene = Scene()
targets = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0,
                       trajectory=lambda t: torch.tensor([[0.0, 0.0, -3.0 + t]], device="cuda"))

radar = Radar(reference_frequency_hz=77e9, waveform=chirp, tx=tx, rx=rx, power=12.0, antenna_unit="half_wavelength",
              leakage=Leakage(isolation=40.0, delay=1e-9), iq=Iq(gain=0.5, phase=math.radians(3.0)))
result = radar.simulate(scene, targets, times=(0.0, 0.05), ad_mode="none")
print(result.times, radar.power_w, result.frame(1).time)
```

Not renamed, on purpose: `Radar.power` and `Emitter.power` (dBm), `los=` / `reflections=`, the default pose, every
name in `witwin.radar.processing` and `witwin.radar.labels` (`range_m`, `velocity_mps`, `Aperture.times_s`,
`MicroDopplerSpectrogram.times_s`, ...), internal spec records, and the flat configuration keys (`fc`, `power`,
TI-style). A flat configuration's `iq` block takes `gain` (dB) and `phase` (rad).

## 3. Changed behaviour

### The default `Motion` for a moving FMCW scene is adaptive

`Motion.auto()`, the default, resolves to `Motion.adaptive()` when anything moves in an FMCW scene. 0.4 evaluated the
world at every ADC instant, which is exact and costs one world observation per sample. Adaptive sampling evaluates a
few probes per frame and interpolates between them under a phase and amplitude tolerance, so cubes differ from 0.4 by
up to that tolerance. The adaptive route covers one uniform TDM ramp, so any other moving scene (OFDM, pulsed, a chirp
sequence, a ramp phase error, DDM or BPM), a scene that declares a discovery cadence, and a moving `Network` still get
the exhaustive schedule.

```python
times = (0.0,)
# 0.4 default for a moving scene, and still the exhaustive reference in 0.5:
reference = radar.simulate(scene, targets, times=times, motion=Motion.adc())
# 0.5 default: Motion.auto() -> adaptive; read what it did
result = radar.simulate(scene, targets, times=times)
print(reference.motion_sampling, result.motion_sampling, result.adaptive_diagnostics[0]["evaluations"])
print(result.semantics)
```

`Result.motion_sampling` and `Paths.kind` now name the schedule that ran (`"chirp"`, `"symbol"`, `"pulse"`, `"adc"`,
`"adaptive"`, ...), and `motion_sampling_exhaustive` is true only for the frame and fast-time schedules.
`Motion.adaptive()` no longer takes `rediscover_every_frames`. A pulsed train under `Motion.adc()` now observes the
world at every range-gate sample (intra-pulse Doppler); `Motion.chirp()` keeps one observation per pulse.

### Processing defaults

| 0.4 | 0.5 | To keep the 0.4 result |
|---|---|---|
| `frame.range_doppler()` tapered slow time with Hann | rectangular, like `processing.range_doppler_map` | `frame.range_doppler(window="hann")` |
| `point_cloud(..., route="phase_comparison")` default; `Frame.points` likewise | `route="scan"` (any array layout) | `route="phase_comparison"` on a layout it supports |
| one point per detected cell | `peak_neighbourhood=(3, 3)`: one point per local maximum | `point_cloud(..., peak_neighbourhood=None)` |
| CFAR thresholded what it was given | square law: complex input is detected on `abs(x)**2`, a real input is read as power | pass complex IQ or `abs(x)**2`, never `abs(x)` or dB |
| `combine_coherent(cube)` returned `abs(sum)` | returns the power `abs(sum)**2` | `combine_coherent(cube).sqrt()` |
| `os_cfar(pfa=)` was a nominal constant | `pfa` is the achieved false-alarm rate | thresholds and counts change for every detector |
| `microdoppler_spectrogram` returned `(times, frequencies, spectrum)` | a `MicroDopplerSpectrogram` record, normalised by `1 / window_slots` | read `.times_s`, `.frequency_hz`, `.data` |
| `ArrayGeometry` offsets in the radar's local frame | `from_axes` maps `(right, up, -boresight)` into the processing frame `(right, boresight, up)`; `from_offsets` takes processing-frame offsets | elevation offsets in a config (`tx_loc` `[x, 1, 0]`) now steer as elevation |
| `fft2_aoa` assumed TI interleaving and four transmitters | reads the array's lattice | none; `phase_comparison_aoa` refuses layouts it cannot read |

Before and after for a detector over a Range-Doppler map:

```python
frame = result.frame(0)
rd = frame.range_doppler()
# 0.4: detections = processing.ca_cfar(rd.data.abs(), pfa=1e-3)
detections = processing.ca_cfar(rd.data, pfa=1e-3)                       # complex in: |x|^2 inside
same = processing.ca_cfar(rd.data.abs().square(), pfa=1e-3)              # or power in
print(detections)
```

```python no-run
# 0.4: times, freqs, spectrum = processing.microdoppler_spectrogram(signal, window_slots=64, hop_slots=8)
spec = processing.microdoppler_spectrogram(signal, window_slots=64, hop_slots=8)
times, freqs, spectrum = spec.times_s, spec.frequency_hz, spec.data
```

`point_cloud`, `beam_cube` and `range_angle_map` refuse a DDM or BPM product whose transmitters are not separated;
call `processing.demultiplex` first, or use the `Frame` facades (`points`, `range_angle`, `virtual_cube`), which do.

### Numbers the physics fixes move

These are corrections, each checked against an analytic oracle in `tests/oracle/`. Drop any compensation you
applied for the old behaviour.

- **Isotropic elements are isotropic.** 0.4 projected every leg onto a dipole along the polarization, so an
  `"isotropic"` element lost `sin^4` of the angle off the plane normal to the polarization (5.0 dB at 30 degrees,
  12.0 dB at 45, 24.1 dB at 60). On the polarization plane nothing changes. `Pattern.dipole()` attenuates once, not
  twice.
- **Pulsed samples are physical amplitudes.** The pulse envelope is unit amplitude, so the pulsed cube is `sqrt(T_p)`
  larger than in 0.4; matched-filter peaks are unchanged. Stop dividing by `sqrt(T_p)`.
- **Noise bandwidth.** Thermal noise integrates over the complex sample rate for every waveform. Pulsed SNR drops by
  `f_s / (B T_p)` relative to 0.4 (it was optimistic).
- **Noise per frame.** Each frame of a noisy run draws its own realisation, keyed by the frame instant; 0.4 reused
  one for every frame of a call. A frame at `t = 0` is unchanged.
- **ADC grid.** `Adc` is a two's-complement mid-tread quantiser: `step == 2 * full_scale / 2**bits`, zero is a code,
  outputs span `[-FS, FS - step]`, and `Adc(bits=1)` is refused.
- **Patterns.** Tables are normalised (power at most 1, `abs(field)` at most 1) and the peak gain is `Pattern.gain`
  in dBi; the record's table field is `gain_table`.

  ```python no-run
  # 0.4: a table whose peak value carried the gain
  # 0.5: a normalised table plus the absolute gain
  pattern = Pattern.table(x_angles, y_angles, values / values.max(), gain=10 * math.log10(float(values.max())))
  ```
- **Moving-target phase.** The round-trip path length is recomputed in float64, which removes 0.4's carrier-phase
  staircase (spurs at -49 dBc at 100 m in 0.4, below -100 dBc now). Doppler spectra of moving targets are cleaner.
- **Surfaces.** A surface scatters from its facets through `MeshTargets` (physical optics). The 0.4 recipe of many
  isotropic `PointTargets` of `sigma / N` over a surface does not converge (it reads `N sigma` for a plate) and should
  be replaced:

  ```python no-run
  # 0.4: targets = PointTargets(positions=samples_on_the_surface, rcs=sigma / len(samples_on_the_surface))
  targets = MeshTargets()                          # every face of the scene's structures scatters
  targets = MeshTargets(structure_ids=(7,), max_sites=2048)   # one structure, facets merged into <= 2048 sites
  ```
- **Outbound leg orientation.** The outbound leg is evaluated from the receive element toward the site (reciprocity);
  composed values move by at most 1.3e-7 relative. Code reading the internal leg records changes:
  `RadarLegBatch.coefficient` (a projected scalar) is `RadarLegBatch.field` (`complex64[rows, 3]`, the field vector at
  the site), `coefficient_band` is `field_band` (`[rows, F, 3]`); `RadarWorldBinding` has one `sites` spec and the
  receivers as `receivers` and `receiver_sources`; `TwoWayComposer.outbound_max_depth` is gone.

### Validation and refusals

Records validate at construction and refuse a bad field by `Record.field` name; a CPU radar is refused by the verbs.

```python
for build in (
    lambda: Fmcw(slope="60e12", sample_rate=4.4e6, samples_per_chirp=64, chirps_per_frame=16,
                 adc_start=6e-6, idle=7e-6, ramp_end=25e-6),   # 0.4: accepted, failed later
    lambda: PointTargets(positions=[(0.0, 0.0)]),               # positions must be (S, 3)
    lambda: radar.replace(device="cpu").simulate(scene, targets, times=(0.0,)),   # refused before any work
):
    try:
        build()
    except (TypeError, ValueError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}")
```

- An FMCW ADC window that ends at or after `idle + ramp_end` is refused (it sampled the next chirp): lengthen
  `ramp_end` or shorten the window.
- Array-dependent waveform rules (BPM needs a power-of-two transmitter count, DDM bands at least the transmitter
  count) refuse `Radar(...)` instead of its first run.
- `Radar.from_dict` refuses a mapping that declares the waveform twice (a `waveform` block next to TI-style FMCW
  keys); a non-FMCW radar is now written as `{"waveform": {"kind": "ofdm", ...}}`.
- A scatter site on an array element is refused by name.

### Results

- `Result.load` returns a result whose `last_propagation`, `last_radar_paths`, `last_snapshot` and
  `last_compiled_scene` are `None` and whose `events` is empty; the fields are typed `Optional`.
- `Result`, `Paths` and per-frame records gain `discovery_complete` and `transmission_dropped_rows`;
  `path_set_complete` is also false when Channel's discovery was not provably complete. `Result.semantics` gives the
  same in Channel's vocabulary.

### Channel 0.6 behaviour you may notice

- A reflection whose specular point lies past a face's edge by more than a few float roundings is not reported.
- A scene edited in place after it was compiled is refused at the next replay ("the compiled scene is stale").

### For code that calls internals

- `ScatterResponse.evaluate(site_count=...)` replaces `evaluate(row_count=...)`.
- `FrozenLegTopology.prepared` is a property that prepares on first use; `ChannelPropagationAdapter.discover` is
  discovery alone.
- `RadarPathTopology` gains `direct_row`; a composed row with `site_id == -1` has `inbound_row == outbound_row == -1`
  (only with `environment=True` or `coupling=True`).
- Native operator schemas changed (`sensor_weight_*`, `facet_response_*`, `pulsed_echo_*`, `fmcw_beat_*`). After
  changing a native operator run `python ci/check_native_bindings.py --write`; the binding manifest is generated.
- `SensorWeightGeometry.FROZEN_FIELDS` no longer lists `pattern_frame`; `SmplPoseDeformation` accepts a pose or
  shape with a gradient; `Coupling.tx`/`rx` keep a complex tensor as given.
