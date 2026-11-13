# R-ADR-050: Public API migration table, 0.4 to 0.5 (the naming pass and validation)

Status: Accepted (0.5.0)

## Context

R-ADR-049 renames public names in one breaking pass, with no aliases, and makes
every record validate at construction. This record is the complete table a
user needs to move code from 0.4.x to 0.5.0 for those two changes; the
release's user migration guide is built from it. Other 0.5.0 breaking changes
(default `Motion`, noise contract, processing defaults, ...) are in their own
records and changelog fragments.

Earlier design records (R-ADR-001 to 048, 051 to 055) are history and keep the
names they were written with.

## Decision: the table

### Renamed (the old name raises `TypeError` or `AttributeError`)

| 0.4 | 0.5 | Note |
|---|---|---|
| `Radar(carrier=f)` / `radar.carrier` | `Radar(reference_frequency_hz=f)` / `radar.reference_frequency_hz` | Channel's name. The flat configuration key stays `fc`. |
| `radar.replace(carrier=f)` | `radar.replace(reference_frequency_hz=f)` | |
| `radar.transmit_power_watts` | `radar.power_w` | Channel's name and unit. `Radar.power` stays, in dBm. |
| `radar.trace/simulate/stream(..., grad=m)` | `...(..., ad_mode=m)` | Same values `"none"`, `"vjp"`, `"jvp"`. Also `Network.trace/simulate/stream`. |
| `paths.grad` | `paths.ad_mode` | |
| `paths.carrier` | `paths.reference_frequency_hz` | `Result.reference_frequency_hz` is unchanged. |
| `result.times_s` | `result.times` | The name `Paths` already used. |
| `result.sample_times_s` | `result.sample_times` | Likewise. |
| `result.birth_lag_bound_s`, `paths.birth_lag_bound_s` | `result.birth_lag_bound`, `paths.birth_lag_bound` | `PathEvents.birth_lag_bound_s` (in `witwin.radar.events`) is unchanged. |
| `frame.time_s` | `frame.time` | `Frame` of `result.frame(i)`. |
| `Result.from_frames(times_s=, sample_times_s=)` / `Result(times_s=, sample_times_s=)` | `times=`, `sample_times=` | Saved results written by a 0.5 pre-release with the old keys do not load. |
| `Leakage(isolation_db=x, delay_s=t)` | `Leakage(isolation=x, delay=t)` | Same units (dB, s). |
| `Iq(gain_db=g, phase_deg=p)` | `Iq(gain=g, phase=math.radians(p))` | **Unit change**: `phase` is radians. `gain` stays dB. The flat config `iq` block follows. |
| `RcsTable(axis, rcs_m2=, frequencies_hz=, phase_rad=)` | `RcsTable(axis, rcs=, frequencies=, phase=)` | Same units. |
| `MeshTargets(cluster_size_m=d)` | `MeshTargets(cluster_size=d)` | m. |
| `Fmcw/Ofdm/Pulsed.to_spec(carrier=f, ...)` | `.to_spec(reference_frequency_hz=f, ...)` | |

### Changed default

| 0.4 | 0.5 | Migration |
|---|---|---|
| `frame.range_doppler()` tapered slow time with Hann | rectangular, like `processing.range_doppler_map` | Pass `window="hann"` for the old map. `frame.points()` still defaults to Hann. |

### Not changed, and why (so nobody looks for them)

`Radar.power` (dBm), `Emitter.power` (dBm), `los=`/`reflections=`, the default
pose `look_at=(0, 0, -1)`, `up=(0, 1, 0)` (R-ADR-049 section 6), every name in
`witwin.radar.processing` and `witwin.radar.labels`, internal spec records,
and the flat configuration keys.

### Added

| Name | What |
|---|---|
| `witwin.radar.capabilities()` | Versioned record of what the build supports. |
| `result.semantics`, `paths.semantics` | Completeness in Channel's `estimator`/`combine`/`complete` keys, plus `exhaustive`. |
| `AdaptiveDiagnostics` (`witwin.radar.adaptive`) | Typed adaptive record; new key `complete`. |
| `frame.virtual_cube()`, `frame.range_angle(az)`, `frame.image(aperture, pixels_m, range_bins=)`, `frame.aperture(radar, ...)` | Facades; DDM/BPM demultiplexed automatically. |
| `Fmcw.sweep_bandwidth`, `.range_resolution`, `.max_range`, `.max_velocity(reference_frequency_hz=, num_tx=)` | Derived FMCW quantities. |
| `LEAVES` on `Radar`, `PointTargets`, `Pattern`, `Coupling`, `SMPLBody` | Differentiable fields and their AD modes. |

### Now refused at construction (were accepted, failed later or silently)

| Input | Error names |
|---|---|
| A numeric string, bool, complex, NaN or infinity where a real number is expected (e.g. `Fmcw(slope="60e12")`, `Iq(gain=nan)`) | `Record.field must be a real number` / `must be finite` |
| An unknown word: `Fmcw.mimo/output/sampling`, `Ofdm.modulation/output`, `Pulsed.pulse_kind`, `Motion.world` | `Record.field must be one of [...]` |
| An FMCW ADC window ending at or after `idle + ramp_end` | `Fmcw: the ADC window ends ...` |
| Array-dependent waveform rules (BPM on a non-power-of-two transmitter count, DDM bands below it) | `Radar.waveform: ...` at `Radar(...)` |
| `positions` not `(S, 3)` or empty; negative host `rcs`; repeated or miscounted `ids`; an `Aspect`/`RcsTable` axis with the wrong row count; non-finite `ScatteringMatrix` factors | `PointTargets.positions`, `PointTargets.rcs[i]`, `PointTargets.ids`, `Aspect.axis`, `ScatteringMatrix.hh` ... |
| A radar whose device is not CUDA, at `trace`/`simulate`/`stream`/`echo` | `this Radar is on device 'cpu' ...` |
| `point_cloud`, `beam_cube`, `range_angle_map` on a DDM/BPM product that was not demultiplexed | `... needs one channel per transmitter ...` |
