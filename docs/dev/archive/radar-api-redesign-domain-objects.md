# Radar API redesign: domain objects, and the trace/echo split

Status: Implemented (2026-09-18).

All five commits of section 10 are on the branch. Commits 1, 4 and 5 shipped the
flat `Radar` record, the target and motion records, the caller migration, the
governance manifests and the living documents; commits 2 and 3 shipped `Paths`,
`trace` and `echo`, the rename of the result to `Result`, and the `Frame`
facades over the processing package.

Sections 4.9, 4.10 and 5 are the draft the work was done against, kept as
written. Section 5.4a is no longer a plan: it records where the cut actually
went and the three places the shipped `Paths`, `Result` and `Frame` differ from
sections 4.9 and 4.10. Section 10a records the three places commit 1 differed
from the draft.

One difference is structural enough to state here rather than in a subsection:
`Paths`, `Result` and `Frame` all live in `witwin/radar/simulation.py`, not in
`paths.py` as section 8 proposed. The session loop the four verbs share is
there, and a `Paths` carries that session with it so its rows can be echoed;
splitting the record away from the loop that produces it would have put the two
halves of one contract in two files.

One item of commit 3 was not carried out and should not be: `frame_synthesis`
was to be deleted, and it survives. It is the rank-3 `(slow, sensor_pair, fast)`
view in the waveform's own layout, which is what a stage wanting the synthesis
ordering rather than the processing one reads; `Frame` carries it as `synthesis`
beside the processing cube. What the commit did delete is the reason a caller
had to go through it, which was assembling `ProcessingAxes` by hand.

The suite was run on the implementing machine, with CUDA and a Channel developer
override: 1590 passed, 12 skipped, none failed, and all thirteen static gates
pass. That is the record of one local run and not a statement about CI.

## 1. Summary

The public Radar API becomes one flat, immutable `Radar` record with SI
units, a few typed sub-records only where a field has variants or coupled
numbers, and four verbs on the radar:

| Verb | In | Out | What it owns |
| --- | --- | --- | --- |
| `radar.trace(scene, targets, ...)` | world | `Paths` | world sampling, Channel epochs, round-trip composition, scattering, antenna weighting |
| `radar.echo(paths)` | `Paths` | `Result` | waveform synthesis, receiver chain, output domain, processing axes |
| `radar.simulate(scene, targets, ...)` | world | `Result` | the two above fused, without materialising `Paths` |
| `radar.stream(scene, targets, ...)` | world | `Iterator[Result]` | `simulate`, one frame per item |

The example that today needs six imports, a flat configuration dictionary in
four unit conventions, a private method, and a mutable diagnostic becomes
four imports and one flat constructor:

```python
from witwin.radar import Radar, Fmcw, Noise, PointTargets

radar = Radar(
    carrier=77e9,
    waveform=Fmcw(slope=60.012e12, sample_rate=4.4e6, samples_per_chirp=256, chirps_per_frame=128,
                  adc_start=6e-6, idle=7e-6, ramp_end=65e-6),
    tx=[(0, 0, 0), (4, 0, 0), (2, 1, 0)],
    rx=[(-6, 0, 0), (-5, 0, 0), (-4, 0, 0), (-3, 0, 0)],
    antenna_unit="half_wavelength",
    power=15.0,
    noise=Noise(figure=10.0),          # default None: no noise
    seed=20260727,
    position=(0, 0, 0),
    look_at=(0, 0, -1),                 # polarization "up" is transverse by construction
)
targets = PointTargets(positions=[(0, 0, -3)], rcs=1.0)           # device and carrier come from the radar

paths = radar.trace(scene, targets, times=(0.0, 0.1, 0.2), reflections=1)
result = radar.echo(paths)              # bit-identical to radar.simulate(scene, targets, times=..., reflections=1)

frame = result.frame(0)
rd = frame.range_doppler(window="hann")                                  # delegates to processing.range_doppler_map
cloud = frame.point_cloud(pfa=1e-4, route="phase_comparison", max_points=64)
```

That block is the draft as written. The shipped detection facade is
`frame.points(...)`, for the reason recorded in section 5.4a.

Physics does not change, with the two decided exceptions in section 9. Processing
math does not change. The Channel boundary does not change.

## 2. Why: the current surface, measured

| Finding | Evidence |
| --- | --- |
| Units are hidden in field names | `slope` is MHz/µs, `sample_rate` is kSPS, `adc_start_time`/`idle_time`/`ramp_end_time` are µs, `power` is dBm, `fc` is Hz, `tx_loc`/`rx_loc` are half-wavelength multiples (`radar.py`, `FmcwWaveformConfig` docstring; `Radar._init_antenna_locations`) |
| Four required keys are not consumed | `frame_per_second`, `num_doppler_bins`, `num_range_bins`, `num_angle_bins` build `ProcessingConfig`, which no simulation or processing code reads |
| Two constructor parameters are camera leftovers | `fov` is stored and never read; `name` has no reader in the package |
| Defaults contradict each other | the default pose looks along -Z and `DEFAULT_POLARIZATION` is `(0, 0, 1)`, parallel to the boresight, so every transport is exactly zero and nothing raises (`examples/single_point.py` documents the trap) |
| "No targets" is not the default | `sites=None` becomes `ScatterSitePolicy.structure_anchor()`, one site per rigid structure |
| Modes are strings that must agree | `ad_mode`, `world_motion`, `motion_sampling`, `components`; `adaptive_motion` without `motion_sampling="adaptive"` raises |
| The radar's own facts are re-fed | `ScalarRcsResponse.from_rcs(rcs, reference_frequency_hz=radar.config.fc, device=radar.device)`; `ScatterSitePolicy.explicit(torch.tensor(..., device=radar.device))` |
| The processing seam needs three objects from two places | `ProcessingAxes.from_synthesis(result.frame_synthesis(), radar.system_config.waveform_spec(), radar.system_config.sensors.array)`; `tests/conftest.py` invented `PointTargetFrame` to hold the pair; two examples call the private `radar._synthesize` instead |
| Diagnostics exist twice | four mutable `last_*` properties on the radar duplicate the same fields on the result |
| Surface shape | 26 modules, 140 declared exports, 487 reachable symbols, 2 root exports; `Radar.simulate` takes 13 keyword arguments |

The frame loop in `simulation.py` reads seven attributes of the radar object
and the test suite substitutes a `MockRadar` for it. The object dependency is
already narrow; the redesign makes the record explicit.

## 3. Goals and non-goals

Goals:

1. Every public field states its unit in its docstring; no name carries a
   unit suffix.
2. One way to build a radar: typed parts. The flat dictionary is a loader.
3. No hidden defaults that change physics silently; refusals where a default
   would be a guess.
4. The world half and the instrument half of the pipeline are separately
   callable, and the fused call is bit-identical to the pair.
5. The result is the only diagnostic; the radar holds no run state.
6. Processing is reachable from a result without re-synthesis or private
   methods, and adds no second implementation of any DSP stage.
7. A record exists only where a field has variants (`waveform`, `pattern`,
   `Motion`) or where several numbers are coupled (`Noise`, `Agc`, `Adc`,
   `PointTargets`). Everything else is a field of `Radar` or a keyword of a
   verb.

Non-goals:

- No change to propagation, composition, scattering, synthesis or processing
  equations, and no new CPU path.
- No compatibility layer. AGENTS.md forbids shims; the cut is atomic per
  surface and every caller in the repository moves in the same commit.
- No new Channel components. The `los` and `reflections` keywords expose
  exactly what `witwin.radar.channel` accepts today.

## 4. The public surface

### 4.1 Root exports

`witwin.radar.__all__` becomes the whole happy path, in the order a caller
meets it:

```
Radar, Fmcw, Ofdm, Pulsed, Pattern, Noise, Agc, Adc,
PointTargets, StructureTargets, Aspect, Motion,
Paths, Result, Frame, processing
```

Advanced records stay importable from their owner modules and are not at the
root: `SensorEndpointIds`, `StableIdAllocator`, `RadarPathBatch`,
`SynthesisResult`, `FmcwSpec`, `ScalarRcsResponse`, `AspectScatterResponse`,
`FrontendChain`, the epoch records. `witwin.radar.processing` is re-exported
as a module, as `witwin.maxwell` re-exports `postprocess`.

### 4.2 `Radar`

A flat, immutable record: the parameters a datasheet or a configuration file
lists, one field each, named the way the datasheet names them. A sub-record appears only
where a field has variants (`waveform`, `pattern`) or where several numbers
are coupled (`Noise`, `Agc`, `Adc`). "Modifying" a radar returns a new one.

| Field | Type | Unit | Default |
| --- | --- | --- | --- |
| `carrier` | float | Hz | required |
| `waveform` | `Fmcw \| Ofdm \| Pulsed` | | required |
| `tx` | `tuple[vec3, ...]` | `antenna_unit` | required |
| `rx` | `tuple[vec3, ...]` | `antenna_unit` | required |
| `antenna_unit` | `"m" \| "half_wavelength"` | | `"m"` |
| `pattern` | `Pattern` | | `Pattern.isotropic()`, section 9.1 |
| `power` | float | dBm | required |
| `noise` | `Noise \| None` | | `None`: no thermal or oscillator noise |
| `lna_gain` | `float \| None` | dB | `None`: no LNA stage |
| `agc` | `Agc \| None` | | `None` |
| `adc` | `Adc \| None` | | `None`: no quantisation |
| `impedance` | float | Ω | 50.0 |
| `seed` | int | | 0; the Philox seed base every receiver stage keys from |
| `position` | vec3 | m | `(0, 0, 0)` |
| `look_at` | vec3 | m | `(0, 0, -1)`: boresight along -Z, as today |
| `up` | vec3 | | `(0, 1, 0)` |
| `polarization` | `"up" \| "right" \| vec3` | | `"up"`: the frame's true up, transverse by construction |
| `device` | `str \| torch.device` | | `"cuda"`; refused when CUDA is absent, as today |

Fourteen of the eighteen fields have defaults; the required ones are the
carrier, the waveform, the two antenna layouts and the transmit power.

No field name carries a unit suffix. The unit is in the type annotation's
documentation and in the field's docstring row, which is where a reader looks
for a tolerance or a valid range anyway. This rule covers the redesigned
surface only: `witwin.radar.processing` is frozen by R-ADR-017 and keeps
`range_m`, `velocity_mps` and its siblings, and the internal records
(`FmcwSpec.sample_period_s`, `RadarPathBatch.total_delay_s`,
`RadarLegBatch.delay_s`) keep theirs, because those are the equation-owning
modules where R-ADR-021 requires units to be explicit.

`carrier` lives on the radar, not on the waveform: it is the reference
frequency of the antenna spacing, the propagation solve and the synthesis, and
today's `fc` already plays all three roles. The waveform records describe the
modulation only.

`tx` and `rx` are read in `antenna_unit`. Half-wavelength positions, the
convention every TI-style configuration uses, are converted to metres once,
in `__post_init__`, against `carrier`. `num_tx` and `num_rx` are the
lengths and are not typed a second time.

The four receiver stages, `impedance` and `seed` are today's
`FrontendSpec` with its wrapper removed. The stage order is fixed in
`frontend.py` and nothing here reorders it. A stage that is one number is a
keyword (`lna_gain`); a stage whose numbers are coupled is a record
(`Noise`, `Agc`, `Adc`). `noise` is one stage of four, so the field is called
`noise` and the others keep their own names.

The pose is look-at, as today, with three vectors instead of a record.
Refusals at construction: a `look_at` equal to `position`, an `up`
collinear with the boresight, a polarization vector parallel to the boresight
(it radiates nothing; today this case returns all-zero cubes without a
message). `position` may be a tensor with a tape. A rotation input is not
offered: it needs an axis-order and angle-unit convention that look-at does
not, and every test and example is written against look-at. If one is needed
later it is one optional field, `rotation`, taking a 3×3 world-from-radar
matrix, with `look_at` and `up` then refused.

Methods:

| Method | Returns | Notes |
| --- | --- | --- |
| `trace(scene, targets, *, times, los=True, reflections=1, motion=Motion.auto(), grad="none", endpoints=None)` | `Paths` | section 5 |
| `echo(paths)` | `Result` | section 5 |
| `simulate(scene, targets, *, times, los=True, reflections=1, motion=..., grad=..., endpoints=None)` | `Result` | fused |
| `stream(...)` | `Iterator[Result]` | same arguments as `simulate`, one frame per item |
| `replace(**fields)` | `Radar` | a new radar with these fields changed; replaces `set_pose` |
| `to(device)` | `Radar` | a new radar on another device |
| `from_dict(mapping, *, device="cuda", **overrides)` | `Radar` | the flat file format, section 6 |
| `from_json(path, ...)` | `Radar` | `from_dict` on a JSON file |
| `waveform_spec()` | `FmcwSpec \| OfdmSpec \| PulsedSpec` | the SI synthesis spec, derived from `waveform`, `tx`, `rx` and `carrier_hz`; advanced |

`endpoints` on `trace` and `simulate` is a `SensorEndpointIds`: the Core
phase-centre IDs of a radar mounted on a moving structure, in `tx`/`rx`
order. It binds the radar to one scene, so it is a call argument, not a
field.

Derived device tensors (`tx_positions_m`, `rx_positions_m` in world
coordinates after the pose) are built once in `__post_init__` and cached on
the record, as `Radar.__init__` does today.

Deleted from the constructor: `fov`, `name`, the flat `config` positional,
`target` (now `look_at`).

### 4.3 Waveforms: `Fmcw`, `Ofdm`, `Pulsed`

`Fmcw` replaces `FmcwWaveformConfig`. Every field is SI.

| Field | Unit | Today | Note |
| --- | --- | --- | --- |
| `slope` | Hz/s | `slope` MHz/µs | 60.012 MHz/µs is `60.012e12` |
| `sample_rate` | Hz | `sample_rate` kSPS | |
| `samples_per_chirp` | count | `adc_samples` | |
| `chirps_per_frame` | count | `chirp_per_frame` | slots per frame are `chirps_per_frame * num_tx` under TDM, as today |
| `adc_start_s` | s | `adc_start_time` µs | |
| `idle_s` | s | `idle_time` µs | |
| `ramp_end_s` | s | `ramp_end_time` µs | chirp period is `idle_s + ramp_end_s` |
| `output` | `"spectrum" \| "beat"` | `output_domain` | default `"spectrum"`, unchanged (AGENTS.md FMCW contract) |

`Fmcw.from_ti(slope_mhz_per_us, sample_rate_ksps, adc_start_us, idle_us, ramp_end_us, samples_per_chirp, chirps_per_frame, output="spectrum")`
is the one conversion site from vendor units and is what the loader calls.

`Ofdm` and `Pulsed` are today's `OfdmWaveformConfig` and `PulsedWaveformConfig`
renamed. Their fields are already SI (`subcarrier_spacing_hz`,
`cyclic_prefix_s`, `pulse_width_s`, `pri_s`, ...) and keep their names.

`to_spec(...)` stays on all three as the derivation of the synthesis spec; it
is called by `Radar.waveform_spec()` and is not part of the happy path.

### 4.4 `Pattern`

`Pattern` replaces `AntennaPatternSpec` and is the one optional record on the
antenna side: it has four variants, so a keyword would need a kind string plus
loose tables. Values are normalised linear power gain, dimensionless, at most
1. It is applied by the round-trip weighting stage in `sensors.py`, unchanged.

| Constructor | Meaning |
| --- | --- |
| `Pattern.isotropic()` | the proven no-op, today's `ISOTROPIC_PATTERN`; the default |
| `Pattern.dipole()` | today's half-wave dipole cut over -90..90 degrees |
| `Pattern.separable(x_angles_deg, x_gain, y_angles_deg, y_gain)` | today's `"separable"` kind |
| `Pattern.table(x_angles_deg, y_angles_deg, gain)` | today's `"map"` kind |

### 4.5 Receiver stages: `Noise`, `Agc`, `Adc`

There is no `Receiver` record. `Radar` holds the stages directly (section
4.2), because the common case, one noise figure, should not need two
constructors, and the stage order and the shared seed are facts of
`frontend.py` rather than something a caller assembles. Stage equations, the
Philox stage keys and `FrontendChain` stay in `frontend.py` unchanged; the
chain is built from the radar's fields.

`Noise` keeps today's physical quantities, with names that neither repeat the
class nor carry a unit:

| Field | Unit | Default | Today |
| --- | --- | --- | --- |
| `figure` | dB | 0.0 | `noise_figure_db` |
| `antenna_temperature` | K | 290.0 | unchanged |
| `bandwidth` | Hz | `None`: the waveform's sampling bandwidth | `0.0`, which silently adds no noise |
| `phase_density` | dBc/Hz | `None`: no oscillator noise | `phase_noise_dbc_per_hz` |
| `phase_offset` | Hz | `None` | `0.0` |
| `phase_sample_rate` | Hz | `None`: the waveform's sample rate | `0.0` |

`bandwidth` and `phase_sample_rate` are resolved by `echo` from the
radar (`sample_rate` for FMCW and pulsed, `num_subcarriers *
subcarrier_spacing_hz` for OFDM). An explicit value overrides; an explicit
`0.0` is refused. Today `NoiseSpec(noise_figure_db=10.0)` is a no-op because
the bandwidth defaults to zero.

`Agc(target_rms, mode="per_rx", min_gain_db=-60.0, max_gain_db=60.0)` and
`Adc(bits, full_scale)` are today's records without the `Spec` suffix. The
LNA is one number and is the `lna_gain` field of `Radar`; `PortSpec` is
the `impedance` field; `SeedSpec` is the `seed` field.

### 4.6 Targets

`PointTargets` merges `ScatterSitePolicy.explicit(...)` and
`ScalarRcsResponse.from_rcs(...)`.

| Field | Type | Unit | Default |
| --- | --- | --- | --- |
| `positions` | `(S, 3)` tensor or sequence | m | required; a live tensor passes through untouched, so a `requires_grad` leaf or a forward dual keeps its tape into both legs |
| `rcs` | float or `(S,)` tensor | m² | required |
| `phase` | float or `(S,)` tensor | rad | 0.0 |
| `trajectory` | `Callable[[float], Tensor]` or `None` | | `None`; `trajectory(time)` returns the same ordered material points at that instant, `(S, 3)` m; replaces today's `at(time) -> Kinematics` (the loop never reads the velocities) |
| `ids` | `tuple[int, ...] \| None` | | `None`, allocated |
| `aspect` | `Aspect \| None` | | `None`, isotropic scalar RCS |

`Aspect(axis, exponent, coherent_interval, phase_rate=0.0)`
carries today's `AspectScatterResponse` parameters; the coefficient math stays
in `scattering.py`.

`StructureTargets(rcs, *, structure_ids=None, ids=None)` is today's
`structure_anchor` policy and is never a default: `targets` is a required
positional argument of `trace` and `simulate`.

Device and carrier are taken from the radar at trace time. The site
excitation power stays the module constant `SITE_EXCITATION_POWER_W` and is
not a field; its own note says changing it double-counts transmit power.

### 4.7 Propagation: two keywords, not a type

`trace`, `simulate` and `stream` take the propagation request as keywords:

| Keyword | Type | Default | Meaning |
| --- | --- | --- | --- |
| `los` | bool | `True` | include the line-of-sight leg |
| `reflections` | int | 1 | maximum specular bounces per leg; 0 means none |

They map to today's `components` and `max_depth`: `{"los"}` when `los`,
`{"reflection"}` when `reflections > 0`, `max_depth = reflections`. Both off
is refused by `trace`. `Paths` records both values as provenance, because
`path_set_complete` is a statement relative to the requested components.

There is no public `Propagation` record. Two fields do not meet the R-ADR-021
bar for a parameter object, and the reason today's `PropagationConfig` exists
is internal: it is the exact request block the Channel adapter is handed, and
a boundary test pins its keyword set. That block stays as a private record in
`propagation.py`, built by `trace` from the two keywords and
`Radar.carrier_hz`; the public surface does not need to mirror it.

### 4.8 `Motion`

One record replaces `motion_sampling`, `adaptive_motion`, `world_motion` and
`motion_event_period_frames`, and removes the rule that two of them must agree.

| Constructor | Meaning today |
| --- | --- |
| `Motion.auto()` | default; resolves to `adc()` when anything moves (structure trajectories or deformations, endpoint trajectories, a target trajectory) or the receiver has phase noise, else to `static()` |
| `Motion.static()` | one observation per frame; refused for a moving world, because it would publish no Doppler |
| `Motion.adc()` | `motion_sampling="adc"`: world refreshed at every ADC instant, the exhaustive reference |
| `Motion.chirp()` | `motion_sampling="chirp"`: stop-and-hop, frozen within a chirp |
| `Motion.adaptive(phase_error_rad=0.02, relative_amplitude_error=0.02, max_interval_s=0.002, nodes=2, max_evaluations=8192, batch_observations=256)` | `AdaptiveMotionSpec`, same fields, `interpolation_nodes` shortened to `nodes` |

Advanced fields on every constructor: `rediscover_every_frames: int | None`
(today `motion_event_period_frames`) and `world: str = "frozen_world"` (today
`world_motion`, Channel vocabulary, unchanged).

`grad` on `trace`/`simulate` replaces `ad_mode` and keeps the capability
vocabulary: `"none"`, `"vjp"`, `"jvp"`.

### 4.9 `Paths`

The product of `trace`. Data only, immutable, one record for the whole run.
Section 5 defines what it holds per motion kind.

Schedule:

| Field | Shape | Meaning |
| --- | --- | --- |
| `kind` | | `"static"`, `"chirp"`, `"adc"` or `"adaptive"` |
| `times` | `(F,)` host | frame instants |
| `observation_times` | `(N,)` host float64 | absolute instant of every observation |
| `observation_frame` | `(N,)` host int | frame index of each observation |
| `waveform` | `Fmcw \| Ofdm \| Pulsed \| None` | the waveform that scheduled the observations; `None` for `static` |

Rows, device tensors, in the Channel phasor convention at `carrier`:

| Field | Shape | Meaning |
| --- | --- | --- |
| `delay_s` | `(R,)` float32 | round-trip delay, today `RadarPathBatch.total_delay_s` |
| `transfer` | `(R,)` complex64 | composed transport, today `complex_transfer_ref` |
| `row_valid` | `(R,)` bool | the sole authority on whether a row means anything |
| `row_pair` | `(R,)` int | sensor pair of the row |
| `topology` | per epoch | today's `RadarPathTopology`: site, inbound and outbound leg identity per row |

Tables that turn rows into observations, host integer arrays as in today's
adaptive route:

| Field | Shape | Meaning |
| --- | --- | --- |
| `evaluated` | `(E,)` | observation indices that were evaluated |
| `observation_rows` | `(E + 1,)` | row offsets per evaluated observation |
| `node_index` | `(N, K)` | which evaluated observations feed each observation |
| `basis` | `(N, K)` | interpolation weights, rows sum to one; `K = 1` for every kind but adaptive |

Session bookkeeping, per frame: `epochs`, `rediscovery_reasons`,
`compile_count`, `discovery_count`, `path_set_complete`,
`motion_sampling_exhaustive`, `adaptive_diagnostics`. These leave `Result`
and live here, because they describe the trace.

Request provenance: `los`, `reflections`, `grad` and the resolved `Motion`,
so a `Paths` states what its completeness is relative to.

`last`: the closing observation's `snapshot`, `compiled_scene`, `legs` and
`composed` batch. This is today's four `last_*` diagnostics, now a field of
the record that produced them. Retention rule unchanged: it aliases that
observation's device tensors and holds no tape.

Methods: `frame(i) -> Paths` (one frame's slice), `rows(frame, observation)
-> RadarPathBatch` (a typed view for inspection), `detach() -> Paths`,
`frame_count`, `observation_count`, `row_count`.

### 4.10 `Result` and `Frame`

`Result` replaces `RadarSimulationResult`.

| Field | Meaning |
| --- | --- |
| `cube` | `[frame, tx, rx, slow, fast]`, unchanged |
| `axes` | `ProcessingAxes`, built by `echo` from the radar's waveform spec and antenna layout, so no caller assembles it |
| `times`, `sample_times` | unchanged |
| `kind`, `phasor`, `time_dependence`, `output_domain`, `carrier` | unchanged conventions |
| `motion` | the resolved `Motion` |
| `los`, `reflections`, `path_set_complete`, `motion_sampling_exhaustive`, `epochs`, `compile_count`, `discovery_count`, `adaptive_diagnostics` | copied from the trace so a fused run still publishes them |
| `radar` | the radar that produced it |

No `last_*` fields. A result retains its cube and nothing else from the run.

`Frame` is `result.frame(i)`: `cube` `[tx, rx, slow, fast]`, `axes`,
`time`. Its methods are facades over `witwin.radar.processing` and add no
math:

| Method | Delegates to |
| --- | --- |
| `processing_cube()` | `ProcessingCube(cube, axes)` |
| `range_profile(*, window="hann", remove_dc=False)` | `range_profile` |
| `range_doppler(*, window="hann")` | `range_profile` then `range_doppler_map` |
| `point_cloud(*, pfa, guard_cells, training_cells, route, max_points, ...)` | `range_doppler`, `combine_incoherent`, `ca_cfar`, `point_cloud`; every keyword forwards by name |

The processing products (`RangeProfile`, `RangeDopplerMap`, `Detections`,
`PointCloud`) do not gain methods; R-ADR-017 keeps that surface functional.

## 5. The trace/echo contract

### 5.1 What each stage owns

Today's frame loop has eight steps. The split puts the first five in `trace`
and the last three in `echo`:

| Step | Stage | Owner module |
| --- | --- | --- |
| sample the Core world at the observation instants | trace | `simulation` |
| compile or reuse the Channel epoch | trace | `channel`, `propagation` |
| discover or re-evaluate one-way topology | trace | `channel`, `propagation` |
| compose round trips, apply the scatter response | trace | `paths`, `scattering` |
| apply antenna pattern weights | trace | `sensors` |
| synthesize the waveform per observation and assemble the frame | echo | `synthesis` |
| apply the receiver chain, then the output-domain transform | echo | `frontend`, `processing.range_doppler.fmcw_range_fft` |
| build the typed result and its processing axes | echo | `simulation`, `processing.signal` |

Two facts decide the boundary. First, the observation schedule is a property
of the waveform: under `adc` sampling the world is evaluated at
`frame + slot * chirp_period + adc_start + m * sample_period`, and the
adaptive phase test includes the beat-frequency sensitivity `|slope| * (...)`.
So `trace` takes its schedule from the radar's waveform and records which
waveform scheduled it. Second, the common-oscillator phase noise is applied
per path at absolute ADC time, before coherent summation; it needs ADC-time
observations but is an instrument effect, so it runs in `echo` and
`echo` refuses static paths when the receiver has phase noise.

### 5.2 What `Paths` holds per motion kind

| Kind | Observations per frame | `K` | Rows retained | Echo route |
| --- | --- | --- | --- | --- |
| `static` | 1 | 1 | one composed batch per frame | frame synthesis, `FROZEN_WEIGHT_WITH_CARRIER_RATE`, native Dirichlet/quadratic-phase kernels |
| `chirp` | `chirps_per_frame * num_tx` (FMCW), symbols (OFDM), pulses (pulsed) | 1 | one batch per slot | per-observation synthesis, `REFRESHED_WEIGHT_NO_RATE` |
| `adc` | slots × `samples_per_chirp` | 1 | one batch per ADC instant | per-observation synthesis in bounded batches |
| `adaptive` | slots × `samples_per_chirp` | `nodes` | one batch per evaluated probe, at most `max_evaluations` | interpolate at the carrier, then per-observation synthesis in bounded batches |

The `adaptive` representation (row table, `node_index`, `basis`) is what
`_adaptive_fmcw` already builds before it synthesizes; `adc` and `chirp` are
the same table with `K = 1` and every observation evaluated, which the
existing final stage already handles ("an evaluated observation is written as
K copies of itself with the first weight one").

### 5.3 `echo` validation

`echo(paths)` refuses, by name:

- `paths.carrier_hz != radar.carrier_hz`;
- a `tx`/`rx` layout whose pair count or order differs from the traced one;
- a sampled `paths.kind` whose `paths.waveform != radar.waveform` (the schedule
  and, for adaptive, the accepted partition belong to that waveform);
- a receiver with phase noise on `static` paths.

It accepts any receiver, any `Fmcw.output`, and, for `static` paths, any
waveform. That is what the split buys: receiver sweeps, seed sweeps and
output-domain choices never re-trace; waveform sweeps re-trace only when the
world moves.

### 5.4 One owner, two consumers

`simulation.py` keeps one generator of observation batches: for each frame,
resolve the epoch, evaluate the observations the motion kind asks for (the
adaptive controller runs its refinement here and emits its accepted table),
compose, weight, and yield the rows with their schedule and bookkeeping.

- `trace` drains the generator into a `Paths` record.
- `simulate` and `stream` hand each batch to the same per-batch synthesis
  that `echo` uses, keep the synthesized samples, and drop the rows.

Because both routes run the same generator and the same per-batch synthesis on
the same rows, `simulate(...)` equals `echo(trace(...))` bit for bit. A
test asserts `torch.equal` for every motion kind, with and without a receiver,
and for FMCW spectrum, FMCW beat, OFDM and pulsed. This replaces today's
`simulate_scene`/`stream_scene` pair, which is the same pattern with one
retention policy fewer.

### 5.4a Where the cut went

This section located the seam before the work; it now records the cut. The seam
held where it was predicted, in `_scene_frames` and in the adaptive route, with
one structural difference and three surface ones.

`_open_session` is the session setup this section assigned to `trace`, and it
returns two things rather than one. The first is a `_Session`: everything the
instrument half needs that does not change between frames - the array, the epoch
loop, the three waveform specs, the TDM offsets, the sampling flags and the
callable that re-derives those specs for another radar. Nothing on it describes
the world, which is exactly the property that lets one traced session be echoed
against a different receive chain. The second is an iterator of `_FrameTrace`,
one per frame, carrying that frame's schedule, its two completeness statements
and its observations.

A `_FrameTrace`'s observations are an ITERATOR, drained exactly once. That is
the whole retention difference between the two routes, and it is the cleanest
part of the cut: `_scene_frames` drains it one observation at a time as it
synthesizes, so a fused run never holds more than one, while `trace_scene`
materialises it into a tuple and keeps it in the `Paths`. `_echo_frame` is the
instrument half and is the only caller of synthesis, so both routes reach the
kernels through one function rather than through two loops that have to be kept
in agreement. `echo_paths` rebuilds only the instrument fields of the stored
session for whatever radar it was handed (`_instrument_specs`), after
`_require_same_instrument` has refused a radar the rows do not describe.

The adaptive cut went where this section said it would. `_adaptive_trace` is the
host-side refinement and publishes the accepted partition as an
`_AdaptiveTable`: `delays`, `transfers`, `validity`, `starts`, `counts`,
`node_index`, `basis` and `clock`, exactly the construction named above.
`_adaptive_echo` is the batching loop below it and is where every kernel launch
happens. An adaptive `_FrameTrace` yields a single observation, the one that
closed the frame, because its rows live in that table rather than in one batch
per instant; `Paths.rows` refuses any other observation index by name instead of
answering with that one.

The bit-exactness test was written first, as this section said it should be. It
is `tests/test_trace_echo_split.py`, and it asserts `torch.equal` on every
motion kind, in both FMCW output domains, with and without a receive chain, and
with oscillator phase noise.

Three places where the shipped records differ from sections 4.9 and 4.10:

1. **`Paths` keeps each motion kind's own row layout.** Section 4.9 proposed one
   table shape for every kind, the adaptive tables with `K = 1` and every
   observation evaluated for the other three. That would have routed static,
   chirp and ADC rows through the interpolation path in order to describe them
   uniformly. The shipped record holds what each kind already produces instead:
   one composed batch per evaluated observation for static, chirp and ADC, and
   the `_AdaptiveTable` for adaptive. No route's numerics moved, which is what
   makes the `torch.equal` assertion available at all, and a uniform description
   is not worth a rewritten kernel path. `Paths.row_count` and
   `Paths.observation_count` read both layouts.

2. **The detection facade is `Frame.points`, not `Frame.point_cloud`.** The
   processing fence forbids the detector, angle-estimator and beamformer names -
   `ca_cfar`, `os_cfar`, `music_spectrum`, `point_cloud` and their siblings - in
   every production module outside the processing package, by name and with no
   allowance list. A facade is not a reason to blunt that fence, so the method is
   named for the product it returns rather than for the stage that produces it.

3. **The axis-name tuple is `Result.axis_names`, so `axes` can be the metadata
   record.** Section 4.10 gave `axes` to `ProcessingAxes`, while the result it
   replaced already used that name for the cube's axis-name tuple. Both are
   needed and neither is derivable from the other: `axis_names` is what the
   result checks the cube's rank against and what the synthesis view reads to
   name the slow and fast axes, and `axes` is the record every processing stage
   takes. They now have one name each.

### 5.5 Memory

`trace` retains every evaluated row. Rows cost about 20 bytes each (float32
delay, complex64 transfer, validity, pair and identity columns), so:

```
bytes ≈ 20 × Σ_frames Σ_evaluated_observations live_rows
```

Worked example, the single-point scene with the wall (3 paths × 12 pairs = 36
rows) under `Motion.adc()`: 128 chirps × 3 TX × 256 samples = 98,304
observations per frame, 3.5 M rows, about 71 MB per frame. A world with 100
paths costs 2.4 GB per frame. `simulate` and `stream` never hold more than one
synthesis batch of rows, as today; `Motion.adaptive()` is bounded by
`max_evaluations` rows per frame in both routes, as today.

The `Paths` docstring states the formula. `trace` does not refuse a large
history and does not warn; the caller who wants bounded memory calls
`trace` per frame or uses `simulate`.

### 5.6 Gradients

`grad="vjp"`: rows in `Paths` carry the tape from the supported leaves (site
positions, element positions, mesh vertices, material parameters, RCS and
phase); `echo` extends it; a loss on `Result.cube` reaches the leaves
through both stages. `grad="jvp"`: forward duals pass through the same way.
`Paths.detach()` is the inspection-only copy. The no-tape-retention rule that
`tests/test_tape_containment.py` enforces on the four `last_*` members
applies to `Paths.last`.

## 6. The flat configuration as a loader

`Radar.from_dict` accepts today's flat mapping with today's units and builds
the record through `Fmcw.from_ti` and `antenna_unit="half_wavelength"`.
This is the only place vendor units are read.

| Key | Unit | Becomes |
| --- | --- | --- |
| `fc` | Hz | `carrier_hz` |
| `slope` | MHz/µs | `Fmcw.slope_hz_per_s` |
| `sample_rate` | kSPS | `Fmcw.sample_rate_hz` |
| `adc_samples` | count | `Fmcw.samples_per_chirp` |
| `adc_start_time`, `idle_time`, `ramp_end_time` | µs | `Fmcw.adc_start_s`, `idle_s`, `ramp_end_s` |
| `chirp_per_frame` | count | `Fmcw.chirps_per_frame` |
| `output_domain` | | `Fmcw.output` |
| `power` | dBm | `power` |
| `tx_loc`, `rx_loc` | half-wavelength | `tx`, `rx` with `antenna_unit="half_wavelength"` |
| `num_tx`, `num_rx` | count | checked against `len(tx_loc)`, `len(rx_loc)` |
| `antenna_pattern` | mapping | `Pattern.separable` or `Pattern.table` |
| `frame_per_second`, `num_doppler_bins`, `num_range_bins`, `num_angle_bins` | | refused with a message naming them: nothing consumes them |

Unknown keys are refused. `RadarConfig` and the `validate_*` functions become
private helpers of the loader.

## 7. Renamed, deleted, moved

Renamed, same meaning:

| Today | New |
| --- | --- |
| `RadarSimulationResult` | `Result` |
| `FmcwWaveformConfig`, `OfdmWaveformConfig`, `PulsedWaveformConfig` | `Fmcw` (SI fields), `Ofdm`, `Pulsed` |
| `SensorArraySpec` | `Radar.tx`, `Radar.rx`, `Radar.antenna_unit` |
| `AntennaPatternSpec` | `Pattern` |
| `FrontendSpec`, `NoiseSpec`, `LnaSpec`, `AgcSpec`, `AdcSpec` | `Radar.noise`, `lna_gain_db`, `agc`, `adc`, `seed`; `Noise`, `Agc`, `Adc` |
| `AdaptiveMotionSpec` | `Motion.adaptive(...)` |
| `Radar.set_pose(...)` | `Radar.replace(position=..., look_at=...)` |
| `ad_mode` | `grad` |

Deleted from the public surface:

| Symbol | Replacement |
| --- | --- |
| `RadarConfig`, `RadarSystemConfig`, `ProcessingConfig`, `SensorConfig`, `WaveformConfig` | `Radar` is the record; `from_dict` is the loader |
| `PropagationConfig` | a private request block in `propagation.py`; `los` and `reflections` keywords on the verbs |
| `Radar.fov`, `Radar.name`, `Radar.c0`, `Radar.system_config`, `Radar.config` | none; `Radar.carrier_hz` and the part records |
| `Radar.last_result`, `last_snapshot`, `last_compiled_scene`, `last_propagation`, `last_radar_paths` | `Paths.last` |
| `ScatterSitePolicy` | `PointTargets`, `StructureTargets` |
| `ScalarRcsResponse.from_rcs` as a caller-facing step | `PointTargets.rcs_m2`; the class stays as the coefficient owner |
| `DEFAULT_POLARIZATION` | `Radar.polarization` |
| `PortSpec`, `SeedSpec`, `TxPowerSpec` | `Radar.impedance_ohm`, `Radar.seed`, `Radar.power_dbm` |
| `simulate(..., sites=, response=, components=, max_depth=, ad_mode=, world_motion=, motion_event_period_frames=, ids=, polarization=, sensor_endpoints=, motion_sampling=, adaptive_motion=)` | `targets`, `los`, `reflections`, `motion`, `grad`, `endpoints` |
| `Radar._synthesize` as the route to processing axes | `Result.axes` |

## 8. Module ownership

New names live with today's concept owners; `ci/architecture-manifest.json`
changes only where a module is added.

| Name | Module |
| --- | --- |
| `Radar`, `Fmcw`, `Ofdm`, `Pulsed`, the loader | `radar.py` |
| `Pattern`, the half-wavelength conversion, the pattern weighting | `sensors.py` |
| `Noise`, `Agc`, `Adc`, the stage chain | `frontend.py` |
| `PointTargets`, `StructureTargets`, `Aspect` | `targets.py`, new: the target set has its own vocabulary (RCS, trajectory, identity) and its own reason to change; `scattering.py` keeps the coefficient math and `simulation.py` keeps the binding |
| `Motion` | `simulation.py` |
| `Paths` | `paths.py` |
| `Result`, `Frame` | `simulation.py` |


## 9. Defaults that change physics (decided 2026-09-17)

Two defaults are decisions, not renames. Both were decided on 2026-09-17.

### 9.1 Antenna pattern: isotropic by default (decided)

Today `Radar.simulate` always passes the stored pattern, and the stored pattern
falls back to a half-wave dipole cut, so every default run is attenuated
off-boresight by a number the caller did not choose. The `_scene_frames`
docstring itself argues against adopting it silently. The default is
`Radar.pattern = Pattern.isotropic()`, with `Pattern.dipole()` one keyword away.
Cost: every test that relied on the implicit dipole sets it explicitly, which
is 12 test files by grep.

### 9.2 `Motion.auto()` resolves to `adc`, the exhaustive reference (decided)

This keeps today's default. `adaptive` is 3 to 64 times faster on the measured
scenes but certifies nothing between probes; a default must be the safe one.

Not changed: the -Z boresight of the default pose, the `"spectrum"` FMCW
output, the unit site excitation, the receiver stage order.

## 10. Migration plan

Five commits, each leaving the suite green on the new surface, no aliases at
any point.

1. **Records and loader.** Add `Fmcw`, `Ofdm`, `Pulsed`, `Pattern`, `Noise`,
   `Agc`, `Adc`, `Motion`, `PointTargets`, `StructureTargets`, `Aspect`.
   Rebuild `Radar` as the flat immutable record with `from_dict`. `simulate`/`stream` take the new keywords. Move every test
   fixture and example construction. Files: `radar.py`, `sensors.py`,
   `frontend.py`, `propagation.py`, `simulation.py`, `targets.py`,
   `tests/conftest.py`, `tests/support/*`, `tests/core/*`, and the 28 test
   files that construct a radar or call `simulate` (list in the appendix).
2. **`Paths`, `trace`, `echo`.** Extract the observation-batch generator
   from `_scene_frames` and `_adaptive_fmcw`; add `Paths` and the two verbs;
   make `simulate`/`stream` fused consumers; delete the `last_*` properties;
   add the bit-exact equality tests and the row-count tests.
3. **`Result`, `Frame`, processing seam.** Build `axes` in `echo`; add
   `Frame` facades; delete `frame_synthesis`; rewrite the three examples and
   the six `tools/validate_*.py` scripts; delete every `_synthesize` call
   outside `synthesis`.
4. **Governance.** `ci/public-api-manifest.json` (root exports and
   `root_class_members`), regenerate `ci/public-api-snapshot.json` with
   `python tests/test_public_api_snapshot.py`, `ci/architecture-manifest.json`
   (`targets.py`), `ci/documentation-manifest.json` retired tokens
   (`RadarConfig`, `ScatterSitePolicy`, `last_radar_paths`, `set_pose`,
   `FrontendSpec`, `AdaptiveMotionSpec`, `ad_mode`).
5. **Documents.** `README.md` Main API, `FEATURE_LIST.md`,
   `docs/pipeline_guide.md`, `AGENTS.md` and `CLAUDE.md` public entry points
   (the current text names a `Radar.synthesize` that does not exist),
   `docs/dev/radar-ad-capability-matrix.md` where it names `ad_mode`.

Verification after every commit, from the repository root:

```bash
python -m ruff format --check witwin/radar tests examples tools ci scripts
python -m ruff check witwin/radar tests examples tools ci scripts
python ci/check_architecture.py
python ci/check_duplicate_code.py
python ci/check_documentation_surface.py
python ci/check_no_compatibility.py
python ci/check_public_api_manifest.py
pytest tests/
pytest tests/ --gpu
```

New tests, all in commit 2 unless noted:

- `simulate == echo(trace)` bit-exact per motion kind and waveform.
- `Paths.row_count` equals the formula in section 5.5 for `adc` and `chirp`;
  is at most `max_evaluations` rows per frame for `adaptive`.
- `echo` refusals of section 5.3, each by message.
- `Motion.auto()` resolution table, including the phase-noise rule.
- `Motion.static()` refused on a moving world.
- Pose refusals on `Radar`; polarization parallel to the boresight is refused
  rather than returning zeros.
- `Noise.bandwidth_hz=None` resolves to the waveform bandwidth; `0.0` refused
  (commit 1).
- Loader unit table round trip: `Radar.from_dict(CONFIG).waveform_spec()`
  equals today's `FmcwSpec` for the standard fixture (commit 1).
- `Frame.range_doppler()` equals the functional chain on the same cube
  (commit 3).

## 10a. Decided while implementing (2026-09-17)

Three things the draft did not settle, decided against the code:

1. **No unit suffix on any redesigned public name.** `carrier`, not
   `carrier_hz`; `times`, not `times_s`; `Fmcw.slope` in Hz/s, not
   `slope_hz_per_s`. The unit is on the field's docstring row. The rule stops
   at this surface: `witwin.radar.processing` is frozen by R-ADR-017 and keeps
   `range_m` and its siblings, and the internal records
   (`FmcwSpec.sample_period_s`, `RadarPathBatch.total_delay_s`) keep theirs,
   because those are the equation-owning modules where R-ADR-021 requires the
   unit to be in the identifier. `Fmcw.from_ti` is the one exception on the
   public surface: its parameters carry the vendor units (`slope_mhz_per_us`,
   `sample_rate_ksps`) because naming the column each number came from is its
   entire job.

2. **A target declares its strength as `rcs` or as `amplitude`, never both.**
   The draft had only the cross section, but `ScalarRcsResponse.from_values`
   exists because a test oracle or an optimiser wants the dimensionless
   strength itself as the leaf, and routing it through the cross section would
   put a square root in its gradient. Both records take either, exactly one,
   and a live tensor keeps its tape.

3. **The radar holds no run state, including `last_result`.** The draft kept
   that one property. It cannot be an instance attribute on a frozen record
   without `object.__setattr__`, and making it a class attribute would share
   one result between every radar in the process. Both are worse than the
   alternative, which is that `simulate` returns the result and the caller
   keeps it. All four typed diagnostics already live on that result, so
   deleting the property removes a second owner rather than a capability, and
   "a failed call leaves no stale world behind" becomes structural instead of
   something the entry point has to remember to do.

## 11. Decisions taken (2026-09-17)

1. Section 9.1: the default antenna pattern is isotropic.
2. Section 9.2: `Motion.auto()` resolves to `adc`.
3. The second verb is `echo`: `paths = radar.trace(...)`,
   `result = radar.echo(paths)`. It is the radar word for the received
   return, it works as a verb, and `synthesis/pulsed.py` already calls the
   received samples echo rows. The result type stays `Result`, as in
   `witwin.maxwell`.
4. Section 8: `targets.py` is a new module.
5. Section 4.7: propagation is two keywords on the verbs, not a public type.
6. `Array` is not a type: `tx`, `rx` and `antenna_unit` are fields of `Radar`.
7. `Receiver` is not a type: the stages are fields of `Radar`; a one-number
   stage is a keyword, a coupled stage is a record.
8. `Pose` is not a type: `position`, `look_at`, `up` and `polarization`
   are fields of `Radar`. Rotation input stays out until someone needs it.

## Appendix: files touched

Tests that construct a radar, call `simulate`/`stream`, read `last_*`, or use
`RadarConfig` or `set_pose` (38):

```
tests/conftest.py
tests/core/__init__.py
tests/core/test_antenna_pattern.py
tests/core/test_radar_config.py
tests/core/test_radar_pose.py
tests/processing/__init__.py
tests/support/exact_bin_grid.py
tests/support/multi_endpoint_driver.py
tests/support/pipeline_chain.py
tests/support/spike_driver.py
tests/test_adaptive_motion.py
tests/test_consolidation_governance_gates.py
tests/test_correlated_phase_noise.py
tests/test_dynamic_motion_sampling.py
tests/test_fmcw_continuous_motion.py
tests/test_frame_streaming.py
tests/test_frontend_output_domain.py
tests/test_native_diagnostics.py
tests/test_static_gates.py
tests/test_antenna_pattern_route.py
tests/test_scene_binding.py
tests/test_simulate_entry.py
tests/test_fmcw_beat_spec.py
tests/test_fmcw_beat_kernel.py
tests/test_config_vocabulary_boundary.py
tests/test_fmcw_analytic.py
tests/test_synthesis_launch_budget.py
tests/test_no_torch_physics.py
tests/test_moving_scene_acceptance.py
tests/test_processing_surface_freeze.py
tests/test_pipeline_budget.py
tests/test_phase8_wideband_ofdm.py
tests/test_ad_chain_coverage.py
tests/test_sensor_constant_refusal.py
tests/test_tape_containment.py
tests/test_public_api_snapshot.py
```

Examples, tools and CI scripts (10):

```
examples/single_point.py
examples/music_imaging.py
examples/rgbd_range_doppler.py
tools/validate_adaptive_motion.py
tools/validate_adaptive_probe_efficiency.py
tools/validate_adaptive_tolerance.py
tools/validate_doppler_motion.py
tools/validate_frame_streaming.py
tools/validate_heavy_multipath.py
ci/check_orphan_modules.py
```

The notebooks under `examples/` mirror the scripts and are regenerated from
them.
