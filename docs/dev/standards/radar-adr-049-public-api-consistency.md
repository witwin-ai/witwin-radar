# R-ADR-049: Public API consistency - typed metadata, capabilities, eager validation, facades and names

Status: Accepted (0.5.0)

## Context

The audit (plan section 3.8: U-2, U-7, U-8, U-9 and the Channel family table)
found the public surface loose in four ways:

* metadata typed `object` or `dict` (`Result.last_snapshot`,
  `last_compiled_scene`, `adaptive_diagnostics: tuple[dict]`), with no
  vocabulary shared with Channel's `ResultSemantics` (`estimator`, `complete`);
* no `capabilities()` record, only a developer markdown matrix;
* records that accepted inputs a run could never use (`Fmcw(slope="60e12")`,
  an ADC window four milliseconds long on a 65 us ramp, `Pulsed(pulse_kind="barker")`,
  `positions=[(0, 0)]`, `rcs=-1`), and a CPU `Radar` that reached Channel and
  failed there on "endpoint tensors must be CUDA tensors";
* names that disagreed with Channel and with the module's own stated rule
  (section 5).

Facades were also missing where the agents of this release asked for them:
`Frame.points` failed on a DDM or BPM cube, and imaging had no `Frame` route.

## Decision

### 1. Typed metadata in Channel's vocabulary

* `witwin.radar.adaptive.AdaptiveDiagnostics` is a `TypedDict` (a plain `dict`
  at run time, so saved results and datasets keep their JSON) with every key
  the adaptive controller publishes, always present: `evaluations`,
  `observation_count`, `topology_refinements`, `accepted_intervals`,
  `max_tested_phase_error_rad`, `max_tested_relative_amplitude_error`,
  `max_interval_s`, `interpolation_nodes`, `verification_probes`,
  `verification_refinements`, `exhaustive`, `topology_proved_complete`,
  `complete`, `synthesis_batches` (`0` until an echo). `complete` is the one
  new key and carries Channel's meaning: no path within the requested bounds
  can have been missed in the frame. The session reads the frame's
  `path_set_complete` from it, so the rule has one owner.
* `Result.last_snapshot` is a Core `SceneSnapshot | None` and
  `last_compiled_scene` a `CompiledWorld | None`, a protocol in
  `propagation.py` naming the one member Radar calls on Channel's compiled
  scene (`require_reference_frequency`). Channel's consumer facade does not
  export `CompiledScene`, and Radar reaches Channel only through that facade,
  so a protocol is the typed name available. `EpochFrame` and the epoch loop's
  compile callable carry the same two types.
* `Result.semantics` and `Paths.semantics` return a `RunSemantics`
  `TypedDict` with Channel's keys and meanings: `estimator="coherent"`,
  `combine="field"`, `complete` (= `path_set_complete`), plus Radar's own
  `exhaustive` (= `motion_sampling_exhaustive`), which Channel has no motion to
  need. The flat fields stay; `semantics` is the cross-package view.

### 2. `witwin.radar.capabilities()`

A versioned (`schema_version = 1`) plain-JSON record, a fresh copy per call,
in the shape of `witwin.channel.capabilities()`. Nothing is listed in
`capabilities.py`: every entry is read from the owner of the vocabulary -
`WAVEFORM_KINDS`, `FMCW_OUTPUT_DOMAINS`, `FMCW_SAMPLINGS`, `MIMO_SCHEMES`,
`OFDM_MODULATIONS`, `OFDM_OUTPUT_DOMAINS`, `PULSE_KINDS`; `MOTION_KINDS`,
`CADENCES`, `WORLD_MOTIONS`; `PROPAGATION_COMPONENTS` (now also what
`Radar._session` builds the request from) and the composer's
`COMPONENT_NAMES`; the processing vocabularies; the native prebuilt's build
record read from disk (`operator_symbols`, architectures, ABI) without loading
the library; Channel's consumer contract through `channel.py` when Channel is
installed. AD leaves are a `LEAVES` mapping (field -> AD modes) declared on
each owner (`Radar`, `PointTargets`, `Pattern`, `Coupling`, `SMPLBody`, and
`facets.MATERIAL_LEAVES` for Core materials on a mesh target); `Radar`'s pose
refusals iterate the same `POSE_LEAVES`. `tests/test_capabilities.py` ties each
section to its owner: every listed option is accepted and an unlisted one
refused, every declared leaf is a field of its owner, the mirrored world
motions equal Channel's live `world_motions`, and the native list equals the
prebuilt's record. The authority for every AD cell stays
`docs/dev/radar-ad-capability-matrix.md`.

### 3. Validation at construction

Every public record refuses a bad field in `__post_init__`, naming
`Record.field`:

* `policy.require_real` is the one host-number admission: a numeric string, a
  bool, a complex value, NaN and infinity are refused rather than coerced by
  `float()`. `require_host_floats` (every frontend and synthesis spec) now
  calls it, so `Iq(gain_db=nan)` is refused; bool flags stay legal there.
* The waveforms check their vocabularies by field name (`Fmcw.output`,
  `.sampling`, `.mimo`, `Ofdm.modulation`, `.output`, `Pulsed.pulse_kind`) and
  then build a one-transmitter probe of their synthesis spec, so the spec's own
  cross-field rules (chips only on a coded pulse, a periodic code's PRI) run at
  construction and are reported as `"Fmcw: ..."`. `Radar` builds its real spec
  in `__post_init__`, so the rules that need the array (BPM's Hadamard order,
  DDM bands per transmitter) refuse the radar rather than its first run.
* New rule, owned by `FmcwSpec`: the ADC window `t_start + (N - 1) / f_s` must
  end inside the chirp period `idle + ramp_end`, the rule a chirp sequence
  already had. A window past it would sample the next chirp. One existing test
  (`tests/core/test_radar_config.py::test_range_resolution_scales_with_adc`)
  built such radars to test axis arithmetic; it now stretches the ramp.
* Targets: `positions` is `(S, 3)` with `S >= 1` (a tensor is checked for shape
  and dtype only, never read, so construction does not synchronize);
  host `rcs` is finite and non-negative, `amplitude`/`phase` finite, per
  target or shared; `ids` and `structure_ids` are distinct non-negative ints
  and `ids` has one per target; a static `Aspect.axis` or `RcsTable.axis` has
  one row per target; `ScatteringMatrix` factors are finite complex numbers.
* `Motion.world` is one of `WORLD_MOTIONS` (Channel's `world_motions`,
  mirrored so a record can be refused without importing Channel and pinned
  against the live record by the capability test).
* `Radar.waveform` and the receive-chain stages are type checked.
* U-2: `Radar.trace`, `simulate`, `stream` and `echo` refuse a radar whose
  device is not CUDA before any work, naming the device.

### 4. Facades

* `Frame.virtual_cube()` is `demultiplex(processing_cube())` (identity for
  TDM). `Frame.points()` and the new `Frame.range_angle(azimuth_rad)` read it,
  so a DDM or BPM frame gets one channel per transmitter. The processing routes
  that cannot demultiplex (they take a range-Doppler map; demultiplexing is a
  slow-time operation) refuse a non-separated DDM/BPM product by name
  (`processing.angle.require_separated`) instead of failing on a shape.
* `Frame.image(aperture, pixels_m, range_bins=)` is
  `backproject(range_profile(virtual_cube(), range_bins=), aperture, pixels_m)`
  and `Frame.aperture(radar, velocity_mps=, pose=)` is
  `Aperture.from_radar(radar, time_s=frame.time_s, ...)` after checking the
  radar against the frame's axes. The radar is an argument, not a member of
  `Result`/`Frame`: a result is saved and loaded, and a radar whose pose or
  power is a leaf has no saved form; holding it would also hold its tape.

### 5. Profiler-derived budget counts

`tests/test_session_budget.py` and the mesh-target budget pinned profiler
kernel and copy counts that moved between runs (`auto_point` h2d 65 vs 62,
`adc_point_per_frame` kernels 862 vs 845). Two sources, both measured:

* **Process history.** The same case measured alone is identical across
  processes and hash seeds; measured after other cases it uploads and
  launches a little less (chirp_structure first frame h2d 173 alone, 170 after
  the other cases). Process-global first-use caches - Channel's compile
  registry rebinds a new source's runtime structures to an earlier one with
  the same geometry, and others - let earlier tests save work. Clearing the
  compile registry alone did not remove it. The harness now measures each
  case in a fresh interpreter (`support.session_ledger.measure_isolated`).
* **Dropped profiler records.** Within one process, two passes of identical
  warm work differed by one upload or a few dozen kernels in the largest case
  (`adc_structure`, 24k kernels per two-frame pass). A dropped record only ever
  lowers a count, so `profiled_copies` takes the largest of three passes; the
  old minimum of two picked the damaged pass.

With both, the profiled columns of an uncontended run are identical between
runs and processes. Under a contended device every pass of a long case can
still lose records (seen: `adc_point` marginal kernels 8284 vs 8292, one
upload), and a loss in the one-frame run raises the marginal column, so the
profiled columns keep a margin of 0.5 % (at least 2), down from 3 %, above the
largest loss measured (0.1 %). Every other counter (Channel calls, native
launches by name, host synchronizations, compositions, trips) stays exact. The
pins were re-measured with this harness; they were stale, the old slack had
absorbed later merges' extra launches (`adc_point_per_frame` kernels 845 ->
862, `adc_point_per_chirp` 1589 -> 1630, `adc_structure` 12106 -> 12202, no
exact counter moved).

### 6. Names: one rule, stated once (M5-6, U-8)

The module docstring of `radar.py` declared "no field name carries a unit
suffix" while `Leakage.isolation_db/delay_s`, `Iq.gain_db/phase_deg`,
`Result.times_s` and `Result.reference_frequency_hz` carried one and `Paths`
said `times`/`carrier` for the same two quantities. Owner decision 5 fixes the
direction: public names align with Channel where they name the same quantity;
the short verb keywords `los=`/`reflections=` stay. The rule, applied once to
the whole public surface (root exports, their fields, the verbs' keywords and
the members `Result`, `Frame` and `Paths` publish):

1. **A quantity Channel also publishes, in Channel's unit, takes Channel's
   name verbatim**, unit suffix included: `reference_frequency_hz` (was
   `Radar.carrier`, `Paths.carrier`; `Result` already used it), `ad_mode`
   (was the verbs' `grad=` and `Paths.grad`; same vocabulary `"none"`,
   `"vjp"`, `"jvp"`), `power_w` (was `Radar.transmit_power_watts`).
2. **Every other public name carries no unit suffix**; its unit is on its
   docstring row. Renamed: `Leakage.isolation` (dB) and `Leakage.delay` (s);
   `Iq.gain` (dB) and `Iq.phase` (**rad**, was degrees: every other phase in
   Radar is in radians, so the unit changes with the name, and a caller gets a
   `TypeError` on the old keyword rather than a silent 57x error);
   `Result.times`, `Result.sample_times` (the names `Paths` already used),
   `Result.birth_lag_bound` / `Paths.birth_lag_bound`, `Frame.time`;
   `RcsTable.rcs`, `.frequencies`, `.phase`; `MeshTargets.cluster_size`; the
   waveform records' `to_spec(reference_frequency_hz=)` and
   `Fmcw.max_velocity(reference_frequency_hz=)` take rule 1's name.
3. **Not renamed, by scope.** Radar's transmit power stays `power` in dBm:
   Channel's `power_w` is the same quantity in a different unit, so it is not
   Channel's name (rule 1), and the watt value is published as `power_w`.
   `Emitter.power` follows `Radar.power`. `Network.carrier_offsets` and
   `Emitter.carrier_offset` name an offset Channel does not publish. The
   processing package (`range_m`, `velocity_mps`, `Aperture.times_s`, ...)
   is frozen by R-ADR-017 and its names are its contract; the ground-truth
   labels match it (`rcs_m2`, `range_m`). Equation-owning internal records and
   helpers (`FmcwSpec.sample_period_s`, `RadarPathBatch.total_delay_s`,
   `Noise.noise_power_watts`, `Iq.image_rejection_db`, `PathEvents.time_s`,
   `SceneFrame`) keep the unit in the identifier, as R-ADR-021 requires of an
   equation owner. The flat configuration keys (`fc`, `power`, TI-style) are a
   file format, not field names. `Fmcw.from_ti` names its vendor units on
   purpose.

**World frame.** Channel and Core document +Z up; `Radar` defaults to
`look_at=(0, 0, -1)`, `up=(0, 1, 0)`, a Y-up world, so in a Z-up scene the
default radar looks at the floor. The default is **kept** and documented on
the two fields: changing a default re-aims every existing script and test
without an error, which is the one kind of break this pass avoids (every
rename here fails loudly at the old name). A Z-up scene states its pose.

**Window defaults.** `Frame.range_doppler` defaulted to a Hann slow-time
window and `processing.range_doppler_map` to rectangular. Both are now
rectangular: a facade applies no taper the stage it names would not.
`Frame.range_angle` follows. `Frame.points` keeps `window="hann"` on purpose
(a detector over a rectangular Doppler transform reports sidelobes as
targets) and its docstring says so.

**`Fmcw` derived quantities** (M5-6): `sweep_bandwidth` (`|S| N / f_s`, Hz),
`range_resolution` (`c / 2B`), `max_range` (`f_s c / 2|S|`, the complex
receiver's unambiguous range) as properties, and
`max_velocity(reference_frequency_hz=, num_tx=)` (`lambda / 4 T_slot`) as a
method because the carrier and the transmitter count are the radar's. There
is no `Fmcw.bandwidth`; U-8's "returns the sample rate" refers to a property
that no longer exists.

No aliases: the old names raise. R-ADR-050 holds the migration table.
