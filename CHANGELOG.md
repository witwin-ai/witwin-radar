# Changelog

All notable changes to `witwin-radar` are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the version numbers follow the package metadata.

## [Unreleased]

## [0.5.0] - 2026-11-13

Radar 0.5.0 is one breaking release. It moves to Channel 0.6, makes the physics of the radar equation, the receive
chain and the detectors match their analytic oracles, makes surfaces scatter as physical-optics facets, makes moving
scenes cheap by default (`Motion.auto()` resolves to adaptive sampling for a moving FMCW scene) and precise (float64
round-trip path length), and adds multipath components, coded and multiplexed waveforms, multi-radar networks,
imaging, calibration, tracking, dataset tooling and a public naming pass aligned with Channel. Every correctness
claim is checked against an independent oracle in `tests/oracle/` (R-ADR-032). Upgrading from 0.4:
[`docs/migration-0.5.md`](docs/migration-0.5.md). User guides: [`docs/user/`](docs/user/README.md).

### Breaking changes

**Names and validation**

- One naming pass, no aliases (R-ADR-049; the full table is in R-ADR-050 and the migration guide). A quantity
  Channel also publishes takes Channel's name; every other public name drops its unit suffix. `Radar(carrier=)` /
  `radar.carrier` -> `reference_frequency_hz`; the verbs' `grad=` -> `ad_mode=` (also on `Network`), `Paths.grad`
  -> `Paths.ad_mode`; `Paths.carrier` -> `Paths.reference_frequency_hz`; `Radar.transmit_power_watts` ->
  `Radar.power_w`; `Result.times_s`, `sample_times_s`, `birth_lag_bound_s` -> `times`, `sample_times`,
  `birth_lag_bound` (also `Paths.birth_lag_bound`); `Frame.time_s` -> `Frame.time`; `Leakage(isolation_db=,
  delay_s=)` -> `Leakage(isolation=, delay=)`; `RcsTable(rcs_m2=, frequencies_hz=, phase_rad=)` ->
  `RcsTable(rcs=, frequencies=, phase=)`; `MeshTargets(cluster_size_m=)` -> `MeshTargets(cluster_size=)`;
  `Fmcw/Ofdm/Pulsed.to_spec(carrier=)` -> `to_spec(reference_frequency_hz=)`. The old names raise.
- `Iq(gain_db=, phase_deg=)` -> `Iq(gain=, phase=)`, and **the phase is now in radians**, like every other phase in
  Radar. Migration: `Iq(gain=g, phase=math.radians(p))`; a flat configuration's `iq` block follows.
- Public records validate every field at construction and refuse a bad one by `Record.field` name: a numeric
  string, bool, complex, NaN or infinite number where a real number is expected; an unknown vocabulary word
  (`Fmcw.mimo`, `Fmcw.output`, `Fmcw.sampling`, `Ofdm.modulation`, `Ofdm.output`, `Pulsed.pulse_kind`,
  `Motion.world`); positions that are not `(S, 3)`; a negative host `rcs`; repeated or miscounted `ids`; an
  `Aspect`/`RcsTable` axis without one row per target. These inputs failed later, or produced NaN cubes, before.
- An FMCW ADC window that ends after the chirp period (`adc_start + (samples_per_chirp - 1) / sample_rate >=
  idle + ramp_end`) is refused at construction; it sampled the next chirp. Lengthen `ramp_end` or shorten the window.
- `Radar(...)` builds its synthesis spec at construction, so array-dependent waveform rules (BPM needs a
  power-of-two transmitter count, DDM bands at least the transmitter count) refuse the radar instead of its first run.
- `Radar.trace`, `simulate`, `stream` and `echo` refuse a radar that is not on a CUDA device, naming the device,
  before any work.
- `Radar.from_dict` accepts every `Radar` field and a `waveform` block of SI waveform fields; a mapping that declares
  the waveform twice (a `waveform` key next to the TI-style FMCW keys) is refused with "the waveform is declared
  twice".
- `Result.last_propagation`, `last_radar_paths`, `last_snapshot` and `last_compiled_scene` are typed `Optional`;
  a result read back with `Result.load` carries `None` there and empty `events`.

**Defaults**

- `Motion.auto()` (the default) resolves to `Motion.adaptive()` for a moving FMCW scene instead of evaluating the
  world at every ADC instant. Any other moving scene (OFDM, pulsed, an FMCW chirp sequence, a ramp phase error, DDM
  or BPM), one that declares a discovery cadence, and a moving `Network` still get the exhaustive `Motion.adc()`. Pass `motion=Motion.adc()` where the
  exhaustive reference is wanted.
- `Frame.range_doppler()` (and the new `Frame.range_angle()`) no longer taper slow time by default: rectangular,
  like `processing.range_doppler_map`. Pass `window="hann"` for the previous map. `Frame.points()` keeps its Hann
  default.
- `point_cloud`'s and `Frame.points`' default `route` is `"scan"` (any array layout) instead of
  `"phase_comparison"`, and `point_cloud` groups peaks by default (`peak_neighbourhood=(3, 3)`): the cells one return
  lights up become one point. Pass `peak_neighbourhood=None` for per-cell points.

**Physics and numbers**

- Element patterns and polarization are Radar's (R-ADR-041). Channel's `complex3_transport` is replayed and the
  join divides out Channel's dipole projection, so an `"isotropic"` element is isotropic: off the plane normal to
  the polarization the echo no longer loses `sin^4` of the angle (it lost 5.0 dB at 30 degrees, 12.0 dB at 45, 24.1
  dB at 60). `Pattern.dipole()` attenuates by its cut once instead of twice. Numbers on the polarization plane are
  unchanged to float32 rounding. Drop any compensation for the old loss.
- Pulsed samples are physical amplitudes: the pulse envelope is unit amplitude (`PulsedSpec.pulse_amplitude == 1`,
  `pulse_normalization="unit_amplitude"`), so an in-pulse sample is `C_rt` in sqrt(W) and the pulsed cube is
  `sqrt(T_p)` larger than before. Matched-filter peaks are unchanged. Code that divided the cube by `sqrt(T_p)` must
  stop.
- The default thermal noise bandwidth is the waveform's complex sample rate for every waveform; a pulsed radar no
  longer integrates over its LFM sweep. `Fmcw.bandwidth` is removed (it was the sample rate); use
  `Fmcw.sample_rate`, or the new `Fmcw.sweep_bandwidth` for the swept bandwidth.
- The ADC is a two's-complement mid-tread grid: `Adc.step == 2 FS / 2^b`, zero is a code, outputs span
  `[-FS, FS - step]`, and the clipped count is the saturated-code count. `Adc(bits=1)` is refused.
- `Pattern` tables are normalised and the bound is enforced (power at most 1, `|field|` at most 1); the peak gain
  moves to `Pattern.gain` (dBi). The map table field is renamed `gain_table`.
- Every frame of a noisy run draws its own thermal realisation, keyed by the frame instant; a frame at `t = 0` is
  unchanged. Multi-frame noisy results change.
- A pulsed train on a moving scene under `Motion.adc()` observes the world at every range-gate sample instead of
  once per pulse (intra-pulse Doppler and LFM range-Doppler coupling), at one world observation per gate sample.
  `Motion.chirp()` keeps the per-pulse schedule.
- The outbound leg is evaluated from the receive element toward the site (Lorentz reciprocity, R-ADR-039); composed
  co-polar values move by at most 1.3e-7 relative per row. `RadarLegBatch.coefficient` (a projected scalar) is
  replaced by `RadarLegBatch.field` (`complex64[rows, 3]`), `coefficient_band` by `field_band`
  (`complex64[rows, F, 3]`); `RadarWorldBinding` has one `sites` spec, with the receivers appearing as `receivers`
  (the site-less leg's sinks) and `receiver_sources` (the outbound leg's 1 W sources); `TwoWayComposer.
  outbound_max_depth` is removed.
- A Channel whose fixed-topology replay cannot return `complex3_transport` for the requested components or AD mode
  is refused by name; there is no scalar fallback.
- The sampled-surface recipe (`N` isotropic `PointTargets` of `sigma / N`) is not a surface model and is gone from
  the oracles and tools; a surface is a `MeshTargets` now. `tools/benchmark_smpl_walk.py` scatters from the body's
  facets, and `--sites` is the site budget; its results are not comparable with earlier runs.

**Processing**

- CFAR detectors are square law. Complex input is detected on `|x|^2`; a real input is read as a POWER map. Pass
  complex IQ or `|x|^2`, never `|x|` or decibels. `combine_coherent` returns the power `|sum_p x_p|^2` instead of the
  magnitude.
- `os_cfar`'s `pfa` is the achieved false-alarm rate (the constant is solved from the ordered-statistic law).
  Thresholds and detection counts of every detector change.
- `ArrayGeometry.from_axes` maps the radar's local offsets `(right, up, -boresight)` into the processing frame
  `(right, boresight, up)`; `ArrayGeometry.from_offsets` takes processing-frame offsets. Elevation offsets declared
  in a radar config (`tx_loc` `[x, 1, 0]`) steer as elevation in `conventional_steering`, `beam_cube` and MUSIC.
- `fft2_aoa` reads the array's lattice instead of assuming TI interleaving and four transmitters;
  `phase_comparison_aoa` refuses layouts it cannot read, naming the reason.
- `microdoppler_spectrogram` returns a `MicroDopplerSpectrogram` record instead of a `(times, frequencies,
  spectrum)` tuple, normalised by `1 / window_slots`. Read `.times_s`, `.frequency_hz`, `.data`.
- `point_cloud`, `beam_cube` and `range_angle_map` refuse a DDM or BPM product whose transmitters are not separated,
  naming `processing.demultiplex`. Demultiplex the `ProcessingCube` first, or use the `Frame` facades, which do.

**Results, motion and module layout**

- `Result.motion_sampling` / `Paths.kind` are derived from `observation_schedule`: a slow-time schedule is named
  `"chirp"`, `"symbol"` (frequency-domain OFDM, whatever was requested) or `"pulse"` (pulsed under
  `Motion.chirp()`), a fast-time one `"adc"`. `motion_sampling_exhaustive` is true only for the `frame` and
  `fast_time` schedules.
- `Motion.adaptive()` no longer takes `rediscover_every_frames`; adaptive sampling owns its discovery cadence.
- `AdaptiveMotionSpec` is removed; use `Motion.adaptive(...)`, whose fields are the same tolerances, now validated
  when the `Motion` is built.
- `witwin.radar.simulation` no longer exports `SensorEndpointIds`; import it from `witwin.radar.binding`. `Motion`,
  `Result`, `Frame` and `Paths` stay exported from `witwin.radar`; their owner modules are `witwin.radar.motion` and
  `witwin.radar.results`. `StableIdAllocator`, `ScatterSitePolicy` and `bind_radar_world` live in
  `witwin.radar.binding`, the adaptive controller in `witwin.radar.adaptive`.
- `Result`, `Paths` and per-frame records gain `discovery_complete` and `transmission_dropped_rows`;
  `path_set_complete` is also false when Channel's own discovery was not provably complete. `RadarPathTopology`
  gains `direct_row`, and a composed row with `site_id == -1` (only with `environment=True` or `coupling=True`) has
  `inbound_row == outbound_row == -1`.
- `ScatterResponse.evaluate` takes `site_count` instead of `row_count` (it always returned one value per site).
- `FrozenLegTopology` (internal adapter record) carries the discovery `topology`; `prepared`, `components` and the
  `prepare_*` counters are properties that prepare the replay partition on first use.
- Native operator schemas changed for callers of the internal operators: `sensor_weight_jvp/backward` and
  `facet_response_jvp/backward` take new optional arguments; `pulsed_echo_*` take a chip table, a per-segment
  schedule, `frequency_step_hz` and `periodic`; `fmcw_beat_*` take `slot_table`, `segment_table` and `chirp_poly`.
- `SensorWeightGeometry.FROZEN_FIELDS` no longer lists `pattern_frame`, and `SensorWeightPlan` accepts a marked
  table: both are differentiable inputs. `SmplPoseDeformation` accepts a pose or shape that carries a gradient;
  `Coupling.tx`/`rx` keep a complex tensor as given when it carries a derivative.
- `PulsedSpec` refuses an unknown `pulse_kind` with a message pointing at the `coded` kind.

**Environment**

- The `channel` extra requires `witwin-channel>=0.6.0,<0.7` (consumer contract 7); Radar 0.5.0 does not run on
  Channel 0.5. Upgrade with `pip install -U "witwin-radar[channel]"`.
- The `channel` extra installs nothing on Python 3.14, because Channel 0.6 publishes wheels for CPython 3.10-3.13
  only. Radar itself installs on 3.10-3.14; scene-driven simulation needs 3.10-3.13.
- `witwin-radar` requires `torch==2.10.0` instead of `torch>=2.10`: the loader only ever accepted a packaged binary
  on the exact Torch it was built against. Install Torch 2.10.0, or build the native library for your Torch with
  `python scripts/build_radar_cuda_prebuilt.py` from a source checkout.
- `ci/native-binding-manifest.json` is generated (schema 3). After changing a native operator, run
  `python ci/check_native_bindings.py --write`.

### Added

**Targets and scattering**

- `MeshTargets(structure_ids=None, cluster_size=None, max_sites=None, seed=0)`: the faces of Core structures are the
  scatterers. Each facet scatters as its vector physical-optics dyadic (exact flat-facet form factor) published to
  the join as a 2x2 scattering matrix per composed row; facets are extracted natively per frame from the deformed
  world vertices (float64 sites), occluded by the scene through the legs, culled on their back side, and
  differentiable with respect to the vertices in reverse mode. Multi-bounce on the target comes from the scene's
  reflections (`reflections=1` for a dihedral, `2` for a trihedral). `cluster_size` or `max_sites` merges facets
  into fewer sites for large meshes (R-ADR-042).
- A facet's return follows its structure's Core material: the layer-stack TE and TM reflection coefficients at the
  incidence angle (R-ADR-043) and, with `roughness_front`, a coherent roughness loss plus an energy-consistent
  Lambertian diffuse term with a seeded per-facet phase (R-ADR-044).
- `SmplPoseDeformation.mesh_target(scene, structure_id=..., material=..., ...)` returns the scene with the posed
  body and the `MeshTargets` that makes the body scatter from its facets.
- `Fluctuation(swerling, coherent_interval, seed)` on `PointTargets` and `StructureTargets`: Swerling I-IV power
  fluctuation per site, seeded and keyed like the receiver noise (R-ADR-045).
- `RcsTable(axis, rcs=, frequencies=None, phase=0.0)` as a `PointTargets.rcs`: a tabulated `sigma(aspect,
  frequency)`, reduced to the carrier and interpolated per composed row at the bistatic aspect angle,
  differentiable in the axis and the leg directions.
- `ScatteringMatrix(hh, hv, vh, vv, up=...)` for polarimetric point targets (`PointTargets(scattering=...)`), with
  `ScatteringMatrix.sphere()` and `.dihedral()`; `Radar.rx_polarization` for cross-polar channels.
- Per-target `rcs`, `amplitude` and `phase` vectors on `PointTargets` and `StructureTargets` without an `Aspect`.
- `Aspect.axis` may be a function of time, `axis(t) -> (S, 3)`, evaluated at every observation instant.
- Float64 target positions: `PointTargets(positions=...)` and a `trajectory` may return float64; the round-trip
  path length is recomputed from them in float64 (R-ADR-040).
- `FacetScatterResponse`, `TableScatterResponse`, `FluctuatingResponse` and `ScatteringMatrixResponse` in
  `witwin.radar.scattering`.

**Propagation and motion**

- `transmission=True` and `diffraction=True` on `trace`, `simulate` and `stream`: paths through walls and
  edge-diffracted paths (with `ad_mode="none"` and a discovery at every evaluated instant).
- `environment=True` adds site-less transmitter-to-receiver rows for the scene's own returns (a wall facing the radar
  echoes at `2d/c`); `coupling=True` adds the elements' straight-segment coupling. Both are off by default.
- `Paths.component_index()` classifies composed rows as target, environment clutter, direct leakage or
  multi-interaction, and `Paths.select(name)` keeps one class, so `radar.echo(paths.select("environment_clutter"))`
  synthesizes the clutter alone.
- `discovery=` (`"auto"`, `"exhaustive"`, `"rays"`) selects Channel's reflection candidate source; `max_paths=`
  bounds one discovery's rows; `max_round_trips=` bounds one sensor pair's composed rows, refused by name before the
  join is built.
- `wideband=True` for OFDM: each subcarrier's transport is evaluated by Channel (spreading, material and layer
  response across the band).
- `Result.events` and `Paths.events`: per frame, a `PathEvents` record of every path birth and death, detected by a
  native kernel; `Result.birth_lag_bound` / `Paths.birth_lag_bound` bound how late a birth can have been seen.
- Discovery cadences for `Motion.adc()`, `Motion.chirp()` and `Motion.auto()`: `rediscover_every_chirps`,
  `rediscover_every_seconds` and `rediscover_on_displacement`, besides `rediscover_every_frames` (at most one).
- `Motion.adaptive(motion_bandwidth=..., max_acceleration=...)`: declared motion bounds that cap the probe spacing.
- `pose=` on `Radar.simulate` / `trace` / `stream`: the radar's ego-motion `pose(t) -> (rotation, translation)`.
- `Result.observation_schedule` and `SynthesisResult.observation_schedule`: `frame`, `slow_time`, `fast_time` or
  `adaptive`, the schedule the frame's synthesis used.
- `RadarPathBatch.total_delay_residual_s` and the `total_delay_residual_s=` argument of the row-level synthesis
  entries: the float64 round-trip delay as a float32 pair.

**Waveforms**

- Coded pulses, `Pulsed(pulse_kind="coded", chips=..., chip_frequencies=...)`, with `Pulsed.pmcw`, `Pulsed.fsk`,
  `Pulsed.arbitrary` and `Pulsed.stepped_frequency`; code generators `Pulsed.m_sequence(order)` and
  `Pulsed.costas(prime)`; `processing.stepped_frequency_profile`.
- Time-domain OFDM: `Ofdm(modulation="bpsk"|"qpsk"|"qam16"|"qam64", pilot_spacing=..., symbol_seed=...,
  output="cfr"|"time")` synthesizes the received samples of a seeded QAM stream with pilots and a cyclic prefix, so
  intra-symbol motion produces ICI and an echo past the prefix ISI; `processing.ofdm_cfr_estimate` and
  `ofdm_demodulate`.
- FMCW chirp sequences, `Fmcw(sequence=(Chirp(slope=..., start_offset=..., idle=..., adc_start=..., ramp_end=...),
  ...))`; `Fmcw(phase_error=(c2, c3, ...))`, a polynomial ramp phase error; `Fmcw(sampling="real")`, an in-phase-only
  receiver.
- `Fmcw(mimo="ddm"|"bpm", ddm_bands=...)`: Doppler-division and binary-phase transmit multiplexing (TDM stays the
  default); `processing.demultiplex` returns the `[TX, RX, slow, fast]` virtual array.
- `Fmcw.sweep_bandwidth`, `range_resolution`, `max_range` and `max_velocity(reference_frequency_hz=, num_tx=)`.

**Receive chain**

- `Pattern.gain` (absolute element gain, dBi), `Radar.rx_pattern`, complex field patterns
  (`Pattern.field_separable`, `Pattern.field_table`) and `Pattern.from_antenna` (samples a `witwin.core`
  `AntennaPattern`, such as Channel's presets, into a field table).
- `Radar.compression = Compression(iip3, p1db, am_pm)`: receiver gain compression after the LNA, with AM/PM.
- Shaped oscillator phase noise: `Noise.phase_flicker_density` / `phase_flicker_offset` (1/f^3) and
  `Noise.phase_floor`, alongside the 1/f^2 region.
- `Radar.coupling = Coupling(tx=..., rx=...)`: mutual coupling matrices on the transmit and receive ports.
- `Leakage` accepts per-pair `[num_tx][num_rx]` tables for `isolation`, `delay` and a new `phase`, and works for
  OFDM and pulsed radars.
- `Radar.dc_offset` and `Noise.flicker_corner` (baseband 1/f noise), for FMCW and pulsed.
- `Agc(time_constant=...)`: a time-dynamic AGC with one gain per slow-time block.
- `Radar(iq=...)` alone builds a receive chain.

**Multi-radar**

- `witwin.radar.Network(radars, links=(), clock_offsets=(), if_bandwidths=(), carrier_offsets=(), emitters=())`:
  several radars in one scene and one call, with `simulate`, `stream`, `trace` and `echo` publishing one `Result` per
  radar; the scene is compiled and each leg family discovered once for all radars; `poses=` gives each radar its
  ego-motion (R-ADR-051).
- Bistatic links: `links=((i, j), ...)` makes radar `j` hear radar `i` through the targets, the direct segment and
  the environment, synthesized with the receiver's own dechirp reference, so slope mismatch, start-time and
  start-frequency offsets appear in the beat and the receiver's IF band gates it. FMCW mutual interference (ghost
  targets, broadband bursts) follows from the same rows.
- Pulsed and OFDM receivers on links (ISI against the prefix, ICI from `carrier_offsets`), and `witwin.radar.Emitter`,
  a non-radar transmitter heard by every radar of a `Network(emitters=...)`.
- Native `fmcw_cross_*`, `pulsed_cross_*` and `ofdm_cross_*` operators: every link is differentiable in both modes.

**Gradients**

- Antenna pattern leaves: the cut or table values of `Pattern.separable`, `table`, `field_separable` and
  `field_table` and every constructor's `gain` may be tensors.
- Radar pose leaves: a tensor `Radar.position`, `look_at` or `up` (and a `pose=` trajectory's translation) keeps its
  tape through every element position and the pattern frame (`ad_mode="vjp"` or `"jvp"`).
- `Radar.power` and `Radar.lna_gain` accept a 0-dim tensor leaf; complex tensor `Coupling` matrices are leaves.
- Mesh-target material leaves: a Core `PhysicalMaterial` whose `eps_r`, `sigma_e`, `thickness_m`, `mu_r` or
  `roughness_front.rms_height_m` is a tensor reaches the facet response.
- SMPL leaves: the pose, shape and body position/rotation of an `SMPLBody` behind
  `SmplPoseDeformation.mesh_target` reach the traced rows in reverse mode.
- A `LEAVES` mapping on `Radar`, `PointTargets`, `Pattern`, `Coupling` and `SMPLBody` naming each differentiable
  field and its AD modes.

**Processing, imaging and calibration**

- `go_cfar`, `so_cfar` and `cfar_false_alarm_probability`; `group_peaks`; `cluster_points` / `PointClusters`
  (DBSCAN, optionally with radial velocity); `scan_aoa` (route `"scan"`) and `ArrayLattice`; `range_angle_map` /
  `RangeAngleMap` / `azimuth_directions`; `KalmanTracker` / `TrackState` and `greedy_assignment`;
  `remove_static_clutter(profile, method="mean" | "mti2" | "mti3")`.
- `PointCloud.frame` and `PointCloud.to_world(radar)`.
- `ProcessingAxes.from_radar(radar)` and `ProcessingCube.from_tensor(cube, radar)` for recorded or external cubes.
- Zero padding: `range_profile(..., range_bins=)`, `range_doppler_map(..., doppler_bins=)`,
  `ProcessingAxes.with_bin_counts`.
- `SlowTimeSignal(wavelength_m=)` adds a velocity axis to the micro-Doppler product.
- `backproject(profile, aperture, pixels_m)`: time-domain back-projection onto any pixel grid, returning a
  `SarImage`, for FMCW, OFDM and pulsed phase laws, differentiable in the samples; `Aperture` and
  `Aperture.from_radar(radar, time_s=..., velocity_mps=... | pose=...)`; `isar_aperture`; `element_responses`.
- Calibration: `estimate_array_calibration` / `ArrayCalibration`, `estimate_range_calibration` /
  `RangeCalibration`, `subtract_background` and `average_background`.
- `Frame.virtual_cube()`, `Frame.range_angle(azimuth_rad)`, `Frame.image(aperture, pixels_m, range_bins=)` and
  `Frame.aperture(radar, velocity_mps=, pose=)`; `Frame.points()` demultiplexes a DDM or BPM cube itself.

**Results, tools and introspection**

- `witwin.radar.capabilities()`: a versioned JSON record of what this build supports, read from the owning modules.
- `Result.semantics` / `Paths.semantics`: completeness in Channel's result-semantics keys (`estimator`, `combine`,
  `complete`) plus `exhaustive`; `witwin.radar.adaptive.AdaptiveDiagnostics`, the typed record of each
  `Result.adaptive_diagnostics` entry (new key `complete`).
- `Result.save(path)` / `Result.load(path, device=)`: a versioned directory of JSON metadata and plain tensors read
  with `weights_only=True`; nothing is pickled.
- Compact `repr` for `Result`, `Frame`, `Paths`, `SynthesisResult` and every processing product.
- `witwin.radar.__version__`; `witwin.radar.deployment.package_version` and `runtime_failures`;
  `Paths.row_identity` and `Paths.observation_times`.
- `witwin.radar.labels`: `ground_truth(radar, scene, targets, times=)` / `GroundTruth.at(t)` (per-target range,
  radial velocity, azimuth, elevation, position, monostatic RCS, ID) and `path_summary(paths)`.
- `witwin.radar.dataset.write_dataset(root, radar, [Episode(scene, targets, times, seed=...)], ...)` and
  `load_episode`.
- `Radar.to_dict()` / `Radar.to_json(path)` and `witwin.radar.config.config_schema()`; field table in
  `docs/configuration.md`.
- `witwin.radar.config.read_ti_cfg(path)` reads a TI mmWave SDK `.cfg`, `TiConfig.to_radar(tx=, rx=)` builds the
  radar, and `preset("awr1843boost" | "iwr6843isk")` builds a board.
- `witwin.radar.rawdata.write_dca1000` / `read_dca1000`: raw ADC in the DCA1000 capture layout.
- `witwin-radar-doctor` / `python -m witwin.radar doctor`: runtime, native-library and Channel checks plus a GPU
  simulate smoke test, with a verdict and a fix per problem.
- The native loader refuses, before loading, a binary whose CUDA architectures cannot run on the current device,
  naming the device, its `sm_XY` and the `TORCH_CUDA_ARCH_LIST` that fixes it.
- `pytest --benchmark` runs the absolute wall-clock budgets, which are skipped by default.
- An analytic oracle and survival suite under `tests/oracle/` (R-ADR-032).
- User guides under `docs/user/`, a migration guide, `CONTRIBUTING.md` and runnable examples
  (`examples/moving_target.py`, `examples/walking_human.py`); `tools/run_doc_snippets.py` runs every Python block of
  the documentation.

### Fixed

- **Carrier-phase staircase of moving targets.** The join recomputes the round-trip length in float64 from the
  endpoints and Channel's interaction points and replaces Channel's float32 phase (R-ADR-040). Against a float64
  trajectory at 77 GHz the integrated phase error is -130 / -111 / -100 dBc and the spur floor -133 / -116 / -107 dBc
  at 2 / 30 / 100 m (was -73 / -55 / -43 and -76 / -57 / -49 dBc); 1 um of motion at 30 m is resolved. Every
  waveform kernel reads the float64 delay pair, so a 4 GHz sweep at 100 m no longer carries the float32 delay's
  7e-4 rad of bandwidth phase; site-less rows and every wideband column carry the float64 phase too.
- **Surface RCS converges.** The cross section of a surface converges to its analytic value as the mesh is refined:
  a 5 cm PEC plate reads -0.01 dB of `4 pi A^2 / lambda^2`, a sphere within 0.3 dB of `pi a^2`, a dihedral within
  0.5 dB and a triangular trihedral within 0.7 dB, where `N` isotropic sites read up to +36 dB.
- **Isotropic elements are isotropic** off the polarization plane (see Breaking changes); the element factor
  follows a radar pose.
- **Pulsed SNR** was optimistic by `f_s / (B T_p)` (+54 dB at 20 MHz, 10 us, 50 MS/s); all three waveforms match the
  radar equation through `k T_sys f_s` and the coherent processing gain within 0.5 dB.
- **CFAR.** CA-CFAR on `Frame.points` produced no false alarms at any `pfa` (a magnitude was thresholded with a
  power-law constant); every detector measures within a few percent of its `pfa` at 1e-2, 1e-3 and 1e-4, edges
  included. CFAR padding wraps along Doppler and truncates (with re-solved constants) along range.
- **Angles.** Estimation no longer assumes a half-wavelength TI layout: 1x8, 2x4 and one-wavelength arrays give
  correct angles on the `fft2` and `scan` routes.
- A coded-pulse replica (`processing.pulse_samples`, used by the matched filter) sampled at a multiple of the chip
  rate put about a fifth of its samples on the previous chip, because `m T_s / T_c` rounded just below an integer in
  float64; the range profile of a PMCW, FSK or arbitrary pulse carried sidelobes next to its peak.
- All frames of a `simulate`/`stream`/`echo` call shared one thermal noise realisation.
- A zero ADC input read `+/- step/2` on I and Q.
- An invalid oscillator query raised a device-side assert that left the CUDA context unusable; it raises
  `ValueError` on the host.
- A target walking behind an occluder no longer stops the run: a declared site with no path at a discovery
  contributes exactly zero (an unknown site ID is still refused).
- Adaptive sampling checks every accepted interval at one jittered instant off its probe grid (motion periodic at the
  grid's step was accepted with about 87 % IQ error), compares delays in float64 (the same motion takes the same
  number of probes at 5 m and 100 m), treats a specular point crossing the edge of two coplanar triangles as one
  continuous reflection in a static world, and includes a declared `Leakage` (it was dropped).
- A scatter site on (or within a fraction of a millimetre of) an array element is refused by name instead of
  returning a finite, unbounded echo.
- `path_set_complete` no longer claims completeness when Channel's reflection discovery fell back to rays.
- `PointTargets` vectors and an `Aspect`'s axes, strengths and phases declared against caller IDs that are not
  ascending are permuted into the join's site order (they were applied in declaration order).
- `AspectScatterResponse` works on reflected outbound rows.
- `Radar(power=tensor)`, `Radar(lna_gain=tensor)`, `Pattern.dipole(gain=tensor)`/`Pattern.isotropic(gain=tensor)`
  and a mesh target's Core material with tensor fields silently converted the tensor to a host float and severed the
  gradient; each is now a leaf.
- `Radar(iq=...)` without another stage was silently ignored.
- `python ci/run_ci_tier.py quick` passes on Linux; a checkout made with `core.autocrlf=true` no longer refuses every
  prebuilt binary (`.gitattributes`); the Linux native library links with `--as-needed`; `pytest tests` collects and
  runs without a CUDA device.

### Performance

Counts are pinned by budget tests (`tests/test_session_budget.py` and the per-family budget tests); timings below are
from a contended desktop GPU.

- The default for a moving FMCW scene evaluates a handful of probes per frame instead of every ADC instant: 6 of 2048
  observations per frame on a 48-site articulated body, 3 to 14 of 128 on the Doppler-precision fixtures.
- An `ad_mode="none"` run reads each rediscovered instant's rows straight from the discovery: one Channel call per
  leg per rediscovery instead of three, and a static world never prepares or replays (bit-identical).
- A world that does not move evaluates its paths and synthesizes its waveform once per session; later frames run only
  the receive chain. `simulate` writes frames into one preallocated cube when no gradient is taken.
- The two-way join is built with vectorized host code and one device copy each way, and a rediscovery that finds
  the same topology reuses the previous join and antenna-pattern tables.
- Chirp- and ADC-refreshed FMCW sessions replay topology-identical instants of a frame together and synthesize each
  group in one launch over only the active transmitter's sensor pairs; `Motion.batch_observations` bounds these
  groups for every kind.
- Slot-batched replays (point, aspect, tabulated and facet responses, directional patterns) are composed by the
  native join striding one slot's tables across the slots instead of copying them per slot (bit-identical).
- Site-less rows are composed by the same native join launch as the site rows; fields, the response and the composed
  value cross the native boundary as real views (no split and recombination launches).
- A still mesh target's facets are placed once per session and reused.
- The isotropic 0 dBi default skips the antenna-pattern launch; each new receiver stage is one native operator call
  per frame.
- `os_cfar` and `cluster_points` on CUDA tensors run native kernels (one and four launches, no host round trip);
  back-projection is one native launch per image.

### Changed

- Reflection rows follow Channel 0.6's path-validity rules: a reflection whose specular point lies past the edge of a
  face by more than a few float roundings is not reported. A scatter site on or near a wall keeps that wall's
  reflection, and discovery and replay agree near a wall.
- A scene edited in place after it was compiled is refused by Channel at the next replay ("the compiled scene is
  stale"); Radar's motion-event tick still detects the edit and recompiles when a motion-event period is declared.
- Channel 0.6 picks its ray-tracing backend when it compiles a scene and uses a pure-CUDA tracer where OptiX is
  unavailable; `witwin-channel-doctor` reports which one a machine gets.
- `Motion.auto()` in a network resolves to `Motion.adc()` when anything moves; adaptive motion and
  `rediscover_every_chirps` are refused for a network, as are the link configurations R-ADR-051 lists.
- Refused with a named reason, before any work: a pose leaf with an `"up"`/`"right"` polarization, a polarization
  leaf, a pose trajectory's rotation leaf, a pose leaf under an `ad_mode` that does not carry it, a roughness rms
  leaf at exactly zero.
- The default radar pose (`look_at=(0, 0, -1)`, `up=(0, 1, 0)`, a Y-up world) is unchanged and documented: in a
  Channel/Core Z-up scene it looks at the floor, so state the pose there.
- `Result.last_snapshot` and `last_compiled_scene` are typed (`SceneSnapshot | None`, `CompiledWorld | None`).
- The adaptive controller compares packed topology identities, and the session's slot-batched table cache is bounded.
- Finished plans, phase records and superseded measurements moved to `docs/dev/archive/`; `PERFORMANCE.md` keeps
  the current conclusions. Comments and docstrings across the package state invariants, units and refusals, and
  `mypy` checks the whole package.

### Dependencies

- `witwin-channel>=0.6.0,<0.7` through the `channel` extra, on CPython 3.10-3.13 only.
- `torch==2.10.0` (CUDA 12.8 build for the release wheels).
- `witwin>=0.4.0,<0.5` (unchanged).
- Python 3.10-3.14 for Radar itself (unchanged).

### Known limitations

- **Cross-polar element patterns.** Element patterns are co-polar: a Jones (cross-polar) element pattern is not
  modelled, because Channel's fixed-topology replay does not publish a polarimetric (Jones) transport for reflection
  rows. Point and facet targets do carry 2x2 scattering matrices.
- **Reflection-row phase precision.** Line-of-sight rows carry a float64 path length; a reflection row's interaction
  point comes from Channel in float32, leaving a residual of `(cos theta_i + cos theta_r) ulp32(x) / 2` (about 0.5 um,
  0.8 mrad of carrier phase, for a wall 4 m away). Transmission rows are float64 to second order, diffraction rows to
  first order across the edge.
- **Finely meshed tilted faces.** With Channel 0.6.0, reflection discovery can return no rows off a finely meshed
  face that is not axis-aligned, while the result still reports `complete=True` (`"auto"` and `"exhaustive"` alike):
  a 5 cm PEC dihedral at 10 m with its faces at 45 degrees loses its double bounce (-43 dB) at 52 or more cells per
  edge (10,816 faces and up), where 50 per edge (10,000 faces) works. A flat plate, upright or tilted 20 degrees, keeps
  its reflection at 18,432 faces. Use the coarsest mesh that resolves the shape; `MeshTargets(max_sites=...)` does
  not help, because the scene's faces are what reflect. Pinned as a strict expected failure in
  `tests/oracle/test_rcs_convergence.py`.
- **FMCW wideband.** `wideband=True` is for OFDM only; it is refused for FMCW, pulsed and time-domain OFDM radars.
- **Imaging.** Back-projection only: no frequency-domain (range-Doppler or omega-k) stripmap algorithm. Image an
  FMCW profile from `Fmcw(output="beat")` with zero padding; aperture and pixel positions are not differentiable.
- **Saved results only.** `Result.save` / `Result.load` exist; `Paths` has no saved form (it holds the session).
- **Gradients not carried.** The receiver compression parameters (`Compression.iip3`, `p1db`, `am_pm`) are not
  leaves; polarizations are not leaves.
- **Multi-radar.** A network resolves a moving scene to `Motion.adc()` (no adaptive sampling); DDM/BPM and
  multi-profile links, emitters with directional elements, phase noise on a receiving radar and wideband are refused.
- **Budget pins.** The profiler-derived kernel and copy counts in the session budget tests keep a 0.5 % margin,
  because the profiler can drop records on a contended device; every other counter is exact.
- **Untested without external files.** The SMPL route is tested on a synthetic rig only, because SMPL model files
  are not shipped. The DCA1000 capture layouts and the TI board presets were written from TI's documentation and
  have not been checked against hardware or a real capture.
- **Requests to Channel** that would lift limitations above: a polarimetric (Jones) fixed-topology transport for
  reflection rows; float64 interaction points (or float64 endpoints) for reflection rows; reflection discovery off
  finely meshed tilted faces; the structure ID of each interaction in the consumer topology; several
  source/sink groups (or a pair mask) per `PropagationRequest` and slot-batched discovery.

[Unreleased]: https://github.com/witwin-ai/witwin-radar/compare/witwin-radar-v0.5.0...HEAD
[0.5.0]: https://github.com/witwin-ai/witwin-radar/releases/tag/witwin-radar-v0.5.0
