# Radar Feature List

What witwin-radar 0.5.0 supports, by area. At run time `witwin.radar.capabilities()` returns the same vocabularies
as a versioned JSON record read from the owning modules. How to use each feature: [`docs/user/`](docs/user/README.md).
Known limitations: the `[0.5.0]` section of [`CHANGELOG.md`](CHANGELOG.md).

## Configuring a radar

- `Radar` is one flat immutable record in SI units. The reference frequency (`reference_frequency_hz`), the waveform,
  the two antenna layouts and the transmit power are required; patterns, polarizations, the receive stages, the port
  impedance, the seed, the pose and the device have defaults. A sub-record appears only where a field has variants
  (`Fmcw`/`Ofdm`/`Pulsed`, `Pattern`, `Motion`) or where several numbers are coupled (`Noise`, `Agc`, `Adc`,
  `Compression`, `Coupling`, `Leakage`, `Iq`). A field naming a quantity Channel also publishes, in Channel's unit,
  takes Channel's name (`reference_frequency_hz`, `ad_mode`, `power_w`); every other public field carries no unit
  suffix and states its unit on its docstring row (R-ADR-049). Exceptions to SI: `power` in dBm, gains in dB or dBi.
- Every public record validates its fields at construction and refuses a bad one by `Record.field` name; array-
  dependent waveform rules refuse the `Radar` itself, and the verbs refuse a radar that is not on a CUDA device.
- `antenna_unit` reads `tx` and `rx` in metres or in half wavelengths, converted once against the carrier.
- The default pose is `position=(0, 0, 0)`, `look_at=(0, 0, -1)`, `up=(0, 1, 0)`: a Y-up world. Channel and Core
  scenes are Z-up, so a Z-up scene states the pose. `polarization` defaults to the pose's own up axis;
  `rx_polarization` defaults to it.
- `Radar.from_dict` / `from_json` read the flat configuration format (TI-style FMCW keys in their vendor units, or a
  `waveform` block in SI) and `to_dict` / `to_json` write it, so `from_dict(to_dict())` is the same radar;
  `witwin.radar.config.config_schema()` is its JSON Schema. `config.read_ti_cfg` reads a TI mmWave SDK `.cfg` and
  `config.preset(...)` builds a board (written from TI documentation, not validated on hardware).
- `Radar.replace(...)` and `Radar.to(device)` return a new radar; a radar holds no run state.

## Simulation and propagation

- Four verbs, two halves and their fusion: `Radar.trace` (world half: sampling, Channel discovery and replay,
  round-trip composition, scattering, antenna weighting) returns `Paths`; `Radar.echo(paths)` (instrument half:
  synthesis, receive chain, output domain, processing axes) returns `Result`; `Radar.simulate` fuses them and stacks
  every frame; `Radar.stream` yields one frame at a time. `echo(trace(...))` is bit-identical to `simulate(...)`
  across every motion kind and output domain, with and without a receive chain and oscillator phase noise.
- One `Paths` can be echoed by several receive chains, seeds or FMCW output domains; a radar the rows do not describe
  is refused by name. `Paths` retains every evaluated observation's rows; the fused verbs retain one replay group (at
  most `Motion.batch_observations` observations) at a time.
- Targets: `PointTargets` (world positions, float64 accepted; per-target or shared `rcs`, `amplitude`, `phase`;
  `trajectory`; stable `ids`; an `Aspect` lobe, possibly time-varying; a polarimetric `ScatteringMatrix`; an
  `RcsTable` of `sigma(aspect, frequency)`; a Swerling I-IV `Fluctuation`), `StructureTargets` (one scatterer at each
  Core-published anchor of a moving structure) and `MeshTargets` (every face of Core structures is a vector
  physical-optics facet of its structure's material: layer-stack TE/TM reflection, rough-surface coherent loss and
  Lambertian diffuse term, occlusion through the legs, back-face culling, multi-bounce from the scene's reflections,
  an optional site budget; R-ADR-042 to R-ADR-045). `SmplPoseDeformation.mesh_target` makes an SMPL body a mesh
  target (needs the SMPL model files, which are not shipped).
- Propagation components: `los`, `reflections` (depth), `transmission` (through walls) and `diffraction` (edges); the
  last two exist only at discovery, so they need `ad_mode="none"` and a discovery at every evaluated instant.
  `environment=True` adds site-less transmitter-to-receiver rows (wall echoes at `2d/c`), `coupling=True` the
  elements' straight-segment coupling. `Paths.component_index()` / `Paths.select(name)` split the rows into target,
  environment clutter, direct leakage and multi-interaction. `discovery=` picks Channel's reflection candidate
  source; `max_paths=` and `max_round_trips=` bound rows and are refused by name on overflow.
- Element patterns and polarization are Radar's (R-ADR-041): Channel's unprojected `complex3_transport` is replayed,
  so an isotropic element is isotropic in every direction.
- The round-trip path length is recomputed in float64 from the endpoints and Channel's interaction points (R-ADR-040),
  and every waveform kernel reads the float64 delay as a float32 pair: moving-target carrier phase is free of float32
  staircases on line-of-sight rows.
- `wideband=True` makes OFDM consume Channel's per-subcarrier transport (R-ADR-035); FMCW and pulsed stay narrowband
  at the reference frequency with an exact delay phase.
- Completeness is published, never assumed: `path_set_complete`, `discovery_complete`, `motion_sampling_exhaustive`,
  `transmission_dropped_rows`, and `Result.semantics` / `Paths.semantics` in Channel's vocabulary.
- `Network(radars, links=..., clock_offsets=..., if_bandwidths=..., carrier_offsets=..., emitters=...)` simulates
  several radars in one scene and one call, one `Result` per radar, with one compile and one discovery per epoch for
  all radars. Links make radar `j` hear radar `i` (bistatic returns, the direct path, the environment) through native
  cross-radar families for FMCW, pulsed and OFDM, so interference (ghosts, bursts, ISI, ICI) follows; an `Emitter` is
  a non-radar transmitter (R-ADR-051).

## Motion

- `Motion` names the in-frame sampling: `auto` (the default), `static`, `chirp` (stop-and-hop, one observation per
  chirp), `adc` (every ADC sample: the exhaustive reference) and `adaptive`.
- `Motion.auto()` resolves to one observation per frame for a still world, to `adaptive` for a moving FMCW scene on
  one uniform TDM ramp, and to `adc` for any other moving scene (OFDM, pulsed, a chirp sequence, a ramp phase error,
  DDM or BPM), when a discovery cadence is declared, and in a moving `Network`. A receiver with oscillator phase noise
  counts as moving, because its phase difference needs observation instants.
- `Motion.adaptive(...)` interpolates the delay, phase and amplitude of each path between probes under per-path phase
  and amplitude tolerances, tests every accepted interval at its midpoints and at one jittered instant, refines on a
  topology change, caps its probe spacing (`max_interval`, `motion_bandwidth`, `max_acceleration`), compares delays
  in float64, and publishes `AdaptiveDiagnostics` per frame. Brief events between probes are not certified absent;
  `path_set_complete` says so.
- Discovery cadences: `rediscover_every_frames`, `rediscover_every_chirps`, `rediscover_every_seconds`,
  `rediscover_on_displacement` (at most one). A longer cadence sets `path_set_complete=False`, and
  `Result.birth_lag_bound` bounds how late a path birth can have been seen.
- `Result.events` / `Paths.events`: every path birth and death per frame, detected natively.
- `pose=` gives the radar an ego-motion `pose(t) -> (rotation, translation)`; elements, polarizations and the pattern
  frame follow it at every observation.
- `Result.observation_schedule` and `motion_sampling` state the schedule that ran.

## Waveform synthesis

- FMCW: native Dirichlet range spectrum by default (`output="spectrum"`) for a delay that holds the whole chirp, the
  beat domain (`output="beat"`) for a walking delay and every ADC-refreshed scene, with native VJP/JVP. TDM by default;
  `mimo="ddm"` and `mimo="bpm"` fire every transmitter every chirp and `processing.demultiplex` returns the virtual
  array (R-ADR-048). Chirp sequences (`Chirp` profiles), a polynomial ramp phase error and a real-only ADC
  (`sampling="real"`) are native and differentiable (R-ADR-047). Derived quantities: `sweep_bandwidth`,
  `range_resolution`, `max_range`, `max_velocity(...)`.
- Pulsed: rectangular, LFM and coded pulses (PMCW with `m_sequence`, FSK with `costas`, arbitrary envelopes, stepped
  frequency). Samples are physical amplitudes; under `Motion.adc()` a pulse train is observed at every gate sample
  (R-ADR-046).
- OFDM: the closed-form CFR by default; with `modulation=` a seeded QAM stream with pilots and a cyclic prefix is
  synthesized in time, so intra-symbol motion produces ICI and a long echo ISI (`output="cfr"` or `"time"`).
- Output-domain metadata travels with every result, so processing never guesses whether a range transform is needed.

## Receive chain

- In signal order: transmit and receive `Coupling` matrices; element `Pattern`s (isotropic, dipole, separable or
  tabulated power or complex field, `Pattern.from_antenna` for Core antenna patterns, `gain` in dBi, separate
  `rx_pattern`); transmit `power`; `Leakage` (per-pair tables of isolation, delay, phase); `lna_gain`; `Compression`
  (IIP3, P1dB, AM/PM); `Noise` (thermal at the complex sample rate, a fresh realisation per frame keyed by the frame
  instant; oscillator phase noise with 1/f^3, 1/f^2 and floor regions over absolute time; baseband flicker);
  `dc_offset`; `Iq` imbalance; `Agc` (static or time-dynamic); `Adc` (two's-complement mid-tread quantiser with
  clipping counts). Each stage is one native operator call per frame; the isotropic 0 dBi default skips the pattern
  launch (R-ADR-029 to R-ADR-031).
- Every stage applies to FMCW and pulsed; OFDM takes the stages its CFR model supports and refuses the rest by name.

## Differentiation

- `ad_mode="none" | "vjp" | "jvp"` on the world-facing verbs. Leaves (R-ADR-052; `LEAVES` on each record and
  `capabilities()`): target positions, trajectories, `rcs`/`amplitude`/`phase`, aspect axes, table values; mesh
  vertices (reverse and forward mode); SMPL pose, shape and transform (reverse mode; a forward tangent is refused because `velocity_at` opens its own forward level); a mesh target's Core material (`eps_r`, `sigma_e`, `thickness_m`, `mu_r`, roughness rms); antenna
  pattern values and `Pattern.gain`; the transmit power and the LNA gain; coupling matrices; the radar pose
  (`position`, `look_at`, `up`, a pose trajectory's translation); every network link.
- A derivative that is not defined (a discrete choice, a quantiser, a polarization) is refused by name before any
  work. The authority for every cell is `docs/dev/radar-ad-capability-matrix.md`.

## Signal processing

The `witwin.radar.processing` facade exports typed products and algorithms; each consumes named-axis metadata.

- Range profiles (zero padding), Range-Doppler and range-angle maps, static clutter removal (mean, MTI), matched
  filtering, `ProcessingAxes.from_radar` and `ProcessingCube.from_tensor` for recorded cubes.
- Micro-Doppler: `SlowTimeSignal` with explicit timestamps, `microdoppler_spectrogram` returning a typed record.
- Detection: square-law CA/GO/SO/OS-CFAR at the requested false-alarm rate (`cfar_false_alarm_probability`), Doppler
  wrap and range truncation at the edges, `group_peaks`.
- Angles: steering-vector scan for any layout (`scan_aoa`), lattice FFT (`fft2_aoa`), phase comparison, conventional,
  MVDR and MUSIC beamforming; `ArrayGeometry` in the processing frame `(right, boresight, up)`.
- Point clouds (`point_cloud`, `PointCloud.to_world(radar)`), DBSCAN clustering (`cluster_points`), constant-velocity
  multi-target tracking (`KalmanTracker`, `greedy_assignment`).
- SAR and ISAR by time-domain back-projection (`Aperture`, `isar_aperture`, `backproject`, `element_responses`;
  one native launch per image, differentiable in the samples; R-ADR-054).
- Calibration: per-channel complex gain from a reflector, round-trip delay bias, background subtraction.
- `os_cfar`, `cluster_points` and `backproject` run native kernels on CUDA tensors; the rest is Torch library calls.
- `Result.frame(i)` returns a `Frame` whose methods (`processing_cube`, `virtual_cube`, `range_profile`,
  `range_doppler`, `range_angle`, `array`, `points`, `image`, `aperture`) are facades over this package. Their
  windows match the stage they name (rectangular), except `points()`, which tapers Doppler with Hann so a detector
  does not report sidelobes.

## Results, data and tools

- `Result` carries the cube `[frame, TX, RX, slow, fast]`, `axis_names`, `axes`, `times`, `sample_times`, the
  completeness and schedule fields, and the last frame's typed diagnostics. `Result.save` / `Result.load` use a
  versioned directory of JSON and plain tensors (nothing pickled); `Paths` has no saved form.
- `witwin.radar.labels` (per-frame, per-target ground truth and per-row path summaries) and
  `witwin.radar.dataset` (a streaming writer of cubes, labels and metadata per episode).
- `witwin.radar.rawdata`: raw ADC in the DCA1000 capture layout (written from TI documentation, not validated on
  hardware).
- `witwin-radar-doctor` / `python -m witwin.radar doctor`; compact `repr` of every result and product;
  `witwin.radar.__version__`.

## Public API and architecture governance

- Public exports are declared in `ci/public-api-manifest.json` and signature-pinned by `ci/public-api-snapshot.json`.
- The package root exports the whole happy path: `Adc`, `Agc`, `Aspect`, `Chirp`, `Compression`, `Coupling`, `Emitter`, `Fluctuation`, `Fmcw`, `Frame`, `Iq`, `Leakage`, `MeshTargets`, `Motion`, `Network`, `Noise`, `Ofdm`, `Paths`, `Pattern`, `PointTargets`, `Pulsed`, `Radar`, `RcsTable`, `Result`, `ScatteringMatrix`, `StructureTargets`, `capabilities` and `processing`. Advanced records stay importable from their owner modules and are not re-exported here.
- Concept owners and permitted module topology are declared in `ci/architecture-manifest.json`; the native binding
  manifest is generated from the sources.
- Compatibility aliases, fallback imports, deprecation shims and legacy package mirrors are forbidden.
- `mypy` checks the whole package; living documentation is checked against deleted paths and retired names.

## Packaging and CI policy

- CPython 3.10-3.14 on Linux x86_64 and Windows x86_64; the `channel` extra (Channel 0.6) on 3.10-3.13.
- Linux release artifacts target `manylinux_2_28_x86_64`.
- Release wheels contain exactly one `_radar_native` binary and its build identity sidecars, built for
  `torch==2.10.0`; the loader enforces exact Torch/CUDA/ABI identity and refuses a binary whose architectures cannot
  run on the device. A loader refusal is a release failure; there is no JIT fallback.
- Required-Channel quality and GPU workflows install the Channel extra, consume its build fingerprint, and permit
  zero skips caused by an absent Channel runtime.

Commands in CI are evidence only after the corresponding job actually runs. The checked-in workflow is policy, not
proof of a successful remote execution.
