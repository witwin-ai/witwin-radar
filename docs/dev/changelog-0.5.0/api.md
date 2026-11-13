## Breaking changes

- Public records validate every field at construction and refuse a bad one by `Record.field` name: a numeric string, bool, complex, NaN or infinite number where a real number is expected; an unknown vocabulary word (`Fmcw.mimo`, `Fmcw.output`, `Fmcw.sampling`, `Ofdm.modulation`, `Ofdm.output`, `Pulsed.pulse_kind`, `Motion.world`); positions that are not `(S, 3)`; a negative host `rcs`; repeated or miscounted `ids`; an `Aspect`/`RcsTable` axis without one row per target. Migration: pass SI numbers and the documented words; these inputs failed later (or produced NaN cubes) before.
- An FMCW ADC window that ends after the chirp period (`adc_start + (samples_per_chirp - 1) / sample_rate >= idle + ramp_end`) is refused at construction; it sampled the next chirp. Migration: lengthen `ramp_end` or shorten the window.
- `Radar(...)` builds its synthesis spec at construction, so array-dependent waveform rules (BPM needs a power-of-two transmitter count, DDM bands at least the transmitter count) refuse the radar instead of its first run.
- `Radar.trace`, `simulate`, `stream` and `echo` refuse a radar that is not on a CUDA device, naming the device, before any work (it used to fail inside Channel).
- `point_cloud`, `beam_cube` and `range_angle_map` refuse a DDM or BPM product whose transmitters are not separated, naming `processing.demultiplex`. Migration: demultiplex the `ProcessingCube` first, or use the `Frame` facades, which do.

## Added

- `witwin.radar.capabilities()`: a versioned JSON record of what this build supports (waveform kinds and options, MIMO schemes, motion kinds and cadences, propagation components, target records, AD leaves per record with their modes, processing vocabularies, the native operators of the installed prebuilt, Channel's consumer contract), read from the owning modules.
- `Result.semantics` / `Paths.semantics`: the run's completeness in Channel's result-semantics keys (`estimator`, `combine`, `complete`) plus `exhaustive`.
- `witwin.radar.adaptive.AdaptiveDiagnostics`, the typed record of each `Result.adaptive_diagnostics` entry; every key is always present, and the new `complete` key is the frame's `path_set_complete`.
- `Frame.virtual_cube()`, `Frame.range_angle(azimuth_rad)`, `Frame.image(aperture, pixels_m, range_bins=)` and `Frame.aperture(radar, velocity_mps=, pose=)`; `Frame.points()` now demultiplexes a DDM or BPM cube itself.
- `Fmcw.sweep_bandwidth`, `Fmcw.range_resolution`, `Fmcw.max_range` and `Fmcw.max_velocity(carrier=, num_tx=)`.
- A `LEAVES` mapping on `Radar`, `PointTargets`, `Pattern`, `Coupling` and `SMPLBody` naming each differentiable field and its AD modes.

## Changed

- `Result.last_snapshot` and `last_compiled_scene` are typed (`SceneSnapshot | None`, `CompiledWorld | None`) instead of `object`.
