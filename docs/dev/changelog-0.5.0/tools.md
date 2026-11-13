## Breaking changes

- `Radar.from_dict` accepts every `Radar` field and a `waveform` block of SI waveform fields. A mapping that declares the waveform twice (a `waveform` key next to the TI-style FMCW keys) is refused with "the waveform is declared twice" instead of "unsupported keys: waveform". Migration: none for TI-style files; a non-FMCW radar can now be written as `{"waveform": {"kind": "ofdm", ...}}`.
- `Result.last_propagation` and `Result.last_radar_paths` are typed `Optional`: a result read back with `Result.load` carries `None` there (and in `last_snapshot`, `last_compiled_scene`) and empty `events`.

## Added

- `Result.save(path)` / `Result.load(path, device=)`: a versioned directory format (`metadata.json`, plain-tensor `.pt` files read with `weights_only=True`, one file per frame); nothing is pickled. See `docs/configuration.md`.
- Compact `repr` for `Result`, `Frame`, `Paths`, `SynthesisResult` and every processing product (`RangeDopplerMap`, `PointCloud`, `ProcessingAxes`, ...): tensors print as dtype, shape and device.
- `witwin.radar.__version__`.
- `witwin.radar.labels`: `ground_truth(radar, scene, targets, times=)` / `GroundTruth.at(t)` give per-frame, per-target range, radial velocity, azimuth, elevation, position, monostatic RCS and target ID (mesh targets: per-structure centroid, area, range extent); `path_summary(paths)` lists each composed row's component, depth, delay, Doppler and amplitude.
- `witwin.radar.dataset.write_dataset(root, radar, [Episode(scene, targets, times, seed=...)], ...)` streams cubes, labels and metadata to disk frame by frame; `load_episode` reads one back.
- `Radar.to_dict()` / `Radar.to_json(path)` (round-trip `from_dict`/`from_json` exactly) and `witwin.radar.config.config_schema()` (JSON Schema); field table in `docs/configuration.md`.
- `witwin.radar.config.read_ti_cfg(path)` reads a TI mmWave SDK `.cfg` (TDM slot order, chirp variations, BPM, real ADC) and `TiConfig.to_radar(tx=, rx=)` builds the radar; `preset("awr1843boost" | "iwr6843isk")` builds a board.
- `witwin.radar.rawdata.write_dca1000` / `read_dca1000`: raw ADC in the DCA1000 capture layout (two-lane xWR16xx/18xx/68xx and four-lane xWR14xx, complex or real).
- `witwin-radar-doctor` / `python -m witwin.radar doctor`: runtime, native-library and Channel checks plus a GPU simulate smoke test, with a verdict and a fix per problem.
- `witwin.radar.deployment.package_version` and `runtime_failures`; `Paths.row_identity` and `Paths.observation_times`.
