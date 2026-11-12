## Breaking changes

- CFAR detectors are square law. Complex input is detected on `|x|^2`; a real input is read as a POWER map. Migration: pass complex IQ or `|x|^2`, never `|x|` or decibels. `combine_coherent` now returns the power `|sum_p x_p|^2` instead of the magnitude.
- `os_cfar`'s `pfa` is now the achieved false-alarm rate (the constant is solved from the ordered-statistic law). Thresholds and detection counts of every detector change.
- `point_cloud`'s default `route` is `"scan"` (any array layout) instead of `"phase_comparison"`.
- `point_cloud` groups peaks by default (`peak_neighbourhood=(3, 3)`): the cells one return lights up become one point. Pass `peak_neighbourhood=None` for per-cell points.
- `ArrayGeometry.from_axes` maps the radar's local offsets `(right, up, -boresight)` into the processing frame `(right, boresight, up)`; `ArrayGeometry.from_offsets` takes processing-frame offsets. Elevation offsets declared in a radar config (`tx_loc` `[x, 1, 0]`) now steer as elevation in `conventional_steering`, `beam_cube` and MUSIC.
- `fft2_aoa` reads the array's lattice instead of assuming TI interleaving and four transmitters; `phase_comparison_aoa` refuses layouts it cannot read, naming the reason.
- `microdoppler_spectrogram` returns a `MicroDopplerSpectrogram` record instead of a `(times, frequencies, spectrum)` tuple, normalised by `1 / window_slots`. Migration: read `.times_s`, `.frequency_hz`, `.data`.

## Added

- `go_cfar`, `so_cfar` and `cfar_false_alarm_probability` (the exact CA/GO/SO/OS false-alarm laws on exponential noise).
- `group_peaks`: local-maximum suppression of detections.
- `cluster_points` / `PointClusters`: DBSCAN over a point cloud, optionally with radial velocity.
- `scan_aoa` (route `"scan"`): steering-vector scan for any array layout; `ArrayLattice` describes lattice-aligned arrays.
- `PointCloud.frame` and `PointCloud.to_world(radar)`.
- `ProcessingAxes.from_radar(radar)` and `ProcessingCube.from_tensor(cube, radar)` for recorded or external cubes (sink-major `[slow, pair, fast]` or `[tx, rx, slow, fast]`).
- Zero padding: `range_profile(..., range_bins=)`, `range_doppler_map(..., doppler_bins=)`, `ProcessingAxes.with_bin_counts`.
- `remove_static_clutter(profile, method="mean" | "mti2" | "mti3")`.
- `range_angle_map` / `RangeAngleMap` / `azimuth_directions`.
- `SlowTimeSignal(wavelength_m=)` adds a velocity axis to the micro-Doppler product.
- `KalmanTracker` / `TrackState` (constant-velocity multi-target tracking) and `greedy_assignment`.

## Fixed

- CA-CFAR on `Frame.points` produced no false alarms at any `pfa` (a magnitude was thresholded with a power-law constant); every detector now measures within a few percent of its `pfa` at 1e-2, 1e-3 and 1e-4, edges included.
- CFAR padding wraps along Doppler and truncates (with re-solved constants) along range instead of replicating.
- Angle estimation no longer assumes a half-wavelength TI layout: 1x8, 2x4 and one-wavelength arrays give correct angles on the `fft2` and `scan` routes.

## Performance

- `os_cfar` and `cluster_points` on CUDA tensors run native kernels (`cfar_order_statistic_forward`, one launch, no patch tensor; `dbscan_labels_forward`, four launches, no host round trip). CPU tensors use the same algorithm in Torch.
