# Radar Feature List

## Public API

- Declarative simulation flow: `Scene -> Simulation -> Result`
- Radar pose can be controlled explicitly with `Sensor(...)` on `Radar`, `Renderer`, or `Simulation`
- Scene assembly uses `Scene.set_sensor(...)`, `Scene.add_structure(...)`, `Scene.add_mesh(...)`, `Scene.add_smpl(...)`, and `Scene.add_structure_motion(...)`
- Multi-radar orchestration is available via `Simulation.mimo_group(...)`, `RadarSpec`, and `MultiResult`
- Optional per-structure motion is available through `Scene.add_structure_motion(...)`, `Scene.set_structure_motion(...)`, and `Scene.clear_structure_motion(...)`. Callers pass `TranslationMotion` / `RotationMotion` instances directly.
- Public string-literal API types: `SolverBackend`, `DetectorType`, `SamplingMode`, and `MotionSampling`
- Low-level radar solver entrypoint: `Radar.chirp()`, `Radar.frame()`, `Radar.mimo()`, and `Radar.apply_noise()`
- Ray-tracing entrypoint: `Renderer.trace()` returns `TraceResult(points, intensities)` and also carries `entry_points`, `fixed_path_lengths`, and `depths` for generalized path tracing
- `Result.signal()`, `Result.trace_points()`, `Result.trace_intensities()`, `Result.trace_entry_points()`, `Result.trace_fixed_path_lengths()`, and `Result.trace_depths()` provide semantic tensor access, with `Result.tensor(...)` retained as a generic fallback

## Configuration

- `RadarConfig` frozen schema validates required radar fields and antenna layouts
- Optional `antenna_pattern` config defaults to a broadside dipole and also supports separable `x/y` 1D gain curves or a direct 2D gain map
- Optional `noise_model` config supports thermal noise, quantization noise, and phase noise with optional deterministic seeding
- Optional `polarization` config supports simplified TX/RX polarization vectors with alias strings (`horizontal` / `vertical`) or per-element 3D vectors
- Optional `receiver_chain` config supports `lna`, `agc`, and `adc` stages plus absolute TX-power scaling via `config["power"]`
- `Radar` accepts `RadarConfig`, `dict`, or JSON config path
- `Simulation` accepts the same validated config inputs as `Radar`
- `Renderer(...)` and `Simulation.mimo(...)` accept `multipath`, `max_reflections`, and `ray_batch_size`
- `Simulation.mimo(...)` and `Simulation.mimo_group(...)` accept `motion_sampling="per_frame" | "per_chirp"` for dynamic scenes

## Backend Execution

- Three solver backends: `pytorch`, `slang`, `dirichlet`
- Backend-specific runtime state lives on `radar.solver`, including Dirichlet FFT metadata such as `pad_factor` and `N_fft`
- `Radar(device=...)` validates CUDA availability explicitly; `slang` and `dirichlet` require CUDA, while `pytorch` honors the selected device
- Time-domain outputs from `Radar.chirp()`, `Radar.frame()`, and `Radar.mimo()` automatically apply `noise_model` when configured; `radar.mimo(..., freq_domain=True)` rejects built-in noise injection
- Time-domain outputs from `Radar.chirp()`, `Radar.frame()`, and `Radar.mimo()` automatically apply `receiver_chain` when configured; enabling it also moves `Radar.gain` onto an absolute transmit-voltage scale
- `receiver_chain.adc` and `noise_model.quantization` are mutually exclusive so only one ADC quantizer is active

## Rendering And Dynamics

- `Renderer.trace()` has a single public signature with no ignored `spp` parameter
- `Scene.compile_renderables(time=...)` and `Renderer.trace(time=...)` expose time-dependent geometry for dynamic scenes
- Multipath tracing is available for `sampling="pixel"` and uses radar-center path tracing with configurable maximum specular reflection depth
- Solver backends consume generalized path samples and apply FSPL from the total `tx -> bounces -> scatter -> rx` distance
- When `polarization` is configured, traced path normals are propagated through the runtime and used for simplified reflection/projection coupling
- Shared core geometry constructors default to `device=None`, while radar `Scene(...)` owns device placement and defaults to CUDA
- `Timeline.from_motion()` uses the renderer trace contract directly
- Dynamic structure motion supports rigid `translation`, `rotation`, and `parent` inheritance so rotational Doppler can be modeled directly from the scene
- `radar.mimo(..., freq_domain=True)` remains available for Dirichlet frequency-domain output

## Signal Processing

- `process_pc(..., detector=...)` accepts the validated detector set `{"cfar", "topk"}`
- `frame2pointcloud(...)` requires a `radar` argument so TDM-MIMO compensation is never skipped silently
- `PointCloudProcessConfig` provides the normalized point-cloud extraction config surface
- `range_fft(...)`, `doppler_fft(...)`, `clutter_removal(...)`, `process_pc(...)`, and `process_rd(...)` keep tensor inputs on the PyTorch path and use `torch.fft` for GPU-native DSP
- `ca_cfar_2d(...)`, `ca_cfar_2d_fast(...)`, and `os_cfar_2d(...)` return `(detections, threshold_map)` with a consistent CFAR contract
- `ca_cfar_2d(...)`, `ca_cfar_2d_fast(...)`, and `os_cfar_2d(...)` all accept NumPy arrays and PyTorch tensors; the reference CA/OS paths now stay on the torch device instead of falling back to CPU
