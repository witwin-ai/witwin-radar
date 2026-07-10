# Radar Feature List

## Packaging And Platform Support

- Prebuilt CUDA 12.8 wheels for CPython 3.10-3.14 on Linux x86_64 and Windows x86_64.
- Each wheel includes one Python-independent LibTorch Stable ABI Dirichlet library targeting PyTorch 2.10 or newer; the same binary is CI load-tested with PyTorch 2.10, 2.11, and 2.12 across CPython 3.10-3.14.
- Native CUDA images cover compute capabilities 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 10.1, and 12.0, with compute 12.0 PTX retained for forward-compatible Blackwell execution.
- Linux release wheels are repaired and validated as `manylinux_2_35_x86_64` artifacts before publication.

## Public API

- Declarative simulation flow: `Radar -> Scene -> torch.Tensor` via `radar.simulate(scene, ...)`
- Radar pose is controlled directly with `Radar(position=..., target=..., up=..., fov=...)` or `radar.set_pose(...)`
- Scene assembly uses `Scene.add_structure(...)`, `Scene.add_mesh(...)`, `Scene.add_smpl(...)`, and `Scene.add_structure_motion(...)`
- `SMPLBody` is owned by `witwin.radar.geometry` and remains exported through `witwin.radar`; shared core stays free of radar-only body-model dependencies.
- Multi-radar orchestration is available via `Radar.simulate_group(...)`, returning a `dict[str, torch.Tensor]`
- Optional per-structure motion is available through `Scene.add_structure_motion(...)` and `Scene.update_structure(...)`. Callers pass `TransformMotion` instances directly.
- Public string-literal API types: `DetectorType`, `SamplingMode`, and `MotionSampling`
- Low-level radar solver entrypoint: `Radar.chirp()`, `Radar.frame()`, `Radar.mimo()`, and `Radar.apply_noise()`
- Ray-tracing entrypoint: `Tracer.trace()` returns `TraceResult(points, intensities)` and also carries `entry_points`, `fixed_path_lengths`, `depths`, and optional `normals` for generalized path tracing
- `radar.simulate(...)` returns the radar data tensor directly. The most recent scene, trace, and tracer are available as `radar.last_scene`, `radar.last_trace`, and `radar.last_tracer` for debugging.

## Configuration

- `RadarConfig` frozen schema validates required radar fields and antenna layouts
- Optional `antenna_pattern` config defaults to a broadside dipole and also supports separable `x/y` 1D gain curves or a direct 2D gain map
- Optional `noise_model` config supports thermal noise, quantization noise, and phase noise with optional deterministic seeding
- Optional `polarization` config supports simplified TX/RX polarization vectors with alias strings (`horizontal` / `vertical`) or per-element 3D vectors
- Optional `receiver_chain` config supports `lna`, `agc`, and `adc` stages plus absolute TX-power scaling via `config["power"]`
- `Radar` accepts `RadarConfig` or raw config dictionaries
- `Tracer(scene, radar, ...)` and `radar.simulate(...)` accept `multipath`, `max_reflections`, and `ray_batch_size`
- `radar.simulate(...)` and `Radar.simulate_group(...)` accept `motion_sampling="per_frame" | "linear" | "per_chirp"` for dynamic scenes

## Solver Execution

- Native Dirichlet runtime state lives on `radar.solver`, including FFT metadata such as `pad_factor` and `N_fft`
- `Radar(device="cuda")` validates CUDA availability explicitly. CPU construction is supported for configuration and helper workflows, but solver execution requires CUDA tensors.
- Linux and Windows are supported targets. Release wheels include prebuilt native CUDA extensions for supported Python/platform combinations; source builds require a CUDA-enabled PyTorch build plus a working CUDA/C++ compilation toolchain.
- Time-domain outputs from `Radar.chirp()`, `Radar.frame()`, and `Radar.mimo()` automatically apply `noise_model` when configured; `radar.mimo(..., freq_domain=True)` rejects built-in noise injection
- Time-domain outputs from `Radar.chirp()`, `Radar.frame()`, and `Radar.mimo()` automatically apply `receiver_chain` when configured; enabling it also moves `Radar.gain` onto an absolute transmit-voltage scale
- `receiver_chain.adc` and `noise_model.quantization` are mutually exclusive so only one ADC quantizer is active
- MIMO frames sample the scene once per TDM chirp slot (`chirp_per_frame * num_tx` evaluations), so the velocity-dependent per-TX phase removed by `_compensate_tdm_phase` is physically present; moving-target elevation is now correct instead of biased by radial velocity
- `radar.mimo(interpolator, ...)` batches all TDM slots into grouped native kernel launches for roughly 8-16x faster dynamic-frame generation
- Public chirp and MIMO autograd uses analytical native CUDA backward kernels; float64 PyTorch implementations remain validation references only
- `motion_sampling="linear"` uses two triangle traces plus the fused range-rate MIMO kernel for fast rigid-motion approximation

## Rendering And Dynamics

- `Tracer.trace()` has a single public signature with no ignored `spp` parameter
- `Scene.compile_renderables(time=...)` and `Tracer.trace(time=...)` expose time-dependent geometry for dynamic scenes
- Multipath tracing is available for `sampling="pixel"` and uses radar-center path tracing with configurable maximum specular reflection depth
- The Dirichlet solver consumes generalized path samples and applies FSPL from the total `tx -> bounces -> scatter -> rx` distance
- When `polarization` is configured, traced path normals are propagated through the runtime and used for simplified reflection/projection coupling
- Shared core geometry constructors default to `device=None`, while radar `Scene(...)` owns device placement and defaults to CUDA
- `Timeline.from_motion()` uses the tracer result contract directly
- Dynamic structure motion supports rigid `translation`, `rotation`, and `parent` inheritance so rotational Doppler can be modeled directly from the scene
- `radar.mimo(..., freq_domain=True)` remains available for Dirichlet frequency-domain output

## Signal Processing

- `process_pc(..., detector=...)` accepts the validated detector set `{"cfar", "topk"}`
- `frame2pointcloud(...)` requires a `radar` argument so TDM-MIMO compensation is never skipped silently
- `PointCloudProcessConfig` provides the normalized point-cloud extraction config surface
- `range_fft(...)`, `doppler_fft(...)`, `clutter_removal(...)`, `process_pc(...)`, and `process_rd(...)` keep tensor inputs on the PyTorch path and use `torch.fft` for GPU-native DSP
- `process_pc_tensor(...)` and `process_rd_tensor(...)` keep outputs on-device; `process_pc(...)` and `process_rd(...)` are NumPy compatibility wrappers
- `ca_cfar_2d(...)`, `ca_cfar_2d_fast(...)`, and `os_cfar_2d(...)` return `(detections, threshold_map)` with a consistent CFAR contract
- `ca_cfar_2d(...)`, `ca_cfar_2d_fast(...)`, and `os_cfar_2d(...)` all accept NumPy arrays and PyTorch tensors; the reference CA/OS paths now stay on the torch device instead of falling back to CPU
