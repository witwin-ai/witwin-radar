# Fixed-Trace MIMO Fast Path

This note documents the latest MIMO simulation path for cases where the ray
tracing result is already known for a frame. It is meant for real-time radar
simulation experiments where scene geometry, visibility, material response, and
surface samples can be preprocessed, and the frame only needs FMCW MIMO signal
generation.

## What Changed

- Added an explicit fixed-trace API that generates a full MIMO cube without
  calling ray tracing or a per-chirp interpolator inside the frame.
- Added an optional first-order velocity model for multi-chirp Doppler. The
  solver reuses the frame-start paths and updates path phase from per-path
  one-way range rates.
- Added `MimoPathCache` so static path distances, amplitudes, and optional
  range rates can be precomputed once and reused across repeated frame
  generation.
- Added a Dirichlet Slang kernel for linear path motion across chirps.
- Updated `Radar.simulate(...)` so non-moving scenes, or moving scenes requested
  with `motion_sampling="per_frame"`, use the fixed-trace MIMO path.
- Kept `Radar.mimo(interpolator, fast=False)` as the legacy per-chirp interface.
  The default remains the exact per-chirp behavior for caller-provided
  interpolators.
- Added `tools/benchmark_realtime.py` to measure fixed-trace MIMO throughput.

## Public API

### `Radar.mimo_from_trace`

```python
frame = radar.mimo_from_trace(
    trace,
    velocities=None,
    t0=0.0,
    freq_domain=False,
    amplitude_update="range_loss",
)
```

Generates one MIMO frame from a single `TraceResult` or legacy
`(intensities, points)` sample.

Arguments:

- `trace`: frame-start trace result. For velocity mode, this must be
  `TraceResult`-like and expose `points` and `entry_points`.
- `velocities`: optional tensor with shape `(N, 3)` in meters per second,
  aligned with `trace.points`.
- `t0`: frame start time in seconds. Dirichlet fast kernels use the fixed trace
  directly; fallback backends use `t0` when reconstructing an interpolator.
- `freq_domain`: Dirichlet-only option. If `True`, returns range-domain spectra
  before IFFT. This cannot be combined with `noise_model` or `receiver_chain`,
  which require time-domain output.
- `amplitude_update`: Dirichlet velocity-mode option. Use `"range_loss"` to
  apply first-order range-loss amplitude scaling, or `"constant"` to freeze
  amplitude across chirps.

### `Radar.path_cache_from_trace`

```python
cache = radar.path_cache_from_trace(trace, velocities=None)
```

Precomputes per-TX/RX one-way distances, amplitude weights, and optional
one-way distance rates from one trace. This is currently implemented by the
Dirichlet backend.

### `Radar.mimo_from_paths`

```python
frame = radar.mimo_from_paths(
    cache,
    freq_domain=False,
    amplitude_update="range_loss",
)
```

Generates a full MIMO frame from a `MimoPathCache`. Use this when the same
preprocessed paths are evaluated repeatedly, for example during throughput
tests, frame replay, or batched parameter sweeps.

### `MimoPathCache`

```python
from witwin.radar import MimoPathCache

cache = MimoPathCache(
    one_way_distances=distances,
    amplitudes=amplitudes,
    one_way_distance_rates=rates,
)
```

All tensors use shape `(num_tx, num_rx, num_paths)`.

- `one_way_distances`: one-way path distance in meters.
- `amplitudes`: amplitude-domain solver weights after radar-equation scaling.
- `one_way_distance_rates`: optional one-way range rate in meters per second.

## Simulation Entry Point

`Radar.simulate(...)` now has two moving-scene sampling modes:

```python
frame = radar.simulate(scene, motion_sampling="per_chirp")
frame = radar.simulate(scene, motion_sampling="per_frame")
```

- `motion_sampling="per_chirp"` keeps the existing behavior for moving scenes:
  ray tracing can be evaluated at each chirp time through the interpolator.
- `motion_sampling="per_frame"` traces once at frame start and calls
  `mimo_from_trace(trace, t0=t0)`.
- Static scenes also use `mimo_from_trace(trace, t0=t0)` automatically.

For real-time experiments where ray tracing is already amortized or
preprocessed, prefer the explicit API:

```python
trace = tracer.trace(time=frame_t0)
frame = radar.mimo_from_trace(trace)
```

With known velocities:

```python
trace = tracer.trace(time=frame_t0)
velocities = world_velocity_per_trace_point
frame = radar.mimo_from_trace(trace, velocities=velocities)
```

With reusable cache:

```python
cache = radar.path_cache_from_trace(trace, velocities=velocities)
frame = radar.mimo_from_paths(cache)
```

## Multi-Chirp Doppler Model

The optimized velocity path avoids recomputing every chirp from updated 3D
points. For each TX/RX/path tuple it precomputes:

```text
d(t) = d0 + d_rate * t
```

where `d0` is the frame-start one-way distance and `d_rate` is the first-order
one-way distance rate from the current path geometry:

```text
total_path_rate =
    dot(entry_point - tx, velocity) / |entry_point - tx|
  + dot(point - rx, velocity) / |point - rx|

d_rate = 0.5 * total_path_rate
```

The Slang kernel then evaluates the Dirichlet FMCW phase per chirp from this
linear distance model. This gives Doppler from path phase evolution without
calling the full per-chirp position, visibility, or material pipeline.

The chirp timing follows the existing solver convention:

```text
t_chirp = chirp_id * T_chirp * num_tx
```

There is no extra per-TX transmit-time offset in this fast path, matching the
current MIMO solver contract.

## Accuracy Contract

The fixed-trace velocity model is a first-order approximation. It is appropriate
when the frame is short enough that the following are effectively constant:

- path visibility and occlusion,
- reflection point identity,
- surface normal and Fresnel/material response,
- antenna pattern lookup,
- polarization terms,
- off-axis geometry changes beyond first-order range rate.

It is exact for the intended static case, and it captures the dominant Doppler
phase term for short-frame radial or near-radial motion. It is less accurate
when a target moves enough within the frame to cross silhouettes, change
reflecting surfaces, or significantly change incidence/antenna angles. Use
`motion_sampling="per_chirp"` or the legacy `radar.mimo(interpolator)` path when
that behavior matters more than throughput.

`amplitude_update="range_loss"` updates amplitude using the first-order range
loss ratio. `amplitude_update="constant"` freezes amplitude and only updates
phase.

## Benchmarking

Use the included benchmark script:

```bash
python tools/benchmark_realtime.py --backend dirichlet --targets 1024 6248 --chirps 128 --warmup 3 --runs 10
```

Measured on an RTX 5080 with 3TX, 4RX, 128 chirps, and 256 ADC samples:

| Path | Targets | Mean Time | Throughput |
| --- | ---: | ---: | ---: |
| `mimo_from_trace` static | 1024 | 4.787 ms | 208.91 FPS |
| `mimo_from_trace` static | 6248 | 5.957 ms | 167.88 FPS |
| `mimo_from_paths` linear cache | 1024 | 1.615 ms | 619.33 FPS |
| `mimo_from_paths` linear cache | 6248 | 10.492 ms | 95.31 FPS |

These numbers isolate MIMO signal generation. End-to-end FPS will also depend
on scene update, trace production, DSP, point-cloud extraction, data transfer,
and visualization.

## Verification

The change was validated with:

```bash
python -m pytest tests/solvers/test_mimo_cross.py --gpu -q
python -m pytest tests/core/test_radar_simulation.py tests/core/test_scene_motion.py -q
python -m pytest tests/solvers/test_chirp_cross.py tests/solvers/test_solver_edge.py --gpu -q
python -m py_compile witwin/radar/radar.py witwin/radar/path_cache.py witwin/radar/solvers/solver_dirichlet.py tools/benchmark_realtime.py
```

Expected results from the validation run:

- `tests/solvers/test_mimo_cross.py --gpu`: 11 passed.
- `tests/core/test_radar_simulation.py tests/core/test_scene_motion.py`: 12
  passed, 1 skipped.
- `tests/solvers/test_chirp_cross.py tests/solvers/test_solver_edge.py --gpu`:
  13 passed.
- `py_compile`: exit code 0.
