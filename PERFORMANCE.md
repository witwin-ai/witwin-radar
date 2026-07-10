# Radar CUDA Kernel Performance

## Overview

Radar signal generation now uses a single native CUDA implementation of the Dirichlet frequency-domain kernel. The package no longer ships legacy solver backends.

The maintained benchmark is:

```bash
python tools/benchmark_dirichlet_cuda.py --targets 1024 16384 262144 1048576 --runs 20 --json
```

It measures:

- native Dirichlet forward spectrum generation
- native Dirichlet backward gradients for distance and amplitude
- public native-autograd forward+backward time and peak allocated memory
- PyTorch FFT/autograd reference implementations for target counts below the configured memory limit

## Legacy Slang Warm Comparison (Historical Kernel)

The legacy `dirichlet.slang` kernel was measured separately from git history (`8ea84e2^`) to compare warmed steady-state performance against the native CUDA implementation. Compilation and module load were excluded from timing; each row used 8 warmup runs and 30 CUDA-event timed runs on an NVIDIA GeForce RTX 5080 with `adc_samples=400`, `N_fft=6400`, `num_bins=3200`, and `targets_per_chunk=256`.

SlangTorch's default CUDA build enables fast math. That default is faster, but it is not numerically equivalent to the native/PyTorch-correct path for this kernel because phase reduction differs for large radar carrier phases. In the default fast-math run, max relative error reached `5.57e-1` for forward output and about `2.0` for gradients at 1,048,576 targets, so those timings are not a correctness-preserving comparison.

With Slang fast math disabled, native CUDA and legacy Slang produced identical forward and backward tensors in this benchmark:

| Targets | Native fwd ms | Slang fwd ms | Native/Slang fwd | Native bwd ms | Slang bwd ms | Native/Slang bwd |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.144 | 0.152 | 1.05x faster | 1.673 | 1.658 | 0.99x |
| 16,384 | 0.440 | 0.471 | 1.07x faster | 1.714 | 1.658 | 0.97x |
| 262,144 | 6.240 | 6.166 | 0.99x | 12.141 | 11.844 | 0.98x |
| 1,048,576 | 24.765 | 24.828 | 1.00x | 44.435 | 39.897 | 0.90x |

This comparison predates the parallel-bin backward kernel documented below. It
remains useful for forward parity with the removed Slang backend, but its native
backward timings are no longer representative of the current implementation.

## Native Autograd Acceptance

Public `Radar.chirp()` autograd now uses analytical CUDA backward kernels instead
of constructing the float64 PyTorch reference graph. A one-block-per-path kernel
parallelizes spectrum bins and reduces gradients with warp shuffles. Batched MIMO
spectra use a separate one-thread-per-path kernel because each spectrum has an
independent output gradient.

RTX 5080 results (`adc_samples=400`, `N_fft=6400`, 20 CUDA-event runs):

| Targets | Native forward | Native backward | Native autograd e2e | PyTorch autograd e2e | E2E speedup | Native peak | PyTorch peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.124 ms | 0.064 ms | 0.510 ms | 1.144 ms | 2.24x | 8.6 MB | 53.2 MB |
| 16,384 | 0.417 ms | 0.626 ms | 1.159 ms | 6.183 ms | 5.33x | 11.6 MB | 703.0 MB |
| 262,144 | 5.514 ms | 9.760 ms | 15.369 ms | skipped | n/a | 85.4 MB | skipped |

Forward and gradient correctness are checked against the float64 reference and
finite differences in `tests/solvers/test_native_dirichlet_cuda.py` and
`tests/solvers/test_mimo_grad.py`.

## MIMO Frame Paths

MIMO frame generation offers three paths with different speed/fidelity trade-offs (measured on an NVIDIA GeForce RTX 5080, 3TX x 4RX, 128 chirp loops, 256 ADC samples, CUDA-event medians):

| Targets | `mimo(interpolator)` | `mimo_from_trace(velocities=...)` | `mimo_from_trace()` static |
| ---: | ---: | ---: | ---: |
| 1,024 | 21.9 ms | 5.7 ms | 2.2 ms |
| 13,776 | 84.4 ms | 44.3 ms | 2.3 ms |
| 131,072 | 863.8 ms | 402.3 ms | 4.5 ms |

- `mimo(interpolator)` resamples the scene once per TDM chirp slot (`chirp_per_frame * num_tx` evaluations) and batches all slots into grouped `forward_chunked` launches. This is the highest-fidelity dynamic path.
- `mimo_from_trace(velocities=...)` uses the fused `forward_mimo_linear_chunked` kernel with a first-order per-path range-rate model and per-TX slot timing.
- The static path evaluates one chirp and expands it across the frame.

Before the slot batching (per-chirp Python loop with one small kernel launch per chirp), the interpolator path took 348/885/6561 ms for the same target counts, so the batched path is roughly 8-16x faster while also simulating per-TX TDM timing.

## End-to-End Scene And DSP

Steady-state RTX 5080 medians for 3TX x 4RX, 128 chirps, 256 ADC samples:

| Operation | Median |
| --- | ---: |
| Pixel trace, 128x128, 6,248 hits | 1.457 ms |
| Cached static MIMO, 13,776 paths | 2.361 ms |
| Cached static MIMO autograd e2e, 13,776 paths | 5.457 ms |
| Tensor Range-Doppler | 0.234 ms |
| Tensor point cloud | 2.038 ms |

Strict dynamic triangle tracing evaluates all 384 TDM slots and remains the
highest-fidelity option. For a translating box scene it measured 917 ms.
`motion_sampling="linear"` traces two adjacent slots, matches triangle IDs, and
uses the fused linear range-rate kernel; the same scene measured 14.65 ms
(62.6x faster). For a 32-chirp, 0.2 m/s translation check, complex-signal
relative L2 error was `5.9e-4` and relative peak-magnitude error was `1.0e-3`.

## Method

The native forward kernel evaluates the Dirichlet closed form directly in frequency space, avoiding the time-domain signal materialization and FFT used by the reference path. The native backward kernel applies the analytical gradients for the same expression and accumulates per-target distance/amplitude gradients.

The PyTorch reference path remains in `witwin.radar.solvers.common` for tests and validation only. It is intentionally not exposed as a runtime backend because it allocates an `O(targets * samples)` intermediate and can run out of memory for large scenes.

## Expected Shape

For the default chirp benchmark (`adc_samples=400`, `pad_factor=16`, `N_fft=6400`, `num_bins=3200`):

| Method | Memory trend | Notes |
| --- | --- | --- |
| Native Dirichlet forward | `O(chunks * bins)` scratch plus output | production path |
| Native Dirichlet backward | `O(targets)` saved inputs and output gradients | production autograd path |
| PyTorch FFT reference | `O(targets * samples)` complex intermediate | validation only |

Use the JSON output from the benchmark in release notes when reporting hardware-specific numbers. Timings depend on GPU model, CUDA toolkit, PyTorch build, and driver.
