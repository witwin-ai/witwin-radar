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
- PyTorch FFT/autograd reference implementations for target counts below the configured memory limit

## Legacy Slang Warm Comparison

The legacy `dirichlet.slang` kernel was measured separately from git history (`8ea84e2^`) to compare warmed steady-state performance against the native CUDA implementation. Compilation and module load were excluded from timing; each row used 8 warmup runs and 30 CUDA-event timed runs on an NVIDIA GeForce RTX 5080 with `adc_samples=400`, `N_fft=6400`, `num_bins=3200`, and `targets_per_chunk=256`.

SlangTorch's default CUDA build enables fast math. That default is faster, but it is not numerically equivalent to the native/PyTorch-correct path for this kernel because phase reduction differs for large radar carrier phases. In the default fast-math run, max relative error reached `5.57e-1` for forward output and about `2.0` for gradients at 1,048,576 targets, so those timings are not a correctness-preserving comparison.

With Slang fast math disabled, native CUDA and legacy Slang produced identical forward and backward tensors in this benchmark:

| Targets | Native fwd ms | Slang fwd ms | Native/Slang fwd | Native bwd ms | Slang bwd ms | Native/Slang bwd |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.144 | 0.152 | 1.05x faster | 1.673 | 1.658 | 0.99x |
| 16,384 | 0.440 | 0.471 | 1.07x faster | 1.714 | 1.658 | 0.97x |
| 262,144 | 6.240 | 6.166 | 0.99x | 12.141 | 11.844 | 0.98x |
| 1,048,576 | 24.765 | 24.828 | 1.00x | 44.435 | 39.897 | 0.90x |

The practical read is that the native kernel matches legacy Slang forward performance at equivalent precision, while backward remains roughly comparable but is about 10% slower at the largest target count measured.

## Method

The native forward kernel evaluates the Dirichlet closed form directly in frequency space, avoiding the time-domain signal materialization and FFT used by the reference path. The native backward kernel applies the analytical gradients for the same expression and accumulates per-target distance/amplitude gradients.

The PyTorch reference path remains in `witwin.radar.solvers.common` for tests and validation only. It is intentionally not exposed as a runtime backend because it allocates an `O(targets * samples)` intermediate and can run out of memory for large scenes.

## Expected Shape

For the default chirp benchmark (`adc_samples=400`, `pad_factor=16`, `N_fft=6400`, `num_bins=3200`):

| Method | Memory trend | Notes |
| --- | --- | --- |
| Native Dirichlet forward | `O(chunks * bins)` scratch plus output | production path |
| Native Dirichlet backward | `O(chunks * targets)` gradient scratch | production autograd path |
| PyTorch FFT reference | `O(targets * samples)` complex intermediate | validation only |

Use the JSON output from the benchmark in release notes when reporting hardware-specific numbers. Timings depend on GPU model, CUDA toolkit, PyTorch build, and driver.
