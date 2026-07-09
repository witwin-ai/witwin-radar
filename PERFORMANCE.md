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
