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

## Processing Chain (Phase 8)

The maintained benchmark is:

```bash
python tools/benchmark_processing.py --runs 200 --warmup 20 --json
```

Same timing convention as the Dirichlet benchmark: CUDA events with an explicit
`torch.cuda.synchronize`, medians, and a peak-allocation delta around a single
call. Measured on an NVIDIA GeForce RTX 5080. Two sizes per stage: the frozen
fixture (8 chirps, 2x2, 256 samples) and one realistic size (128 chirps, 3x4,
256 samples).

A caveat that must not be dropped: a synchronization inside cuFFT plan creation
is invisible from Python. The `fft` and `host` columns count DISPATCHES and
HOST-VISIBLE observations. Wall time is measured with CUDA events and is never
inferred from a counter.

### Transforms

| Stage | fixture | realistic | fft | note |
| --- | ---: | ---: | ---: | --- |
| `range_profile` FMCW | 0.079 ms | 0.079 ms | 1 | |
| `range_profile` OFDM | 0.103 ms | 0.084 ms | 1 | the CIR inversion |
| `range_profile` pulsed | 0.196 ms | 0.210 ms | 4 | matched filter |
| `range_doppler` | 0.142 ms | 0.145 ms | 2 | transform plus shift |
| window multiply alone | 0.062 ms | 0.062 ms | 0 | |
| transform alone | 0.019 ms | 0.020 ms | 1 | |
| `matched_filter` float32 | 0.254 ms | 0.253 ms | 3 | |
| `matched_filter` complex128 | 0.245 ms | 0.261 ms | 3 | the deleted upcast |
| `fft2_aoa` | 0.396 ms | 0.393 ms | 1 | |
| micro-Doppler framing copy | 0.022 ms | 0.015 ms | 0 | |
| micro-Doppler transform | 0.021 ms | 0.036 ms | 1 | |

Every stage is flat in problem size to within noise across a 48x larger cube, so
the chain is dispatch bound rather than work bound. The **window multiply costs
3.2x the transform it feeds** (0.062 against 0.019 ms) - the largest single
finding in this table and a Torch-side fusion candidate, not a cuFFT one.

The `complex128` upcast the cutover deleted bought **no measurable time** at
either size (its cost is 2.2x the peak allocation: 13.5 MB against 6.25 MB); the
deletion is a memory and precision-honesty change, not a speed change.

### Detectors

| Detector | shape | median | peak |
| --- | --- | ---: | ---: |
| `ca_cfar` | `[128, 256]` | 0.530 ms | 1.18 MB |
| `ca_cfar_fast` | `[128, 256]` | 0.115 ms | 0.62 MB |
| `os_cfar` | `[128, 256]` | 1.265 ms | **138.0 MB** |
| `ca_cfar` | `[12, 128, 256]` | 0.544 ms | 10.76 MB |
| `ca_cfar_fast` | `[12, 128, 256]` | 0.107 ms | 7.76 MB |
| `os_cfar` | `[12, 128, 256]` | 14.53 ms | **1644 MB** |
| `ca_cfar_1d` | `[12, 256]` | 0.207 ms | 0.08 MB |

`os_cfar` is the memory outlier by a factor of 222 on one map and 212 on twelve:
it materialises `[batch, D * R, n_outer]` training patches and sorts them. The
number is pinned as a budget in `tests/test_phase8_pipeline_budget.py` rather
than left to be discovered.

The docstring claim that `ca_cfar_fast` is "~100x faster" was measured and is
false; it is 4.6x on CUDA here, 4.1-6.1x in the separate cutover measurement,
and SLOWER than `ca_cfar` on CPU (0.72x on one map, 0.38x on eight). The claim
has been deleted from the source.

### Angle of arrival

| Stage | fixture | realistic | note |
| --- | ---: | ---: | --- |
| `tdm_compensate` | 0.085 ms | 0.093 ms | one broadcast multiply |
| the deleted Python TX loop | 0.139 ms | 0.136 ms | 1.6x, and it cloned |
| `phase_comparison_aoa` | 0.526 ms | 0.531 ms | two padded transforms |
| `fft2_aoa` | 0.391 ms | 0.421 ms | one `fft2` |
| `music_spectrum` | 0.704 ms | 0.711 ms | whole call |
| MUSIC smoothing, `unfold` | 0.014 ms | 0.014 ms | |
| the deleted `stack` comprehension | 0.100 ms | 0.081 ms | 5.9x |
| MUSIC `eigh` | 0.231 ms | 0.251 ms | a third of the call |

### Cube formation

| Stage | fixture | realistic | note |
| --- | ---: | ---: | --- |
| `assemble_frame_cube` | 0.018 ms | 0.018 ms | permute/reshape/permute/contiguous |
| `ProcessingCube.from_synthesis` | 0.019 ms | 0.019 ms | |
| `conventional_steering` | 0.078 ms | 0.078 ms | scene static, cacheable |
| `beam_cube` | 0.041 ms | 0.063 ms | the second full copy |

The two full copies together are 0.081 ms against a 2.23 ms pipeline: 3.6 percent.

### Full pipeline, and the frozen budgets

`synthesize -> cube -> range profile -> Range-Doppler -> CFAR -> AoA -> point
cloud`, one call, real Channel fixture at 3 TX x 4 RX.

| Detector | median | peak delta | fft dispatches | host observations |
| --- | ---: | ---: | ---: | ---: |
| `ca_cfar_fast` | **2.23 ms** | **1.13 MB** | 7 | 1 |
| `ca_cfar` | 2.68 ms | 1.13 MB | 7 | 1 |
| `os_cfar` | 2.40 ms | 3.33 MB | 7 | 1 |

Process-to-process medians spanned 2.19 to 2.34 ms over four independent runs.

The frozen budgets, each asserted in `tests/test_phase8_pipeline_budget.py` with
its measured number and headroom factor inline:

| Budget | Measured | Frozen at |
| --- | ---: | ---: |
| full-pipeline latency | 2.23 ms | `x 1.30` = 2.90 ms |
| full-pipeline peak allocation delta | 1.13 MB | `x 1.25` = 1.41 MB |
| host observations per pipeline call | 1 | exactly 1 |
| `torch.fft` dispatches per pipeline call | 7 | exactly 7 |
| per-frame simulation cost | 3.88 ms | `x 1.30` = 5.04 ms |
| `os_cfar` peak, one `[128, 256]` map | 138.0 MB | `x 1.25` = 172.5 MB |
| wideband D2H copies / synchronizations per leg | 1 / 1 | exactly, for `F` in {1, 8, 64} |
| two-way join launches | `1 + F` | exactly, for `F` in {1, 8, 64} |

The single host observation is the `torch.argwhere` inside `point_cloud`, and it
IS the stage: a point cloud has a data-dependent length. The seven dispatches are
one range transform, two for the Doppler stage, two building the velocity axis,
and two inside the phase comparison.

**On the per-frame simulation cost.** The Phase-7 report recorded 2.30 ms/frame
for two leg reevaluations plus one composition. That figure is not reproducible
in this environment and the reason is not a Phase-8 regression: measured at the
Phase-8 BASE commit `4bb059a` in the same session on the same fixture, the same
call costs **3.911 ms**, against **3.880 ms** at HEAD - a ratio of 0.992. The
2.30 ms figure describes a different machine state. The portable claim is the
ratio, and it says nothing regressed.

**Component export multiplier.** Exporting components costs one synthesis launch
each and zero host observations. On the multi-endpoint fixture: unseparated
0.160 ms, the two populated classes 0.393 ms (2.46x), all four classes 0.797 ms
(4.98x). Against a 4.94 ms frame that is 1.05x.

**Wideband `F`-loop join cost.** 2 TX x 2 sites x 2 RX, 11 composed rows:

| `F` | leg reevaluation | compose | marginal join |
| ---: | ---: | ---: | ---: |
| none | 3.501 ms | 0.267 ms | - |
| 1 | 5.683 ms | 0.449 ms | 0.182 ms/column |
| 8 | 21.83 ms | 1.665 ms | 0.175 ms/column |
| 16 | 38.52 ms | 3.304 ms | 0.190 ms/column |
| 64 | 140.4 ms | 9.629 ms | 0.146 ms/column |

The join loop is 6-8 percent of a wideband frame; the dominant term is Channel's
`(1 + F)` native launches. A strided `[K, F]` native join is not justified by
this table and would need its own decision record and a measurement that beats
it, shipping primal, JVP and VJP in the same change.

### The native-DSP gate

Measured and recorded: **no native DSP in Phase 8.** Of the four criteria, only
(a) dispatch-bound is tripped, and a cuFFT wrapper replaces one dispatch with
another dispatch - it removes no launch. What the data argues for is fewer,
larger launches or a captured CUDA graph, both Torch-side. The reasoning and the
per-criterion numbers are in
`docs/dev/standards/radar-adr-017-processing-facade-and-frozen-dsp-surface.md`.

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
