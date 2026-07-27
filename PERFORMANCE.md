# Radar Performance

## Overview

Two benchmarks are maintained, and everything in this document comes from one of
them or from a pinned test.

```bash
python tools/benchmark_processing.py --runs 200 --warmup 20 --json
python -m pytest tests/test_phase8_pipeline_budget.py --gpu -q -s
```

One timing convention throughout: CUDA events or `perf_counter` with an explicit
`torch.cuda.synchronize` on both sides, medians rather than means, and a peak
allocation delta measured around a single call. Measured on an NVIDIA GeForce
RTX 5080.

The Phase-11 cutover deleted `tools/benchmark_dirichlet_cuda.py` and the whole
Dirichlet route it measured (`Radar.chirp`, `Radar.frame`, `Radar.mimo`,
`mimo_from_trace`, `witwin.radar.solvers`). Every number that came from it has
been removed rather than carried forward: those tables described a code path
that no longer exists, and a benchmark table for deleted code is worse than no
table. The pre-cutover numbers remain in git history at `27829a9^`.

## Scene-Driven Simulation

`Radar.simulate` cost against scatter-site count. Line of sight only,
`max_depth=0`, 3 TX x 4 RX, 128 chirps, 256 samples, one frame; one warmup call
per size then the median of five, `torch.cuda.synchronize` on both sides.

| sites | composed rows | median | peak allocation |
| ---: | ---: | ---: | ---: |
| 512 | 6,144 | 26.5 ms | 6.5 MB |
| 2,048 | 24,576 | 105.3 ms | 8.2 MB |
| 4,096 | 49,152 | 209.8 ms | 10.3 MB |

Site count is the cost driver: every site is a Channel endpoint in BOTH legs, so
the composed row count is `num_tx * num_rx * sites`. These are single-process
medians, not a warmed benchmark suite; `tools/benchmark_processing.py` is the
instrument for the post-processing half.

### The per-frame budget

`tests/test_phase8_pipeline_budget.py::test_the_simulation_frame_cost_has_not_regressed`
pins the MARGINAL per-frame cost of `Radar.simulate` - `(T(2K) - T(K)) / K`, so
the one-off compile and discovery of the first frame do not enter the number.
The pin measures the minimum within each timing rather than the median, because
a difference of two medians is not a stable quantity (the measured spread across
repeats was 4.52 to 7.62 ms); it takes the best of three such differences after
an explicit warmup.

| where | measured | budget |
| --- | ---: | ---: |
| isolation | 4.5372 ms | 5.044 ms |
| inside the full `--gpu` suite | 4.3822 ms | 5.044 ms |
| inside `ci/run_ci_tier.py cuda` (under `coverage run`) | 4.1623 ms | 5.044 ms |

The third row exists because it did not always hold. `ci/run_ci_tier.py cuda`
runs the GPU suite under `coverage run`, and both wall-clock pins are dispatch
bound, so the per-line C tracer is charged straight to the measurement: 5.27 ms
against 4.54 ms for the same code, a 16 percent instrument tax that on its own
exceeded the budget. The pins now suspend the tracer and the profiler around the
timed region and restore them afterwards (`_untraced` in the budget file). The
thresholds are unchanged and the coverage floor still passes at 83 percent.

**Open item for the owner, recorded rather than acted on.** The 5.044 ms budget
is `3.88 x 1.30`, and 3.88 ms described two leg replays plus one composition
WITHOUT synthesis - a strictly smaller quantity than what the pin now measures.
The measurement passes unchanged, in isolation and under load, but it sits at
roughly 84 percent of a threshold derived for something else. Re-deriving it
(`MEASURED_SIMULATION_FRAME_MS = 4.25`, same 1.30 factor, 5.53 ms) is proposed
and deliberately NOT taken: raising a budget is an owner decision, and no phase
should raise its own.

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
| per-frame simulation cost | 3.88 ms (see above) | `x 1.30` = 5.04 ms |
| `os_cfar` peak, one `[128, 256]` map | 138.0 MB | `x 1.25` = 172.5 MB |
| wideband D2H copies / synchronizations per leg | 1 / 1 | exactly, for `F` in {1, 8, 64} |
| two-way join launches | `1 + F` | exactly, for `F` in {1, 8, 64} |

The single host observation is the `torch.argwhere` inside `point_cloud`, and it
IS the stage: a point cloud has a data-dependent length. The seven dispatches are
one range transform, two for the Doppler stage, two building the velocity axis,
and two inside the phase comparison.

**On the per-frame simulation cost.** The pin changed subject in Phase 11 and
the threshold did not. It measured `MultiEndpointSpike.frame()` - two leg
reevaluations plus one composition - and now measures the marginal per-frame
cost of the production `Radar.simulate`, which additionally runs synthesis,
frame assembly and the receive chain. That is more work against the same
number, and it still passes: 4.5372 ms in isolation,
4.3822 ms inside the full `--gpu` suite, against 5.044 ms. The
Phase-8 note behind the 3.88 ms constant is preserved because it is the
provenance of the threshold: the Phase-7 report recorded 2.30 ms/frame, which is
not reproducible in this environment for a reason that is not a regression -
measured at the Phase-8 base commit `4bb059a` the same call cost **3.911 ms**
against **3.880 ms** at that HEAD, a ratio of 0.992.

**On the full-pipeline latency pin.** It got the same treatment in Phase 11
(warmup, synchronize on both sides, best of three) and its budget was likewise
not raised: 2.3911 ms against 2.899 ms.

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

Every number above is a wall-clock or allocation measurement of code that is in
the tree at the recorded commit. Where a claim is a ratio rather than an
absolute (the per-frame cost, the Channel reverse-pass surcharge), that is
deliberate: absolute wall times on an otherwise idle device drifted by up to
1.5x between processes in this environment, and a pin that cannot survive that
drift is a flake rather than a budget.

Timings depend on GPU model, CUDA toolkit, PyTorch build and driver. Use the
`--json` output of `tools/benchmark_processing.py` when reporting
hardware-specific numbers in release notes.
