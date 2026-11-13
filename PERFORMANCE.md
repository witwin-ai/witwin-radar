# Radar Performance

Status: current conclusions for witwin-radar 0.5.0 (Channel 0.6.0). Performance evidence is counts - Channel calls,
native launches, host synchronizations, device copies and CUDA kernels per frame - pinned by budget tests, plus
same-process timings labelled contended (the development GPU is shared). Superseded runs and the 2026-09 fused-CUDA
measurements are in [the performance history](docs/dev/archive/performance-history-2026-09.md).

## What 0.5.0 changes, in counts

`tests/test_session_budget.py` pins, per motion kind, every counter of a two-frame run of a small fixture (2 x 2
array, 2 chirps x 8 samples, one wall, `reflections=1`), measured in a fresh interpreter; Channel calls, native
launches, synchronizations, compositions and joins are exact, the three profiled columns hold to 0.5 %. The table
compares the 0.4.1 session (measured on Channel 0.6 before this release's session work) with the 0.5.0 pins, first
frame / marginal frame.

| Case | Counter | 0.4.1 | 0.5.0 |
|---|---|---|---|
| still world | Channel replays | 2 / 2 | 0 / 0 |
| still world | native launches | 3 / 3 | 2 / 0 |
| still world | host synchronizations | 160 / 4 | 45 / 0 |
| still world | CUDA kernels | ~935 / ~304 | 410 / 1 |
| moving point, `Motion.chirp()` | native launches | 12 / 12 | 3 / 3 |
| moving point, `Motion.chirp()` | host synchronizations | 554 / 524 | 79 / 41 |
| moving point, `Motion.adc()` | Channel replays | 64 / 64 | 0 / 0 |
| moving point, `Motion.adc()` | native launches | 96 / 96 | 3 / 3 |
| moving point, `Motion.adc()` | joins built | 32 / 32 | 1 / 0 |
| moving point, `Motion.adc()` | host synchronizations | 4222 / 4192 | 359 / 321 |
| moving point, `Motion.adc()` | CUDA kernels | 26333 / 26248 | 8421 / 8292 |
| moving structure, `Motion.adc()` | joins built | 32 / 32 | 1 / 0 |
| moving structure, `Motion.adc()` | host synchronizations | 5027 / 5024 | 1179 / 1152 |

What moved them:

- **Discovery rows are the primal rows.** An `ad_mode="none"` run reads each rediscovered instant's rows from the
  discovery itself: one Channel call per leg per rediscovery instead of three (discover, prepare, replay), bit-identical
  to the replay route.
- **A still world runs once.** Paths are evaluated and the waveform synthesized once per session; a later frame runs
  only the receive chain (one kernel when there is none).
- **Joins and pattern tables are reused** when a rediscovery finds the same topology, and the join is built with
  vectorized host code and one copy each way.
- **Grouped replay and synthesis.** Topology-identical instants of a frame replay together, and each group is
  synthesized in one launch over only the active transmitter's pairs; slot-batched replays stride one slot's tables
  natively instead of copying them per slot (point, aspect, tabulated and facet responses, directional patterns).
- **Fewer launches per composition.** Fields, responses and composed values cross the native boundary as real views
  (18 -> 12 kernels per composition), site-less rows are composed in the same join launch, and the isotropic 0 dBi
  default skips the antenna-pattern launch.

## The default for a moving scene

0.4.1 resolved `Motion.auto()` to per-ADC sampling whenever anything moved; 0.5.0 resolves a moving FMCW scene to
adaptive sampling. On a 48-site articulated body (one 2048-observation frame) the adaptive route evaluates 6
observations instead of 2048, at a relative cube difference of 3.6e-3 against `Motion.adc()`; one frame took 0.31 s
against 157 s for `Motion.adc()` (contended, same process). On the Doppler-precision fixtures it evaluates 3 to 14 of
128 observations, 2.9e-4 to 9.9e-3 relative to `Motion.adc()` (R-ADR-037). Because the adaptive phase test compares
float64 delays, the same motion takes the same number of probes at 5 m and at 100 m.

No contended A/B against an installed 0.4.1 was taken for this release: 0.4.1 needs Channel 0.5, so such a
comparison would mix two Channel versions. The evidence above is the budget-test counts and the same-process
comparison of the two `Motion` routes.

## Known limits

- **Topology discovery dominates heavy multipath.** A probe in a world that cannot be certified complete needs its
  own discovery. One discovery costs about the same for 1 and 1024 sites (a flat per-call cost), so packing would pay,
  but consuming a packed discovery needs several source/sink groups (or a pair mask) per Channel request, or
  slot-batched discovery; Channel 0.6 offers neither (R-ADR-033).
- **Exhaustive ADC sampling is a reference, not a real-time route.** It pays a discovery at every observation, so
  its cost scales with chirps x transmitters x ADC samples, and a moving mesh also pays compile work. `Motion.chirp()`
  freezes fast-time geometry; longer rediscovery cadences may miss path births. Result metadata records both.
- **A network has no adaptive route.** A moving `Network` resolves to `Motion.adc()`.
- **A walking delay has no closed-form range spectrum.** The spectrum family evaluates a delay that holds for the
  whole chirp as a Dirichlet kernel, one evaluation per bin; a walking delay is synthesized by the beat family in N per
  chirp and finished by the range transform in N log N. Derivatives take the term-by-term sum in both families.
- **Mesh targets cost a site per facet.** Each facet is a site with `N_tx + N_rx` leg rows; `MeshTargets(max_sites=)`
  or `cluster_size=` merges facets (on a 36,480-face sphere, 512 sites cut the composed rows 68x at no measurable
  RCS error). Clustering runs on the host at session start.
- **Profiled counts.** The CUDA profiler can drop records on a contended device, so the profiled budget columns keep
  a 0.5 % margin; every other counter is exact.

## Maintained benchmark

```bash
python tools/benchmark_processing.py --runs 200 --warmup 20 --json
python tools/benchmark_processing.py --groups pipeline --runs 10 --warmup 3 --json   # CI smoke
```

GPU pipeline budgets and measurement fixtures live in `tests/test_pipeline_budget.py`; the session budgets in
`tests/test_session_budget.py`; the backward budgets in `tests/test_backward_budget.py`. Those tests define the
executable thresholds; this document does not duplicate their constants.

## Measurement protocol

- Record GPU model, driver, CUDA toolkit/runtime, PyTorch version, Radar build fingerprint, and Channel build
  fingerprint.
- Warm up the exact route being timed; synchronize CUDA before and after each timed sample, or use CUDA events.
- Report the minimum (contended device) or the median (idle device) and the run count, never mixed in one table;
  label a contended measurement as contended and compare routes in the same process.
- Measure peak allocation around one call after warmup.
- Keep tensor shapes, scene and path counts, receive chain and detector choices identical when comparing routes, and
  report the path/site count and output domain with every number.
- Separate dispatch counts from wall time. A dispatch counter does not measure hidden synchronization or kernel
  duration.

## Current performance gates

Launch, host observation, synchronization, tape-byte and peak-allocation budgets run in the GPU suite; the absolute
wall-clock budgets are machine-dependent and run only when asked for, on an idle device:

```bash
pytest tests/test_session_budget.py tests/test_pipeline_budget.py tests/test_backward_budget.py --gpu -q
pytest tests/test_pipeline_budget.py tests/test_backward_budget.py --gpu --benchmark -q
```

The manually dispatched GPU workflow runs the CUDA tier and the maintained processing pipeline smoke benchmark. Its
existence does not prove a GPU result until the workflow actually runs and the artifact/log is retained.

## Native processing policy

Signal processing is Torch library calls (FFTs, matrix products) except where a stage has a per-element hot loop:
the ordered-statistic CFAR, DBSCAN clustering and back-projection run native kernels on CUDA tensors, each with its
launch budget, and back-projection with VJP/JVP companions. Moving another stage native needs a measured bottleneck
and a before/after count.
