# Radar Performance

Status: local Doppler repair measurements recorded on 2026-09-16; release-platform benchmarks remain separate.

## What changed

FMCW defaults to a normalized range spectrum. Stationary rows use native Dirichlet synthesis; linear moving rows include continuous fast-time phase, and refreshed dynamic scenes evaluate each ADC observation. The synthesized beat-signal route is explicit with `output_domain="beat"`. That changes the default pipeline's work: spectrum input must not pay a second range FFT, while beat input still does.

For that reason, pre-consolidation latency, FFT-count, launch-count, and allocation tables are not presented as current evidence here. They measured the former default beat pipeline and deleted module layout. They remain available in repository history, but must not be copied into release notes for the spectrum-first implementation.

## Doppler repair measurements

Both checkouts used witwin2, Torch 2.10.0+cu128, and RTX 5080. The pre-repair
`e0c79ad` was exported into `output/doppler-repair/baseline`, rebuilt, and ran
the exact same `tests/test_phase8_pipeline_budget.py` measurement recipe.

| Measured route | Before repair | Repaired (isolated targeted run) |
| --- | ---: | ---: |
| Full DSP pipeline, best of three medians | 3.6963 ms | 3.1512 ms |
| Static scene, marginal frame | 8.66685 ms | 8.65955 ms |
| Pipeline peak allocation | 0.9404 MB | 0.9404 MB |

The old 2.899/5.044 ms limits fail on the unmodified baseline as well. Budgets
are re-derived from rounded baseline measurements (3.70/8.67 ms) with the same
1.30 factor; exact operation counts and memory limits are unchanged. These are
local environment measurements, not a claim of a general speed improvement.
Logs: `output/doppler-repair/baseline-performance.log` and
`output/doppler-repair/stage4-migration.log`.

Dynamic ADC sampling deliberately pays discovery/replay at every observation.
It is a correctness reference, not a real-time implementation. The cost scales
with chirps * transmitters * ADC samples, and a moving mesh also incurs compile
work. Explicit chirp sampling freezes fast-time geometry; longer rediscovery
cadences may miss path births. Those tradeoffs are recorded in result metadata.
The native finite-sum spectrum for nonzero linear delay rates costs O(N^2) per
path rather than the stationary Dirichlet O(N). `tools/validate_doppler_motion.py`
records actual scene timings together with independent IQ/STFT errors.

## Maintained benchmark

The maintained DSP benchmark is:

```bash
python tools/benchmark_processing.py --runs 200 --warmup 20 --json
```

For a shorter CI smoke:

```bash
python tools/benchmark_processing.py --groups pipeline --runs 10 --warmup 3 --json
```

GPU pipeline budgets and measurement fixtures live in `tests/test_phase8_pipeline_budget.py`. FMCW spectrum/beat correctness and domain-routing coverage live with the FMCW spectrum tests and processing-axis tests. Those tests define executable thresholds; this document does not duplicate their numeric constants.

## Required rebaseline matrix

A publishable performance record for the consolidated architecture must measure both FMCW domains on the same machine and software stack:

| Route | Required products | Required observations |
| --- | --- | --- |
| default spectrum | synthesis, range-profile domain conversion, Range-Doppler, detection, point cloud | median latency, peak allocation delta, native launches, FFT dispatches, host observations |
| explicit beat | beat synthesis, range FFT, Range-Doppler, detection, point cloud | the same observations and the delta versus spectrum |
| scene-driven simulation | Channel propagation, round-trip composition, synthesis, frontend | marginal per-frame latency, composed rows, compile/discovery counts |
| AD | supported FMCW forward/JVP/VJP paths | forward and backward latency, saved-tensor bytes, companion launch count |

At minimum, use the maintained small fixture and one realistic array/frame configuration. Report path/site count and output domain with every number; a latency without those two values is not reproducible.

## Measurement protocol

- Record GPU model, driver, CUDA toolkit/runtime, PyTorch version, Radar build fingerprint, and Channel build fingerprint.
- Warm up the exact route being timed.
- Synchronize CUDA before and after each timed sample, or use CUDA events with correct synchronization.
- Report medians and the run count; do not mix best-case and median values in one table.
- Measure peak allocation around one call after warmup.
- Keep spectrum and beat tensor shapes, scene/path counts, frontend, and detector choices identical when comparing routes.
- Separate dispatch counts from wall time. A dispatch counter does not measure hidden synchronization or kernel duration.
- Retain the JSON output as an artifact when a number is used for an acceptance or release claim.

## Current performance gates

The repository currently enforces performance through executable tests and workflow references, not through copied prose numbers:

```bash
pytest tests/test_phase8_pipeline_budget.py --gpu -q -s
python ci/check_workflow_references.py
```

The manually dispatched GPU workflow runs the CUDA tier and the maintained processing pipeline smoke benchmark. It publishes no wheel. Its existence does not prove a GPU result until the workflow actually runs and the artifact/log is retained.

## Native DSP policy

Signal processing remains PyTorch-owned. Moving a DSP stage into the native extension requires a measured bottleneck, a documented owner decision, forward/backward policy where relevant, and a before/after result from the rebaseline matrix above. The concept-axis consolidation alone is not performance evidence for such a move.