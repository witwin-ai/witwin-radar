# Radar Performance

Status: local Doppler repair measurements recorded on 2026-09-16; release-platform benchmarks remain separate.

## What changed

FMCW defaults to a normalized range spectrum. Stationary rows use native Dirichlet synthesis; linear moving rows include continuous fast-time phase, and refreshed dynamic scenes evaluate each ADC observation. The synthesized beat-signal route is explicit with `output_domain="beat"`. That changes the default pipeline's work: spectrum input must not pay a second range FFT, while beat input still does.

For that reason, pre-consolidation latency, FFT-count, launch-count, and allocation tables are not presented as current evidence here. They measured the former default beat pipeline and deleted module layout. They remain available in repository history, but must not be copied into release notes for the spectrum-first implementation.

## Adaptive motion control (2026-09-17)

Two changes to how the adaptive route chooses its partition, measured on RTX 5080 / Ryzen 7
9800X3D in witwin2. The probe-spacing bound is enforced only where a path birth is possible,
and an accepted interval interpolates through `interpolation_nodes` samples instead of two.
Neither change touches the phase or amplitude tolerance.

| Public scene entry | Probes/frame before | after | ms/frame before | after |
| --- | ---: | ---: | ---: | ---: |
| Rotor point, 128 chirps x 128 ADC, 4 MHz | 73-101 | 9 | 75.8-81.6 | 19.0 |
| MIMO walker, 3TX x 4RX, 128 chirps x 256 ADC | 65 | 9 | 133-148 | 128 |
| Three-wall multipath, 64 paths, 32 x 64 | 37 | 19 | 1580 | 332 |

Against the exhaustive per-ADC route in the same session, the four `validate_adaptive_motion.py`
fixtures now measure 196x (radial), 134x (rotor), 218x (two articulated points) and 129x (heavy
multipath), against 2.93x/2.91x/3.08x/64.14x for the 2026-09-16 tables below. Those older
absolute seconds came from a differently loaded desktop and are not comparable directly; the
speedup ratios are, because each is measured within one session.

Accuracy moves toward the declared tolerance rather than past it: at the default 0.02 rad the
rotor fixture goes from 8.5e-3 to 2.7e-3 relative IQ error while the two-point proxy goes from
2.2e-4 to 4.5e-3, both inside tolerance. `tools/validate_adaptive_tolerance.py` records what a
tolerance buys; realized IQ relative L2 measured 0.55 to 0.66 times the largest per-path phase
residual the controller tested, on single-dominant-path, null-free fixtures.

Evidence: [the interval-bound and tolerance report](docs/dev/audit/radar-adaptive-interval-bound-and-tolerance-2026-09-17.md)
and [the interpolation-order report](docs/dev/audit/radar-adaptive-interpolation-order-2026-09-17.md).

### Probe efficiency, and why there is no warm start

`tools/validate_adaptive_probe_efficiency.py` measures the adaptive probe count against the
floor a published partition cannot go below: `2*(nodes-1)*intervals + 1`, because each accepted
interval needs its own nodes and the instants between them. Everything above that floor went to
a rejected interval, and that excess is the whole budget a cross-frame partition warm start
could recover.

| Fixture | Frames | Tolerance | Probes | Partition floor | Recoverable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three-wall multipath, 32 x 64 | 3 | 0.02 | 69 | 69 | 0 (0.000%) |
| Rotor point, 128 x 128 | 4 | 0.02 | 36 | 36 | 0 (0.000%) |
| Same rotor, tight tolerance | 2 | 0.002 | 23511 | 23498 | 13 (0.055%) |

The probe cache already shares a rejected interval's grid with its children, so no probe is paid
twice, and the coarsest-initial-partition change removed the one systematically wasted level.
A warm start is therefore not implemented: it would still pay the same floor, for at most 0.055%,
while carrying a stale partition across frames. Recorded in
[the warm-start headroom report](docs/dev/audit/radar-adaptive-warm-start-headroom-2026-09-17.md).

## Long frame sequences

### Streamed against stacked frames (2026-09-17)

`tools/validate_frame_streaming.py` produces the same 3TX x 4RX, 128-chirp, 256-sample adaptive
walker session through both public entries in witwin2 on RTX 5080, after one warm-up sequence.
Peak allocation is measured around the run and is relative to the resident bytes before it.
Every frame is compared with `torch.equal` against the stacked cube's frame; a per-frame
comparison rather than one reduction over the whole cube, because the two summation trees differ
in the last float32 digits for no physical reason.

| Frames | Streamed peak | Stacked peak | Peak ratio | Streamed ms/frame | Stacked ms/frame | Mismatched frames |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 65.5 MB | 101.2 MB | 1.55x | 138.3 | 119.8 | 0 |
| 32 | 66.5 MB | 307.6 MB | 4.62x | 126.9 | 116.5 | 0 |
| 128 | 66.5 MB | 1214.5 MB | 18.26x | 149.9 | 161.9 | 0 |

Streamed peak allocation is flat; stacked peak tracks the frame count and reaches 1.21 GB for the
402.7 MB cube at 128 frames, because the frame list and `torch.stack` are both live. Per-frame
latency is the same route in both cases and the spread here is shared-desktop noise, not a
streaming penalty. This measures one LOS walker fixture: it does not establish a per-frame cost
for heavy multipath or moving meshes.

Reproduce: `python tools/validate_frame_streaming.py --frames 8 32 128`.
Evidence: `output/frame-streaming/results.json`.

## Doppler repair measurements

### Rotor scheduling optimization (2026-09-17)

Completed-GPU end-to-end timing in witwin2, RTX 5080 / Ryzen 7 9800X3D, with one warmup and
three measured calls to `Radar.simulate`. Same 77 GHz, 128 ADC samples, 1024 chirps, 80 Hz
orbital point target, empty LOS world, unit-RCS/antenna and 0.02 rad adaptive tolerance as the
MATLAB comparison. This is a complete public scene call, including discovery, rather than known-path synthesis.

| Scene | Median | Measured range | Independent ADC-oracle IQ relative L2 |
| --- | ---: | ---: | ---: |
| Rotor, before this optimization | 16.165 s | 14.870–19.294 s | 0.7207% |
| Rotor, optimized | 608.7 ms | 571.8–818.5 ms | 0.7207% |
| Accelerating point, optimized | 242.2 ms | 217.3–306.2 ms | 0.8132% |
| Two articulated point proxies, optimized | 519.9 ms | 442.9–599.3 ms | 0.5643% |
| Static point, 128 chirps | 19.46 ms | 18.16–19.55 ms | 0.0463% |

Rotor speedup is 26.56x against the same-session baseline; the full IQ relative change is
3.26e-8. The earlier optimized run measured 497–586 ms, median 566 ms; the final repeat above
is retained rather than selecting the faster run. Shared-desktop latency is not a real-time guarantee.
677 phase probes and 169 accepted intervals remain; discoveries fall from 677 to 1, and native
ADC synthesis calls from 512 to 4. The initial 263062 per-observation `full_like` calls disappear.

The three-wall, depth-two, 64-path fixture still requires 37 discoveries: one measured run took
1.356 s versus the exhaustive ADC reference's 96.321 s (71.0x), IQ error 0.0566%, RD power
error 0.00649%. Its native ADC synthesis uses one batch. Geometry-dependent discovery remains
the limit; hundreds of milliseconds are not established for arbitrary heavy multipath or moving meshes.

The retained MATLAB CPU rotor result is 250.7 ms. The optimized WiTwin rotor is still about 2.43x
slower for that simple point-target comparison; this work does not establish a general MATLAB speed advantage.
MATLAB was not rerun in this optimization pass. Different precision, devices, and fractional-delay models
remain as described in the original comparison report.

Reproduce: `python tools/validate_rotor_performance.py --output output/rotor-optimization/final
--baseline output/rotor-optimization/baseline --cases rotor acceleration limbs static` (one command).
Evidence: `output/rotor-optimization/final/acceptance.json`, the saved MAT cubes, and
[the optimization acceptance report](docs/dev/audit/radar-rotor-optimization-2026-09-17.md).
The following 2026-09-16 tables are historical pre-optimization measurements.

### Adaptive motion experiment (2026-09-16)

`tools/validate_adaptive_motion.py` compares complete public simulations in witwin2 on RTX 5080,
after warming native loading. The ADC and adaptive routes use identical waveform and scene inputs.
The heavy fixture has three reflecting walls, depth two per leg, 64 round-trip paths, 32 chirps and 64 ADC samples.

| Scene | ADC time | Adaptive time | Speedup | IQ relative L2 error | RD power relative L2 error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Radial point | 8.464 s | 2.888 s | 2.93x | 0.0002601 | 0.00004237 |
| Rotor point | 8.555 s | 2.935 s | 2.91x | 0.0002014 | 0.00006526 |
| Two articulated points | 8.853 s | 2.871 s | 3.08x | 0.0002150 | 0.00007299 |
| Heavy multipath | 101.330 s | 1.580 s | 64.14x | 0.0005659 | 0.00006488 |

Heavy discovery count falls from 2048 to 37; the other fixtures fall from 512 to 193.
These are single measured runs, not latency percentiles or a real-time guarantee. Adaptive controls use
0.02 rad phase tolerance, 0.02 relative amplitude tolerance and 2 ms maximum interval. Different motion,
topology churn, tolerances and batch sizes change both accuracy and cost. The ADC reference remains available.
Evidence: `output/doppler-repair/adaptive/results.json` and the saved full complex cubes.

### Actual MATLAB material/motion comparison (2026-09-16)

Actual R2025b Update 4 comparisons now distinguish known-path synthesis from the public scene entry.
On RTX 5080 / Ryzen 7 9800X3D, known static paths to beat IQ (including CUDA transfers) measured
0.241–2.252 ms versus MATLAB CPU double at 16.671–2637.869 ms. These are different native precisions
and devices, not a same-hardware algorithm speedup. MATLAB submitted full frames with NumRepetitions.

The public adaptive dynamic scene entry was slower: acceleration 6.479 s versus 0.302 s,
rotor 13.247 s versus 0.251 s, and two articulated point proxies 11.872 s versus 0.343 s.
The profiles identify 677 rediscoveries and 263062 per-observation full_like calls in one rotor frame.
Fast CUDA synthesis therefore does not establish an end-to-end dynamic-scene advantage.
Measurements used a shared desktop; medians and raw ranges are retained, not real-time guarantees.

See [the full accuracy, sampling-control, and performance report](docs/dev/audit/radar-matlab-material-motion-performance-2026-09-16.md)
and its JSON evidence index. The earlier heavy 64.14x result below is against WiTwin's own ADC reference.

### Earlier baseline measurements

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
