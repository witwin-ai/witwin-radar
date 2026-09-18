# Radar Performance

Status: local Doppler repair measurements recorded on 2026-09-16; release-platform benchmarks remain separate.

## What changed

FMCW defaults to a normalized range spectrum. Stationary rows use native Dirichlet synthesis; linear moving rows include continuous fast-time phase, and refreshed dynamic scenes evaluate each ADC observation. The synthesized beat-signal route is explicit with `output_domain="beat"`. That changes the default pipeline's work: spectrum input must not pay a second range FFT, while beat input still does.

For that reason, pre-consolidation latency, FFT-count, launch-count, and allocation tables are not presented as current evidence here. They measured the former default beat pipeline and deleted module layout. They remain available in repository history, but must not be copied into release notes for the spectrum-first implementation.

## Adaptive motion control (2026-09-17)

Two changes to how the adaptive route chooses its partition, measured on RTX 5080 / Ryzen 7
9800X3D in witwin2. The run starts from the coarsest partition the probe-spacing bound allows
instead of bisecting down to it, and an accepted interval may interpolate through
`nodes` samples instead of two. Neither change touches the phase or amplitude
tolerance, and neither relaxes the bound.

An earlier revision of this work DID relax that bound for a topologically certified family, on the
argument that it only guarded against path births. That was wrong and is reverted: every tolerance
here is checked by sampling, motion periodic at the probe grid's step is invisible to all of them,
and this bound is what sets that step. See
[the correction](docs/dev/audit/radar-probe-spacing-bound-correction-2026-09-17.md), which carries
the reproduction - a grid-periodic 0.4 mm motion passing the phase test at 0.0095 rad with 0.87
relative IQ error.

Against the exhaustive per-ADC route in the same session, `tools/validate_adaptive_motion.py`:

| Fixture | Probes before | after | IQ relative L2 before | after | Speedup before | after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| radial point | 193 | 97 | 2.601e-4 | 2.80e-4 | 2.93x | 120.4x |
| rotor point | 193 | 97 | 2.014e-4 | 2.26e-4 | 2.91x | 119.1x |
| two articulated points | 193 | 97 | 2.150e-4 | 2.49e-4 | 3.08x | 92.5x |
| heavy multipath | 37 | 19 | 5.659e-4 | 5.68e-4 | 64.14x | 141.3x |

Accuracy is unchanged within 0.3e-4; the speedup comes from the initial partition, not from
sampling the motion any more coarsely.

`nodes` defaults to 2, the linear rule. Raise it only where the phase test, not the
bound, is what shortens an interval. Measured per frame at 2/3/5 nodes: an 80 Hz rotor on a
4.096 ms frame gives 38/21/25 probes and 36.0/22.2/21.7 ms, while a 24.96 ms MIMO frame gives
27/53/105 probes and 95.5/166.7/173.5 ms for one unchanged 13-interval partition.

`tools/validate_adaptive_tolerance.py` records what a tolerance buys. Realized IQ relative L2
divided by the largest per-path phase residual the controller tested measured **0.44 to 1.19**
across both fixtures and six tolerances; the ratio exceeds one at the tightest setting, where 126
interval errors add in the coherent sum rather than averaging down. Set the tolerance to the
target IQ error rather than above it.

### Probe efficiency, and why there is no warm start

`tools/validate_adaptive_probe_efficiency.py` measures the adaptive probe count against the
floor a published partition cannot go below: `2*(nodes-1)*intervals + 1`, because each accepted
interval needs its own nodes and the instants between them. Everything above that floor went to
a rejected interval, and that excess is the whole budget a cross-frame partition warm start
could recover.

| Fixture | Frames | Tolerance | Probes | Partition floor | Recoverable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three-wall multipath, 32 x 64 | 3 | 0.02 | 69 | 69 | 0 (0.0%) |
| Rotor point, 128 x 128 | 4 | 0.02 | 171 | 160 | 11 (6.4%) |
| Same rotor, tight tolerance | 2 | 0.002 | 2173 | 2064 | 109 (5.0%) |

The probe cache already shares a rejected interval's grid with its children, so no probe is paid
twice. A warm start would still pay the same floor, for at most 6.4%, while carrying a stale
partition across frames; it is therefore not implemented. Recorded in
[the warm-start headroom report](docs/dev/audit/radar-adaptive-warm-start-headroom-2026-09-17.md).

### Topology discovery is the remaining heavy-multipath limit

A probe in a world that cannot be certified complete needs its own topology discovery: that is
what a probe is. In the three-wall fixture those discoveries are 69% of the frame, spent across
two Channel calls per probe. `tools/validate_discovery_batching.py` measures whether that is
per-call overhead by packing P endpoint pairs into one discovery, which Channel evaluates as the
full P x P cross product because `PropagationRequest` carries no pairing restriction.

| Packed probes | Pairs solved | One call | P separate calls | Batching speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 9.49 ms | 9.49 ms | 1.00x |
| 8 | 64 | 21.94 ms | 75.88 ms | 3.46x |
| 19 | 361 | 43.48 ms | 180.22 ms | 4.15x |
| 38 | 1444 | 79.47 ms | 360.45 ms | 4.54x |

Fixed overhead is about 9.4 ms per call against a 0.05 ms marginal cost per pair, so batching
would pay roughly 4x on discovery and 2.4x end to end on that frame even paying the cross
product. It is not implemented: consuming a packed discovery per probe needs either a pairing
restriction on the request or a way to split a `PreparedFixedTopology` by endpoint id, and
Channel offers neither. Per-probe path families are ragged, which is exactly what the probe
detects, so Radar cannot rebuild them from a packed handle. Recorded, with the required
capability spelled out, in
[the discovery-batching blocker report](docs/dev/audit/radar-discovery-batching-blocker-2026-09-17.md).

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
| 8 | 73.2 MiB | 106.2 MiB | 1.45x | 109.9 | 246.9 | 0 |
| 32 | 73.2 MiB | 303.6 MiB | 4.15x | 112.1 | 108.1 | 0 |
| 128 | 73.2 MiB | 1233.4 MiB | 16.85x | 115.2 | 111.3 | 0 |

All sizes are MiB. Streamed peak allocation is flat; stacked peak tracks the frame count and
reaches 1233.4 MiB against a 384 MiB cube at 128 frames - about 3.2x the cube - because the frame
list and `torch.stack` are both live. The 8-frame stacked figure is a warm-up outlier. Per-frame
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

The retained MATLAB CPU rotor result was 250.7 ms and the optimized WiTwin rotor 608.7 ms, about
2.43x slower. That comparison is superseded: both sides were re-executed on 2026-09-17 after the
adaptive-control work and WiTwin measured 270.99 ms against MATLAB's 211.86 ms on the same
fixture at the default order, or 174.28 ms with `nodes=5`. See the rerun section
below. MATLAB was not rerun in THIS optimization pass, and
different precision, devices and fractional-delay models remain as the comparison report states.

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

### Actual MATLAB comparison, rerun (2026-09-17)

Both sides re-executed in one session on RTX 5080 / Ryzen 7 9800X3D against MATLAB R2025b
Update 4 with Radar Toolbox and Phased Array System Toolbox 25.2, using the same fixtures,
sampling parameters, alignment rule and repeat counts as the 2026-09-16 run.

That report concluded the public dynamic scene entry was 21 to 53 times slower than MATLAB's
analytic point-target entry. It is now faster on four of the six scenes and slower on the two
rotor scenes, at the default `nodes=2`:

| Scene | WiTwin median | MATLAB median | MATLAB / WiTwin | WiTwin on 2026-09-16 |
| --- | ---: | ---: | ---: | ---: |
| static | 7.66 ms | 23.11 ms | 3.02x | 14.01 ms |
| acceleration | 103.31 ms | 248.20 ms | 2.40x | 6478.87 ms |
| limbs | 206.19 ms | 290.18 ms | 1.41x | 11871.91 ms |
| static_os4 | 15.18 ms | 33.89 ms | 2.23x | accuracy-only |
| rotor | 270.99 ms | 211.86 ms | 0.78x | 13247.48 ms |
| rotor_os4 | 446.48 ms | 303.37 ms | 0.68x | accuracy-only |

The rotor is the micro-Doppler case where the phase test, not the probe-spacing bound, shortens
the intervals, which is what `nodes` is for. With
`--interpolation-nodes 5` the same rotor scene measures 174.28 ms, 1.22x faster than MATLAB;
`rotor_os4` stays slower at 537.14 ms because its 524288 observations per frame are dominated by
ADC synthesis rather than by probes. The default stays at 2 because the other fixtures are
fastest there.

Accuracy against the independent continuous-delay oracle is within 0.3 percentage points of the
2026-09-16 run - acceleration 0.814% to 0.763%, rotor 0.721% to 0.800%, limbs 0.564% to 0.825% -
all inside the 0.02 rad tolerance. Ground four-path multipath moved the same way, 0.511% to
0.762% and 0.486% to 0.682%. WiTwin is not uniformly more accurate: at four-times oversampling
MATLAB measures 0.291% against 0.819%.

Materials are unchanged at 1.9348e-6 maximum complex absolute error over 360 combinations, and
known static paths to beat IQ measured 0.206-1.533 ms against MATLAB CPU double at
12.99-2167.10 ms. Those two routes were not modified. All of these are different native
precisions and devices, not a same-hardware algorithm speedup.

See [the 2026-09-17 comparison](docs/dev/audit/radar-matlab-comparison-2026-09-17.md); the
[2026-09-16 report](docs/dev/audit/radar-matlab-material-motion-performance-2026-09-16.md) is
retained as the historical record of the implementation it measured. The heavy 64.14x result
below is against WiTwin's own ADC reference.

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
