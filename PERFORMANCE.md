# Radar Performance

Status: current conclusions, from local measurements recorded 2026-09-16 to 2026-09-19 on one RTX 5080 desktop;
release-platform benchmarks remain separate. Superseded runs, the retracted 2026-09-18 latency tables and the
Doppler-repair measurements are in [the performance history](docs/dev/archive/performance-history-2026-09.md).

## Motion sequence to complete ADC: fused CUDA (2026-09-19)

The current adaptive FMCW route fuses compact probe interpolation and dechirped
ADC synthesis in CUDA. It retains the same float32 boundaries, double phase
evaluation, serial path accumulation, fixed observation schedule and first-order
VJP/JVP. It does not generate RF transmit/receive chirps. Dirichlet remains the
constant-delay direct-spectrum route; dynamic ADC synthesis is a different
output/model contract, not a replacement of Dirichlet.

The full SMPL benchmark remains 128 material sites, 3 TX x 4 RX, 128 chirps x
256 samples, unchanged adaptive tolerances, and 300 frames over a 30 s walk
within 15 m. Beat output includes every complex ADC sample; diffusion, setup,
file output and all DSP remain outside timing. Other desktop GPU applications
remain running. The control is the **previously optimized implementation**,
interleaved with the new implementation at every frame of this same run.

| Completed-CUDA measurement | Previous optimized control | Fused CUDA |
| --- | ---: | ---: |
| Full 300-frame simulation | 39.143 s | **14.275 s** |
| Median frame | 127.96 ms | **45.18 ms** |
| P95 frame | 153.83 ms | **67.01 ms** |
| Throughput over the full sequence | 7.66 frames/s | **21.02 frames/s** |
| Synthesis batches per frame | 49 | **2** |
| Maximum incremental Torch allocation, paired single frames | 151.33 MiB | **34.33 MiB** |

The incremental sequence speedup is **2.74x**. Maximum ADC relative L2 difference
is **zero** across all 300 paired frames; adaptive probe counts match throughout.
All 15 paired standalone frames are bit-identical. The small exhaustive ADC
oracle remains at **0.2323%** adaptive error. Neither motion fidelity nor
sampling tolerance was reduced. This fixture now exceeds its 10 Hz target;
this is not a throughput guarantee for denser sites, multipath or phase noise.

Nsight Systems correlates the echo's CUDA launches with its NVTX range:
**2,371 -> 27 kernels**, **79.80 -> 12.44 ms** cumulative device kernel time in
one instrumented frame. The removed gather alone cost 48.23 ms. These profiler
numbers are separate from clean sequence timings. Synchronized stage timing
places the world half at 60.26 ms, including 15.86 ms SMPL and 11.80 ms Channel;
echo is 13.40 ms. World sampling and propagation now dominate that frame.

Oscillator phase noise retains separate interpolation and synthesis because a
receiver effect intervenes. The native boundary is **ABI 10**, with 41 operators
and a locally rebuilt developer binary. Both paired variants use that binary;
this comparison isolates compact fusion versus the previous expanded route,
not independently loaded historical DLLs. See the
[CUDA fusion report](docs/dev/audit/radar-adc-cuda-fusion-2026-09-19.md).

## Known limits

- **Range against the phase tolerance.** One float32 step of a position at range R is `R * 2^-23` metres, which is
  `4*pi*R*2^-23/lambda` radians of two-way phase; at 77 GHz that reaches the default 0.02 rad adaptive tolerance
  near 50 m. Past it no interval can be certified and the adaptive controller bisects toward exhaustion: measured
  probe counts on one frame at 6, 20, 40, 55, 70 and 78 m were 27, 27, 29, 31, 38 and 162. Peak allocation is flat
  in the frame count of a streamed sequence, but not in the probe count.
- **Topology discovery dominates heavy multipath.** A probe in a world that cannot be certified complete needs its
  own discovery; in the three-wall fixture those discoveries were 69% of the frame. Packing probes into one
  discovery would pay roughly 4x on discovery, but consuming a packed discovery needs a pairing restriction on the
  Channel request or a way to split a prepared topology by endpoint, and Channel offers neither. See
  [the discovery-batching blocker report](docs/dev/audit/radar-discovery-batching-blocker-2026-09-17.md).
- **Exhaustive ADC sampling is a reference, not a real-time route.** It pays discovery and replay at every
  observation, so its cost scales with chirps x transmitters x ADC samples, and a moving mesh also pays compile
  work. Explicit chirp sampling freezes fast-time geometry; longer rediscovery cadences may miss path births. Those
  trade-offs are recorded in result metadata.
- **A walking delay has no closed-form range spectrum.** The spectrum family takes no delay rate and refuses one by
  name. Evaluating that DFT term by term would cost N per bin, where the beat family synthesizes the same
  linear-delay model in N and the range transform finishes the axis in N log N. A delay that holds for the whole
  chirp collapses to a Dirichlet kernel, one evaluation per bin, with no samples materialised and no transform.
  Derivatives take the term-by-term sum in both families, so a spectrum backward costs N per bin where its forward
  costs one.

## Maintained benchmark

The maintained DSP benchmark is:

```bash
python tools/benchmark_processing.py --runs 200 --warmup 20 --json
```

For a shorter CI smoke:

```bash
python tools/benchmark_processing.py --groups pipeline --runs 10 --warmup 3 --json
```

GPU pipeline budgets and measurement fixtures live in `tests/test_pipeline_budget.py`. FMCW spectrum/beat correctness and domain-routing coverage live with the FMCW spectrum tests and processing-axis tests. Those tests define executable thresholds; this document does not duplicate their numeric constants.

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

The repository enforces performance through executable tests, not through copied prose numbers. Launch, host
observation, synchronization, tape-byte and peak-allocation budgets run in the GPU suite; the absolute wall-clock
budgets are machine-dependent and run only when asked for, on an idle device:

```bash
pytest tests/test_pipeline_budget.py tests/test_backward_budget.py --gpu -q
pytest tests/test_pipeline_budget.py tests/test_backward_budget.py --gpu --benchmark -q
```

The manually dispatched GPU workflow runs the CUDA tier and the maintained processing pipeline smoke benchmark. It publishes no wheel. Its existence does not prove a GPU result until the workflow actually runs and the artifact/log is retained.

## Native DSP policy

Signal processing remains PyTorch-owned. Moving a DSP stage into the native extension requires a measured bottleneck, a documented owner decision, forward/backward policy where relevant, and a before/after result from the rebaseline matrix above. The concept-axis consolidation alone is not performance evidence for such a move.
