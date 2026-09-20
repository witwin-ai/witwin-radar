# Motion sequence to complete ADC: optimization and acceptance

Historical first optimization pass. The subsequent
[CUDA fusion report](radar-adc-cuda-fusion-2026-09-19.md) supersedes its current
performance and native ABI descriptions. Preserve this run's measurements as
its own paired comparison; desktop load differs between runs.

Measured locally on 2026-09-19, Windows / RTX 5080 / Torch 2.10.0 / CUDA runtime
12.8. Other desktop GPU applications remained running, as requested.

## Scope and fixed workload

The timer starts from a fixed motion sequence and ends at the full complex ADC
cube. It includes live SMPL, trajectory placement, adaptive probes, Channel
propagation, round-trip composition, scattering, antenna weighting, interpolation
and waveform synthesis. It excludes diffusion/SMPL fitting, setup, file output,
range FFT and Range-Doppler. `output_domain="beat"` is explicit.

The frozen asset is `tools/fixtures/smpl_walk_genesis_seed10.npz`, SHA-256
`203c690f6b03d12f4b39d0aea9668907b9fb26e2290629bd8faebcc340d38329`.
The full neutral model has 6,890 vertices and 13,776 faces. It is evaluated at
every probe. There are 128 fixed area-sampled material sites, total scalar RCS
1 m², 3 TX x 4 RX, 128 chirps per transmitter and 256 samples per chirp.
Each frame produces 393,216 complex samples and 50,331,648 path contributions.

The smooth 30 s return route stays within 15 m: measured vertex ranges are
2.7516–13.3279 m. The 40 MHz/us slope gives a 16.4886 m unambiguous range.
The complete run covers 300 frames at an authored 10 Hz. Adaptive settings
remain phase error 0.02 rad, relative amplitude error 0.02, maximum interval
0.002 s and two interpolation nodes. No sample count, site count or tolerance
was reduced to obtain the speedup.

The physical scene remains LOS scalar material-point scattering. Body
self-occlusion, room multipath and a receiver/noise/quantization chain are not
part of this benchmark. The root route is authored and the generated joint clip
is smoothly looped; this is not a measurement-calibrated human radar model.

## Implemented stages

1. **SMPL model evaluation.** `smpl.py` computes shape blending once, constructs
   all local joint matrices together, retains parent indices on the model's
   device, and clones the already resident face tensor. Returning an owned face
   tensor prevents caller mutation from corrupting the shared model. Live
   pose/shape gradients and forward-mode vertex velocity remain supported.

2. **Observation preparation.** The instrument half converts compact probe
   values to double precision once per frame, prepares node starts and ADC
   clocks once, derives the observation from the existing receiver segment,
   and reuses that CSR inverse in waveform synthesis. It does not create a
   second route for `trace`/`echo`, `simulate` or `stream`.

3. **Fixed-motion adapter scheduling.** `WalkingBody` captures the complete
   SMPL/world-placement expression in a CUDA Graph. Each query uploads pose and
   time in one pinned buffer, replays all vertices, and clones the output so
   the next query cannot overwrite retained vertices. There is no precomputed
   vertex animation. This serial, inference-only adapter belongs to the
   benchmark; arbitrary public trajectories are not silently graph-captured.
   Its analytic velocity diagnostic continues through the eager differentiable
   expression. Measured setup, including graph capture, was 0.227 s.

4. **Native indexed interpolation and bounded batching.** The existing native
   interpolation family reads compact (delay, real coefficient, imaginary
   coefficient) triples through node indices. Packed controller probes and
   indexed ADC rows share one templated transport equation. Native backward
   and JVP remain present; backward sums repeated probe cotangents in the
   Torch layout boundary. The 32 MiB index-plus-basis budget permits 49 batches
   instead of 193 at 128 sites. Higher interpolation order lowers the row cap.
   This is a throughput/memory trade, not a claim of lower total peak memory.

The binding schemas and registry use ABI 9. All 38 operators resolved on the
freshly built Windows developer library. Its retained build fingerprint is
`cb97613ee80ed798e3015eb975989eb0480f03f71eb309f07ea247388f6de725`.
No Linux wheel or release matrix was executed by this optimization run.

## Paired end-to-end acceptance

The harness alternates baseline/current order every frame. Both streams see
the exact same motion, surface layout, radar configuration and timestamps.
Every timed `next(stream)` has CUDA synchronization on both sides. Validation,
sample export and diagnostic profiling happen outside those timers.

The baseline is a frozen copy of the pre-change `smpl.py`, `simulation.py`
and walking adapter under `output/adc-optimization/baseline/`. Both variants
use the rebuilt native DLL, through its packed and indexed modes respectively;
the comparison does not load two conflicting Torch dispatcher libraries.
Source digests, model digest, Channel identity and native identity are retained
in the environment/results records. The native indexed layout may change
last-bit rounding; exact old/new equality is not claimed for the final stage.

| Full 300-frame measurement | Before | Current |
| --- | ---: | ---: |
| Completed simulation | 148.324019 s | **45.399347 s** |
| Mean frame | 494.413 ms | **151.331 ms** |
| Median frame | 459.412 ms | **147.547 ms** |
| P95 frame | 745.832 ms | **182.489 ms** |
| Maximum frame, no removal | 1733.545 ms | **593.764 ms** |
| Throughput | 2.02 frames/s | **6.61 frames/s** |
| Batches/frame | 193 | **49** |

Total-time speedup: **3.2671x**. Computing 30 s of scene time still takes 45.40 s;
this does not meet a sustained 10 Hz real-time target on the measured desktop.
The additional 15 paired standalone calls have medians 510.45 / 167.19 ms.
Those medians are a separate measurement, not a replacement for the full stream.

Maximum old/new ADC relative L2 error over the 300 frames is **4.96375e-7**.
All frames are finite and nonzero, and every frame has the same probe count
as the baseline. Full vertices at 31 route instants match exactly; a retained
vertex tensor also stays unchanged across graph replays. The small exhaustive
ADC oracle gives relative L2 **0.0023231944** (0.2323%), and analytic vertex
velocity versus central differences gives **0.0003810705** (0.0381%).
The loop's pose and pose-rate jumps are zero.

The paired standalone measurements show maximum incremental Torch CUDA memory
of approximately **90 MiB baseline / 151 MiB current**. Larger batches spend
roughly 61 MiB to reduce launch overhead. The paired streaming process peaks at
**504.98 MiB**, which includes both variants and retained comparison state;
it must not be labeled the optimized simulator's standalone peak, or the
desktop's total VRAM use. The unpaired 300-frame run is separately retained in
`output/adc-optimization/final/` and is not the denominator of the speedup.

## Remaining measured bottleneck

A separate synchronized instrumentation pass at t=2.3 s totals 170.981 ms:

| Stage | Time | Interpretation |
| --- | ---: | --- |
| Echo | 107.043 ms | Row routing, native interpolation, payload conversion, synthesis, cube assembly |
| SMPL vertices | 16.561 ms | 53 complete mesh evaluations through graph replay |
| Channel discovery and replay | 14.324 ms | Two discovery and four replay calls |
| Other world-half work | 31.466 ms | Adaptive control, binding, composition and remaining bookkeeping |
| Remaining outer work | 1.587 ms | Session/result overhead |

The world half is 62.351 ms inclusive of SMPL and Channel. Its nested times
must not be added to it again. Echo remains about **63%** of this instrumented
frame. Geometry is no longer the dominant stage. The next substantial target
is fusing ADC row routing/interpolation/payload assembly with waveform work;
native waveform phase evaluation alone is not established as the bottleneck.

The Torch profiler trace and cProfile report corroborate the remaining many
indexing, copying and conversion operations. Instrumentation changes latency,
and Torch's Windows event attribution also charges time to metadata operations;
its summed CUDA totals are not reported as exclusive kernel wall time or as
a percentage of clean end-to-end time.

## Evidence and reproduction

Acceptance validation executed on the final implementation:

- `pytest tests/ --gpu -q -rs`: **1458 passed, 1 skipped**, 373.38 s. The skip
  is `test_native_coexistence_smoke.py`: this checkout has no nightly coexistence
  evidence. It is not claimed as passing; there are zero missing-Channel skips.
- All **18 non-CPU-test quick gates** passed, including full-tree Ruff formatting
  and lint, duplication, native binding registry, ownership/import rules,
  architecture, public API, documentation and release-claim checks.
- Native indexed interpolation tests include 2/5/9 nodes, repeated probe
  indices, an independent complex oracle, reverse gradients, JVP, empty queries
  and refusal of derivatives through the fixed partition.
- Existing adaptive reflection/scene AD tests and bit-identical
  `echo(trace(...))` / `simulate(...)` tests pass as part of that full run.
- SMPL tests retain the independent smplpytorch oracle, analytic velocity and
  pose gradients, and add returned-face ownership and live shape mutation.

Logs: `output/adc-optimization/full-gpu-tests.log`, `static-checks.log` and
`native-build.log`. The full quick tier's separate CPU coverage command was not
run; the statement above names exactly the static/import gates that were run.

![Paired latency and remaining stage cost](../../../output/adc-optimization/performance.png)

- `tools/benchmark_motion_adc.py`: reusable ADC-only driver, paired streams,
  vertex ownership check, full-frame comparisons and exhaustive small oracle.
- `output/adc-optimization/acceptance/results.json`: final summaries,
  environment identities, validation and inclusive stage timings.
- `output/adc-optimization/acceptance/frames.jsonl`: every paired frame.
- `output/adc-optimization/acceptance/paired.json`: standalone alternating runs,
  individual allocation increments and errors.
- `output/adc-optimization/acceptance/adc-0000.npz`, `adc-0150.npz`,
  `adc-0299.npz`: representative complete ADC cubes. All 300 cubes are computed;
  file I/O is intentionally outside the simulation timer.
- `output/adc-optimization/stage1/`, `stage2/`, `stage3/`, `stage4/`: interim
  paired evidence. Do not divide measurements taken in separate stages.
- `output/adc-optimization/batch-experiment.json`: alternating 1x/2x/4x row
  budget experiment; the first cold call is explicitly retained.
- `output/adc-optimization/gpu-trace.json`, `gpu-profile.json`, and
  `acceptance/profile.txt`: diagnostic traces, separate from clean timers.

PowerShell, after supplying the locally licensed neutral model:

```powershell
$env:PYTHONPATH='E:/Code/witwin-platform/channel;E:/Code/witwin-platform/radar'
$env:WITWIN_CHANNEL_DEVELOPER_OVERRIDE='1'
$env:WITWIN_CHANNEL_EXTENSION_PATH='E:/Code/witwin-platform/channel/build/cmake_rayd080/_channel.cp311-win_amd64.pyd'
$env:WITWIN_CHANNEL_EXPECTED_FINGERPRINT=(Get-Content ../channel/build/cmake_rayd080/_channel.build-fingerprint).Trim()
python tools/benchmark_motion_adc.py --baseline output/adc-optimization/baseline --rounds 3 --frames 300 --paired-stream --output output/adc-optimization/reproduction
```

Omit `--baseline` and `--paired-stream` for a current-only run. The generator
is not needed: the frozen motion is already in `tools/fixtures/`.

Rebuild the packaged native library after updating its sources:

```powershell
$env:WITWIN_RADAR_NATIVE_BUILD_DIR='E:/Code/witwin-platform/radar/output/adc-optimization/native-build'
$env:TORCH_CUDA_ARCH_LIST='12.0'
python -c 'import subprocess,os,sys; subprocess.run([sys.executable,"scripts/build_radar_cuda_prebuilt.py","--verbose"],env=dict(os.environ),check=True)'
```

The clean child environment collapses the duplicate `PATH`/`Path` entries in
this desktop shell before the normal build script initializes MSVC. There is
no loader bypass. This local build used installed NVCC 12.9 with the recorded
Torch CUDA 12.8 runtime; the sidecar's complete identity validated successfully.
