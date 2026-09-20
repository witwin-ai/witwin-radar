# Compact adaptive ADC CUDA fusion

Measured locally on 2026-09-19, Windows, RTX 5080, Torch 2.10.0, CUDA runtime
12.8. Other GPU desktop applications remained running as requested.

## Scope and model

The benchmark starts from a frozen Genesis-generated SMPL motion and computes
the full complex ADC cube. It excludes diffusion, setup, file output and DSP.
The body has 6,890 vertices and 13,776 faces; every probe evaluates the complete
SMPL expression through the existing inference-only CUDA Graph adapter.
There are 128 stable material sites, 3 TX x 4 RX, 128 chirps x 256 samples,
393,216 complex values per frame and 50,331,648 contributing path samples.
The sequence is 300 frames at 10 Hz over a 30 s loop, with body ranges
2.7516-13.3279 m. The scene is empty LOS, with no room multipath, self-occlusion,
oscillator phase noise or ADC quantization. Motion generation is not timed.

Adaptive settings remain 0.02 rad phase tolerance, 0.02 relative amplitude
tolerance, 0.002 s maximum interval and two interpolation nodes. This change
neither relaxes the controller nor replaces the mesh or motion.

The fixture hash is
`203c690f6b03d12f4b39d0aea9668907b9fb26e2290629bd8faebcc340d38329`.
Model, source and binary hashes are recorded in each `environment.json`.

## Why ADC still needs synthesis

Dirichlet directly evaluates the normalized DFT of a constant-delay chirp.
It remains the direct-spectrum route. The benchmark explicitly requests
`output_domain="beat"` and allows delay and transfer to vary within a chirp.
It directly evaluates the analytic **dechirped complex ADC** expression;
it does not sample RF transmit and receive chirps or simulate a mixer.
Motion probes provide the path delays and complex transfers. Interpolation
transports the carrier phase, synthesis evaluates each requested sample, and
path contributions add at the correct TDM receiver. A direct Dirichlet spectrum
would change either the requested output or the within-chirp motion model.

## Implementation and numeric invariants

`fmcw_adaptive_forward/backward/jvp` in `cuda/fmcw_beat.cu` consume compact
probe triples, per-observation node starts/basis, pair bounds and ADC clocks.
One thread accumulates the ordered paths of an observation/receiver segment.
It performs interpolation and dechirped beat evaluation without expanded
observation-by-path indices, coefficients, delays or waveform payloads.

`cuda/path_interpolation.cuh` owns the interpolation equation and Jacobian for
both standalone interpolation and fused ADC. Common helpers in `fmcw_beat.cu`
own observation evaluation and AD for both routes. The existing phase helper
remains the sole beat phase owner. No second production equation was introduced.

Interpolation and synthesis retain their existing float64 arithmetic and
float32 interface rounding, including the rounding in VJP/JVP. Channel transfer
is conjugated exactly once. Dead rows zero the coefficient at the same point.
Primal accumulation retains path order and double precision. Reverse uses
double atomics to accumulate shared probe gradients; its reduction order is
not deterministic. No fast-math flag or parallel primal reduction was added.

All four verbs use `_adaptive_echo`. Oscillator phase noise keeps the expanded
interpolation/effect/synthesis boundary, because a receiver operation intervenes.
There is no AD-dependent primal route. Native operations use the current CUDA
stream and device guard and reject incompatible payload types and shapes.
The public API and motion defaults are unchanged.

The native interface is ABI 10 with 41 operators. Developer binary fingerprint:
`dd2b4ca0cff28a425bdab853221d2ba12197d6ade5ce618be08e870a67a2be74`.
Required Channel fingerprint:
`183118e96d75856e71df2f90c621fa191d6151d2b63ec12fd5f71062e49ed73f`.

## Paired acceptance measurements

The control is the prior optimized source snapshot in
`output/adc-fusion/baseline/`, not the original unoptimized implementation.
Both variants use the same newly built native library; the control uses its
indexed interpolation and CSR synthesis, while the candidate uses fusion.
This controls the dataflow change, not historical DLL identity.

`tools/benchmark_motion_adc.py` alternates execution order at every frame,
synchronizes completed CUDA work, retains every frame's timing and checks
the ADC result and probe count. Fifteen standalone pairs precede the stream.

| Metric | Previous optimized control | Fused CUDA |
| --- | ---: | ---: |
| 300-frame completed simulation | 39.143276 s | 14.274506 s |
| Median frame | 127.960700 ms | 45.182700 ms |
| P95 frame | 153.827425 ms | 67.012725 ms |
| Maximum frame | 234.175900 ms | 85.191000 ms |
| Sequence throughput | 7.664 frames/s | 21.016 frames/s |
| ADC synthesis batches | 49 | 2 |
| Maximum incremental Torch allocation in standalone pairs | 151.329590 MiB | 34.326172 MiB |

Sequence speedup is **2.74218x**. Peak incremental allocation falls **77.3%**.
The paired process's full-stream peak of 514.096 MiB includes both variants,
both bodies and retained session state; it is not fused-only memory or total
desktop VRAM. Shared desktop load explains variation between smoke,
acceptance and historical runs; do not multiply unrelated speedup ratios.

All 300 relative ADC L2 differences are zero and probe counts agree. The
15 standalone pairs pass `torch.equal`. All body vertices at 31 instants
match exactly and retained outputs survive later graph replays. The small
64-observation exhaustive ADC oracle has relative error 0.0023231944 (0.2323%),
unchanged from the previous implementation. This is adaptive-model error,
separate from optimization equivalence. Velocity versus central differences
remains 0.0003810705 relative L2. Loop pose and pose-rate jumps are zero.

## Profiler evidence and remaining costs

`tools/profile_motion_adc.py` warms both implementations, brackets two frames
with CUDA profiler start/stop, and labels full simulation and echo with NVTX.
Nsight Systems 2025.1.3 records CUDA activity. Device kernels are assigned to
echo by their CUDA runtime correlation IDs and the launching NVTX range.

| Echo activity, one instrumented frame at t=2.3 s | Control | Fused |
| --- | ---: | ---: |
| Device kernel launches | 2,371 | 27 |
| Summed device kernel duration | 79.800860 ms | 12.442843 ms |
| Path interpolation launches | 49 | fused |
| CSR synthesis launches | 49 | fused |
| Fused interpolation/synthesis launches | 0 | 2 |

The old gather kernels alone cost 48.234562 ms; standalone interpolation and
synthesis cost 8.202335 and 7.009877 ms. The fused kernels cost 12.393627 ms.
Most benefit therefore comes from eliminating expanded routing and payload
movement, not a cheaper physical equation. The host echo NVTX duration is
89.647905 vs 1.366969 ms; the latter is submission time, **not completed GPU
time**. Nsight capture overhead and enabled CUDA event tracing mean these
instrumented numbers must not replace the clean paired latency measurement.

A separate synchronized stage profile measures 75.318600 ms for one frame:
world 60.259000 ms, echo 13.402600 ms, with SMPL 15.857900 ms and Channel
discovery/replay 11.798300 ms included in the world total. The remaining
32.602800 ms in the world half includes orchestration, site binding, topology,
composition, scattering and adaptive-control work. These inclusive timings
must not be added to their parent or substituted for streaming medians.

Further opportunities are batched world probes, fewer Python/host transitions
and repeated topology/metadata validation, and tuning the remaining fused
kernel. Nsight Compute counters were not collected; this report does not
claim a verified FP64, bandwidth or occupancy bottleneck. Dense sites,
multipath, phase noise and full-frame gradient throughput were not benchmarked
here. The current forward LOS fixture exceeds its 10 Hz target.

## Evidence and reproduction

The full GPU-enabled suite executed: **1466 passed, 2 failed, 1 skipped** in
338.40 s. Both failures were explicit inventory counts: the new autograd
owner changes 11 to 12 backwards and 22 to 24 saved-tensor reads. Updating
those inventories preserved their decorator and tape-containment assertions.
The final targeted run passed **98 tests**, covering both corrected inventories,
all fused native primal/VJP/JVP tests, negative interpolation weights, fixed
schedule refusals in both AD modes, non-default streams, empty rows, batch
cuts, path interpolation, CSR synthesis, and the trace/echo split (including
phase noise). Four schedule-refusal cases were added after full-suite collection
and are included in that targeted run. The complete suite was not rerun after
the inventory-only fixes. No numeric regression remained unresolved.

The one skip is the nightly coexistence evidence check at
`tests/test_native_coexistence_smoke.py:199`; no missing-Channel integration
skip occurred. All 18 non-CPU-test quick gates passed after Ruff formatting
was corrected (17 in `static-final.log`, final format in `format-final.log`).
The quick CPU coverage run and Linux release cells were not executed.

Raw local evidence is retained under `output/adc-fusion/`:

- `acceptance/environment.json`, `results.json`, `paired.json`, `frames.jsonl`
  and ADC/oracle NPZ files: complete timing and numerical evidence.
- `native-build.log`: developer native rebuild and stamped identity.
- `paired-nsys.nsys-rep`, `paired-nsys.sqlite`, `nsys-stats_*.csv` and
  `nsys-analysis.json`: profiler capture and correlation analysis.
- `unit.log`, `full-suite.log`, `targeted-final.log`, `static-final.log`,
  `format-final.log`, `ruff-final.log`: validation logs.

With the recorded Channel developer binary selected in the environment:

```powershell
python tools/benchmark_motion_adc.py --baseline output/adc-fusion/baseline --rounds 3 --frames 300 --paired-stream --output output/adc-fusion/acceptance
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output=output/adc-fusion/paired-nsys python tools/profile_motion_adc.py --baseline output/adc-fusion/baseline
pytest tests/ --gpu -q -rs -p no:cacheprovider
```

Choose fresh output paths for another run; the benchmark refuses to overwrite
existing timing evidence. No Linux release wheel or remote workflow is claimed.
