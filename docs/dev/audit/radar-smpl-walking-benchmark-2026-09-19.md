# Real SMPL walking benchmark within 15 m (2026-09-19)

Historical pre-optimization measurement. The current motion-to-ADC implementation
and paired acceptance measurements are recorded in
[the optimization report](radar-motion-adc-optimization-2026-09-19.md).
This report's spectrum-output timings were taken under different desktop load;
they are not a denominator for current speedup claims.

## Workload and provenance

This benchmark uses an actual Genesis-generated SMPL motion and the complete
neutral SMPL mesh: 6,890 vertices and 13,776 faces. It is not the single-point
`walker` in `benchmark_adaptive_frame.py`, nor the two-point limb oracle.

The runtime behind the `wt_genesis` plugin generated the prompt
`a person walks forward at a natural steady pace, swinging both arms`, seed 10,
6 seconds, 20 Hz, 120 poses, guidance 2.5, 20 SMPLify iterations. The existing
licensed model and MDM assets were loaded from
`E:/Code/Vibe/Kimodo/MDM_TINY/data`. Generation plus fitting took **154.191 s**.
This one-time cost is measured separately from radar simulation; the saved
motion is reused on every benchmark run.

Motion SHA-256:
`203c690f6b03d12f4b39d0aea9668907b9fb26e2290629bd8faebcc340d38329`.
The fixed motion and its generation record are checked in as
`tools/fixtures/smpl_walk_genesis_seed10.npz` and the adjacent JSON. Runtime
stdout and the local model copy are under `output/smpl-walk/`. Licensed model
files are not committed or redistributed.

The radar configuration derives from the existing 77 GHz `MIMO` fixture: 3 TX, 4 RX,
128 chirps per transmitter, 256 ADC samples, 4.4 MS/s, 65 us slot period,
with the slope changed from 60.012 to **40 MHz/us**. The original fixture has
only a 10.99 m complex unambiguous range; the change gives **16.489 m**, covering
the requested 15 m volume without range folding. The validator checks this
against the result's actual processing axes. Antennas are isotropic, and the
output is the default normalized range spectrum. The resulting per-frame
cube has shape `(1, 3, 4, 128, 256)` and covers 98,304 observation instants.
`Motion.adaptive()` retains its default 0.02 rad phase tolerance, 0.02 relative
amplitude tolerance, 2 ms spacing bound and two interpolation nodes.

At **every evaluated probe**, the benchmark calls the existing `SMPLBody`
implementation on the full mesh, then evaluates the same material surface
points using fixed face IDs and barycentric weights. It does not bake vertex
positions once and substitute an interpolated point animation. The points are
area-sampled once with trimesh and seed 10; each layout is retained as JSON.
128 surface sites is the practical starting point recommended by `wt_genesis`.

## Motion and scattering assumptions

- Genesis supplies all 23 articulated joint rotations and the exported shape.
  Root heading and translation are explicitly reauthored for this bounded route.
- The 5.95 s span between source endpoints is closed with a 0.5 s bridge using
  a periodic cubic spline of joint axis-angle coordinates: loop period 6.45 s.
  This is an authored continuous benchmark trajectory, not a motion-capture
  ground-truth recording. Foot contact and floor penetration are not constrained.
- The pelvis follows `x=8-5 cos(wt)`, `y=1.5 sin(wt)`, with
  `w=2*pi/30 s^-1`. The heading follows its tangent, giving continuous turns.
  Root speed varies from 0.314 to 1.047 m/s. SMPL +Y is world +Z and SMPL +Z
  is the forward direction. The radar is at `(0,0,1)` m.
- Pelvis height is fixed from the first posed mesh; the source root's vertical
  bob and global orientation are not replayed. This adaptation is part of the
  fixture, and must not be confused with unmodified Genesis root motion.
- Each of N material sites has scalar isotropic RCS `1/N m^2`. That fixes an
  incoherent strength budget, not the coherent aggregate RCS of the human.
  Phase interference between the moving points is retained.
- This is the complete **LOS material-point radar pipeline**, including Channel
  one-way propagation, round-trip composition, scattering, antenna weighting,
  native synthesis, range transform and separately timed range-Doppler DSP.
  It does not include body self-occlusion, mesh electromagnetic scattering,
  skin dielectric calibration, room multipath, noise, or a receiver chain.
  A full mesh is evaluated for motion; the mesh is not inserted as an occluder.

## Measurement method

The machine is an RTX 5080 with Torch 2.10.0 / CUDA 12.8, Windows. The user
explicitly requested measurement under the current shared desktop load,
including a running Unreal editor. These are shared-load measurements, not an
idle-machine throughput claim or a real-time guarantee. Do not compare these
absolute latencies with the 2026-09-18 tables as an implementation regression.

`completed()` synchronizes CUDA before and after each measured operation.
Single-frame calls include session construction and discovery. Sequence calls
use one public `Radar.stream` session. The sum of synchronized `next()` times
is reported separately from DSP and the wall time that also includes checks,
progress logging and three saved sample cubes. Model loading/layout authoring
and first-call costs are recorded separately. Streamed cubes are not retained.

The profiler runs **after** clean timing. Instrumented stages explicitly
synchronize and can slow the run down; their times are inclusive and nested,
so they must not all be added. `smpl_vertices`, discovery and replay are inside
the trace stage. A separate cProfile dump retains host-call attribution.

`velocity_at` evaluates the derivative of the SAME mesh/world-transform
expression using forward AD. Its microbenchmark is separate: the public
dynamic `PointTargets` route consumes positions and does not call this velocity
diagnostic. Adding that diagnostic time to radar time would count work the
pipeline did not execute.

## Completed 30-second sequence

The corrected 128-site run completed all **300 frames at 10 Hz**:

| Measurement | Result |
| --- | ---: |
| Completed simulation time, sum of synchronized stream calls | **1033.615 s (17 min 13.6 s)** |
| Range-Doppler DSP, all 300 frames | **1.249 s** |
| Wall time including DSP, checks, logging and three sample exports | **1036.696 s** |
| Mean / median simulation time per frame | **3.445 / 3.355 s** |
| P95 / maximum simulation time per frame | **5.320 / 20.778 s** |
| Minimum simulation time per frame | 0.535 s |
| Simulation throughput | **0.290 frames/s** |
| Peak Torch CUDA allocation, including diagnostic checks and DSP | **245.23 MiB** |
| Peak allocation above resident model/session state | 218.01 MiB |
| SMPL evaluations over the sequence | **14,844** |
| Probe count per frame, minimum / median / maximum | **27 / 53 / 101** |
| Native synthesis batches per frame | **193**, every frame |
| Discovery epochs across the streamed sequence | **1** |

Every cube was finite and nonzero. No slow frame was removed. The two slowest
frames took 20.778 and 19.286 seconds, with 65 and 33 probes respectively:
probe count alone does not explain the tail. GPU load snapshots and the raw
frame records are retained. These measurements cannot isolate OS scheduling,
GPU competition or paging from compute without a separate controlled run.

The separately repeated standalone frame calls had median **0.745 s**, ranging
from 0.540 to 3.408 s. They were measured earlier than the sequence, under
changing desktop load. This difference is **not** evidence of a stream penalty;
do not divide those two medians to claim a regression or speedup.

### Where this workload spends its time

A separate synchronized profile at scene time 2.3 s took **2.851 s**:

| Stage | Time | Share of this instrumented frame |
| --- | ---: | ---: |
| Full SMPL/world vertex evaluation, 53 calls | **1.350 s** | **47.3%** |
| Remaining adaptive trace work | 0.285 s | 10.0% |
| Echo interpolation, row expansion and synthesis | **1.202 s** | **42.2%** |
| Remaining session work | 0.014 s | 0.5% |

The full trace is 1.634 s and **includes** SMPL. Channel discovery's two leg
calls (77.5 ms) and four batched replays (24.1 ms) are also inside the trace,
not additional costs to add to the table. No topology refinements occurred
over the sequence. The heavy-multipath discovery bottleneck from older reports
does not describe this empty-world LOS fixture.

The main amplification is concrete: `98304 observations * 4 receivers * 128 sites`
is **50,331,648 row contributions per frame**, versus 393,216 for one point.
The fixed temporary-row budget splits that into 193 synthesis batches. The
complete SMPL pose is independently evaluated at every probe, including repeat
GPU construction of the face-index tensor in `SMPLBody.to_mesh`.

A separate cProfile run attributes 1.382 s to 53 `to_mesh` calls, including
0.691 s in `_evaluate`; `_fast_smpl_forward` accounts for 0.636 s. The difference
between `to_mesh` and its vertex evaluator is substantial, and the source shows
the invariant topology being uploaded again on each call. The echo profile also
shows repeated gathers, dtype conversions and interpolation dispatch across
193 batches. These are concrete optimization targets; no gain from changing
them has been measured by this task.

Standalone geometry microbench medians were **42.56 ms** for a material-point
position evaluation (full SMPL included) and **57.01 ms** for the analytic
full-vertex velocity diagnostic. They are separate samples under shared load,
not additive components of the sequence timer. This task did not sum isolated
native-kernel CUDA-event times, so the older single-point "3% native device"
figure must not be transferred to this human-body workload.

Evidence: `output/smpl-walk/range15m/results.json`, the per-frame
`128/frames.jsonl`, `128/frame.prof`, `128/profile.txt`, sample NPZ cubes and
`output/smpl-walk/shared-gpu-load.csv`.

### Interleaved site-count comparison

Three rounds at scene times 0 and 2.3 s, with 32/128/512 sites interleaved in
each round; six completed calls per count:

| Sites | Median | Min–max | Probe counts at the two instants | Synthesis batches |
| --- | ---: | ---: | --- | ---: |
| 32 | **1.274 s** | 0.454–2.692 s | 27 / 39 | 49 |
| 128 | **3.114 s** | 0.783–6.691 s | 27 / 53 | 193 |
| 512 | **9.211 s** | 2.516–9.657 s | 27 / 53 | 775 |

All counts evaluate the full SMPL mesh; changing count changes the declared
surface sites and the echo workload. Sampling uses the same seed but each count
has its own material layout, so this measures real workload scaling, not a
controlled isolated kernel-complexity law. Raw measurements are in
`output/smpl-walk/scaling/results.json`. The combined performance figure is
`output/smpl-walk/performance.png`.

## Validation

The corrected 15 m, 128-site run measured:

| Check | Result |
| --- | ---: |
| Adaptive vs exhaustive ADC, 64 observations, complex IQ relative L2 | 0.002323 |
| Analytic vertex velocity vs central difference, relative L2 | 0.000381 |
| Periodic pose endpoint jump | 0 rad |
| Periodic pose-rate endpoint jump | 0 rad/s |
| All-vertex range sampled over the 30 s route | 2.752–13.328 m |
| Actual processing-axis unambiguous range | 16.489 m |

The small exhaustive oracle validates the declared motion-to-IQ integration;
it does not establish that every frame of the long adaptive sequence has the
same error. Every long-sequence cube is checked for finite, nonzero samples.
The saved workload preview shows four generated gait poses and the root route.

The preliminary `output/smpl-walk/full` sequence was intentionally stopped
after its distance-window mismatch was identified. Its partial timing and the
separate `smoke` profile are diagnostic evidence only, not the completed 15 m
benchmark. The corrected run is under `output/smpl-walk/range15m`.

## Reproduction

Use an environment with this Radar checkout, matching Core/Channel/RayD native
runtimes, SMPL, scipy, trimesh and matplotlib. This run used `witwin2` and the
validated developer Channel binary in `channel/build/cmake_rayd080`, fingerprint
`183118e96d75856e71df2f90c621fa191d6151d2b63ec12fd5f71062e49ed73f`.
No native validation was bypassed.

The local shell selected that runtime explicitly (these paths describe this
machine, not a dependency to install into other checkouts):

```powershell
$env:PYTHONPATH = 'E:/Code/witwin-platform/channel;E:/Code/witwin-platform/radar'
$env:WITWIN_CHANNEL_DEVELOPER_OVERRIDE = '1'
$env:WITWIN_CHANNEL_EXTENSION_PATH = 'E:/Code/witwin-platform/channel/build/cmake_rayd080/_channel.cp311-win_amd64.pyd'
$env:WITWIN_CHANNEL_EXPECTED_FINGERPRINT = (Get-Content ../channel/build/cmake_rayd080/_channel.build-fingerprint).Trim()
```

For generation, expose the Genesis package, set `WITWIN_GENESIS_DATA_DIR` to
the asset directory, and run `python tools/generate_smpl_walk.py`. The local
environment needed `chumpy` for its old licensed pickle; it was installed into
`output/smpl-walk/python-deps`, not into the project's production dependencies.
cuDNN's DLL directory also needed to be on PATH for Genesis fitting.

For SMPLBody, provide the neutral model under its expected filename
`basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` in the model-root directory. The
benchmark records the actual file hash, motion hash, script hash and native
build identity in its results.

```powershell
python tools/benchmark_smpl_walk.py --motion tools/fixtures/smpl_walk_genesis_seed10.npz --model-root output/smpl-walk/models --sites 128 --frames 300 --runs 3 --validate --preview --profile --output output/smpl-walk/range15m
```

Run the count comparison **after** the sequence, not concurrently:

```powershell
python tools/benchmark_smpl_scaling.py --motion tools/fixtures/smpl_walk_genesis_seed10.npz --model-root output/smpl-walk/models
```

This interleaves 32/128/512 sites at two body instants in three rounds. It uses
the exact same geometry and simulation helpers; it is not a second simulator.

No simulator production equations or public APIs were changed for this benchmark.

## Checks executed

- `pytest tests/test_smpl_pose_refusal.py tests/test_moving_structures.py::test_the_smpl_pose_velocity_matches_a_two_snapshot_difference --gpu -q`:
  **11 passed**, no skips; log retained at `output/smpl-walk/tests.log`.
- Full-tree Ruff formatting and lint checks: passed.
- Duplicate-code, architecture, documentation-surface, no-compatibility,
  public-API-manifest, release-claims, required-Channel-coverage and workflow-
  reference checks: passed. No complete GPU regression-suite run is claimed.
- The benchmark's own ADC, velocity, range-window, motion-continuity and
  per-frame finite/nonzero assertions all passed on the completed run.
