# Radar Performance

Status: local measurements recorded 2026-09-16 to 2026-09-18; release-platform benchmarks remain separate.

## Retraction and corrected measurement (2026-09-18)

An independent acceptance run could not reproduce the latency tables the two
2026-09-18 sections below published, and it was right. **Those tables are retracted.**
They were built from measurements taken tens of minutes apart, with test suites in
between, and this machine's absolute latency drifts by up to 2x over that span: the
same harness on the same commit measured the rotor fixture at 47.6 ms and, hours
later, at 97.9 ms. A ratio taken across two such measurements is not a result.

The corrected numbers below are measured round-robin - every revision, one fixture
per process, three rounds, medians across rounds - so a drifting machine cannot be
mistaken for a code change. The fixtures are now `tools/benchmark_adaptive_frame.py`
rather than a description in a report, because "1x1, 128 x 128" is not reproducible.

`7c93c8e` is the pre-change baseline, `f00bd8e` the per-observation bookkeeping work,
`a2775db` the device-side row expansion and deferred leg narrowing.

| Fixture | observations | probes | `7c93c8e` | `f00bd8e` | HEAD | first | second | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| walker, bound-limited MIMO | 98304 | 27 | 118.2 ms | 61.5 ms | **38.7 ms** | 1.92x | 1.59x | **3.05x** |
| rotor, phase-limited | 16384 | 155 | 121.0 ms | 115.0 ms | **98.9 ms** | 1.05x | 1.16x | **1.22x** |
| pair, two scatterers | 6144 | 9 | 31.7 ms | 26.3 ms | **25.1 ms** | 1.21x | 1.05x | **1.26x** |

What the retracted tables got wrong is the SIZE of the gain outside the
observation-heavy regime, and they got it wrong in the flattering direction. The
published 2.57x on the rotor is really **1.22x**, and the published 2.01x on the
two-scatterer frame is really **1.26x**. The 16 ms the deferred narrowing saves
there matches the 19.7 ms of `slot()` the profile attributed to it; it is the
115 ms frame it sits in that the retracted baseline understated.

The work is still worth what the observation-heavy regime says it is - a
98304-observation frame went 118.2 to 38.7 ms - and every bit-identity, accuracy,
memory, synchronization and call-count claim in the sections below was independently
reproduced. Only the latency ratios moved.

### A minute of data, measured rather than extrapolated

The sections below extrapolate 128 frames to "about 12 s for one minute at 10 fps".
That is wrong twice: it is not what a minute costs, and per-frame cost is not flat.
Streamed, 600 frames at 10 fps:

| Sequence | Wall time | ms/frame | Segment range | Probes | Peak |
| --- | ---: | ---: | --- | --- | ---: |
| Target pacing inside 3-8 m | **16.7 s** | 27.8 | 26.1 to 28.9 | 27 throughout | 79.4 MiB |
| Target receding to 78 m | **29.5 s** | 49.2 | 24.9 to **224.1** | 27 to **615** | 99.1 MiB |

The second row is a real limit, not a fixture artifact to wave away.
**float32 position quantization interacts with the phase tolerance.** One float32
step of a position at range R is `R * 2^-23` metres, which is
`4*pi*R*2^-23/lambda` radians of two-way phase; at 77 GHz that reaches the default
0.02 rad tolerance near 50 m. Past it no interval can be certified and the
controller bisects toward exhaustion. Measured probe counts on one frame at 6, 20,
40, 55, 70 and 78 m: 27, 27, 29, 31, 38, **162**.

So peak allocation is flat in the FRAME count, as the sections below say, but not in
the probe count: the receding run peaks at 99.1 MiB against the 76.4 MiB those
sections present. A long sequence whose target stays in a room is flat in both.

Reproduce: `python tools/benchmark_adaptive_frame.py --runs 5` and
`python tools/benchmark_adaptive_frame.py --fixtures walker --sequence 600`.

## Device-side row expansion and deferred leg narrowing (2026-09-18)

The two limits the host-cost section below left standing, measured on RTX 5080 / Ryzen 7 9800X3D
in witwin2. They are different limits in different regimes, which is why both were needed: a frame
with many observations spends its time expanding rows, and a frame with many probes spends its
time per probe.

**The echo expands its row maps on the device.** The host holds a partition of tens of intervals
and a compact per-observation column set; expanding those into one integer per row and sending the
result is work the device can do from what it already has. At this frame's 393216 rows, measured
standalone, the two routes are 9.23 ms of numpy plus transfer against 0.64 ms of device work, for
identical indices. An independent run of the same comparison measured 15.2 ms for the host side
and 0.571 ms for the device side, so 9.23 ms understates the host route and the real ratio is
nearer 26x than 14x. Host-to-device traffic falls from about 15 MB per frame to about 5 MB, uploaded
once instead of per batch. Every `repeat_interleave` declares its `output_size` from the host's own
row totals, so the expansion adds no synchronization of its own, and `_adaptive_echo` still observes
the device zero times - which `tests/test_import_boundary.py` checks, by scanning for `cpu`, `numpy`,
`tolist` and `item` by name.

**It does still synchronize.** A pageable host-to-device copy is synchronous, so each `upload`
is one synchronizing CUDA operation: six per frame after this change against five per BATCH before
it, measured with `torch.cuda.set_sync_debug_mode` against positive and negative controls. That is
a large reduction on a multi-batch frame and one MORE synchronization on a single-batch frame. An
earlier revision of this section claimed "nothing here synchronizes" and that the import-boundary
test enforced it; both were wrong. That test is an AST scan for four attribute names and cannot see
an upload, an `int(tensor)`, a `torch.nonzero`, or a `repeat_interleave` missing its `output_size`.

**A batched group no longer narrows its slots eagerly.** `RadarLegBatch.slot(...)` builds a
revalidated single-slot batch, and the replay loop built two of them for every observation. A
batched group composes once and reads nothing from the individual slots - the rows it needs are
already sliced out of the composed batch - so those narrowings served only the record's `legs`
member, which on the adaptive route is published for exactly one observation per frame. Deferring
it behind a callable takes the count from 310 narrowings per frame to **2**, measured on the
155-probe rotor.

Per-frame latency: **retracted, see the correction above.** The table published here
gave 50.3/124.2/20.3 ms before and 21.6/48.3/10.1 ms after, for 2.33x, 2.57x and 2.01x.
Measured round-robin the same three fixtures are 61.5/115.0/26.3 and 38.7/98.9/25.1,
for **1.59x, 1.16x and 1.05x**. The probe count in the third row was wrong too: 9, not 27.

Every cube is bit-identical to the pre-change cube under `torch.equal` at each step, on all three.
That statement is about the CUBE. `Result.adaptive_diagnostics[...]["synthesis_batches"]` does move,
because a row budget that no longer pays for the inactive transmitters' rows reaches further per
batch: the walker frame reports 2 where it reported 5. Gradients are not bit-reproducible on either
side of this change - the backward accumulates atomically and two runs of the same revision differ
by about 1.8e-7 absolute - so the equality claimed here does not extend to them.

`tools/validate_frame_streaming.py --frames 8 32 128`, same fixture and machine:

| Frames | Streamed peak | Stacked peak | Peak ratio | Streamed ms/frame | Stacked ms/frame | Mismatched frames |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 76.4 MiB | 113.4 MiB | 1.48x | 18.4 | 19.8 | 0 |
| 32 | 75.4 MiB | 293.6 MiB | 3.89x | 20.2 | 22.7 | 0 |
| 128 | 76.4 MiB | 1223.4 MiB | 16.01x | 19.6 | 20.1 | 0 |

128 frames at 10 fps is 12.8 s of scene time produced in **2.51 s**, against 4.90 s before this
section and 13.86 s before the host-cost work. The "about 12 s for a minute" that stood here is
retracted: a minute of a target pacing in a room measures 16.7 s, and a minute of THIS receding
walker measures 29.5 s. See the correction above. Streamed peak allocation is still flat in the frame count, which is the property
`stream` exists for, but it ROSE from 56.2 MiB to 76.4 MiB: the row maps are now device tensors and
several of them are live at once inside a batch. The row budget still bounds them - it is the same
knob against a larger constant - and the trade is 20 MiB of flat allocation for half the latency.

Accuracy is unchanged. `tools/validate_adaptive_motion.py` measures IQ relative L2 of 2.8020e-4,
2.2612e-4, 2.4938e-4 and 5.6830e-4, `tools/validate_doppler_motion.py` 1.144e-6, 1.845e-4 and
4.085e-4, and `tools/validate_heavy_multipath.py` still places its four strongest peaks within one
range/velocity bin of image geometry - every one of these the same digits as before the change.

Reproduce: `python tools/validate_frame_streaming.py --frames 8 32 128`. Evidence:
[the row-expansion report](docs/dev/audit/radar-row-expansion-and-probe-cost-2026-09-18.md).

## Per-observation host cost (2026-09-18)

The adaptive route's frame was host-bound, not device-bound. On the public scene entry with a
3 TX x 4 RX, 128-chirp, 256-sample walker at 10 fps, RTX 5080 / Ryzen 7 9800X3D in witwin2,
CUDA events around the two native calls measured **3.2 ms of a 99 ms frame**, 3.1%. The rest was
the interpreter carrying a 98304-entry per-observation schedule through structures whose distinct
answers number in the tens: the adaptive controller evaluated 27 probes and accepted 13 intervals
for that same frame, and discovery and compilation each ran once.

Four changes, none of which touches a tolerance, a partition decision or the physics. Every cube
below is bit-identical to the pre-change cube under `torch.equal`, on both a bound-limited MIMO
walker and a phase-limited rotor.

| Change | What it replaced |
| --- | --- |
| The accepted partition writes one slice per interval | One dict entry and one numpy row per observation |
| The observation schedule is a `float64` array | Three full-length passes building Python floats |
| The echo synthesizes only each slot's own transmitter | All `num_tx * num_rx` pairs, then discarding `1 - 1/num_tx` |
| Pair row bounds are held per evaluated observation | An `[observations, pairs]` table that was zeros except at the probes |

The third is the largest and needs the layout stated: under `PAIR_RANK_LAYOUT` a pair's
transmitter rank is `pair % num_tx`, a TDM observation sits in one slot and hears one
transmitter, and the slot gather at the end of the frame already discarded the rest. The rows
were synthesized and thrown away.

Frame latency on that walker, median of seven complete `Radar.simulate` calls:

**Retracted, see the correction above.** The staged table published here ran
104.9 -> 75.9 -> 72.4 -> 46.8 -> 41.0 ms for a claimed 2.56x. Each stage was a
working-tree state rather than a commit, so none of them can be re-measured; the
end-to-end step this section is responsible for, `7c93c8e` to `f00bd8e`, measures
**118.2 to 61.5 ms, 1.92x**, round-robin. The ordering conclusions the stages
support - that the partition dict and the transmitter restriction were the two large
items - rest on profile attribution, which is not latency and does not depend on the
machine's clock.

The maintained sequence measurement, `tools/validate_frame_streaming.py --frames 8 32 128`, on
the same fixture and machine:

| Frames | Streamed peak | Stacked peak | Peak ratio | Streamed ms/frame | Stacked ms/frame | Mismatched frames |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 56.2 MiB | 93.1 MiB | 1.66x | 39.4 | 39.9 | 0 |
| 32 | 56.2 MiB | 300.6 MiB | 5.35x | 40.1 | 40.5 | 0 |
| 128 | 56.2 MiB | 1230.3 MiB | 21.91x | 38.3 | 39.4 | 0 |

128 frames at 10 fps is 12.8 s of scene time produced in **4.90 s**, against 13.86 s before.
The minute figure that stood here extrapolated those 128 frames and is retracted; see the
correction above for a measured one. Streamed peak allocation is flat and
fell from 67.4 MB, because the per-observation pair table is gone. This is one LOS walker
fixture: it does not establish a per-frame cost for heavy multipath or moving meshes, whose limit
remains topology discovery.

Accuracy is unchanged, not improved. `tools/validate_adaptive_motion.py` in the same session
measures IQ relative L2 of 2.80e-4, 2.26e-4, 2.49e-4 and 5.68e-4 on radial, rotor, limbs and
heavy multipath, the same four values the 2026-09-17 table records. The published
`synthesis_batches` diagnostic falls where the transmitter restriction lets more observations fit
one row budget, which is the change working, not a new bound.

Do not read that tool's speedup COLUMN as a result of this work in either direction. Its four
fixtures are 512 and 2048 observations against 97 and 19 probes, so they measure the probe path
and not the per-observation bookkeeping this section changed; the ratios it reported in the same
session are 96.1x, 94.7x, 95.9x and 143.5x against the 2026-09-17 row's 120.4x, 119.1x, 92.5x and
141.3x, which is single-run spread on a shared desktop at that fixture size. The frame-latency and
sequence tables above are the measurement this work is claimed on.

Reproduce: `python tools/validate_frame_streaming.py --frames 8 32 128` and
`python tools/validate_adaptive_motion.py`. Evidence: `output/frame-streaming/results.json`,
`output/doppler-repair/adaptive/results.json` and
[the host-cost report](docs/dev/audit/radar-frame-host-cost-2026-09-18.md).

## What changed

FMCW defaults to a normalized range spectrum, which native Dirichlet synthesis evaluates in closed form for a delay that holds the whole chirp. A walking delay is synthesized in the beat domain, where the continuous fast-time phase is evaluated per sample, and refreshed dynamic scenes evaluate each ADC observation. The synthesized beat-signal route is explicit with `output_domain="beat"`. That changes the default pipeline's work: spectrum input must not pay a second range FFT, while beat input still does.

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

Superseded by the 2026-09-18 host-cost section above, which reran this tool on the same fixture
and overwrote the results file cited below. The retention property this section establishes still
holds; only the latencies moved.

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
the exact same `tests/test_pipeline_budget.py` measurement recipe.

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
A walking delay has no closed-form range spectrum, so the spectrum family takes
no delay rate and refuses one by name. Evaluating that DFT term by term would
cost N per bin, or O(N^2) per path per chirp, where the beat family synthesizes
the same linear-delay model in N and the range transform finishes the axis in
N log N. The closed form is still the reason the spectrum family exists: a delay
that holds for the whole chirp collapses to a Dirichlet kernel, one evaluation
per bin, with no samples materialised and no transform. Derivatives take the
term-by-term sum in both families, so a spectrum backward costs N per bin where
its forward costs one. `tools/validate_doppler_motion.py` records actual scene
timings together with independent IQ/STFT errors.

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

The repository currently enforces performance through executable tests and workflow references, not through copied prose numbers:

```bash
pytest tests/test_pipeline_budget.py --gpu -q -s
python ci/check_workflow_references.py
```

The manually dispatched GPU workflow runs the CUDA tier and the maintained processing pipeline smoke benchmark. It publishes no wheel. Its existence does not prove a GPU result until the workflow actually runs and the artifact/log is retained.

## Native DSP policy

Signal processing remains PyTorch-owned. Moving a DSP stage into the native extension requires a measured bottleneck, a documented owner decision, forward/backward policy where relevant, and a before/after result from the rebaseline matrix above. The concept-axis consolidation alone is not performance evidence for such a move.
