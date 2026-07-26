# R-ADR-017: The processing facade, and the frozen vendor DSP surface

Status: Accepted (Phase 8)

## Context

Before Phase 8 the Radar package had no processing owner. It had a `sigproc`
directory holding three separate range-FFT implementations with three different
windowing choices, two frame packers, two point-cloud pipelines differing only in
the detector, a `MUSICImager` with a hard-coded half-wavelength spacing and its
own `numpy` angle grids, magic range gates written as `[:, :25] = -100`, and a
Python loop over transmitters doing an in-place multiply on a clone. There was no
named range-profile stage, no OFDM channel-impulse-response inversion anywhere,
no batched detector, no point-cloud contract, and no tracking interface.

There was also no statement of what the Radar package is allowed to do in Torch.
The plan's Torch/DSP exception (R-ADR-007) says post-processing may be Torch, but
"post-processing" was a word rather than a directory, so a `torch.fft` call in a
solver and a `torch.fft` call in a detector were indistinguishable to any guard.

Three facts shape the answer.

1. **The owner directive is that all radar post-processing stays PyTorch.**
   Range profile, Range-Doppler, cube formation, AoA, beamforming, CFAR, point
   cloud and tracking are Torch. Only simulation kernels are native CUDA. A
   native cuFFT wrapper needs a MEASURED dispatch, layout, fusion or tape
   bottleneck plus its own decision record, and Phase 8 ships none.
2. **Three waveforms, one chain.** FMCW's beat cube is the CONJUGATE of
   Channel's phasor convention and OFDM's and pulsed's are not, so the same
   closing target lands on opposite Doppler bins in a magnitude plot where the
   difference is invisible. Nothing reconciled it.
3. **The old public names have callers.** The repository's no-legacy rule
   deletes old INTERNAL paths; it does not delete a public surface without a
   migration.

## Decision

### 1. `witwin/radar/processing/` is the single processing owner

One package, one entry per stage, rank generic with an arbitrary leading batch:

```
SynthesisResult -> ProcessingCube -> range_profile -> range_doppler
                -> beam_cube -> ca_cfar -> point_cloud -> DetectionFrame
```

`witwin/radar/sigproc/` keeps its entire public surface. Every module under it is
re-export only, asserted by AST: no function definition, no class definition, not
one expression. The adapters live INSIDE the facade, in
`processing/adapters.py`, which is what lets the fence below be a statement about
a DIRECTORY rather than a list of exceptions.

### 2. `ProcessingAxes` is the one metadata, axes and units record

Built from the waveform SPECS by `ProcessingAxes.from_synthesis(result, spec,
array)`, never from the flat engineering-unit `RadarConfig`, which has exactly
one documented conversion site and must not grow a second home. It publishes the
range and velocity axes in SI, the phasor convention, the array layout, and the
derived `doppler_sign`.

The legacy `Radar` is NOT made multi-waveform. `RadarSystemConfig.axes()` raises
for anything but FMCW, `Radar.__init__` calls `_init_axes` unconditionally, and
`from_radar_config` hard-codes FMCW, so a non-FMCW `Radar` is unconstructible
today. `ProcessingAxes` is built one level up, at the
`RadarSystemConfig`/`SynthesisResult` level, where all three waveforms already
exist. `as_fmcw_axes()` returns the Phase-6 `RadarAxes` for the legacy callers.

### 3. One canonical Doppler convention, reconciled exactly once

`PROCESSING_DOPPLER_CONVENTION = "positive_doppler_bin_is_closing"`.

A closing radial speed `v` gives `tau_rate = -2 v / c`, so the canonical
frequency is `-f_ref tau_rate = +2 v / lambda` and `v = lambda f / 2` is the
published velocity axis for all three waveforms.

`doppler_sign` is DERIVED from the cube's published `phasor` string
(`BEAT_PHASOR -> +1`, `CHANNEL_PHASOR -> -1`, anything else is a `ValueError`),
and the reconciliation is applied in exactly one place, inside `range_doppler`,
as a frequency-index reversal `X[k] -> X[(-k) mod D]` performed with
`index_select` BEFORE the `fftshift`. It is a gather with no arithmetic, so it is
exact.

Before the shift and not after: negating a frequency index is a wrap in the
unshifted order, and for even `D` the shifted axis runs `[-D/2, D/2)` and is
asymmetric about zero, so reversing the shifted axis would move every bin by one.

The same conjugation reaches the SPATIAL phase across the virtual array.
`conventional_steering` and the FFT angle estimators read the same derived
`axes.doppler_sign` / `array.phase_sign`, so there is one derived quantity behind
both reconciliations rather than two sign decisions that can drift.

### 4. The facade fence

No `torch.fft`, no CFAR, no angle estimator and no beamformer expression appears
anywhere under `witwin/radar/` outside `witwin/radar/processing/`.

One named allowance, with a reason: `witwin/radar/solvers/solver_dirichlet.py`
inverts a SYNTHESIZED spectrum into time samples. That transform is part of
producing the received signal, not of reading it, and it predates the processing
chain. The allowance is not a blanket - a test asserts that the module still
CALLS `ifft` and contains no forward transform, no `fftshift` and no detector, so
the exception cannot quietly grow into a processing path.

### 5. The frozen vendor DSP primitive list

**Scope, stated because an unstated scope drifts.** In scope: transforms;
pooling, padding and patch extraction; decompositions and solves; order
statistics and selection. Out of scope: elementwise arithmetic (`exp`, `angle`,
`polar`, `log10`, `sqrt`), shape manipulation and construction (`arange`,
`stack`, `zeros`, `reshape`), contraction (`einsum`, `tensordot` - a contraction
is a sum of products, and a beamformer's is not a signal-processing algorithm),
and random sampling. Freezing those would be freezing arithmetic.

The frozen list, asserted by EQUALITY in four cells so it can neither grow nor
silently shrink:

| Cell | Frozen |
|---|---|
| transforms | `torch.fft.{fft, ifft, fft2, fftshift, ifftshift, fftfreq}` |
| pooling / patch extraction | `torch.nn.functional.{avg_pool2d, unfold, pad}` |
| decompositions / solves | `torch.linalg.{eigh, solve}` |
| order statistics / selection | `torch.{argsort, argwhere, gather, sort, topk, where}`, `Tensor.{cumsum, index_select, unfold}` |

Four entries beyond the design's list, each earned when the facade was built:

* `argsort` - the tracker orders its association candidates;
* `gather` - the angle estimators read a peak out of a padded spectrum;
* `index_select` - the Doppler sign reconciliation, chosen BECAUSE it has no
  arithmetic and is therefore exact;
* `Tensor.unfold` - the MUSIC sub-aperture view and the micro-Doppler framing.
  It replaced an `(L + 1) ** 2`-way `torch.stack` over a list comprehension.

`torch.linalg.inv` and `torch.linalg.pinv` are asserted ABSENT: `mvdr_weights`
solves, it never forms an inverse.

The vendor window constructors are absent too, and that is deliberate.
`torch.hamming_window(N, periodic=False)` and `torch.hamming_window(N,
periodic=True)` are DIFFERENT sequences and the difference is invisible at a call
site. The facade owns one window family with an explicit periodic/symmetric
distinction; the legacy adapters use `hamming_symmetric` because every legacy
transform used `periodic=False`.

Six adjacent vendor calls are RECORDED, also by equality, so nobody has to decide
again whether they are DSP: `torch.angle`, `torch.einsum`, `torch.polar`,
`torch.randint`, `torch.randperm`, `torch.tensordot`.

### 6. The native-DSP gate: measured, and the answer is no

The decision rule, applied to `tools/benchmark_processing.py`'s output. Native
DSP is justified only if the measurement shows one of:

| # | Criterion | Measured | Tripped |
|---|---|---|---|
| (a) | dispatch overhead dominating actual transform time | every stage is flat in problem size: `range_profile` 0.079 ms at both the fixture and a 48x larger cube; `range_doppler` 0.142 / 0.145 ms; `fft2_aoa` 0.396 / 0.393 ms. Dispatch DOES dominate | see below |
| (b) | a layout conversion costing more than the transform it feeds | `assemble_frame_cube` 0.018 ms against a 0.079 ms range profile (0.23x); `beam_cube` 0.041 ms; the micro-Doppler framing copy 0.022 ms against its 0.021 ms transform (1.05x) | no |
| (c) | a fusion opportunity removing a materialized intermediate LARGER than the output | the windowed tensor is exactly the size of the output. The one genuine outlier is `os_cfar`, 138 MB for one `[128, 256]` map against `ca_cfar_fast`'s 0.62 MB - but that is an ALGORITHM choice with a Torch-side fix (chunking), not a kernel-fusion argument | no |
| (d) | a tape or AD cost a native primal+JVP+VJP would remove | processing carries no production tape. The chain is post-synthesis and the plan already declares CFAR, peak selection and tracking non-differentiable | no |

**(a) is tripped and it does not argue for a cuFFT wrapper.** Every stage being
flat in problem size means the cost is per-launch, not per-element. A cuFFT
wrapper replaces one dispatch with another dispatch; it removes no launch. What
the data argues for is FEWER, LARGER launches or a captured CUDA graph, and both
are Torch-side changes with their own evidence requirements. Recorded here so a
later reader does not mistake "launch bound" for "needs a native kernel".

**The recorded answer is therefore: no native DSP in Phase 8.**

Two Torch-side optimizations the measurement found, both deferred with their
numbers rather than taken here, because this stage measures and freezes and does
not optimize:

* the window multiply costs 0.062 ms against the 0.019 ms transform it feeds -
  3.2x - so a fused windowed transform, if `torch.fft` ever offers one, is worth
  more than the transform itself;
* `os_cfar` at 1.64 GB for an 8-beam `[128, 256]` cube is a chunking candidate.

### 7. The cutover deletion list

Ten items, each with a test that the symbol or the code path is gone:
`FrameConfig`'s seven raw `radar.config.*` reads; the duplicate range/Doppler FFT
bodies; `frame_reshape`; `_process_pc_cfar_tensor`; `reg_data`'s `numpy`/
`np.random` path; the magic range-gate bins; `MUSICImager`'s hard-coded `0.5`
spacing; the `numpy` leaks in the tensor path; `_compensate_tdm_phase`'s Python
transmitter loop; and `matched_filter`'s unconditional `complex128` upcast.

## Consequences

* One owner per stage, and the fence is a statement about a directory.
* Five real defects were found and fixed by the cutover, all with named boundary
  constants preserving the legacy behaviour for the legacy entry points: a
  sink-major steering table multiplied against a tx-major cube; FFT angle
  estimators that never reconciled the beat conjugation in space and reported
  every target at its mirror image; a phase-comparison elevation cosine pointing
  against `z`; a MUSIC pseudo-spectrum evaluated at the conjugate steering
  vector; and a CFAR `max_points` thinning that reordered peaks by energy while
  reading angles in mask order.
* The frozen list is asserted by equality, so adding a vendor primitive is a
  deliberate edit to this record and its test, not a silent import.
* The two wall-time budget pins are device specific. That is what a frozen budget
  means; the response to a failure is to record a new measurement on purpose,
  never to widen the factor.

## Alternatives rejected

**A native cuFFT wrapper.** Rejected on the measurement above: the chain is
dispatch bound, and a wrapper trades one dispatch for another.

**Keeping the three range FFTs and adding a fourth for OFDM.** Rejected: the
three differ in their windowing and one applies an unconditional DC removal that
is a clutter operation. Collapsing them into one owner with an explicit `window`
and an explicit `remove_dc` (defaulting OFF) is what makes a clutter-export test
mean what it says.

**Reconciling the Doppler sign at every consumer.** Rejected: there are five
consumers and the sign is invisible in a magnitude plot. One derived quantity,
applied once.

**Deleting the `sigproc` public names.** Rejected: the plan's work item 6 asks
for migration adapters plus a cutover deletion of the old INTERNAL paths, which
is what shipped.
