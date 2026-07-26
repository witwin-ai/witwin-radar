# R-ADR-015: How Radar consumes a band, and what it deliberately does not

Status: Accepted (Phase 8)

## Context

Channel's ADR-042 added `FixedTopologyRequest.frequency_offsets_hz`: a host
tuple of offsets in Hz that makes a fixed-topology replay publish a `[K, F]`
transport alongside the `[K]` one, with column `j` evaluated natively at
`f_ref + df_j`. Its launch law is `(1 + F) * buckets`, its ADR-032 budget stays
at one device-to-host copy and one synchronization regardless of `F`, and a
column with `df == 0` is bit-identical to the reference coefficient.

Phase 6 anticipated this. `SynthesisPathBatch` already declared
`frequency_response` / `frequency_offsets_hz` with full shape, dtype and device
validation, and rule R8 refused every non-`None` value by name, citing Phase 8.
That refusal is what this decision retires.

Four things constrain the answer.

1. **A wideband column is not a shifted narrowband one.** It carries
   `exp(-j 2 pi f_n tau_rt)` - its own absolute phase, at the frozen delay - and
   the material response, the layer-stack fringes and the `lambda/(4 pi d)`
   spreading evaluated at `f_n`. It is a complete field, not a correction.
2. **Only one of the three waveforms has a discrete frequency grid.** OFDM
   transmits on subcarriers; FMCW and pulsed LFM sweep continuously in fast
   time.
3. **The two-way join is native, with primal, JVP and VJP.** Widening
   `two_way_join.cu` to a strided `[K, F]` means widening all three in one
   change.
4. **Channel publishes the frequency resolution but not the phase bound it
   implies.** Evaluating the bound needs `max(delay_s)`, a device reduction plus
   a host read that the ADR-032 per-call budget does not have.

## Decision

### 1. The band is a lifetime declaration on the adapter, in Hz

`ChannelPropagationAdapter(..., frequency_offsets_hz=...)`. It is a tuple of
frequencies and nothing else: not a subcarrier count, not a spacing, not an FFT
size, not a bandwidth. The waveform-to-Hz mapping is
`OfdmCfrSpec.frequency_offsets_hz`, on the synthesis side, which is the side of
the boundary that knows what a subcarrier is. The propagation request-keyword
equality assertion in `tests/test_phase6_config_boundary.py` was updated
deliberately to admit it, with that reasoning written into the test.

Structural validation of the grid stays in Channel, which owns the launch grid
those rules describe. The adapter checks only what Channel cannot see: whether
its declared component set is inside `capabilities().wideband_components`. That
set is a radar declaration made before any scene is touched.

### 2. `row_valid` stays `[K]` and broadcasts over the band

Whether a stationary point exists is a geometric fact about the endpoints. It
cannot depend on frequency, and widening the mask would invite an
implementation that thought it could.

### 3. Both legs carry a band, or neither

`TwoWayComposer.compose` refuses a banded leg joined with a narrowband one.
Broadcasting the narrowband leg's single coefficient across the band would
reintroduce the narrowband approximation silently, on exactly half of the round
trip.

### 4. The frequency axis of the join is a Python loop, and here is the number

The loop calls the existing `[K]` join primitive once per column: `1 + F`
launches, the reference column plus one per band column. `tau_rt` and `rate_rt`
are recomputed by every column and discarded, because they are functions of the
two delays alone.

Measured on the multi-endpoint fixture (2 TX, 2 sites, 2 RX, 11 composed rows):

| `F` | leg reevaluation, ms | compose, ms | join, ms per column |
|---|---|---|---|
| none | 3.50 | 0.267 | - |
| 1 | 5.68 | 0.449 | 0.182 |
| 8 | 21.83 | 1.665 | 0.175 |
| 16 | 38.52 | 3.304 | 0.190 |
| 64 | 140.40 | 9.629 | 0.146 |

The join loop costs about 0.15-0.19 ms per column against 2.15 ms per column for
the band evaluation itself, so it is 6-8% of a wideband frame. **A strided
`[K, F]` join is therefore not justified by this measurement**: removing all of
it would recover under a tenth of the cost, and the dominant term is the
`(1 + F)` native launches on the Channel side, which is Channel's own recorded
follow-up. A future R-ADR that wants the strided kernel must beat this table,
and must ship primal, JVP and VJP in the same change.

### 5. The scatter response is evaluated once and reused across the band

A composed column is `H_in(f_j) * S(f_ref) * H_out(f_j)`. The propagation and
material band shape is exact; the target's is frozen at the reference frequency.

This is a **declared deferral**, not an oversight. A wideband RCS is a different
capability with its own owner: `ScatterResponse.evaluate` returns one value per
site, `AspectScatterResponse` evaluates per row in a native kernel, and neither
has a frequency argument. Adding one means a new native contract with its own
AD family. Until then `RadarPathBatch` says so in its docstring rather than
leaving a reader to assume the whole round trip is wideband.

### 6. FMCW and pulsed LFM refuse a band, by name

Rule R8 now reads an opt-in, `consumes_frequency_response`, absent by default
and declared only by `OfdmCfrSpec`. A spec without it refuses a band rather than
discarding it.

The reason is physical rather than budgetary. OFDM's transmit grid IS a discrete
set of frequencies, so `[K, F]` with `F = num_subcarriers` is exact. FMCW's
instantaneous transmit frequency is `f_ref + slope * t`, continuous over fast
time, so a wideband beat needs either one column per fast-time sample - `F` in
the thousands, at `(1 + F)` native launches each - or a coarse grid the kernel
interpolates in frequency. Interpolation is a NEW approximation with its own
error term, and it is not a free consequence of the OFDM contract. Phase 8 ships
OFDM exact and declares the rest deferred.

R8 additionally refuses a band on a weight whose provenance says it carries no
reference phase: every column carries the phase it was evaluated with, so the
two statements contradict each other.

### 7. The kernel change is not the weight index

This is the part most likely to be got wrong, so it is stated here as well as in
the kernel header.

`ofdm_cfr.cu` gains `weight_columns`, 1 for `C[k]` and `num_subcarriers` for
`C[k][n]`. Indexing alone would be a defect:

* narrowband: the weight holds `exp(-j 2 pi f_ref tau_rt)` and NOTHING at
  `n * df`, so the kernel owns the whole subcarrier phase and the subcarrier
  term multiplies the FULL delay `tau_k(l)`;
* wideband: column `n` already holds `exp(-j 2 pi (f_ref + n df) tau_rt)`, so
  only its slow-time CHANGE is missing and the subcarrier term multiplies the
  DRIFT.

Applying `f_sub * tau` to a wideband weight counts `n * df * tau_rt` twice and
puts every tap at twice its delay - a plausible-looking range profile that is
wrong by a factor of two in range. `d(phi)/d(tau_rt)` loses its `f_sub` term to
match, which routes the delay gradient through Channel's response rather than
through the kernel phase; `d(phi)/d(tau_rate)` is unchanged.

Primal, JVP and VJP change together. The backward kernel keeps two loop nests: a
narrowband weight has one gradient slot per path and reduces over the whole grid
in symbol-major order, while a wideband weight has one slot per subcarrier and
reduces over symbols only. The narrowband nest is preserved verbatim, which is
what keeps a narrowband gradient bit-identical to the pre-band one.

### 8. Radar owns the frequency-resolution phase budget

`WIDEBAND_FREQUENCY_RESOLUTION_PHASE_BUDGET_RAD = 0.1`, checked once per frozen
topology at freeze time, where a host read is already paid:

```
pi * native_frequency_resolution_hz(f_ref) * max(delay_s) <= budget
```

Channel publishes the resolution and the law and deliberately does not evaluate
the bound. The value matches the `ASPECT_PHASE_BUDGET_RAD = 0.1` precedent. At
77 GHz the resolution is 8192 Hz, so the bound binds at `tau = 3.9 us`, a 580 m
round trip; a 150 m round trip sits at 2.6e-2 rad. The refusal names both
numbers and there is no clamped mode.

It is named for the frequency RESOLUTION rather than for what a signal engineer
would call it: `witwin/radar/propagation/` is held to a vocabulary that excludes
front-end terms, and `resolution` is also the word Channel's capability record
uses.

## Consequences

### Accepted, with closed-form evidence

| Fixture | Reference | Worst relative error | Bound |
|---|---|---|---|
| line of sight, 3.7 GHz sweep at 3 GHz | `sqrt(P) lambda/(4 pi d) exp(-j k d)` | 4.5e-5 | 1e-4 |
| lossy half space, same sweep | single-interface Fresnel `r_TE(f)` | 9.6e-5 | 2e-4 |
| 0.05 m slab, 2.4 Airy fringes | transfer-matrix Airy stack | 9.6e-5 | 2e-4 |
| single-tap range profile, wideband | inverse transform of the analytic band | 3.3e-5 | 2e-4 |
| single-tap range profile, narrowband | the pure delay kernel | 3.3e-8 | 1e-4 |

The float32 forward sets the floor: the reflection row carries about 80 cycles
of absolute phase at 3 GHz, and a complex64 result cannot beat a few float32
ULPs on that.

The slab's `|r|` swings by a factor of 14.2 across the sweep, its measured
envelope minima land in the same grid bins as the analytic ones, and the
narrowband offset law is wrong by a factor of 3.8 on the same grid. The
wideband tap departs from the pure delay kernel by 2.68 (bound 1e-2) and its
smear is 64% asymmetric, which a symmetric loss of resolution is not.

### Refused, with a number rather than a sentence

* **Dispersive materials.** Channel evaluates a `DispersionSpec` once, at
  compile, so a band would reuse `eps_r(f_ref)` at every column. Over the slab
  sweep a `PowerLawDispersion` with exponent -0.3 drifts `eps_r` by 32.7%, which
  is the error a frozen record would hide. The refusal comes from Channel and
  radar passes it through without re-wrapping it.
* **Rough / phase-screen materials.** Refused for a lifetime reason: the
  Kirchhoff table is built per material cache token and that token hashes the
  frequency.
* **Grid spacings below one float32 step.** Refused by Channel; the offset does
  not exist and a duplicate column under a different label would be worse than
  an error.
* **`transmission` and `diffraction`.** Not freezable components, so they cannot
  ride the fixed-topology inner loop at all. This is inherited from ADR-042, not
  decided here.

### Not changed

`row_valid` semantics, row identity, row order, pair segmentation, the `hybrid`
join-mode refusal, `ASPECT_PHASE_BUDGET_RAD`, `fmcw_beat.cu`, `pulsed_echo.cu`,
`two_way_join.cu`, and every narrowband number the package published before.
