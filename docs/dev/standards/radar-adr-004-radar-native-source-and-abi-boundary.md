# R-ADR-004: The `_radar_native` source and ABI boundary

Status: Accepted (Phase 4)

## Context

Radar's production compute policy allows exactly one native backend. Phase 4 adds
the first operator family that is not the Dirichlet spectrum, so the rules for
what that binary may contain, and what its operators look like, need to be
written down before the second family arrives and sets a precedent by accident.

## Decision

### Scope of the binary

`_radar_native` compiles Radar-owned target, two-way, Doppler, synthesis, and
front-end kernels. It must NOT:

- link RayD or Channel,
- contain copied numerical source from RayD or Channel,
- introduce a second RF binary or a second dispatcher/registry,
- expose a symbol without a Python owner.

Channel and RayD numerics reach Radar as compact typed CUDA tensors through the
consumer contract, never as linked code.

### Library name: an explicit decision, not drift

The new operators are registered in the EXISTING `witwin_radar_dirichlet_cuda`
Stable ABI library rather than a new one. `pyproject.toml`'s wheel artifacts,
`scripts/verify_cuda_binary_arches.py --stem`, `hatch_build.py`, and
`publish-witwin-radar.yml`'s `assert len(native) == 1` all assume a single native
stem. Splitting the stem is a packaging change, and packaging changes do not
belong in an AD spike.

`_radar_native` is the LOGICAL owner name for Phase 4. The physical rename, with
`dirichlet.cu` moving under `synthesis/`, is Phase-10 work. The binding manifest
records both names so the gap is visible rather than forgotten.

### Operator shape

- Out-parameter style, matching the existing family: outputs are preallocated by
  the Python owner and passed as `Tensor(a!)`.
- One Python owner per symbol. For the beat family that owner is
  `witwin/radar/synthesis/fmcw_beat.py`.
- Three registered operators per differentiable family: `_forward`, `_backward`,
  `_jvp`. Finite-difference derivatives in production are forbidden.
- All tensors CUDA, contiguous, and validated in the host wrapper before launch.

### The carrier has two homes, and both parameters are needed to say which

The beat operators take `carrier_hz` AND `carrier_rate_hz`. Exactly one is
nonzero:

- `carrier_hz = fc`, `carrier_rate_hz = 0`: the kernel owns the whole carrier
  phase. This reproduces `dirichlet.cu`'s phase structure exactly and is what the
  equivalence test uses.
- `carrier_hz = 0`, `carrier_rate_hz = fc`: the production path for
  Channel-sourced weights, where the absolute carrier phase already sits inside
  the natively computed coefficient. That placement is more accurate: the
  coefficient's phase was formed against a float64 delay inside a native kernel,
  whereas a float32 `tau` re-multiplied by 77 GHz loses roughly 2e-4 rad at 2 m
  and 1e-2 rad at 100 m.

Setting both to `fc` double counts the carrier and `FmcwBeatSpec.__post_init__`
refuses it.

`carrier_rate_hz` is not a second carrier and not a knob. It exists because a
Channel coefficient is frozen at the per-frame `tau_rt`, so the carrier phase it
holds is CONSTANT across chirps. Without this term the slow-time phase walk keeps
only `slope * (t_start - tau + t_m) * tau_rate`, and the measured intra-frame
Doppler is understated by 215x at fast-time sample 0 and 21x at sample 255 on the
fixture - silently, because the primal still looks like a plausible radar cube.
`carrier_rate_hz` applies the carrier to the delay CHANGE `(tau - tau_rt)` only,
which is exactly the missing `fc * tau_rate * t_c`.

This also makes the two derivative slots genuinely distinct:
`d(phi)/d(tau_rate)` is `t_c * (d(phi)/d(tau_rt) + 2 pi carrier_rate_hz)`, not
`t_c * d(phi)/d(tau_rt)`. The JVP and VJP companions carry both slots, and
`tests/test_phase4_fmcw_beat_ad.py` runs its float64-oracle comparisons at the
production placement so the extra term is exercised rather than zeroed.

Neither supported setting is a fallback for the other.

#### Superseded by this section

An earlier revision of this ADR stated that a single `carrier_hz` had two exact
settings. That was true only for `tau_rate = 0`. `carrier_hz = 0` combined with
a nonzero `delay_rate` was accepted without complaint and produced the
understatement above; the record is corrected here rather than in a footnote
because the wrong version named `carrier_hz = 0` "the production path".

### Phase convention and the single conversion site

Channel publishes `exp(-j k d)` under `exp(+j 2 pi f t)`. FMCW de-chirping
multiplies by the conjugate of the transmitted chirp, so the beat-domain phasor
advances with `+j`. The two conventions are conjugates.

The conversion has exactly ONE call site,
`witwin.radar.synthesis.fmcw_beat.channel_phasor_to_beat_weight`. A complex
target response is authored in the Channel convention, because it multiplies
transports authored there, and is converted along with them.

### `tau` is the round-trip delay

The beat operators consume `tau_rt` directly. The Dirichlet family's
`tau = 2 * distance / c0` is a monostatic assumption and is not reproduced. A
two-leg round trip already knows its own total delay; doubling it would give a
self-consistent, plausible, exactly 2x wrong range.

### Precision

Cycle counts accumulate in `double` and wrap to `[0, 1)` before `sincosf`. Fast
math stays off. At the fixture's ~47 cycles of `f_beat * t_m`, a naive float32
phase costs about 1e-2 rad, which is the magnitude of the gradients being
measured.

### Manifest

`ci/native-binding-manifest.json` maps every symbol to owner, contract test, and
end-to-end caller. It is seeded with the six Dirichlet operators plus the three
new ones. A symbol without a production caller is cleanup debt, not a feature.

## Consequences

The load-time presence check now names one operator per family, so a stale binary
fails at load instead of deep inside a kernel call. Adding a family without
updating the check is caught by
`tests/test_phase4_binding_manifest.py::test_the_load_check_covers_every_operator_family`.

## Alternatives rejected

**A new `_radar_native` library now.** Correct destination, wrong phase: it
changes four packaging surfaces and the release workflow's single-artifact
assumption for no AD-spike benefit.

**Hard-code the carrier inside the kernel.** Forces one of the two exact
placements to be unreachable and makes the Dirichlet equivalence test
unwritable.

**Forbid `delay_rate` when `carrier_hz = 0` instead of adding
`carrier_rate_hz`.** This removes the silent wrong answer but also removes
Doppler from the production placement, leaving no configuration that is both
phase-accurate and Doppler-correct. The caller would have to choose between the
float64-delay phase accuracy of a Channel weight and a usable Doppler cube.
Rejected: one extra `double` on three operators buys both.

## Acceptance evidence

- `tests/test_phase4_fmcw_beat_kernel.py::test_production_carrier_placement_carries_the_same_doppler`
- `tests/test_phase4_fmcw_beat_kernel.py::test_the_two_carrier_homes_cannot_both_be_used`
- `tests/test_phase4_fmcw_beat_kernel.py::test_matches_the_dirichlet_path_when_carrier_is_the_carrier`
- `tests/test_phase4_fmcw_beat_kernel.py::test_tau_is_the_round_trip_delay_and_is_never_doubled`
- `tests/test_phase4_fmcw_beat_kernel.py::test_conjugation_is_the_only_channel_to_beat_conversion`
- `tests/test_phase4_fmcw_beat_ad.py` (VJP and JVP against the float64 oracle)
- `tests/test_phase4_binding_manifest.py` (declared, implemented, and manifested
  operators agree; every operator has an owner, a test, and a caller)
