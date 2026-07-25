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

### `carrier_hz` is a parameter, and both settings are exact

The beat operators take an explicit `carrier_hz`:

- `carrier_hz = fc` reproduces `dirichlet.cu`'s phase structure exactly, and is
  what the equivalence test uses.
- `carrier_hz = 0` is the production path for Channel-sourced weights, where the
  carrier already sits inside the natively computed coefficient. This placement
  is strictly more accurate: the coefficient's phase was formed against a float64
  delay inside a native kernel, whereas a float32 `tau` re-multiplied by 77 GHz
  loses roughly 2e-4 rad at 2 m and 1e-2 rad at 100 m.

Neither setting is a fallback for the other. The parameter exists because the
carrier has two legitimate homes, not because one of them sometimes fails.

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

## Acceptance evidence

- `tests/test_phase4_fmcw_beat_kernel.py::test_matches_the_dirichlet_path_when_carrier_is_the_carrier`
- `tests/test_phase4_fmcw_beat_kernel.py::test_tau_is_the_round_trip_delay_and_is_never_doubled`
- `tests/test_phase4_fmcw_beat_kernel.py::test_conjugation_is_the_only_channel_to_beat_conversion`
- `tests/test_phase4_fmcw_beat_ad.py` (VJP and JVP against the float64 oracle)
- `tests/test_phase4_binding_manifest.py` (declared, implemented, and manifested
  operators agree; every operator has an owner, a test, and a caller)
