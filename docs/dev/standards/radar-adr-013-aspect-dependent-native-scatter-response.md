# R-ADR-013: An aspect-dependent scatter response is a kernel, not a composition

Status: Accepted (Phase 7)

## Context

Plan work item 6 composes micro-Doppler out of three things: scatter-site
velocity, aspect change, and target-specific phase.

The first is already carried. A scatter site is an ENDPOINT in this
architecture - the inbound leg is `TX -> site` and the outbound `site -> RX` -
so its `omega x r` or deformation velocity reaches `delay_rate` through the
propagation JVP the moment the R-ADR-012 seam exists. Nothing further was
needed for it.

The other two are per-path physics, and `TwoWayComposer.compose` refuses them
outright:

```
a geometry-dependent scatter response varies per path and must be evaluated
in a native kernel, not composed here
```

That refusal exists precisely to stop an aspect-dependent RCS becoming a Torch
expression in the composer, which is the shortest path from "one more feature"
to a second numerical backend on the hot path. Deleting it to ship item 6 would
be exactly the trade it was written to prevent.

Three further facts constrain the answer.

1. **The join already indexes its response.** `two_way_join.cu` reads
   `s[idx_s[k]]` per composed row and reduces the response's gradient through a
   CSR whose owner family is the site. Neither the forward nor the JVP entry
   constrains the response's length against the site count; only the backward
   does, through its declared `num_sites`.
2. **The legs publish a direction basis.** Channel's `PropagationGeometry`
   carries `field_direction`, the row's final-segment propagation direction, and
   the Radar adapter was simply dropping it.
3. **An aspect-dependent argument is a second Doppler term.** If `S` varies with
   aspect then `d(arg S)/dt` contributes to the target's spectrum, and the join
   publishes `tan_rate_rt = 0` on the stated policy that the WHOLE rate lives in
   `tau_rt`. Nothing in the pipeline would carry it.

## Decision

### 1. The refusal narrows by name; it does not disappear

`witwin.radar.scattering.NATIVE_ROW_RESPONSE_OWNERS` is an explicit
frozenset of fully qualified class names the composer will dispatch. A
geometry-dependent response that is not on it still raises `NotImplementedError`
with the original message.

The check is against the response's own declared `native_row_owner` string and
NOT against a `runtime_checkable` protocol. A protocol check can see that a
method called `evaluate_rows` exists; it cannot see whether a kernel or a Torch
expression runs behind it, and the whole content of the refusal is that
distinction. The list is likewise explicit rather than an `isinstance` against a
base class, because a subclass can override the evaluator and would inherit the
permission with it.

### 2. The response is a new native family, and the join is untouched

`scatter_response_aspect_{forward,jvp,backward}` in
`witwin/radar/cuda/kernels/scatter_response.cu` evaluate

```
ci = -dot(dir_in[i], axis[s])          incidence cosine at the site
co =  dot(dir_out[o], axis[s])         scattering cosine at the site
S  = amplitude[s] * clamp(ci)^n * clamp(co)^n * exp(-i * phase[s])
```

per composed row. `ci` is negated because `dir_in` is a propagation direction
and points INTO the site; getting that sign wrong gives a lobe that is exactly
backwards and still looks like a lobe. The clamp is physical - a negative cosine
is a direction on the far side of the aspect plane, which a separable forward
lobe does not illuminate - and its derivative is the right-hand limit, zero for
`n >= 1`.

`two_way_join.cu` is unchanged. When the response is row evaluated the composer
hands the join an identity site index and an identity CSR, so the site family
becomes the row family and `num_sites = path_count`. The identity tables are
built once at freeze, not per frame. **A row response therefore costs exactly
one extra kernel launch per frame** and adds no launch to the join.

The response crosses the autograd boundary as a real/imaginary PAIR, matching
the join and the beat family: no complex tensor at the seam, so the
conjugate-Wirtinger convention cannot be got wrong. That also means the composer
passes the pair straight through with no `torch.complex` and no `.contiguous()`
copy.

The VJP consumes the JOIN's own frozen CSR tables. One thread owns one gradient
slot, there are no atomics, and the summation order is a property of the frozen
composition - which is what keeps a bit-identical gradient comparison across a
permuted leg order a legitimate assertion. `exponent` is a host scalar and takes
no gradient; it selects the law.

### 3. A leg's published direction is a departure direction only at depth 0

`field_direction` is the row's FINAL segment direction. For the inbound leg that
is the arrival direction at the site at ANY depth, which is exactly what an
incidence cosine wants. For the outbound leg it is the arrival direction at the
RECEIVER, which equals the departure direction from the site only when the row
is line of sight.

`AspectScatterResponse` therefore REFUSES an outbound leg whose frozen rows are
not all line of sight, by name, from a host-known `outbound_max_depth` recorded
at freeze. No device column is read, so the refusal costs no host observation.

This is a real limitation and it is stated rather than papered over. Reading the
receiver-side direction as a departure direction for a reflected outbound row
would be wrong by the reflection angle and entirely plausible on a plot.
Carrying a true departure direction needs either a new consumer field or a
kernel that reconstructs it from the interaction positions, and both are
separate decisions.

### 4. The aspect phase rate is refused, not approximated

`witwin.radar.synthesis.contracts.require_aspect_phase_rate_bounded` refuses a
configuration whose `|d(arg S)/dt| * T_frame` reaches
`ASPECT_PHASE_BUDGET_RAD = 0.1`. It runs at construction time - once per epoch,
on the host - and the rate is DECLARED by the caller, exactly as
`PulsedEchoSpec.max_expected_delay_rate` is. Reducing over device rows to find a
maximum would be the hot-path device-to-host transfer the fixed-topology
capability exists to avoid.

This follows the pulsed spec's own precedent of refusing range migration rather
than approximating it. Phase 7 does NOT fold `d(arg S)/dt` into `delay_rate`:
that would change `tan_rate_rt = 0`, which is a numerical decision with its own
ADR and its own evidence.

The separable law shipped here has a real, non-negative magnitude and a per-site
constant phase, so its own aspect phase rate is identically zero. The
declaration exists for a caller that composes a further phase onto it and for
this law's successors, and the guard is a contract rather than a measurement.
That is recorded here so a reader does not mistake a vacuous measurement for a
tight one.

### 5. `axis` is required to be unit, and is not normalised

A kernel-side normalisation would add a division to every row and a quotient
rule to both AD companions, to hide a caller error that one host check catches
once. Worse, a silently renormalised axis returns the gradient of a different
parameterisation than the caller wrote.

## Consequences

- Item 6b and 6c are built, and `compose` still contains no per-path physics.
- Three new ABI symbols, each with a manifest entry, a direct contract test, a
  negative no-fallback test, and a production end-to-end caller. The caller-free
  symbol budget of 1 does not move.
- `RadarLegBatch` gains an optional `field_direction` column, aliased from the
  consumer. Optional because a fabricated leg has no geometry behind it; a
  response that needs one refuses a batch without it rather than inventing one.
- The composed frame costs one more launch when an aspect response is used, and
  exactly the same as before when it is not.
- Micro-Doppler ANALYSIS stays in Torch under `sigproc/microdoppler.py`, per the
  owner directive of 2026-07-25. A test asserts both directions: no native call
  in the analysis module, and no analysis vocabulary under `cuda/`.

## Alternatives rejected

- **Evaluate the aspect law in `compose`.** This is the thing the refusal
  exists to stop, and it would put a Torch numerical backend on the per-frame
  hot path where a kernel belongs.
- **Widen the join kernel to take a per-row response natively.** The join
  already indexes its response and reduces it through a CSR; a second code path
  for a shape it can already express would be duplicate physics with no
  numerical gain.
- **Refuse the response as an `isinstance` check against a base class.** A
  subclass can override the evaluator, so the check would grant permission to
  code it never saw.
- **Reconstruct the outbound departure direction in the response.** That is
  geometry, and reconstructing it in Torch is the same violation by another
  route. Doing it natively needs positions and interaction points the leg
  contract does not carry, which is a separate change.
- **Fold `d(arg S)/dt` into `delay_rate`.** A numerical change to a frozen
  contract, mixed into a feature commit. Refusing the configuration is the
  honest interim and is what this repository already does for range migration.
