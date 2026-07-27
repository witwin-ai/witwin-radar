# R-ADR-018: AD capability states, the non-differentiability wall, and first order only

Status: Accepted (Phase 9)

## Context

By the end of Phase 8 the Radar package had a working differentiable chain and
no statement of what it was. Phase 4 proved the vertical route from a Core leaf
to a scalar loss; Phase 5 added the two-way join's companions; Phase 6 gave
every synthesis family a primal, a JVP and a VJP; Phase 7 added kinematics duals
and the aspect response; Phase 8 added wideband frequency-offset AD. Each of
those phases proved the cells it introduced.

What none of them produced was an answer to the question a caller actually asks:
**for this leaf, in this mode, on this route - does a derivative exist, and if
not, what happens?**

The survey that opened Phase 9 found 89 Radar cells and 295 Channel cells. Most
of the Radar ones worked and had no test. A smaller and much worse group did
something else:

- `CFAR` published `d(threshold)/d(power)` with a measured absolute sum of
  3595.02 - the derivative of a value at a frozen threshold comparison;
- `point_cloud` published `d(energy)/d(cube)` of 58.36 - the derivative of a
  value at a frozen `argmax`;
- `DetectionFrame`'s refusal checked `requires_grad` only, so a forward dual
  walked straight through it, and it fired AFTER `point_cloud` had already built
  a full result;
- every frontend and waveform spec scalar swallowed a `requires_grad` tensor
  through `float(...)` with at most a `UserWarning`, and returned `grad = None`;
- `create_graph=True` succeeded at every autograd boundary and handed back a
  silently detached first gradient;
- a `grad_output` carrying a forward tangent - a mixed second derivative
  request - was accepted everywhere, computed the correct first derivative, and
  published a tangent of `None`, which a caller reads as an exact zero mixed
  partial with no error anywhere;
- `SmplPoseDeformation` detached its vertices before building the `Mesh`, so a
  pose gradient could never reach the compiled scene and nothing said so;
- `frequency_offsets_hz` refused a tensor grid but accepted a sequence whose
  ENTRIES were tensors, so a marked offset ran a whole wideband solve and
  returned no gradient.

Every one of those is the same defect wearing different clothes: **a question
that could not be answered was answered anyway**. That class is worse than an
unimplemented feature, because an unimplemented feature fails and a plausible
zero does not.

## Decision

### 1. Four target states, and a cell with no declaration is a defect

Every `(route, leaf-or-output, mode)` cell in the Radar package is in exactly
one of four states:

| State | Meaning | Required evidence |
|---|---|---|
| `SUP` | Supported. A nonzero derivative is published and it is correct. | A named test at the boundary that publishes it, validated against finite differences, an independent float64 oracle, an analytic closed form, a jvp/vjp adjoint identity, or - for a structural claim such as row identity or context aliasing - a structural declaration test. |
| `ZERO` | Structurally zero. The leaf genuinely does not enter this physics, and exact zero is the complete answer. | A named test asserting EXACT zero, plus a falsifier showing the zero is a fact about the physics rather than a severed wire. |
| `REF` | Refused. Fails loudly BEFORE any numerical work and before any result object exists. | A named test asserting the raise, its owner, and that no partial result was produced. |
| `DECL` | Declared non-differentiable OUTPUT. The published tensor deliberately carries no graph and no tangent. | The capability record names the field and the route, a test pins the declaration against observed behaviour, and the contract document states it. Legal for outputs only, never for inputs. |

**`SILENT` is not a state and never becomes one.** A cell that answers with a
severed derivative, a `grad = None`, or a plausible zero and does not fail is a
defect. A cell nobody has decided is also a defect - not a status, not a
backlog entry.

There is no `TODO` state. A cell this phase is deliberately not doing is `REF`
or `DECL` with a named deferral, and every deferral carries a reason and a
follow-up owner.

`docs/dev/radar-ad-capability-matrix.md` is the authoritative record, and
`tests/test_phase9_capability_matrix.py` is what makes it authoritative rather
than decorative: it parses the document, closes the four vocabularies, resolves
every cited test node id against the source, freezes the row count per section,
and pins the mirrored Channel rows against the live `capabilities()` record.

### 2. The wall sits at the first discrete decision, not at "post-processing"

`witwin/radar/ad_contracts.py::refuse_derivative` is the single guard, and it
checks BOTH `requires_grad` and `unpack_dual(...).tangent`, at function entry,
before any compute.

Above the wall, and differentiable: matched filter, range profile, range-Doppler,
beam cube, MUSIC pseudo-spectrum, TDM compensation. All of them are smooth
functions of their input, all of them are Torch under R-ADR-007's DSP
exception, and all of them stay live.

Below it, and refused: ADC quantization (`round`), CFAR (a threshold
comparison), peak selection (`topk`), point cloud (`argwhere`, and a value read
at a frozen index), AoA bin selection (`argmax`), detection and tracking (a
discrete association).

"Post-processing" would have been a much later and much less defensible line: it
would have refused a range FFT, which is linear and whose derivative is exactly
what a caller optimising a waveform wants.

**CFAR's threshold derivative is refused deliberately, and it is the decision in
this ADR most likely to be questioned.** `d(threshold)/d(power)` is technically
well defined - the threshold is a smooth function of the training cells - and
was measurably nonzero before Phase 9. It is refused because a caller who takes
that gradient is not differentiating the detector; the detector's output is the
comparison, and the comparison's derivative is zero almost everywhere and
undefined on the boundary. A gradient that flows through the threshold and not
through the comparison is a plausible number describing a different function. A
straight-through estimator or a soft-threshold surrogate is a legitimate future
modelling choice, and it is an explicit design with its own ADR - never
something a stage may choose by not refusing.

Every refusal message says WHY, not merely that: which discrete decision the
stage is built on. "Not differentiable" alone sends a reader looking for a bug
rather than for the modelling decision.

### 3. Every noise and receiver continuous parameter has a decision

`witwin/radar/host_parameters.py::require_host_floats` refuses **any**
`torch.Tensor` spec value, not only a grad-carrying one. Raising on any tensor
is deliberate twice over: a tensor that happens not to require grad today is the
exact input that starts requiring grad tomorrow, and `float()` on a device
tensor is a silent host synchronisation as well as a silent detach. The previous
`UserWarning` path is deleted, not kept alongside.

| Parameter | Decision | Reason |
|---|---|---|
| `LnaSpec.gain_db` | `REF` | Device configuration, not scene state. Its derivative would be perfectly well defined - a smooth multiplicative factor on the whole signal - and the native frontend operator carries no tangent or gradient slot for it. Deferred with a named owner rather than pretended away. |
| `AgcSpec.target_rms` | `REF` | A control setpoint. The stage is already non-linear in the signal, and a global AGC makes a magnitude loss EXACTLY constant - measured - so a gradient here would be a correctly-zero number with a misleading name. |
| `NoiseSpec.noise_figure_db`, `antenna_temperature_k`, `bandwidth_hz`, `phase_noise_dbc_per_hz` | `REF` | Each parameterises a counter-based Philox draw. A pathwise derivative through an RNG stream is not defined by any accepted contract here. A reparameterised noise model is the shape that would make these meaningful and is a separate ADR. |
| `PortSpec.reference_impedance_ohm` | `REF` | A unit convention. |
| `AdcSpec.full_scale`, `AdcSpec.bits` | `REF` | Behind the ADC wall regardless of their own smoothness. |
| FMCW slope, carrier and period; OFDM `subcarrier_spacing_hz`; pulse-shape scalars | `REF` | Host declarations that SELECT a waveform. Several of them change the SHAPE of the output as well as its value, and a derivative taken across a sampling-grid change is not the derivative of a fixed function. |
| `AspectScatterResponse.exponent` | `REF` | Selects the scattering law rather than parameterising it continuously in a way any consumer optimises. |
| `frequency_offsets_hz`, including a SEQUENCE whose entries are tensors | `REF` | The declared band is a host grid. This one was found by Phase 9's own combined-input matrix and closed in the same phase; before the fix a marked offset ran the whole wideband solve and returned no gradient with only a Torch `UserWarning`. |
| `ScalarRcsResponse.from_rcs(sigma_m2=...)` | **`SUP`** | The one exception, below. |

### 4. `from_rcs` is the one capability added, and `torch-orchestration` is legal there

A radar cross section is the canonical inverse-design leaf - "how big does this
target have to look" - and it is the only configuration scalar in the package
that is genuine scene state rather than a device, waveform or unit declaration.
`rcs_amplitude` therefore accepts a 0-dim tensor and produces the amplitude as
`torch.sqrt(4 pi sigma) / lambda` with its graph intact.

This does not breach the single-backend policy, and the reason is structural
rather than a plea:

- it runs **once per response**, not once per path, and it is off the per-path
  loop entirely;
- it produces **one number** that the response broadcasts; every per-path
  product downstream of it is still evaluated by a native kernel;
- it is result **construction**, not a second numerical owner: there is no
  native `rcs_amplitude` for it to be a Torch replay of.

The capability matrix records its mechanism as `torch-orchestration`, which
R-ADR-007 allows outside the hot path, and
`tests/test_phase6_no_torch_physics.py` pins `scattering/rcs.py`'s whole matched
Torch set by EQUALITY, so a second arithmetic expression in that module fails.

Two consequences of the tensor route are deliberate. `requires_grad=True` is
refused together with a tensor cross section, because the leaf is `sigma_m2`
which the caller already marked and the derived amplitude is not a leaf. And the
tensor route does not range check its input: a value check is a host read, and
this module sits inside the import boundary's no-host-observation scan precisely
so a per-frame construction cannot hide a synchronisation. A non-positive tensor
therefore produces `nan`, visibly, rather than a clamped plausible number.

### 5. First order only, everywhere, with `torch.is_grad_enabled()` as the mechanism

Radar publishes first derivatives and nothing higher. Every second-order request
fails loudly, before any partial second-order result, naming the owner.

`witwin/radar/ad_contracts.py::first_order_only` decorates every registered
`backward` in the package - ten of them - and refuses three compositions:

- **reverse over reverse.** `create_graph=True` is exactly what leaves grad mode
  enabled while a backward runs, so `torch.is_grad_enabled()` inside the
  decorator is a precise detector that fires before any launch. Without it the
  first gradient came back silently detached and the failure surfaced one step
  later as a generic Torch message naming Torch, or - with `allow_unused=True` -
  as a silent `None`.
- **forward over reverse.** A `grad_output` carrying a forward tangent is a
  mixed second derivative request. This is the worse of the two: before Phase 9
  it did not fail at all, the gradient value was right, and the mixed partial
  read as an exact zero.
- **a `grad_output` that itself requires grad**, which is the same request
  spelled differently and equally unanswerable.

`torch.autograd.function.once_differentiable` is applied UNDERNEATH, from the
decorator rather than from every call site, as defence in depth. The nesting
order is not cosmetic and was measured: `once` runs the backward body inside
`torch.no_grad()`, so a check written inside the body would see grad mode
already off even under `create_graph=True`.

Nested forward levels stay Torch-owned. Torch refuses a second `dual_level`
itself, Radar adds nothing, and that absence is pinned by a test so it is a
recorded decision rather than a gap.

The policy is queryable rather than folklore: Channel's
`capabilities().supports_higher_order_ad` is `False` and Radar pins it.

### 6. The tape stays inside its owner

Ten autograd contexts exist in the package, in nine files. Each is created by
its owner's `setup_context` and released when the graph that holds it is
released. Two rules:

- **no public result field holds a tape** - not the saved tensors, not the
  context. A `grad_fn` is not a tape and is supposed to be there; a field
  holding the context turns a data record into a handle on somebody else's
  memory and makes the tape's lifetime the result's lifetime.
- **no module outside a tape's own owner reads one.** There are exactly twenty
  `ctx.saved_tensors` reads in the package, in the eight owner files, each
  inside a `backward` or a `jvp`.

Both are tested (`tests/test_phase9_tape_non_leak.py`) rather than inspected,
and the ledger `docs/dev/ad-tape-and-budget-ledger.md` records every owner's
saved tensor names, a symbolic byte formula, the measured bytes, launch counts,
backward wall time and - the column the document exists for - the context
lifetime.

### 7. Backward budgets are pinned, and only five of them

One backward launch per forward launch at every boundary; the `_compose_band`
tape as a predicted linear law in the band column count; the full FMCW
pipeline's backward wall time and peak allocation; and the Channel `reevaluate`
inner loop forward and reverse. Nothing else. A pinned wall-time number is
maintenance debt and the cheapest way to make a budget suite useless is to pin
everything.

Measured constants are the WORST median over four independent processes, with
the headroom applied on top of that. No existing budget was weakened.

## Consequences

- A caller can ask, before making a call, whether a leaf is live: for Channel
  cells through `capabilities()`, and for Radar cells through the matrix
  document, which a test keeps true.
- Several things that "worked" before Phase 9 now raise. That is the point.
  CFAR, point cloud, AoA, tracking, every spec scalar and every velocity leaf
  refuse where they previously returned a number or a `None`.
- The refusals are all at function entry, so no caller is left holding a
  half-built result. `tests/test_phase9_processing_wall.py`'s `_ComputeWatch`
  measures that rather than asserting it, and is calibrated against the same
  stages running normally so the zero is never vacuous.
- Adding a leaf now costs a matrix row. Forgetting one fails
  `tests/test_phase9_capability_matrix.py::test_every_section_carries_its_frozen_row_count`.
- Adding a tenth-plus autograd `Function` costs a boundary entry in
  `tests/support/ad_boundaries.py`. Forgetting it fails
  `tests/test_phase9_backward_budget.py::test_the_launch_ledger_covers_every_tape_owner_in_the_package`
  and takes the higher-order rejection coverage with it.

## Deferrals

Nine, each with a reason and a named follow-up owner, recorded in the "Deferred"
section of `docs/dev/radar-ad-capability-matrix.md` and summarised here:

| Deferral | Follow-up owner |
|---|---|
| `field_direction` on transmission, wedge and coupled diffraction | Channel, in the RayD ADR that ADR-043 defers to |
| discovery-route geometry liveness | Channel, ADR-043 |
| a pose derivative into the compiled scene (SMPL) | Radar `geometry/smpl.py` plus `witwin.core` Mesh construction, as one accepted design |
| a material-only forward tangent | Radar `propagation/channel_consumer.py`, whose dead-tangent guard owns the decision |
| the LNA voltage gain as a leaf | Radar `frontend/`, one extra slot on the fused operator |
| waveform-parameter optimisation | Radar `synthesis/`, as a new R-ADR |
| a pathwise derivative through the noise realisation | Radar `frontend/`, as a reparameterised noise-model R-ADR |
| a sensor pattern table as a leaf | Radar `sensors/` |
| the Channel diffraction primal `IndexError` | Channel, as a primal reachability fix separate from any AD work |

## Consistency with Channel

This ADR and Channel's ADR-043 state the same policy from the two sides of the
consumer boundary and must not diverge. Channel owns the propagation cells and
publishes them in `PropagationCapabilities`; Radar owns everything above the
adapter and mirrors the Channel rows it consumes. The mirror is pinned against
the live record by
`tests/test_phase9_capability_matrix.py::test_the_mirrored_channel_rows_agree_with_the_live_capability_record`,
so a Channel change that contradicts this document fails in Radar's suite rather
than in a reader's expectations.

Both sides agree that: `SILENT` is not a state; diffraction advertises `{"none"}`
and its AD column is refused pre-compute rather than advertised and unreachable;
`field_direction` is live for `{los, reflection}` under a frozen topology and
declared elsewhere; discovery-route geometry is declared rather than live; and
higher-order AD is refused, in both packages, at the owner that cannot answer.

## Alternatives considered

**Put the wall at "post-processing".** Rejected: it is a directory name, not a
mathematical property, and it would refuse a range FFT while admitting a
threshold.

**Make CFAR differentiable through a soft threshold.** Rejected for this phase,
not forever. It is a modelling decision that changes what the detector IS, and
shipping it behind a flag would mean two detectors with the same name.

**Warn instead of refuse on a tensor spec value.** Rejected: that is exactly
what the tree did, and the survey found the warning had been silently accepted
in production paths for three phases.

**Accept `create_graph=True` and let Torch produce whatever it produces.**
Rejected: what it produces is a detached first gradient and a `None` second one,
which is the plausible-zero defect at its purest.

**One capability matrix per waveform.** Rejected: the waveform is chosen AFTER
`prepare_fixed_topology`, so the AD scenario is waveform independent, and three
matrices would have meant three topologies where the acceptance criterion asks
for one.
