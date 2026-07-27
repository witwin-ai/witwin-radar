# R-ADR-020: The production scene binding, its Channel crossing, and where scatter sites come from

Status: Accepted (Phase 11)

## Context

Phase 11 work item 1 says `Radar.simulate` delegates to the new Core `Scene` ->
`CompiledScene` -> propagation -> two-way -> synthesis pipeline. Every step of
that pipeline was already production before this phase; what was missing was the
first one. Three specific things had no production owner at all, and all three
existed only inside `tests/support/multi_endpoint_world.py`:

1. **The compile crossing.** `compile_fixture_scene` (`:127-134`) and
   `compile_snapshot` (`:210-223`) were the only callers of
   `witwin.channel.scene.compile` anywhere. No Radar production module could
   call it: `tests/test_phase4_import_boundary.py:390`
   (`test_only_the_adapter_crosses_the_channel_boundary`) pins
   `ChannelPropagationAdapter` as the sole Radar module allowed to name
   `witwin.channel`, and `propagation/epochs.py:336-345` takes `compile_scene`
   as a *callable argument* precisely so that it is not a second crossing.
2. **The endpoint specs.** `endpoint_batch` (`:226-262`) turned
   `(stable_id, position)` pairs into a `RadarEndpointSpec`. Nothing in
   production did.
3. **The stable IDs.** The fixture hard-codes them
   (`multi_endpoint_geometry.TX_A_STABLE_ID` and friends). Radar had no
   allocator, and `witwin.core.identity` deliberately does not provide one:
   `new_*_id()` is a process counter whose values depend on construction order,
   and its own module docstring says choosing a reproducible identity scheme is
   the caller's decision.

A fourth question had no owner in any repository. `simulate(scene, ...)` must
answer **"where are the scatter sites?"**, and a Core `Scene` describes
structures, not scatterers. `MultiEndpointSpike` sidesteps it by taking the
sites as an explicit constructor argument.

## Decision

### 1. The compile facade lives in the adapter module

`witwin.radar.propagation.channel_consumer.compile_scene(scene_or_snapshot, *,
reference_frequency_hz)`.

The alternative was to allowlist a second Radar module that names
`witwin.channel`. It is rejected: one crossing is a property worth more than the
convenience, and the module that already owns this boundary is the honest place
for a second Channel lifecycle entry.
`test_only_the_adapter_crosses_the_channel_boundary` is **unchanged** and still
asserts a single file.

Three shape decisions follow from that placement:

* It is a **module function, not an adapter method.** It produces the compiled
  scene an adapter is *constructed with*, so there is no adapter in existence
  when it is called. A `staticmethod` would suggest a per-adapter binding that
  does not exist, and the adapter's `__init__` parameter set is asserted by
  equality (`tests/test_phase6_config_boundary.py:300-330`) precisely so that
  this kind of surface growth is reviewed rather than tolerated.
* The `witwin.channel.scene` import is **function-local.** Importing
  `witwin.radar.propagation.channel_consumer` therefore still loads nothing
  beyond the consumer facade's own closure, which is what
  `test_radar_adds_nothing_to_the_consumer_facade_closure` measures at runtime,
  independently of any allowlist.
* `propagation/epochs.py` is **not edited.** It keeps taking `compile_scene` as
  a callable, its design comment stays true, the compile count stays observable,
  and the production driver simply passes this function in.

The static-closure allowlist `ALLOWED_CHANNEL_IMPORTS` in
`tests/test_phase4_import_boundary.py` gains `witwin.channel.scene` and
`witwin.channel.scene.compile`. That list is asserted by equality on purpose, so
a new name has to be added deliberately; this is that deliberate addition, and
it is the *only* change to that file's assertions.

### 2. A reference-frequency mismatch is refused, never recompiled

`compile_scene` validates the scene it gets back against the frequency that was
asked for, and `require_reference_frequency(compiled_scene,
reference_frequency_hz)` is published for a driver that receives a compiled
scene from somewhere else. Both are host-only; neither launches anything.

The refusal is Channel's own `CompiledScene.require_reference_frequency`, quoted
rather than re-derived, because the exactness rule (a hex comparison, not a
tolerance) belongs to the side that owns the compiled constant. Channel already
refuses a mismatch inside `evaluate`/`reevaluate`; checking at the binding means
a scene that came out of the compile cache at another frequency is refused where
the two numbers are still readable side by side. There is deliberately **no**
implicit recompile: Channel's CLAUDE.md forbids it, and a silent second compile
would hide a caller that is confused about which radar it is simulating.

`ChannelPropagationAdapter.__init__` is intentionally left alone.
`tests/test_phase4_adapter.py:295-325` constructs a deliberately mismatched
adapter and pins that the refusal arrives at `reevaluate`; moving the check into
the constructor would break that pin for no gain, since the constructor is not
the only way to acquire a compiled scene.

### 3. Stable IDs are three contiguous blocks over a declared base

`StableIdAllocator(transmitter_base, receiver_base, site_base)` with defaults
`1_000_000`, `2_000_000`, `3_000_000`. An ID is
`base + array_index` and therefore a pure function of `(role, index)`: not of
construction order, not of the process, not of how many frames have run.

That is the property a frozen leg topology depends on. `FrozenLegTopology`
records `source_id` and `sink_id`, the two-way composer joins on identity rather
than on row position, and a replay in a later frame has to be able to say it is
talking about the same endpoints. A process counter cannot say that; two runs of
one script would disagree.

The defaults sit far above Core's own zero-based counters so that a radar
endpoint ID cannot collide with a structure, material, assignment or antenna ID
in the same world. Overlap between the three blocks is checked at
`allocate(...)`, not at construction, because whether two bases collide depends
on the counts. An overlap is an error: two endpoints sharing one ID is not a
smaller answer, it is a join that pairs the wrong rows and still publishes a
full result. Site IDs may be overridden per policy, and an override that lands
in the transmitter or receiver block is refused the same way.

Allocation is one-time setup. `bind_radar_world` runs once per topology epoch,
builds three small constant tensors, and reads nothing back to the host.

### 4. Scatter sites are declared, never derived from geometry

`ScatterSitePolicy` has exactly two sources:

* **`explicit`** - the caller hands over an `(S, 3)` tensor or a sequence of
  triples. A live tensor is passed through **untouched**: not moved, not cast,
  not made contiguous, because each of those is a new node that would leave the
  caller holding a tensor that is no longer the one the legs differentiate
  through. A device or dtype mismatch is refused with a message rather than
  fixed with a silent copy.
* **`structure_anchor`** - one site per selected structure at the world
  translation `StructureState.rigid_motion` publishes in the snapshot. This is a
  Core-owned world quantity read as it stands. `torch.stack` preserves whatever
  tape it carries, so a site riding a `LinearTrajectory` reaches both legs
  differentiably without Radar ever forming a position of its own. Selection and
  ordering are by ascending structure ID, so the site array order is a function
  of world identity rather than of the snapshot's tuple order.

A structure with no rigid motion has no Core-owned anchor and is refused by name.

**The named deferral.** Deriving a site by *sampling* a structure - a surface
sample, a facet centroid, a bounding-box centre, a visibility-weighted scatterer
set - is out of scope for Phase 11 and the refusal message says so. A sampling
rule is a geometry algorithm, and a geometry algorithm written as a Torch
expression on the production path is exactly what the single-backend policy
exists to keep out of Radar: geometry belongs to Channel and RayD. Closing this
deferral needs (a) a decision about which repository owns radar scatterer
extraction, (b) a native or Channel-side owner for the sampling itself, and (c)
an accuracy statement relating the sampled site set to the structure's radar
cross-section, since a scatter site is a point re-radiator and a surface is not.
None of the three is a Phase-11 question.

`power_w` defaults to exactly 1 W. The site is a re-radiator, not a second
transmitter: the whole target strength lives in the two-way join's
`S = sqrt(4 pi sigma) / lambda`. A site excitation of anything else multiplies
that factor again, and at a transmit power of 1 W the extra `sqrt(P)` would be
numerically invisible - which is how a squared transmit power ships.

### 5. `RadarWorldBinding` is one type because the site tensor is one object

The sites are the **sinks** of the inbound leg and the **sources** of the
outbound leg. `RadarWorldBinding` hands the *same* `positions_m` object to both
specs and asserts that identity in `__post_init__` rather than assuming it.
Rebuilding the tensor for the second role halves a reverse gradient and zeroes a
forward tangent, and both failures look like a plausible answer. Four loose
arguments cannot state that invariant; one frozen record can.

The record also re-states the Channel power contract at its own boundary - a
source carries `powers_w`, a sink must not - so the complaint names the leg
endpoint the caller actually built.

The ID tuples on the record are host tuples, not tensors, because the composer's
declared identity lists are host lists and reading them back out of a device
tensor would be a host observation on a path that has none.

### 6. The configuration surface: block keywords, not new flat fields

`components` and `max_depth` reach the propagation block through
`RadarSystemConfig.from_radar_config(..., components=..., max_depth=...)` and
`RadarSystemConfig.with_propagation(...)`. They do **not** become flat
`RadarConfig` fields.

`RadarConfig` is the file format, and the five-block split exists because a field
that nobody groups is a field anybody may read. `components` and `max_depth` are
propagation-request quantities with exactly one legitimate reader, and until this
phase they could only ever be their `PropagationConfig` defaults because nothing
surfaced them at all. The scene-driven entry takes them as keywords and applies
them with `with_propagation`, which returns a new configuration: a per-solve
override that edited the radar's stored configuration would silently change every
later solve as well.

`reference_frequency_hz` is deliberately not overridable there. It is tied to the
array element spacing by `RadarSystemConfig.__post_init__` and to the compiled
scene by Channel, so changing it would produce a configuration that is refused
later rather than one that means something else.

The waveform block becomes **selectable**:
`from_radar_config(..., waveform=<block>)` uses the block verbatim, and `None`
builds the FMCW block from the flat fields exactly as before, bit for bit. That
is what makes the OFDM and pulsed synthesis owners reachable from a
configuration instead of only from a hand-assembled `RadarSystemConfig`. It stays
a keyword rather than a flat discriminator because an OFDM block shares none of
the FMCW block's fields, and inferring which half of a merged flat form is live
is the exact defect the stored `kind` removed.

**Known limitation, handed to the `simulate-entry` stage.** `Radar.__init__`
calls `_init_axes`, which calls `RadarSystemConfig.axes()`, which raises
`NotImplementedError` for any non-FMCW waveform. A non-FMCW `RadarSystemConfig`
is therefore constructible and usable by the synthesis owners today, but a
`Radar` object carrying one is not. Making `Radar` accept a non-FMCW waveform is
a `radar.py` change and belongs to the stage that owns that file.

## Consequences

* One Radar module names `witwin.channel`, and it now names two things from it.
  The boundary property that matters - a single crossing file, no solver, no
  enumerated engine, no internal contracts, no raw extension - is unchanged and
  still asserted.
* `witwin/radar/scene_binding.py` joins the scanned module set in
  `tests/test_phase4_import_boundary.py`, so it is held to the same
  no-host-observation and no-Dr.Jit rules as the rest of the per-frame path.
* Radar has a reproducible endpoint identity scheme for the first time. Two runs
  of one script agree on every `source_id` and `sink_id`, which is what makes a
  frozen topology comparable across processes.
* Mesh-derived scatter sites remain unavailable. A caller with a mesh target
  declares its sites, or gives the structure a rigid motion so that Core
  publishes an anchor. The refusal names the deferral rather than approximating.

## Related

* R-ADR-005 - the adapter holds a `CompiledScene` as an opaque token.
* R-ADR-009 - `witwin.core.Mesh` defaults `recenter=True` and rewrites authored
  world coordinates. Every mesh a binding consumes must be built with
  `recenter=False`; `tests/test_phase11_scene_binding.py` pins that the default
  still bites, measured on `world_vertices` (a compiler consumes those;
  `Mesh.vertices` returns the authored tensor unchanged whatever `recenter`
  says).
* R-ADR-014 - the scene-driven epoch cadence. `SceneEpochLoop` is the consumer of
  `compile_scene`, and `bind_radar_world` is what its `bind` callback calls
  before freezing the legs.
* R-ADR-016 - the scene component taxonomy. `components` is the propagation-side
  half of that vocabulary, and this ADR is what finally makes it settable.
