# R-ADR-016: Scene components as a sidecar over one topology

Status: Accepted (Phase 8)

## Context

A radar frame is a target sitting inside an environment that returns energy of
its own. Everything a user wants to do with that - export the clutter alone,
measure the target against it, recombine the two coherently, feed one of them
to a suppressor - needs the composed rows to carry a NAME. Until Phase 8 they
did not.

The ingredients were all present and all native. Channel publishes
`component_id`, `depth`, `primitive_sequence` and `material_sequence` on every
frozen leg row, and `RadarPathTopology` publishes `inbound_row` and
`outbound_row`, so the map from a composed row back to the two leg rows that
produced it has always been recoverable. What was missing was the name, the
export, and the statement of what putting the pieces back together means.

Phase 5's plan named a third `join_mode`, `hybrid`, that would distinguish
target echo from environment clutter from multi-interaction echo. It was never
built: the only occurrence of the word in the tree is a test asserting that the
mode is refused. Phase 8 inherits that unpaid acceptance criterion.

Three facts shape the answer.

1. **The waveform kernels already accumulate linearly over the rows of a pair
   segment**, and `synthesize_fmcw_beat` already masks a dead row's WEIGHT with
   `torch.where` before the launch - explicitly so the row is inert in the
   primal and carries no gradient. A component mask is that operation with a
   different predicate, so exporting a component needs no kernel change at all.
2. **The freezable component set is `{los, reflection}`.** A clutter class that
   needs `transmission` or `diffraction` cannot ride the fixed-topology inner
   loop and must rediscover. This is a capability constraint, not a discovery.
3. **A fixed-winner replay is subtractive.** A row that stops existing is
   published inert; a row that STARTS existing is absent and nothing on the
   device reports its absence.

## Decision

### 1. The discriminator is a SIDECAR, not a column

`RadarComponentIndex` holds `class_id: [K] int32` on the device, the ordered
host tuple of class names, the per-class host counts, and the declaration that
produced it. `RadarPathTopology` gains nothing.

The reason is the acceptance criterion. "Processing does not change propagation
row identity" is directly assertable only if every component export shares the
same topology OBJECT, and `select_component` asserts exactly that with `is`. A
new column would have respecified the frozen dataclass and every identity test
that keys on it, and would have made the criterion a claim about equality
rather than about identity.

The index is built ONCE per topology epoch from the frozen topology and the
frozen legs. It performs the same sanctioned freeze-time host read that
`paths/_identity.py` already performs, after the consumer has synchronized, and
`test_phase4_import_boundary.py` records the allowance by name. Per frame it
reads nothing: `mask` is a device comparison and `count` returns a host int
decided at build time.

### 2. `hybrid` is paid by a component index over a `multipath` join

`join_mode` stays `{direct, multipath}` and the refusal test for `hybrid` stays
green and unweakened. The Phase-5 distinction arrives as a DECLARATION over one
topology instead of as a third composer:

| class | predicate |
|---|---|
| `target` | a scatter site in the declared target set, no declared clutter interaction, no leg deeper than the declared depth |
| `environment_clutter` | an interaction with a declared clutter material slot, or a site in the declared clutter set |
| `direct_leakage` | no scatter site AND no declared clutter interaction |
| `multi_interaction` | either leg deeper than the declared depth |

A third join mode would have been a second composer, a second frozen order and
a second set of index tables producing the same rows. The declaration is
strictly more expressive: the same frozen topology yields a different partition
under a different declaration, at no propagation cost.

**`direct_leakage` is narrower than "no scatter site", deliberately.** The
transmitter-to-wall-to-receiver path also has no site, and filing it as leakage
would put the strongest environment return in the frame under the name of the
antenna coupling term. On the multi-endpoint fixture the direct route has four
rows and exactly one of them is that reflection; the test asserts the split.

**A structure is named by its compiled material slot.** That is what a frozen
row carries - Channel publishes `material_sequence` and no structure column -
and the consequence is stated rather than hidden: two structures sharing one
material are indistinguishable here, and a scene that needs them apart must
give them different materials or declare its clutter by site.

### 3. Every row belongs to exactly one class, checked at build

The four predicates are evaluated independently rather than as an if/elif
ladder. A ladder makes "exactly one class" true by construction and therefore
unassertable. Evaluating all four lets the build refuse:

- a row matching NONE, which is an undeclared site. It would otherwise vanish
  from every export and from the coherent recombination law, silently.
- a row matching TWO, which is a declaration that contradicts itself. The
  commonest case - a site declared both target and clutter - is refused earlier
  still, in `ComponentDeclaration.__post_init__`.

### 4. Coherent recombination is a tolerance, not `torch.equal`

`sum_j cube(component_j) == cube(every row)` up to float re-association of the
partial sums, and no further. The kernel accumulates in the same row ORDER in
every selection and a masked row contributes a literal `0.0` in its own slot,
but `(a + 0 + c) + (0 + b + 0)` is not `(a + b + c)` in float32.

The tolerance is derived: `atol = 8 * eps_f32 * K * max|per-row contribution|`,
`rtol = 1e-6`. Measured on the multi-endpoint fixture at `K = 11`:

| quantity | value |
|---|---|
| residual, absolute | 3.2539e-11 |
| residual, relative to peak | 8.665e-08 |
| derived `atol` | 2.4797e-09 |
| margin | 76x |

A companion test asserts that the recombination is NOT bitwise, so that if it
ever becomes bitwise the tolerance is tightened deliberately rather than left
over-generous.

### 5. Incoherent combination is a power-domain helper, and no kernel gains a flag

`witwin.radar.processing.combine_incoherent(cubes) -> sum_j |cube_j|^2`,
post-synthesis Torch, returning a REAL tensor.

An "incoherent" flag on a waveform kernel would put a second summation semantic
inside a fused op whose whole contract is that it sums complex amplitudes over
a pair segment. The refusal is structural, not stylistic.

**DEFERRED, with the reason.** The physically honest incoherent model is not a
power sum: it is a per-realization random phase drawn into the scatter response,
so an ensemble of frames averages to the power sum while each individual frame
remains a legitimate coherent field with speckle. That is a native response
change needing an RNG and a seed contract consistent with the frontend's, and
it is out of Phase-8 scope. Phase 8 ships the power-domain law and says so
rather than shipping a random phase with an undeclared seed.

The test makes the semantic visible rather than only asserting the formula:
where one component owns a range bin outright the two laws agree to within
`2/sqrt(power ratio)` (measured 1.75e-2 at a ratio of 1e4, against the derived
bound of 2e-2), and where they overlap they differ by the cross term
`2 Re(a conj(b))`, asserted against its exact bound `2|a||b|`.

### 6. Mobility is a declaration, and it resolves to ONE loop configuration

`ClutterComponentSpec{name, mobility, components, rediscovery_period_frames}`
with `mobility in {static, replay, rediscover}`. Whether a moving structure can
GAIN a path is not detectable from inside a replay, so mobility is recorded and
never inferred.

`epoch_policy(specs, *, fixed_topology_components)` resolves a set of specs into
the two `SceneEpochLoop` arguments. There is one compiled scene and one loop per
session, so the resolution is total:

| declaration | `world_motion` | cadence |
|---|---|---|
| all static | `frozen_world` | none |
| static + replay | `fixed_winner_replay` | none |
| static + rediscover | `frozen_world` | shortest declared |
| replay + rediscover | `fixed_winner_replay` | shortest declared |

The mixed case downgrades neither declaration: frames between the ticks replay,
and the tick pays the discovery.

A component whose declared propagation components are outside the consumer's
`fixed_topology_components` is REFUSED unless it declares `rediscover` with a
cadence, and the refusal quotes the live capability record rather than a local
copy. Diffraction and transmission clutter must rediscover.

The policy is resolved by the caller and handed to the loop as two ordinary
arguments. The loop never reads a component declaration, so "which parts does
this scene have" and "when does the pipeline pay" stay separable questions with
separate owners, and `epochs.py` still names no `witwin` package.

### 7. The subtractive boundary is demonstrated, not asserted away

`test_replay_cannot_gain_a_clutter_row_and_a_cadence_recovers_it` drives a wall
parked clear of the geometry and arriving over one second. Under
`mobility="replay"` the composed row count does not change, the born
`TX_A -> SITE_P` reflection round trips are absent, and no signal reports them.
Declaring a rediscovery cadence alongside the replay recovers them on the tick.
Both halves are asserted. That is honest evidence of a designed limitation.

Validity and class are orthogonal and the tests say so: a clutter row that
stops existing publishes `row_valid=False` with an exactly zero payload, and
the component mask still classifies it.

## Consequences

- Exporting `n` components costs `n` synthesis launches. Measured on the
  multi-endpoint fixture (11 rows, 8 chirps, 4 pairs, 256 samples), the FMCW
  synthesis goes from 0.17-0.44 ms unseparated to 0.42-0.96 ms for the two
  populated classes (2.6-2.7x) and 1.28-1.93 ms for all four (5.5-7.8x). The
  spread is machine noise across runs at this fixture size; the RATIO is the
  stable quantity. Propagation, composition, discovery and the frame's
  host-observation budget are all unchanged: a component export reads no device
  value and performs no synchronization, asserted with a counter.
- An empty component still launches and still allocates a full-size cube of
  zeros. A caller that wants only target and clutter should export those two
  rather than iterate the whole taxonomy.
- The scatter response is shared by every component of a frame, because it is a
  property of the site and not of the classification. A per-component response -
  a fluctuating clutter model - would be a new geometry-dependent response class,
  which needs a native kernel and an entry in `NATIVE_ROW_RESPONSE_OWNERS`.
  Phase 8 adds neither.
- Nothing crosses into Channel. No capability, API, ABI or native-manifest field
  names a component, a class, or a mobility.

## Alternatives rejected

**A `component_id` column on `RadarPathTopology`.** Cleaner to read, but it
respecifies a frozen dataclass and its identity tests, and it turns the
acceptance criterion into a claim about equality rather than identity.

**One adapter per component.** Expressible today - the adapter takes
`components` at construction - but it multiplies the tier-2 inner-loop cost by
the component count and gives up the shared pair partition. It stays the right
tool for a clutter class that needs a different `max_depth` or a non-freezable
component, and the mobility refusal is what routes such a class there.

**A third `join_mode`.** See decision 2.

**An incoherent flag on the synthesis kernels.** See decision 5.
