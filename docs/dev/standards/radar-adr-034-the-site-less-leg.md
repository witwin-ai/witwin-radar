# R-ADR-034: The site-less TX->RX leg, and who classifies its rows

Status: Accepted

## Context

Radar composed only transmitter -> site -> receiver rows (PH-1, MP-1). Geometry produced no echo of its own: a wall
facing the radar returned nothing at `2d/c`, a vehicle body with no declared site was invisible, and the geometric
coupling between separated transmit and receive elements did not exist. `RadarComponentIndex`, `ComponentDeclaration`
and `select_component` (R-ADR-016) named a `direct_leakage` class and site-less clutter rows that no production code
could produce.

## Decision

### 1. A third frozen leg, transmitter straight to receiver

Two verb keywords, both off by default:

- `environment=True` asks the site-less leg for the session's interacting components (`reflection`, `diffraction`):
  the environment's own returns. It is refused by name without either.
- `coupling=True` asks it for the straight segment between the elements: `los`, plus `transmission` when the run
  requests transmission.

The leg is discovered by the same adapter as the two site legs (`discover(..., components=)` narrows one leg's request
to a subset of the adapter's components), so the epoch loop rebinds and polls it with the others: `FrozenEpoch.handles`
is `(inbound, outbound, direct)`. It follows every route the site legs follow - discovered rows on a primal run,
slot-batched replay otherwise, the wideband band (R-ADR-035) - and it is part of the epoch identity, so a rediscovery
that finds the same three topologies reuses the join.

Why two keywords. The straight segment between co-located elements is not a propagation path: free-space transport
is singular at zero distance, and a monostatic radar's self-coupling belongs to `Radar(leakage=...)`, not to Channel.
Channel's endpoint-separation check refuses such a pair, which is right for `coupling` and wrong for `environment`:
a wall echo of a co-located monostatic radar is well defined. Separating the two makes the common monostatic case
(`environment=True`) work and the singular one (`coupling=True` with co-located elements) a named refusal, with no
silent per-pair filtering.

### 2. Site-less rows join the sensor-pair partition by a gather

A site-less row is its leg row as it stands: delay, coefficient, validity, with no scatter factor. `TwoWayComposer`
places it in its pair's segment after that pair's site rows, ordered by its own leg key, through one frozen `merge`
permutation over `[site rows | direct rows]` (`torch.cat` then `index_select`, autograd- and forward-AD-native). The
native join, its CSR families and `inbound_row`/`outbound_row`/`response_slot` still address the site rows only, so
the join kernels are untouched; `sensor_pair_index`, `pair_offsets` and `topology` address the composed rows.
`RadarPathTopology` marks a site-less row with `site_id == NO_SITE (-1)`, `inbound_row == outbound_row == -1` and a
new `direct_row` column (`None` when no third leg was composed). `TwoWayComposer.segment_targets` gives the pattern
stage each row's first and last segment ends from whichever leg owns it. `for_slots` replicates `merge` and the
optional columns with whole-block shifts.

A run without the third leg builds no `merge`, gathers nothing and is bit-identical to the previous composer.

### 3. Off by default

Default-on would add a row to every scene with geometry and a coupling row to every separated array, moving every
existing baseline, and would refuse every co-located fixture with `coupling`. A physically complete simulator
includes environment returns; the decision here is that a caller opts in per call, and that the default may flip in a
later release once baselines are regenerated deliberately. No baseline moved.

### 4. The component taxonomy has a production owner

`Paths.component_index(frame, observation, declaration=None)` builds a `RadarComponentIndex` for one observation from
its epoch's composer and frozen legs (`from_two_way` now reads a site-less row's depth and materials from the direct
leg). The default declaration is `default_component_declaration(epoch)`: every declared site a target, every compiled
material any leg interacted with clutter. So the site LOS-LOS row is `target`, every row that touches a wall
(including the site-less wall echo) is `environment_clutter`, the site-less line of sight is `direct_leakage`, and a
row deeper than the declared depth is `multi_interaction`.

`Paths.select(name, declaration=None)` returns the same paths with every row outside the class marked invalid, so
`radar.echo(paths.select("environment_clutter"))` synthesizes the clutter alone and the per-class echoes sum to the
full echo. An adaptive frame (one interpolation table, not one batch per instant) is refused by name.
`select_component` on a `SynthesisPathBatch` remains the row-level primitive.

## Oracles

`tests/test_multipath_discovery.py`: a PEC wall 4 m in front of a co-located radar returns one site-less row at
`8 m / c` with `|C| = sqrt(P) lambda / (4 pi 8 m)` (image method, `|Gamma| = 1`) to 2e-4 (measured 9e-5); elements 1 cm apart couple
at `0.01 m / c` with the free-space level; co-located coupling is refused by name; the per-class echoes sum to the
full echo; the slot-batched and replayed routes give the same cube.
