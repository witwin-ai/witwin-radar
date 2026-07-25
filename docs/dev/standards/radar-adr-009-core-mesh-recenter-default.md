# R-ADR-009: `witwin.core.Mesh` should not silently recentre world coordinates

Status: **Proposed** (cross-repository; NOT implemented)

This document proposes a change to `witwin/core`, which this Radar change does
not touch. Plan 18.1 forbids patching upstream from a Radar pull request, so the
Phase-4 deliverable is this text plus a Radar-side defensive assertion.

## Context

`witwin/core/geometry/mesh.py:297` defaults `recenter=True`, and
`_local_vertices_tensor` (`mesh.py:444`) subtracts the bounding-box centre from
the authored vertices. `Mesh.from_obj` (`mesh.py:486`) carries the same default.

The consequence: a caller who authors world coordinates and omits the keyword
gets geometry silently relocated to the origin. Nothing raises. The scene
compiles, propagation runs, paths are found, and every number is
self-consistent -- and wrong, because the wall is not where the caller put it.

This is not hypothetical. It cost a five-variable bisection during Stage-II
readiness work to locate. Channel's own `tests/support/core_world.py` already
passes `recenter=False` in `make_mesh_structure`, which is evidence that the
default is wrong for world-authored geometry and that the workaround is already
being copied around by hand.

The failure mode is the expensive kind: no exception, no warning, plausible
output, and a root cause several layers away from the symptom.

## Proposal

One of:

**A. Flip the default to `recenter=False`.** World coordinates are the common
case for scene authoring; recentring is an asset-import convenience.

**B. Make `recenter` a required keyword argument.** Every caller states its
intent. Noisier, but there is no silent wrong answer and no default to argue
about.

Both are breaking changes. They need a Core-side inventory of every `Mesh` and
`Mesh.from_obj` call site, including Radar's own asset loaders, before either is
chosen. That inventory is the actual prerequisite and it has not been done.

This is an owner call, not a Radar decision.

## Radar-side mitigation, implemented

Every `Mesh` built by the Phase-4 fixture passes `recenter=False` explicitly, and
`tests/support/phase4_world.py::assert_world_coordinates_survived` checks after
construction that the authored wall plane is still at `x = 4`. That asserts the
property the spike depends on instead of trusting an upstream default.

## Consequences of doing nothing

The workaround keeps spreading by copy-paste, and the next caller who omits the
keyword rediscovers the same silent failure. The cost per rediscovery is high
precisely because the symptom is "the physics is slightly wrong" rather than an
error.

## Acceptance evidence

None yet. This is Proposed. `tests/support/phase4_world.py` carries the Radar-side
assertion; nothing upstream has changed.

## Appendix: a second Core footgun, observed in Phase 5

`Box(...).to_mesh()` returns a `(vertices, faces)` tuple with BOTH tensors on
the CPU. Promoting only `vertices` to CUDA - the natural thing to write, since
`faces` is an index table - trips `_resolve_device` in
`core/witwin/core/geometry/base.py` with

```
ValueError: Geometry tensor device cpu conflicts with resolved device cuda:0.
```

which names neither tensor, so the reader looks at the one they just moved.
Measured here rather than quoted from a survey.

This is recorded, not patched: Core is read-only from here. It matters because
it will resurface in any new fixture built from a Core primitive, and because
the error message sends the reader looking at the wrong tensor. Any fixture that
promotes a `to_mesh()` result must move `faces` as well as `vertices`.
