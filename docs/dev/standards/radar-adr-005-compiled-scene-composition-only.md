# R-ADR-005: Radar receives typed tensors, never a scene handle

Status: Accepted (Phase 4)

## Context

Channel's `CompiledScene` owns native resources: a RayD scene, GPU stores,
material tables, and cached resident state. Radar needs the results of tracing
that scene, and a shared handle would in principle let Radar's own kernels trace
it directly.

## Decision

Radar receives compact typed CUDA tensors from the consumer contract and nothing
else. It never receives:

- a RayD scene handle or `SceneResource`,
- a native pointer or an integer encoding one,
- a PyCapsule or any other opaque native object,
- an entry in a Channel cache or registry.

Radar holds the `CompiledScene` only as an opaque token to pass back to
`evaluate` and `reevaluate`. It does not read its attributes and does not depend
on its type beyond that.

A cross-extension scene lease -- letting `_radar_native` trace a Channel-owned
RayD scene -- is deliberately OUT OF SCOPE. It would create a second numerical
owner for the same geometry, a lifetime contract spanning two independently
released binaries, and an ABI coupling that neither repository's release process
currently models. It needs its own ADR in both repositories, with a lifetime and
failure story, before any code.

## Consequences

Radar cannot trace new rays. It can only reevaluate topologies Channel
discovered. For Phase 4 that is exactly right: the frozen-topology capability is
what makes per-frame reevaluation cheap, and it is a contract-level feature, not
a native-handle feature.

## Acceptance evidence

`tests/test_phase4_import_boundary.py::test_static_closure_of_the_new_modules_names_nothing_forbidden`
asserts the adapter names only `witwin.channel.propagation[.consumer]`, so no
scene-internal type is reachable. The adapter's `compiled_scene` parameter is
typed `object` and is only ever passed through.
