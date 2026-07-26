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

## Amendment (Phase 7): the token may be REPLACED, and a frozen handle names one

The adapter used to bind one `CompiledScene` at construction and hold it for its
whole lifetime. A moving structure, a deformed mesh, or a new `DynamicScene`
snapshot produces a NEW compiled scene, so that adapter replayed frozen rows
against geometry that had moved on - silently, at full strength, and with a
plausible delay.

`ChannelPropagationAdapter.refreeze(compiled_scene)` replaces the token. It is
still only a token: `refreeze` stores it and passes it back, and the adapter
reads no attribute of it. Two rules make the replacement safe:

- Every `FrozenLegTopology` records the adapter EPOCH it was frozen at, and a
  handle from a retired epoch is refused by name. The epoch is a host int and
  is deliberately broader than Channel's world-provenance check: a compiled
  scene and the rows discovered on it always agree with each other, so
  recompiling an unchanged world produces a handle Channel cannot call stale
  while the caller has still moved on.
- `ChannelPropagationAdapter.rediscovery_required(frozen)` forwards Channel's
  host-only version-domain poll, so a caller learns WHICH domain moved before
  it decides to refreeze. It costs no device work and no host observation, and
  it is meant to be polled every frame.

`refreeze` deliberately does not rediscover. Which paths exist is exactly the
question a moved world reopens, and answering it implicitly would hide the cost
of a discovery inside what looks like a rebind.

## Acceptance evidence (Phase 7)

- `tests/test_phase7_rediscovery_cadence.py::test_refreeze_is_required_after_a_structure_moves`
- `tests/test_phase7_rediscovery_cadence.py::test_a_retired_handle_is_refused_even_when_no_version_moved`
- `tests/test_phase7_rediscovery_cadence.py::test_rediscovery_required_costs_no_host_observation`
