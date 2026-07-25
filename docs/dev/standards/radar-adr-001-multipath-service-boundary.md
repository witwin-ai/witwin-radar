# R-ADR-001: Radar consumes the propagation consumer, not the enumerated engine

Status: Accepted (Phase 4)

## Numbering

Radar ADRs are numbered `radar-adr-NNN` with the `R-ADR-NNN` short form. Channel
uses `adr-0NN` in its own repository, and a Radar document numbered `adr-038`
would be ambiguous the first time someone cites it in a cross-repository review.
The prefix costs four characters and removes the ambiguity permanently.

## Context

Channel's ADR-008 makes exactly one exception to the rule that no solver imports
another's internals: `montecarlo.bdpt.pipeline` may call the public
`evaluate_enumerated_paths` entry read-only. That exception is narrow, named, and
allowlisted.

Radar needs multipath propagation. The tempting shortcut is to reach for the same
entry, or for a Path/Deterministic `Result`, and read the path rows out of it.

## Decision

Radar consumes `witwin.channel.propagation.consumer` and nothing else from
Channel's propagation stack. Specifically:

- Radar is NOT a second ADR-008 exception. It gets no allowlist entry.
- Radar does not import `witwin.channel.propagation.enumerated.*`,
  `propagation.models`, `propagation.topology`, `propagation.geometry`, or
  `propagation.fields`.
- Radar does not import a Channel solver (`path`, `deterministic`,
  `montecarlo`), and does not obtain path rows indirectly by constructing a
  solver and reading its `Result`.
- Radar does not touch `witwin.channel._channel` or
  `witwin.channel.runtime.extension`.

Exactly one Radar module crosses the boundary:
`witwin/radar/propagation/channel_consumer.py`.

## Consequences

Radar is insulated from Channel's internal refactoring, which is the point of
the consumer contract having a version. When Radar needs something the contract
does not publish, the answer is a contract change in Channel with its own ADR,
not a Radar-side reach-through. That is slower and correct.

The consumer facade's own package initialization loads a large part of
`witwin.channel`, including `witwin.channel.runtime.*`. That is Channel
initializing itself, not Radar reaching. The boundary test therefore asserts
that Radar adds nothing to the facade's own module closure, and separately
asserts absolutely that no solver and no internal propagation module is ever
loaded.

## Alternatives rejected

**Extend the ADR-008 exception to Radar.** The exception exists because BDPT
needs a discrete-path oracle inside Channel, where the numerical owner and the
caller ship together. Radar is a separate distribution with a separate release
cadence; an internal-API dependency across that line becomes a permanent
coupling.

**Read path rows from a solver `Result`.** Same coupling, one indirection later,
and it additionally binds Radar to a solver's configuration and metadata schema.

## Appendix: upstream consumer observations, recorded not patched

Channel is read-only from here, and the boundary above is exactly why: what
Radar can do with a consumer behaviour it disagrees with is describe it, not
work around it. Each entry below was measured against the live extension from
this worktree.

**Degenerate reflection is zero-but-valid, and asymmetric by endpoint role.**
With the fixture wall at `x = 4` and a site at distance `d` from it, the frozen
reflection row's coefficient magnitude is smooth and converges to its grazing
value as `d` shrinks -- until the leg where the site is the SOURCE snaps to
exactly `0.0` at `d <= 1e-6 m` while `row_valid` stays `True`. The leg where the
same site is the SINK keeps its finite coefficient at every `d`, including
`d = 0`:

```
d_to_wall    site-as-sink  |c|            site-as-source  |c|
3.0e-06      True          4.14433271e-05  True           4.30694963e-05
1.0e-06      True          4.14433489e-05  True           0.00000000e+00
0.0e+00      True          4.14433562e-05  True           0.00000000e+00
```

Two things are notable. A valid row with a discontinuously zeroed payload sits
against the contract's own statement that `row_valid` is the sole authority on
whether a payload means anything: a consumer that trusts the flag reads a zero
as physics. And the source/sink asymmetry points at an epsilon inside the
fixed-topology reflection reevaluation that treats the two endpoint roles
differently, rather than at a shared geometric degeneracy.

This is measure-zero geometry and the join propagates it consistently -- the
products go to zero and nothing produces a NaN -- so Radar is not exposed today.
It is recorded because a Radar-side guard would be exactly the compensating
workaround this boundary exists to prevent.

## Acceptance evidence

`tests/test_phase4_import_boundary.py`:

- `test_no_channel_solver_or_internal_module_is_ever_loaded` (subprocess probe)
- `test_radar_adds_nothing_to_the_consumer_facade_closure`
- `test_static_closure_of_the_new_modules_names_nothing_forbidden`, which asserts
  the exact set of named Channel imports is
  `{witwin.channel.propagation, witwin.channel.propagation.consumer}`
- `test_only_the_adapter_crosses_the_channel_boundary`
