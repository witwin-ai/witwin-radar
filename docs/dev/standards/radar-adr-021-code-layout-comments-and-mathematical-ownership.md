# R-ADR-021: Code layout, comments, duplication, and mathematical ownership

Status: Accepted (2026-07-29)

## Context

The concept-axis consolidation removed the old file-count and maximum-line
constraints. The resulting modules now follow physical and processing concepts,
but four kinds of drift could recreate the old architecture:

1. formatting every parameter on its own line makes a short contract occupy
   dozens of vertical lines;
2. splitting a concept merely to shorten a file recreates deep package trees;
3. copied helpers and equations acquire different conventions over time;
4. comments and documentation continue to describe deleted modules or an older
   mathematical owner.

The repository needs a rule that is both readable by contributors and
executable by CI.

## Decision

### 1. Layout follows concepts, not line counts

`ci/architecture-manifest.json` is the executable source of truth for production
modules and their conceptual owners. There is no maximum file length and no
target file count. A module is split only when the new module has:

- a distinct concept and vocabulary;
- an independent dependency boundary or runtime lifecycle; and
- an independent reason to change.

Directories stay shallow. A new package layer requires an architectural
boundary, not merely several related files. Public convenience imports remain
facades; implementation and mathematics have one owner.

### 2. Ruff owns Python layout

The canonical Python width is 120 columns. Ruff formatting uses
`skip-magic-trailing-comma = true` and isort uses
`split-on-trailing-comma = false`.

- A signature, call, literal, or annotation that fits in 120 columns stays on
  one line.
- A construct that does not fit uses Ruff's hanging indentation.
- Contributors do not manually align parameters or preserve one-argument-per-
  line formatting with a trailing comma.
- A function is not redesigned merely to make its header shorter.
- When many parameters describe one domain concept and evolve together, replace
  them with a typed `dataclass`/spec/options object owned by that concept.
  Unrelated parameters must not be hidden in a generic bag.

The executable checks are:

```bash
python -m ruff format --check witwin/radar tests examples tools ci scripts
python -m ruff check witwin/radar tests examples tools ci scripts
```

### 3. Comments record contracts and reasons

Comments explain invariants, ownership, sign conventions, units, numerical
choices, refusal boundaries, and non-obvious trade-offs. They do not narrate
syntax or restate the next line.

Public APIs use concise docstrings. Add `Args`, `Returns`, and `Raises` sections
when they convey information that is not already explicit in names and types.
Math-heavy module and class docstrings may use a narrative contract when a
single convention governs many functions.

Documentation and comments must not:

- use source line numbers as durable references;
- call a completed migration an active phase;
- describe deleted compatibility paths as current APIs;
- duplicate a formula without naming its canonical owner; or
- claim a benchmark, GPU run, wheel load, or workflow result without retained
  evidence.

Historical documents stay historical. Living documents are listed in
`ci/documentation-manifest.json` and are checked against retired surfaces.

### 4. Mathematical ownership is explicit

| Concept | Canonical owner | Required boundary |
| --- | --- | --- |
| world geometry, structure state, identity | `witwin.core` | Radar consumes typed Core state; it does not recreate geometry |
| one-way delay, delay rate, spreading, transmit-power/carrier/polarization transport | `witwin.channel` through `witwin.radar.channel` | this is the only production Channel import boundary |
| Radar propagation policy and epoch selection | `witwin.radar.propagation` | policy only; no second one-way transfer law |
| round-trip composition and path topology | `witwin.radar.paths` | joins Channel legs and scattering once |
| scattering response | `witwin.radar.scattering` | response coefficient only; no sensor or waveform gain |
| antenna, sensor, and array weighting | `witwin.radar.sensors` | no repeated propagation or scattering factor |
| receiver, noise, gain, and ADC effects | `witwin.radar.frontend` | frontend effects occur after field synthesis |
| waveform grids and synthesis phasors | `witwin.radar.synthesis.*` | FMCW spectrum is direct Dirichlet CUDA by default; beat synthesis is explicit |
| FFT axes, range/Doppler/angle products, detection and tracking | `witwin.radar.processing.*` | consume result metadata; do not infer the domain from tensor shape |

Every new or changed equation must state, in its owning module or ADR:

1. symbols and SI units;
2. phasor/time/sign convention;
3. amplitude, power, and transform normalization;
4. validity domain and singular/degenerate behavior;
5. the module that owns each input factor; and
6. an oracle: closed form, exact-bin identity, independent reference, gradient
   check, or cross-package contract test.

A second implementation is permitted only as an explicitly named independent
test/reference oracle. Production code must call the canonical implementation.

### 5. Duplication is rejected by semantics and by syntax

Reuse code only when the callers share meaning, units, convention, lifecycle,
and reason to change. Similar-looking code with different physical meaning must
remain separate and document why.

Exact non-trivial production function clones are rejected by
`ci/check_duplicate_code.py`. The gate normalizes the AST, ignores names,
comments, docstrings, formatting, and source locations, and compares functions
with at least three executable statements. This catches copied implementation
logic without treating tiny protocol accessors as shared domain algorithms.

## Consequences

- Long files are acceptable when they are one coherent conceptual owner.
- Compact formatting is deterministic and does not depend on contributor taste.
- Parameter-object refactors are architectural decisions rather than formatting
  workarounds.
- Mathematical factors have a single production owner and a stated oracle.
- Exact copied production logic fails the quick CI tier.

## Verification

```bash
python ci/check_architecture.py
python ci/check_duplicate_code.py
python ci/check_documentation_surface.py
python -m ruff format --check witwin/radar tests examples tools ci scripts
python -m ruff check witwin/radar tests examples tools ci scripts
pytest tests/
pytest tests/ --gpu
```
