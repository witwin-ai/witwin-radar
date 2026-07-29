# Radar Phase G before-evidence

Status: active debt baseline.

Recorded: 2026-07-28.

Execution used:

```text
Python 3.11.14
Torch 2.10.0
Core 7791ce21 / tree 274d6b9e (clean export)
Channel c07b489e / native 2dd9d779 (clean wheel)
RayD 94cf6eaf / integration 57f83ea4 (clean checkout)
Radar ff2d9cc8 + Phase 0/G working changes
```

## Calibration

`tests/test_consolidation_governance_gates.py` and
`tests/test_consolidation_policy_gates.py` plant:

- a second Channel importer;
- a misleading plain string that must not become an import edge;
- comment-only workflow commands that must not become executable evidence;
- a retired Python path and native compatibility flag;
- retired and missing living-document paths;
- a missing workflow script;
- a required workflow without Channel, fingerprint or skip budget;
- duplicate concept ownership and duplicate canonical public targets;
- a duplicate target in the symbol-level public API manifest;
- retired manylinux and cross-Torch claims plus a false-success loader refusal;
- open and closed-without-evidence governance debt rows.

Result: **10 passed** with isolated writable basetemp.

All new gate sources pass `py_compile` and Ruff. Public API manifest validation,
single-definition validation and governance inventory schema-only validation
pass.

## Target-audit failures

The real tree is expected to fail target gates until the owning phase closes.
The before-evidence is:

- architecture: 11 target modules missing, fragmented current modules reported
  as unexpected, the Channel importer is still the old owner, and five internal
  import cycles are visible;
- compatibility: `_REMOVED`, `sigproc`, `processing/adapters.py`,
  `FmcwSpec`, `RadarAxes`, `SensorWeightModes`, `PolarizationSpec`,
  `from_real_amplitudes`, `synthesize_fmcw` and the native
  `legacy_real_polarization` flag detected;
- documentation: stale AGENTS/CLAUDE paths, current `sigproc` claims and other
  missing living-document paths detected;
- public API: root and every owner facade differ from the approved symbol-level
  target; `witwin.radar.smpl` does not yet exist;
- workflow references: GPU regression invokes missing
  `tools/benchmark_dirichlet_cuda.py`;
- required Channel coverage: quality and GPU workflows install no Channel,
  record no observed fingerprint and enforce no consumed skip budget;
- release claims: README/FEATURE_LIST claim retired `manylinux_2_35` and
  cross-Torch Stable ABI support, while publish CI counts the documented loader
  refusal as a successful matrix cell;
- governance inventory: every open GOV row is reported.

These failures are not allowlisted. Each is tied to a debt row and must
disappear from the real-tree gate output before consolidation completion. The
terminal no-compatibility gate remains red across intermediate phases for the
explicit typed-handoff, native-ABI and Phase-F rows; its expected remainder may
only shrink.
