# Radar development archive

Everything in this directory is a **historical record**. It explains how the current tree came to be and is kept
for that reason only: none of it is a current instruction, and none of it is checked against the current tree by
`ci/check_documentation_surface.py`. Paths, line numbers, operator counts, ABI versions and statuses in these
documents are as of the date each was written. Where one disagrees with the code, a living document, or a standard
under `docs/dev/standards/`, the latter is right.

The living documents are listed in `ci/documentation-manifest.json`.

## Index

| Document | What it records | Superseded by |
| --- | --- | --- |
| [channel-radar-architecture-plan.md](channel-radar-architecture-plan.md) | The Stage I/II plan for Radar consuming Channel's propagation core (phases 1-11), in Chinese | The code, `AGENTS.md`, R-ADR-001 to R-ADR-020 |
| [stage2-phase4-entry-prep.md](stage2-phase4-entry-prep.md) | The inventory taken before Stage II Phase 4 started, in Chinese | The same |
| [phase10-acceptance-matrix.md](phase10-acceptance-matrix.md) | Phase 10 acceptance criteria mapped to evidence | `ci/run_ci_tier.py` and the tests it runs |
| [phase10-extension-boundary.md](phase10-extension-boundary.md) | Phase 10 audit of the `_radar_native` / `_channel` source and link boundary | R-ADR-004, R-ADR-019 |
| [phase11-acceptance-matrix.md](phase11-acceptance-matrix.md) | Phase 11 acceptance criteria, commands and verdicts | The same |
| [phase11-cutover-migration-note.md](phase11-cutover-migration-note.md) | The Phase 11 public API break and its migration | `ci/public-api-snapshot.json`, `README.md` |
| [radar-concept-axis-layout-and-module-consolidation-plan.md](radar-concept-axis-layout-and-module-consolidation-plan.md) | The completed concept-axis consolidation | `AGENTS.md`, R-ADR-021, `ci/architecture-manifest.json` |
| [radar-governance-debt-and-drift-inventory.md](radar-governance-debt-and-drift-inventory.md) | The governance-debt register, every row closed | R-ADR-026 |
| [radar-api-redesign-domain-objects.md](radar-api-redesign-domain-objects.md) | The domain-object API redesign and the trace/echo split | `AGENTS.md` "Public entry points", the public API snapshot |
| [radar-doppler-correctness-repair.md](radar-doppler-correctness-repair.md) | The scope of the 2026-09 Doppler correctness repair | `docs/dev/audit/radar-doppler-correctness-*` |
| [radar-frontends-adaptive-motion-and-matlab.md](radar-frontends-adaptive-motion-and-matlab.md) | The scope of the 2026-09 receiver, adaptive-motion and MATLAB work | `docs/dev/audit/radar-frontends-adaptive-motion-acceptance-2026-09-16.md` |
| [phase9-ad-acceptance-record.md](phase9-ad-acceptance-record.md) | The Phase 9 AD acceptance verdicts, formerly the tail of the capability matrix | `docs/dev/radar-ad-capability-matrix.md` |
| [performance-history-2026-09.md](performance-history-2026-09.md) | Superseded and retracted performance sections of `PERFORMANCE.md` | `PERFORMANCE.md` |

## Evidence under `output/`

Several of these records, and the dated reports under `docs/dev/audit/`, cite files under `output/`. That directory
is gitignored: it was the recording machine's local run directory, it was never committed, and it no longer exists
on any machine this archive was assembled on. Those citations say where evidence was written at the time, not where
it can be found now. Evidence that is retained is committed beside the report that uses it (the `*.json` files in
`docs/dev/audit/`) or as a fixture under `tools/fixtures/`.
