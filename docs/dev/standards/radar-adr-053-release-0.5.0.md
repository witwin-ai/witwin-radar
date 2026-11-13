# R-ADR-053: Release record for witwin-radar 0.5.0

Status: Accepted (0.5.0)

## Context

0.5.0 is one breaking release that carries every work item of the 0.5 development plan that was done: the move to
Channel 0.6, the correctness and physics fixes, Doppler precision and path events, multipath components and their
performance, mesh scattering and polarization, the API and documentation pass, governance trimming, and the feature
additions (waveforms, MIMO, networks, imaging, tooling). Each item has its own design record; this record says what
the release contains as a whole, what evidence stands behind it, and what it does not do. The user-facing account is
`CHANGELOG.md` (section `[0.5.0]`) and `docs/migration-0.5.md`.

## Decision: what 0.5.0 is

| Area | What shipped | Records |
|---|---|---|
| Channel 0.6 | the `channel` extra pins `witwin-channel>=0.6.0,<0.7` on Python 3.10-3.13; Channel's validity rules and stale-scene refusal; `torch==2.10.0` | R-ADR-022, 023 |
| Session | the scene session split into `motion`, `binding`, `adaptive`, `results`, typed `Session`/`WorldHalf`; join and pattern-table reuse; a still world evaluated once; grouped replay and one synthesis launch per group; count budgets per motion kind | R-ADR-024 |
| Governance | generated native manifest, trimmed gates, the docs archive, the loader's architecture refusal | R-ADR-026 |
| Processing | square-law CFAR at the requested false-alarm rate, geometry-driven angles for any layout, recorded cubes, clustering, tracking, range-angle maps | R-ADR-027, 028 |
| Receive chain | the noise contract (complex sample rate, a realisation per frame), the ADC grid, element patterns with absolute gain and complex fields, compression, shaped phase noise, coupling, per-pair leakage, DC offset and flicker, dynamic AGC | R-ADR-029 to 031 |
| Oracles | `tests/oracle/`: radar equation, RCS convergence, image-method ghosts, Doppler and angle, carrier-phase precision, per-waveform SNR, wall-bounce level, survival | R-ADR-032 |
| Multipath | discovery rows as primal rows, transmission and diffraction, the site-less leg (environment and coupling), component classification, wideband OFDM | R-ADR-033 to 035 |
| Motion | path events and birth-lag bounds, discovery cadences, `Motion.auto()` resolving to adaptive for a moving FMCW scene, orientation trajectories and radar ego-motion | R-ADR-036 to 038 |
| Transport | the outbound leg from the receiver, float64 round-trip path length, Radar-owned element polarization and site scattering matrices | R-ADR-039 to 041 |
| Scattering | physical-optics facet targets, facet reflectivity from Core materials, rough-facet backscatter, Swerling fluctuation and tabulated cross sections | R-ADR-042 to 045 |
| Waveforms | coded pulses, fast-time observation of pulsed trains, time-domain OFDM, chirp sequences, ramp phase error, real ADC, DDM and BPM | R-ADR-046 to 048 |
| API | typed metadata, `capabilities()`, validation at construction, facades, the naming pass and its migration table | R-ADR-049, 050 |
| Networks | several radars, bistatic links, interference, emitters | R-ADR-051 |
| Gradients | pattern, gain, power, pose, coupling, material and SMPL leaves | R-ADR-052 |
| Imaging | back-projection SAR/ISAR, calibration | R-ADR-054 |
| Tools | saved results, labels, the dataset writer, configuration I/O, TI import, DCA1000, the doctor | R-ADR-055 |
| Documentation | user guides in `docs/user/` whose Python blocks run (`tools/run_doc_snippets.py`), `CHANGELOG.md`, `CONTRIBUTING.md`, the migration guide, maintained examples; comments state invariants and refusals only; `mypy` over the whole package | this record |

The version is set once, in `pyproject.toml`; `witwin.radar.__version__` reads it from the package metadata (or a
source tree's `pyproject.toml`). There is no per-milestone changelog section and no compatibility shim between
milestones. The changelog fragments each work item wrote (one file per item in a docs/dev/changelog-0.5.0 directory) were assembled into
`CHANGELOG.md` in the post-rename names and removed.

## Evidence

- **Physics** is checked against independent oracles, not against earlier Radar output: the analytic oracle suite in
  `tests/oracle/` (R-ADR-032) and the per-item oracles each design record names (float64 trajectories for carrier
  phase, the Airy slab for materials, UTD for knife-edge diffraction, brute-force physical-optics quadrature for facets,
  the exact CFAR false-alarm laws, long-double signal oracles for cross-radar beats). Known defects that remain are
  strict expected failures naming their cause, so a fix that lands is detected.
- **Performance** is counts pinned by budget tests (`tests/test_session_budget.py` per motion kind, and the per-family
  launch and synchronization budgets), plus same-process contended timings; `PERFORMANCE.md` holds the current
  conclusions and the 0.4.1-versus-0.5.0 count table. Structural and performance changes were verified bit-identical
  (`torch.equal`) against the route they replaced.
- **Suites.** The coordinator's full GPU suite on the integration branch after wave 3 (head d09022f) ran 2060 passed,
  25 skipped, 5 xfailed and 3 failed; the three failures were test-side and were fixed in that head. The final full
  GPU run on the release candidate, after Channel 0.6.0 is retagged, is the coordinator's; the release agent ran the
  quick tier, the documentation snippets, the examples and the tests touching what it changed (its report states
  each).

## Known limitations at release

Listed for users in `CHANGELOG.md` ("Known limitations"). In short: co-polar element patterns only (no Jones
transport from Channel's fixed-topology replay); a float32 residual on reflection-row phase; Channel 0.6.0's
missing reflection rows off finely meshed tilted faces (pinned as a strict
xfail in `tests/oracle/test_rcs_convergence.py`); FMCW wideband refused; back-projection is the only imaging route;
`Paths` has no saved form; the compression parameters are not leaves; networks have no adaptive route; profiled budget columns keep a 0.5 % margin; SMPL tested only on
a synthetic rig; the DCA1000 layouts and TI presets written from documentation, not validated on hardware.

## Requests to Channel

Each would lift a limitation above, and none is worked around with a Radar-side fallback: a polarimetric (Jones)
fixed-topology transport for reflection rows; float64 interaction points (or float64 endpoints) for reflection rows;
reflection discovery off finely meshed tilted faces (and a `complete`
flag that is false when candidates were dropped); the structure ID of
each interaction in the consumer topology; several source/sink groups (or a pair mask) per request and slot-batched
discovery.

## Consequences

The next release starts a new `[Unreleased]` section in `CHANGELOG.md`; a user-visible change adds its line there in
the same commit. The design records 022 to 055 keep the names they were written with; R-ADR-050 maps them to the
0.5.0 names.
