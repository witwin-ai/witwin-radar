# Radar consolidation execution amendments

Status: binding amendments produced by the adversarial audit.

Recorded: 2026-07-28.

These decisions resolve contradictions discovered after execution baseline
capture. They override any broader preservation statement in the layout plan.

1. Removing the legacy real-amplitude sensor path is an intentional native API
   break. The `legacy_real_polarization` mode, its polarization projection,
   unused pointer arguments, manifest fields and tests are removed together.
   Only the Channel-sourced numerical slice is required to remain unchanged.
2. Bare-tensor processing overloads are not deleted until a typed
   `ProcessingCube.from_simulation` (or equivalent typed frame API) connects
   `RadarSimulationResult` to processing. Callers migrate atomically; there is
   no adapter left behind.
3. `radar.py` is the single configuration owner after consolidation. The flat
   `RadarConfig`, `RadarSystemConfig`, waveform configuration blocks,
   validation and SI conversion move there. FMCW output domain is stored on the
   FMCW waveform block and converted exactly once into its synthesis spec.
4. `ProcessingAxes` is the single physical-axis owner. `RadarAxes`,
   `Radar.ranges`, `Radar.velocities`, resolution/max-range convenience
   properties and adapter-derived axes are deleted after the typed simulation
   handoff is live.
5. Native preservation gates exclude two explicitly approved API changes:
   legacy sensor-mode removal and Phase-F FMCW spectrum operators. Every other
   numerical family, launch count, AD role and host-observation count remains
   frozen.
6. FMCW parity covers narrowband provenance. Both spectrum and beat preserve
   the current explicit refusal of discrete wideband `frequency_response`;
   “wideband provenance parity” does not authorize a new interpolation model.
7. The direct spectrum uses the normalized length-`num_samples` DFT, double
   phase reduction and analytic `x -> 0` Dirichlet/derivative limits. Historical
   `dirichlet.cu` is evidence only: no one-way distance, padding, truncation,
   internal range loss, float32 phase, or old registered operator returns.
8. The only accepted root API is `Radar` and `RadarConfig`. The symbol-level
   owner facades are frozen in `ci/public-api-manifest.json`; compatibility
   deletion cannot precede that manifest.
9. `slow_time_mode` is removed from the public `Radar.simulate` signature: the
   scene driver has exactly one supported mode and owns it internally.
10. Antenna-pattern ownership must become single-valued. A stored pattern is
    either applied by simulation or removed from stored configuration; `None`
    cannot simultaneously mean default dipole and no pattern.
11. Required integration CI installs the pinned Channel dependency and records
    its version/native fingerprint. Core→Channel→Radar tests have a zero-skip
    budget; a green job that skipped the main chain is not evidence.
12. Phase 0/G is the only active implementation scope until immutable Core and
    Channel identities are recorded (Core commit/tree, Channel native fingerprint)
    plus quick/cuda and numerical smoke evidence. No production deletion or
    module movement begins earlier.
13. The no-compatibility gate is a terminal zero gate, not a Phase-1 zero gate.
    Phase 1 removes the root proxy, `sigproc`, adapters and immediately
    deletable aliases. Typed processing overloads close only with the typed
    handoff, sensor modes close with their native ABI removal, and FMCW beat
    names close atomically with Phase F. Every scheduled remainder stays visible
    in the debt ledger and target gate output; none is an accepted compatibility
    surface or an allowlist.
