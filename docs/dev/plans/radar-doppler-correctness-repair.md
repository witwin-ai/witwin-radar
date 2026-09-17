# Doppler correctness repair

Requested scope: close audit A1-A8, commit each validated stage, and retain final numerical and test evidence.

1. Separate parameter JVP from physical time derivatives (A4).
2. Sample dynamic scenes, sites, and path births on the waveform timeline (A1, A2, A5, A6).
3. Consume actual multipath departure/arrival geometry for antenna and scatter responses (A3).
4. Complete continuous FMCW motion and typed micro-Doppler processing (A7, A8).
5. Run full CPU/GPU suites, static gates, independent physical experiments, and record limitations and performance.

Environment: witwin2, Windows, RTX 5080. Preserve the user's untracked notebook and recording. Build changed native sources before testing them; record native fingerprints. No compatibility shims or derivative-by-path-row finite differences.

## Stage evidence

Stage 1: the scene entry composes parameter JVPs without publishing them as delay rates. A regression varies the tangent seed at identical primal positions and checks an identical primal cube.

Stage 2 (`15b876d`): waveform observation sampling, trajectories, endpoint mappings,
and completeness metadata; 26 entry/motion GPU tests passed.
Stage 3 (`db74db7`): actual multipath sensor segments and outbound scatter bearings,
including native first-order companions. Reflection and pattern oracles passed.
Stage 4 (`94c3a83`): continuous-delay FMCW phase, per-ADC scene sampling, typed
micro-Doppler timestamps/phasor, and frame metadata extraction. 32 independent
waveform/micro-Doppler tests passed; the final stage migrates old fixtures and
executes the complete suite and experiments.

Stage 5: full rebuilt GPU suite: 1499 passed, 12 skipped (11 missing SMPL
assets, one external nightly coexistence record); zero missing-Channel skips.
CPU quick: 697 passed, 813 skipped. All ten static gates and Ruff passed.
Independent continuous-chirp, radial, rotor, two-limb proxy, moving-wall,
and coplanar triangle-winner checks passed. Final results and runtime identity:
`../audit/radar-doppler-correctness-acceptance-2026-09-16.md` and the adjacent
`radar-doppler-correctness-evidence-2026-09-16.json`.
