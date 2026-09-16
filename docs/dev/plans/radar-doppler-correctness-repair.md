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
