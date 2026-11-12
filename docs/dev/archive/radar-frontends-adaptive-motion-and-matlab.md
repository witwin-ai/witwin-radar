# Receiver physics, adaptive motion, and MATLAB comparison

User-authorized work, 2026-09-16, witwin2 on Windows / RTX 5080.

1. Correct receiver domain ordering and pin hardware-equivalent beat/spectrum output.
2. Use physical timestamps and path-dependent common-oscillator delayed phase differences.
3. Add phase-error-controlled temporal interpolation, batched propagation, and topology-event checks; retain exact ADC sampling as an independent reference.
4. Compare matched scenarios against the locally installed MATLAB Radar Toolbox, with explicit runtime/toolbox identity, numerical normalization, and timing boundaries.

Every implementation phase is committed after relevant tests. Final acceptance includes GPU regression, static gates, independent phase/noise oracles, adaptive-versus-exact errors and timings, and actual MATLAB evidence where available. Missing licenses or startup failure cannot be described as a passing comparison.

Stage 1: 44 receiver/domain/scene-entry tests passed. Receiver-enabled FMCW generates beat samples, applies the hardware chain once, then computes the requested normalized spectrum. Ideal receivers retain the direct-spectrum route.

Stage 2: 38 oscillator/frontend/RNG/domain tests passed. A shared continuous-time Wiener oscillator supplies each path's delayed phase difference at absolute ADC time. Brownian-bridge queries are independent of path ordering and refinement. Tests cover delay covariance, idle gaps, spectral delay cancellation, zero-delay cancellation, and beat/spectrum parity. This is white frequency noise, not a multi-region oscillator mask. Wiener timestamp/delay derivatives are explicitly refused; signal derivatives at fixed queries remain supported. Native ABI is 5, with 32 operators.

Stage 2 final rebuilt verification: 58 tests passed, including the native binding registry. Commit 52f50bc.

Stage 3 implements explicit adaptive FMCW sampling, retaining ADC as the exhaustive reference default. Native carrier-aware interpolation has analytic VJP/JVP; adaptive decisions hold the chosen partition fixed for differentiation. Full leg keys and validity changes trigger subdivision. Channel replay batches only identical topology against the same compiled geometry. Moving structures are evaluated before their handles are retired. Synthesis batches equal fast-time offsets without allocating a quadratic observation grid.

Measured heavy scene: 64 round-trip rows, 2048 ADC observations, 2048 to 37 discoveries; 101.330 s to 1.580 s (64.14x), relative IQ L2 error 0.00056595 and RD-power L2 error 0.00006488. Radial, rotor and two-point articulated proxies measured 2.91–3.08x with IQ errors below 0.000261. Raw cubes and measurements are in output/doppler-repair/adaptive. These sampled tests do not certify arbitrary trajectories or arbitrarily short unseen topology events.

Final local acceptance: 1531 GPU tests passed, 12 skipped (11 missing SMPL assets and one external nightly coexistence record), zero missing-Channel skips. CPU quick: 697 passed, 846 skipped, 57% coverage. All nine additional static gates passed; Ruff checked 215 files. Phase 3 commit: e6210eb.

MATLAB follow-up: after user login and authorized Radar Toolbox installation, seven actual R2025b Update 4 radarTransceiver cases completed. All declared-path acceptance checks passed: raw IQ errors 0.50%/0.58%/1.07% for static/radial/three-path cases, reduced by 4x oversampling; integer-delay control error 9.9e-8; all path-local RD peak bins match. A separate linear fractional-delay diagnostic agrees with MATLAB within 2.4e-11, explaining the dominant model difference. These checks do not cover MATLAB scene discovery, animated micro-Doppler, heavy geometry, hardware masks, or comparable performance. Full evidence and installation caveat: docs/dev/audit/radar-frontends-adaptive-motion-acceptance-2026-09-16.md.
