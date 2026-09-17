# Receiver physics, adaptive motion, and MATLAB comparison

User-authorized work, 2026-09-16, witwin2 on Windows / RTX 5080.

1. Correct receiver domain ordering and pin hardware-equivalent beat/spectrum output.
2. Use physical timestamps and path-dependent common-oscillator delayed phase differences.
3. Add phase-error-controlled temporal interpolation, batched propagation, and topology-event checks; retain exact ADC sampling as an independent reference.
4. Compare matched scenarios against the locally installed MATLAB Radar Toolbox, with explicit runtime/toolbox identity, numerical normalization, and timing boundaries.

Every implementation phase is committed after relevant tests. Final acceptance includes GPU regression, static gates, independent phase/noise oracles, adaptive-versus-exact errors and timings, and actual MATLAB evidence where available. Missing licenses or startup failure cannot be described as a passing comparison.

Stage 1: 44 receiver/domain/scene-entry tests passed. Receiver-enabled FMCW generates beat samples, applies the hardware chain once, then computes the requested normalized spectrum. Ideal receivers retain the direct-spectrum route.

Stage 2: 38 oscillator/frontend/RNG/domain tests passed. A shared continuous-time Wiener oscillator supplies each path's delayed phase difference at absolute ADC time. Brownian-bridge queries are independent of path ordering and refinement. Tests cover delay covariance, idle gaps, spectral delay cancellation, zero-delay cancellation, and beat/spectrum parity. This is white frequency noise, not a multi-region oscillator mask. Wiener timestamp/delay derivatives are explicitly refused; signal derivatives at fixed queries remain supported. Native ABI is 5, with 32 operators.
