# R-ADR-045: Swerling fluctuation and tabulated cross sections

Status: Accepted

## Context

PH-6: every target response was a deterministic scalar or a separable aspect
lobe. There was no fluctuation (speckle, Swerling), and no way to give a target
a measured or precomputed `sigma(aspect, frequency)`.

## Decision

### 1. Swerling fluctuation

`Fluctuation(swerling, coherent_interval, seed)` on `PointTargets` and
`StructureTargets` wraps the target's response in a
`FluctuatingResponse`. Each site's power is multiplied by an independent draw
`X` with `E[X] = 1`:

* Swerling I and II: `X ~ Exp(1)` (chi-square with 2 degrees of freedom,
  `E X^2 = 2`), the many-equal-scatterer target;
* Swerling III and IV: `X ~ Gamma(2, 1/2)` (chi-square with 4 degrees of
  freedom, `E X^2 = 3/2`), one dominant scatterer among many.

The amplitude factor `sqrt(X)` multiplies the response; the phase is not
fluctuated. I and III hold one draw for every observation in one coherent
interval `T` (the "scan") and redraw at `floor(t / T)` boundaries; II and IV
redraw at every observation instant ("pulse to pulse", where an observation is
what `Motion` samples). The draw is the native `fluctuation_factor_forward`:
splitmix64 uniforms of a key and the site index, transformed by
`-log U` (and the mean of two for 4 degrees of freedom). The key is the
frontend's frame key (`frontend.frame_seed`, MurmurHash3's finaliser of the
instant's IEEE-754 bits mixed into the seed) of the interval's start for I/III
and of the instant for II/IV, so every verb sees one realisation and a run is
reproducible from its seed.

The factor has no differentiable input (the manifest records it under
`no_backward`). The product `S sqrt(X)` is one tensor multiply in the
response, so the base response keeps its derivatives with respect to its own
parameters. A fluctuating response varies per observation, so the session
binds it per observation and never slot-batches it or reuses a still world's
rows.

### 2. Tabulated cross sections

`RcsTable(axis, rcs_m2, frequencies_hz=None, phase_rad=0.0)` is accepted as a
`PointTargets.rcs`. `rcs_m2` samples `sigma` uniformly in the aspect angle on
`[0, pi]` (`(A,)`), or at ascending frequencies (`(F, A)`). At session start the
table is reduced to the carrier on the host: the complex amplitude
`sqrt(4 pi sigma) / lambda exp(-j phase)` is interpolated linearly in
frequency, and a carrier outside the table's span is refused rather than
extrapolated. The waveform is narrowband at the carrier everywhere else in
Radar, so a per-frequency table is consumed at one frequency.

Per composed row the native `scatter_response_table` family evaluates the
bistatic aspect `theta = acos(axis . b / |b|)`, `b = u_in + u_out` the
bisector of the leg directions at the site (the monostatic aspect when the
two coincide), and interpolates the table linearly in `theta`. Its VJP and JVP
are exact jets over the leg directions and the axis (one owner slot per thread,
the join's frozen CSR, no atomics). The table is host data and takes no
derivative; a requires-grad table is refused. On the axis (`theta = 0, pi`),
where `d theta / d cos theta` is singular, and for exact forward scatter
(`b = 0`), the derivative is the zero one-sided limit.

**Validity.** The bistatic-equivalent rule (reading a monostatic table at the
bisector) holds for small bistatic angles; a table measured bistatically
should be authored per bistatic geometry. Linear interpolation in `theta`
needs the table to resolve its lobes.

## Evidence

`tests/test_target_fluctuation_and_tables.py`: over 1500 sites the draw
moments are `E X = 1`, `E X^2 = 2` (I) and `3/2` (III); Swerling I holds its
draw inside a coherent interval and redraws across it, II redraws every
observation, and a rerun with the same seed is bit-identical; a two-frequency
table read back at four aspects through the radar equation equals the
interpolated closed form to `1e-3`; the table operator's JVP equals central
differences and its VJP's adjoint.
