# R-ADR-040: The round-trip path length is recomputed in float64 inside the join

Status: Accepted (0.5.0)

## Context

Channel's consumer contract is float32: endpoint positions, interaction
points, path lengths, delays and the wave number. A moving target's carrier
phase therefore advanced in steps of `4 pi ulp32(R) / lambda` (D-1). The
oracle suite (R-ADR-032) measured the consequence against a float64
trajectory, at 77 GHz, 64 samples at 10 MS/s, two chirps 1 ms apart:

| Range | Integrated error | Spur floor |
|---|---|---|
| 2 m | -72.9 dBc | -75.8 dBc |
| 30 m | -54.8 dBc | -56.7 dBc |
| 100 m | -42.7 dBc | -48.6 dBc |

The target is -80 dBc at every range. Owner decision 3 prefers a Radar-side
correction over a Channel contract change.

Measured on Channel 0.6.0 before choosing the law (a static leg, a sink swept
in sub-ulp steps):

- the coefficient's phase is exactly `-k32 L32` with `L32` the published
  float32 `path_length_m` and `k32 = float32((2 pi_f32) f_f32 / c_f32)`
  evaluated in float32: the residual is 3.5e-9 rad at 100 m for a
  line-of-sight row and constant to 1.4e-8 rad for a reflection row (the
  material phase). `k64 L32` leaves 7.7e-4 rad at 100 m and `k64 c tau32`
  (the published float32 delay) leaves a 4e-4 rad staircase, so neither is a
  usable reference;
- the unfolded length through Channel's float32 specular point is stationary
  for a displacement ALONG the reflecting surface (Fermat): with the point's
  normal component put back on the wall it equals the image-method length to
  1e-12 m. ACROSS the surface it changes at first order, and Channel's float32
  point is off the plane by up to half a float32 ulp of its coordinate: 4.7e-7
  m of length at a wall 4 m away, near normal incidence.

## Decision

The join recomputes, per composed row, the round-trip length in double,
`L64 = sum |P_{j+1} - P_j|` over the transmit element, the inbound leg's
interaction points, the site, the outbound leg's interaction points (reversed,
R-ADR-039) and the receive element, and replaces Channel's phase:

```
psi = -(k64 (L64 - L32) + (k64 - k32) L32)      k64 = 2 pi f / c0, c0 = 299792458 m/s
V   = V_channel exp(j psi)
tau = L64 / c0
```

written in that form so the large terms cancel analytically rather than in
floating point. A wideband band (R-ADR-035) is composed column by column with
that column's own pair: the join reads `(k64, k32)` per frequency column from a
float64 table, `k64 = 2 pi (f_ref + n df) / c0` and `k32` Channel's float32
wave number at its float32 launch frequency `float32(f_ref + n df)` (the sum in
float64; Channel's `launch_grid=float32`). A replica that rounded the sum in
float32 misses some columns by a `k32` ulp, 9 mrad at 37 m, which is what the
measurement test below caught. Site-less rows (R-ADR-041) take the same law in
the same launch. `tau` is published as the float32 pair
`RadarPathBatch.total_delay_s = float(tau)` and
`total_delay_residual_s = float(tau - total_delay_s)`; the residual is primal
only, the delay's derivative is carried by `total_delay_s`.

Precision of the inputs:

- Endpoint positions: a site authored as a float64 tensor (positions or a
  trajectory callable returning float64) is kept as
  `RadarEndpointSpec.precise_positions_m` and rounded to float32 for Channel
  through a differentiable cast; the adapter gathers it per leg row as
  `RadarLegBatch.precise_arrival_target_m`. Array element positions are the
  radar's float32 pose and are used exactly as Channel received them.
- Interaction points are Channel's float32 points. Line-of-sight rows have
  none, so they are exact in float64. Reflection rows: the in-plane error
  costs nothing (second order) but the error across the surface remains, so a
  reflected leg's length carries up to `(cos theta_i + cos theta_r)
  ulp32(x)/2` - comparable to the float32 length it replaces, without the
  endpoint quantization. Transmission straight through: the points lie on the
  straight line and enter at second order. Diffraction: the edge point is
  stationary only along the edge, so its error perpendicular to the edge
  enters at first order. Channel 0.6 freezes `{los, reflection}` only; a
  discovery-direct transmission or diffraction row inherits these bounds.
  0.5.0 accepts and documents this residual (owner decision); closing it
  needs float64 interaction points from Channel, which is filed as a Channel
  request.

Native: everything above is inside the one join launch (forward, VJP, JVP),
templated on the site precision. The VJP owns the geometry gradients by leg
row through the frozen CSR (no atomics); the reduction from leg rows to
endpoint tensors is Torch's own gather backward. Inputs: element and site
positions, interaction points, `L32`; `psi` couples them through
`dV/dL64 = -j k64 V` and `dV/dL32 = +j k32 V`, so Channel's own derivative of
its phase is removed exactly and replaced by the float64 one.

The adaptive controller (D-9, M2-3) compares delays as the float64 pair and
interpolates between float64 node delays; it subtracts a quantization
allowance (two delays at the pair's `2^-46` relative resolution, two complex64
phases at `2^-22` rad) from its phase test.

Every waveform kernel reads the pair. The FMCW beat and range-spectrum, OFDM
CFR and time-domain, and pulsed kernels take the residual as a
non-differentiable `tau_lo` input and form every phase and envelope term from
`tau_hi + tau_lo` in float64 (`path_delay.cuh`); an empty `tau_lo` reads
`tau_hi` alone, bit for bit as before. The range spectrum's closed form also
evaluates its Dirichlet argument in float64 when a residual is present: in
float32 the argument alone is quantized to `N ulp32(x)`, the size of the
float32 delay error it would otherwise refine. The fused adaptive FMCW kernel
keeps its interpolated delay in float64 (`precise_delay`) when every probe
carries the pair, and so does the unfused observation route; a table of bare
float32 delays keeps the float32 boundary. `SynthesisPathBatch` carries
`total_delay_residual_s`, and the session's slot flush gathers it with the
rows.

## Evidence

`tests/oracle/test_doppler_precision.py` (strict xfails removed, the
trajectory now returns float64), same scenario as the table above:

| Range | Trajectory | Integrated error | Spur floor | Phase rms | Phase peak |
|---|---|---|---|---|---|
| 2 m | const. velocity | -130.0 dBc | -132.9 dBc | 2.9e-7 rad | 9.7e-7 rad |
| 2 m | const. acceleration | -130.1 dBc | -133.8 dBc | 2.9e-7 rad | 8.2e-7 rad |
| 30 m | const. velocity | -111.2 dBc | -115.8 dBc | 2.7e-6 rad | 7.5e-6 rad |
| 30 m | const. acceleration | -111.4 dBc | -115.3 dBc | 2.7e-6 rad | 7.1e-6 rad |
| 100 m | const. velocity | -100.2 dBc | -107.6 dBc | 9.8e-6 rad | 2.8e-5 rad |
| 100 m | const. acceleration | -102.8 dBc | -106.9 dBc | 7.2e-6 rad | 2.0e-5 rad |

What remains is the float32 complex64 coefficient; the waveform kernels read
the float64 delay for the slope term too.

- `tests/test_float64_path_length.py`: Channel's phase is `-k32 L32` (residual
  below 1e-6 rad at 100 m, and `k64` would be visibly wrong); a reflection
  row's length is stationary along the wall (1e-12 m) and within the
  across-the-wall bound; the composed delay pair equals the closed-form
  float64 round trip to 1e-20 s on line-of-sight rows; 1 um of motion at 30 m
  moves the carrier phase by `4 pi dR / lambda` to 1% (a float32-authored
  position is off by more than 30%).
- `tests/test_join_geometry_ad.py`: forward against an independent float64
  Torch statement of the law; every reverse-mode input gradient against
  float64 autograd; the JVP through the adjoint identity; float32 and float64
  sites.
- `tests/test_adaptive_float64_delays.py`: the same 5 m/s motion takes the
  same number of adaptive probes at 5 m and 100 m (within 2x).
- `tests/test_float64_path_length.py::test_every_wideband_column_carries_the_float64_phase_at_its_own_frequency`:
  every column of a 64 x 10 MHz band at 37 m is `exp(-j 2 pi n df tau64)`
  relative to column 0 to 1e-5 rad (measured 6e-7).
- `tests/test_delay_residual_synthesis.py`: one oracle per waveform family
  (FMCW beat, range spectrum, fused adaptive FMCW, OFDM CFR, OFDM time,
  pulsed LFM), a 4 GHz sweep at 100 m against a float64 statement of the
  family's equation: under 1e-5 relative with the residual, over 1e-4 without
  it, and an all-zero residual bit-identical to none (the spectrum, whose
  residual route is the float64 Dirichlet, is held to the float64 statement at
  the float32 delay instead).

## Consequences

Waveform synthesis reads the float64 delay, so a chirp's `2 pi B ulp32(tau)`
(about 7e-4 rad for 4 GHz at 100 m) no longer enters the cube. No Channel
request is needed for the delay; float64 interaction points remain the filed
request for reflection rows.
