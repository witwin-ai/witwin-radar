# R-ADR-032: The analytic oracle and survival suites are the acceptance gates for physics changes

Status: Accepted

## Context

A physics change is accepted against an independent statement of the physics,
not against the previous release's numbers. Before this record the suite had
float64 references for single waveform kernels and a radar equation checked
only where the polarization projection is exactly one, and several audit
findings were invisible to it: the off-plane polarization factor (PH-5), the
non-convergence of sampled surfaces (PH-3), ghost angles of arrival (MP-15),
the float32 phase staircase of a moving target (D-1) behind a reference that
shared the float32 trajectory (D-2), the pulsed noise normalisation (FE-1),
the offline-only material comparison (MAT-3), and the hard failure of an
occluded target (MP-2).

## Decision

`tests/oracle/` holds the gates. Every oracle runs through the public verbs
(`Radar.simulate`, `Radar.trace` and its `Paths.rows`), on GPU, marked `gpu`,
and compares against a closed form in `tests/reference/radar_oracles.py`,
which imports nothing from `witwin` (a review rule; no automated scan enforces this
direction) while `ci/check_test_oracle_isolation.py` keeps production from importing it.
Scenes and the user-side sampling recipes live in
`tests/support/oracle_scenes.py`. A defect the plan has not fixed yet is a
`pytest.mark.xfail(strict=True)` whose reason starts with the finding ID, so a
fix flips it to XPASS, fails the suite, and makes whoever fixed it remove the
marker.

All oracles share one pose: radar at the origin, boresight world +x, up world
+y, `polarization="up"` (world +y), carrier 77 GHz.

| Oracle (file) | Reference formula | Tolerance and why | Today (Channel 0.5.0) |
|---|---|---|---|
| Radar equation off the polarization plane, PH-5 (`test_radar_equation.py`) | `P_r = P_t G(u)^2 lambda^2 sigma / ((4 pi)^3 R^4)`, `G` the declared pattern gain toward the target (1 for isotropic, the tabulated cut for dipole), no other angular factor; measured as `mean |x|^2` of a static beat cube | `rel 1e-3` on power (0.004 dB): float32 reproduces the equation to 3e-8 on the plane | H-plane passes at all angles for both patterns. E-plane, isotropic: 0.00 / -5.00 / -12.04 / -24.08 dB at 0/30/45/60 deg; dipole: 0.00 / -8.52 / -20.12 / -39.24 dB against the expected 0.00 / -3.52 / -8.08 / -15.16 dB. Six strict xfails (PH-5). |
| Sampled-surface RCS, PH-3 (`test_rcs_convergence.py`) | plate `4 pi A^2 / lambda^2`, sphere `pi a^2`, trihedral `4 pi L^4 / (3 lambda^2)`; 5 cm shapes at 10 m (far field); measured as the coherent sum of the composed rows inverted through the radar equation; recipe `N` isotropic sites of `sigma / N` | within 1 dB at each of the three finest levels: 1 dB is the usual RCS-measurement standard, and requiring three levels stops a coincidental phase sum from passing | plate +24.1 / +30.1 / +36.1 dB (= `10 log N`); sphere -11.0 / -5.1 / +1.0 dB; trihedral +11.7 / +3.9 / +8.2 dB. Three strict xfails (PH-3); the single-site control passes. |
| Image-method multipath, MP-15 (`test_image_multipath.py`) | per class, `L = |T_tx - tx| + |T_rx - rx|` with `T_tx, T_rx` the target or its mirror; phase `-k L`; Doppler `-(dL/dt) / lambda`; arrival from `T_rx` | lengths 2e-5 m (float32 near 28 m); array phase and Doppler phase 0.03 rad after one constant per class; direction cosine 0.01; processed cube: one range bin, one velocity bin, each cell's array snapshot inside the span of its predicted steering vectors (residual < 0.05) and outside every wrong set (> 0.3), beam peak within 1 deg in single-source cells | passes: lengths agree to ~2e-6 m, every class including the two mixed-path angles. A beam-scan peak in the mixed cell is biased 1.7 deg by the other source's sidelobe, which is why that cell is checked as a subspace. |
| Doppler precision, D-1/D-2 (`test_doppler_precision.py`) | float64 `p(t) = p0 + v t + a t^2/2`, quasi-static beat `A exp(j 2 pi (f_c tau + K tau (t - tau/2)))`, `tau = 2 |p(t)| / c` at each ADC instant | M2-2 target: integrated error and fast-time spur floor both below -80 dBc; plus a passing envelope (amplitude 1e-5, phase `4 k ulp32(R)`) proving the reference is the production model | see "Measured precision" below. Six strict xfails (D-1). |
| Per-waveform SNR, FE-1 (`test_waveform_snr.py`) | `SNR = P_r T_obs / (k T_sys)`, `T_obs` the observed echo time; measured normalisation-free as `sum |s|^2 / E|n|^2` from a 0 K and a 290 K run with one seed | 0.5 dB: five standard errors of the noise-variance estimate, well below a factor-of-two slip | FMCW beat -0.04 dB, FMCW spectrum -0.04 dB, OFDM -0.04 dB (pass); pulsed +53.88 dB (`f_s / (B T_p)`), strict xfail (FE-1). |
| Wall-bounce level, MAT-3 (`test_material_multipath.py`) | TE Airy slab coefficient `r12 (1 - e^{-2 j delta}) / (1 - r12^2 e^{-2 j delta})`; rows relative to direct-direct are `Gamma (R/L) e^{-jk(L-R)}` and `Gamma^2 (R/L)^2 e^{-2jk(L-R)}` | magnitude 5e-4, phase 5 mrad | passes on three materials (10 cm concrete, 6 mm glass, 2 mm board) at three incidence angles; agreement 3e-5 and 0.3 mrad. |
| Survival, M0-6 (`test_survival.py`, `survival_cases.py`) | no unnamed exception, a correct result (finite plus a per-case check) or a `ValueError`/`TypeError`/`NotImplementedError` that names its cause, and a usable CUDA context afterwards; each case in its own interpreter with a timeout | exact | see "Survival" below. |

### Measured precision (the M2-1 table)

Channel 0.5.0, 64 samples at 10 MS/s, two chirps 1 ms apart, `Motion.adc()`,
co-located element, radial motion. Velocity case 10 m/s; acceleration case
1 m/s plus 400 m/s^2.

| Range | Trajectory | Integrated error | Spur floor | Phase rms | Phase peak | Float32 step `4 pi ulp32(R) / lambda` |
|---|---|---|---|---|---|---|
| 2 m | constant velocity | -72.9 dBc | -75.8 dBc | 2.3e-4 rad | 4.2e-4 rad | 7.7e-4 rad |
| 2 m | constant acceleration | -73.0 dBc | -75.6 dBc | 2.2e-4 rad | 4.1e-4 rad | 7.7e-4 rad |
| 30 m | constant velocity | -54.8 dBc | -56.7 dBc | 1.8e-3 rad | 3.5e-3 rad | 6.2e-3 rad |
| 30 m | constant acceleration | -55.0 dBc | -57.2 dBc | 1.8e-3 rad | 3.5e-3 rad | 6.2e-3 rad |
| 100 m | constant velocity | -42.7 dBc | -48.6 dBc | 7.3e-3 rad | 1.4e-2 rad | 2.5e-2 rad |
| 100 m | constant acceleration | -42.5 dBc | -46.8 dBc | 7.5e-3 rad | 1.4e-2 rad | 2.5e-2 rad |

Amplitude agrees to 3e-7 everywhere, so the whole error is phase. The
integrated errors reproduce the audit's computed -73 / -55 / -43 dBc; the
measured spur floor at 100 m (-47 to -49 dBc) matches its -48 dBc estimate.
The M2-2 target (-80 dBc at 100 m) is 32 to 37 dB below today, and even 2 m
misses it by 4 dB on the spur floor and 7 dB on the integrated error.

### Survival

Each case runs in its own interpreter; every one exits cleanly and leaves the
CUDA context usable today.

| Case | Today | Verdict |
|---|---|---|
| target occluded mid-sequence | `ValueError: site ... has no inbound leg row in the frozen topology` | strict xfail (MP-2); expected: completes, the occluded frame exactly zero |
| site on a PEC wall | 0.5.0: 4 rows, echo exactly 0; 0.6.0: 1 row (line of sight), free-space echo | passes on 0.5.0 (the tangential field vanishes on a conductor); strict xfail on Channel >= 0.6 (C-12) |
| site 0.05 mm in front of a PEC wall | 0.5.0: 4 rows, echo `(2 sin(k delta))^4` of free space (-32 dB); 0.6.0: 1 row, free-space echo (+32 dB too strong) | passes on 0.5.0 against the image method to 3%; strict xfail on Channel >= 0.6 (C-12) |
| site on an array element | 1 row, finite cube of mean power 7.6e29 W, no refusal | strict xfail (no audit ID; cited as M0-6); expected: a named refusal |
| empty scene | 1 row, radar-equation level | passes |
| no targets | `ValueError: site_count must be a positive int, got 0` | passes as a named refusal |
| single triangle | 4 rows | passes |
| degenerate mesh (zero-area and duplicate faces) | 4 rows, no duplicated reflection | passes |
| 50 km | radar-equation level (1.2e-29 W) | passes |
| 20 000 sites | 20 000 rows | passes |

### Channel 0.6.0

The whole directory was also run in `radar-rel` (Channel 0.6.0, tag build
65b6da0). Every oracle number above is unchanged to the printed precision,
including the D-1 table. The one difference is C-12: Channel 0.6 rejects
reflection segments shorter than about 1e-5 of the scene diagonal, so a site on
or 0.05 mm from a wall keeps only its line of sight. The two wall-adjacent
survival cases carry a strict xfail conditioned on the installed Channel
release, so the suite is green on both and flips when either side restores the
cancellation.

## Consequences

- A physics change cites the oracle it moves. The fix for a marked finding
  removes its marker in the same commit; a strict XPASS is a failure, so a
  fix that forgets cannot land green.
- The references are deliberately narrow: narrowband at the carrier, the
  quasi-static observation model (no light-time retardation), TE incidence
  for the material oracle, far-field canonical shapes. A model change that
  goes beyond them (wideband sub-bands, retardation, TM or cross-polarization)
  extends the reference rather than loosening a tolerance.
- The survival child prints a record and never judges it, so the verdict and
  its acceptance rules stay in one file.

## Amendment: the published Channel 0.6.0

The published Channel 0.6.0 (tag c287e11) closes C-12: a site on or 0.05 mm in front of a PEC wall
keeps its four rows, the on-wall echo cancels below -60 dB and the near-wall echo matches the image
method to 3 %. `site_on_wall` and `site_near_wall` are ordinary cases; the table above records what
the unpublished earlier 0.6.0 build did.
