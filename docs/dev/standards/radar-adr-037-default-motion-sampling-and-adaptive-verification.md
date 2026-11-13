# R-ADR-037: Adaptive sampling is the default for a moving FMCW scene, and its tests no longer share one grid

Status: Accepted

## Context

* Any moving scene defaulted to the exhaustive ADC route, which rediscovers at
  every ADC instant: 98,304 discoveries for a 128 x 3 x 256 frame (P-1, U-11).
  The owner decided (plan 4.11, decision 4) that the default for a dynamic
  scene becomes `adaptive`, with `Motion.adc()` the named exhaustive reference.
* A moving OFDM or pulsed run published the sampling kind `"adc"` and
  `motion_sampling_exhaustive=True` although it evaluates the world once per
  symbol or pulse (D-6).
* Every adaptive error test sampled the same nested, commensurate grid, so
  motion periodic at the grid's step passed with 87 % IQ error, and the only
  control over the grid was `max_interval` (D-8).
* The adaptive identity named the triangle a reflection lies on, so a
  specular point crossing the shared edge of two coplanar triangles counted as
  a topology change and forced refinement down to enumeration; a subdivided
  wall multiplied the probe count (D-7, MP-13).

## Decision

### The published schedule (D-6)

`motion_sampling` names the schedule that ran, not the request: `"static"`,
`"chirp"`, `"adc"`, `"adaptive"`, and, for a moving OFDM or pulsed session
whatever was requested, `"symbol"` or `"pulse"`. `motion_sampling_exhaustive`
is true exactly when the world was evaluated at every instant the waveform
puts a sample at: an ADC-refreshed FMCW frame or a still world. Stop-and-hop
(`"chirp"`), `"symbol"` and `"pulse"` hold the world over a chirp, symbol or
pulse and are not exhaustive; an adaptive frame is decided by its controller.
If pulsed fast-time refresh lands (plan M2-9), the rule is unchanged: a pulsed
schedule that evaluates every fast-time sample publishes `exhaustive=True`.

### The default (P-1, owner decision 4)

`Motion.auto()` resolves to `adaptive` with the default tolerances when the
session moves and the waveform is FMCW and no discovery cadence is declared.
An OFDM or pulsed moving session, or one that declares a cadence, resolves to
the exhaustive schedule; a still world to one observation a frame.
`Motion.adc()` is unchanged and is the reference every adaptive test compares
against. The adaptive echo now adds a declared `Leakage` (the coupling is a
static row per pair, so it is the full-frame beat cube whatever the sampling);
before, the adaptive route dropped it.

Measured, same process, contended GPU (`scratchpad/motion/measure_auto.py`),
`Motion.auto()` against `Motion.adc()` and against the float64 reference of
`tests/oracle/test_doppler_precision.py` (64 samples, 2 chirps 1 ms apart):

| Fixture | auto vs adc (relative IQ) | adc vs float64 | auto vs float64 | probes / observations | adc s | auto s |
|---|---|---|---|---|---|---|
| 2 m, 10 m/s | 2.9e-4 | -72.9 dBc | -75.6 dBc | 3 / 128 | 3.46 | 0.10 |
| 2 m, 1 m/s + 400 m/s^2 | 2.4e-3 | -73.0 dBc | -52.4 dBc | 3 / 128 | 5.12 | 0.05 |
| 30 m, 10 m/s | 2.2e-3 | -54.8 dBc | -60.0 dBc | 3 / 128 | 1.41 | 0.09 |
| 30 m, 1 m/s + 400 m/s^2 | 2.6e-3 | -54.9 dBc | -55.6 dBc | 3 / 128 | 2.31 | 0.02 |
| 100 m, 10 m/s | 9.9e-3 | -42.7 dBc | -41.9 dBc | 14 / 128 | 1.96 | 0.05 |
| 100 m, 1 m/s + 400 m/s^2 | 8.4e-3 | -42.5 dBc | -48.0 dBc | 3 / 128 | 1.54 | 0.02 |
| 48-site articulated body (SMPL-like limb swing 1.5 Hz, 0.6 rad), 32 x 64, two frames | 3.6e-3 | - | - | 6 / 2048 per frame | 157.27 | 0.31 |

The default's error against the exhaustive route is within its declared
phase tolerance (0.02 rad, about 2e-2 relative) everywhere, and against the
float64 reference it is at the level of the float32 staircase the exhaustive
route itself carries (D-1), except the 2 m accelerating case, where the
linear interpolant's curvature error (-52 dBc, inside the tolerance) shows
above the staircase. Callers who need the float32 floor everywhere name
`Motion.adc()` or tighten `phase_error`. The articulated-body fixture is a
many-site stand-in built in the measurement script; the SMPL model files are
not in this checkout.

### Verification off the grid and declared motion bounds (D-8)

Every candidate interval wide enough to carry the grid is also tested at one
verification instant: an interior schedule index chosen by golden-ratio
jitter of the interval's ends, moved to the nearest index that is not a grid
point. It is interpolated from the same nodes as the grid's test points, and
the interval is accepted only if both pass. The instant is deterministic for a
partition and never a point of the nested dyadic grid's first levels, so a
motion that sits at the grid's zeros at every level is seen. It costs one
probe per candidate interval, evaluated in the same batch as the grid.
`adaptive_diagnostics` reports `verification_probes` and
`verification_refinements` (intervals only the verification rejected).

`Motion.adaptive(motion_bandwidth=B)` caps the accepted interval at
`(nodes - 1) / (2 B)`, four probes per period of the declared bandwidth.
`Motion.adaptive(max_acceleration=a)` caps it at `sqrt(2 c phi / (pi f_c a))`,
the interval over which a linear delay interpolant's carrier-phase error
`pi f_c a h^2 / (2 c)` reaches the phase tolerance `phi`; the linear bound is
used at every order because it is the conservative one. `max_interval` still
applies; the smallest cap wins and is reported as `max_interval_s`.

### Plane groups instead of triangles (D-7, MP-13)

In a world whose compiled faces never move, the adaptive identity names the
PLANE a reflection lies on. Channel publishes no kept-face or plane-group
table, so Radar learns it: each frozen leg keeps the discovery's interaction
types, normals and points (aliased from the consumer's evaluation, read once
per discovery), and a session-wide registry assigns each reflecting face, the
first time a discovery reflects off it, to a plane group: unit normals equal
to `1e-4` per component (sign fixed so the first non-zero component is
positive) and offsets equal to `1e-5` relative plus `1e-6 m`. The identity the
controller compares is the composed rows in composed order, keyed by group
for reflecting slots (raw IDs, in a separate label space, for any other
interaction) and by everything else exactly as before. Equal identities
therefore mean row `r` of one probe is row `r` of the other, which is what the
interpolation requires.

A moving or deforming structure keeps the triangle identity: a face's plane
moves with it, and a rigidly moving face's group could only be learned in the
structure's own frame, which the consumer does not publish. Two disjoint
coplanar walls share a group; a reflection jumping between them across a gap
still dies in the gap, and the error tests see a jump that lands between
probes.

## Evidence

* `tests/test_motion_schedule.py`: OFDM and pulsed publish `"symbol"` /
  `"pulse"` and `exhaustive=False` under `auto`, `adc` and `chirp`; FMCW
  `chirp` is not exhaustive and `adc` is; `auto` resolves to `adaptive` for a
  moving FMCW scene within 0.012 of `adc`; the adaptive route reproduces a
  declared leakage to `1e-3` of the coupling's own norm.
* `tests/test_adaptive_motion.py::test_the_probe_grid_bound_is_enforced_even_for_a_certified_family`:
  the motion periodic at the grid step, which used to pass with an IQ error
  above 0.5, is now refined by the verification probe to below 0.05; a
  declared bandwidth sets `max_interval_s = 1 / (2 B)` and resolves it.
* `tests/test_adaptive_motion.py::test_a_subdivided_wall_needs_no_more_probes_than_an_undivided_one`:
  a target walking past a PEC wall, 2 triangles against 32: the same probe
  count, no topology refinement, cubes equal to `1e-3`.
* `tests/test_session_budget.py` (2 x 2 array, 2 chirps of 8 samples, one
  wall, `reflections=1`), marginal frame, integration head a111481 against
  this change, same environment:

  | Case | evaluate | native | synchronizations | d2h |
  |---|---|---|---|---|
  | `adaptive_point` | 6 -> 8 | 3 -> 4 | 68 -> 96 | 61 -> 88 |
  | `adaptive_structure` | 6 -> 8 | 5 -> 7 | 159 -> 211 | 100 -> 133 |
  | `adc_point` | 64 -> 64 | 2 -> 3 | 580 -> 581 | 578 -> 578 |
  | `adc_point_per_chirp` (new) | 8 | 3 | 105 | 74 |
  | `auto_point` (new) | 8 | 4 | 96 | 88 |

  The adaptive increase is the two verification probes per frame (each a
  discovery and a replay) and, in a static world, one host read of each
  leg's interaction geometry per discovery for the plane groups; `native` is
  the event launch. `auto_point` costs what `adaptive_point` does and an
  eighth of `adc_point`'s Channel evaluations. The committed pins before this
  change were ceilings well above the measured values (for example 2884
  synchronizations for `adc_point` against 580 measured on a111481); they are
  now the measured values.
