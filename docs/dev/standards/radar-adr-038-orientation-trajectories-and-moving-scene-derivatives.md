# R-ADR-038: Orientation trajectories, fixed-winner replay on curved reflectors, and moving-scene derivative oracles

Status: Accepted

## Context

`Aspect.axis` was a static world-frame tensor, so a rotating body's lobe did
not turn and a blade flash could not be expressed; the radar's pattern
orientation was the authored pose even when its platform moved (D-10). The
derivative of the scene route had only been checked as VJP against JVP, never
against finite differences, with no test across a path event, no moving OFDM
or pulsed scene and one `fixed_winner_replay` test; the dropout of a curved
reflector under fixed-winner replay was inferred, not shown (D-12, D-11).

## Decision

### A time-varying aspect axis

`Aspect.axis` may be `axis(t) -> (S, 3)` unit vectors. The session counts as
moving, and the record resolves at each replay group's instants into an
ordinary `AspectScatterResponse` with one site per (instant, target), slot
major. The slot-batched join gives slot `g`'s site `s` the response site
`g S + s` (`TwoWayComposer.for_slots(..., per_slot_response=True)`: a shifted
site index and a per-slot site CSR, integer tables only), so the native aspect
kernel evaluates every row with its own instant's axis and reduces each axis
gradient into its own slot. A tensor the function returns keeps its tape.
Building the per-group response re-runs that record's unit-norm check, one
synchronization per replay group.

### A radar pose trajectory

`Radar.simulate/trace/stream(..., pose=...)` takes
`pose(t) -> (rotation (3, 3), translation (3,))`: a local-to-world rotation
composed on the left of the authored orientation, about the radar's authored
position, and a world translation. At each observation the elements are
carried, `p(t) = c + T(t) + R(t)(p - c)`, and their polarizations turned. The
pattern stage reads only directions and maps them through the authored
pattern frame, so the session hands it each slot's points turned by
`R_g^T` (one batched `torch.matmul` per point set, nothing per row): each
direction is then expressed in the body frame and the authored frame gives
the pattern orientation the pose carried. The isotropic default skips this
entirely. The session counts as moving whenever a pose is given.

### Fixed-winner replay on a tessellated curved reflector

Not fixed; documented and made visible. Fixed-winner replay holds each
discovered interaction sequence while the geometry moves. On a curved,
tessellated reflector the specular point slides off its frozen facet; the row
then reports `row_valid=False` and the neighbouring facet's reflection is not
born until the next discovery. The event log (R-ADR-036) records the death and
no birth, and `birth_lag_bound_s` reports the whole gap. A caller whose
reflectors are curved and moving either rediscovers per observation
(`frozen_world`, the default) or declares a cadence short enough for its
facet size.

## Evidence

* `tests/test_orientation_trajectories.py`: a lobe turning at `Omega` gives
  per-chirp magnitudes equal, to `1e-4`, to `max(-cos Omega t, 0)^(2 n)` times
  one constant, with flashes `2 pi / Omega` apart; a yawing radar with a
  dipole pattern matches, chirp by chirp, separate static radars authored at
  each instant's heading (`2e-4`); a pose translation closing on a still
  target equals the target closing on a still radar.
* `tests/test_fixed_winner_replay.py`: on a translating flat wall fixed-winner
  replay with one discovery a frame matches per-observation rediscovery
  (`1e-4`) with the same compile count; on a twelve-facet cylinder it logs
  deaths and no birth, with the lag bound equal to the frame span, while
  per-observation rediscovery hands the reflection on.
* `tests/test_motion_gradients.py`: the reverse-mode derivatives with respect
  to a target's initial position and velocity match central finite
  differences (step `2^-16` m and `2^-6` m/s) to 2 %, and forward mode matches
  reverse to `1e-3`, under `Motion.adc()` and the adaptive default, and across
  a frame that contains an occlusion death.
* `tests/test_motion_schedule.py`: moving OFDM and pulsed scenes carry the
  slow-time phase `-2 pi f d tau` of a float64 geometric delay (OFDM per
  subcarrier, `2e-3` rad; pulsed mid-pulse, `5e-3` rad) and the `R^-4` level.
