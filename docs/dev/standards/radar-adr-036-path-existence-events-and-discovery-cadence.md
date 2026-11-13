# R-ADR-036: Path existence across observations: occluded sites, the event log and discovery cadences

Status: Accepted

## Context

A moving session rediscovers its topology at every observation by default, so
which round trips exist is re-decided at every instant. Three things were
wrong or missing at that boundary.

1. A declared scatter site that a discovery found no path to (a target behind
   an occluder) made the two-way join raise `ValueError` (MP-2). Every
   observation rediscovers, so a target walking behind a plate stopped the run.
2. Births and deaths were invisible (D-3). A geometric-optics path appears and
   disappears as a step; nothing recorded when, which path, or how strong it
   was, so a user saw only the step's broadband smear in a range-Doppler map.
3. The discovery cadence had two settings, every observation and every N
   frames (D-4). Nothing in between, and the only signal about what a longer
   cadence could have missed was the boolean `path_set_complete`.

## Decision

### Occluded sites compose to zero

`TwoWayComposer.freeze` takes `leg_site_ids`, the site endpoints both legs
were discovered for (the session passes its binding's site IDs). A declared
site in that set with no row in a leg owns no composed row and contributes
exactly zero. A declared site outside it is an unknown stable ID and is still
refused. Without `leg_site_ids` the rows are the only evidence of which sites
the legs were asked about, so the old refusal applies; the direct composer
tests keep pinning it. A join with no row at all publishes an empty
`row_valid` column, so replayed batches have one contract whatever the
topology.

### The event log

`Result.events` and `Paths.events` hold one `PathEvents` per frame: for each
birth or death the instant it was first seen (`time_s`), the last instant the
old state was seen (`previous_time_s`), its index in the frame's schedule, the
round trip's `(radar_source_id, site_id, radar_sink_id)`, the two legs'
component IDs, the summed interaction depth, and `|C|` of the live side
(the new row for a birth, the last live row for a death). The true instant
lies in `(previous_time_s, time_s]`.

Detection is the native `path_events` operator (`cuda/path_events.cu`). The
host authors, once per pair of frozen joins, which composed row of the earlier
observation is which row of the later one:

* same join object (a rediscovery that found the same topology reuses its
  join, R-ADR-024): the identity;
* different joins: rows matched by their whole key (endpoint IDs, components,
  depths, interaction sequences) with array operations (`np.unique` over the
  stacked keys), unmatched rows becoming birth or death candidates.

The kernel compares existence (`present and row_valid`) per candidate and
appends a record through an atomic counter. Candidate tables are cached by
the joins and offsets they were built from, so a frame whose topology repeats
uploads nothing. Records stay on the device until `PathEvents` is first read;
recording costs one launch per replay group (one per frame on the adaptive
route, over its probe table) and no synchronization. A world whose structures
move replays every observation alone, so there it is one launch per
observation, next to the compile and discovery each of those observations
already pays. The buffer holds
`EVENT_CAPACITY = 65536` records a frame; more are counted in `dropped`, never
silently lost. The first observation of a session has no predecessor, so its
paths are not births.

The log is discrete: payloads are read detached, the operator has no
derivative companion (`no_backward` in the native manifest), and nothing
differentiable is built from it.

A specular point crossing a triangle edge changes the row's interaction
sequence, so on the exhaustive routes the log records a death and a birth at
the same instant with nearly equal amplitudes. The adaptive route's plane
groups (R-ADR-037) are not applied to the log.

### Discovery cadences and the birth lag bound

`Motion.adc`, `Motion.chirp` and `Motion.auto` accept at most one of:

| Cadence | Meaning |
|---|---|
| (none) | rediscover at every observation; `path_set_complete` |
| `rediscover_every_chirps=N` | once per N TDM slots, replaying at every ADC instant between |
| `rediscover_every_seconds=T` | once `T` of world time has passed since the last discovery (relative tolerance `1e-9` of `T`) |
| `rediscover_on_displacement=d` | once any site or sensor endpoint has moved `d` metres since the last discovery |
| `rediscover_every_frames=N` | once per N frames |

The displacement cadence binds the replay group's positions up front and
reads them to the host in one copy per group; the decision then walks the
group in time order. Adaptive sampling owns its cadence and refuses all five.
Any declared cadence makes `path_set_complete` false on a moving session.

`Result.birth_lag_bound_s` (and `Paths.birth_lag_bound_s`, and
`PathEvents.birth_lag_bound_s`) is per frame the largest gap between
consecutive discoveries ending inside the frame, and the topology's age at the
frame's last evaluated observation. A path born just after a discovery is
first seen at the next one, so no birth the frame reports was seen later than
this. It is zero for a still world and when the candidate family is certified
complete for all time (an empty line-of-sight world on the adaptive route).

## Evidence

* `tests/oracle/test_survival.py::occluded_mid_sequence` (strict xfail
  removed): the occluded frame's power is exactly 0, the others non-zero.
* `tests/test_row_validity_ad.py`: a fresh freeze at the occluded geometry
  composes the surviving site's rows, equal to the frozen replay's live rows;
  an unknown site ID is refused.
* `tests/test_path_events.py`: a target walking behind a plate under
  `Motion.chirp()`, `Motion.chirp(rediscover_every_frames=1)` (validity
  flips), `Motion.adc()` and `Motion.adaptive()` logs one death and one birth,
  each bracketing the analytic shadow-edge instant `(-1 - y0) / v` and
  `(1 - y0) / v`; the occluded chirps are exactly zero. A birth's
  range-Doppler signature: the occluded cube equals the unobstructed cube
  times the gate the log reports, to `1e-6`, and its Hann Doppler cut leaks
  above -25 dB away from the main lobe where the ungated tone is below -40 dB.
* `tests/test_motion_schedule.py`: the per-chirp cadence makes `1 / samples`
  of the ADC route's discoveries with the same cube to `1e-5`; the time and
  displacement cadences discover exactly where a numpy walk of the schedule
  says; the lag bound equals the measured discovery gap.
