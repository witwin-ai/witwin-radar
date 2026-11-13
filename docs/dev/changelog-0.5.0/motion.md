## Breaking changes

- `Motion.auto()` (the default) resolves to `Motion.adaptive()` for a moving FMCW scene instead of evaluating the world at every ADC instant. A moving OFDM or pulsed scene, or one that declares a discovery cadence, still gets the exhaustive schedule. Migration: pass `motion=Motion.adc()` where the exhaustive reference is wanted.
- `Result.motion_sampling` / `Paths.kind` name the schedule that ran: a moving OFDM run publishes `"symbol"` and a moving pulsed run `"pulse"` (previously `"adc"`).
- `motion_sampling_exhaustive` is true only when the world was evaluated at every waveform sample instant (an ADC-refreshed FMCW frame or a still world). Moving OFDM and pulsed runs and FMCW `Motion.chirp()` now report `False`.
- `Motion.adaptive()` no longer takes `rediscover_every_frames`; adaptive sampling owns its discovery cadence (it was already refused when the session opened).

## Added

- `Result.events` and `Paths.events`: per frame, a `PathEvents` record of every path birth and death, with the instant it was seen and the last instant before it, the round trip's source/site/sink IDs, the legs' component IDs, the interaction depth and the amplitude at the event. Detected by a native kernel; materialised on first read.
- `Result.birth_lag_bound_s` / `Paths.birth_lag_bound_s`: per frame, how late a path birth can have been seen under the discovery cadence actually used.
- Discovery cadences for `Motion.adc()`, `Motion.chirp()` and `Motion.auto()`: `rediscover_every_chirps` (discover per chirp, replay per ADC sample), `rediscover_every_seconds` and `rediscover_on_displacement`, besides `rediscover_every_frames`. At most one may be declared.
- `Motion.adaptive(motion_bandwidth=..., max_acceleration=...)`: declared motion bounds that cap the adaptive probe spacing.
- `Aspect.axis` may be a function of time, `axis(t) -> (S, 3)`: a rotating lobe or blade flash, evaluated at every observation instant.
- `Radar.simulate` / `trace` / `stream` take `pose=`, the radar's ego-motion `pose(t) -> (rotation, translation)`: element positions, polarizations and the antenna-pattern orientation follow it at every observation.

## Fixed

- A target walking behind an occluder no longer stops the run: a declared site with no path at a discovery contributes exactly zero (an unknown site ID is still refused).
- Adaptive sampling checks every accepted interval at one jittered instant off its probe grid, so motion periodic at the grid's step (previously accepted with ~87 % IQ error) is refined.
- Adaptive sampling treats a specular point crossing the edge of two coplanar triangles as one continuous reflection in a static world: a subdivided wall needs the same number of probes as an undivided one.
- The adaptive route now includes a declared `Leakage` (it was dropped).

## Performance

- The default for a moving FMCW scene evaluates a handful of probes per frame instead of every ADC instant: 6 of 2048 observations per frame on a 48-site articulated body (0.31 s against 157 s for `Motion.adc()`, contended), 3 to 14 of 128 on the Doppler-precision fixtures.
