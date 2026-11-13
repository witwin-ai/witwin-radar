# Motion: how often the world is resampled

A frame is not an instant. An FMCW frame samples the echo at every ADC
instant of every chirp; a target, a structure or the radar itself moves
between those instants, and that motion is the Doppler and micro-Doppler the
radar measures. `motion=` on `trace`, `simulate` and `stream` chooses how
often the world (propagation, round trips, scattering, antenna weights) is
evaluated inside a frame. It never changes how often the waveform is sampled.

| `Motion` | Observes the world | Exhaustive | Waveforms |
|---|---|---|---|
| `Motion.auto()` (default) | resolves to one of the kinds below | - | all |
| `Motion.adaptive(...)` | at error-controlled probes, interpolated between them | no | FMCW, one uniform TDM ramp |
| `Motion.adc(...)` | at every ADC (fast-time) instant | yes | all (OFDM CFR: per symbol) |
| `Motion.chirp(...)` | once per chirp, symbol or pulse (stop and hop) | no | all |
| `Motion.static()` | once per frame; refused if anything moves | yes, for a still world | all |

**What `auto` resolves to.** A session "moves" if a target has a
`trajectory`, a Core structure has a trajectory or deformation, an endpoint
moves, the radar has an ego-motion `pose=`, an `Aspect.axis` is a function of
time, or the receiver has oscillator phase noise (which needs real
observation instants). Then:

- a moving FMCW session on **one uniform TDM ramp** with no declared discovery
  cadence resolves to **`adaptive`** with its default tolerances (a
  phase-noise receiver on such a ramp included);
- any other moving session - OFDM, pulsed, an FMCW chirp `sequence`, a ramp
  `phase_error`, DDM or BPM MIMO, or one that declares a discovery cadence -
  resolves to **`adc`**, the exhaustive reference; so does every moving
  `Network`;
- a still session is evaluated **once per frame**.

## Setup: one radar, four choices

The radar below is deliberately small (16 samples x 16 chirps, 256 ADC
instants a frame) so that the exhaustive route is cheap enough to compare.

```python
import dataclasses

import torch
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure
from witwin.radar import Aspect, Fluctuation, Fmcw, Motion, Network, PointTargets, Radar

chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=16, chirps_per_frame=16,
             adc_start=1e-6, idle=4e-6, ramp_end=6e-6)
radar = Radar(reference_frequency_hz=77e9, waveform=chirp, power=15.0, tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0),))
# 3 m ahead, closing at 20 m/s.
mover = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0,
                     trajectory=lambda t: torch.tensor([[0.0, 0.0, -3.0 + 20.0 * t]], device="cuda"))

runs = {}
for name, motion in (("auto", None), ("adaptive", Motion.adaptive()), ("adc", Motion.adc()),
                     ("chirp", Motion.chirp()), ("adc, per chirp", Motion.adc(rediscover_every_chirps=1))):
    result = radar.simulate(Scene(), mover, times=(0.0,), motion=motion)
    runs[name] = result
    probes = result.adaptive_diagnostics[0]["evaluations"] if result.adaptive_diagnostics else "-"
    print(f"{name:15s} ran {result.motion_sampling:8s} schedule {result.observation_schedule:9s} "
          f"instants {len(result.sample_times[0]):3d} discoveries {result.discovery_count:3d} probes {probes} "
          f"complete {result.path_set_complete!s:5s} exhaustive {result.motion_sampling_exhaustive}")

reference = runs["adc"].cube
for name in ("adaptive", "chirp"):
    error = float((runs[name].cube - reference).norm() / reference.norm())
    print(f"{name} vs adc: relative error {error:.1e}")
assert torch.equal(runs["auto"].cube, runs["adaptive"].cube)
assert float((runs["adaptive"].cube - reference).norm() / reference.norm()) < 1e-2
```

What the table shows:

- `auto` ran `adaptive`: same cube, bit for bit, as `Motion.adaptive()`.
- `adc` evaluates the world at all 256 instants and rediscovers the path
  topology at each one. It is the reference; its cost is one discovery per
  ADC instant of every chirp, which for the quickstart radar (256 samples x 64
  chirps) is 16,384 per frame.
- `adaptive` evaluates a handful of probes (the `probes` column) and
  interpolates the transport between them. It matches `adc` to the tested
  phase tolerance. On the quickstart radar a moving target typically costs
  about ten probes per frame instead of 16,384 instants.
- `chirp` holds the world over each chirp: 16 evaluations, but no range
  migration or phase change inside a chirp, which is why it differs more.
- `adc, per chirp` samples every ADC instant but discovers the topology once
  per chirp and replays it in between: exhaustive, but no longer complete.

`Result.motion_sampling` names the route that ran (`"static"`, `"adaptive"`,
`"adc"`, `"chirp"`, and `"symbol"`/`"pulse"` for OFDM/pulsed held per slot);
`Result.observation_schedule` is the synthesis owner's name for the same
thing (`"frame"`, `"adaptive"`, `"fast_time"`, `"slow_time"`).
`Result.sample_times[i]` is frame `i`'s whole observation schedule, a float64
array of absolute times in seconds.

## Two completeness statements

A run publishes two separate statements, and a consumer that needs to know how
far to trust a cube needs both:

- `path_set_complete`: no path within the requested propagation can have been
  missed - every observation rediscovered, or the candidate family certified
  for all time - and Channel's own discovery was complete
  (`discovery_complete`, see [Multipath](multipath.md)).
- `motion_sampling_exhaustive`: the world was evaluated at every instant the
  waveform samples; nothing was interpolated or held over a slot.

`Result.semantics` publishes them in Channel's vocabulary:
`{"estimator": "coherent", "combine": "field", "complete": ..., "exhaustive": ...}`.
An adaptive run is never exhaustive, and it is complete only when the probed
topology was certified for all time. Its probes cannot see an arbitrarily
brief event between them.

```python
for name, result in runs.items():
    print(f"{name:15s} {result.semantics}")
```

## Adaptive sampling

`Motion.adaptive(...)` probes the world on a nested grid, interpolates each
path's delay, phase and amplitude between probes, and accepts an interval only
when the error at the probes between its nodes is inside the tolerances.

| Field | Default | Meaning |
|---|---|---|
| `phase_error` | 0.02 | largest tested round-trip phase error per path, rad |
| `relative_amplitude_error` | 0.02 | largest tested relative amplitude error per path |
| `max_interval` | 0.002 | maximum probe spacing, s; always enforced |
| `nodes` | 2 | instants one accepted interval interpolates through (2 linear, 5 quartic) |
| `max_evaluations` | 8192 | probe budget per frame; exhausting it raises |
| `batch_observations` | 256 | instants replayed, composed and synthesized together |
| `motion_bandwidth` | `None` | declared bound on the motion's bandwidth, Hz; refines the grid to four probes per period |
| `max_acceleration` | `None` | declared bound on any site's or endpoint's acceleration, m/s^2; caps the probe spacing |

Every tolerance is checked by sampling, so it bounds only what it samples:
motion periodic at the grid's step is invisible to the probes. `max_interval`
sets that step (the default 2 ms resolves roughly 250 Hz at two nodes), and
each accepted interval is also checked at one jittered off-grid instant.
Declaring `motion_bandwidth` or `max_acceleration` is how you tell the
controller about fast motion it could otherwise alias. Raise `nodes` only
when the phase test is what shortens intervals (micro-Doppler at a high
carrier); when `max_interval` sets the length, a higher order only adds probes.

`Result.adaptive_diagnostics` holds one plain `dict` per frame:

```python
tight = radar.simulate(Scene(), mover, times=(0.0,),
                       motion=Motion.adaptive(phase_error=0.005, nodes=3, motion_bandwidth=2e3))
stats = tight.adaptive_diagnostics[0]
for key in ("evaluations", "observation_count", "accepted_intervals", "topology_refinements",
            "verification_probes", "verification_refinements", "max_tested_phase_error_rad",
            "max_interval_s", "interpolation_nodes", "complete"):
    print(f"{key:28s} {stats[key]}")
assert stats["max_tested_phase_error_rad"] <= 0.005
```

`evaluations` is the probe count, `observation_count` the instants the frame
schedules, `topology_refinements` the intervals split because a probe saw a
different path set, `complete` the frame's share of `path_set_complete`.

Adaptive sampling is refused, by name, for:

- OFDM and pulsed waveforms (`adaptive motion currently requires FMCW`);
- an FMCW chirp `sequence`, a start offset, a ramp `phase_error`, DDM and BPM
  (`adaptive motion interpolates one uniform TDM ramp ...`);
- a `Fluctuation` of Swerling II or IV, which redraws at every observation and
  cannot be interpolated;
- a `Network` (`a network merges every radar's observation instants into one
  schedule ...`);
- `Paths.select(...)` on an adaptive trace, whose rows live in one
  interpolation table.

Adaptive sampling owns its discovery cadence: `Motion.adaptive()` takes none.

## The exhaustive reference and stop-and-hop

`Motion.adc()` evaluates the world at every fast-time instant: every ADC
sample of every chirp and transmitter for FMCW, every sample of every pulse for
pulsed, every sample of every symbol for time-domain OFDM. An OFDM CFR symbol
has no fast-time grid of its own and is evaluated once per symbol. Without a
cadence it rediscovers the topology at every instant, which is the only
schedule that cannot miss a path birth. Use it to validate a faster choice,
not as the default for long runs.

`Motion.chirp()` freezes geometry and weights within each chirp (symbol,
pulse), the classical stop-and-hop approximation. It is the cheap choice for
OFDM and pulsed radars whose fast-time refresh is expensive.

`Motion.static()` asks for one observation per frame and is refused for a
session that moves (`Motion.static() asks for one observation per frame, but
this session moves within a frame ...`), because it would publish no Doppler.

## Discovery cadences

`Motion.auto/adc/chirp` accept **at most one** discovery cadence. Between
discoveries the frozen path topology is replayed at every observation, so new
paths (a target stepping out of a shadow) are seen late:

| Cadence | Rediscovers |
|---|---|
| `rediscover_every_frames=n` | once per `n` frames |
| `rediscover_every_chirps=n` | once per `n` chirps (TDM slots); not accepted by a `Network` |
| `rediscover_every_seconds=t` | once `t` seconds of world time have passed |
| `rediscover_on_displacement=d` | once any site or endpoint moved `d` metres (one host read per replay group) |

A declared cadence makes `auto` resolve to `adc`, publishes
`path_set_complete=False` (unless structure motion already forces discovery),
and records the lag it actually incurred: `Result.birth_lag_bound[i]` is the
largest gap, in seconds, between the discoveries frame `i` relied on. For
frame 0 that is the gap within the frame; later frames also count the gap
back to the previous frame's last discovery. Transmission and diffraction,
which Channel cannot replay, refuse every cadence
([Multipath](multipath.md)).

```python
print("discoveries per frame:", runs["adc"].discovery_count, "vs", runs["adc, per chirp"].discovery_count)
print("birth lag bound:", runs["adc"].birth_lag_bound, "vs", runs["adc, per chirp"].birth_lag_bound)
try:
    Motion.adc(rediscover_every_chirps=1, rediscover_every_seconds=1e-3)
except ValueError as error:
    print("refused:", error)
```

`Motion.world` is Channel's replay vocabulary: `"frozen_world"` (default)
rediscovers whenever compiled geometry moves; `"fixed_winner_replay"` replays
the frozen winners against moved geometry, which can drop the return of a
tessellated curved reflector between discoveries (visible in the event log
and the lag bound).

`batch_observations` (default 256) is how many observation instants one
replay group composes and synthesizes together, and therefore how many rows
the fused route holds at once. `Motion.adaptive(batch_observations=...)` sets
it; for the other kinds use `dataclasses.replace`:

```python
small_groups = dataclasses.replace(Motion.adc(), batch_observations=64)
assert torch.equal(radar.simulate(Scene(), mover, times=(0.0,), motion=small_groups).cube, runs["adc"].cube)
```

## Path births and deaths

Paths appear and disappear as steps (a target walking into a shadow). Every
frame publishes the steps it saw in `Result.events[i]`, a `PathEvents` record
read lazily: `birth` (true for a birth, false for a death), `previous_time_s`
and `time_s` (the event lies in between), `observation` (its index in
`sample_times`), the round trip's `radar_source_id`, `site_id`,
`radar_sink_id`, `component` and `depth`, and `amplitude` (`|C|` of the live
side). Events are diagnostics: discrete, never differentiable. The first
observation of a session has no predecessor, so its paths are not births.

Here a target slides behind a conducting plate 50 us into the frame:

```python
plate = Structure(geometry=Mesh(vertices=torch.tensor([[0.3, -1.0, -2.0], [1.3, -1.0, -2.0],
                                                       [1.3, 1.0, -2.0], [0.3, 1.0, -2.0]]),
                                faces=torch.tensor([[0, 1, 2], [0, 2, 3]]), recenter=False),
                  material=PhysicalMaterial.perfect_conductor())
walker = PointTargets(positions=[(0.595, 0.0, -4.0)], rcs=1.0,
                      trajectory=lambda t: torch.tensor([[0.595 + 100.0 * t, 0.0, -4.0]], device="cuda"))
for motion in (Motion.adc(), Motion.adaptive(), Motion.chirp()):
    result = radar.simulate(Scene(structures=(plate,)), walker, times=(0.0,), reflections=0, motion=motion)
    events = result.events[0]
    print(f"{result.motion_sampling:8s} {len(events)} event(s): birth={events.birth.tolist()} "
          f"between {events.previous_time_s[0] * 1e6:.1f} us and {events.time_s[0] * 1e6:.1f} us")
    assert len(events) == 1 and not events.birth[0]
    assert events.previous_time_s[0] < 50e-6 <= events.time_s[0] + 1e-12
```

The line of sight is cut when the target crosses `x = 0.6 m`, 50 us into
the frame. The exhaustive route brackets it between adjacent ADC instants;
adaptive between adjacent probes; stop-and-hop between chirps.

## Ego-motion and rotating aspect

`pose=` on the verbs is the radar's own motion, `pose(t) -> (rotation,
translation)`: a `(3, 3)` local-to-world rotation applied on the left of the
authored orientation, about the authored position, and a `(3,)` world
translation in metres. Element positions, polarizations and the antenna
pattern all follow it at every observation. A pose counts as motion.

```python
def ego(t):
    """The radar drives toward the target at 20 m/s."""
    return torch.eye(3), torch.tensor([0.0, 0.0, -20.0 * t])


still = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0)
driving = radar.simulate(Scene(), still, times=(0.0,), motion=Motion.adc(), pose=ego)
relative = float((driving.cube - runs["adc"].cube).norm() / runs["adc"].cube.norm())
print(f"ego-motion vs moving target: relative difference {relative:.1e}")
assert relative < 1e-2
```

An `Aspect` whose `axis` is a function of time, `axis(t) -> (S, 3)` unit
vectors, is a rotating lobe or a blade flash; it is evaluated at every
observation instant and also makes the session move:

```python
def spinning(t):
    angle = 2.0 * torch.pi * 2000.0 * t  # 2 kHz rotation about +y
    return torch.tensor([[float(torch.sin(torch.tensor(angle))), 0.0, float(torch.cos(torch.tensor(angle)))]],
                        device="cuda")


blade = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0,
                     aspect=Aspect(axis=spinning, exponent=2.0, coherent_interval=1.0))
flash = radar.simulate(Scene(), blade, times=(0.0,), motion=Motion.chirp())
per_chirp = flash.cube[0, 0, 0].abs().amax(dim=-1)
print("per-chirp peak, relative:", [round(float(v), 2) for v in per_chirp / per_chirp.max()])
```

A time-varying `Aspect.axis` and `pose=` are refused with `wideband=True`.

## More refusals

```python
def refused(call):
    try:
        call()
    except (ValueError, NotImplementedError) as error:
        return f"{type(error).__name__}: {str(error)[:110]}..."
    return "not refused"


swerling_ii = dataclasses.replace(mover, fluctuation=Fluctuation(swerling=2, coherent_interval=1e-3))
print(refused(lambda: radar.simulate(Scene(), mover, times=(0.0,), motion=Motion.static())))
print(refused(lambda: radar.simulate(Scene(), swerling_ii, times=(0.0,), motion=Motion.adaptive())))
print(refused(lambda: Network((radar,)).simulate(Scene(), mover, times=(0.0,), motion=Motion.adaptive())))
```

See also: [Targets](targets.md), [Multipath](multipath.md),
[Waveforms](waveforms.md), [Pipeline guide](../pipeline_guide.md).
