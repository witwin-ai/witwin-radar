# Several radars: networks, bistatic links and interference

A `Network` simulates several radars in one scene with one shared world half:
the scene is compiled once and every propagation leg is discovered once per
topology epoch for all the radars' elements together, so the Channel cost of
a network is about that of one radar. Each radar publishes its own `Result`,
and a radar hears its own transmitters exactly as it would alone. Declared
`links` let a radar hear another radar too: bistatic target returns, the
direct line of sight (mutual interference) and the environment's returns.
`Emitter` adds a non-radar transmitter.

## The record and its verbs

```text
Network(radars, links=(), clock_offsets=(), if_bandwidths=(), carrier_offsets=(), emitters=())
```

| Field | Meaning |
|---|---|
| `radars` | the radars, a tuple; results come back in this order |
| `links` | `(i, j)` pairs: radar `j` hears radar `i`'s transmissions |
| `clock_offsets` | per radar, s: radar `k`'s frame `f` starts at `times[f] + clock_offsets[k]` |
| `if_bandwidths` | per FMCW receiving radar, Hz: its IF (anti-alias) band; `None` is half its ADC rate |
| `carrier_offsets` | per pulsed or OFDM radar, Hz: its oscillator offset from the shared carrier |
| `emitters` | `Emitter` records, heard by every radar |

The verbs mirror the radar's: `network.trace(scene, targets, times=...)`
returns the shared paths, `network.echo(paths)`, `network.simulate(...)` and
`network.stream(...)` return one `Result` per radar (`stream` yields one tuple
per frame). They take the radar verbs' keywords (`los`, `reflections`,
`transmission`, `diffraction`, `environment`, `coupling`, `discovery`,
`max_paths`, `max_round_trips`, `motion`, `ad_mode`) plus:

* `poses=`: one `pose(t)` (or `None`) per radar, the same ego-motion contract as
  `Radar.simulate(pose=...)`;
* `direct=True`: keep the straight segment between the radars of each link
  (and from each emitter), the direct interference path;
* `environment=True`: add the environment's returns between every pair;
  `coupling=True`: the straight segment between a radar's own transmit and
  receive elements.

The world is shared, so every radar has one carrier
(`reference_frequency_hz`), one device and one element-pattern pair.

## Independent radars sharing a world

Without links each radar's cube is exactly what `Radar.simulate` gives it
alone. The two radars below face each other 4 m apart in a Y-up world (the
default `up=(0, 1, 0)`), with a target between them.

```python
import dataclasses
import math

import torch
from witwin.core import Scene
from witwin.radar import Chirp, Emitter, Fmcw, Motion, Network, Ofdm, PointTargets, Radar

C = 299_792_458.0
chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=256, chirps_per_frame=16,
             adc_start=6e-6, idle=7e-6, ramp_end=65e-6, output="beat")
elements = dict(antenna_unit="half_wavelength", tx=((0.0, 0.0, 0.0),), rx=((1.0, 0.0, 0.0),))
a = Radar(reference_frequency_hz=77e9, waveform=chirp, power=15.0, **elements,
          position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0))
b = Radar(reference_frequency_hz=77e9, waveform=chirp, power=15.0, **elements,
          position=(4.0, 0.0, 0.0), look_at=(0.0, 0.0, 0.0))
scene = Scene()
target = PointTargets(positions=[(2.0, 1.0, 0.0)], rcs=1.0)
times = (1e-3,)

results = Network((a, b)).simulate(scene, target, times=times)
alone = b.simulate(scene, target, times=times)
print("radar b in the network == radar b alone:", torch.equal(results[1].cube, alone.cube))
assert torch.equal(results[1].cube, alone.cube)


def range_peak(cube, axes):
    """The range (m) of the strongest bin of a beat cube [frame, tx, rx, chirp, sample]."""
    spectrum = torch.fft.fft(cube[0], dim=-1).abs().square().sum(dim=(0, 1, 2))
    return float(axes.range_m[int(spectrum.argmax())])
```

## A bistatic link

`links=((0, 1),)` puts radar `a`'s transmissions into radar `b`'s receiver.
With synchronized clocks, `b` hears the target at the bistatic range: an FMCW
beat at `S (R1 + R2) / c`, which `b`'s range axis reads as `(R1 + R2) / 2`.
`direct=False` leaves out the line of sight between the radars so the
difference is the target's bistatic return alone:

```python
linked = Network((a, b), links=((0, 1),)).simulate(scene, target, times=times, direct=False)
bistatic = linked[1].cube - alone.cube
r1 = math.dist((0.0, 0.0, 0.0), (2.0, 1.0, 0.0))
r2 = math.dist((4.0, 0.0, 0.0), (2.0, 1.0, 0.0))
heard = range_peak(bistatic, alone.axes)
print(f"bistatic return at {heard:.2f} m; (R1 + R2) / 2 = {(r1 + r2) / 2:.2f} m")
assert abs(heard - (r1 + r2) / 2) <= alone.axes.range_bin_m
print("radar a is unchanged by the link:", torch.equal(linked[0].cube, results[0].cube))
```

## FMCW mutual interference

With `direct=True` (the default) a link also carries the line of sight from
`a`'s transmitter to `b`'s receiver. Each FMCW receiver de-chirps the other
radar's ramp with its own: the instantaneous beat frequency
`S_B v - S_A w` (plus any start-frequency difference) reaches the samples
only while it lies inside the receiver's IF band.

**Same slope: a ghost target.** Two radars with the same ramp and a
start-time offset `dT` (`clock_offsets`) make a steady tone at
`S (tau + dT)`: a ghost at range `c (tau + dT) / 2` with zero Doppler.

```python
d_t = 10e-9
network = Network((a, b), links=((0, 1),), clock_offsets=(d_t, 0.0))
with_direct = network.simulate(scene, target, times=times, direct=True)[1]
without = network.simulate(scene, target, times=times, direct=False)[1]
ghost = with_direct.cube - without.cube
tau = 4.0 / C
print(f"ghost at {range_peak(ghost, alone.axes):.2f} m; c (tau + dT) / 2 = {C * (tau + d_t) / 2:.2f} m")
assert abs(range_peak(ghost, alone.axes) - C * (tau + d_t) / 2) <= alone.axes.range_bin_m
```

**Different slopes: a broadband burst.** When the slopes differ the beat
frequency sweeps through the IF band, and the receiver sees a short burst,
`f_s / |S_A - S_B|` seconds long, where
`|(S_B - S_A) v + S_A (tau + dT)| <= f_s / 2`. Its spectrum is spread over
every range bin: it raises the noise floor rather than adding a ghost.

```python
steep = dataclasses.replace(a, waveform=dataclasses.replace(chirp, slope=61e12))
d_t = 0.48e-6
network = Network((steep, b), links=((0, 1),), clock_offsets=(d_t, 0.0))
burst = (network.simulate(scene, target, times=times, direct=True)[1].cube
         - network.simulate(scene, target, times=times, direct=False)[1].cube)[0, 0, 0]  # [chirp, sample]
hit = (burst.abs() > 0).sum(dim=-1)
expected = chirp.sample_rate ** 2 / 1e12  # samples: (f_s / |dS|) * f_s
print(f"burst length per chirp: {hit.tolist()[:4]} samples; f_s^2 / |dS| = {expected:.1f}")
assert all(abs(int(n) - expected) <= 1.5 for n in hit)
```

`if_bandwidths` narrows a receiver's IF band (default `sample_rate / 2`). A
start-frequency offset between FMCW radars is a one-profile chirp sequence,
`Fmcw(sequence=(Chirp(start_offset=d),))`: the ghost moves by `-d` in beat
frequency and rotates by `-2 pi d T_c` from chirp to chirp.

## Pulsed and OFDM links

A pulsed or OFDM receiver samples what it hears on its own grid and with its
own LO, and its matched filter or demodulator then uses its own replica or
symbols. An OFDM radar heard by an identical one whose frame starts `dT`
later (inside the cyclic prefix) is seen as the CFR
`C exp(-j 2 pi n df (tau + dT))`, so the phase slope across subcarriers reads
the delay. A hearing pulsed or OFDM radar has one transmitter.

```python
ofdm = Ofdm(subcarrier_spacing=1e6, num_subcarriers=64, cyclic_prefix=0.5e-6, num_symbols=8,
            max_expected_delay=0.3e-6)
oa = dataclasses.replace(a, waveform=ofdm)
ob = dataclasses.replace(b, waveform=ofdm)
d_t = 0.1e-6
network = Network((oa, ob), links=((0, 1),), clock_offsets=(d_t, 0.0))
cfr = (network.simulate(scene, target, times=times, direct=True)[1].cube
       - network.simulate(scene, target, times=times, direct=False)[1].cube)[0, 0, 0, 0]  # [subcarrier]
slope = torch.angle(cfr[1:] * cfr[:-1].conj()).mean()
delay = -float(slope) / (2 * math.pi * ofdm.subcarrier_spacing)
print(f"delay from the phase slope: {delay * 1e9:.2f} ns; tau + dT = {(tau + d_t) * 1e9:.2f} ns")
assert abs(delay - (tau + d_t)) < 1e-9
```

`carrier_offsets` gives a pulsed or OFDM radar's oscillator offset from the
shared carrier (an OFDM carrier offset appears as inter-carrier interference).
It acts on what other radars hear from it; the radar's own rows keep the
nominal carrier.

## Emitters

`Emitter(position, samples, rate, power, carrier_offset=0.0, start=0.0,
repeat=True, polarization=(0, 1, 0))` is a non-radar transmitter: a complex
baseband `samples` (each magnitude at most 1) at `rate` samples per second,
full-scale `power` in dBm, an oscillator at the carrier plus `carrier_offset`
Hz from world time zero, its first sample leaving at `start` s, repeated or
sent once. It radiates isotropically from a fixed world position and every
radar hears it (through the targets, the line of sight with `direct=True`,
and the environment). A network with emitters needs isotropic radar elements.

A CW emitter is heard by an FMCW receiver only while the ramp sweeps across
it: a burst of `f_s / S` seconds where `|S v - d_E| <= f_s / 2`.

```python
slow = dataclasses.replace(b, waveform=dataclasses.replace(chirp, slope=5e12, sample_rate=10e6))
crossing = 30e-6  # the ramp passes the emitter's frequency 30 us into the ramp
cw = Emitter(position=(0.0, 0.0, 0.0), samples=(1.0,), rate=1e6, power=0.0, carrier_offset=5e12 * crossing)
heard = Network((slow,), emitters=(cw,)).simulate(scene, target, times=times)[0]
burst = (heard.cube - slow.simulate(scene, target, times=times).cube)[0, 0, 0, 0]
samples = int((burst.abs() > 0).sum())
print(f"burst of {samples} samples; f_s^2 / S = {10e6 ** 2 / 5e12:.0f}")
assert abs(samples - 20) <= 1
print("radars that hear another transmitter:", Network((slow,), emitters=(cw,)).hearing())
```

## Moving radars: `poses=`

Each radar may follow its own `pose(t) -> (rotation (3, 3), translation (3,))`,
the same contract as `Radar.simulate(pose=...)`. A radar's observations are
merged with the others' into one schedule, and each radar reads its own
instants. When anything moves, the default `Motion.auto()` resolves to the
exhaustive `Motion.adc()` schedule in a network (a radar alone would resolve
to `Motion.adaptive()`), so compare a network radar with a standalone one under
`Motion.adc()`.

```python
small = dataclasses.replace(chirp, samples_per_chirp=64, chirps_per_frame=4)
wa, wb = (dataclasses.replace(r, waveform=small) for r in (a, b))


def walk(t):
    return torch.eye(3), torch.tensor([0.0, 0.0, 0.5 * float(t)])  # 0.5 m/s along +z


pair = Network((wa, wb)).simulate(scene, target, times=(0.0,), poses=(walk, None))
solo = wa.simulate(scene, target, times=(0.0,), pose=walk, motion=Motion.adc())
print("schedule:", pair[0].motion_sampling, "| walking radar == alone under Motion.adc():",
      torch.equal(pair[0].cube, solo.cube))
assert torch.equal(pair[0].cube, solo.cube)
```

## What is refused

```python
attempts = {
    "two carriers": lambda: Network((a, b.replace(reference_frequency_hz=78e9))),
    "a radar hearing itself": lambda: Network((a, b), links=((1, 1),)),
    "FMCW heard by OFDM": lambda: Network((a, ob), links=((0, 1),)),
    "DDM on a link": lambda: Network((dataclasses.replace(a, waveform=dataclasses.replace(chirp, mimo="ddm"),
                                                          tx=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))), b),
                                     links=((0, 1),)),
    "adaptive motion": lambda: Network((a, b)).simulate(scene, target, times=times, motion=Motion.adaptive()),
    "a power leaf": lambda: Network((a.replace(power=torch.tensor(15.0, requires_grad=True)), b)),
}
for name, attempt in attempts.items():
    try:
        attempt()
    except (ValueError, NotImplementedError) as error:
        print(f"{name}: {type(error).__name__}: {str(error)[:100]}")
    else:
        print(f"{name}: accepted")
```

The full list:

* radars on different carriers or devices, or with different element patterns;
* an FMCW transmitter heard by a pulsed or OFDM receiver, and a pulsed or OFDM
  transmitter heard by an FMCW receiver (either needs the receiver's filter
  response to a foreign sweep);
* a pulsed or OFDM receiver hearing a band wider than its sample rate, or a
  carrier offset by more than half of it; a hearing pulsed or OFDM radar with
  several transmitters; a nonzero `carrier_offsets` entry on an FMCW radar
  (use `Chirp(start_offset=...)`);
* DDM or BPM MIMO, multi-profile chirp sequences and ramp phase errors on an
  FMCW link;
* emitters with directional radar elements, and moving emitters (there is no
  field for it);
* `Motion.adaptive()` and `Motion(rediscover_every_chirps=...)` in a network;
* oscillator phase noise on a radar that hears another transmitter, wideband
  OFDM, and `endpoints=`;
* `transmission=True` together with `direct`, `coupling` or `environment` rows;
* a transmit-power leaf on a radar of a network, and a `pose=` rotation leaf.

A transmitter is heard only in its same-index frame; its previous and next
frames are not. The IF band is an ideal gate on the instantaneous beat
frequency (no filter ripple). See [Gradients](gradients.md) for the leaves a
network carries: each radar's pose translation, and a shared pattern.
