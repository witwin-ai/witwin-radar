# Waveforms

A `Radar` carries one waveform record: `Fmcw`, `Ofdm` or `Pulsed`. Every
waveform is synthesized natively from the same composed round trips, so a
target has one range, one radial velocity (positive closing) and one amplitude
whichever waveform looks at it; `Result.axes` carries each waveform's own
range and velocity axes. Each record validates its fields at construction, and
the rules that need the array or the carrier run when the `Radar` is built.

| | FMCW | OFDM | Pulsed |
|---|---|---|---|
| Cube `[slow, fast]` axes | `chirp`, `range_bin` (or `sample` for `output="beat"`) | `symbol`, `subcarrier` (or `sample` for `output="time"`) | `pulse`, `sample` |
| Cube phasor | `conj(exp(-j*k*d))` (de-chirped) | `exp(-j*k*d)` | `exp(-j*k*d)` |
| MIMO | TDM, DDM, BPM | TDM (ideally orthogonal) | TDM (ideally orthogonal) |
| `Motion.adaptive()` | uniform TDM ramp only | refused | refused |
| `Motion.adc()` observes | every ADC sample | every symbol (CFR) or sample (time) | every gate sample |
| `wideband=True` | refused | CFR output only | refused |
| Oscillator phase noise in a scene | yes | refused | refused |

## Setup

```python
import math

import torch
from witwin.core import Scene
from witwin.radar import Chirp, Fmcw, Motion, Ofdm, PointTargets, Pulsed, Radar, processing


def make_radar(waveform, tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0),), **fields):
    return Radar(reference_frequency_hz=77e9, waveform=waveform, power=15.0, antenna_unit="half_wavelength",
                 tx=tx, rx=rx, **fields)


still = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0)  # 3 m down the boresight


def range_peak(result):
    """Range, m, of the strongest range bin of frame 0 (power summed over everything but range)."""
    profile = result.frame(0).range_profile().data.abs().square()
    profile = profile.reshape(-1, profile.shape[-1]).sum(dim=0)
    return float(result.axes.range_m[int(profile.argmax())])
```

## FMCW

`Fmcw(slope, sample_rate, samples_per_chirp, chirps_per_frame, adc_start,
idle, ramp_end, ...)`, all SI: slope in Hz/s, sample rate in Hz, times in
seconds. The chirp period is `idle + ramp_end`, the ADC window must end
before it, and `chirps_per_frame` counts chirps per transmitter.

```python
chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=64, chirps_per_frame=32,
             adc_start=6e-6, idle=7e-6, ramp_end=20e-6)
print(f"sweep {chirp.sweep_bandwidth / 1e6:.0f} MHz, range resolution {chirp.range_resolution * 100:.1f} cm, "
      f"max range {chirp.max_range:.2f} m, chirp period {chirp.chirp_period * 1e6:.0f} us, "
      f"max velocity {chirp.max_velocity(reference_frequency_hz=77e9, num_tx=1):.1f} m/s")
fmcw = make_radar(chirp)
result = fmcw.simulate(Scene(), still, times=(0.0,))
print(result.axis_names, result.output_domain, result.phasor)
assert abs(range_peak(result) - 3.0) <= chirp.range_resolution
```

- `sweep_bandwidth` is `|slope| * N / f_s`, the band the ADC window sweeps
  (narrower than the whole ramp); `range_resolution` is `c / (2 B)`;
  `max_range` is `f_s c / (2 |slope|)`, the unambiguous range of a complex
  (IQ) receiver; `max_velocity(reference_frequency_hz=, num_tx=)` is
  `lambda / (4 T_slot)` with `T_slot = num_tx` chirp periods under TDM.
- `Fmcw.from_ti(slope_mhz_per_us=, sample_rate_ksps=, samples_per_chirp=,
  chirps_per_frame=, adc_start_us=, idle_us=, ramp_end_us=)` reads the vendor
  units of a TI configuration; `Radar.from_dict` / `Radar.from_json` read the
  whole flat configuration format, and `witwin.radar.config` imports a TI
  mmWave `.cfg`.

```python
ti = Fmcw.from_ti(slope_mhz_per_us=60.0, sample_rate_ksps=4400, samples_per_chirp=64, chirps_per_frame=32,
                  adc_start_us=6, idle_us=7, ramp_end_us=20)
assert all(math.isclose(getattr(ti, name), getattr(chirp, name))
           for name in ("slope", "sample_rate", "adc_start", "idle", "ramp_end"))
```

### Output domain: spectrum or beat

`output="spectrum"` (default) publishes the normalized range spectrum: the
fast axis is already range, so `range_profile()` applies no second transform
and refuses a fast-time window. `output="beat"` publishes the de-chirped
time samples (`sample` axis); `range_profile()` transforms them and accepts a
`window=`. The same target lands in the same range bin either way.

```python
beat = make_radar(Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=64, chirps_per_frame=32,
                       adc_start=6e-6, idle=7e-6, ramp_end=20e-6, output="beat"))
beat_result = beat.simulate(Scene(), still, times=(0.0,))
print(f"{beat_result.axis_names[-1]} axis, {beat_result.output_domain} domain, peak {range_peak(beat_result):.2f} m")
assert range_peak(beat_result) == range_peak(result)
hann = beat_result.frame(0).range_profile(window="hann")
```

### Chirp sequences, ramp phase error, real sampling

- `sequence=(Chirp(...), ...)` is a TI-style chirp configuration: one profile
  per slot of a loop, repeated across the frame. A `Chirp` field left `None`
  takes the ramp's own value; `start_offset` (Hz) shifts the start frequency.
  Every profile keeps the sample rate and count. `Result.axes` publishes the
  base slope's range axis, so process one profile's slots with its own slope.
- `phase_error=(c2, c3, ...)` adds `phi(t) = sum_k c_k t^k` (rad) to the
  transmitted ramp's phase, `t` from the ramp start: ramp nonlinearity.
- `sampling="real"` digitises the in-phase channel only. The cube is exactly
  the real part of the IQ cube, so its spectrum is Hermitian: every target
  also appears at its mirror bin, and the usable range is half of `max_range`.

```python
two_profiles = make_radar(Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=64, chirps_per_frame=32,
                               adc_start=6e-6, idle=7e-6, ramp_end=20e-6,
                               sequence=(Chirp(), Chirp(slope=30e12))))
profile = two_profiles.simulate(Scene(), still, times=(0.0,)).frame(0).range_profile().data.abs()[0, 0]
fast_bin, slow_bin = int(profile[0].argmax()), int(profile[1].argmax())
print(f"profile 0 peaks in bin {fast_bin}, profile 1 (half the slope) in bin {slow_bin}")
assert abs(fast_bin - 2 * slow_bin) <= 1

real = make_radar(Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=64, chirps_per_frame=32,
                       adc_start=6e-6, idle=7e-6, ramp_end=20e-6, sampling="real"))
line = real.simulate(Scene(), still, times=(0.0,)).frame(0).range_profile().data.abs()[0, 0, 0]
print("real sampling, two strongest bins:", sorted(torch.topk(line, 2).indices.tolist()), "of", line.numel())
```

A chirp sequence, a start offset, a phase error, DDM and BPM all run with
`Motion.adc()` or `Motion.chirp()`; `Motion.adaptive()` refuses them and
`Motion.auto()` resolves to `adc` for them ([Motion](motion.md)).

## MIMO: TDM, DDM and BPM

With several transmitters the default `mimo="tdm"` fires them in turn: the
cube has one `tx` entry per transmitter and each transmitter's chirps are
`num_tx` chirp periods apart. `mimo="ddm"` (Doppler-division, with
`ddm_bands` sub-bands, at least the transmitter count; 0 means the count)
and `mimo="bpm"` (binary phase coding, a power-of-two transmitter count) fire
every transmitter on every chirp. Their physical cube has one transmit
channel; `Frame.virtual_cube()` (`processing.demultiplex`) separates it into
`[tx, rx, slow, fast]`, decimating slow time by the code period
(`axes.code_period`), so the demultiplexed unambiguous velocity is the
physical cube's divided by that period. `Frame.points()`,
`Frame.range_angle(...)` and `Frame.image(...)` demultiplex automatically;
`processing.point_cloud`, `beam_cube` and `range_angle_map` called directly on
a DDM/BPM product that was not demultiplexed are refused. DDM and BPM are
FMCW only, and array mutual coupling (`Radar.coupling`) is refused with them.

```python
mimo_chirp = dict(slope=60e12, sample_rate=4.4e6, samples_per_chirp=64, chirps_per_frame=32,
                  adc_start=6e-6, idle=7e-6, ramp_end=20e-6)
array = dict(tx=((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)), rx=tuple((float(i), 0.0, 0.0) for i in range(4)))
off_axis = PointTargets(positions=[(3.0 * math.sin(math.radians(20)), 0.0, -3.0 * math.cos(math.radians(20)))],
                        rcs=1.0)
azimuth = torch.linspace(-math.pi / 3, math.pi / 3, 241, dtype=torch.float64)
for scheme in ("tdm", "ddm", "bpm"):
    frame = make_radar(Fmcw(**mimo_chirp, mimo=scheme), **array).simulate(Scene(), off_axis, times=(0.0,)).frame(0)
    virtual = frame.virtual_cube()
    angle_map = frame.range_angle(azimuth).data.abs()
    best = math.degrees(float(azimuth[int(angle_map.amax(dim=1).argmax())]))
    print(f"{scheme}: physical cube {tuple(frame.cube.shape)}, virtual {tuple(virtual.data.shape)}, "
          f"max speed {frame.axes.max_unambiguous_speed_mps:.1f} -> {virtual.axes.max_unambiguous_speed_mps:.1f} m/s, "
          f"beam peak at {best:+.1f} deg")
    assert abs(best - 20.0) < 3.0
```

`processing.tdm_compensate` corrects the TDM motion phase between
transmitters before angle estimation; it is the identity for DDM and BPM.

## OFDM

`Ofdm(subcarrier_spacing, num_subcarriers, cyclic_prefix, num_symbols,
max_expected_delay, modulation="ones", pilot_spacing=0, symbol_seed=0,
output="cfr")`. The occupied band is `num_subcarriers * subcarrier_spacing`,
which sets the range resolution `c / (2 B)`; the unambiguous range is
`c / (2 df)`; the cyclic prefix must cover `max_expected_delay`.

- `output="cfr"` publishes the per-subcarrier channel estimate `Y / X`
  (`symbol`, `subcarrier` axes); `range_profile()` inverts it.
- `output="time"` publishes the received post-prefix samples (`symbol`,
  `sample` axes). `range_profile()` refuses such a cube;
  `processing.ofdm_demodulate(frame.processing_cube())` turns it into the CFR
  estimate first.
- `modulation` is `"ones"` (the unmodulated comb), `"bpsk"`, `"qpsk"`,
  `"qam16"` or `"qam64"`, drawn from `symbol_seed`; every `pilot_spacing`-th
  subcarrier carries the pilot `1`. A modulated stream is synthesized as
  received time samples, so motion inside a symbol produces inter-carrier
  interference; the unmodulated CFR is the closed-form ideal.
- `wideband=True` evaluates the transport per subcarrier, CFR output only
  ([Multipath](multipath.md#wideband-ofdm)).

```python
ofdm = Ofdm(subcarrier_spacing=4e6, num_subcarriers=128, cyclic_prefix=0.1e-6, num_symbols=32,
            max_expected_delay=0.05e-6)
cfr = make_radar(ofdm).simulate(Scene(), still, times=(0.0,))
print(cfr.axis_names, f"range bin {cfr.axes.range_bin_m:.3f} m, peak {range_peak(cfr):.2f} m")
assert abs(range_peak(cfr) - 3.0) <= cfr.axes.range_bin_m

qpsk = Ofdm(subcarrier_spacing=4e6, num_subcarriers=128, cyclic_prefix=0.1e-6, num_symbols=32,
            max_expected_delay=0.05e-6, modulation="qpsk", pilot_spacing=8, output="time")
samples = make_radar(qpsk).simulate(Scene(), still, times=(0.0,))
demodulated = processing.ofdm_demodulate(samples.frame(0).processing_cube())
line = processing.range_profile(demodulated).data.abs()[0, 0, 0]
print(samples.axis_names, f"demodulated peak {float(demodulated.axes.range_m[int(line.argmax())]):.2f} m")
assert abs(float(demodulated.axes.range_m[int(line.argmax())]) - range_peak(cfr)) < 1e-9
```

OFDM refuses `Motion.adaptive()`, oscillator phase noise in a scene, the
receive chain's DC offset and flicker stages, and a declared `Leakage` on a
moving time-domain frame. Under `Motion.adc()` a CFR radar is observed once
per symbol (it has no fast-time grid of its own) and a time-domain one at every
sample. Velocity resolution needs many symbols: this example's 32 symbols of
0.35 us resolve only `axes.velocity_bin_mps` = 174 m/s.

## Pulsed

`Pulsed(pulse_kind, pulse_width, bandwidth, pri, num_pulses, sample_rate,
num_samples, range_gate_start, max_expected_delay_rate=0.0, ...)`: a pulse
train mixed down with a fixed local oscillator and sampled in a range gate
that opens `range_gate_start` after each pulse starts. `pulse_kind` is
`"rect"`, `"lfm"` (a linear chirp of `bandwidth` across the pulse) or
`"coded"`. `range_profile()` is the matched filter; the range bin is one
sample, `c / (2 f_s)`.

```python
lfm = Pulsed(pulse_kind="lfm", pulse_width=0.2e-6, bandwidth=300e6, pri=2e-6, num_pulses=32,
             sample_rate=600e6, num_samples=256, range_gate_start=0.0)
pulsed = make_radar(lfm)
echo = pulsed.simulate(Scene(), still, times=(0.0,))
print(echo.axis_names, f"range bin {echo.axes.range_bin_m:.3f} m, peak {range_peak(echo):.2f} m")
assert abs(range_peak(echo) - 3.0) <= echo.axes.range_bin_m
```

The coded and stepped members have their own constructors:

- `Pulsed.pmcw(code, chip_rate=, repetitions=, sample_rate=)`: a binary or
  polyphase code transmitted continuously, one code period per slow-time
  sample, matched by cyclic correlation; one period must be a whole number of
  samples. `Pulsed.m_sequence(order)` (orders 2-16) generates a +1/-1
  maximal-length sequence.
- `Pulsed.fsk(frequencies, chip_width=, pri=, num_pulses=, sample_rate=,
  num_samples=)`: a frequency-hopped pulse, one tone per chip.
  `Pulsed.costas(prime)` gives a Welch-Costas hop order; multiply it by a tone
  spacing of `1 / chip_width`.
- `Pulsed.stepped_frequency(step=, num_steps=, pulse_width=, pri=,
  sample_rate=, num_samples=)`: rectangular pulses whose carrier steps by
  `step` Hz from pulse to pulse. `processing.stepped_frequency_profile(signal,
  spec)` inverts across the pulses: resolution `c / (2 num_steps step)`,
  unambiguous range `c / (2 step)`.
- `Pulsed.arbitrary(samples, rate=, pri=, num_pulses=, sample_rate=,
  num_samples=)`: a user-sampled complex envelope (magnitudes at most one),
  each sample held for `1 / rate` s.

Any +/-1 code can be passed as `code` or `chips`; only m-sequences and Costas
orders are generated.

```python
coded = {
    "pmcw": Pulsed.pmcw(Pulsed.m_sequence(7), chip_rate=500e6, repetitions=32, sample_rate=500e6),
    "fsk": Pulsed.fsk([25e6 * k for k in Pulsed.costas(7)], chip_width=40e-9, pri=2e-6, num_pulses=16,
                      sample_rate=600e6, num_samples=256),
    "arbitrary": Pulsed.arbitrary([1.0, 1j, -1.0, -1j] * 8, rate=200e6, pri=2e-6, num_pulses=16,
                                  sample_rate=400e6, num_samples=256),
}
for name, waveform in coded.items():
    result = make_radar(waveform).simulate(Scene(), still, times=(0.0,))
    print(f"{name:9s} cube {tuple(result.cube.shape)}, range bin {result.axes.range_bin_m:.3f} m, "
          f"peak {range_peak(result):.2f} m")
    assert abs(range_peak(result) - 3.0) <= 1.5 * result.axes.range_bin_m

stepped = make_radar(Pulsed.stepped_frequency(step=10e6, num_steps=64, pulse_width=0.1e-6, pri=1e-6,
                                              sample_rate=100e6, num_samples=32))
train = stepped.simulate(Scene(), still, times=(0.0,)).cube[0, 0, 0]  # [pulse, sample]
synthetic, delay_s = processing.stepped_frequency_profile(train, stepped.waveform_spec())
gate = int(train.abs().sum(dim=0).argmax())  # a fast-time sample inside the echo
bins = synthetic[:, gate].abs()
print(f"stepped frequency: synthetic range {float(delay_s[int(bins.argmax())]) * 299792458.0 / 2:.2f} m")
```

A coded receiver samples the chip that has arrived at each instant, so the
peak lands on the first sample at or after the true delay and the range can
read up to one bin long.

Under `Motion.adc()` (and `auto` for a moving scene) a pulsed radar observes
the world at every gate sample, which is exhaustive but costs one world
evaluation per sample; `Motion.chirp()` holds the world per pulse.
`max_expected_delay_rate` is the largest `d(tau)/dt` the matched filter must
tolerate. Pulsed radars refuse `Motion.adaptive()`, oscillator phase noise in
a scene, and a declared `Leakage` on a moving (refreshed) frame.

```python
mover = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0,
                     trajectory=lambda t: torch.tensor([[0.0, 0.0, -3.0 + 5.0 * t]], device="cuda"))
per_pulse = pulsed.simulate(Scene(), mover, times=(0.0,), motion=Motion.chirp())
print(per_pulse.motion_sampling, per_pulse.observation_schedule, len(per_pulse.sample_times[0]), "observations")
try:
    pulsed.simulate(Scene(), mover, times=(0.0,), motion=Motion.adaptive())
except NotImplementedError as error:
    print("refused:", error)
```

See also: [Motion](motion.md), [Multipath](multipath.md),
[Conventions](conventions.md), [Receive chain](receive-chain.md).
