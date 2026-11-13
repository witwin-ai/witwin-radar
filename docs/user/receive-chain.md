# The receive chain

Everything between the transmitter's power amplifier and the ADC codes is a
field of `Radar`. This guide walks the chain in signal order, one section per
field, each with a runnable example on the quickstart radar. Every stage is
off by default; a radar with none of them set is an ideal receiver whose cube
is in sqrt(W) at the receive port.

The order is fixed and physical. Nothing reorders it:

```text
transmit power -> transmit element pattern -> scene -> receive element pattern
  (leakage rows join here, synthesized like any other echo)
-> port (sqrt(W) to volts, `impedance`) -> coupling
-> oscillator phase noise, thermal noise, LNA -> compression -> IQ imbalance
-> baseband DC offset and flicker -> AGC -> ADC
```

Once any of `noise`, `lna_gain`, `agc`, `adc`, `iq`, `compression`,
`dc_offset` or `coupling` is set, the radar has a receive chain and its cube
is in volts across `Radar.impedance` (50 ohm by default): the port stage
multiplies by `sqrt(impedance)` once. The FMCW spectrum output is formed after
the chain, so the chain always acts on the de-chirped time samples.

## Setup

The radar, scene and target below are used by every later block. The helper
`peak_db` returns the strongest cell of the range-Doppler power summed over
the array, in dB.

```python
import dataclasses
import math

import torch
from witwin.core import Scene
from witwin.radar import (Adc, Agc, Compression, Coupling, Fmcw, Iq, Leakage, Noise, Ofdm, Pattern, PointTargets,
                          Radar)

chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=256, chirps_per_frame=64,
             adc_start=6e-6, idle=7e-6, ramp_end=65e-6)
radar = Radar(reference_frequency_hz=77e9, waveform=chirp, power=15.0, antenna_unit="half_wavelength",
              tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
scene = Scene()
# A static 1 m^2 target 3 m down the boresight (-z).
target = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0)


def power_map(r, targets=target, times=(0.0,)):
    """Range-Doppler power summed over the array, [Doppler, range]."""
    rd = r.simulate(scene, targets, times=times).frame(0).range_doppler()
    return rd.data.abs().square().sum(dim=(0, 1))


def peak_db(r, targets=target):
    return 10.0 * math.log10(float(power_map(r, targets).max()))


base_db = peak_db(radar)
print(f"ideal receiver peak: {base_db:.2f} dB(W)")
```

## Element patterns: `pattern`, `rx_pattern`, `gain`

`Radar.pattern` is the transmit element pattern and, unless `rx_pattern` is
given, the receive one too. A `Pattern` is a normalised table (power in
`[0, 1]`, or a complex field with `|F| <= 1`) over two off-boresight angles in
degrees, plus an absolute peak `gain` in dBi. The angles are, for a direction
`v` in the radar's local frame (right, up, -boresight),
`x = atan2(v_x, -v_z)` and `y = atan2(v_y, -v_z)`. Outside the tabulated
support the response is exactly zero.

`gain` is applied exactly once per side as the voltage factor
`10^(gain/20)`, so 10 dBi elements on both sides put a boresight target 20 dB
above the isotropic radar, and 10 dBi on transmit only puts it 10 dB above:

```python
both = peak_db(radar.replace(pattern=Pattern.isotropic(gain=10.0)))
tx_only = peak_db(radar.replace(pattern=Pattern.isotropic(gain=10.0), rx_pattern=Pattern.isotropic()))
print(f"10 dBi both sides: {both - base_db:+.2f} dB, transmit side only: {tx_only - base_db:+.2f} dB")
assert abs(both - base_db - 20.0) < 0.01 and abs(tx_only - base_db - 10.0) < 0.01
```

The constructors:

| Constructor | What it tabulates |
|---|---|
| `Pattern.isotropic(gain=0.0)` | unit power everywhere (the default) |
| `Pattern.dipole(gain=0.0)` | a half-wave dipole cut in both planes over -90..90 degrees |
| `Pattern.separable(x_angles, x_gain, y_angles, y_gain, gain=)` | power, the product of two cuts |
| `Pattern.table(x_angles, y_angles, values, gain=)` | power on the full grid, `values[y][x]` |
| `Pattern.field_separable(x_angles, x_field, y_angles, y_field, gain=)` | complex field, product of two cuts |
| `Pattern.field_table(x_angles, y_angles, field, gain=)` | complex field on the full grid |
| `Pattern.from_antenna(antenna, x_angles=, y_angles=)` | samples a `witwin.core.AntennaPattern` |

Interpolation is piecewise linear (in power, or in the field's real and
imaginary parts). A field pattern's phase is the element's phase response and
is applied on both sides of the round trip. `from_antenna` takes Core's custom
patterns, which is what Channel's `witwin.channel.presets.dipole_pattern()`
and `tr38901_pattern()` return; the sampled table is normalised by its
largest magnitude and that magnitude becomes `gain`.

A target 45 degrees off boresight shows the patterns' shape:

```python
from witwin.channel.presets import tr38901_pattern

off_axis = PointTargets(positions=[(3.0 * math.sin(math.radians(45)), 0.0, -3.0 * math.cos(math.radians(45)))],
                        rcs=1.0)
iso_off = peak_db(radar, off_axis)
cuts = [float(a) for a in range(-90, 91, 5)]
cosine = [math.cos(math.radians(a)) ** 2 for a in cuts]  # power cut, cos^2
patterns = {
    "dipole": Pattern.dipole(),
    "separable cos^2": Pattern.separable(cuts, cosine, cuts, cosine),
    "table cos^2 (x cut only)": Pattern.table(cuts, cuts, [cosine for _ in cuts]),
    "field cos (pi/4 phase)": Pattern.field_separable(
        cuts, [complex(math.cos(math.radians(a))) * complex(math.cos(math.pi / 4), math.sin(math.pi / 4)) for a in cuts],
        cuts, [1.0 + 0.0j for _ in cuts]),
    "TR 38.901 sector, 8 dBi": Pattern.from_antenna(tr38901_pattern(max_gain_dbi=8.0)),
}
for name, pattern in patterns.items():
    loss = peak_db(radar.replace(pattern=pattern), off_axis) - iso_off
    print(f"{name:26s} at 45 deg: {loss:+6.2f} dB (both sides)")
# cos^2 power at 45 degrees is 0.5, -3.01 dB per side.
assert abs(peak_db(radar.replace(pattern=patterns["separable cos^2"]), off_axis) - iso_off + 6.02) < 0.05
```

`Pattern.from_antenna` reads the element frame of Core's patterns (boresight
`+x`, zenith `+z`) and rotates it into the radar's local frame for you.

### Polarization: `polarization`, `rx_polarization`

`polarization` is the transmit elements' polarization: `"up"` (the default,
the radar's local up axis), `"right"`, or a world vector. `rx_polarization`
defaults to the same value; a different one measures a cross-polar channel.
A `PointTargets` with a scalar `rcs` is a co-polar scatterer, so a cross-polar
receiver hears nothing from it; a target that depolarises needs a
`ScatteringMatrix`, and walls and meshes depolarise through the scene's
materials.

```python
co = float(power_map(radar.replace(polarization="right")).max())
cross = float(power_map(radar.replace(rx_polarization="right")).max())
print(f"co-polar 'right': {10 * math.log10(co) - base_db:+.2f} dB, cross-polar power: {cross:.1e}")
assert cross == 0.0
```

## Transmit power: `power`, `power_w`

`Radar.power` is the transmit power in dBm per transmit element;
`Radar.power_w` is the same in watts. The received amplitude follows
`sqrt(P)`:

```python
print(f"{radar.power} dBm = {radar.power_w * 1e3:.2f} mW")
louder = peak_db(radar.replace(power=radar.power + 6.0))
print(f"+6 dB of power: {louder - base_db:+.2f} dB at the target")
assert abs(louder - base_db - 6.0) < 0.01
```

## Transmit leakage: `Leakage(isolation=, delay=, phase=)`

Part of every transmitted chirp reaches the receiver on the board. After
de-chirping it is a strong tone at `slope * delay`, a "target" a few
centimetres away, tens of dB above the scene. It spends the receiver's
dynamic range (the AGC and ADC see it) and its window skirt covers the near
range bins.

`isolation` is the transmit-to-receive isolation in dB (positive), so the
coupled amplitude is `sqrt(P_tx) * 10^(-isolation/20)`; `delay` is the
coupling path's delay in seconds; `phase` is the coupling coefficient's
argument in radians. Each may be one number or a `[num_tx][num_rx]` table.
Leakage is synthesized by the same waveform kernel as the echo, for FMCW,
OFDM and pulsed radars alike.

```python
leaky = radar.replace(leakage=Leakage(isolation=25.0, delay=0.3e-9))
profile = leaky.simulate(scene, target, times=(0.0,)).frame(0).range_profile()
power = profile.data.abs().square().sum(dim=(0, 1, 2))  # [range]
axes = profile.axes
near = int(power.argmax())
at_target = int((axes.range_m - 3.0).abs().argmin())
print(f"strongest range bin: {axes.range_m[near]:.2f} m, "
      f"{10 * math.log10(float(power[near]) / float(power[at_target])):.1f} dB above the target's bin at "
      f"{axes.range_m[at_target]:.2f} m")
assert axes.range_m[near] < 0.2
# A per-pair table: one transmitter, four receivers.
per_pair = Leakage(isolation=[[25.0, 30.0, 35.0, 40.0]], delay=[[0.3e-9] * 4], phase=[[0.0, 0.5, 1.0, 1.5]])
radar.replace(leakage=per_pair)
```

[Processing](processing.md) shows how to remove leakage and static clutter by
background subtraction.

## Mutual coupling: `Coupling(tx=, rx=)`

Port coupling matrices of the array. `rx[i][j]` is the complex response port
`i` reads from a unit field on element `j` (the identity is an uncoupled
array; an off-diagonal `0.1` is a -20 dB neighbour), and `tx[i][j]` the
excitation element `i` receives when port `j` is driven. The chain applies
`Y = C_tx^T H C_rx^T` over the cube's (TX, RX) axes before any active stage,
so the echoes are mixed and the thermal noise is not. Either side may be
`None`.

```python
c_rx = torch.eye(4, dtype=torch.complex64)
for i in range(3):
    c_rx[i, i + 1] = c_rx[i + 1, i] = 0.1
coupled = radar.replace(coupling=Coupling(rx=c_rx.tolist()))
plain = radar.replace(lna_gain=0.0)  # a unit chain, so both cubes are in volts
h = plain.simulate(scene, target, times=(0.0,)).cube[0]
y = coupled.simulate(scene, target, times=(0.0,)).cube[0]
expected = torch.einsum("tjsf,rj->trsf", h, c_rx.to(h.device))
print(f"coupled cube vs C_rx applied by hand: max error {float((y - expected).abs().max() / h.abs().max()):.1e}")
assert torch.allclose(y, expected, rtol=1e-4, atol=1e-6 * float(h.abs().max()))
```

## Noise: `Noise(...)`

`Noise` holds the thermal noise, the oscillator phase noise and the baseband
flicker noise.

**Thermal noise.** `figure` is the noise figure in dB and
`antenna_temperature` the antenna temperature in K (290 by default), so
`T_sys = T_ant + 290 (F - 1)`. Every waveform publishes samples in sqrt(W),
and the thermal stage adds circular complex Gaussian noise of variance
`k T_sys f_s R` per complex sample: the noise bandwidth is the waveform's
complex sample rate `f_s` (`waveform.sample_rate`), for FMCW, OFDM and pulsed
radars alike. `bandwidth=` overrides it. On the beat samples:

```python
beat = radar.replace(waveform=dataclasses.replace(chirp, output="beat"))
clean = beat.replace(lna_gain=0.0)  # unit chain: volts, no noise
noisy = beat.replace(noise=Noise(figure=8.0))
n = noisy.simulate(scene, target, times=(0.0,)).cube - clean.simulate(scene, target, times=(0.0,)).cube
t_sys = 290.0 + 290.0 * (10 ** 0.8 - 1.0)
expected = 1.380649e-23 * t_sys * chirp.sample_rate * radar.impedance
measured = float(n.abs().square().mean())
print(f"noise per complex sample: {measured:.3e} V^2, k T_sys f_s R = {expected:.3e} V^2")
assert abs(measured / expected - 1.0) < 0.03
```

**Realisations.** The thermal draws are keyed by the radar's `seed` and the
frame instant, so every frame of a run has its own realisation, the same frame
instant always gives the same one (in `simulate`, `stream` and `echo`), and
`radar.replace(seed=...)` draws another:

```python
quiet = beat.replace(noise=Noise(figure=8.0))
a = quiet.simulate(scene, target, times=(0.0, 0.01)).cube
b = quiet.simulate(scene, target, times=(0.0, 0.01)).cube
c = quiet.replace(seed=7).simulate(scene, target, times=(0.0, 0.01)).cube
print("frames differ:", not torch.equal(a[0], a[1]), "| reproducible:", torch.equal(a, b),
      "| other seed differs:", not torch.equal(a, c))
assert not torch.equal(a[0], a[1]) and torch.equal(a, b) and not torch.equal(a, c)
```

**Oscillator phase noise.** One common oscillator is evaluated at the receive
and the delayed transmit instants of every path, so the range correlation of
phase noise (`phi(t) - phi(t - tau)`) is modelled. Three single-sideband
regions can be declared, each by its level in dBc/Hz at its own offset in Hz;
their sum is the mask:

* `phase_flicker_density` at `phase_flicker_offset`: the `1/f^3` (flicker FM)
  region;
* `phase_density` at `phase_offset`: the `1/f^2` (white FM, free-running)
  region;
* `phase_floor`: the white floor, held over cells of `1/phase_sample_rate`
  (the waveform's sample rate by default).

**Baseband flicker.** `flicker_corner` (Hz) adds the mixer and IF amplifier
`1/f` noise after the quadrature split: its density equals the thermal density
at the corner and rises as `1/f` below it. It needs the thermal stage.

The oscillator and the flicker are processes over absolute time and keep the
base seed; only the white thermal draws are re-keyed per frame.

```python
shaped = beat.replace(noise=Noise(figure=8.0, phase_flicker_density=-60.0, phase_flicker_offset=10e3,
                                  phase_density=-90.0, phase_offset=1e6, phase_floor=-150.0,
                                  flicker_corner=100e3))
cube = shaped.simulate(scene, target, times=(0.0,)).cube
print("shaped phase noise and flicker:", tuple(cube.shape), cube.dtype)
```

## LNA gain: `lna_gain`

`lna_gain` is the LNA voltage gain in dB. Thermal noise is input-referred, so
the gain scales signal and noise together:

```python
amplified = peak_db(radar.replace(lna_gain=20.0)) - peak_db(radar.replace(lna_gain=0.0))
print(f"lna_gain=20: {amplified:+.2f} dB")
assert abs(amplified - 20.0) < 0.01
```

## Compression: `Compression(iip3=, p1db=, am_pm=)`

A memoryless gain compression after the LNA, referred to the receiver input
like a datasheet: `iip3` and `p1db` in dBm, `am_pm` the phase rotation at the
1-dB point in degrees. With `u = P_in / P_IIP3` the complex gain is
`(1 - u + c5 u^2) exp(j k u)`, held at its first output maximum (a soft
limiter). Without `p1db` the 1-dB point is 9.64 dB below `iip3`. The input
power is the complex envelope's `|x|^2 / R` at the port.

The quickstart target arrives at about -88 dBm per sample, so a deliberately
weak receiver shows the effect:

```python
port = beat.replace(lna_gain=0.0).simulate(scene, target, times=(0.0,)).cube
p_in_dbm = 10 * math.log10(float(port.abs().square().max()) / radar.impedance * 1e3)
weak = radar.replace(compression=Compression(iip3=p_in_dbm + 9.64, am_pm=5.0))
loss = peak_db(weak) - peak_db(radar.replace(lna_gain=0.0))
print(f"input {p_in_dbm:.1f} dBm at the 1-dB point: peak {loss:+.2f} dB")
assert -1.3 < loss < -0.7
```

## IQ imbalance: `Iq(gain=, phase=)`

A real quadrature split has a gain ratio `g` (`gain` = `20 log10 g`, dB) and a
phase error `phi` (`phase`, radians) between its branches, which leaks a
conjugate copy: `y = alpha x + beta conj(x)`. The conjugate mirrors the Doppler
axis (a target closing at `+v` also appears receding at `-v`) and, for complex
FMCW sampling, the range axis, at `Iq.image_rejection_db` below the true
return:

```python
moving = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0,
                      trajectory=lambda t: torch.tensor([[0.0, 0.0, -3.0 + t]], device="cuda"))
iq = Iq(gain=0.5, phase=math.radians(3.0))
power = power_map(radar.replace(iq=iq), moving)
true_peak = float(power[:, : power.shape[1] // 2].max())  # positive beat frequencies
image_peak = float(power[:, power.shape[1] // 2:].max())  # the mirrored half
measured = 10 * math.log10(true_peak / image_peak)
print(f"image {measured:.2f} dB below the target; Iq.image_rejection_db = {iq.image_rejection_db:.2f} dB")
assert abs(measured - iq.image_rejection_db) < 0.1
```

## Baseband DC offset: `dc_offset`

`dc_offset` is added after the quadrature split, in volts: one complex value
for every receiver, or one per receiver. It appears in the zero-range bin.

```python
dc = beat.replace(dc_offset=[1e-6, 2e-6, 0.0, 1e-6j])
mean = dc.simulate(scene, PointTargets(positions=[(0.0, 0.0, -300.0)], rcs=1e-6), times=(0.0,)).cube.mean(dim=(0, 1, 3, 4))
print("per-receiver mean (V):", [complex(round(v.real, 9), round(v.imag, 9)) for v in mean.tolist()])
assert torch.allclose(mean.cpu(), torch.tensor([1e-6, 2e-6, 0.0, 1e-6j], dtype=torch.complex64), atol=1e-8)
```

## AGC: `Agc(target_rms=, mode=, min_gain=, max_gain=, time_constant=)`

The AGC scales the signal towards `target_rms` volts, per receiver
(`mode="per_rx"`, the default) or with one gain for the whole array
(`mode="global"`), within `[min_gain, max_gain]` dB.

* `time_constant=None` (the default) is a frame-static AGC: one gain per group,
  measured over the whole frame.
* A positive `time_constant` (s) is a causal first-order loop in log gain with
  one gain per slow-time block (a chirp slot, an OFDM symbol, a pulse):
  `l_k = clamp((1 - beta) l_{k-1} + beta ln(target / r_{k-1}))`,
  `beta = 1 - exp(-T_block / time_constant)`, the first block acquired at its
  own ideal gain. The loop state does not carry across frames.

The AGC depends on the signal's own level, so the chain is not linear with an
AGC on.

```python
agc_radar = beat.replace(noise=Noise(figure=8.0), lna_gain=30.0, agc=Agc(target_rms=0.05))
cube = agc_radar.simulate(scene, target, times=(0.0,)).cube[0, 0]  # [rx, slow, fast]
rms = cube.abs().square().mean(dim=(1, 2)).sqrt()
print("per-receiver RMS (V):", [round(v, 4) for v in rms.tolist()])
assert torch.allclose(rms, torch.full_like(rms, 0.05), rtol=1e-3)
looped = beat.replace(noise=Noise(figure=8.0), lna_gain=30.0, agc=Agc(target_rms=0.05, time_constant=200e-6))
print("dynamic AGC cube:", tuple(looped.simulate(scene, target, times=(0.0,)).cube.shape))
```

## ADC: `Adc(bits=, full_scale=)`

The only quantiser in the chain, per component, on a two's-complement
mid-tread grid: `step = 2 FS / 2^bits` (`Adc.step`, volts), codes
`-2^(bits-1) ... 2^(bits-1) - 1`, levels `code * step`. Zero is a code (an
idle channel reads exactly zero), the levels span `[-FS, FS - step]`, and an
input outside the no-overload range `[-FS - step/2, FS - step/2)` saturates at
the end code. Rounding is half-to-even. `bits` lies in `[2, 30]`.

A simulated result carries the codes as volts; it does not publish a clipped
count, so count the components sitting on the two end levels. The AGC above
holds the target's tone at about 0.05 V, so a 0.1 V full scale has headroom
and a 0.04 V one clips:

```python
for full_scale in (0.1, 0.04):
    adc = Adc(bits=10, full_scale=full_scale)
    digitised = agc_radar.replace(adc=adc).simulate(scene, target, times=(0.0,)).cube
    codes = torch.view_as_real(digitised) / adc.step
    on_grid = torch.allclose(codes, codes.round(), atol=1e-3)
    clipped = int(((codes <= -2 ** (adc.bits - 1)) | (codes >= 2 ** (adc.bits - 1) - 1)).sum())
    print(f"full scale {full_scale} V: step {adc.step * 1e3:.4f} mV, on the grid: {on_grid}, "
          f"components at an end code: {clipped} of {codes.numel()}")
    assert on_grid and (clipped == 0) == (full_scale == 0.1)
```

## What is refused

Each of these raises before any simulation work:

```python
attempts = {
    "one-bit ADC": lambda: Adc(bits=1, full_scale=1.0),
    "p1db above iip3": lambda: Compression(iip3=-20.0, p1db=-10.0),
    "coupling of the wrong size": lambda: radar.replace(coupling=Coupling(rx=torch.eye(3, dtype=torch.complex64).tolist())),
    "a tensor noise figure": lambda: Noise(figure=torch.tensor(8.0, requires_grad=True)),
    "DC offset on an OFDM radar": lambda: radar.replace(
        waveform=Ofdm(subcarrier_spacing=1e6, num_subcarriers=64, cyclic_prefix=0.2e-6, num_symbols=8,
                      max_expected_delay=0.1e-6),
        dc_offset=1e-6),
}
for name, attempt in attempts.items():
    try:
        attempt()
    except (TypeError, ValueError, NotImplementedError) as error:
        print(f"{name}: {type(error).__name__}: {str(error)[:110]}")
    else:
        print(f"{name}: accepted")
```

* Receive-chain configuration (`Noise`, `Agc`, `Adc`, `Compression`,
  `impedance`, `seed`, `dc_offset`) takes host numbers, not tensors that carry
  a derivative. `lna_gain`, `power`, the pattern tables and gains, and the
  `Coupling` matrices are the differentiable ones; see [Gradients](gradients.md).
* The ADC refuses a signal that carries a gradient or a forward tangent:
  rounding has no derivative, and no straight-through surrogate is applied for
  you.
* A DC offset or baseband flicker on an OFDM radar is refused
  (`NotImplementedError`): the OFDM cube is a frequency-domain CFR.
* `Radar.echo` refuses a phase-noise receiver over paths that were traced
  without the ADC instants the oscillator needs.
* In a [network](networks.md), oscillator phase noise on a receiving radar is
  refused.
* Interference from other radars is not a receive-chain stage; it is
  synthesized by a [`Network`](networks.md).
