# frontend

## Breaking changes

- Pulsed samples are physical amplitudes: the pulse envelope is unit amplitude (`PulsedSpec.pulse_amplitude == 1`, `pulse_normalization="unit_amplitude"`), so an in-pulse sample is `C_rt` in sqrt(W) and the pulsed cube is `sqrt(T_p)` larger than before. Matched-filter peaks are unchanged (the replica carries `PulsedSpec.replica_amplitude = 1 / T_p`). Migration: pass `pulse_normalization="unit_amplitude"` or omit it; code that divided the cube by `sqrt(T_p)` must stop.
- The default thermal noise bandwidth is the waveform's complex sample rate for every waveform; a pulsed radar no longer integrates over its LFM sweep. `Fmcw.bandwidth` is removed (it was the sample rate); use `Fmcw.sample_rate`.
- The ADC is a two's-complement mid-tread grid: `Adc.step == 2 FS / 2^b`, zero is a code, outputs span `[-FS, FS - step]`, and the clipped count is the saturated-code count. `Adc(bits=1)` is refused.
- `Pattern` tables are normalised and the bound is enforced (power at most 1, `|field|` at most 1); the peak gain moves to `Pattern.gain` (dBi). The map table field is renamed `gain_table`.
- Every frame of a noisy run draws its own thermal realisation, keyed by the frame instant; a frame at `t = 0` is unchanged. Multi-frame noisy results change.

## Added

- `Pattern.gain` (absolute element gain, dBi, applied once per side), `Radar.rx_pattern` (separate receive element), complex field patterns (`Pattern.field_separable`, `Pattern.field_table`) with amplitude and phase, and `Pattern.from_antenna` to sample a `witwin.core.AntennaPattern` custom callable (Channel's TR 38.901 and dipole presets) into a field table.
- `Radar.compression = Compression(iip3, p1db, am_pm)`: receiver gain compression after the LNA, input-referred, with AM/PM and a saturating hold; differentiable in the signal.
- Shaped oscillator phase noise: `Noise.phase_flicker_density`/`phase_flicker_offset` (1/f^3) and `Noise.phase_floor` (white floor) alongside the existing 1/f^2 region, one oscillator over absolute time with the range-correlation effect.
- `Radar.coupling = Coupling(tx=..., rx=...)`: mutual coupling matrices on the transmit and receive ports, applied before any active stage.
- `Leakage` accepts per-pair `[num_tx][num_rx]` tables for `isolation_db`, `delay_s` and a new `phase`, and works for OFDM and pulsed radars as well as FMCW.
- `Radar.dc_offset` (per-receiver complex baseband offset) and `Noise.flicker_corner` (baseband 1/f noise with the thermal density at the corner), for FMCW and pulsed.
- `Agc(time_constant=...)`: a time-dynamic AGC with one gain per slow-time block from a first-order loop; the gains are published per (group, block).
- `Radar(iq=...)` alone now builds a receive chain; before, an IQ imbalance without another stage was silently ignored.

## Fixed

- Pulsed SNR was optimistic by `f_s / (B T_p)` (+54 dB at 20 MHz, 10 us, 50 MS/s); all three waveforms now match the radar equation through `k T_sys f_s` and the coherent processing gain within 0.5 dB.
- All frames of a `simulate`/`stream`/`echo` call shared one thermal noise realisation.
- A zero ADC input read `+/- step/2` on I and Q.
- An invalid oscillator query (non-finite time or delay, negative delay) raised a device-side assert that left the CUDA context unusable; it now raises `ValueError` on the host and the context stays usable.

## Performance

- The isotropic 0 dBi default skips the antenna-pattern launch (the stage is the identity).
- Each new receiver stage is one native operator call per frame (compression one launch, baseband one launch, dynamic AGC three launches).
