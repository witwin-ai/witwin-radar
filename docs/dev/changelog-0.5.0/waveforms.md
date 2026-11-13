# waveforms

## Breaking changes

- A pulsed train on a moving scene under `Motion.adc()` (and the `Motion.auto()` default that resolves to it) now observes the world at every range-gate sample instead of once per pulse, so it carries intra-pulse Doppler and LFM range-Doppler coupling and costs one world observation per gate sample, like FMCW. `Motion.chirp()` keeps the per-pulse schedule.
- The native `pulsed_echo_*` operators take a chip table, a per-segment schedule, `frequency_step_hz` and `periodic`; the `fmcw_beat_*` operators take `slot_table`, `segment_table` and `chirp_poly` in place of `segment_t_start`. Only code calling the native operators directly is affected.
- `PulsedSpec` refuses an unknown `pulse_kind` with a message pointing at the new `coded` kind (it used to say only analytic kinds exist).

## Added

- Coded pulses, `Pulsed(pulse_kind="coded", chips=..., chip_frequencies=...)`, with constructors `Pulsed.pmcw` (a phase code transmitted continuously, compressed by a cyclic matched filter), `Pulsed.fsk` (frequency-hopped chips), `Pulsed.arbitrary` (a user-sampled complex envelope held per sample) and `Pulsed.stepped_frequency` (`frequency_step`: the carrier steps from pulse to pulse); code generators `Pulsed.m_sequence(order)` and `Pulsed.costas(prime)`; processing `stepped_frequency_profile`.
- Time-domain OFDM: `Ofdm(modulation="bpsk"|"qpsk"|"qam16"|"qam64", pilot_spacing=..., symbol_seed=..., output="cfr"|"time")` synthesizes the received post-prefix samples of a seeded QAM stream with a pilot comb and a cyclic prefix, so intra-symbol motion produces inter-carrier interference and an echo past the prefix inter-symbol interference; under `Motion.adc()` it observes the world per sample. `output="cfr"` publishes the demodulated estimate `Y / X`; processing `ofdm_cfr_estimate` and `ofdm_demodulate` turn a time cube into it. The unmodulated default is unchanged.
- FMCW chirp sequences: `Fmcw(sequence=(Chirp(slope=..., start_offset=..., idle=..., adc_start=..., ramp_end=...), ...))` runs several TI-style chirp profiles in one frame, one profile per slot, repeated across the frame.
- `Fmcw(phase_error=(c2, c3, ...))`: a polynomial phase error on the transmitted ramp (rad/s^k), entering the beat as `E(u) - E(u - tau)`.
- `Fmcw(sampling="real")`: an in-phase-only receiver; the published samples are the real part of the complex baseband after the receive chain.
- `Fmcw(mimo="ddm"|"bpm", ddm_bands=...)`: Doppler-division and binary-phase transmit multiplexing (TDM stays the default). The result is the physical one-transmit-channel cube; `witwin.radar.processing.demultiplex` returns the `[TX, RX, slow, fast]` virtual array.
- `Result.observation_schedule` and `SynthesisResult.observation_schedule`: `frame`, `slow_time`, `fast_time` or `adaptive`, the schedule the frame's synthesis actually used.

