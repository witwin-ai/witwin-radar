# R-ADR-046: Coded and stepped pulses, fast-time observations, and time-domain OFDM

Status: Accepted (0.5.0)

## Context

Two waveforms froze the world for a whole slow-time slot (D-5): a pulsed train
observed the scene once per pulse and OFDM once per symbol, even under
`Motion.adc()`, so intra-pulse Doppler, LFM range-Doppler coupling and OFDM
inter-carrier interference (ICI) did not exist. OFDM was an ideal CFR with no
symbols, prefix or pilots, and the pulsed family had only a rectangle and an LFM
(F-4). Nothing told a consumer which observation schedule a frame had actually
used (M1-5).

## Decision

**One coded pulse family.** `PulsedSpec.pulse_kind = "coded"` is a piecewise
envelope of `num_chips` chips of width `T_c = T_p / num_chips`; chip `i` is
`a_i exp(j 2 pi f_i (u - i T_c))`, `|a_i| <= 1` (one is the full transmit
amplitude, so samples stay in sqrt(W), R-ADR-029). `pulsed_echo.cu` reads a
float64 `[num_chips, 3]` table (`|a_i|`, `arg(a_i) / 2pi` in cycles, `f_i` Hz)
and still evaluates the envelope in closed form at the exact fractional delay.
That one family is:

* PMCW: `Pulsed.pmcw(code, chip_rate, repetitions, sample_rate)`, a phase code
  transmitted continuously (`periodic = True`: the envelope is the periodic
  extension of one code period, `pri = T_p`, the gate one period). Its matched
  filter is the CYCLIC correlation (`correlate(..., periodic=True)`).
  `Pulsed.m_sequence(order)` generates maximal-length codes (orders 2-16).
* FSK: `Pulsed.fsk(frequencies, chip_width, ...)`, unit chips at their own
  tones; `Pulsed.costas(prime)` gives a Welch-Costas order.
* Arbitrary: `Pulsed.arbitrary(samples, rate, ...)`, a user-sampled complex
  envelope held for one chip per sample, the zero-order hold a DAC outputs.

The derivative with respect to delay is the almost-everywhere one: `d|s|/du =
0`, `d(arg s)/du = 2 pi f_i`; chip boundaries are discontinuities, like the
rectangle's edges. The matched-filter replica is `s(u) / E_s`, `E_s = int |s|^2
du` (`PulsedSpec.envelope_energy_s`), so an isolated row still compresses to
exactly `C_rt`.

**Stepped frequency.** `frequency_step_hz` puts pulse `l` at `f_ref + l df`.
The Channel weight is evaluated at `f_ref`, so the offset multiplies the FULL
delay, like an OFDM subcarrier: `cycles -= l df tau`. The synthetic range
profile is the `1/L` inverse transform across pulses
(`processing.stepped_frequency_profile`), resolution `c / (2 L df)`.

**Fast-time observations for pulsed trains.** Under `Motion.adc()` a pulsed
session observes the world at every gate instant `t_frame + l T_pri + t_g + m
T_s`, the FMCW pattern. Each slot batch is ONE `pulsed_echo_forward` launch
whose segments carry their own `(gate start, stepped offset)` from a float64
`segment_table` (the model is the session's per-segment beat kernel); a segment
equals a one-instant launch bit for bit. A refreshed weight carries
`exp(-j 2 pi f_c tau(t))` at its own instant, so the carrier walks inside the
pulse (intra-pulse Doppler) and an up-chirp compresses at `tau + f_c tau_dot /
k`, `k = B / T_p` (range-Doppler coupling). `Motion.chirp()` keeps the per-pulse
schedule and goes through the same batched route, whose schedule carries each
pulse's stepped offset (a one-pulse spec alone would see pulse 0).

**Time-domain OFDM.** `Ofdm(modulation=..., pilot_spacing=..., symbol_seed=...,
output="cfr"|"time")`. A modulated stream is synthesized as received
post-prefix samples by the new `ofdm_time_{forward,backward,jvp}` family
(`ofdm_time.cu`): symbol `j` carries `x_j(t) = N^-1/2 sum_n X[j][n] exp(j 2 pi n
df t)`, periodic in `T_u`, so its first `T_cp` is the cyclic prefix; an echo
longer than the prefix reads the previous symbol (ISI), and the delay walks
with the row's delay rate or refreshes per sample, so motion produces ICI.
`X` is seeded PCG64 QAM (BPSK, QPSK, 16-, 64-QAM) at unit mean power with
pilots `1` on every `pilot_spacing`-th subcarrier (`OfdmSpec.symbol_table()`).
`output="cfr"` publishes the demodulated estimate `Y / X`
(`processing.ofdm_cfr_estimate`, after the receive chain); `output="time"`
publishes the samples and `processing.ofdm_demodulate` maps the cube. The
unmodulated comb as a CFR keeps the closed-form CFR kernel, bit for bit. The
backward writes per-(path, symbol) partials that one Torch sum reduces.
Pulsed and time-domain OFDM refreshed frames refuse a declared leakage, which
keeps one launch per observation.

**The schedule that ran.** `SynthesisResult.observation_schedule` and
`Result.observation_schedule` are `frame`, `slow_time`, `fast_time` or
`adaptive` (`assembly.OBSERVATION_SCHEDULES`), set by the echo from what it
did, not from the requested `Motion`: an unmodulated OFDM CFR under
`Motion.adc()` reports `slow_time`.

## Equations, units and owners

`pulsed_echo.cu` and `ofdm_time.cu` state their closed forms, units, sign
(Channel convention) and derivatives in their headers. Delay, delay rate and
weight are Channel's and the composer's; chip tables, symbol tables and
schedules are synthesis-owned waveform data and refuse a derivative.

## Evidence

* `tests/test_pulsed_coded.py`: kernel = float64 closed form (FSK, stepped,
  PMCW); m-sequence cyclic MF = `C` with every sidelobe `-C/N` (atol 2e-6);
  Costas MF peak at the delay; stepped profile peak `C` at `L df tau`; native
  VJP/JVP vs the FD-validated oracle; scheduled launch = per-segment launches,
  `torch.equal` on forward, VJP and JVP.
* `tests/test_pulsed_fast_time.py`: 300 m/s at 1.5 m. In-pulse phase step
  `-2 pi f_c tau_dot T_s` (rel 1e-3); refreshed / per-pulse train =
  `(r_l/r)^2 exp(-j 2 pi f_c tau_dot t)` sample by sample; LFM coupling 3.85
  samples (rel 2 %), zero for the per-pulse schedule.
* `tests/test_ofdm_time.py`: kernel = float64 closed form including ISI;
  still echo demodulates to the CFR kernel (2e-5); ICI: measured signal-to-ICI
  vs `|I_0|^2 / (1 - |I_0|^2)`, `|I_0| = sin(pi eps) / (N sin(pi eps / N))`,
  within 0.5 dB at eps = 0.05-0.3, and end to end (fast-time, 47 m/s) within
  1 dB; native VJP/JVP vs the oracle.
* `tests/oracle/test_waveform_snr.py`: the radar equation over `k T_sys`
  within 0.5 dB. Measured minus predicted: `pmcw` +0.136 dB, `fsk` -0.099,
  `stepped` -0.069, `arbitrary` -0.099, `ofdm-qpsk` -0.040 (the existing
  cases: FMCW -0.040, OFDM -0.040, pulsed -0.099).
* `tests/test_waveform_budget.py`: one forward and one backward launch per call,
  no host read; a fast-time pulsed frame is `ceil(observations / batch)` launches.

## Consequences

* `pulsed_echo_*` gained `chips`, `segment_table`, `frequency_step_hz` and
  `periodic`; rect and LFM numbers are unchanged (the added terms are exact
  zeros there).
* Dynamic pulsed scenes under `Motion.adc()` now cost one world observation per
  gate sample, like FMCW.
* Not done: adaptive motion for pulsed and OFDM (it interpolates one FMCW
  ramp), APAS code generation (any `+/-1` code, an APAS included, is accepted as
  `chips`), and the frontend's DC/flicker stages on time-domain OFDM (still
  refused by name).
