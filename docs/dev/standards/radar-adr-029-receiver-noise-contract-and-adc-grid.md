# R-ADR-029: One receiver noise contract for every waveform, and a mid-tread ADC with a zero code

Status: Accepted (0.5.0)

## Context

The receive chain converts a synthesis cube from sqrt(W) to volts once and adds
circular complex Gaussian noise of variance `k T_sys B R` per element. That is
correct only if every waveform publishes samples whose `|x|^2` is a power in W
and if `B` is the noise bandwidth of one sample. Three things broke it:

1. **Pulsed samples were not in sqrt(W).** The pulsed kernel evaluated a
   unit-ENERGY envelope `C_rt / sqrt(T_p)`, so a sample was in sqrt(W / s), and
   the default noise bandwidth was the LFM sweep `B` rather than the sample
   rate. The SNR was optimistic by `f_s / (B T_p)`: +54 dB at 20 MHz, 10 us,
   50 MS/s. Measured before this change (table below): +54.0 dB.
2. **Every frame shared one thermal realisation.** The Philox key was the base
   seed alone, so frame 1 of a `simulate` call carried exactly the noise of
   frame 0 (`torch.equal` on the noise of two frames was true). Averaging frames
   did not reduce noise.
3. **The ADC grid had no zero code.** `step = 2 FS / (2^b - 1)` with levels
   `-FS + k step` is a mid-RISER grid: a zero input read `+/- step/2` on I and
   Q, a DC offset on every idle channel, while the documentation said mid-tread.
4. **A device assertion guarded the oscillator.** `torch._assert_async` on
   non-finite times or negative delays aborts with a device-side assert, which
   leaves the CUDA context unusable for the rest of the process.

## Decision

**Noise contract.** Every waveform publishes PHYSICAL-amplitude complex
samples in sqrt(W) at the receive port, and the thermal stage adds
`CN(0, k T_sys f_s R)` per complex sample, `f_s` being the waveform's complex
sample rate (the equivalent noise bandwidth of a white complex stream at
`f_s`). `Radar` resolves `Noise.bandwidth` and `Noise.phase_sample_rate` from
`waveform.sample_rate` for all three waveforms.

* FMCW: unchanged. A beat sample is `C_rt` in sqrt(W) at `f_s`.
* OFDM: unchanged. A CFR bin is one output of a unitary `N`-point transform of
  time samples at `N df`, so its noise is `k T_sys N df`, which is `f_s`.
* Pulsed: the envelope is now UNIT AMPLITUDE (`PulsedSpec.pulse_amplitude = 1`,
  `pulse_normalization = "unit_amplitude"`), so an in-pulse sample is `C_rt`.
  The matched-filter replica carries `1 / T_p`
  (`PulsedSpec.replica_amplitude`), which keeps the processing convention that
  an isolated row's matched-filter peak is exactly `C_rt`; its output noise is
  `k T_sys / T_p`, and the per-pulse SNR is the textbook `E / N0`.

Under this contract the coherent SNR of any matched statistic over `K`
echo-bearing samples is `P_r (K / f_s) / (k T_sys)`, with `K / f_s` equal to
`N T_s` per chirp, `T_u` per symbol and `T_p` per pulse.

**Frame keying.** `FrontendChain.apply(..., frame_time_s=t)` keys the white
draws (thermal) with `frame_seed(seed, t)`: the base seed XOR MurmurHash3's
64-bit finaliser of the IEEE-754 bits of `t`, masked to 63 bits. The finaliser
maps zero to zero, so a frame at `t = 0` keeps the base seed and every
single-frame result is bit-identical to before. The time-keyed processes - the
oscillator and the baseband flicker - keep the base seed, because they are one
realisation over absolute time. An explicit `seed_base` keys everything.
`simulation._finish_frame` passes the trace's frame instant.

**ADC grid.** Two's-complement mid-tread: `step = 2 FS / 2^b`, codes
`-2^(b-1) ... 2^(b-1) - 1`, levels `k step`, formed in double inside the
kernel. Zero is a code; the grid spans `[-FS, FS - step]`; a component whose
unclamped code is out of range saturates and is counted as clipped, so the
no-overload range is `[-FS - step/2, FS - step/2)`. One bit is refused (its
codes are `{-FS, 0}`).

**Oscillator check.** The oscillator kernel writes NaN for an invalid query and
increments a device counter. `Noise.phase_difference` decodes it on the host
(one `.item()`); the per-observation route accumulates into a counter owned by
the chain, decoded once by the frame's `apply`. No device assertion remains.

## Evidence

`tests/test_noise_contract_snr.py` (GPU), 77 GHz, 10 dBm, NF 6 dB, 1 m^2,
isotropic elements on boresight. Oracle: the radar equation, `k T_sys`, and the
coherent time `K / f_s`, in float64 host arithmetic; the noise realisation is
the difference of a noisy and an ideal cube at the same seed.

| waveform | predicted SNR | before: measured - predicted | after: measured - predicted |
|---|---|---|---|
| FMCW (10 MS/s, 256 x 64, 6 m) | 37.82 dB | 0.00 dB | -0.004 dB |
| OFDM (256 x 1 MHz, 64 symbols, 30 m) | -4.22 dB | 0.00 dB | -0.004 dB |
| Pulsed LFM (20 MHz, 10 us, 50 MS/s, 32 pulses, 150 m) | -25.19 dB | +54.00 dB (echo power +50.0, noise -4.0) | +0.021 dB |

The received-power and per-sample-noise components are each asserted within
0.5 dB as well, so two compensating errors cannot pass as one correct SNR.

ADC: `test_a_zero_input_reads_exactly_zero_on_both_components` (2 to 24 bits,
inputs within half a step of zero), the reference grid test against
the `quantize` oracle in `tests/support/reference_frontend.py`, the `step^2/12` variance test and
the full-scale-sine SQNR test (`6.02 b + 1.76` dB within 0.3 dB for a sine at
the no-overload full scale `FS - step/2`).

## Consequences

* Pulsed cubes are larger by `sqrt(T_p)` (numerical change); matched-filter
  peaks are unchanged. Pulsed noise is now correctly scaled.
* Any multi-frame noisy result changes (frames are independent now); a
  single frame at `t = 0` is bit-identical.
* ADC outputs move by up to half a step; `Adc.step` is `2 FS / 2^b`.
* `Fmcw.bandwidth` is removed: it was the sample rate under a misleading name,
  and the noise bandwidth is now always `waveform.sample_rate`.
