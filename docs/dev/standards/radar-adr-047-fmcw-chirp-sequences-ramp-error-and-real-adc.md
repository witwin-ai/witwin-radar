# R-ADR-047: FMCW chirp sequences, start offsets, ramp phase error, and an in-phase ADC

Status: Accepted (0.5.0)

## Context

FMCW had one ramp per frame and a complex ADC only (F-4). TI-style
configurations run several chirp profiles in one frame (start frequency, slope,
idle and ADC start per chirp), real receivers sample the in-phase channel only,
and a synthesized VCO ramp is not linear. The frontend deferred chirp
nonlinearity to the waveform owner (R-ADR-031): it is a property of the
transmitted signal, not a receive stage.

## Decision

**Per-slot profiles.** `Fmcw(sequence=(Chirp(...), ...))`: one `Chirp` profile
per slot of a loop, repeated cyclically across the frame's slots (slot `s` =
`chirp * num_tx + tx` under TDM). A `Chirp` field left `None` is the ramp's own
value; `start_offset` [Hz] is the start frequency relative to the carrier.
Slot `s` occupies its ramp then its idle, so its ramp starts at the float64
running sum of the earlier slots' `idle + ramp_end`. Sample rate and sample
count stay per waveform (one cube). `FmcwSpec.slot_profiles()` is the one
owner of the `[slots, 4]` table (ramp start, slope, offset, ADC start) the beat
kernel reads.

**The phase.** `fmcw_phase.cuh::fmcw_profile_terms` adds two terms to the
uniform phase owner without touching it:

* start offset `f0`: `+ f0 tau` (the weight carries only `f_ref`), derivative
  `f0`;
* ramp phase error `E(t) = sum_i c_i t^(i+2)` cycles, `t` the time since the ramp
  start: the beat is `tx(u) conj(tx(u - tau))`, so `+ E(u) - E(u - tau)`,
  derivative `E'(u - tau)`. `Fmcw.phase_error` takes the coefficients in
  rad/s^k (converted once to cycles); orders 0 and 1 are a constant phase and a
  frequency offset, not a nonlinearity.

With neither, the function returns the uniform terms unchanged, and an empty
table runs the kernel's original scalar expression: uniform frames are
bit-identical. The beat operators take `slot_table` [slots, 4], `segment_table`
[segments, 3] (window start, slope, offset per segment: one launch over many ADC
instants of a refreshed sequence) and `chirp_poly`. They replace the
single-column `segment_t_start`.

**Spectrum.** The Dirichlet spectrum is the closed form of one linear ramp, so
a sequence, a start offset, a phase error or a real ADC is synthesized in the
beat domain and transformed by processing; `synthesize_fmcw_rows` refuses the
direct spectrum for them by name.

**In-phase ADC.** `Fmcw(sampling="real")` publishes `Re` of the complex baseband after
every receive stage (`simulation._finish_frame`). The real channel's noise is
`k T_sys f_s / 2` per sample, a real ADC's at `f_s`; the matched SNR is the IQ
receiver's (signal and noise power both halve), while a single spectral bin
loses 3 dB to the image at `-f_b`.

**Processing axes.** A sequence publishes its mean slot period (frame length
over chirps) and the range axis of the waveform's base slope; a mixed-slope
frame is processed per profile by selecting that profile's slots.

## Evidence

* `tests/test_fmcw_profiles.py`: two profiles (slopes 30/45 MHz/us, 150 MHz
  offset, different idle and ADC start, TDM with two transmitters) and a cubic
  ramp error: kernel = float64 closed form (2e-4 of peak); native VJP and JVP
  against the oracle's autograd (3e-3); a table holding the scalars is
  bit-identical to the scalar path; end to end each profile beats at `S tau`;
  `sampling="real"` publishes exactly `Re` of the IQ cube, and its spectrum is
  Hermitian with each image at half the IQ amplitude.
* `tests/oracle/test_waveform_snr.py::fmcw-real-iq`: radar equation over
  `k T_sys` within 0.5 dB (measured minus predicted -0.187 dB).
* `tests/test_waveform_budget.py`: a sequenced, nonlinear frame is one beat
  launch and one backward launch, with no host read.

## Consequences

* Adaptive motion refuses a sequence, a start offset and a phase error (it
  interpolates one uniform ramp).
* `fmcw_beat_*` schemas changed (three tables in place of `segment_t_start`).
