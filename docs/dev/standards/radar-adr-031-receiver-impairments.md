# R-ADR-031: Receiver impairments in their physical place in the chain

Status: Accepted (0.5.0)

## Context

The receive chain modelled thermal noise, a free-running (1/f^2) oscillator, an
LNA gain, a quadrature imbalance, a frame-static AGC and the ADC. Missing were
the impairments a real radar front end shows and a learned model can latch on
to: gain compression, the close-in (1/f^3) and far-out (floor) phase-noise
regions, mutual coupling between array elements, per-pair transmit leakage for
every waveform, DC offset and flicker noise, and a time-dynamic AGC.

## Decision

Every stage is off by default (a radar without the new fields is bit-identical)
and has one native kernel. The order is fixed and physical:

    port -> coupling -> phase, thermal, LNA (fused) -> compression -> IQ
         -> baseband (DC offset + flicker) -> AGC -> ADC

**Mutual coupling** (`Radar.coupling = Coupling(tx=..., rx=...)`,
`witwin/radar/sensors.py`). Port coupling matrices applied as
`Y = C_tx^T H C_rx^T` over the cube's (TX, RX) axes, ahead of every active
device, so the coupled echoes are mixed and the thermal noise is not. Two
cuBLAS products through `torch.matmul`; the derivative is their transpose. In a
TDM frame the neighbouring elements' responses are taken at the port's own
slot: the coupled cube is assembled from one frame, not re-synthesised per slot.

**Leakage** (`Leakage(isolation_db, delay_s, phase)`). Each of the three is a
scalar or a `[num_tx][num_rx]` table, and the coupled rows are synthesised by
the waveform's own row entry for all three waveforms: FMCW de-chirps it into a
near-range tone, OFDM sees a short-delay CFR term, and a pulsed receiver sees it
only where its range gate overlaps the transmitted pulse.

**Compression** (`Radar.compression = Compression(iip3, p1db, am_pm)`), after
the LNA and referred to the receiver input:
`G(u) = (1 - u + c5 u^2) exp(j k u)`, `u = P_in / P_IIP3`, held at the first
maximum of the output amplitude (a soft limiter). The cubic term sets IIP3
(the 1-dB point falls 9.64 dB below it); `c5` places a declared P1dB; `k` is the
AM/PM at that point. The analytic 2x2 Jacobian is the backward and the JVP.

**Shaped-PSD oscillator** (`Noise.phase_flicker_density/offset`,
`Noise.phase_floor`). The existing Brownian bridge stays the exact white-FM
(1/f^2) region and is bit-identical when the others are absent. Flicker FM
(1/f^3) is the same dyadic Schauder expansion with tent amplitude `w` instead of
`0.5 sqrt(w)`: its derivative is a Haar sum of equal height in every octave, a
Voss-McCartney flicker-FM process with `L(f) = c^2 / (2 pi^2 ln2 f^3)` up to an
octave ripple of -0.36/+0.21 dB. The floor is a zero-order hold of i.i.d.
draws over cells of `1 / phase_sample_rate`, `L = s^2 / f_cell`. All three
are keyed by absolute time, so `phi(t) - phi(t - tau)` keeps the range
correlation: exactly for 1/f^2, at dyadic scales for 1/f^3, and as the
triangle `2 s^2 min(tau f_cell, 1)` for the floor.

**Baseband DC offset and flicker** (`Radar.dc_offset`, `Noise.flicker_corner`).
Added after the quadrature split, per receive channel. Flicker is a Haar sum
over octaves from 1 s down to two samples, per-component density
`h^2 / (2 ln2 f)`, with `h` set so that the complex density equals the thermal
density `k T_sys R` at the corner (times the LNA gain, being after it). The kernel
writes the additive term and the Python owner adds it to the signal, so the
identity derivative is carried by that addition and no companion kernel exists. The OFDM cube is a frequency-domain CFR,
so a time-domain baseband impairment is refused for OFDM by name.

**Time-dynamic AGC** (`Agc(time_constant=tau)`). One gain per group and
slow-time block (a chirp index across the TDM cycle, a symbol, a PRI): a causal
first-order loop in log gain,
`l_k = clamp((1 - beta) l_{k-1} + beta ln(target / r_{k-1}))`,
`beta = 1 - exp(-T_block / tau)`, the first block acquired at its own ideal
gain; the state does not carry across frames. The forward is a block-RMS
reduction, a per-group scan and an apply; the backward and JVP are an inner
reduction, the adjoint or tangent scan and one rank-one combine. `tau = None`
keeps the frame-static AGC.

**Not done here.** Interference from other radars and chirp nonlinearity
(FE-4 lists them) are waveform-synthesis effects rather than receive-chain
stages and are left for a later task.

## Evidence

`tests/test_receiver_impairments.py`, `tests/test_shaped_phase_noise.py`,
`tests/test_instrument_artifacts.py` (GPU): the 1-dB point at the declared input
P1dB with and without an LNA; a two-tone IM3 extrapolating to the declared IIP3
within 0.1 dB; AM/PM at the 1-dB point; monotone saturation; compression and
dynamic-AGC VJP/JVP against float64 Torch oracles; the dynamic AGC's gains equal
the loop's closed form and decay with the declared time constant; DC offset
exact; flicker density `k T_sys R f_c / f` within 1.5 dB per octave; coupling
equals `einsum(C_tx, H, C_rx)` and enters before noise; the 1/f^3 region at
-30 dB/decade at its declared level and the floor flat at its level, each within
1 dB (0.5 dB for the floor) of the textbook power law by an independent Welch
estimator; the floor's range correlation; the three-region composite mask;
per-pair leakage for OFDM and pulsed against the closed forms; each new stage is
one native operator call per frame.
