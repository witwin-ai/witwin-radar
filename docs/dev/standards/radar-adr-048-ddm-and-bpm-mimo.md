# R-ADR-048: DDM and BPM transmit multiplexing, and the demultiplexer

Status: Accepted (0.5.0)

## Context

FMCW MIMO was TDM only, and OFDM and pulsed trains assume ideal orthogonality
between transmitters (F-5). Automotive radars use Doppler-division (DDM) and
binary phase (BPM) multiplexing to keep every transmitter on every chirp.

## Decision

`Fmcw(mimo="tdm" | "ddm" | "bpm", ddm_bands=0)`, TDM the default.

**Physics.** Under DDM and BPM every transmitter fires every chirp, so the slow
time of chirp `c` is `c T_c` for every pair (the beat kernel runs with one slot
per chirp) and transmitter `t` carries a physical code:

* DDM: `exp(+j 2 pi t c / B)`, `B = ddm_bands or num_tx` Doppler sub-bands;
* BPM: the Sylvester Hadamard sign `H[c mod num_tx, t]` (`num_tx` a power of
  two, `num_chirps` a multiple of it).

A receiver hears the coded sum over transmitters.
`synthesis.fmcw.multiplex_transmitters` contracts the sink-major pair axis with
the code in ONE einsum, `[chirp, pair, sample] -> [chirp, rx, sample]`, before
the receive chain, so noise is added once per receiver. The beat cube carries
the CONJUGATE of the physical code (the receiver mixes against an uncoded ramp);
`FmcwSpec.transmit_code()` owns the code.

**What is published.** A DDM or BPM result is the physical cube
`[frame, 1, RX, chirp, sample]`: one transmit channel. `ProcessingAxes` carries
`mimo`, the code in the cube's phasor convention, the code period `P` (`B` for
DDM, `num_tx` for BPM) and `transmit_channels`; `ProcessingCube` checks the cube
against `transmit_channels`.

**Demultiplexing** (`processing.demultiplex`, beside `tdm_compensate`) returns
the `[TX, RX, C / P, S]` virtual cube at `P` chirp periods:

* DDM: multiply by `conj(code[:, t])` (transmitter `t` to zero Doppler, the
  others `(t' - t) / P` of the band away), keep the `C / P` slow-time bins
  around zero, inverse transform with `C / P` points, scale `(C / P) / C`. A
  target faster than the sub-band edge folds into a neighbour: DDM's ambiguity.
* BPM: `z_t[b] = (1/P) sum_i conj(code[bP + i, t]) y[bP + i]`, exact while the
  scene holds still across a block.

The record is `demultiplexed`, its unambiguous velocity `1/P` of the physical
cube's. `tdm_compensate` is the identity for DDM and BPM: there is no slot
phase between transmitters that fire together.

**Refusals.** Array mutual coupling (it acts on separately recorded transmit
channels) and adaptive motion are refused under DDM and BPM.

## Evidence

* `tests/test_mimo_ddm_bpm.py`: a still target at 17 degrees, 2 x 4 array: the
  eight demultiplexed virtual channels at the range peak match TDM's under DDM
  and BPM within 1e-3, and step by `pi sin(theta)` per half wavelength (rel
  2e-3); shapes, demultiplexed axes and the refusal.
* `tests/oracle/test_waveform_snr.py::fmcw-ddm`, `::fmcw-bpm`: two co-located
  transmitters, the coherent SNR is twice one transmitter's within 0.5 dB
  (measured minus predicted -0.040 dB for both).

## Consequences

OFDM and pulsed MIMO remain ideally orthogonal; CDM is not provided.
