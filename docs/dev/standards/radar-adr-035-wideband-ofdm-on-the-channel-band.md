# R-ADR-035: OFDM consumes Channel's `[K, F]` band; FMCW stays narrowband

Status: Accepted (supersedes the band consumption of R-ADR-015)

## Context

Every waveform composed its rows at the reference frequency (PH-7, MP-8). For OFDM that freezes the spreading term,
the material reflectivity and the layer-stack Airy fringes across the band: a 0.1 m `eps_r = 4` wall has a ~750 MHz
fringe, so a few-hundred-MHz OFDM band sees its reflectivity change by tens of percent, and the narrowband law cannot
express it. R-ADR-015 had described a band route and was superseded when that route was deleted. Channel 0.6 replays a
frozen topology at `F` offsets in one call (`FixedTopologyRequest.frequency_offsets_hz`, `ScalarTransport.
coefficient_offsets`, `[K, F]`, `(1 + F)` launches per bucket, one copy and one synchronization whatever `F`), and
the native OFDM CFR kernel already accepts one weight per subcarrier (`weight_columns`).

## Decision

### 1. `wideband=True` on the verbs, OFDM only

A wideband run asks every replay for `frequency_offsets_hz = (n * df for n in range(num_subcarriers))`, which is
exactly the CFR kernel's subcarrier grid (`f_n = f_ref + n df`). The adapter publishes the band as
`RadarLegBatch.coefficient_band` beside the reference `coefficient`. Discovery has no band, so a wideband run always
replays (it never takes the discovery route of R-ADR-033), and `transmission`/`diffraction`, which cannot be frozen,
are refused with it by name.

### 2. The band is composed by the existing equations, columns as slots

`propagation.band_as_slots` lays a leg's `F` columns out as outer slots: row `j * rows + r` carries
`coefficient_band[r, j]`, and every frequency-independent member (delay, directions, validity, segment ends) is
repeated. That is the block-diagonal slot layout, so `TwoWayComposer.for_slots(F)` and
`RoundTripPatternStage.for_slots(F)` compose and pattern-weight every column in one launch each with the native join
and sensor-weight kernels, their VJPs and JVPs unchanged. The tables are built once per (topology, group size, `F`)
and cached with the slot tables. The result is reshaped to `RadarPathBatch.complex_transfer_band` `[rows, F]`, whose
column 0 (offset 0) is bit-identical to `complex_transfer_ref`. The element patterns are frequency independent, so
one pattern stage serves every column; the site-less leg is banded the same way.

`SynthesisPathBatch` carries the band; `synthesize_ofdm` checks that its grid is the spec's subcarrier grid and hands
the `[K, N]` weight to the CFR kernel, which then applies only the slow-time change of each column's phase.
`require_compatible` refuses a band for FMCW and pulsed specs rather than silently using its reference column.

Cost: one replay of `(1 + F)` launches per bucket per leg instead of one, and `F` times the composed rows for the
join and pattern launches. Off by default.

### 3. Oracle

`tests/test_multipath_discovery.py::test_m3_8_ofdm_subcarrier_response_of_a_slab_bounce_matches_the_analytic_slab`:
a 0.1 m `eps_r = 4` slab 4 m in front of the radar, 64 subcarriers x 10 MHz. The site-less bounce's per-subcarrier
magnitude equals `sqrt(P) c / (4 pi f_n 8 m) |Gamma_slab(f_n)|` (normal-incidence Airy reflection) to 1e-3 at every
subcarrier (measured worst 6.3e-4), across a band where that value changes by more than 20 %.

### 4. FMCW: not done, and why

An FMCW chirp sweeps continuously, so a per-frequency weight is a per-ADC-sample weight. Doing it natively means a
weight column per sample or a coarse band interpolated per sample inside `fmcw_beat.cu` (forward, VJP and JVP), and
abandoning the closed-form Dirichlet spectrum family for every banded run, because that family exists only for a
weight that is constant over the chirp. The Channel side is cheap (a handful of columns); the Radar side is a new
native interpolation mode in the beat family and a forced beat-plus-FFT route. Across a 77-81 GHz sweep the omitted
terms are the spreading tilt (about 0.44 dB end to end, a smooth amplitude taper the range window dominates) and the
material and layer fringes of thick dielectric walls. The FMCW delay phase is exact either way. This is left for a
release that also needs per-sample weights for another reason; `wideband=True` on an FMCW radar is refused by name
and cites this record.
