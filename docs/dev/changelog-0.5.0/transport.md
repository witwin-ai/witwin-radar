# transport

## Breaking changes

- Element patterns and polarization are Radar's (R-ADR-041). Channel's `complex3_transport` is replayed and the join divides out Channel's dipole projection, so an `"isotropic"` element is isotropic: off the plane normal to the polarization the echo no longer loses `sin^4` of the angle (it lost -5.0 dB at 30 degrees, -12.0 dB at 45, -24.1 dB at 60). `Pattern.dipole()` attenuates by its cut once instead of twice. Numbers on the polarization plane are unchanged to float32 rounding. Migration: none needed; drop any compensation for the old loss.
- A Channel whose fixed-topology replay cannot return `complex3_transport` for the requested components or AD mode is refused by name; there is no scalar fallback.
- `RadarLegBatch.coefficient` (a projected scalar) is replaced by `RadarLegBatch.field`, the `complex64[rows, 3]` field vector at the site. `RadarLegBatch.coefficient_band` is replaced by `RadarLegBatch.field_band` (`complex64[rows, F, 3]`). `RadarWorldBinding` has one `sites` spec instead of `site_sources`/`site_sinks`; the receivers appear twice over one set of tensors, as `receivers` (the site-less leg's sinks, no power) and `receiver_sources` (the outbound leg's 1 W sources). `TwoWayComposer.outbound_max_depth` is removed.

## Added

- `ScatteringMatrix` (`witwin.radar`): `PointTargets(scattering=ScatteringMatrix(hh, hv, vh, vv, up=...))` gives a target a 2x2 polarimetric scattering matrix in its local backscatter-aligned basis; `ScatteringMatrix.sphere()` and `ScatteringMatrix.dihedral()` (HH and VV 180 degrees apart). `witwin.radar.scattering.ScatteringMatrixResponse` is the response interface a polarimetric scatterer implements (`matrix_width = 4`, `evaluate -> complex64[sites, 4]`, `site_up -> float32[sites, 3]`).
- `Radar.rx_polarization`: the receive elements' polarization (default: `polarization`), for cross-polar channels.
- Float64 target positions: `PointTargets(positions=...)` and a `trajectory` may return float64; Channel receives them rounded, the round-trip path length is recomputed from them.
- `RadarPathBatch.total_delay_residual_s`: `total_delay_s.double() + total_delay_residual_s.double()` is the float64 round-trip delay. `SynthesisPathBatch.total_delay_residual_s` carries it into synthesis, and the row-level entries (`synthesize_fmcw_rows`, `synthesize_cfr_rows`, `synthesize_ofdm_time_rows`, `synthesize_echo_rows`) take it as `total_delay_residual_s=`.

## Fixed

- Carrier-phase staircase of moving targets (D-1): the join recomputes the round-trip length in float64 from the endpoints and Channel's interaction points and replaces Channel's float32 phase (R-ADR-040). Against a float64 trajectory at 77 GHz the integrated phase error is -130 / -111 / -100 dBc and the spur floor -133 / -116 / -107 dBc at 2 / 30 / 100 m (was -73 / -55 / -43 and -76 / -57 / -49 dBc); 1 um of motion at 30 m is resolved (the float32 ulp there is 1.9 um).
- The adaptive motion test compares delays in float64 with a quantization allowance (D-9): the same motion takes the same number of probes at 5 m and 100 m instead of refining on float32 rounding beyond about 70 m.
- Every waveform kernel (FMCW beat and range spectrum, fused adaptive FMCW, OFDM CFR and time samples, pulsed) reads the float64 delay pair, so a 4 GHz sweep at 100 m no longer carries the float32 delay's 7e-4 rad of bandwidth phase (now below 1e-5); without a residual every kernel is bit-identical to before.
- Site-less rows (wall echoes, direct paths) carry the float64 path length, the element factors and the receiver's polarization like the site rows; a discovery that keeps its rows keeps the float64 site.
- A wideband band's columns each remove Channel's phase at the column's own float32 launch frequency, so every subcarrier carries the float64 phase (6e-7 rad measured, where one wave number for the band would be off by up to 1e-2 rad at 37 m).
- The element factor follows a radar pose: it reads each leg row's launch polarization, which a `pose=` turns.
- `AspectScatterResponse` works on reflected outbound rows (it read the receiver's arrival direction as the site's departure direction and therefore refused them).

## Performance

- Site-less rows are composed by the same native join launch as the site rows (a fourth row family, written straight to their frozen composed positions), not by a separate Torch path and a merge gather.
- The join composes the field vectors with no extra launch: fields, the response and the composed value cross the native boundary as `view_as_real`/`view_as_complex` views, which removes the per-composition split and recombination launches.
- A directional antenna pattern on a slot-batched replay strides one slot's routing tables natively instead of copying them per slot (bit-identical to the copied tables).

## Changed

- The outbound leg is evaluated from the receive element toward the site (Lorentz reciprocity, still a separate Channel evaluation, R-ADR-039); composed co-polar scalars move by at most 1.3e-7 relative per row.
