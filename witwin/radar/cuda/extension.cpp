#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(witwin_radar_dirichlet_cuda, m) {
  // Dirichlet range spectrum. The path weight is COMPLEX, carried as the two
  // real tensors (a, a_im) exactly like the beat and join families: no complex
  // tensor crosses the autograd boundary. A real weight is a_im = 0 and is
  // bit-identical to what this family produced before it gained the component.
  //
  // Two additive switches, both defaulting to the legacy meaning:
  //
  //   fc is the carrier home, mirroring carrier_hz in the beat family.
  //     fc != 0  - the kernel owns the absolute phase 2 pi fc tau.
  //     fc == 0  - the weight owns it and the kernel applies none.
  //   A Channel coefficient already carries the reference-frequency phase, so
  //   pairing one with fc != 0 double counts the carrier. The Python contract
  //   (synthesis/contracts.py, rule R1) refuses that combination before any
  //   launch; this comment records why the kernel does not need to.
  //
  //   tau_is_seconds says what the first tensor holds.
  //     0 - a ONE-WAY distance in metres, round trip assumed monostatic,
  //         tau = 2 d / c0, with k0_per_meter = (slope * 2 / c0) * n_fft / fs.
  //     1 - a ROUND-TRIP delay in seconds, consumed directly, with the matching
  //         scale k0_per_meter = slope * n_fft / fs.
  //   The second form exists because every Phase-6 contract speaks round-trip
  //   delay, and reconstructing a distance from one only to halve it again is
  //   how a path becomes self-consistently 2x wrong.
  //
  // See R-ADR-004.
  m.def(
      "forward_chunked(Tensor d, Tensor a, Tensor a_im, Tensor(a!) output_re, "
      "Tensor(b!) output_im, float n, float k0_per_meter, int num_bins, int n_fft, "
      "int num_targets, int targets_per_chunk, float fc, float slope, float t_start, "
      "int tau_is_seconds) -> ()");
  m.def(
      "forward_mimo_linear_chunked(Tensor d0, Tensor d_rate, Tensor a0, Tensor a0_im, "
      "Tensor(a!) output_re, Tensor(b!) output_im, float n, float k0_per_meter, "
      "int num_bins, int n_fft, int targets_per_pair, int chirp_per_frame, "
      "float chirp_period, int num_tx, int range_loss_update, float fc, float slope, "
      "float t_start, int tau_is_seconds) -> ()");
  // AD companions of forward_mimo_linear_chunked. They exist so that the frame
  // path has ONE owner: the Torch expression that used to stand in for a
  // reverse-mode call - `dist = d0 + rate * t` plus the `d0 / dist` range-loss
  // update - was a second implementation of this kernel's physics, evaluated in
  // a different dtype and a different order, and reachable only when an input
  // happened to require grad.
  //
  //   d(dist)/d(d0) = 1, d(dist)/d(d_rate) = t_slot
  //   with range_loss_update, d(amp)/d(d0)     =  a (dist - d0) / dist^2
  //                           d(amp)/d(d_rate) = -a d0 t_slot   / dist^2
  //
  // The backward owns one gradient slot per target row and uses no atomics; the
  // jvp keeps the forward's own grid, so a tangent costs one launch. See
  // R-ADR-004.
  m.def(
      "mimo_linear_backward(Tensor d0, Tensor d_rate, Tensor a0, Tensor a0_im, "
      "Tensor grad_output_re, Tensor grad_output_im, Tensor(a!) grad_d0, "
      "Tensor(b!) grad_d_rate, Tensor(c!) grad_a0, Tensor(d!) grad_a0_im, "
      "float n, float k0_per_meter, int num_bins, int n_fft, "
      "int targets_per_pair, int chirp_per_frame, float chirp_period, "
      "int num_tx, int range_loss_update, float fc, float slope, float t_start, "
      "int tau_is_seconds) -> ()");
  m.def(
      "mimo_linear_jvp(Tensor d0, Tensor d_rate, Tensor a0, Tensor a0_im, "
      "Tensor tan_d0, Tensor tan_d_rate, Tensor tan_a0, Tensor tan_a0_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, float n, "
      "float k0_per_meter, int num_bins, int n_fft, int targets_per_pair, "
      "int chirp_per_frame, float chirp_period, int num_tx, "
      "int range_loss_update, float fc, float slope, float t_start, "
      "int tau_is_seconds) -> ()");
  m.def(
      "dirichlet_jvp(Tensor d, Tensor a, Tensor a_im, Tensor tan_d, Tensor tan_a, "
      "Tensor tan_a_im, Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, float n, "
      "float k0_per_meter, int num_bins, int n_fft, int num_targets, "
      "int targets_per_chunk, float fc, float slope, float t_start, "
      "int tau_is_seconds) -> ()");
  m.def(
      "backward(Tensor d, Tensor a, Tensor a_im, Tensor grad_output_re, "
      "Tensor grad_output_im, Tensor(a!) grad_d, Tensor(b!) grad_a, "
      "Tensor(c!) grad_a_im, float n, float k0_per_meter, "
      "int num_bins, int n_fft, int num_targets, float fc, float slope, float t_start, "
      "int tau_is_seconds) -> ()");
  m.def(
      "backward_batched(Tensor d, Tensor a, Tensor a_im, Tensor grad_output_re, "
      "Tensor grad_output_im, Tensor(a!) grad_d, Tensor(b!) grad_a, "
      "Tensor(c!) grad_a_im, float n, float k0_per_meter, "
      "int num_bins, int n_fft, int num_targets, int targets_per_spectrum, "
      "float fc, float slope, float t_start, int tau_is_seconds) -> ()");
  m.def(
      "backward_parallel_bins(Tensor d, Tensor a, Tensor a_im, Tensor grad_output_re, "
      "Tensor grad_output_im, Tensor(a!) grad_d, Tensor(b!) grad_a, "
      "Tensor(c!) grad_a_im, float n, "
      "float k0_per_meter, int num_bins, int n_fft, int num_targets, "
      "float fc, float slope, float t_start, int tau_is_seconds) -> ()");
  m.def(
      "backward_per_bin(Tensor d, Tensor a, Tensor a_im, Tensor grad_output_re, "
      "Tensor grad_output_im, Tensor(a!) grad_d, Tensor(b!) grad_a, "
      "Tensor(c!) grad_a_im, float n, float k0_per_meter, "
      "int num_bins, int n_fft, int num_targets, int bins_per_chunk, "
      "float fc, float slope, float t_start, int tau_is_seconds) -> ()");

  // Phase-4 FMCW beat synthesis over a chirp's fast-time axis. The carrier has
  // two homes and exactly one of the two parameters names it:
  //
  //   carrier_hz = fc, carrier_rate_hz = 0   reproduces the Dirichlet path's
  //     phase structure exactly; the kernel owns the whole carrier phase.
  //   carrier_hz = 0, carrier_rate_hz = fc   is the production path for a
  //     Channel-sourced weight, which already carries exp(j 2 pi fc tau_rt) at
  //     the frozen per-frame delay. carrier_rate_hz supplies the intra-frame
  //     Doppler term fc * (tau - tau_rt) that the frozen weight cannot express.
  //
  // Setting both to fc double counts the carrier, and the Python contract
  // refuses it. Both supported settings are exact; neither is a fallback.
  // See R-ADR-004.
  //
  // `segment_tx_index` (int32, one entry per sensor-pair segment) and `num_tx`
  // carry TDM-MIMO slow time: the slow-time coordinate of a (chirp, segment)
  // cell is its TDM slot, (chirp * num_tx + segment_tx_index[segment]) *
  // chirp_period_s, not the chirp index. They are kernel ARGUMENTS rather than
  // a second pass, so TDM costs no extra launch. `num_tx = 1` with a zero table
  // reduces the slot to the chirp index exactly.
  m.def(
      "fmcw_beat_forward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor segment_tx_index, "
      "Tensor(a!) out_re, "
      "Tensor(b!) out_im, int num_paths, int num_segments, int num_tx, "
      "int num_chirps, "
      "int num_samples, float sample_period_s, float chirp_period_s, "
      "float slope_hz_per_s, float carrier_hz, float carrier_rate_hz, "
      "float t_start_s) -> ()");
  m.def(
      "fmcw_beat_backward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_segment, Tensor segment_tx_index, "
      "Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_tau_rt, Tensor(b!) grad_tau_rate, "
      "Tensor(c!) grad_weight_re, Tensor(d!) grad_weight_im, int num_paths, "
      "int num_segments, int num_tx, int num_chirps, int num_samples, "
      "float sample_period_s, float chirp_period_s, float slope_hz_per_s, "
      "float carrier_hz, float carrier_rate_hz, float t_start_s) -> ()");
  m.def(
      "fmcw_beat_jvp(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor segment_tx_index, "
      "Tensor tan_tau_rt, "
      "Tensor tan_tau_rate, Tensor tan_weight_re, Tensor tan_weight_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, int num_paths, "
      "int num_segments, int num_tx, int num_chirps, int num_samples, "
      "float sample_period_s, float chirp_period_s, float slope_hz_per_s, "
      "float carrier_hz, float carrier_rate_hz, float t_start_s) -> ()");

  // Phase-6 OFDM channel frequency response over the (symbol, subcarrier)
  // grid. The cube is published in the CHANNEL phasor convention exp(-j k d),
  // NOT conjugated: OFDM demodulation is per-subcarrier equalisation H = Y / X,
  // which removes the transmitted symbol but not the carrier convention. With
  // n = 0 pinned to the reference frequency, H[0][p][0] == C_rt exactly. The
  // FMCW beat cube stays conjugated; the two are different products.
  //
  // The subcarrier term multiplies the FULL delay tau_k(l) while the
  // carrier-rate term multiplies the drift only. That asymmetry is deliberate:
  // a Channel coefficient carries exp(-j 2 pi f_ref tau_rt) and nothing else,
  // so only the f_ref phase's slow-time CHANGE is missing, while the n * df
  // phase is absent from the weight entirely.
  //
  // weight_columns selects the two weight layouts. 1 is the narrowband weight
  // C[k]; num_subcarriers is the ADR-042 wideband weight C[k][n], row major,
  // holding the response evaluated at f_ref + n * df. A wideband column already
  // carries the whole subcarrier phase at the frozen delay, so in that mode the
  // subcarrier term multiplies the DRIFT rather than the full delay - indexing
  // alone would count the n * df tau_rt phase twice and put every tap at twice
  // its delay. See the kernel header.
  //
  // carrier_hz / carrier_rate_hz are the same two carrier homes as in the beat
  // family and exactly one may be nonzero. Dropping the rate term leaves only
  // the n * df slow-time phase and understates Doppler by f_ref / (n * df) -
  // a factor of 1e4 at the top of a 64 x 120 kHz band at 77 GHz, and infinite
  // at n = 0. See R-ADR-004.
  m.def(
      "ofdm_cfr_forward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor(a!) out_re, "
      "Tensor(b!) out_im, int num_paths, int num_segments, int num_symbols, "
      "int num_subcarriers, int weight_columns, float subcarrier_spacing_hz, "
      "float symbol_period_s, float carrier_hz, float carrier_rate_hz) -> ()");
  m.def(
      "ofdm_cfr_backward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_segment, Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_tau_rt, Tensor(b!) grad_tau_rate, "
      "Tensor(c!) grad_weight_re, Tensor(d!) grad_weight_im, int num_paths, "
      "int num_segments, int num_symbols, int num_subcarriers, "
      "int weight_columns, float subcarrier_spacing_hz, float symbol_period_s, "
      "float carrier_hz, float carrier_rate_hz) -> ()");
  m.def(
      "ofdm_cfr_jvp(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor tan_tau_rt, "
      "Tensor tan_tau_rate, Tensor tan_weight_re, Tensor tan_weight_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, int num_paths, "
      "int num_segments, int num_symbols, int num_subcarriers, "
      "int weight_columns, float subcarrier_spacing_hz, float symbol_period_s, "
      "float carrier_hz, float carrier_rate_hz) -> ()");

  // Phase-6 pulsed echo train over the (pulse, fast-time sample) grid. What
  // this family emits is the matched-filter INPUT: the received complex
  // baseband pulse train. The matched filter itself is a correlation and lives
  // in DSP glue, because synthesis owns the received waveform and processing
  // owns the filter.
  //
  // The pulse is evaluated at the exact FRACTIONAL delay from its analytic
  // form, never snapped to a sample. `pulse_kind` selects between the two
  // analytic unit-energy shapes - 0 rectangular, 1 linear FM - and
  // `pulse_amplitude` is 1/sqrt(pulse_width_s), passed in so the unit-ENERGY
  // normalisation lives on the Python spec rather than inside the kernel. That
  // normalisation is what makes the matched-filter peak exactly C_rt with no
  // sample-count factor.
  //
  // The train is published in the CHANNEL phasor convention exp(-j k d), like
  // the OFDM CFR cube and unlike the conjugated FMCW beat cube: there is no
  // de-chirping here, so there is nothing to conjugate.
  //
  // carrier_hz / carrier_rate_hz are the same two carrier homes as in the other
  // two families and exactly one may be nonzero. Dropping the rate term is
  // worse here than anywhere else: the envelope carries no carrier, so what
  // survives is only the LFM's own phase moving with the drifting envelope
  // position, and for a rectangular pulse or at the pulse's leading edge that
  // is EXACTLY ZERO - the Doppler disappears completely. See R-ADR-004.
  m.def(
      "pulsed_echo_forward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor(a!) out_re, "
      "Tensor(b!) out_im, int num_paths, int num_segments, int num_pulses, "
      "int num_samples, float sample_period_s, float pri_s, "
      "float range_gate_start_s, int pulse_kind, float pulse_width_s, "
      "float bandwidth_hz, float pulse_amplitude, float carrier_hz, "
      "float carrier_rate_hz) -> ()");
  m.def(
      "pulsed_echo_backward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_segment, Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_tau_rt, Tensor(b!) grad_tau_rate, "
      "Tensor(c!) grad_weight_re, Tensor(d!) grad_weight_im, int num_paths, "
      "int num_segments, int num_pulses, int num_samples, "
      "float sample_period_s, float pri_s, float range_gate_start_s, "
      "int pulse_kind, float pulse_width_s, float bandwidth_hz, "
      "float pulse_amplitude, float carrier_hz, float carrier_rate_hz) -> ()");
  m.def(
      "pulsed_echo_jvp(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor tan_tau_rt, "
      "Tensor tan_tau_rate, Tensor tan_weight_re, Tensor tan_weight_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, int num_paths, "
      "int num_segments, int num_pulses, int num_samples, "
      "float sample_period_s, float pri_s, float range_gate_start_s, "
      "int pulse_kind, float pulse_width_s, float bandwidth_hz, "
      "float pulse_amplitude, float carrier_hz, float carrier_rate_hz) -> ()");

  // Phase-6 sensor weight: array geometry, antenna pattern, transmit power,
  // and the legacy receive projection, applied to a path weight exactly once
  // each. This is the native owner of what `solvers/common.py` computes in
  // Torch today.
  //
  // THE THREE MODE FLAGS ARE THE SINGLE-COUNT RULE, AS ARGUMENTS. They are
  // driven directly by the batch's provenance booleans:
  //
  //   spreading_mode           1 applies wavelength/(4 pi L), 0 applies nothing.
  //   tx_power_mode            1 applies tx_amplitude, 0 applies nothing.
  //   legacy_real_polarization 1 applies the mirrored TX-onto-RX projection.
  //
  // A Channel-sourced weight already carries free-space spreading per leg,
  // sqrt(tx_power) from the source endpoint's powers_w, and the endpoint
  // polarization projection, so it arrives with all three set to 0 and CANNOT
  // have any of them applied a second time. The legacy real-amplitude route
  // carries none of them, passes 1 for all three, and reproduces the Torch
  // expression it replaces.
  //
  // `row_kind` is 0 for a row that interacts at a site and 1 for a direct
  // transmitter-to-receiver row, whose length is |rx - tx| with no site term.
  //
  // Outputs are the scaled complex weight, the ROUND-TRIP delay tau_rt in
  // seconds, its rate d(tau_rt)/dt, and the POWER pattern product G_t * G_r as
  // a diagnostic. The differentiable inputs are the four endpoint positions,
  // the intensity, and the incoming weight; the velocities, normals,
  // polarization vectors, local frame, and pattern table are constants.
  //
  // The backward operator takes two per-row scratch tensors and uses NO
  // atomics. Many rows share one antenna, so the antenna-position gradient is a
  // real reduction; doing it in a second kernel over ascending rows makes the
  // summation order a property of the frozen row set rather than of the
  // schedule, exactly as the two-way join does. See R-ADR-004.
  m.def(
      "sensor_weight_forward(Tensor tx_pos, Tensor rx_pos, Tensor tx_velocity, "
      "Tensor rx_velocity, Tensor site_in, Tensor site_out, "
      "Tensor site_velocity, Tensor fixed_length_m, Tensor tx_index, "
      "Tensor rx_index, Tensor row_kind, Tensor intensity, Tensor weight_in_re, "
      "Tensor weight_in_im, Tensor normals, Tensor pol_tx, Tensor pol_rx, "
      "Tensor local_axes, Tensor pattern_x_axis, Tensor pattern_y_axis, "
      "Tensor pattern_x_values, Tensor pattern_y_values, Tensor pattern_values, "
      "Tensor(a!) out_re, Tensor(b!) out_im, Tensor(c!) tau_rt, "
      "Tensor(d!) tau_rate, Tensor(e!) pattern_gain, int num_paths, int num_tx, "
      "int num_rx, int pattern_kind, float c0, float wavelength_m, "
      "float tx_amplitude, int spreading_mode, int tx_power_mode, "
      "int legacy_real_polarization, int reflection_flip) -> ()");
  m.def(
      "sensor_weight_backward(Tensor tx_pos, Tensor rx_pos, Tensor tx_velocity, "
      "Tensor rx_velocity, Tensor site_in, Tensor site_out, "
      "Tensor site_velocity, Tensor fixed_length_m, Tensor tx_index, "
      "Tensor rx_index, Tensor row_kind, Tensor intensity, Tensor weight_in_re, "
      "Tensor weight_in_im, Tensor normals, Tensor pol_tx, Tensor pol_rx, "
      "Tensor local_axes, Tensor pattern_x_axis, Tensor pattern_y_axis, "
      "Tensor pattern_x_values, Tensor pattern_y_values, Tensor pattern_values, "
      "Tensor grad_out_re, Tensor grad_out_im, Tensor grad_tau_rt, "
      "Tensor grad_tau_rate, Tensor(a!) grad_tx_pos, Tensor(b!) grad_rx_pos, "
      "Tensor(c!) grad_site_in, Tensor(d!) grad_site_out, "
      "Tensor(e!) grad_intensity, Tensor(f!) grad_weight_re, "
      "Tensor(g!) grad_weight_im, Tensor(h!) tx_row_scratch, "
      "Tensor(i!) rx_row_scratch, int num_paths, int num_tx, int num_rx, "
      "int pattern_kind, float c0, float wavelength_m, float tx_amplitude, "
      "int spreading_mode, int tx_power_mode, int legacy_real_polarization, "
      "int reflection_flip) -> ()");
  m.def(
      "sensor_weight_jvp(Tensor tx_pos, Tensor rx_pos, Tensor tx_velocity, "
      "Tensor rx_velocity, Tensor site_in, Tensor site_out, "
      "Tensor site_velocity, Tensor fixed_length_m, Tensor tx_index, "
      "Tensor rx_index, Tensor row_kind, Tensor intensity, Tensor weight_in_re, "
      "Tensor weight_in_im, Tensor normals, Tensor pol_tx, Tensor pol_rx, "
      "Tensor local_axes, Tensor pattern_x_axis, Tensor pattern_y_axis, "
      "Tensor pattern_x_values, Tensor pattern_y_values, Tensor pattern_values, "
      "Tensor tan_tx_pos, Tensor tan_rx_pos, Tensor tan_site_in, "
      "Tensor tan_site_out, Tensor tan_intensity, Tensor tan_weight_re, "
      "Tensor tan_weight_im, Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, "
      "Tensor(c!) tan_tau_rt, Tensor(d!) tan_tau_rate, int num_paths, "
      "int num_tx, int num_rx, int pattern_kind, float c0, float wavelength_m, "
      "float tx_amplitude, int spreading_mode, int tx_power_mode, "
      "int legacy_real_polarization, int reflection_flip) -> ()");

  // Phase-6 receiver frontend. Three families, ONE fixed order, and the order
  // lives in the Python runtime rather than in a caller:
  //
  //   port -> phase -> thermal -> lna -> agc -> adc
  //
  // `frontend_noise` fuses stages 1 to 3 deliberately. Two independently
  // callable runtimes let the caller decide whether thermal noise lands before
  // or after the LNA, and that decision is worth a factor of g_lna^2 in output
  // noise power. Thermal noise is INPUT-REFERRED: it is added before the gain.
  //
  //   y = ( x * exp(j theta) + n ) * g_lna,   n ~ CN(0, 2 thermal_sigma^2)
  //
  // The draws are counter-based Philox keyed by seed_base and countered by
  // (stage_id, linear element index), so the realisation is independent of the
  // block size and toggling one stage leaves every other stage bit-identical. A
  // per-thread curand state would make both of those false. `block_size` is an
  // argument precisely so a test can prove the independence.
  //
  // The accumulated Wiener phase is PUBLISHED. backward and jvp consume that
  // saved phase rather than regenerating it, which is what makes the derivative
  // exactly consistent with the realisation it was taken at.
  //
  // `frontend_agc` is data-dependent, so the frontend is not linear in the
  // signal and the physics linearity invariant holds only with AGC off. The
  // gain and the measured RMS are device tensors; reading either to the host
  // would be a per-frame transfer. The signal is viewed as
  // [dim0, num_groups, dim2] so a global AGC and a per-receiver one differ only
  // in the group count, with no copy and no permute.
  //
  // `frontend_quantize_forward` has NO backward and NO jvp, on purpose: `round`
  // is not differentiable and a straight-through surrogate is a Phase-9
  // modelling decision. Its Python owner raises on a grad-enabled or
  // forward-dual input rather than silently detaching. This is the one
  // deliberate exception to R-ADR-004's three-per-family rule. See R-ADR-004.
  m.def(
      "frontend_noise_forward(Tensor x_re, Tensor x_im, Tensor(a!) out_re, "
      "Tensor(b!) out_im, Tensor(c!) phase_rad, int num_outer, int num_phase, "
      "float phase_sigma, float thermal_sigma, float lna_gain, int seed_base, "
      "int phase_stage_id, int thermal_stage_id, int block_size) -> ()");
  m.def(
      "frontend_noise_backward(Tensor phase_rad, Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_x_re, Tensor(b!) grad_x_im, "
      "int num_outer, int num_phase, float lna_gain, int block_size) -> ()");
  m.def(
      "frontend_noise_jvp(Tensor phase_rad, Tensor tan_x_re, Tensor tan_x_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, int num_outer, "
      "int num_phase, float lna_gain, int block_size) -> ()");
  m.def(
      "frontend_agc_forward(Tensor x_re, Tensor x_im, Tensor(a!) out_re, "
      "Tensor(b!) out_im, Tensor(c!) gain, Tensor(d!) rms, int dim0, "
      "int num_groups, int dim2, float target_rms, float min_gain, "
      "float max_gain, int block_size) -> ()");
  m.def(
      "frontend_agc_backward(Tensor x_re, Tensor x_im, Tensor gain, Tensor rms, "
      "Tensor grad_out_re, Tensor grad_out_im, Tensor(a!) grad_x_re, "
      "Tensor(b!) grad_x_im, Tensor(c!) inner, int dim0, int num_groups, "
      "int dim2, float target_rms, float min_gain, float max_gain, "
      "int block_size) -> ()");
  m.def(
      "frontend_agc_jvp(Tensor x_re, Tensor x_im, Tensor gain, Tensor rms, "
      "Tensor tan_x_re, Tensor tan_x_im, Tensor(a!) tan_out_re, "
      "Tensor(b!) tan_out_im, Tensor(c!) inner, int dim0, int num_groups, "
      "int dim2, float target_rms, float min_gain, float max_gain, "
      "int block_size) -> ()");
  m.def(
      "frontend_quantize_forward(Tensor x_re, Tensor x_im, Tensor(a!) out_re, "
      "Tensor(b!) out_im, Tensor(c!) clipped_count, int num_elements, int bits, "
      "float full_scale, int block_size) -> ()");

  // Phase-5 two-way join: inbound leg x outbound leg -> radar round trip.
  //
  //   tau_rt  = tau_in + tau_out,  rate_rt = rate_in + rate_out
  //   C_rt    = (C_out * S) * C_in
  //
  // Complex values cross as separate real and imaginary tensors, matching the
  // beat family: no complex tensor crosses the autograd boundary, so the
  // conjugate-Wirtinger convention cannot be got wrong at the seam.
  //
  // row_valid is int32 and is the sole authority on whether a composed row
  // means anything. It is the conjunction of the two legs' masks, formed at
  // the Python boundary as row selection rather than physics. A dead row's
  // payload is exactly zero in every one of the three operators.
  //
  // The backward operator consumes the frozen CSR tables so each thread owns
  // one gradient slot and needs no atomics; the summation order is therefore a
  // property of the frozen join rather than of the schedule. See R-ADR-004.
  m.def(
      "two_way_join_forward(Tensor tau_in, Tensor tau_out, Tensor rate_in, "
      "Tensor rate_out, Tensor c_in_re, Tensor c_in_im, Tensor c_out_re, "
      "Tensor c_out_im, Tensor s_re, Tensor s_im, Tensor row_valid, "
      "Tensor idx_in, Tensor idx_out, Tensor idx_s, Tensor(a!) tau_rt, "
      "Tensor(b!) rate_rt, Tensor(c!) c_rt_re, Tensor(d!) c_rt_im, "
      "int num_rows) -> ()");
  m.def(
      "two_way_join_backward(Tensor c_in_re, Tensor c_in_im, Tensor c_out_re, "
      "Tensor c_out_im, Tensor s_re, Tensor s_im, Tensor row_valid, "
      "Tensor idx_in, Tensor idx_out, Tensor idx_s, Tensor by_in_offsets, "
      "Tensor by_in_rows, Tensor by_out_offsets, Tensor by_out_rows, "
      "Tensor by_s_offsets, Tensor by_s_rows, Tensor grad_tau_rt, "
      "Tensor grad_c_rt_re, Tensor grad_c_rt_im, Tensor(a!) grad_tau_in, "
      "Tensor(b!) grad_tau_out, Tensor(c!) grad_c_in_re, "
      "Tensor(d!) grad_c_in_im, Tensor(e!) grad_c_out_re, "
      "Tensor(f!) grad_c_out_im, Tensor(g!) grad_s_re, Tensor(h!) grad_s_im, "
      "int num_rows, int num_in, int num_out, int num_sites) -> ()");
  m.def(
      "two_way_join_jvp(Tensor c_in_re, Tensor c_in_im, Tensor c_out_re, "
      "Tensor c_out_im, Tensor s_re, Tensor s_im, Tensor row_valid, "
      "Tensor idx_in, Tensor idx_out, Tensor idx_s, Tensor tan_tau_in, "
      "Tensor tan_tau_out, Tensor tan_c_in_re, Tensor tan_c_in_im, "
      "Tensor tan_c_out_re, Tensor tan_c_out_im, Tensor tan_s_re, "
      "Tensor tan_s_im, Tensor(a!) tan_tau_rt, Tensor(b!) tan_rate_rt, "
      "Tensor(c!) tan_c_rt_re, Tensor(d!) tan_c_rt_im, int num_rows) -> ()");

  // Aspect-dependent scatter response, evaluated per COMPOSED row from the
  // direction basis the two legs publish. This is what makes item 6b/6c of the
  // Phase-7 plan expressible at all: TwoWayComposer.compose refuses to
  // evaluate a geometry-dependent response in Torch, and this family is the
  // route THROUGH that refusal rather than around it.
  //
  //   ci = -dot(dir_in[i], axis[s])   incidence cosine, negated because
  //                                   dir_in propagates INTO the site
  //   co =  dot(dir_out[o], axis[s])  scattering cosine
  //   S  = amplitude[s] * clamp(ci)^n * clamp(co)^n * exp(-i phase[s])
  //
  // The output is exactly the per-row (s_re, s_im) pair the join already
  // consumes, so two_way_join is unchanged and the join adds no launch: the
  // composer hands the join an identity site index and an identity CSR when
  // the response is row evaluated.
  //
  // exponent is a host scalar and takes no gradient - it selects the law. The
  // three differentiable owner families are the inbound leg directions, the
  // outbound leg directions, and the per-site (axis, amplitude, phase). The
  // backward consumes the join's OWN frozen CSR tables, so one thread owns one
  // gradient slot, there are no atomics, and the summation order is a property
  // of the frozen composition. See R-ADR-013.
  m.def(
      "scatter_response_aspect_forward(Tensor dir_in, Tensor dir_out, "
      "Tensor idx_in, Tensor idx_out, Tensor idx_site, Tensor axis, "
      "Tensor amplitude, Tensor phase_rad, Tensor row_valid, "
      "Tensor(a!) s_re, Tensor(b!) s_im, float exponent, int num_rows) -> ()");
  m.def(
      "scatter_response_aspect_jvp(Tensor dir_in, Tensor dir_out, "
      "Tensor idx_in, Tensor idx_out, Tensor idx_site, Tensor axis, "
      "Tensor amplitude, Tensor phase_rad, Tensor row_valid, "
      "Tensor tan_dir_in, Tensor tan_dir_out, Tensor tan_axis, "
      "Tensor tan_amplitude, Tensor tan_phase_rad, Tensor(a!) tan_s_re, "
      "Tensor(b!) tan_s_im, float exponent, int num_rows) -> ()");
  m.def(
      "scatter_response_aspect_backward(Tensor dir_in, Tensor dir_out, "
      "Tensor idx_in, Tensor idx_out, Tensor idx_site, Tensor axis, "
      "Tensor amplitude, Tensor phase_rad, Tensor row_valid, "
      "Tensor by_in_offsets, Tensor by_in_rows, Tensor by_out_offsets, "
      "Tensor by_out_rows, Tensor by_site_offsets, Tensor by_site_rows, "
      "Tensor grad_s_re, Tensor grad_s_im, Tensor(a!) grad_dir_in, "
      "Tensor(b!) grad_dir_out, Tensor(c!) grad_axis, "
      "Tensor(d!) grad_amplitude, Tensor(e!) grad_phase_rad, float exponent, "
      "int num_rows, int num_in, int num_out, int num_sites) -> ()");
}
