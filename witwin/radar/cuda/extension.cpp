#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(_radar_native, m) {
  // Phase-4 FMCW beat synthesis over a chirp's fast-time axis. The carrier has
  // two homes and exactly one of the two parameters names it:
  //
  //   carrier_hz = fc, carrier_rate_hz = 0   the kernel owns the whole carrier
  //     phase. This is the absolute-carrier form, which the deleted Dirichlet
  //     family also used; a weight carrying no reference phase still needs it.
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
  // Direct normalized Dirichlet range spectrum; same path and TDM contract.
  m.def(
      "fmcw_spectrum_forward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor segment_tx_index, "
      "Tensor(a!) out_re, "
      "Tensor(b!) out_im, int num_paths, int num_segments, int num_tx, "
      "int num_chirps, "
      "int num_bins, float sample_period_s, float chirp_period_s, "
      "float slope_hz_per_s, float carrier_hz, float carrier_rate_hz, "
      "float t_start_s) -> ()");
  m.def(
      "fmcw_spectrum_backward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_segment, Tensor segment_tx_index, "
      "Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_tau_rt, Tensor(b!) grad_tau_rate, "
      "Tensor(c!) grad_weight_re, Tensor(d!) grad_weight_im, int num_paths, "
      "int num_segments, int num_tx, int num_chirps, int num_bins, "
      "float sample_period_s, float chirp_period_s, float slope_hz_per_s, "
      "float carrier_hz, float carrier_rate_hz, float t_start_s) -> ()");
  m.def(
      "fmcw_spectrum_jvp(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor segment_tx_index, "
      "Tensor tan_tau_rt, "
      "Tensor tan_tau_rate, Tensor tan_weight_re, Tensor tan_weight_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, int num_paths, "
      "int num_segments, int num_tx, int num_chirps, int num_bins, "
      "float sample_period_s, float chirp_period_s, float slope_hz_per_s, "
      "float carrier_hz, float carrier_rate_hz, float t_start_s) -> ()");

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

  // Sensor pattern weighting. Channel already owns spreading, transmit power,
  // reference phase, and polarization projection. These operators therefore
  // take only geometry, the incoming complex transfer, a world-to-pattern
  // frame, and resident antenna-pattern tables.
  m.def(
      "sensor_weight_forward(Tensor tx_pos, Tensor rx_pos, Tensor tx_velocity, "
      "Tensor rx_velocity, Tensor site_in, Tensor site_out, "
      "Tensor site_velocity, Tensor fixed_length_m, Tensor tx_index, "
      "Tensor rx_index, Tensor row_kind, Tensor intensity, Tensor weight_in_re, "
      "Tensor weight_in_im, Tensor pattern_frame, Tensor pattern_x_axis, "
      "Tensor pattern_y_axis, Tensor pattern_x_values, Tensor pattern_y_values, "
      "Tensor pattern_values, Tensor(a!) out_re, Tensor(b!) out_im, "
      "Tensor(c!) tau_rt, Tensor(d!) tau_rate, Tensor(e!) pattern_gain, "
      "int num_paths, int num_tx, int num_rx, int pattern_kind, float c0) -> ()");
  m.def(
      "sensor_weight_backward(Tensor tx_pos, Tensor rx_pos, Tensor tx_velocity, "
      "Tensor rx_velocity, Tensor site_in, Tensor site_out, "
      "Tensor site_velocity, Tensor fixed_length_m, Tensor tx_index, "
      "Tensor rx_index, Tensor row_kind, Tensor intensity, Tensor weight_in_re, "
      "Tensor weight_in_im, Tensor pattern_frame, Tensor pattern_x_axis, "
      "Tensor pattern_y_axis, Tensor pattern_x_values, Tensor pattern_y_values, "
      "Tensor pattern_values, Tensor grad_out_re, Tensor grad_out_im, "
      "Tensor grad_tau_rt, Tensor grad_tau_rate, Tensor(a!) grad_tx_pos, "
      "Tensor(b!) grad_rx_pos, Tensor(c!) grad_site_in, Tensor(d!) grad_site_out, "
      "Tensor(e!) grad_intensity, Tensor(f!) grad_weight_re, "
      "Tensor(g!) grad_weight_im, Tensor(h!) tx_row_scratch, "
      "Tensor(i!) rx_row_scratch, int num_paths, int num_tx, int num_rx, "
      "int pattern_kind, float c0) -> ()");
  m.def(
      "sensor_weight_jvp(Tensor tx_pos, Tensor rx_pos, Tensor tx_velocity, "
      "Tensor rx_velocity, Tensor site_in, Tensor site_out, "
      "Tensor site_velocity, Tensor fixed_length_m, Tensor tx_index, "
      "Tensor rx_index, Tensor row_kind, Tensor intensity, Tensor weight_in_re, "
      "Tensor weight_in_im, Tensor pattern_frame, Tensor pattern_x_axis, "
      "Tensor pattern_y_axis, Tensor pattern_x_values, Tensor pattern_y_values, "
      "Tensor pattern_values, Tensor tan_tx_pos, Tensor tan_rx_pos, "
      "Tensor tan_site_in, Tensor tan_site_out, Tensor tan_intensity, "
      "Tensor tan_weight_re, Tensor tan_weight_im, Tensor(a!) tan_out_re, "
      "Tensor(b!) tan_out_im, Tensor(c!) tan_tau_rt, Tensor(d!) tan_tau_rate, "
      "int num_paths, int num_tx, int num_rx, int pattern_kind, float c0) -> ()");
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
