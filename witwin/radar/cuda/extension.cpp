#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(_radar_native, m) {
  // Back-projection imaging (processing_imaging.cu): linear in the samples, so
  // the jvp is the forward map of the tangent and the backward its adjoint.
  m.def("backprojection_forward(Tensor samples, Tensor tx, Tensor rx, Tensor pixels, Tensor(a!) image, int groups, float range_origin_m, float range_bin_m, float inverse_speed_s_per_m, float carrier_hz, float curvature_hz_per_s, float phase_sign, float scale) -> ()");
  m.def("backprojection_backward(Tensor grad_image, Tensor tx, Tensor rx, Tensor pixels, Tensor(a!) grad_samples, int groups, float range_origin_m, float range_bin_m, float inverse_speed_s_per_m, float carrier_hz, float curvature_hz_per_s, float phase_sign, float scale) -> ()");
  m.def("backprojection_jvp(Tensor tangent_samples, Tensor tx, Tensor rx, Tensor pixels, Tensor(a!) tangent_image, int groups, float range_origin_m, float range_bin_m, float inverse_speed_s_per_m, float carrier_hz, float curvature_hz_per_s, float phase_sign, float scale) -> ()");
  m.def("cfar_order_statistic_forward(Tensor padded, Tensor offsets, Tensor rank, Tensor(a!) level, int window_height, int window_width) -> ()");
  m.def("dbscan_labels_forward(Tensor features, Tensor(a!) parent, Tensor(b!) core, float eps, int min_points) -> ()");
  m.def("fmcw_adaptive_forward(Tensor samples, Tensor valid, Tensor starts, Tensor basis, Tensor bounds, Tensor owners, Tensor clock, Tensor vector, Tensor(a!) output, int begin, int end, int num_tx, int num_samples, float interpolation_carrier, float slope, float carrier, int precise_delay) -> ()");
  m.def("fmcw_adaptive_backward(Tensor samples, Tensor valid, Tensor starts, Tensor basis, Tensor bounds, Tensor owners, Tensor clock, Tensor vector, Tensor(a!) output, int begin, int end, int num_tx, int num_samples, float interpolation_carrier, float slope, float carrier, int precise_delay) -> ()");
  m.def("fmcw_adaptive_jvp(Tensor samples, Tensor valid, Tensor starts, Tensor basis, Tensor bounds, Tensor owners, Tensor clock, Tensor vector, Tensor(a!) output, int begin, int end, int num_tx, int num_samples, float interpolation_carrier, float slope, float carrier, int precise_delay) -> ()");
  m.def("fmcw_observation_forward(Tensor x, Tensor offsets, Tensor segment, Tensor vector, Tensor(a!) output, float slope, float carrier) -> ()");
  m.def("fmcw_observation_backward(Tensor x, Tensor offsets, Tensor segment, Tensor vector, Tensor(a!) output, float slope, float carrier) -> ()");
  m.def("fmcw_observation_jvp(Tensor x, Tensor offsets, Tensor segment, Tensor vector, Tensor(a!) output, float slope, float carrier) -> ()");
  // Path births and deaths between consecutive observations; discrete, so it
  // has no derivative companion (see path_events.cu).
  m.def("path_events_forward(Tensor prev_valid, Tensor prev_transfer, Tensor cur_valid, Tensor cur_transfer, Tensor before, Tensor after, Tensor(a!) event_code, Tensor(b!) event_amplitude, Tensor(c!) counter, int base) -> ()");
  m.def("path_interpolate_forward(Tensor x, Tensor indices, Tensor weights, Tensor vector, Tensor(a!) output, float carrier) -> ()");
  m.def("path_interpolate_backward(Tensor x, Tensor indices, Tensor weights, Tensor vector, Tensor(a!) output, float carrier) -> ()");
  m.def("path_interpolate_jvp(Tensor x, Tensor indices, Tensor weights, Tensor vector, Tensor(a!) output, float carrier) -> ()");
  // Physical-optics facets (facet_scatter.cu). facet_geometry gathers one
  // frame's facet corners and places each site at the mean of its member
  // facets' centroids; it is linear, so its JVP is itself on the tangent.
  // facet_response is one complex factor per composed row: the coherent sum of
  // the site's facets' PO terms with the material's co-polar reflection, the
  // coherent roughness factor and a Lambertian diffuse term. The material table,
  // wavenumber, polarization and seed take no derivative.
  m.def("facet_geometry_forward(Tensor vertices, Tensor faces, Tensor site_offsets, Tensor site_facets, Tensor(a!) corners, Tensor(b!) sites) -> ()");
  m.def("facet_geometry_jvp(Tensor tan_vertices, Tensor faces, Tensor site_offsets, Tensor site_facets, Tensor(a!) tan_corners, Tensor(b!) tan_sites) -> ()");
  m.def("facet_geometry_backward(Tensor vertex_offsets, Tensor vertex_entries, Tensor facet_site, Tensor site_offsets, Tensor grad_corners, Tensor grad_sites, Tensor(a!) grad_vertices) -> ()");
  m.def("facet_response_forward(Tensor dir_in, Tensor dir_out, Tensor idx_in, Tensor idx_out, Tensor idx_site, Tensor corners, Tensor sites, Tensor site_offsets, Tensor site_facets, Tensor facet_material, Tensor material_params, Tensor layer_offsets, Tensor layers, Tensor row_valid, Tensor(a!) s_re, Tensor(b!) s_im, float wavenumber, float pol_x, float pol_y, float pol_z, int seed, int num_rows) -> ()");
  m.def("facet_response_jvp(Tensor dir_in, Tensor dir_out, Tensor idx_in, Tensor idx_out, Tensor idx_site, Tensor corners, Tensor sites, Tensor site_offsets, Tensor site_facets, Tensor facet_material, Tensor material_params, Tensor layer_offsets, Tensor layers, Tensor row_valid, Tensor tan_dir_in, Tensor tan_dir_out, Tensor tan_corners, Tensor tan_sites, Tensor(a!) tan_s_re, Tensor(b!) tan_s_im, float wavenumber, float pol_x, float pol_y, float pol_z, int seed, int num_rows) -> ()");
  m.def("facet_response_backward(Tensor dir_in, Tensor dir_out, Tensor idx_in, Tensor idx_out, Tensor idx_site, Tensor corners, Tensor sites, Tensor site_offsets, Tensor site_facets, Tensor facet_material, Tensor material_params, Tensor layer_offsets, Tensor layers, Tensor row_valid, Tensor by_in_offsets, Tensor by_in_rows, Tensor by_out_offsets, Tensor by_out_rows, Tensor by_site_offsets, Tensor by_site_rows, Tensor facet_site, Tensor grad_s_re, Tensor grad_s_im, Tensor(a!) grad_dir_in, Tensor(b!) grad_dir_out, Tensor(c!) grad_corners, Tensor(d!) grad_sites, float wavenumber, float pol_x, float pol_y, float pol_z, int seed, int num_rows) -> ()");
  // Tabulated response (scatter_response.cu): S(theta) interpolated per
  // composed row at the bistatic aspect angle between the site axis and the
  // bisector of the leg directions; the table takes no derivative. And the
  // Swerling amplitude factor, one per site, keyed on the host; no input of it
  // is differentiable.
  m.def("scatter_response_table_forward(Tensor dir_in, Tensor dir_out, Tensor idx_in, Tensor idx_out, Tensor idx_site, Tensor axis, Tensor table_re, Tensor table_im, Tensor row_valid, Tensor(a!) s_re, Tensor(b!) s_im, int num_rows) -> ()");
  m.def("scatter_response_table_jvp(Tensor dir_in, Tensor dir_out, Tensor idx_in, Tensor idx_out, Tensor idx_site, Tensor axis, Tensor table_re, Tensor table_im, Tensor row_valid, Tensor tan_dir_in, Tensor tan_dir_out, Tensor tan_axis, Tensor(a!) tan_s_re, Tensor(b!) tan_s_im, int num_rows) -> ()");
  m.def("scatter_response_table_backward(Tensor dir_in, Tensor dir_out, Tensor idx_in, Tensor idx_out, Tensor idx_site, Tensor axis, Tensor table_re, Tensor table_im, Tensor row_valid, Tensor by_in_offsets, Tensor by_in_rows, Tensor by_out_offsets, Tensor by_out_rows, Tensor by_site_offsets, Tensor by_site_rows, Tensor grad_s_re, Tensor grad_s_im, Tensor(a!) grad_dir_in, Tensor(b!) grad_dir_out, Tensor(c!) grad_axis, int num_rows) -> ()");
  m.def("fluctuation_factor_forward(Tensor(a!) out, int key, int degrees) -> ()");
  m.def("scatter_direction_forward(Tensor origin, Tensor target, Tensor(a!) out) -> ()");
  m.def("scatter_direction_backward(Tensor origin, Tensor target, Tensor cotangent, Tensor(a!) out) -> ()");
  m.def("scatter_direction_jvp(Tensor origin, Tensor target, Tensor tangent, Tensor(a!) out) -> ()");
  // Direct normalized Dirichlet range spectrum; same path and TDM contract as
  // the beat family below. It takes no delay rate: a delay that walks has no
  // closed form here, and the cheap route to its spectrum is the beat family
  // plus the processing range transform.
  m.def(
      "fmcw_spectrum_forward(Tensor tau_rt, Tensor tau_lo, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor segment_tx_index, "
      "Tensor(a!) out_re, "
      "Tensor(b!) out_im, int num_paths, int num_segments, int num_tx, "
      "int num_chirps, "
      "int num_bins, float sample_period_s, float chirp_period_s, "
      "float slope_hz_per_s, float carrier_hz, float carrier_rate_hz, "
      "float t_start_s) -> ()");
  m.def(
      "fmcw_spectrum_backward(Tensor tau_rt, Tensor tau_lo, Tensor weight_re, "
      "Tensor weight_im, Tensor path_segment, Tensor segment_tx_index, "
      "Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_tau_rt, "
      "Tensor(b!) grad_weight_re, Tensor(c!) grad_weight_im, int num_paths, "
      "int num_segments, int num_tx, int num_chirps, int num_bins, "
      "float sample_period_s, float chirp_period_s, float slope_hz_per_s, "
      "float carrier_hz, float carrier_rate_hz, float t_start_s) -> ()");
  m.def(
      "fmcw_spectrum_jvp(Tensor tau_rt, Tensor tau_lo, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor segment_tx_index, "
      "Tensor tan_tau_rt, "
      "Tensor tan_weight_re, Tensor tan_weight_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, int num_paths, "
      "int num_segments, int num_tx, int num_chirps, int num_bins, "
      "float sample_period_s, float chirp_period_s, float slope_hz_per_s, "
      "float carrier_hz, float carrier_rate_hz, float t_start_s) -> ()");

  // FMCW beat synthesis over a chirp's fast-time axis. The carrier has two
  // homes and exactly one of the two parameters names it:
  //
  //   carrier_hz = fc, carrier_rate_hz = 0   the kernel owns the whole carrier
  //     phase. This is the absolute-carrier form; a weight carrying no
  //     reference phase needs it.
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
  // reduces the slot to the chirp index exactly. Three float64 tables, each
  // empty for the uniform ramp (R-ADR-047): `slot_table` [slots, 4] gives each
  // TDM slot its own ramp start, slope, start offset and ADC start (a chirp
  // sequence); `segment_table` [segments, 3] gives each segment its own ADC
  // window start, slope and start offset, so one launch covers many ADC
  // instants of a refreshed frame; `chirp_poly` holds the transmitted ramp's
  // polynomial phase error.
  m.def(
      "fmcw_beat_forward(Tensor tau_rt, Tensor tau_lo, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor segment_tx_index, "
      "Tensor slot_table, Tensor segment_table, Tensor chirp_poly, Tensor(a!) out_re, "
      "Tensor(b!) out_im, int num_paths, int num_segments, int num_tx, "
      "int num_chirps, "
      "int num_samples, float sample_period_s, float chirp_period_s, "
      "float slope_hz_per_s, float carrier_hz, float carrier_rate_hz, "
      "float t_start_s) -> ()");
  m.def(
      "fmcw_beat_backward(Tensor tau_rt, Tensor tau_lo, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_segment, Tensor segment_tx_index, "
      "Tensor slot_table, Tensor segment_table, Tensor chirp_poly, Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_tau_rt, Tensor(b!) grad_tau_rate, "
      "Tensor(c!) grad_weight_re, Tensor(d!) grad_weight_im, int num_paths, "
      "int num_segments, int num_tx, int num_chirps, int num_samples, "
      "float sample_period_s, float chirp_period_s, float slope_hz_per_s, "
      "float carrier_hz, float carrier_rate_hz, float t_start_s) -> ()");
  m.def(
      "fmcw_beat_jvp(Tensor tau_rt, Tensor tau_lo, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor segment_tx_index, "
      "Tensor slot_table, Tensor segment_table, Tensor chirp_poly, Tensor tan_tau_rt, "
      "Tensor tan_tau_rate, Tensor tan_weight_re, Tensor tan_weight_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, int num_paths, "
      "int num_segments, int num_tx, int num_chirps, int num_samples, "
      "float sample_period_s, float chirp_period_s, float slope_hz_per_s, "
      "float carrier_hz, float carrier_rate_hz, float t_start_s) -> ()");

  // OFDM channel frequency response over the (symbol, subcarrier)
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
      "ofdm_cfr_forward(Tensor tau_rt, Tensor tau_lo, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor(a!) out_re, "
      "Tensor(b!) out_im, int num_paths, int num_segments, int num_symbols, "
      "int num_subcarriers, int weight_columns, float subcarrier_spacing_hz, "
      "float symbol_period_s, float carrier_hz, float carrier_rate_hz) -> ()");
  m.def(
      "ofdm_cfr_backward(Tensor tau_rt, Tensor tau_lo, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_segment, Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_tau_rt, Tensor(b!) grad_tau_rate, "
      "Tensor(c!) grad_weight_re, Tensor(d!) grad_weight_im, int num_paths, "
      "int num_segments, int num_symbols, int num_subcarriers, "
      "int weight_columns, float subcarrier_spacing_hz, float symbol_period_s, "
      "float carrier_hz, float carrier_rate_hz) -> ()");
  m.def(
      "ofdm_cfr_jvp(Tensor tau_rt, Tensor tau_lo, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor tan_tau_rt, "
      "Tensor tan_tau_rate, Tensor tan_weight_re, Tensor tan_weight_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, int num_paths, "
      "int num_segments, int num_symbols, int num_subcarriers, "
      "int weight_columns, float subcarrier_spacing_hz, float symbol_period_s, "
      "float carrier_hz, float carrier_rate_hz) -> ()");

  // Time-domain OFDM: the received post-prefix samples of a modulated symbol
  // stream (`symbols`, float32 [table symbols, N, 2]), so intra-symbol motion
  // produces inter-carrier interference and a long echo inter-symbol
  // interference. `segment_table` (float64, empty or [segments, 2]: window
  // start within the symbol s, symbol index) places each segment on its own
  // instant. The backward writes per-(path, symbol) partial sums the caller
  // reduces. R-ADR-046.
  m.def(
      "ofdm_time_forward(Tensor tau_rt, Tensor tau_lo, Tensor tau_rate, Tensor weight_re, Tensor weight_im, Tensor path_offsets, "
      "Tensor symbols, Tensor segment_table, Tensor(a!) out_re, Tensor(b!) out_im, int num_paths, "
      "int num_segments, int num_symbols, int num_subcarriers, int num_samples, float subcarrier_spacing_hz, "
      "float cyclic_prefix_s, float carrier_hz, float carrier_rate_hz) -> ()");
  m.def(
      "ofdm_time_backward(Tensor tau_rt, Tensor tau_lo, Tensor tau_rate, Tensor weight_re, Tensor weight_im, Tensor path_segment, "
      "Tensor symbols, Tensor segment_table, Tensor grad_re, Tensor grad_im, Tensor(a!) partial, int num_paths, "
      "int num_segments, int num_symbols, int num_subcarriers, int num_samples, float subcarrier_spacing_hz, "
      "float cyclic_prefix_s, float carrier_hz, float carrier_rate_hz) -> ()");
  m.def(
      "ofdm_time_jvp(Tensor tau_rt, Tensor tau_lo, Tensor tau_rate, Tensor weight_re, Tensor weight_im, Tensor path_offsets, "
      "Tensor symbols, Tensor segment_table, Tensor tan_tau_rt, Tensor tan_tau_rate, Tensor tan_weight_re, "
      "Tensor tan_weight_im, Tensor(a!) out_re, Tensor(b!) out_im, int num_paths, int num_segments, "
      "int num_symbols, int num_subcarriers, int num_samples, float subcarrier_spacing_hz, "
      "float cyclic_prefix_s, float carrier_hz, float carrier_rate_hz) -> ()");

  // Pulsed echo train over the (pulse, fast-time sample) grid: the received
  // complex baseband, i.e. the matched-filter INPUT, in the CHANNEL phasor
  // convention. `pulse_kind` is 0 rect, 1 LFM, 2 coded; a coded pulse reads
  // `chips` (float64 [num_chips, 3]: |a|, arg(a) / 2pi, chip frequency Hz) and
  // every other kind takes an empty table. `periodic` extends one code period
  // periodically (a continuous code). `frequency_step_hz` is the stepped
  // carrier offset of pulse l. `segment_table` (float64, empty or
  // [num_segments, 2]: gate start s, carrier offset Hz) replaces the uniform
  // gate start and the stepped offset per segment, so one launch covers many
  // fast-time instants of a refreshed frame. The two carrier homes follow
  // R-ADR-004; the coded and stepped families are R-ADR-046.
  m.def(
      "pulsed_echo_forward(Tensor tau_rt, Tensor tau_lo, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor chips, Tensor segment_table, "
      "Tensor(a!) out_re, Tensor(b!) out_im, int num_paths, int num_segments, int num_pulses, "
      "int num_samples, float sample_period_s, float pri_s, "
      "float range_gate_start_s, int pulse_kind, float pulse_width_s, "
      "float bandwidth_hz, float pulse_amplitude, float carrier_hz, "
      "float carrier_rate_hz, float frequency_step_hz, int periodic) -> ()");
  m.def(
      "pulsed_echo_backward(Tensor tau_rt, Tensor tau_lo, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_segment, Tensor chips, Tensor segment_table, Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_tau_rt, Tensor(b!) grad_tau_rate, "
      "Tensor(c!) grad_weight_re, Tensor(d!) grad_weight_im, int num_paths, "
      "int num_segments, int num_pulses, int num_samples, "
      "float sample_period_s, float pri_s, float range_gate_start_s, "
      "int pulse_kind, float pulse_width_s, float bandwidth_hz, "
      "float pulse_amplitude, float carrier_hz, float carrier_rate_hz, "
      "float frequency_step_hz, int periodic) -> ()");
  m.def(
      "pulsed_echo_jvp(Tensor tau_rt, Tensor tau_lo, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor chips, Tensor segment_table, Tensor tan_tau_rt, "
      "Tensor tan_tau_rate, Tensor tan_weight_re, Tensor tan_weight_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, int num_paths, "
      "int num_segments, int num_pulses, int num_samples, "
      "float sample_period_s, float pri_s, float range_gate_start_s, "
      "int pulse_kind, float pulse_width_s, float bandwidth_hz, "
      "float pulse_amplitude, float carrier_hz, float carrier_rate_hz, "
      "float frequency_step_hz, int periodic) -> ()");

  // Sensor pattern weighting. Channel already owns spreading, transmit power,
  // reference phase, and polarization projection. These operators therefore
  // take only geometry, the incoming complex transfer, a world-to-pattern
  // frame, and the two packed element patterns (transmit and receive, each
  // with its own peak gain; power or complex field).
  m.def(
      "sensor_weight_forward(Tensor tx_pos, Tensor rx_pos, Tensor tx_velocity, "
      "Tensor rx_velocity, Tensor site_in, Tensor site_out, "
      "Tensor site_velocity, Tensor fixed_length_m, Tensor tx_index, "
      "Tensor rx_index, Tensor row_kind, Tensor intensity, Tensor weight_in_re, "
      "Tensor weight_in_im, Tensor pattern_frame, Tensor tx_pattern, "
      "Tensor rx_pattern, Tensor(a!) out_re, Tensor(b!) out_im, "
      "Tensor(c!) tau_rt, Tensor(d!) tau_rate, Tensor(e!) pattern_gain, "
      "int num_paths, int num_tx, int num_rx, float c0, int slots) -> ()");
  m.def(
      "sensor_weight_backward(Tensor tx_pos, Tensor rx_pos, Tensor tx_velocity, "
      "Tensor rx_velocity, Tensor site_in, Tensor site_out, "
      "Tensor site_velocity, Tensor fixed_length_m, Tensor tx_index, "
      "Tensor rx_index, Tensor row_kind, Tensor intensity, Tensor weight_in_re, "
      "Tensor weight_in_im, Tensor pattern_frame, Tensor tx_pattern, "
      "Tensor rx_pattern, Tensor grad_out_re, Tensor grad_out_im, "
      "Tensor grad_tau_rt, Tensor grad_tau_rate, Tensor(a!) grad_tx_pos, "
      "Tensor(b!) grad_rx_pos, Tensor(c!) grad_site_in, Tensor(d!) grad_site_out, "
      "Tensor(e!) grad_intensity, Tensor(f!) grad_weight_re, "
      "Tensor(g!) grad_weight_im, Tensor(h!) tx_row_scratch, "
      "Tensor(i!) rx_row_scratch, int num_paths, int num_tx, int num_rx, "
      "float c0, int slots) -> ()");
  m.def(
      "sensor_weight_jvp(Tensor tx_pos, Tensor rx_pos, Tensor tx_velocity, "
      "Tensor rx_velocity, Tensor site_in, Tensor site_out, "
      "Tensor site_velocity, Tensor fixed_length_m, Tensor tx_index, "
      "Tensor rx_index, Tensor row_kind, Tensor intensity, Tensor weight_in_re, "
      "Tensor weight_in_im, Tensor pattern_frame, Tensor tx_pattern, "
      "Tensor rx_pattern, Tensor tan_tx_pos, Tensor tan_rx_pos, "
      "Tensor tan_site_in, Tensor tan_site_out, Tensor tan_intensity, "
      "Tensor tan_weight_re, Tensor tan_weight_im, Tensor(a!) tan_out_re, "
      "Tensor(b!) tan_out_im, Tensor(c!) tan_tau_rt, Tensor(d!) tan_tau_rate, "
      "int num_paths, int num_tx, int num_rx, float c0, int slots) -> ()");
  // Receiver frontend. Three families, ONE fixed order, and the order
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
  // is not differentiable and a straight-through surrogate is a modelling
  // decision this file does not make. Its Python owner raises on a grad-enabled or
  // forward-dual input rather than silently detaching. This is the one
  // deliberate exception to R-ADR-004's three-per-family rule. See R-ADR-004.
  m.def(
      "oscillator_phase_forward(Tensor times, Tensor delays, Tensor(a!) phase, "
      "Tensor(b!) invalid, float diffusion, float flicker_scale, "
      "float floor_sigma, float floor_cell_rate, float sign, int seed_base, "
      "int block_size) -> ()");
  // `frontend_compress` is the memoryless receiver compression (polynomial
  // AM/AM with AM/PM, held past saturation); its backward and jvp are the
  // analytic 2x2 Jacobian. `frontend_baseband_forward` writes the
  // per-receiver DC offset plus the 1/f flicker process, which the Python owner
  // ADDS to the signal; that addition carries the identity derivative, so the
  // family ships no companion. `frontend_agc_dynamic` is the
  // per-block first-order AGC loop with its adjoint and tangent scans.
  m.def(
      "frontend_compress_forward(Tensor x_re, Tensor x_im, Tensor(a!) out_re, "
      "Tensor(b!) out_im, float scale, float c5, float k_pm, float u_sat, "
      "int block_size) -> ()");
  m.def(
      "frontend_compress_backward(Tensor x_re, Tensor x_im, Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_x_re, Tensor(b!) grad_x_im, "
      "float scale, float c5, float k_pm, float u_sat, int block_size) -> ()");
  m.def(
      "frontend_compress_jvp(Tensor x_re, Tensor x_im, Tensor tan_x_re, "
      "Tensor tan_x_im, Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, "
      "float scale, float c5, float k_pm, float u_sat, int block_size) -> ()");
  m.def(
      "frontend_baseband_forward(Tensor(a!) out_re, "
      "Tensor(b!) out_im, Tensor times, Tensor dc_re, Tensor dc_im, "
      "int num_outer, int num_run, int num_rx, float flicker_level, "
      "int levels, int seed_base, int block_size) -> ()");
  m.def(
      "frontend_agc_dynamic_forward(Tensor x_re, Tensor x_im, Tensor(a!) out_re, "
      "Tensor(b!) out_im, Tensor(c!) gain, Tensor(d!) rms, Tensor(e!) free_mask, "
      "int dim0, int num_groups, int num_blocks, int dim2, float target_rms, "
      "float min_gain, float max_gain, float beta, int block_size) -> ()");
  m.def(
      "frontend_agc_dynamic_backward(Tensor x_re, Tensor x_im, Tensor gain, "
      "Tensor rms, Tensor free_mask, Tensor grad_out_re, Tensor grad_out_im, "
      "Tensor(a!) grad_x_re, Tensor(b!) grad_x_im, Tensor(c!) scratch, "
      "int dim0, int num_groups, int num_blocks, int dim2, float beta, "
      "int block_size) -> ()");
  m.def(
      "frontend_agc_dynamic_jvp(Tensor x_re, Tensor x_im, Tensor gain, "
      "Tensor rms, Tensor free_mask, Tensor tan_x_re, Tensor tan_x_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, Tensor(c!) scratch, "
      "int dim0, int num_groups, int num_blocks, int dim2, float beta, "
      "int block_size) -> ()");
  m.def(
      "frontend_noise_forward(Tensor x_re, Tensor x_im, Tensor(a!) out_re, "
      "Tensor(b!) out_im, Tensor phase_rad, int num_outer, int num_phase, "
      "float thermal_sigma, float lna_gain, int seed_base, "
      "int thermal_stage_id, int block_size) -> ()");
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

  // Two-way join: inbound leg x outbound leg -> radar round trip.
  //
  //   V   = a_t a_r exp(j psi) H^T M E      (R-ADR-039/040/041)
  //   tau = L64 / c0, published as (tau_hi, tau_lo)
  //
  // Complex values cross as interleaved float32 (re, im) pairs, views of the
  // complex tensors taken outside the autograd Function, so no complex tensor
  // crosses the boundary and the conjugate-Wirtinger convention cannot be got
  // wrong at the seam.
  //
  // row_valid is int32 and is the sole authority on whether a composed row
  // means anything. It is the conjunction of the two legs' masks, formed at
  // the Python boundary as row selection rather than physics. A dead row's
  // payload is exactly zero in every one of the three operators.
  //
  // The backward operator consumes the frozen CSR tables so each thread owns
  // one gradient slot and needs no atomics; the summation order is therefore a
  // property of the frozen join rather than of the schedule. See R-ADR-004.
  // Inputs shared by the three join operators (see two_way_join.cu for the
  // equations): per leg, the delay, the complex3 field at the site as float32
  // [rows, 3, 2] pairs, the final-segment direction (width-4 responses only),
  // and the geometry (element, site, interaction points, depth, float32
  // length; all empty to disable the float64 correction). The response is
  // [owners, width, 2] pairs with width 1 (scalar) or 4 ([HH, HV, VH, VV]),
  // and up is the per-owner site axis of a width-4 response. ``polarization``
  // is each leg row's launch polarization, [inbound | outbound | site-less
  // source | site-less sink] rows, or empty; ``waves`` is (k64, k32) per
  // frequency column (empty: no geometry); ``place`` is one slot's composed
  // position of every site row and site-less row (empty: no site-less rows).
  m.def(
      "two_way_join_forward(" "Tensor tau_in, Tensor tau_out, Tensor field_in, Tensor field_out, "
      "Tensor k_in, Tensor k_out, Tensor src_in, Tensor sink_in, Tensor points_in, Tensor depth_in, "
      "Tensor length_in, Tensor src_out, Tensor sink_out, Tensor points_out, Tensor depth_out, "
      "Tensor length_out, Tensor response, Tensor up, Tensor polarization, Tensor field_d, "
      "Tensor src_d, Tensor sink_d, Tensor points_d, Tensor depth_d, Tensor length_d, "
      "Tensor valid_d, Tensor waves, Tensor place, Tensor row_valid, Tensor idx_in, "
      "Tensor idx_out, Tensor idx_s, "
      "Tensor(a!) tau_hi, Tensor(b!) tau_lo, Tensor(c!) value, "
      "int response_width, int slots, int slot_local_sites, int columns) -> ()");
  m.def(
      "two_way_join_backward(" "Tensor tau_in, Tensor tau_out, Tensor field_in, Tensor field_out, "
      "Tensor k_in, Tensor k_out, Tensor src_in, Tensor sink_in, Tensor points_in, Tensor depth_in, "
      "Tensor length_in, Tensor src_out, Tensor sink_out, Tensor points_out, Tensor depth_out, "
      "Tensor length_out, Tensor response, Tensor up, Tensor polarization, Tensor field_d, "
      "Tensor src_d, Tensor sink_d, Tensor points_d, Tensor depth_d, Tensor length_d, "
      "Tensor valid_d, Tensor waves, Tensor place, Tensor row_valid, Tensor idx_in, "
      "Tensor idx_out, Tensor idx_s, "
      "Tensor by_in_offsets, Tensor by_in_rows, Tensor by_out_offsets, Tensor by_out_rows, Tensor by_s_offsets, "
      "Tensor by_s_rows, Tensor grad_tau, Tensor grad_value, Tensor(a!) grad_tau_in, Tensor(b!) grad_tau_out, "
      "Tensor(c!) grad_field_in, Tensor(d!) grad_field_out, Tensor(e!) grad_k_in, Tensor(f!) grad_k_out, "
      "Tensor(g!) grad_src_in, Tensor(h!) grad_sink_in, Tensor(i!) grad_points_in, Tensor(j!) grad_length_in, "
      "Tensor(k!) grad_src_out, Tensor(l!) grad_sink_out, Tensor(m!) grad_points_out, "
      "Tensor(n!) grad_length_out, Tensor(o!) grad_response, Tensor(p!) grad_field_d, Tensor(q!) grad_src_d, "
      "Tensor(r!) grad_sink_d, Tensor(s!) grad_points_d, Tensor(t!) grad_length_d, "
      "int response_width, int slots, int slot_local_sites, int columns) -> ()");
  m.def(
      "two_way_join_jvp(" "Tensor tau_in, Tensor tau_out, Tensor field_in, Tensor field_out, "
      "Tensor k_in, Tensor k_out, Tensor src_in, Tensor sink_in, Tensor points_in, Tensor depth_in, "
      "Tensor length_in, Tensor src_out, Tensor sink_out, Tensor points_out, Tensor depth_out, "
      "Tensor length_out, Tensor response, Tensor up, Tensor polarization, Tensor field_d, "
      "Tensor src_d, Tensor sink_d, Tensor points_d, Tensor depth_d, Tensor length_d, "
      "Tensor valid_d, Tensor waves, Tensor place, Tensor row_valid, Tensor idx_in, "
      "Tensor idx_out, Tensor idx_s, "
      "Tensor tan_tau_in, Tensor tan_tau_out, Tensor tan_field_in, Tensor tan_field_out, Tensor tan_k_in, "
      "Tensor tan_k_out, Tensor tan_src_in, Tensor tan_sink_in, Tensor tan_points_in, Tensor tan_length_in, "
      "Tensor tan_src_out, Tensor tan_sink_out, Tensor tan_points_out, Tensor tan_length_out, "
      "Tensor tan_response, Tensor tan_field_d, Tensor tan_src_d, Tensor tan_sink_d, Tensor tan_points_d, "
      "Tensor tan_length_d, Tensor(a!) tan_tau, Tensor(b!) tan_value, "
      "int response_width, int slots, int slot_local_sites, int columns) -> ()");

  // Aspect-dependent scatter response, evaluated per COMPOSED row from the
  // direction basis the two legs publish. TwoWayComposer.compose refuses to
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
