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
  // carrier_hz / carrier_rate_hz are the same two carrier homes as in the beat
  // family and exactly one may be nonzero. Dropping the rate term leaves only
  // the n * df slow-time phase and understates Doppler by f_ref / (n * df) -
  // a factor of 1e4 at the top of a 64 x 120 kHz band at 77 GHz, and infinite
  // at n = 0. See R-ADR-004.
  m.def(
      "ofdm_cfr_forward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor(a!) out_re, "
      "Tensor(b!) out_im, int num_paths, int num_segments, int num_symbols, "
      "int num_subcarriers, float subcarrier_spacing_hz, float symbol_period_s, "
      "float carrier_hz, float carrier_rate_hz) -> ()");
  m.def(
      "ofdm_cfr_backward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_segment, Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_tau_rt, Tensor(b!) grad_tau_rate, "
      "Tensor(c!) grad_weight_re, Tensor(d!) grad_weight_im, int num_paths, "
      "int num_segments, int num_symbols, int num_subcarriers, "
      "float subcarrier_spacing_hz, float symbol_period_s, float carrier_hz, "
      "float carrier_rate_hz) -> ()");
  m.def(
      "ofdm_cfr_jvp(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor tan_tau_rt, "
      "Tensor tan_tau_rate, Tensor tan_weight_re, Tensor tan_weight_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, int num_paths, "
      "int num_segments, int num_symbols, int num_subcarriers, "
      "float subcarrier_spacing_hz, float symbol_period_s, float carrier_hz, "
      "float carrier_rate_hz) -> ()");

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
}
