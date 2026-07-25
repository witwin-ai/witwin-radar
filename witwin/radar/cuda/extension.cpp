#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(witwin_radar_dirichlet_cuda, m) {
  m.def(
      "forward_chunked(Tensor d, Tensor a, Tensor(a!) output_re, Tensor(b!) output_im, "
      "float n, float k0_per_meter, int num_bins, int n_fft, int num_targets, "
      "int targets_per_chunk, float fc, float slope, float t_start) -> ()");
  m.def(
      "forward_mimo_linear_chunked(Tensor d0, Tensor d_rate, Tensor a0, "
      "Tensor(a!) output_re, Tensor(b!) output_im, float n, float k0_per_meter, "
      "int num_bins, int n_fft, int targets_per_pair, int chirp_per_frame, "
      "float chirp_period, int num_tx, int range_loss_update, float fc, float slope, "
      "float t_start) -> ()");
  m.def(
      "backward(Tensor d, Tensor a, Tensor grad_output_re, Tensor grad_output_im, "
      "Tensor(a!) grad_d, Tensor(b!) grad_a, float n, float k0_per_meter, "
      "int num_bins, int n_fft, int num_targets, float fc, float slope, float t_start) -> ()");
  m.def(
      "backward_batched(Tensor d, Tensor a, Tensor grad_output_re, Tensor grad_output_im, "
      "Tensor(a!) grad_d, Tensor(b!) grad_a, float n, float k0_per_meter, "
      "int num_bins, int n_fft, int num_targets, int targets_per_spectrum, "
      "float fc, float slope, float t_start) -> ()");
  m.def(
      "backward_parallel_bins(Tensor d, Tensor a, Tensor grad_output_re, "
      "Tensor grad_output_im, Tensor(a!) grad_d, Tensor(b!) grad_a, float n, "
      "float k0_per_meter, int num_bins, int n_fft, int num_targets, "
      "float fc, float slope, float t_start) -> ()");
  m.def(
      "backward_per_bin(Tensor d, Tensor a, Tensor grad_output_re, Tensor grad_output_im, "
      "Tensor(a!) grad_d, Tensor(b!) grad_a, float n, float k0_per_meter, "
      "int num_bins, int n_fft, int num_targets, int bins_per_chunk, "
      "float fc, float slope, float t_start) -> ()");

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
  m.def(
      "fmcw_beat_forward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor(a!) out_re, "
      "Tensor(b!) out_im, int num_paths, int num_segments, int num_chirps, "
      "int num_samples, float sample_period_s, float chirp_period_s, "
      "float slope_hz_per_s, float carrier_hz, float carrier_rate_hz, "
      "float t_start_s) -> ()");
  m.def(
      "fmcw_beat_backward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_segment, Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_tau_rt, Tensor(b!) grad_tau_rate, "
      "Tensor(c!) grad_weight_re, Tensor(d!) grad_weight_im, int num_paths, "
      "int num_segments, int num_chirps, int num_samples, "
      "float sample_period_s, float chirp_period_s, float slope_hz_per_s, "
      "float carrier_hz, float carrier_rate_hz, float t_start_s) -> ()");
  m.def(
      "fmcw_beat_jvp(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor tan_tau_rt, "
      "Tensor tan_tau_rate, Tensor tan_weight_re, Tensor tan_weight_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, int num_paths, "
      "int num_segments, int num_chirps, int num_samples, "
      "float sample_period_s, float chirp_period_s, float slope_hz_per_s, "
      "float carrier_hz, float carrier_rate_hz, float t_start_s) -> ()");

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
