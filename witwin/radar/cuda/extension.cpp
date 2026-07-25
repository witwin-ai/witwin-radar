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

  // Phase-4 FMCW beat synthesis over a chirp's fast-time axis. `carrier_hz` is
  // explicit: set it to the carrier to reproduce the Dirichlet path's phase
  // structure exactly, or to 0 when the weight already carries the carrier
  // phase (which is the case for a natively computed Channel coefficient).
  // Both settings are exact; neither is a fallback. See R-ADR-004.
  m.def(
      "fmcw_beat_forward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor(a!) out_re, "
      "Tensor(b!) out_im, int num_paths, int num_segments, int num_chirps, "
      "int num_samples, float sample_period_s, float chirp_period_s, "
      "float slope_hz_per_s, float carrier_hz, float t_start_s) -> ()");
  m.def(
      "fmcw_beat_backward(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_segment, Tensor grad_out_re, "
      "Tensor grad_out_im, Tensor(a!) grad_tau_rt, Tensor(b!) grad_tau_rate, "
      "Tensor(c!) grad_weight_re, Tensor(d!) grad_weight_im, int num_paths, "
      "int num_segments, int num_chirps, int num_samples, "
      "float sample_period_s, float chirp_period_s, float slope_hz_per_s, "
      "float carrier_hz, float t_start_s) -> ()");
  m.def(
      "fmcw_beat_jvp(Tensor tau_rt, Tensor tau_rate, Tensor weight_re, "
      "Tensor weight_im, Tensor path_offsets, Tensor tan_tau_rt, "
      "Tensor tan_tau_rate, Tensor tan_weight_re, Tensor tan_weight_im, "
      "Tensor(a!) tan_out_re, Tensor(b!) tan_out_im, int num_paths, "
      "int num_segments, int num_chirps, int num_samples, "
      "float sample_period_s, float chirp_period_s, float slope_hz_per_s, "
      "float carrier_hz, float t_start_s) -> ()");
}
