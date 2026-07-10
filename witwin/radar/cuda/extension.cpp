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
}
