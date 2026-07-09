#include <torch/extension.h>

#include <vector>

bool is_available() {
  return torch::cuda::is_available();
}

void forward_chunked_cuda(
    const at::Tensor& d,
    const at::Tensor& a,
    at::Tensor output_re,
    at::Tensor output_im,
    double n,
    double k0_per_meter,
    int64_t num_bins,
    int64_t n_fft,
    int64_t num_targets,
    int64_t targets_per_chunk,
    double fc,
    double slope,
    double t_start);

void forward_mimo_linear_chunked_cuda(
    const at::Tensor& d0,
    const at::Tensor& d_rate,
    const at::Tensor& a0,
    at::Tensor output_re,
    at::Tensor output_im,
    double n,
    double k0_per_meter,
    int64_t num_bins,
    int64_t n_fft,
    int64_t targets_per_pair,
    int64_t chirp_per_frame,
    double chirp_period,
    int64_t num_tx,
    int64_t range_loss_update,
    double fc,
    double slope,
    double t_start);

void backward_cuda(
    const at::Tensor& d,
    const at::Tensor& a,
    const at::Tensor& grad_output_re,
    const at::Tensor& grad_output_im,
    at::Tensor grad_d,
    at::Tensor grad_a,
    double n,
    double k0_per_meter,
    int64_t num_bins,
    int64_t n_fft,
    int64_t num_targets,
    double fc,
    double slope,
    double t_start);

void backward_per_bin_cuda(
    const at::Tensor& d,
    const at::Tensor& a,
    const at::Tensor& grad_output_re,
    const at::Tensor& grad_output_im,
    at::Tensor grad_d,
    at::Tensor grad_a,
    double n,
    double k0_per_meter,
    int64_t num_bins,
    int64_t n_fft,
    int64_t num_targets,
    int64_t bins_per_chunk,
    double fc,
    double slope,
    double t_start);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_available", &is_available, "Return whether CUDA is available to PyTorch.");
  m.def("forward_chunked", &forward_chunked_cuda, "Dirichlet forward over target chunks.");
  m.def(
      "forward_mimo_linear_chunked",
      &forward_mimo_linear_chunked_cuda,
      "Dirichlet MIMO forward with a linear per-path range model.");
  m.def("backward", &backward_cuda, "Dirichlet backward over targets.");
  m.def("backward_per_bin", &backward_per_bin_cuda, "Dirichlet backward over bins.");
}
