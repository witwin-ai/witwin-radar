#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
constexpr float kC0 = 299792458.0f;

struct Complex {
  float re;
  float im;
};

__device__ __forceinline__ Complex cmul(const Complex a, const Complex b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

__device__ __forceinline__ Complex cexp_f(const float phase) {
  float s;
  float c;
  sincosf(phase, &s, &c);
  return {c, s};
}

__device__ __forceinline__ float dirichlet_kernel_real(const float x, const float n) {
  constexpr float eps = 1e-7f;
  float sin_half;
  float cos_half;
  sincosf(0.5f * x, &sin_half, &cos_half);
  if (fabsf(sin_half) < eps) {
    return 2.0f * n + 1.0f;
  }

  const float nph = n + 0.5f;
  float sin_nph;
  float cos_nph;
  sincosf(nph * x, &sin_nph, &cos_nph);
  return sin_nph / sin_half;
}

__device__ __forceinline__ float dirichlet_kernel_real_grad(const float x, const float n) {
  constexpr float eps = 1e-7f;
  float sin_half;
  float cos_half;
  sincosf(0.5f * x, &sin_half, &cos_half);
  if (fabsf(sin_half) < eps) {
    return 0.0f;
  }

  const float nph = n + 0.5f;
  float sin_nph;
  float cos_nph;
  sincosf(nph * x, &sin_nph, &cos_nph);
  return (nph * cos_nph * sin_half - 0.5f * sin_nph * cos_half) / (sin_half * sin_half);
}

__device__ __forceinline__ Complex dirichlet_kernel(const float x, const float n) {
  const float scale = dirichlet_kernel_real(x, n);
  const Complex phase = cexp_f(-n * x);
  return {phase.re * scale, phase.im * scale};
}

__device__ __forceinline__ Complex dirichlet_kernel_grad(const float x, const float n) {
  const float scale = dirichlet_kernel_real(x, n);
  const float scale_grad = dirichlet_kernel_real_grad(x, n);
  const Complex bracket = {scale_grad, -n * scale};
  const Complex phase = cexp_f(-n * x);
  return cmul(bracket, phase);
}

__device__ __forceinline__ Complex path_response(
    const float distance,
    const float n,
    const float k0_per_meter,
    const int n_fft,
    const int bin,
    const float fc,
    const float slope,
    const float t_start) {
  const float tau = 2.0f * distance / kC0;
  const float phi0 = kTwoPi * (fc * tau + slope * tau * (t_start - 0.5f * tau));
  const float k0 = distance * k0_per_meter;
  const float x = kTwoPi * (static_cast<float>(bin) - k0) / static_cast<float>(n_fft);
  return cmul(dirichlet_kernel(x, n), cexp_f(phi0));
}

__global__ void forward_chunked_kernel(
    const float* __restrict__ d,
    const float* __restrict__ a,
    float* __restrict__ output_re,
    float* __restrict__ output_im,
    const float n,
    const float k0_per_meter,
    const int num_bins,
    const int n_fft,
    const int num_targets,
    const int targets_per_chunk,
    const float fc,
    const float slope,
    const float t_start) {
  const int bin = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunk_idx = blockIdx.y;
  if (bin >= num_bins) {
    return;
  }

  const int target_start = chunk_idx * targets_per_chunk;
  const int target_end = min(target_start + targets_per_chunk, num_targets);

  float sum_re = 0.0f;
  float sum_im = 0.0f;
  for (int i = target_start; i < target_end; ++i) {
    const float amp = a[i];
    if (amp == 0.0f) {
      continue;
    }
    const Complex response = path_response(d[i], n, k0_per_meter, n_fft, bin, fc, slope, t_start);
    sum_re += amp * response.re;
    sum_im += amp * response.im;
  }

  const int out_idx = chunk_idx * num_bins + bin;
  output_re[out_idx] = sum_re;
  output_im[out_idx] = sum_im;
}

__global__ void forward_mimo_linear_chunked_kernel(
    const float* __restrict__ d0,
    const float* __restrict__ d_rate,
    const float* __restrict__ a0,
    float* __restrict__ output_re,
    float* __restrict__ output_im,
    const float n,
    const float k0_per_meter,
    const int num_bins,
    const int n_fft,
    const int targets_per_pair,
    const int chirp_per_frame,
    const float chirp_period,
    const int num_tx,
    const int range_loss_update,
    const int num_pairs,
    const float fc,
    const float slope,
    const float t_start) {
  const int bin = blockIdx.x * blockDim.x + threadIdx.x;
  const int pair_id = blockIdx.y;
  const int chirp_id = blockIdx.z;
  if (bin >= num_bins || pair_id >= num_pairs || chirp_id >= chirp_per_frame) {
    return;
  }

  // TDM-MIMO: TX antennas fire sequentially, so pair (tx, rx) samples the
  // scene at slot chirp_id * num_tx + tx, one chirp_period per slot.
  const int num_rx = num_pairs / num_tx;
  const int tx_id = pair_id / num_rx;
  const float slot = static_cast<float>(chirp_id) * static_cast<float>(num_tx) + static_cast<float>(tx_id);
  const float chirp_time = slot * chirp_period;
  const int target_start = pair_id * targets_per_pair;
  float sum_re = 0.0f;
  float sum_im = 0.0f;

  for (int i = 0; i < targets_per_pair; ++i) {
    const int target_idx = target_start + i;
    const float dist0 = d0[target_idx];
    const float dist = dist0 + d_rate[target_idx] * chirp_time;
    if (dist <= 0.0f) {
      continue;
    }

    float amp = a0[target_idx];
    if (amp == 0.0f) {
      continue;
    }
    if (range_loss_update != 0) {
      amp *= dist0 / fmaxf(dist, 1e-6f);
    }

    const Complex response = path_response(dist, n, k0_per_meter, n_fft, bin, fc, slope, t_start);
    sum_re += amp * response.re;
    sum_im += amp * response.im;
  }

  const int out_idx = (chirp_id * num_pairs + pair_id) * num_bins + bin;
  output_re[out_idx] = sum_re;
  output_im[out_idx] = sum_im;
}

__global__ void backward_kernel(
    const float* __restrict__ d,
    const float* __restrict__ a,
    const float* __restrict__ grad_output_re,
    const float* __restrict__ grad_output_im,
    float* __restrict__ grad_d,
    float* __restrict__ grad_a,
    const float n,
    const float k0_per_meter,
    const int num_bins,
    const int n_fft,
    const int num_targets,
    const float fc,
    const float slope,
    const float t_start) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_targets) {
    return;
  }

  const float dist = d[i];
  const float amp = a[i];
  const float k0 = dist * k0_per_meter;
  const float tau = 2.0f * dist / kC0;
  const float phi0 = kTwoPi * (fc * tau + slope * tau * (t_start - 0.5f * tau));
  const Complex phase_corr = cexp_f(phi0);
  const float dphi0_dd = kTwoPi * 2.0f / kC0 * (fc + slope * t_start - slope * tau);
  const float dx_dd = -kTwoPi * k0_per_meter / static_cast<float>(n_fft);

  float dL_dd = 0.0f;
  float dL_da = 0.0f;
  for (int bin = 0; bin < num_bins; ++bin) {
    const float gout_re = grad_output_re[bin];
    const float gout_im = grad_output_im[bin];
    const float x = kTwoPi * (static_cast<float>(bin) - k0) / static_cast<float>(n_fft);
    const Complex result = cmul(dirichlet_kernel(x, n), phase_corr);
    const Complex result_grad = cmul(dirichlet_kernel_grad(x, n), phase_corr);

    dL_da += gout_re * result.re + gout_im * result.im;
    dL_dd += amp * (gout_re * result_grad.re + gout_im * result_grad.im) * dx_dd;
    dL_dd += amp * (-gout_re * result.im + gout_im * result.re) * dphi0_dd;
  }

  grad_d[i] = dL_dd;
  grad_a[i] = dL_da;
}

__global__ void backward_per_bin_kernel(
    const float* __restrict__ d,
    const float* __restrict__ a,
    const float* __restrict__ grad_output_re,
    const float* __restrict__ grad_output_im,
    float* __restrict__ grad_d,
    float* __restrict__ grad_a,
    const float n,
    const float k0_per_meter,
    const int num_bins,
    const int n_fft,
    const int num_targets,
    const int bins_per_chunk,
    const float fc,
    const float slope,
    const float t_start) {
  const int chunk_idx = blockIdx.x;
  const int bin = chunk_idx * bins_per_chunk + threadIdx.x;
  if (bin >= num_bins) {
    return;
  }

  const float gout_re = grad_output_re[bin];
  const float gout_im = grad_output_im[bin];
  if (gout_re == 0.0f && gout_im == 0.0f) {
    return;
  }

  for (int i = 0; i < num_targets; ++i) {
    const float dist = d[i];
    const float amp = a[i];
    const float k0 = dist * k0_per_meter;
    const float tau = 2.0f * dist / kC0;
    const float phi0 = kTwoPi * (fc * tau + slope * tau * (t_start - 0.5f * tau));
    const Complex phase_corr = cexp_f(phi0);
    const float dphi0_dd = kTwoPi * 2.0f / kC0 * (fc + slope * t_start - slope * tau);
    const float x = kTwoPi * (static_cast<float>(bin) - k0) / static_cast<float>(n_fft);
    const Complex result = cmul(dirichlet_kernel(x, n), phase_corr);
    const Complex result_grad = cmul(dirichlet_kernel_grad(x, n), phase_corr);
    const float dx_dd = -kTwoPi * k0_per_meter / static_cast<float>(n_fft);

    const float dL_da = gout_re * result.re + gout_im * result.im;
    float dL_dd = amp * (gout_re * result_grad.re + gout_im * result_grad.im) * dx_dd;
    dL_dd += amp * (-gout_re * result.im + gout_im * result.re) * dphi0_dd;

    const int out_idx = chunk_idx * num_targets + i;
    atomicAdd(grad_d + out_idx, dL_dd);
    atomicAdd(grad_a + out_idx, dL_da);
  }
}

void check_cuda_float(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float, name, " must have dtype torch.float32.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<int>::max()), name, " is out of int32 range.");
  return static_cast<int>(value);
}

}  // namespace

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
    double t_start) {
  check_cuda_float(d, "d");
  check_cuda_float(a, "a");
  check_cuda_float(output_re, "output_re");
  check_cuda_float(output_im, "output_im");
  TORCH_CHECK(output_re.sizes() == output_im.sizes(), "output_re and output_im must have the same shape.");
  TORCH_CHECK(output_re.dim() == 2, "output tensors must have shape (chunks, bins).");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int targets = checked_int(num_targets, "num_targets");
  const int chunk_size = checked_int(targets_per_chunk, "targets_per_chunk");
  TORCH_CHECK(chunk_size > 0, "targets_per_chunk must be positive.");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(d));
  constexpr int block_size = 256;
  const dim3 block(block_size, 1, 1);
  const dim3 grid((bins + block_size - 1) / block_size, output_re.size(0), 1);
  forward_chunked_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      d.data_ptr<float>(),
      a.data_ptr<float>(),
      output_re.data_ptr<float>(),
      output_im.data_ptr<float>(),
      static_cast<float>(n),
      static_cast<float>(k0_per_meter),
      bins,
      fft,
      targets,
      chunk_size,
      static_cast<float>(fc),
      static_cast<float>(slope),
      static_cast<float>(t_start));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    double t_start) {
  check_cuda_float(d0, "d0");
  check_cuda_float(d_rate, "d_rate");
  check_cuda_float(a0, "a0");
  check_cuda_float(output_re, "output_re");
  check_cuda_float(output_im, "output_im");
  TORCH_CHECK(d0.sizes() == d_rate.sizes() && d0.sizes() == a0.sizes(), "d0, d_rate, and a0 must have the same shape.");
  TORCH_CHECK(output_re.sizes() == output_im.sizes(), "output_re and output_im must have the same shape.");
  TORCH_CHECK(output_re.dim() == 3, "output tensors must have shape (chirps, pairs, bins).");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int per_pair = checked_int(targets_per_pair, "targets_per_pair");
  const int chirps = checked_int(chirp_per_frame, "chirp_per_frame");
  const int tx = checked_int(num_tx, "num_tx");
  const int update = checked_int(range_loss_update, "range_loss_update");
  const int pairs = checked_int(output_re.size(1), "num_pairs");
  TORCH_CHECK(per_pair > 0, "targets_per_pair must be positive.");
  TORCH_CHECK(tx > 0 && pairs % tx == 0, "num_pairs must be a positive multiple of num_tx.");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(d0));
  constexpr int block_size = 256;
  const dim3 block(block_size, 1, 1);
  const dim3 grid((bins + block_size - 1) / block_size, pairs, chirps);
  forward_mimo_linear_chunked_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      d0.data_ptr<float>(),
      d_rate.data_ptr<float>(),
      a0.data_ptr<float>(),
      output_re.data_ptr<float>(),
      output_im.data_ptr<float>(),
      static_cast<float>(n),
      static_cast<float>(k0_per_meter),
      bins,
      fft,
      per_pair,
      chirps,
      static_cast<float>(chirp_period),
      tx,
      update,
      pairs,
      static_cast<float>(fc),
      static_cast<float>(slope),
      static_cast<float>(t_start));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    double t_start) {
  check_cuda_float(d, "d");
  check_cuda_float(a, "a");
  check_cuda_float(grad_output_re, "grad_output_re");
  check_cuda_float(grad_output_im, "grad_output_im");
  check_cuda_float(grad_d, "grad_d");
  check_cuda_float(grad_a, "grad_a");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int targets = checked_int(num_targets, "num_targets");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(d));
  constexpr int block_size = 256;
  const dim3 block(block_size, 1, 1);
  const dim3 grid((targets + block_size - 1) / block_size, 1, 1);
  backward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      d.data_ptr<float>(),
      a.data_ptr<float>(),
      grad_output_re.data_ptr<float>(),
      grad_output_im.data_ptr<float>(),
      grad_d.data_ptr<float>(),
      grad_a.data_ptr<float>(),
      static_cast<float>(n),
      static_cast<float>(k0_per_meter),
      bins,
      fft,
      targets,
      static_cast<float>(fc),
      static_cast<float>(slope),
      static_cast<float>(t_start));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    double t_start) {
  check_cuda_float(d, "d");
  check_cuda_float(a, "a");
  check_cuda_float(grad_output_re, "grad_output_re");
  check_cuda_float(grad_output_im, "grad_output_im");
  check_cuda_float(grad_d, "grad_d");
  check_cuda_float(grad_a, "grad_a");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int targets = checked_int(num_targets, "num_targets");
  const int chunk = checked_int(bins_per_chunk, "bins_per_chunk");
  TORCH_CHECK(chunk > 0, "bins_per_chunk must be positive.");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(d));
  const dim3 block(chunk, 1, 1);
  const dim3 grid((bins + chunk - 1) / chunk, 1, 1);
  backward_per_bin_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      d.data_ptr<float>(),
      a.data_ptr<float>(),
      grad_output_re.data_ptr<float>(),
      grad_output_im.data_ptr<float>(),
      grad_d.data_ptr<float>(),
      grad_a.data_ptr<float>(),
      static_cast<float>(n),
      static_cast<float>(k0_per_meter),
      bins,
      fft,
      targets,
      chunk,
      static_cast<float>(fc),
      static_cast<float>(slope),
      static_cast<float>(t_start));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
