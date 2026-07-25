// Complex FMCW beat synthesis over a chirp's fast-time axis.
//
// This is the Phase-4 `_radar_native` synthesis primitive. It is registered in
// the existing `witwin_radar_dirichlet_cuda` library because the packaging
// chain assumes a single native artifact stem; the physical rename is Phase-10
// work and is recorded in R-ADR-004.
//
// Convention, taken from `dirichlet.cu` `path_response` and NOT re-derived:
//
//   cycles(tau, t_c, t_m) = carrier_hz * tau
//                         + slope * tau * (t_start - 0.5 * tau)
//                         + slope * tau * t_m
//   tau(t_c)              = tau_rt + tau_rate * t_c
//   s[c][p][m]            = sum_k w[k] * exp(+j * 2 * pi * cycles)
//
// Three rules this file encodes, each pinned by a test:
//
//  1. `tau_rt` is the ROUND-TRIP delay and is consumed directly. The existing
//     Dirichlet path computes `tau = 2 * distance / c0`, which is a monostatic
//     assumption; a two-leg round trip already knows its own total delay and
//     doubling it here would produce a self-consistent, exactly 2x wrong range.
//  2. The phasor is `+j`. Channel publishes `exp(-j * k * d)`; the two are
//     conjugates and the single conversion site is the Python synthesis facade.
//  3. The cycle count is accumulated in double and wrapped to [0, 1) before
//     `sincosf`. At ~47 cycles of `f_beat * t_m` a naive float32 phase costs
//     about 1e-2 rad, which is the same magnitude as the gradients under test.
//     Fast math stays off.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr double kTwoPiD = 6.283185307179586476925286766559;

struct BeatPhase {
  float sin_phi;
  float cos_phi;
  // d(phi) / d(tau) in radians per second.
  double dphi_dtau;
};

// Phase of one path at one (chirp, sample) grid point. The cycle count is a
// large number  -  hundreds of cycles for a metre-scale target  -  so it is
// formed and wrapped in double before it is handed to the single-precision
// trigonometric unit.
__device__ __forceinline__ BeatPhase beat_phase(
    const double tau,
    const double t_m,
    const double slope,
    const double carrier_hz,
    const double t_start) {
  const double cycles = carrier_hz * tau + slope * tau * (t_start - 0.5 * tau) +
      slope * tau * t_m;
  const double frac = cycles - floor(cycles);
  float sin_phi;
  float cos_phi;
  sincosf(static_cast<float>(kTwoPiD * frac), &sin_phi, &cos_phi);
  return {
      sin_phi,
      cos_phi,
      kTwoPiD * (carrier_hz + slope * t_start - slope * tau + slope * t_m)};
}

__global__ void fmcw_beat_forward_kernel(
    const float* __restrict__ tau_rt,
    const float* __restrict__ tau_rate,
    const float* __restrict__ weight_re,
    const float* __restrict__ weight_im,
    const int64_t* __restrict__ path_offsets,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    const int num_paths,
    const int num_segments,
    const int num_samples,
    const double sample_period_s,
    const double chirp_period_s,
    const double slope,
    const double carrier_hz,
    const double t_start) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int chirp = blockIdx.z;
  if (sample >= num_samples || segment >= num_segments) {
    return;
  }

  // Clamped so a malformed offset table cannot walk off the path arrays. The
  // table is validated for shape on the host; its VALUES are never read there.
  int64_t start = path_offsets[segment];
  int64_t end = path_offsets[segment + 1];
  start = start < 0 ? 0 : start;
  end = end > num_paths ? num_paths : end;

  const double t_c = static_cast<double>(chirp) * chirp_period_s;
  const double t_m = static_cast<double>(sample) * sample_period_s;

  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = start; k < end; ++k) {
    const double tau =
        static_cast<double>(tau_rt[k]) + static_cast<double>(tau_rate[k]) * t_c;
    const BeatPhase phase =
        beat_phase(tau, t_m, slope, carrier_hz, t_start);
    const float w_re = weight_re[k];
    const float w_im = weight_im[k];
    acc_re += w_re * phase.cos_phi - w_im * phase.sin_phi;
    acc_im += w_re * phase.sin_phi + w_im * phase.cos_phi;
  }

  const int64_t out_idx =
      (static_cast<int64_t>(chirp) * num_segments + segment) * num_samples +
      sample;
  out_re[out_idx] = acc_re;
  out_im[out_idx] = acc_im;
}

__global__ void fmcw_beat_jvp_kernel(
    const float* __restrict__ tau_rt,
    const float* __restrict__ tau_rate,
    const float* __restrict__ weight_re,
    const float* __restrict__ weight_im,
    const int64_t* __restrict__ path_offsets,
    const float* __restrict__ tan_tau_rt,
    const float* __restrict__ tan_tau_rate,
    const float* __restrict__ tan_weight_re,
    const float* __restrict__ tan_weight_im,
    float* __restrict__ tan_out_re,
    float* __restrict__ tan_out_im,
    const int num_paths,
    const int num_segments,
    const int num_samples,
    const double sample_period_s,
    const double chirp_period_s,
    const double slope,
    const double carrier_hz,
    const double t_start) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int chirp = blockIdx.z;
  if (sample >= num_samples || segment >= num_segments) {
    return;
  }

  int64_t start = path_offsets[segment];
  int64_t end = path_offsets[segment + 1];
  start = start < 0 ? 0 : start;
  end = end > num_paths ? num_paths : end;

  const double t_c = static_cast<double>(chirp) * chirp_period_s;
  const double t_m = static_cast<double>(sample) * sample_period_s;

  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = start; k < end; ++k) {
    const double tau =
        static_cast<double>(tau_rt[k]) + static_cast<double>(tau_rate[k]) * t_c;
    const BeatPhase phase =
        beat_phase(tau, t_m, slope, carrier_hz, t_start);
    const float w_re = weight_re[k];
    const float w_im = weight_im[k];
    const float re = w_re * phase.cos_phi - w_im * phase.sin_phi;
    const float im = w_re * phase.sin_phi + w_im * phase.cos_phi;

    const double dtau = static_cast<double>(tan_tau_rt[k]) +
        static_cast<double>(tan_tau_rate[k]) * t_c;
    const float dphi = static_cast<float>(phase.dphi_dtau * dtau);
    const float tw_re = tan_weight_re[k];
    const float tw_im = tan_weight_im[k];
    acc_re += tw_re * phase.cos_phi - tw_im * phase.sin_phi - dphi * im;
    acc_im += tw_re * phase.sin_phi + tw_im * phase.cos_phi + dphi * re;
  }

  const int64_t out_idx =
      (static_cast<int64_t>(chirp) * num_segments + segment) * num_samples +
      sample;
  tan_out_re[out_idx] = acc_re;
  tan_out_im[out_idx] = acc_im;
}

// One thread per path, looping the whole (chirp, sample) grid. Each path owns
// exactly one output slot in each gradient array, so the reduction needs no
// atomics and the summation order is fixed by the loop nest.
__global__ void fmcw_beat_backward_kernel(
    const float* __restrict__ tau_rt,
    const float* __restrict__ tau_rate,
    const float* __restrict__ weight_re,
    const float* __restrict__ weight_im,
    const int64_t* __restrict__ path_segment,
    const float* __restrict__ grad_out_re,
    const float* __restrict__ grad_out_im,
    float* __restrict__ grad_tau_rt,
    float* __restrict__ grad_tau_rate,
    float* __restrict__ grad_weight_re,
    float* __restrict__ grad_weight_im,
    const int num_paths,
    const int num_segments,
    const int num_chirps,
    const int num_samples,
    const double sample_period_s,
    const double chirp_period_s,
    const double slope,
    const double carrier_hz,
    const double t_start) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_paths) {
    return;
  }

  int64_t segment = path_segment[k];
  segment = segment < 0 ? 0 : segment;
  segment = segment >= num_segments ? num_segments - 1 : segment;

  const double base_tau = static_cast<double>(tau_rt[k]);
  const double rate = static_cast<double>(tau_rate[k]);
  const float w_re = weight_re[k];
  const float w_im = weight_im[k];

  double d_tau_rt = 0.0;
  double d_tau_rate = 0.0;
  double d_w_re = 0.0;
  double d_w_im = 0.0;

  for (int chirp = 0; chirp < num_chirps; ++chirp) {
    const double t_c = static_cast<double>(chirp) * chirp_period_s;
    const double tau = base_tau + rate * t_c;
    const int64_t row_base =
        (static_cast<int64_t>(chirp) * num_segments + segment) * num_samples;
    for (int sample = 0; sample < num_samples; ++sample) {
      const double t_m = static_cast<double>(sample) * sample_period_s;
      const BeatPhase phase =
          beat_phase(tau, t_m, slope, carrier_hz, t_start);
      const float g_re = grad_out_re[row_base + sample];
      const float g_im = grad_out_im[row_base + sample];
      const float re = w_re * phase.cos_phi - w_im * phase.sin_phi;
      const float im = w_re * phase.sin_phi + w_im * phase.cos_phi;

      d_w_re += static_cast<double>(g_re) * phase.cos_phi +
          static_cast<double>(g_im) * phase.sin_phi;
      d_w_im += -static_cast<double>(g_re) * phase.sin_phi +
          static_cast<double>(g_im) * phase.cos_phi;

      const double d_phi = -static_cast<double>(g_re) * im +
          static_cast<double>(g_im) * re;
      const double d_tau = d_phi * phase.dphi_dtau;
      d_tau_rt += d_tau;
      d_tau_rate += d_tau * t_c;
    }
  }

  grad_tau_rt[k] = static_cast<float>(d_tau_rt);
  grad_tau_rate[k] = static_cast<float>(d_tau_rate);
  grad_weight_re[k] = static_cast<float>(d_w_re);
  grad_weight_im[k] = static_cast<float>(d_w_im);
}

void check_cuda_float(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Float,
      name,
      " must have dtype torch.float32.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void check_cuda_long(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Long,
      name,
      " must have dtype torch.int64.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

int checked_int(int64_t value, const char* name) {
  STD_TORCH_CHECK(
      value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<int>::max()),
      name,
      " is out of int32 range.");
  return static_cast<int>(value);
}

cudaStream_t current_cuda_stream(const torch::stable::Tensor& tensor) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

void check_path_inputs(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    int num_paths) {
  check_cuda_float(tau_rt, "tau_rt");
  check_cuda_float(tau_rate, "tau_rate");
  check_cuda_float(weight_re, "weight_re");
  check_cuda_float(weight_im, "weight_im");
  STD_TORCH_CHECK(
      tau_rt.numel() == num_paths && tau_rate.numel() == num_paths &&
          weight_re.numel() == num_paths && weight_im.numel() == num_paths,
      "tau_rt, tau_rate, weight_re, and weight_im must each hold num_paths values.");
}

void check_output(
    const torch::stable::Tensor& out_re,
    const torch::stable::Tensor& out_im,
    int num_chirps,
    int num_segments,
    int num_samples,
    const char* name_re,
    const char* name_im) {
  check_cuda_float(out_re, name_re);
  check_cuda_float(out_im, name_im);
  STD_TORCH_CHECK(
      out_re.sizes().equals(out_im.sizes()),
      "beat output components must have the same shape.");
  STD_TORCH_CHECK(
      out_re.dim() == 3,
      "beat output must have shape (chirps, segments, samples).");
  STD_TORCH_CHECK(
      out_re.size(0) == num_chirps && out_re.size(1) == num_segments &&
          out_re.size(2) == num_samples,
      "beat output shape disagrees with the declared grid.");
}

dim3 sample_grid(int num_samples, int num_segments, int num_chirps, int block) {
  return dim3((num_samples + block - 1) / block, num_segments, num_chirps);
}

}  // namespace

void fmcw_beat_forward_cuda(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_offsets,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    int64_t num_paths,
    int64_t num_segments,
    int64_t num_chirps,
    int64_t num_samples,
    double sample_period_s,
    double chirp_period_s,
    double slope_hz_per_s,
    double carrier_hz,
    double t_start_s) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int chirps = checked_int(num_chirps, "num_chirps");
  const int samples = checked_int(num_samples, "num_samples");
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(chirps > 0, "num_chirps must be positive.");
  STD_TORCH_CHECK(samples > 0, "num_samples must be positive.");
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths);
  check_cuda_long(path_offsets, "path_offsets");
  STD_TORCH_CHECK(
      path_offsets.numel() == static_cast<int64_t>(segments) + 1,
      "path_offsets must hold num_segments + 1 values.");
  check_output(out_re, out_im, chirps, segments, samples, "out_re", "out_im");

  const torch::stable::accelerator::DeviceGuard device_guard(
      out_re.get_device_index());
  constexpr int block_size = 256;
  fmcw_beat_forward_kernel<<<
      sample_grid(samples, segments, chirps, block_size),
      dim3(block_size, 1, 1),
      0,
      current_cuda_stream(out_re)>>>(
      tau_rt.const_data_ptr<float>(),
      tau_rate.const_data_ptr<float>(),
      weight_re.const_data_ptr<float>(),
      weight_im.const_data_ptr<float>(),
      path_offsets.const_data_ptr<int64_t>(),
      out_re.mutable_data_ptr<float>(),
      out_im.mutable_data_ptr<float>(),
      paths,
      segments,
      samples,
      sample_period_s,
      chirp_period_s,
      slope_hz_per_s,
      carrier_hz,
      t_start_s);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void fmcw_beat_jvp_cuda(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_offsets,
    const torch::stable::Tensor& tan_tau_rt,
    const torch::stable::Tensor& tan_tau_rate,
    const torch::stable::Tensor& tan_weight_re,
    const torch::stable::Tensor& tan_weight_im,
    torch::stable::Tensor& tan_out_re,
    torch::stable::Tensor& tan_out_im,
    int64_t num_paths,
    int64_t num_segments,
    int64_t num_chirps,
    int64_t num_samples,
    double sample_period_s,
    double chirp_period_s,
    double slope_hz_per_s,
    double carrier_hz,
    double t_start_s) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int chirps = checked_int(num_chirps, "num_chirps");
  const int samples = checked_int(num_samples, "num_samples");
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(chirps > 0, "num_chirps must be positive.");
  STD_TORCH_CHECK(samples > 0, "num_samples must be positive.");
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths);
  check_path_inputs(
      tan_tau_rt, tan_tau_rate, tan_weight_re, tan_weight_im, paths);
  check_cuda_long(path_offsets, "path_offsets");
  STD_TORCH_CHECK(
      path_offsets.numel() == static_cast<int64_t>(segments) + 1,
      "path_offsets must hold num_segments + 1 values.");
  check_output(
      tan_out_re,
      tan_out_im,
      chirps,
      segments,
      samples,
      "tan_out_re",
      "tan_out_im");

  const torch::stable::accelerator::DeviceGuard device_guard(
      tan_out_re.get_device_index());
  constexpr int block_size = 256;
  fmcw_beat_jvp_kernel<<<
      sample_grid(samples, segments, chirps, block_size),
      dim3(block_size, 1, 1),
      0,
      current_cuda_stream(tan_out_re)>>>(
      tau_rt.const_data_ptr<float>(),
      tau_rate.const_data_ptr<float>(),
      weight_re.const_data_ptr<float>(),
      weight_im.const_data_ptr<float>(),
      path_offsets.const_data_ptr<int64_t>(),
      tan_tau_rt.const_data_ptr<float>(),
      tan_tau_rate.const_data_ptr<float>(),
      tan_weight_re.const_data_ptr<float>(),
      tan_weight_im.const_data_ptr<float>(),
      tan_out_re.mutable_data_ptr<float>(),
      tan_out_im.mutable_data_ptr<float>(),
      paths,
      segments,
      samples,
      sample_period_s,
      chirp_period_s,
      slope_hz_per_s,
      carrier_hz,
      t_start_s);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void fmcw_beat_backward_cuda(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_segment,
    const torch::stable::Tensor& grad_out_re,
    const torch::stable::Tensor& grad_out_im,
    torch::stable::Tensor& grad_tau_rt,
    torch::stable::Tensor& grad_tau_rate,
    torch::stable::Tensor& grad_weight_re,
    torch::stable::Tensor& grad_weight_im,
    int64_t num_paths,
    int64_t num_segments,
    int64_t num_chirps,
    int64_t num_samples,
    double sample_period_s,
    double chirp_period_s,
    double slope_hz_per_s,
    double carrier_hz,
    double t_start_s) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int chirps = checked_int(num_chirps, "num_chirps");
  const int samples = checked_int(num_samples, "num_samples");
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(chirps > 0, "num_chirps must be positive.");
  STD_TORCH_CHECK(samples > 0, "num_samples must be positive.");
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths);
  check_cuda_long(path_segment, "path_segment");
  STD_TORCH_CHECK(
      path_segment.numel() == static_cast<int64_t>(paths),
      "path_segment must hold one segment index per path.");
  check_output(
      grad_out_re,
      grad_out_im,
      chirps,
      segments,
      samples,
      "grad_out_re",
      "grad_out_im");
  check_path_inputs(
      grad_tau_rt, grad_tau_rate, grad_weight_re, grad_weight_im, paths);

  if (paths == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      grad_tau_rt.get_device_index());
  constexpr int block_size = 256;
  fmcw_beat_backward_kernel<<<
      dim3((paths + block_size - 1) / block_size, 1, 1),
      dim3(block_size, 1, 1),
      0,
      current_cuda_stream(grad_tau_rt)>>>(
      tau_rt.const_data_ptr<float>(),
      tau_rate.const_data_ptr<float>(),
      weight_re.const_data_ptr<float>(),
      weight_im.const_data_ptr<float>(),
      path_segment.const_data_ptr<int64_t>(),
      grad_out_re.const_data_ptr<float>(),
      grad_out_im.const_data_ptr<float>(),
      grad_tau_rt.mutable_data_ptr<float>(),
      grad_tau_rate.mutable_data_ptr<float>(),
      grad_weight_re.mutable_data_ptr<float>(),
      grad_weight_im.mutable_data_ptr<float>(),
      paths,
      segments,
      chirps,
      samples,
      sample_period_s,
      chirp_period_s,
      slope_hz_per_s,
      carrier_hz,
      t_start_s);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(witwin_radar_dirichlet_cuda, CUDA, m) {
  m.impl("fmcw_beat_forward", TORCH_BOX(&fmcw_beat_forward_cuda));
  m.impl("fmcw_beat_backward", TORCH_BOX(&fmcw_beat_backward_cuda));
  m.impl("fmcw_beat_jvp", TORCH_BOX(&fmcw_beat_jvp_cuda));
}
