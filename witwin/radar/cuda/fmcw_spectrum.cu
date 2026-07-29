// Complex FMCW Dirichlet spectrum synthesis over a chirp's fast-time axis.
//
// This is the Phase-4 `_radar_native` synthesis primitive. Radar ships ONE
// native artifact, so every family registers in the same `_radar_native`
// library (R-ADR-004; the Phase-10 rename made the physical stem match this
// logical name).
//
// Convention, taken from `dirichlet.cu` `path_response` and NOT re-derived:
//
//   cycles(tau, t_slot, t_m) = carrier_hz * tau
//                            + carrier_rate_hz * (tau - tau_rt)
//                            + slope * tau * (t_start - 0.5 * tau)
//                            + slope * tau * t_m
//   t_slot(c, p)          = (c * num_tx + segment_tx_index[p]) * chirp_period
//   tau(c, p)             = tau_rt + tau_rate * t_slot(c, p)
//   s[c][p][m]            = sum_k w[k] * exp(+j * 2 * pi * cycles)
//
// TDM-MIMO fires the transmitters SEQUENTIALLY, one chirp period apart, so the
// slow-time coordinate of a (chirp, sensor pair) cell is its TDM SLOT, not its
// chirp index. `segment_tx_index[p]` is which transmitter owns sensor pair `p`,
// and `num_tx` is how many slots one chirp loop holds. This is the same slot
// time `dirichlet.cu`'s MIMO kernel, `solvers/common.py::collect_interpolated_
// samples`, and `sigproc/pointcloud.py::_compensate_tdm_phase` already assume;
// synthesizing it here is what makes the per-TX motion phase a physical
// consequence of the kernel rather than a downstream reinterpretation.
// `num_tx = 1` with a zero index table gives `(c * 1 + 0) * Tc == c * Tc`, i.e.
// bit-identical output to the pre-TDM kernel.
//
// `carrier_rate_hz` exists because the carrier has two homes and only one of
// them can express intra-frame Doppler. When the weight carries the carrier
// phase (a Channel coefficient, `carrier_hz = 0`), that phase was formed at the
// FROZEN per-frame `tau_rt` and is constant across chirps, so the slow-time
// phase walk loses its dominant term `f_c * tau_rate * t_c` and understates
// Doppler by 21x to 215x over the fixture's fast-time axis. Applying
// `carrier_rate_hz` to the delay CHANGE `(tau - tau_rt) = tau_rate * t_c`
// restores exactly that term without re-applying the absolute carrier phase the
// weight already holds. Setting both to `fc` would double count and is refused
// by the Python contract.
//
// Four rules this file encodes, each pinned by a test:
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

struct Complex {
  float re;
  float im;
};

struct SpectrumResponse {
  Complex value;
  Complex d_tau_rt;
  Complex d_tau_rate;
};

__device__ __forceinline__ Complex cmul(const Complex a, const Complex b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

__device__ __forceinline__ Complex cexp_cycles(const double cycles) {
  const double frac = cycles - floor(cycles);
  float s;
  float c;
  sincosf(static_cast<float>(kTwoPiD * frac), &s, &c);
  return {c, s};
}

__device__ __forceinline__ float dirichlet_scale(const float x, const float n) {
  const float sh = sinf(0.5f * x);
  if (fabsf(sh) < 1.0e-7f) {
    return 2.0f * n + 1.0f;
  }
  return sinf((n + 0.5f) * x) / sh;
}

__device__ __forceinline__ float dirichlet_scale_grad(const float x, const float n) {
  const float sh = sinf(0.5f * x);
  if (fabsf(sh) < 1.0e-7f) {
    return 0.0f;
  }
  const float ch = cosf(0.5f * x);
  const float sn = sinf((n + 0.5f) * x);
  const float cn = cosf((n + 0.5f) * x);
  return ((n + 0.5f) * cn * sh - 0.5f * sn * ch) / (sh * sh);
}

__device__ __forceinline__ Complex dirichlet(const float x, const float n) {
  const float scale = dirichlet_scale(x, n);
  const Complex phase = cexp_cycles(-static_cast<double>(n) * x / kTwoPiD);
  return {scale * phase.re, scale * phase.im};
}

__device__ __forceinline__ Complex dirichlet_grad(const float x, const float n) {
  const float scale = dirichlet_scale(x, n);
  const Complex phase = cexp_cycles(-static_cast<double>(n) * x / kTwoPiD);
  return cmul({dirichlet_scale_grad(x, n), -n * scale}, phase);
}

__device__ __forceinline__ double slot_time(
    const int chirp,
    const int tx_index,
    const int num_tx,
    const double chirp_period_s) {
  const int64_t slot = static_cast<int64_t>(chirp) * num_tx + tx_index;
  return static_cast<double>(slot) * chirp_period_s;
}

__device__ __forceinline__ int clamped_tx_index(
    const int32_t* __restrict__ segment_tx_index,
    const int segment,
    const int num_tx) {
  int tx = static_cast<int>(segment_tx_index[segment]);
  tx = tx < 0 ? 0 : tx;
  return tx >= num_tx ? num_tx - 1 : tx;
}

__device__ __forceinline__ SpectrumResponse spectrum_response(
    const double tau,
    const double drift,
    const double t_slot,
    const int bin,
    const int num_bins,
    const double sample_period_s,
    const double slope,
    const double carrier_hz,
    const double carrier_rate_hz,
    const double t_start) {
  const double base_cycles = carrier_hz * tau + carrier_rate_hz * drift +
      slope * tau * (t_start - 0.5 * tau);
  const Complex phase = cexp_cycles(base_cycles);
  const double k0 = slope * tau * sample_period_s * num_bins;
  const float x = static_cast<float>(
      kTwoPiD * (static_cast<double>(bin) - k0) / num_bins);
  const float n = 0.5f * static_cast<float>(num_bins - 1);
  const float inv_n = 1.0f / static_cast<float>(num_bins);
  const Complex d = dirichlet(x, n);
  const Complex dg = dirichlet_grad(x, n);
  Complex value = cmul(d, phase);
  value.re *= inv_n;
  value.im *= inv_n;

  const double dx_dtau = -kTwoPiD * slope * sample_period_s;
  const double dphi_dtau = kTwoPiD *
      (carrier_hz + slope * t_start - slope * tau);
  Complex shape_grad = cmul(dg, phase);
  shape_grad.re *= static_cast<float>(dx_dtau * inv_n);
  shape_grad.im *= static_cast<float>(dx_dtau * inv_n);
  const Complex d_tau = {
      shape_grad.re - static_cast<float>(dphi_dtau) * value.im,
      shape_grad.im + static_cast<float>(dphi_dtau) * value.re};
  const double dphi_rate = t_slot * (dphi_dtau + kTwoPiD * carrier_rate_hz);
  Complex rate_shape = cmul(dg, phase);
  rate_shape.re *= static_cast<float>(dx_dtau * t_slot * inv_n);
  rate_shape.im *= static_cast<float>(dx_dtau * t_slot * inv_n);
  const Complex d_rate = {
      rate_shape.re - static_cast<float>(dphi_rate) * value.im,
      rate_shape.im + static_cast<float>(dphi_rate) * value.re};
  return {value, d_tau, d_rate};
}
__global__ void fmcw_spectrum_forward_kernel(
    const float* __restrict__ tau_rt,
    const float* __restrict__ tau_rate,
    const float* __restrict__ weight_re,
    const float* __restrict__ weight_im,
    const int64_t* __restrict__ path_offsets,
    const int32_t* __restrict__ segment_tx_index,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    const int num_paths,
    const int num_segments,
    const int num_tx,
    const int num_bins,
    const double sample_period_s,
    const double chirp_period_s,
    const double slope,
    const double carrier_hz,
    const double carrier_rate_hz,
    const double t_start) {
  const int bin = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int chirp = blockIdx.z;
  if (bin >= num_bins || segment >= num_segments) {
    return;
  }
  int64_t start = max(static_cast<int64_t>(0), path_offsets[segment]);
  int64_t end = min(static_cast<int64_t>(num_paths), path_offsets[segment + 1]);
  const double t_slot = slot_time(
      chirp, clamped_tx_index(segment_tx_index, segment, num_tx),
      num_tx, chirp_period_s);
  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = start; k < end; ++k) {
    const double drift = static_cast<double>(tau_rate[k]) * t_slot;
    const double tau = static_cast<double>(tau_rt[k]) + drift;
    const SpectrumResponse response = spectrum_response(
        tau, drift, t_slot, bin, num_bins, sample_period_s, slope,
        carrier_hz, carrier_rate_hz, t_start);
    const float wr = weight_re[k];
    const float wi = weight_im[k];
    acc_re += wr * response.value.re - wi * response.value.im;
    acc_im += wr * response.value.im + wi * response.value.re;
  }
  const int64_t out_idx =
      (static_cast<int64_t>(chirp) * num_segments + segment) * num_bins + bin;
  out_re[out_idx] = acc_re;
  out_im[out_idx] = acc_im;
}
__global__ void fmcw_spectrum_jvp_kernel(
    const float* __restrict__ tau_rt,
    const float* __restrict__ tau_rate,
    const float* __restrict__ weight_re,
    const float* __restrict__ weight_im,
    const int64_t* __restrict__ path_offsets,
    const int32_t* __restrict__ segment_tx_index,
    const float* __restrict__ tan_tau_rt,
    const float* __restrict__ tan_tau_rate,
    const float* __restrict__ tan_weight_re,
    const float* __restrict__ tan_weight_im,
    float* __restrict__ tan_out_re,
    float* __restrict__ tan_out_im,
    const int num_paths,
    const int num_segments,
    const int num_tx,
    const int num_bins,
    const double sample_period_s,
    const double chirp_period_s,
    const double slope,
    const double carrier_hz,
    const double carrier_rate_hz,
    const double t_start) {
  const int bin = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int chirp = blockIdx.z;
  if (bin >= num_bins || segment >= num_segments) {
    return;
  }
  int64_t start = path_offsets[segment];
  int64_t end = path_offsets[segment + 1];
  start = start < 0 ? 0 : start;
  end = end > num_paths ? num_paths : end;
  const double t_slot = slot_time(
      chirp, clamped_tx_index(segment_tx_index, segment, num_tx),
      num_tx, chirp_period_s);
  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = start; k < end; ++k) {
    const double drift = static_cast<double>(tau_rate[k]) * t_slot;
    const double tau = static_cast<double>(tau_rt[k]) + drift;
    const SpectrumResponse response = spectrum_response(
        tau, drift, t_slot, bin, num_bins, sample_period_s, slope,
        carrier_hz, carrier_rate_hz, t_start);
    const float wr = weight_re[k];
    const float wi = weight_im[k];
    const float twr = tan_weight_re[k];
    const float twi = tan_weight_im[k];
    const float tr = tan_tau_rt[k];
    const float tv = tan_tau_rate[k];
    const Complex dq = {
        tr * response.d_tau_rt.re + tv * response.d_tau_rate.re,
        tr * response.d_tau_rt.im + tv * response.d_tau_rate.im};
    acc_re += twr * response.value.re - twi * response.value.im;
    acc_im += twr * response.value.im + twi * response.value.re;
    acc_re += wr * dq.re - wi * dq.im;
    acc_im += wr * dq.im + wi * dq.re;
  }
  const int64_t out_idx =
      (static_cast<int64_t>(chirp) * num_segments + segment) * num_bins + bin;
  tan_out_re[out_idx] = acc_re;
  tan_out_im[out_idx] = acc_im;
}
// One thread per path, looping the whole (chirp, sample) grid. Each path owns
// exactly one output slot in each gradient array, so the reduction needs no
// atomics and the summation order is fixed by the loop nest.
__global__ void fmcw_spectrum_backward_kernel(
    const float* __restrict__ tau_rt,
    const float* __restrict__ tau_rate,
    const float* __restrict__ weight_re,
    const float* __restrict__ weight_im,
    const int64_t* __restrict__ path_segment,
    const int32_t* __restrict__ segment_tx_index,
    const float* __restrict__ grad_out_re,
    const float* __restrict__ grad_out_im,
    float* __restrict__ grad_tau_rt,
    float* __restrict__ grad_tau_rate,
    float* __restrict__ grad_weight_re,
    float* __restrict__ grad_weight_im,
    const int num_paths,
    const int num_segments,
    const int num_tx,
    const int num_chirps,
    const int num_bins,
    const double sample_period_s,
    const double chirp_period_s,
    const double slope,
    const double carrier_hz,
    const double carrier_rate_hz,
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
  const float wr = weight_re[k];
  const float wi = weight_im[k];
  const int tx = clamped_tx_index(
      segment_tx_index, static_cast<int>(segment), num_tx);
  double d_tau = 0.0;
  double d_rate = 0.0;
  double d_wr = 0.0;
  double d_wi = 0.0;
  for (int chirp = 0; chirp < num_chirps; ++chirp) {
    const double t_slot = slot_time(chirp, tx, num_tx, chirp_period_s);
    const double drift = rate * t_slot;
    const double tau = base_tau + drift;
    const int64_t row =
        (static_cast<int64_t>(chirp) * num_segments + segment) * num_bins;
    for (int bin = 0; bin < num_bins; ++bin) {
      const SpectrumResponse response = spectrum_response(
          tau, drift, t_slot, bin, num_bins, sample_period_s, slope,
          carrier_hz, carrier_rate_hz, t_start);
      const float gr = grad_out_re[row + bin];
      const float gi = grad_out_im[row + bin];
      d_wr += static_cast<double>(gr) * response.value.re +
          static_cast<double>(gi) * response.value.im;
      d_wi += -static_cast<double>(gr) * response.value.im +
          static_cast<double>(gi) * response.value.re;
      const Complex dz_tau = {
          wr * response.d_tau_rt.re - wi * response.d_tau_rt.im,
          wr * response.d_tau_rt.im + wi * response.d_tau_rt.re};
      const Complex dz_rate = {
          wr * response.d_tau_rate.re - wi * response.d_tau_rate.im,
          wr * response.d_tau_rate.im + wi * response.d_tau_rate.re};
      d_tau += static_cast<double>(gr) * dz_tau.re +
          static_cast<double>(gi) * dz_tau.im;
      d_rate += static_cast<double>(gr) * dz_rate.re +
          static_cast<double>(gi) * dz_rate.im;
    }
  }
  grad_tau_rt[k] = static_cast<float>(d_tau);
  grad_tau_rate[k] = static_cast<float>(d_rate);
  grad_weight_re[k] = static_cast<float>(d_wr);
  grad_weight_im[k] = static_cast<float>(d_wi);
}
void check_cuda_float(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Float,
      name,
      " must have dtype torch.float32.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void check_cuda_int(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Int,
      name,
      " must have dtype torch.int32.");
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
      "spectrum output components must have the same shape.");
  STD_TORCH_CHECK(
      out_re.dim() == 3,
      "spectrum output must have shape (chirps, segments, samples).");
  STD_TORCH_CHECK(
      out_re.size(0) == num_chirps && out_re.size(1) == num_segments &&
          out_re.size(2) == num_samples,
      "spectrum output shape disagrees with the declared grid.");
}

dim3 sample_grid(int num_samples, int num_segments, int num_chirps, int block) {
  return dim3((num_samples + block - 1) / block, num_segments, num_chirps);
}

void check_tdm(
    const torch::stable::Tensor& segment_tx_index,
    int num_segments,
    int num_tx) {
  STD_TORCH_CHECK(num_tx > 0, "num_tx must be positive.");
  check_cuda_int(segment_tx_index, "segment_tx_index");
  STD_TORCH_CHECK(
      segment_tx_index.numel() == static_cast<int64_t>(num_segments),
      "segment_tx_index must hold one transmitter index per sensor-pair segment.");
}

}  // namespace

void fmcw_spectrum_forward_cuda(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_offsets,
    const torch::stable::Tensor& segment_tx_index,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    int64_t num_paths,
    int64_t num_segments,
    int64_t num_tx,
    int64_t num_chirps,
    int64_t num_samples,
    double sample_period_s,
    double chirp_period_s,
    double slope_hz_per_s,
    double carrier_hz,
    double carrier_rate_hz,
    double t_start_s) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int transmitters = checked_int(num_tx, "num_tx");
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
  check_tdm(segment_tx_index, segments, transmitters);
  check_output(out_re, out_im, chirps, segments, samples, "out_re", "out_im");

  const torch::stable::accelerator::DeviceGuard device_guard(
      out_re.get_device_index());
  constexpr int block_size = 256;
  fmcw_spectrum_forward_kernel<<<
      sample_grid(samples, segments, chirps, block_size),
      dim3(block_size, 1, 1),
      0,
      current_cuda_stream(out_re)>>>(
      tau_rt.const_data_ptr<float>(),
      tau_rate.const_data_ptr<float>(),
      weight_re.const_data_ptr<float>(),
      weight_im.const_data_ptr<float>(),
      path_offsets.const_data_ptr<int64_t>(),
      segment_tx_index.const_data_ptr<int32_t>(),
      out_re.mutable_data_ptr<float>(),
      out_im.mutable_data_ptr<float>(),
      paths,
      segments,
      transmitters,
      samples,
      sample_period_s,
      chirp_period_s,
      slope_hz_per_s,
      carrier_hz,
      carrier_rate_hz,
      t_start_s);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void fmcw_spectrum_jvp_cuda(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_offsets,
    const torch::stable::Tensor& segment_tx_index,
    const torch::stable::Tensor& tan_tau_rt,
    const torch::stable::Tensor& tan_tau_rate,
    const torch::stable::Tensor& tan_weight_re,
    const torch::stable::Tensor& tan_weight_im,
    torch::stable::Tensor& tan_out_re,
    torch::stable::Tensor& tan_out_im,
    int64_t num_paths,
    int64_t num_segments,
    int64_t num_tx,
    int64_t num_chirps,
    int64_t num_samples,
    double sample_period_s,
    double chirp_period_s,
    double slope_hz_per_s,
    double carrier_hz,
    double carrier_rate_hz,
    double t_start_s) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int transmitters = checked_int(num_tx, "num_tx");
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
  check_tdm(segment_tx_index, segments, transmitters);
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
  fmcw_spectrum_jvp_kernel<<<
      sample_grid(samples, segments, chirps, block_size),
      dim3(block_size, 1, 1),
      0,
      current_cuda_stream(tan_out_re)>>>(
      tau_rt.const_data_ptr<float>(),
      tau_rate.const_data_ptr<float>(),
      weight_re.const_data_ptr<float>(),
      weight_im.const_data_ptr<float>(),
      path_offsets.const_data_ptr<int64_t>(),
      segment_tx_index.const_data_ptr<int32_t>(),
      tan_tau_rt.const_data_ptr<float>(),
      tan_tau_rate.const_data_ptr<float>(),
      tan_weight_re.const_data_ptr<float>(),
      tan_weight_im.const_data_ptr<float>(),
      tan_out_re.mutable_data_ptr<float>(),
      tan_out_im.mutable_data_ptr<float>(),
      paths,
      segments,
      transmitters,
      samples,
      sample_period_s,
      chirp_period_s,
      slope_hz_per_s,
      carrier_hz,
      carrier_rate_hz,
      t_start_s);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void fmcw_spectrum_backward_cuda(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_segment,
    const torch::stable::Tensor& segment_tx_index,
    const torch::stable::Tensor& grad_out_re,
    const torch::stable::Tensor& grad_out_im,
    torch::stable::Tensor& grad_tau_rt,
    torch::stable::Tensor& grad_tau_rate,
    torch::stable::Tensor& grad_weight_re,
    torch::stable::Tensor& grad_weight_im,
    int64_t num_paths,
    int64_t num_segments,
    int64_t num_tx,
    int64_t num_chirps,
    int64_t num_samples,
    double sample_period_s,
    double chirp_period_s,
    double slope_hz_per_s,
    double carrier_hz,
    double carrier_rate_hz,
    double t_start_s) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int transmitters = checked_int(num_tx, "num_tx");
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
  check_tdm(segment_tx_index, segments, transmitters);
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
  fmcw_spectrum_backward_kernel<<<
      dim3((paths + block_size - 1) / block_size, 1, 1),
      dim3(block_size, 1, 1),
      0,
      current_cuda_stream(grad_tau_rt)>>>(
      tau_rt.const_data_ptr<float>(),
      tau_rate.const_data_ptr<float>(),
      weight_re.const_data_ptr<float>(),
      weight_im.const_data_ptr<float>(),
      path_segment.const_data_ptr<int64_t>(),
      segment_tx_index.const_data_ptr<int32_t>(),
      grad_out_re.const_data_ptr<float>(),
      grad_out_im.const_data_ptr<float>(),
      grad_tau_rt.mutable_data_ptr<float>(),
      grad_tau_rate.mutable_data_ptr<float>(),
      grad_weight_re.mutable_data_ptr<float>(),
      grad_weight_im.mutable_data_ptr<float>(),
      paths,
      segments,
      transmitters,
      chirps,
      samples,
      sample_period_s,
      chirp_period_s,
      slope_hz_per_s,
      carrier_hz,
      carrier_rate_hz,
      t_start_s);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("fmcw_spectrum_forward", TORCH_BOX(&fmcw_spectrum_forward_cuda));
  m.impl("fmcw_spectrum_backward", TORCH_BOX(&fmcw_spectrum_backward_cuda));
  m.impl("fmcw_spectrum_jvp", TORCH_BOX(&fmcw_spectrum_jvp_cuda));
}
