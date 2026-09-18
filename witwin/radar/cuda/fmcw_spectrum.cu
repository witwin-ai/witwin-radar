// Complex FMCW range-spectrum synthesis; the phase equation is owned by fmcw_phase.cuh.
//
// SI units: tau0 is round-trip delay [s], rate=d(tau)/dt [s/s], slope [Hz/s],
// carrier and carrier_rate [Hz], u=t_start+m*sample_period [s]. The TDM slot
// starts at (chirp*num_tx+tx_index)*chirp_period [s]. The linear-delay model
// evaluates tau=tau0+rate*(slot+u), including motion during the ADC window.
//
// The beat is tx*conj(rx), with exp(+j*2*pi*cycles). Channel weights are
// conjugated once by the Python facade. When the weight owns the carrier at
// tau0, carrier=0 and carrier_rate=fc applies only the delay change; otherwise
// carrier=fc and carrier_rate=0. Round-trip delay is never doubled here.
// Channel owns spreading; Radar owns scattering and sensor weighting.
//
// Spectrum is the 1/N DFT: stationary rows use Dirichlet, moving rows finite sums.
// The model holds weights and delay rate fixed. Curved trajectories, moving
// reflectors, amplitude changes and topology events require refreshed scene
// observations; Radar.simulate defaults to per-ADC sampling for dynamic FMCW.
// Phase uses double accumulation and cycle wrapping before sincosf.
// Independent oracle: tests/test_fmcw_continuous_motion.py (primal/JVP/VJP).

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cuda_runtime.h>
#include "fmcw_phase.cuh"
#include "fmcw_tdm.cuh"

#include <cstdint>

namespace {

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

__device__ __forceinline__ Complex dirichlet(const float x, const float n) {
  const float scale = dirichlet_scale(x, n);
  const Complex phase = cexp_cycles(-static_cast<double>(n) * x / kTwoPiD);
  return {scale * phase.re, scale * phase.im};
}

__device__ __forceinline__ SpectrumResponse spectrum_response(
    const double tau,
    const double rate,
    const double t_slot,
    const int bin,
    const int num_bins,
    const double sample_period_s,
    const double slope,
    const double carrier_hz,
    const double carrier_rate_hz,
    const double t_start, const bool derivatives) {
  if (rate != 0.0 || derivatives) {
    double re=0., im=0., tr=0., ti=0., rr=0., ri=0.;
    for (int m=0; m<num_bins; ++m) {
      const auto term = fmcw_phase_terms(tau, rate, t_slot, t_start+m*sample_period_s, slope, carrier_hz, carrier_rate_hz);
      const auto z = cexp_cycles(term.cycles-static_cast<double>(bin)*m/num_bins);
      re+=z.re; im+=z.im;
      tr-=term.d_tau*z.im; ti+=term.d_tau*z.re;
      rr-=term.d_rate*z.im; ri+=term.d_rate*z.re;
    }
    const double n=num_bins;
    return {{static_cast<float>(re/n),static_cast<float>(im/n)},
            {static_cast<float>(tr/n),static_cast<float>(ti/n)},
            {static_cast<float>(rr/n),static_cast<float>(ri/n)}};
  }
  // Stationary closed form: one delay for the whole chirp makes the fast-time
  // phase linear in m, so the 1/N DFT is a Dirichlet kernel times the phase at
  // m = 0 (the phase owner evaluated with rate 0 at u = t_start). Reached only
  // with derivatives == false, so the derivative slots are never read.
  const double base_cycles =
      fmcw_phase_terms(tau, 0.0, t_slot, t_start, slope, carrier_hz, carrier_rate_hz).cycles;
  const Complex phase = cexp_cycles(base_cycles);
  const double k0 = slope * tau * sample_period_s * num_bins;
  const float x = static_cast<float>(
      kTwoPiD * (static_cast<double>(bin) - k0) / num_bins);
  const float n = 0.5f * static_cast<float>(num_bins - 1);
  const float inv_n = 1.0f / static_cast<float>(num_bins);
  const Complex d = dirichlet(x, n);
  Complex value = cmul(d, phase);
  value.re *= inv_n;
  value.im *= inv_n;
  return {value, {0.0f, 0.0f}, {0.0f, 0.0f}};
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
  const SegmentBounds bounds = segment_bounds(path_offsets, segment, num_paths);
  const double t_slot = slot_time(
      chirp, clamped_tx_index(segment_tx_index, segment, num_tx),
      num_tx, chirp_period_s);
  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = bounds.start; k < bounds.end; ++k) {
    const double tau = static_cast<double>(tau_rt[k]);
    const double rate = static_cast<double>(tau_rate[k]);
    const SpectrumResponse response = spectrum_response(
        tau, rate, t_slot, bin, num_bins, sample_period_s, slope,
        carrier_hz, carrier_rate_hz, t_start, false);
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
  const SegmentBounds bounds = segment_bounds(path_offsets, segment, num_paths);
  const double t_slot = slot_time(
      chirp, clamped_tx_index(segment_tx_index, segment, num_tx),
      num_tx, chirp_period_s);
  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = bounds.start; k < bounds.end; ++k) {
    const double tau = static_cast<double>(tau_rt[k]);
    const double rate = static_cast<double>(tau_rate[k]);
    const SpectrumResponse response = spectrum_response(
        tau, rate, t_slot, bin, num_bins, sample_period_s, slope,
        carrier_hz, carrier_rate_hz, t_start, true);
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
    const double tau = base_tau;
    const int64_t row =
        (static_cast<int64_t>(chirp) * num_segments + segment) * num_bins;
    for (int bin = 0; bin < num_bins; ++bin) {
      const SpectrumResponse response = spectrum_response(
          tau, rate, t_slot, bin, num_bins, sample_period_s, slope,
          carrier_hz, carrier_rate_hz, t_start, true);
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
    int64_t num_bins,
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
  const int bins = checked_int(num_bins, "num_bins");
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(chirps > 0, "num_chirps must be positive.");
  STD_TORCH_CHECK(bins > 0, "num_bins must be positive.");
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths);
  check_cuda_long(path_offsets, "path_offsets");
  STD_TORCH_CHECK(
      path_offsets.numel() == static_cast<int64_t>(segments) + 1,
      "path_offsets must hold num_segments + 1 values.");
  check_tdm(segment_tx_index, segments, transmitters);
  check_output(out_re, out_im, chirps, segments, bins, "bins", "out_re", "out_im");

  const torch::stable::accelerator::DeviceGuard device_guard(
      out_re.get_device_index());
  constexpr int block_size = 256;
  fmcw_spectrum_forward_kernel<<<
      sample_grid(bins, segments, chirps, block_size),
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
      bins,
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
    int64_t num_bins,
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
  const int bins = checked_int(num_bins, "num_bins");
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(chirps > 0, "num_chirps must be positive.");
  STD_TORCH_CHECK(bins > 0, "num_bins must be positive.");
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
      bins,
      "bins",
      "tan_out_re",
      "tan_out_im");

  const torch::stable::accelerator::DeviceGuard device_guard(
      tan_out_re.get_device_index());
  constexpr int block_size = 256;
  fmcw_spectrum_jvp_kernel<<<
      sample_grid(bins, segments, chirps, block_size),
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
      bins,
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
    int64_t num_bins,
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
  const int bins = checked_int(num_bins, "num_bins");
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(chirps > 0, "num_chirps must be positive.");
  STD_TORCH_CHECK(bins > 0, "num_bins must be positive.");
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
      bins,
      "bins",
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
      bins,
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
