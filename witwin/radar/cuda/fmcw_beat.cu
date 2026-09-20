// Complex FMCW beat synthesis; the phase equation is owned by fmcw_phase.cuh.
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
// Beat samples are unnormalized coherent sums over the weighted path rows.
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
#include "path_interpolation.cuh"

#include <cstdint>
#include <cmath>

namespace {

struct BeatPhase {
  float sin_phi;
  float cos_phi;
  // Phase derivatives at fixed slot and ADC time; see the shared phase owner.
  double dphi_dtau_rt;
  double dphi_dtau_rate;
};

// Phase of one path at one (chirp, sample) grid point. The cycle count is a
// large number  -  hundreds of cycles for a metre-scale target  -  so it is
// formed and wrapped in double before it is handed to the single-precision
// trigonometric unit.
__device__ __forceinline__ BeatPhase beat_phase(
    const double tau0, const double rate, const double t_slot, const double t_m,
    const double slope, const double carrier_hz, const double carrier_rate_hz, const double t_start) {
  const auto terms = fmcw_phase_terms(tau0, rate, t_slot, t_start+t_m, slope, carrier_hz, carrier_rate_hz);
  const double frac = terms.cycles - floor(terms.cycles);
  float sin_phi, cos_phi;
  sincosf(static_cast<float>(kTwoPiD * frac), &sin_phi, &cos_phi);
  return {sin_phi, cos_phi, terms.d_tau, terms.d_rate};
}

__global__ void fmcw_beat_forward_kernel(
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
    const int num_samples,
    const double sample_period_s,
    const double chirp_period_s,
    const double slope,
    const double carrier_hz,
    const double carrier_rate_hz,
    const double t_start) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int chirp = blockIdx.z;
  if (sample >= num_samples || segment >= num_segments) {
    return;
  }

  const SegmentBounds bounds = segment_bounds(path_offsets, segment, num_paths);

  const double t_slot = slot_time(
      chirp,
      clamped_tx_index(segment_tx_index, segment, num_tx),
      num_tx,
      chirp_period_s);
  const double t_m = static_cast<double>(sample) * sample_period_s;

  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = bounds.start; k < bounds.end; ++k) {
    const double tau = static_cast<double>(tau_rt[k]);
    const double rate = static_cast<double>(tau_rate[k]);
    const BeatPhase phase = beat_phase(
        tau, rate, t_slot, t_m, slope, carrier_hz, carrier_rate_hz, t_start);
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
    const int num_samples,
    const double sample_period_s,
    const double chirp_period_s,
    const double slope,
    const double carrier_hz,
    const double carrier_rate_hz,
    const double t_start) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int chirp = blockIdx.z;
  if (sample >= num_samples || segment >= num_segments) {
    return;
  }

  const SegmentBounds bounds = segment_bounds(path_offsets, segment, num_paths);

  const double t_slot = slot_time(
      chirp,
      clamped_tx_index(segment_tx_index, segment, num_tx),
      num_tx,
      chirp_period_s);
  const double t_m = static_cast<double>(sample) * sample_period_s;

  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = bounds.start; k < bounds.end; ++k) {
    const double tau = static_cast<double>(tau_rt[k]);
    const double rate = static_cast<double>(tau_rate[k]);
    const BeatPhase phase = beat_phase(
        tau, rate, t_slot, t_m, slope, carrier_hz, carrier_rate_hz, t_start);
    const float w_re = weight_re[k];
    const float w_im = weight_im[k];
    const float re = w_re * phase.cos_phi - w_im * phase.sin_phi;
    const float im = w_re * phase.sin_phi + w_im * phase.cos_phi;

    const double dphi_d = phase.dphi_dtau_rt * static_cast<double>(tan_tau_rt[k]) +
        phase.dphi_dtau_rate * static_cast<double>(tan_tau_rate[k]);
    const float dphi = static_cast<float>(dphi_d);
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
    const int num_samples,
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

  // Same backstop. On the production route path_segment is DERIVED from the
  // same offsets table by assembly.segment_of_each_row, so it cannot disagree
  // with it.
  int64_t segment = path_segment[k];
  segment = segment < 0 ? 0 : segment;
  segment = segment >= num_segments ? num_segments - 1 : segment;

  const double base_tau = static_cast<double>(tau_rt[k]);
  const double rate = static_cast<double>(tau_rate[k]);
  const float w_re = weight_re[k];
  const float w_im = weight_im[k];
  const int tx_index = clamped_tx_index(
      segment_tx_index, static_cast<int>(segment), num_tx);

  double d_tau_rt = 0.0;
  double d_tau_rate = 0.0;
  double d_w_re = 0.0;
  double d_w_im = 0.0;

  for (int chirp = 0; chirp < num_chirps; ++chirp) {
    const double t_slot = slot_time(chirp, tx_index, num_tx, chirp_period_s);
    const double tau = base_tau;
    const int64_t row_base =
        (static_cast<int64_t>(chirp) * num_segments + segment) * num_samples;
    for (int sample = 0; sample < num_samples; ++sample) {
      const double t_m = static_cast<double>(sample) * sample_period_s;
      const BeatPhase phase = beat_phase(
          tau, rate, t_slot, t_m, slope, carrier_hz, carrier_rate_hz, t_start);
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
      d_tau_rt += d_phi * phase.dphi_dtau_rt;
      d_tau_rate += d_phi * phase.dphi_dtau_rate;
    }
  }

  grad_tau_rt[k] = static_cast<float>(d_tau_rt);
  grad_tau_rate[k] = static_cast<float>(d_tau_rate);
  grad_weight_re[k] = static_cast<float>(d_w_re);
  grad_weight_im[k] = static_cast<float>(d_w_im);
}

}  // namespace

void fmcw_beat_forward_cuda(
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
  check_output(out_re, out_im, chirps, segments, samples, "samples", "out_re", "out_im");

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

void fmcw_beat_jvp_cuda(
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
      "samples",
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

void fmcw_beat_backward_cuda(
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
      "samples",
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

// Refreshed rows at arbitrary ADC times. x=[tau(s), Re(weight), Im(weight),
// chirp-local u(s)]; carrier already belongs to the conjugated Channel weight
// in scene synthesis. Each CSR segment is one observation/sensor pair. The
// phase owner and float sin/cos rounding are shared with the regular beat
// kernel above. Time is a fixed schedule, not an AD input. Oracle:
// test_fmcw_observations.py (independent complex phase and directional AD).
struct ObservationValue { BeatPhase phase; double re,im; };

__device__ __forceinline__ ObservationValue observation_value(
    double tau, double re, double im, double time, double slope, double carrier) {
  const auto p=beat_phase(tau,0,0,time,slope,carrier,0,0);
  return {p,re*p.cos_phi-im*p.sin_phi,re*p.sin_phi+im*p.cos_phi};
}

__device__ __forceinline__ void observation_vjp(
    ObservationValue value, double gr, double gi, double out[3]) {
  const auto p=value.phase;
  out[0]=p.dphi_dtau_rt*(-gr*value.im+gi*value.re);
  out[1]=gr*p.cos_phi+gi*p.sin_phi;
  out[2]=-gr*p.sin_phi+gi*p.cos_phi;
}

__device__ __forceinline__ void observation_jvp(
    ObservationValue value, const double d[3], double& re, double& im) {
  const auto p=value.phase;
  re+=d[1]*p.cos_phi-d[2]*p.sin_phi-value.im*p.dphi_dtau_rt*d[0];
  im+=d[1]*p.sin_phi+d[2]*p.cos_phi+value.re*p.dphi_dtau_rt*d[0];
}

template<int Mode>
__global__ void fmcw_observation_kernel(const double* x, const int64_t* offsets,
    const int64_t* segment, const double* v, double* out, int64_t rows,
    int64_t segments, double slope, double carrier) {
  const int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i >= (Mode==1 ? rows : segments)) return;
  if constexpr(Mode==1) {
    const int64_t owner=segment[i];
    if(owner<0 || owner>=segments) { for(int k=0;k<4;++k) out[4*i+k]=0; return; }
    const double* a=x+4*i;
    const auto value=observation_value(a[0],a[1],a[2],a[3],slope,carrier);
    observation_vjp(value,v[2*owner],v[2*owner+1],out+4*i);
    out[4*i+3]=0;
  } else {
    const int64_t lo=offsets[i], hi=offsets[i+1];
    double r=0,j=0;
    if(lo>=0 && hi>=lo && hi<=rows) for(int64_t k=lo;k<hi;++k) {
      const double* a=x+4*k;
      const auto value=observation_value(a[0],a[1],a[2],a[3],slope,carrier);
      if constexpr(Mode==0) { r+=value.re; j+=value.im; }
      else {
        const double* d=v+4*k;
        observation_jvp(value,d,r,j);
      }
    }
    out[2*i]=r; out[2*i+1]=j;
  }
}

template<int Mode>
void fmcw_observation_run(const torch::stable::Tensor& x, const torch::stable::Tensor& offsets,
    const torch::stable::Tensor& segment, const torch::stable::Tensor& v,
    torch::stable::Tensor& out, double slope, double carrier) {
  const torch::stable::Tensor* payloads[3]={&x,&v,&out};
  for(const auto* t : payloads) {
    STD_TORCH_CHECK(t->is_cuda() && t->is_contiguous() &&
      t->scalar_type()==torch::headeronly::ScalarType::Double,
      "observation payloads require contiguous CUDA doubles");
    STD_TORCH_CHECK(t->get_device_index()==x.get_device_index(), "observation devices differ");
  }
  for(const auto* t : {&offsets,&segment}) {
    check_cuda_long(*t,"observation routing");
    STD_TORCH_CHECK(t->get_device_index()==x.get_device_index(), "observation routing devices differ");
  }
  STD_TORCH_CHECK(x.dim()==2 && x.size(1)==4, "observation x must be [N,4]");
  STD_TORCH_CHECK(offsets.dim()==1 && offsets.numel()>=2, "observation offsets must be [segments+1]");
  const int64_t rows=x.size(0), segments=offsets.numel()-1;
  STD_TORCH_CHECK(segment.numel()==rows, "observation segment count mismatch");
  STD_TORCH_CHECK(out.numel()==(Mode==1 ? rows*4 : segments*2), "observation output shape");
  STD_TORCH_CHECK(v.numel()==(Mode==0 ? 0 : Mode==1 ? segments*2 : rows*4), "observation vector shape");
  STD_TORCH_CHECK(std::isfinite(slope) && std::isfinite(carrier), "observation phase constants must be finite");
  const torch::stable::accelerator::DeviceGuard guard(x.get_device_index());
  const int64_t n=Mode==1 ? rows : segments;
  if(n) {
    fmcw_observation_kernel<Mode><<<(n+255)/256,256,0,current_cuda_stream(x)>>>(
      x.const_data_ptr<double>(),offsets.const_data_ptr<int64_t>(),segment.const_data_ptr<int64_t>(),
      v.const_data_ptr<double>(),out.mutable_data_ptr<double>(),rows,segments,slope,carrier);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// Compact adaptive observations. Each output is one (ADC instant, active RX)
// segment. Its node starts and pair-local bounds address the retained probes
// directly; no observation-by-path payload is allocated. Interpolation and
// beat numerics are the same helpers as the standalone operators above.
struct AdaptivePathNodes {
  const double* samples;
  const int64_t* starts;
  const double* basis;
  int64_t local, sample_count;
  __device__ InterpolationNode operator()(int64_t j) const {
    const int64_t row=starts[j]+local;
    if(row<0 || row>=sample_count) return {0,0,0,0};
    const double* a=samples+3*row;
    return {a[0],a[1],a[2],basis[j]};
  }
};

__device__ __forceinline__ double float_round(double value) {
  // Preserve the existing interpolation -> float32 -> synthesis boundary.
  // Keeping all intermediates double would change the model's phase inputs.
  return static_cast<double>(static_cast<float>(value));
}

template<int Mode>
__global__ void fmcw_adaptive_kernel(const double* samples, const int32_t* valid,
    const int64_t* starts, const double* basis, const int64_t* bounds,
    const int64_t* owners, const double* clock, const double* vector, double* out,
    int64_t sample_count, int64_t nodes, int64_t bound_rows, int64_t pairs,
    int64_t begin, int64_t end, int64_t num_tx, int64_t num_samples,
    double interpolation_carrier, double slope, double carrier) {
  const int64_t receiver_count=pairs/num_tx;
  const int64_t segment=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(segment>=(end-begin)*receiver_count) return;
  const int64_t observation=begin+segment/receiver_count;
  const int64_t pair=(segment%receiver_count)*num_tx+(observation/num_samples)%num_tx;
  const int64_t owner=owners[observation];
  double re=0,im=0;
  if(owner>=0 && owner<bound_rows) {
    const int64_t lo=bounds[owner*(pairs+1)+pair], hi=bounds[owner*(pairs+1)+pair+1];
    if(lo>=0 && hi>=lo && hi<=sample_count) for(int64_t local=lo;local<hi;++local) {
      const AdaptivePathNodes read{samples,starts+observation*nodes,basis+observation*nodes,local,sample_count};
      const auto interpolated=interpolate_path(read,nodes,interpolation_carrier);
      const int64_t first=read.starts[0]+local;
      const bool alive=first>=0 && first<sample_count && valid[first]!=0;
      // Channel -> dechirped beat conjugation occurs exactly once, after
      // interpolation, just as in the unfused FMCW facade. Dead coefficients
      // are zeroed after interpolation, before phase evaluation.
      const double wr=alive ? float_round(interpolated.re) : 0;
      const double wi=alive ? -float_round(interpolated.im) : 0;
      const auto value=observation_value(float_round(interpolated.tau),wr,wi,clock[observation],slope,carrier);
      if constexpr(Mode==0) {
        // Serial path order and double accumulation match CSR synthesis.
        re+=value.re; im+=value.im;
      } else if constexpr(Mode==1) {
        double grad[3];
        observation_vjp(value,vector[2*segment],vector[2*segment+1],grad);
        grad[0]=float_round(grad[0]);
        grad[1]=alive ? float_round(grad[1]) : 0;
        grad[2]=alive ? -float_round(grad[2]) : 0;
        for(int64_t j=0;j<nodes;++j) {
          const int64_t row=read.starts[j]+local;
          if(row<0 || row>=sample_count) continue;
          double jac[3][3];
          interpolation_jacobian(read(j),interpolated,interpolation_carrier,jac);
          for(int col=0;col<3;++col) {
            double sum=0;
            for(int r=0;r<3;++r) sum+=jac[r][col]*grad[r];
            // Several queries share a retained probe. Double accumulation
            // matches the indexed interpolant's scatter-add precision.
            atomicAdd(out+3*row+col,sum);
          }
        }
      } else {
        const AdaptivePathNodes tangent{vector,read.starts,read.basis,local,sample_count};
        double direction[3]={0,0,0};
        for(int64_t j=0;j<nodes;++j) {
          double jac[3][3];
          interpolation_jacobian(read(j),interpolated,interpolation_carrier,jac);
          const auto d=tangent(j);
          const double dv[3]={d.tau,d.re,d.im};
          for(int r=0;r<3;++r) for(int col=0;col<3;++col) direction[r]+=jac[r][col]*dv[col];
        }
        direction[0]=float_round(direction[0]);
        direction[1]=alive ? float_round(direction[1]) : 0;
        direction[2]=alive ? -float_round(direction[2]) : 0;
        observation_jvp(value,direction,re,im);
      }
    }
  }
  if constexpr(Mode!=1) { out[2*segment]=re; out[2*segment+1]=im; }
}

template<int Mode>
void fmcw_adaptive_run(const torch::stable::Tensor& samples, const torch::stable::Tensor& valid,
    const torch::stable::Tensor& starts, const torch::stable::Tensor& basis,
    const torch::stable::Tensor& bounds, const torch::stable::Tensor& owners,
    const torch::stable::Tensor& clock, const torch::stable::Tensor& vector,
    torch::stable::Tensor& out, int64_t begin, int64_t end, int64_t num_tx,
    int64_t num_samples, double interpolation_carrier, double slope, double carrier) {
  const torch::stable::Tensor* payloads[5]={&samples,&basis,&clock,&vector,&out};
  for(const auto* t : payloads) {
    STD_TORCH_CHECK(t->is_cuda() && t->is_contiguous() &&
      t->scalar_type()==torch::headeronly::ScalarType::Double,
      "adaptive FMCW payloads require contiguous CUDA doubles");
    STD_TORCH_CHECK(t->get_device_index()==samples.get_device_index(), "adaptive FMCW devices differ");
  }
  for(const auto* t : {&starts,&bounds,&owners}) {
    check_cuda_long(*t,"adaptive FMCW routing");
    STD_TORCH_CHECK(t->get_device_index()==samples.get_device_index(), "adaptive FMCW routing devices differ");
  }
  check_cuda_int(valid,"adaptive FMCW validity");
  STD_TORCH_CHECK(valid.get_device_index()==samples.get_device_index(), "adaptive FMCW validity device differs");
  STD_TORCH_CHECK(samples.dim()==2 && samples.size(1)==3, "adaptive FMCW samples must be [S,3]");
  STD_TORCH_CHECK(valid.dim()==1 && valid.numel()==samples.size(0), "adaptive FMCW validity shape");
  STD_TORCH_CHECK(starts.dim()==2 && starts.size(1)>=2, "adaptive FMCW starts must be [observations,K], K>=2");
  const int64_t observations=starts.size(0), nodes=starts.size(1);
  STD_TORCH_CHECK(basis.dim()==2 && basis.size(0)==observations && basis.size(1)==nodes,
    "adaptive FMCW basis must match starts");
  STD_TORCH_CHECK(owners.dim()==1 && owners.numel()==observations && clock.dim()==1 && clock.numel()==observations,
    "adaptive FMCW schedule shape");
  STD_TORCH_CHECK(bounds.dim()==2 && bounds.size(1)>=2 && num_tx>0 && num_samples>0,
    "adaptive FMCW pair layout");
  const int64_t pairs=bounds.size(1)-1;
  STD_TORCH_CHECK(pairs%num_tx==0, "adaptive FMCW pairs must partition by transmitter");
  STD_TORCH_CHECK(begin>=0 && end>=begin && end<=observations, "adaptive FMCW observation bounds");
  const int64_t segments=(end-begin)*(pairs/num_tx);
  STD_TORCH_CHECK(out.numel()==(Mode==1 ? samples.numel() : segments*2), "adaptive FMCW output shape");
  STD_TORCH_CHECK(vector.numel()==(Mode==0 ? 0 : Mode==1 ? segments*2 : samples.numel()),
    "adaptive FMCW vector shape");
  STD_TORCH_CHECK(std::isfinite(interpolation_carrier) && interpolation_carrier>0 &&
    std::isfinite(slope) && std::isfinite(carrier), "adaptive FMCW phase constants");
  const torch::stable::accelerator::DeviceGuard guard(samples.get_device_index());
  if(segments) {
    fmcw_adaptive_kernel<Mode><<<(segments+255)/256,256,0,current_cuda_stream(samples)>>>(
      samples.const_data_ptr<double>(),valid.const_data_ptr<int32_t>(),starts.const_data_ptr<int64_t>(),
      basis.const_data_ptr<double>(),bounds.const_data_ptr<int64_t>(),owners.const_data_ptr<int64_t>(),
      clock.const_data_ptr<double>(),vector.const_data_ptr<double>(),out.mutable_data_ptr<double>(),
      samples.size(0),nodes,bounds.size(0),pairs,begin,end,num_tx,num_samples,interpolation_carrier,slope,carrier);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("fmcw_adaptive_forward", TORCH_BOX(&fmcw_adaptive_run<0>));
  m.impl("fmcw_adaptive_backward", TORCH_BOX(&fmcw_adaptive_run<1>));
  m.impl("fmcw_adaptive_jvp", TORCH_BOX(&fmcw_adaptive_run<2>));
  m.impl("fmcw_observation_forward", TORCH_BOX(&fmcw_observation_run<0>));
  m.impl("fmcw_observation_backward", TORCH_BOX(&fmcw_observation_run<1>));
  m.impl("fmcw_observation_jvp", TORCH_BOX(&fmcw_observation_run<2>));
  m.impl("fmcw_beat_forward", TORCH_BOX(&fmcw_beat_forward_cuda));
  m.impl("fmcw_beat_backward", TORCH_BOX(&fmcw_beat_backward_cuda));
  m.impl("fmcw_beat_jvp", TORCH_BOX(&fmcw_beat_jvp_cuda));
}
