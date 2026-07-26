// Pulsed echo train over the (pulse, fast-time sample) grid.
//
// This is the Phase-6 pulsed synthesis primitive. Like the beat and CFR
// families it is registered in the existing `witwin_radar_dirichlet_cuda`
// library because the packaging chain assumes a single native artifact stem;
// the physical rename is Phase-10 work and is recorded in R-ADR-004.
//
// Closed form, stated verbatim:
//
//   t_l         = l * pri_s
//   drift_k(l)  = tau_rate[k] * t_l
//   tau_k(l)    = tau_rt[k] + drift_k(l)
//   u           = range_gate_start_s + m * sample_period_s - tau_k(l)
//
//   cycles_k(l) = - ( carrier_rate_hz * drift_k(l) + carrier_hz * tau_k(l) )
//
//   y[l][p][m]  = sum_{k in segment p} C[k] * p(u)
//                 * exp(+j * 2 * pi * cycles_k(l))
//
// with the two analytic unit-energy pulses, `A = pulse_amplitude = 1/sqrt(T_p)`:
//
//   rect: p(u) = A                                   for 0 <= u < T_p, else 0
//   lfm : p(u) = A * exp(+j * pi * B * u^2 / T_p)     for 0 <= u < T_p, else 0
//
// The support is HALF-OPEN, and that is a contract rather than an accident of
// writing the comparison one way. A closed support puts exactly one extra
// sample inside the pulse whenever the delay lands on the grid and not
// otherwise, so the received pulse would be 501 samples long at one delay and
// 500 at the next. The matched filter's replica has a fixed length, so that one
// sample is a mismatched tap: it costs about 0.2 percent of the peak magnitude
// and biases the estimated delay by nearly two thousandths of a sample, at one
// delay in every five hundred. Half-open makes the sampled pulse exactly
// `round(T_p / T_s)` samples long at EVERY delay, and it leaves the continuous
// unit-energy integral unchanged because a single point has measure zero.
//
// THE POINT OF THIS FILE: `u` is CONTINUOUS. The pulse is evaluated at the exact
// fractional delay from its analytic form and is never snapped to the nearest
// sample. Snapping quantises the delay by `sample_period_s / 2`, which at
// 50 MSPS is 10 ns and three metres of range, and it destroys the closed form
// every acceptance assertion is written against. That is why both supported
// pulse kinds are analytic functions of a real argument: there is no lookup
// table, no gather, and no interpolation in this kernel, and adding one would be
// a different design rather than an optimisation.
//
// The structural parallel to the other two waveforms is exact and deliberate.
// Only one factor differs between them:
//
//   fmcw_beat.cu   exp(+j 2 pi S tau t_m)      the dechirped ramp
//   ofdm_cfr.cu    exp(-j 2 pi n df tau)       the subcarrier phase
//   pulsed_echo.cu p(t_g + m T_s - tau)        the pulse envelope
//
// The slow-time factor - the carrier applied to the delay CHANGE - is identical
// in all three. That is what a shared input contract buys, and it is why this
// kernel needs no notion of where its weight came from.
//
// The sign is NEGATIVE inside `cycles`, so the published train is in the CHANNEL
// phasor convention `exp(-j k d)` under `exp(+j 2 pi f t)` time dependence, like
// the OFDM CFR cube and unlike the conjugated FMCW beat cube. There is no
// de-chirping here, so there is nothing to conjugate and no conversion site
// anywhere in this family.
//
// What this kernel emits is the matched-filter INPUT. The matched filter itself
// is a correlation with the conjugated replica and lives in DSP glue
// (witwin/radar/sigproc/matched_filter.py) under the plan's Torch/FFT exception.
// Synthesis owns the received waveform; processing owns the filter. Fusing the
// correlation in here would bake a modelling choice - which replica, which
// window, which oversampling - into the physics.
//
// The carrier has the same two homes as in the other two families and exactly
// one of them may be nonzero:
//
//   carrier_hz = f_ref, carrier_rate_hz = 0   the kernel owns the whole absolute
//     carrier phase, which multiplies the FULL tau_k(l) and therefore already
//     walks across pulses.
//   carrier_hz = 0, carrier_rate_hz = f_ref   the production path for a
//     Channel-sourced weight, which already carries exp(-j 2 pi f_ref tau_rt) at
//     the frozen per-frame delay. carrier_rate_hz supplies the inter-pulse
//     Doppler term the frozen weight cannot express.
//
// Dropping that rate term is worse here than in either other waveform. The
// envelope carries no carrier at all, so what survives is only the LFM's own
// phase moving with the drifting envelope position, `B u / T_p` against
// `f_ref` - a factor of about 3.9e3 understatement at the far end of a 20 MHz,
// 10 us sweep at 77 GHz, and EXACTLY ZERO at the leading edge and for a
// rectangular pulse, where the Doppler disappears completely while the train
// still looks entirely reasonable. The Python contract refuses a spec that names
// the carrier in both homes.
//
// Differentiability. The four differentiable inputs are tau_rt, tau_rate,
// weight_re, and weight_im, and the geometry enters only through the phase:
//
//   du/dtau_rt      = -1
//   du/dtau_rate    = -t_l
//   d|p|/du         = 0 almost everywhere, for BOTH kinds
//   d(arg p)/du     = 2 pi B u / T_p for the LFM, 0 for the rectangle
//
//   dphi/dtau_rt    = -2 pi (carrier_hz + B u / T_p)
//   dphi/dtau_rate  = -2 pi t_l (carrier_hz + carrier_rate_hz + B u / T_p)
//
// The rectangle's two support edges are a genuine non-differentiability. This
// kernel returns the ALMOST-EVERYWHERE derivative there: the envelope gradient
// is exactly zero at an edge sample, not a delta. A finite difference that
// straddles an edge will disagree with that, and the disagreement is correct
// behaviour rather than a defect - the finite-difference oracle must avoid the
// edge, and the analytic AD tests use the LFM.
//
// Numerics, identical to `fmcw_beat.cu` and `ofdm_cfr.cu` and not re-derived:
// the LFM's own cycle count and the carrier's are summed in double and wrapped
// to [0, 1) before a SINGLE `sincosf`, fast math stays off in this translation
// unit, and no window is applied. Any range or Doppler window is a downstream
// DSP choice and is not in the synthesis contract.

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

// Pulse-kind selector, mirroring PULSE_KIND_RECT / PULSE_KIND_LFM on the Python
// spec. An integer rather than a template parameter because the alternative is
// two more instantiations of three kernels for a branch that is uniform across
// the whole launch and costs nothing.
constexpr int kPulseRect = 0;
constexpr int kPulseLfm = 1;

struct PulseTerm {
  float sin_phi;
  float cos_phi;
  // 1/sqrt(T_p) inside the pulse support, exactly 0 outside it. The support
  // test is the ONLY branch in the inner loop.
  float amplitude;
  // d(phi) / d(tau_rt) in radians per second, at fixed (t_l, m).
  double dphi_dtau_rt;
  // d(phi) / d(tau_rate) in radians. This is NOT `dphi_dtau_rt * t_l`:
  // carrier_rate_hz multiplies the drift, which depends on tau_rate but not on
  // tau_rt, so the rate derivative carries an extra `carrier_rate_hz * t_l`
  // that the base-delay derivative does not.
  double dphi_dtau_rate;
};

// One path's contribution at one (pulse, sample) grid point. The cycle count is
// large when the kernel owns the carrier - thousands of cycles for a metre-scale
// target at 77 GHz - and the LFM's own phase reaches B * T_p / 2 = 100 cycles,
// so both are formed and wrapped in double before the single-precision
// trigonometric unit sees them.
__device__ __forceinline__ PulseTerm pulse_term(
    const double tau,
    const double tau_drift,
    const double t_l,
    const double u,
    const int pulse_kind,
    const double pulse_width_s,
    const double bandwidth_hz,
    const double pulse_amplitude,
    const double carrier_hz,
    const double carrier_rate_hz) {
  PulseTerm term;
  if (!(u >= 0.0 && u < pulse_width_s)) {
    // Outside the envelope support the whole contribution is zero, primal and
    // derivative alike. Returning zeros rather than an edge delta is the
    // almost-everywhere derivative this family documents.
    term.sin_phi = 0.0f;
    term.cos_phi = 0.0f;
    term.amplitude = 0.0f;
    term.dphi_dtau_rt = 0.0;
    term.dphi_dtau_rate = 0.0;
    return term;
  }

  double cycles = -(carrier_rate_hz * tau_drift + carrier_hz * tau);
  double chirp_hz = 0.0;
  if (pulse_kind == kPulseLfm) {
    // pi * B * u^2 / T_p radians is B * u^2 / (2 * T_p) cycles.
    cycles += 0.5 * bandwidth_hz * u * u / pulse_width_s;
    chirp_hz = bandwidth_hz * u / pulse_width_s;
  }
  const double frac = cycles - floor(cycles);
  sincosf(static_cast<float>(kTwoPiD * frac), &term.sin_phi, &term.cos_phi);
  term.amplitude = static_cast<float>(pulse_amplitude);
  term.dphi_dtau_rt = -kTwoPiD * (carrier_hz + chirp_hz);
  term.dphi_dtau_rate =
      -kTwoPiD * t_l * (carrier_hz + carrier_rate_hz + chirp_hz);
  return term;
}

__global__ void pulsed_echo_forward_kernel(
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
    const double pri_s,
    const double range_gate_start_s,
    const int pulse_kind,
    const double pulse_width_s,
    const double bandwidth_hz,
    const double pulse_amplitude,
    const double carrier_hz,
    const double carrier_rate_hz) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int pulse = blockIdx.z;
  if (sample >= num_samples || segment >= num_segments) {
    return;
  }

  // A memory-safety backstop, not a validation policy: the host wrapper checks
  // the table's SHAPE but never reads its values, because doing so per frame
  // would be the D2H the fixed-topology capability exists to avoid.
  int64_t start = path_offsets[segment];
  int64_t end = path_offsets[segment + 1];
  start = start < 0 ? 0 : start;
  end = end > num_paths ? num_paths : end;

  const double t_l = static_cast<double>(pulse) * pri_s;
  const double t_fast =
      range_gate_start_s + static_cast<double>(sample) * sample_period_s;

  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = start; k < end; ++k) {
    const double drift = static_cast<double>(tau_rate[k]) * t_l;
    const double tau = static_cast<double>(tau_rt[k]) + drift;
    const PulseTerm term = pulse_term(
        tau,
        drift,
        t_l,
        t_fast - tau,
        pulse_kind,
        pulse_width_s,
        bandwidth_hz,
        pulse_amplitude,
        carrier_hz,
        carrier_rate_hz);
    const float w_re = weight_re[k];
    const float w_im = weight_im[k];
    acc_re += term.amplitude * (w_re * term.cos_phi - w_im * term.sin_phi);
    acc_im += term.amplitude * (w_re * term.sin_phi + w_im * term.cos_phi);
  }

  const int64_t out_idx =
      (static_cast<int64_t>(pulse) * num_segments + segment) * num_samples +
      sample;
  out_re[out_idx] = acc_re;
  out_im[out_idx] = acc_im;
}

__global__ void pulsed_echo_jvp_kernel(
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
    const double pri_s,
    const double range_gate_start_s,
    const int pulse_kind,
    const double pulse_width_s,
    const double bandwidth_hz,
    const double pulse_amplitude,
    const double carrier_hz,
    const double carrier_rate_hz) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int pulse = blockIdx.z;
  if (sample >= num_samples || segment >= num_segments) {
    return;
  }

  int64_t start = path_offsets[segment];
  int64_t end = path_offsets[segment + 1];
  start = start < 0 ? 0 : start;
  end = end > num_paths ? num_paths : end;

  const double t_l = static_cast<double>(pulse) * pri_s;
  const double t_fast =
      range_gate_start_s + static_cast<double>(sample) * sample_period_s;

  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = start; k < end; ++k) {
    const double drift = static_cast<double>(tau_rate[k]) * t_l;
    const double tau = static_cast<double>(tau_rt[k]) + drift;
    const PulseTerm term = pulse_term(
        tau,
        drift,
        t_l,
        t_fast - tau,
        pulse_kind,
        pulse_width_s,
        bandwidth_hz,
        pulse_amplitude,
        carrier_hz,
        carrier_rate_hz);
    const float w_re = weight_re[k];
    const float w_im = weight_im[k];
    const float re =
        term.amplitude * (w_re * term.cos_phi - w_im * term.sin_phi);
    const float im =
        term.amplitude * (w_re * term.sin_phi + w_im * term.cos_phi);

    // The envelope MAGNITUDE has zero derivative almost everywhere for both
    // pulse kinds, so the geometry tangent is a pure rotation. Outside the
    // support the amplitude is zero and so are both terms below.
    const double dphi_d =
        term.dphi_dtau_rt * static_cast<double>(tan_tau_rt[k]) +
        term.dphi_dtau_rate * static_cast<double>(tan_tau_rate[k]);
    const float dphi = static_cast<float>(dphi_d);
    const float tw_re = tan_weight_re[k];
    const float tw_im = tan_weight_im[k];
    acc_re +=
        term.amplitude * (tw_re * term.cos_phi - tw_im * term.sin_phi) -
        dphi * im;
    acc_im +=
        term.amplitude * (tw_re * term.sin_phi + tw_im * term.cos_phi) +
        dphi * re;
  }

  const int64_t out_idx =
      (static_cast<int64_t>(pulse) * num_segments + segment) * num_samples +
      sample;
  tan_out_re[out_idx] = acc_re;
  tan_out_im[out_idx] = acc_im;
}

// One thread per path, looping the whole (pulse, sample) grid. Each path owns
// exactly one output slot in each gradient array, so the reduction needs no
// atomics and the summation order is fixed by the loop nest.
__global__ void pulsed_echo_backward_kernel(
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
    const int num_pulses,
    const int num_samples,
    const double sample_period_s,
    const double pri_s,
    const double range_gate_start_s,
    const int pulse_kind,
    const double pulse_width_s,
    const double bandwidth_hz,
    const double pulse_amplitude,
    const double carrier_hz,
    const double carrier_rate_hz) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_paths) {
    return;
  }

  // Same backstop. On the production route path_segment is DERIVED from the
  // same offsets table, so it cannot disagree with it.
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

  for (int pulse = 0; pulse < num_pulses; ++pulse) {
    const double t_l = static_cast<double>(pulse) * pri_s;
    const double drift = rate * t_l;
    const double tau = base_tau + drift;
    const int64_t row_base =
        (static_cast<int64_t>(pulse) * num_segments + segment) * num_samples;
    for (int sample = 0; sample < num_samples; ++sample) {
      const double t_fast =
          range_gate_start_s + static_cast<double>(sample) * sample_period_s;
      const PulseTerm term = pulse_term(
          tau,
          drift,
          t_l,
          t_fast - tau,
          pulse_kind,
          pulse_width_s,
          bandwidth_hz,
          pulse_amplitude,
          carrier_hz,
          carrier_rate_hz);
      const float g_re = grad_out_re[row_base + sample];
      const float g_im = grad_out_im[row_base + sample];
      const float re =
          term.amplitude * (w_re * term.cos_phi - w_im * term.sin_phi);
      const float im =
          term.amplitude * (w_re * term.sin_phi + w_im * term.cos_phi);

      d_w_re += term.amplitude *
          (static_cast<double>(g_re) * term.cos_phi +
           static_cast<double>(g_im) * term.sin_phi);
      d_w_im += term.amplitude *
          (-static_cast<double>(g_re) * term.sin_phi +
           static_cast<double>(g_im) * term.cos_phi);

      const double d_phi =
          -static_cast<double>(g_re) * im + static_cast<double>(g_im) * re;
      d_tau_rt += d_phi * term.dphi_dtau_rt;
      d_tau_rate += d_phi * term.dphi_dtau_rate;
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
    int num_pulses,
    int num_segments,
    int num_samples,
    const char* name_re,
    const char* name_im) {
  check_cuda_float(out_re, name_re);
  check_cuda_float(out_im, name_im);
  STD_TORCH_CHECK(
      out_re.sizes().equals(out_im.sizes()),
      "Pulsed echo output components must have the same shape.");
  STD_TORCH_CHECK(
      out_re.dim() == 3,
      "Pulsed echo output must have shape (pulses, segments, samples).");
  STD_TORCH_CHECK(
      out_re.size(0) == num_pulses && out_re.size(1) == num_segments &&
          out_re.size(2) == num_samples,
      "Pulsed echo output shape disagrees with the declared grid.");
}

void check_pulse(int pulse_kind, double pulse_width_s, double pulse_amplitude) {
  STD_TORCH_CHECK(
      pulse_kind == kPulseRect || pulse_kind == kPulseLfm,
      "pulse_kind must be 0 (rect) or 1 (lfm).");
  STD_TORCH_CHECK(pulse_width_s > 0.0, "pulse_width_s must be positive.");
  STD_TORCH_CHECK(pulse_amplitude > 0.0, "pulse_amplitude must be positive.");
}

dim3 sample_grid(
    int num_samples,
    int num_segments,
    int num_pulses,
    int block) {
  return dim3((num_samples + block - 1) / block, num_segments, num_pulses);
}

}  // namespace

void pulsed_echo_forward_cuda(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_offsets,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    int64_t num_paths,
    int64_t num_segments,
    int64_t num_pulses,
    int64_t num_samples,
    double sample_period_s,
    double pri_s,
    double range_gate_start_s,
    int64_t pulse_kind,
    double pulse_width_s,
    double bandwidth_hz,
    double pulse_amplitude,
    double carrier_hz,
    double carrier_rate_hz) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int pulses = checked_int(num_pulses, "num_pulses");
  const int samples = checked_int(num_samples, "num_samples");
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(pulses > 0, "num_pulses must be positive.");
  STD_TORCH_CHECK(samples > 0, "num_samples must be positive.");
  check_pulse(
      checked_int(pulse_kind, "pulse_kind"), pulse_width_s, pulse_amplitude);
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths);
  check_cuda_long(path_offsets, "path_offsets");
  STD_TORCH_CHECK(
      path_offsets.numel() == static_cast<int64_t>(segments) + 1,
      "path_offsets must hold num_segments + 1 values.");
  check_output(out_re, out_im, pulses, segments, samples, "out_re", "out_im");

  const torch::stable::accelerator::DeviceGuard device_guard(
      out_re.get_device_index());
  constexpr int block_size = 256;
  pulsed_echo_forward_kernel<<<
      sample_grid(samples, segments, pulses, block_size),
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
      pri_s,
      range_gate_start_s,
      static_cast<int>(pulse_kind),
      pulse_width_s,
      bandwidth_hz,
      pulse_amplitude,
      carrier_hz,
      carrier_rate_hz);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void pulsed_echo_jvp_cuda(
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
    int64_t num_pulses,
    int64_t num_samples,
    double sample_period_s,
    double pri_s,
    double range_gate_start_s,
    int64_t pulse_kind,
    double pulse_width_s,
    double bandwidth_hz,
    double pulse_amplitude,
    double carrier_hz,
    double carrier_rate_hz) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int pulses = checked_int(num_pulses, "num_pulses");
  const int samples = checked_int(num_samples, "num_samples");
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(pulses > 0, "num_pulses must be positive.");
  STD_TORCH_CHECK(samples > 0, "num_samples must be positive.");
  check_pulse(
      checked_int(pulse_kind, "pulse_kind"), pulse_width_s, pulse_amplitude);
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
      pulses,
      segments,
      samples,
      "tan_out_re",
      "tan_out_im");

  const torch::stable::accelerator::DeviceGuard device_guard(
      tan_out_re.get_device_index());
  constexpr int block_size = 256;
  pulsed_echo_jvp_kernel<<<
      sample_grid(samples, segments, pulses, block_size),
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
      pri_s,
      range_gate_start_s,
      static_cast<int>(pulse_kind),
      pulse_width_s,
      bandwidth_hz,
      pulse_amplitude,
      carrier_hz,
      carrier_rate_hz);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void pulsed_echo_backward_cuda(
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
    int64_t num_pulses,
    int64_t num_samples,
    double sample_period_s,
    double pri_s,
    double range_gate_start_s,
    int64_t pulse_kind,
    double pulse_width_s,
    double bandwidth_hz,
    double pulse_amplitude,
    double carrier_hz,
    double carrier_rate_hz) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int pulses = checked_int(num_pulses, "num_pulses");
  const int samples = checked_int(num_samples, "num_samples");
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(pulses > 0, "num_pulses must be positive.");
  STD_TORCH_CHECK(samples > 0, "num_samples must be positive.");
  check_pulse(
      checked_int(pulse_kind, "pulse_kind"), pulse_width_s, pulse_amplitude);
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths);
  check_cuda_long(path_segment, "path_segment");
  STD_TORCH_CHECK(
      path_segment.numel() == static_cast<int64_t>(paths),
      "path_segment must hold one segment index per path.");
  check_output(
      grad_out_re,
      grad_out_im,
      pulses,
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
  pulsed_echo_backward_kernel<<<
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
      pulses,
      samples,
      sample_period_s,
      pri_s,
      range_gate_start_s,
      static_cast<int>(pulse_kind),
      pulse_width_s,
      bandwidth_hz,
      pulse_amplitude,
      carrier_hz,
      carrier_rate_hz);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(witwin_radar_dirichlet_cuda, CUDA, m) {
  m.impl("pulsed_echo_forward", TORCH_BOX(&pulsed_echo_forward_cuda));
  m.impl("pulsed_echo_backward", TORCH_BOX(&pulsed_echo_backward_cuda));
  m.impl("pulsed_echo_jvp", TORCH_BOX(&pulsed_echo_jvp_cuda));
}
