// Pulsed echo train over the (pulse, fast-time sample) grid.
//
// Closed form, stated verbatim:
//
//   t_l         = l * pri_s
//   drift_k(l)  = tau_rate[k] * t_l
//   tau_k(l)    = tau_rt[k] + drift_k(l)
//   t_m         = t_fast + m * sample_period_s
//   u           = t_m - tau_k(l)                      (wrapped into [0, T_p) when periodic)
//   f_l         = l * frequency_step_hz               (the stepped carrier offset)
//
//   cycles_k(l) = - ( carrier_rate_hz * drift_k(l) + (carrier_hz + f_l) * tau_k(l) )
//
//   y[l][p][m]  = sum_{k in segment p} C[k] * s(u) * exp(+j * 2 * pi * cycles_k(l))
//
// `t_fast` is `range_gate_start_s` for every segment, or, when the per-segment
// table is given, `segment_table[p][0]` with `f_l = segment_table[p][1]`. The
// table is how one launch synthesizes segments that are different fast-time
// instants of a refreshed frame: each segment's value is the one a launch with
// that gate start and that frequency offset would produce. It is schedule data
// and carries no derivative.
//
// The envelope s(u), with `A = pulse_amplitude = 1` (R-ADR-029: a sample with
// |s| = 1 is C[k] in sqrt(W) at the receive port, so |s| <= 1 everywhere):
//
//   rect : s(u) = A                                          0 <= u < T_p
//   lfm  : s(u) = A * exp(+j * pi * B * u^2 / T_p)            0 <= u < T_p
//   coded: s(u) = A * a_i * exp(+j * 2 * pi * f_i * (u - i T_c)),
//          i = floor(u / T_c), T_c = T_p / num_chips          0 <= u < T_p
//
// and zero outside. The coded table holds, per chip, `|a_i|`, `arg(a_i) / 2pi`
// and `f_i` [Hz], so a binary phase code (PMCW), a frequency-hopped code (FSK)
// and a user-sampled complex envelope held for one chip each (a zero-order-hold
// DAC) are one family. `periodic` makes the envelope the periodic extension of
// one code period, which is a continuously transmitted code: every delay is
// then inside the support. R-ADR-046 records the coded and stepped families.
//
// The support is HALF-OPEN, so the sampled pulse is exactly round(T_p / T_s)
// samples long at every delay and the matched filter's fixed-length replica
// never meets a mismatched tap. `u` is CONTINUOUS: the envelope is evaluated at
// the exact fractional delay and never snapped to a sample.
//
// The sign is NEGATIVE inside `cycles`, so the published train is in the
// CHANNEL phasor convention `exp(-j k d)` under `exp(+j 2 pi f t)` time
// dependence. There is no de-chirping, so there is nothing to conjugate.
//
// The carrier has two homes and exactly one is nonzero: `carrier_hz = f_ref`
// (the kernel owns the absolute phase, which multiplies the full delay and so
// walks by itself) or `carrier_rate_hz = f_ref` (a Channel weight already holds
// exp(-j 2 pi f_ref tau_rt); the rate term supplies its slow-time change). The
// stepped offset `f_l` is never in a Channel weight, which is evaluated at
// f_ref, so it multiplies the FULL delay in both homes, like an OFDM
// subcarrier.
//
// Differentiability. The differentiable inputs are tau_rt, tau_rate, weight_re
// and weight_im; the geometry enters through u and the carrier:
//
//   d|s|/du       = 0 almost everywhere, for every kind
//   d(arg s)/du   = 2 pi * nu(u), nu = B u / T_p (lfm), f_i (coded), 0 (rect)
//   dphi/dtau_rt   = -2 pi (carrier_hz + f_l + nu)
//   dphi/dtau_rate = -2 pi t_l (carrier_hz + carrier_rate_hz + f_l + nu)
//
// The support edges and the chip boundaries are genuine discontinuities; the
// kernel returns the ALMOST-EVERYWHERE derivative (zero envelope gradient,
// never a delta). A finite-difference oracle must keep each sample off them.
//
// Numerics: cycle counts are summed in double and wrapped to [0, 1) before a
// single `sincosf`; fast math stays off; no window is applied. What this file
// emits is the matched-filter INPUT; the filter lives in processing.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cuda_runtime.h>
#include "radar_checks.cuh"

#include <cstdint>

namespace {

// Pulse-kind selector, mirroring the Python spec's constants.
constexpr int kPulseRect = 0;
constexpr int kPulseLfm = 1;
constexpr int kPulseCoded = 2;

// The envelope description every operator of this family reads, packed once by
// the host wrapper.
struct Envelope {
  int kind;
  double width_s;
  double bandwidth_hz;
  double amplitude;
  // float64 [num_chips, 3]: |a_i|, arg(a_i) / 2pi, f_i [Hz]. Null unless coded.
  const double* chips;
  int num_chips;
  bool periodic;
};

// The fast-time schedule: a uniform gate and stepped carrier, or one
// (gate start, frequency offset) pair per segment.
struct Schedule {
  double gate_start_s;
  double sample_period_s;
  double pri_s;
  double frequency_step_hz;
  // float64 [num_segments, 2]: t_fast [s], f_l [Hz]. Null for the uniform grid.
  const double* segment_table;
};

struct PulseTerm {
  float sin_phi;
  float cos_phi;
  // |s(u)| inside the support, exactly 0 outside it.
  float amplitude;
  // d(phi) / d(tau_rt) in radians per second, at fixed (t_l, m).
  double dphi_dtau_rt;
  // d(phi) / d(tau_rate) in radians. Not `dphi_dtau_rt * t_l`: the carrier
  // rate multiplies the drift only, so this carries an extra rate term.
  double dphi_dtau_rate;
};

__device__ __forceinline__ double segment_fast_start(const Schedule& schedule, const int segment) {
  return schedule.segment_table == nullptr ? schedule.gate_start_s : schedule.segment_table[2 * segment];
}

__device__ __forceinline__ double segment_frequency(const Schedule& schedule, const int segment, const int pulse) {
  return schedule.segment_table == nullptr ? static_cast<double>(pulse) * schedule.frequency_step_hz
                                           : schedule.segment_table[2 * segment + 1];
}

// One path's contribution at one (pulse, sample) grid point. Cycle counts are
// formed and wrapped in double: a kernel-owned carrier is thousands of cycles,
// and an LFM's own phase reaches B T_p / 2.
__device__ __forceinline__ PulseTerm pulse_term(
    const double tau,
    const double tau_drift,
    const double t_l,
    double u,
    const double f_l,
    const Envelope& envelope,
    const double carrier_hz,
    const double carrier_rate_hz) {
  PulseTerm term{0.0f, 0.0f, 0.0f, 0.0, 0.0};
  if (envelope.periodic) {
    u -= floor(u / envelope.width_s) * envelope.width_s;
  }
  if (!(u >= 0.0 && u < envelope.width_s)) {
    return term;
  }

  double cycles = -(carrier_rate_hz * tau_drift + (carrier_hz + f_l) * tau);
  double nu = 0.0;
  double magnitude = 1.0;
  if (envelope.kind == kPulseLfm) {
    // pi B u^2 / T_p radians is B u^2 / (2 T_p) cycles.
    cycles += 0.5 * envelope.bandwidth_hz * u * u / envelope.width_s;
    nu = envelope.bandwidth_hz * u / envelope.width_s;
  } else if (envelope.kind == kPulseCoded) {
    const double chip_width = envelope.width_s / static_cast<double>(envelope.num_chips);
    int chip = static_cast<int>(floor(u / chip_width));
    chip = chip < 0 ? 0 : (chip >= envelope.num_chips ? envelope.num_chips - 1 : chip);
    const double* entry = envelope.chips + 3 * chip;
    magnitude = entry[0];
    nu = entry[2];
    cycles += entry[1] + nu * (u - static_cast<double>(chip) * chip_width);
  }
  const double frac = cycles - floor(cycles);
  sincosf(static_cast<float>(kTwoPiD * frac), &term.sin_phi, &term.cos_phi);
  term.amplitude = static_cast<float>(envelope.amplitude * magnitude);
  term.dphi_dtau_rt = -kTwoPiD * (carrier_hz + f_l + nu);
  term.dphi_dtau_rate = -kTwoPiD * t_l * (carrier_hz + carrier_rate_hz + f_l + nu);
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
    const Envelope envelope,
    const Schedule schedule,
    const double carrier_hz,
    const double carrier_rate_hz) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int pulse = blockIdx.z;
  if (sample >= num_samples || segment >= num_segments) {
    return;
  }

  const SegmentBounds bounds = segment_bounds(path_offsets, segment, num_paths);
  const double t_l = static_cast<double>(pulse) * schedule.pri_s;
  const double t_m = segment_fast_start(schedule, segment) + static_cast<double>(sample) * schedule.sample_period_s;
  const double f_l = segment_frequency(schedule, segment, pulse);

  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = bounds.start; k < bounds.end; ++k) {
    const double drift = static_cast<double>(tau_rate[k]) * t_l;
    const double tau = static_cast<double>(tau_rt[k]) + drift;
    const PulseTerm term = pulse_term(tau, drift, t_l, t_m - tau, f_l, envelope, carrier_hz, carrier_rate_hz);
    const float w_re = weight_re[k];
    const float w_im = weight_im[k];
    acc_re += term.amplitude * (w_re * term.cos_phi - w_im * term.sin_phi);
    acc_im += term.amplitude * (w_re * term.sin_phi + w_im * term.cos_phi);
  }

  const int64_t out_idx = (static_cast<int64_t>(pulse) * num_segments + segment) * num_samples + sample;
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
    const Envelope envelope,
    const Schedule schedule,
    const double carrier_hz,
    const double carrier_rate_hz) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int pulse = blockIdx.z;
  if (sample >= num_samples || segment >= num_segments) {
    return;
  }

  const SegmentBounds bounds = segment_bounds(path_offsets, segment, num_paths);
  const double t_l = static_cast<double>(pulse) * schedule.pri_s;
  const double t_m = segment_fast_start(schedule, segment) + static_cast<double>(sample) * schedule.sample_period_s;
  const double f_l = segment_frequency(schedule, segment, pulse);

  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = bounds.start; k < bounds.end; ++k) {
    const double drift = static_cast<double>(tau_rate[k]) * t_l;
    const double tau = static_cast<double>(tau_rt[k]) + drift;
    const PulseTerm term = pulse_term(tau, drift, t_l, t_m - tau, f_l, envelope, carrier_hz, carrier_rate_hz);
    const float w_re = weight_re[k];
    const float w_im = weight_im[k];
    const float re = term.amplitude * (w_re * term.cos_phi - w_im * term.sin_phi);
    const float im = term.amplitude * (w_re * term.sin_phi + w_im * term.cos_phi);
    // |s| has zero derivative almost everywhere, so the geometry tangent is a
    // pure rotation.
    const double dphi_d = term.dphi_dtau_rt * static_cast<double>(tan_tau_rt[k]) +
        term.dphi_dtau_rate * static_cast<double>(tan_tau_rate[k]);
    const float dphi = static_cast<float>(dphi_d);
    const float tw_re = tan_weight_re[k];
    const float tw_im = tan_weight_im[k];
    acc_re += term.amplitude * (tw_re * term.cos_phi - tw_im * term.sin_phi) - dphi * im;
    acc_im += term.amplitude * (tw_re * term.sin_phi + tw_im * term.cos_phi) + dphi * re;
  }

  const int64_t out_idx = (static_cast<int64_t>(pulse) * num_segments + segment) * num_samples + sample;
  tan_out_re[out_idx] = acc_re;
  tan_out_im[out_idx] = acc_im;
}

// One thread per path, looping its segment's whole (pulse, sample) grid. Each
// path owns exactly one output slot in each gradient array, so the reduction
// needs no atomics and the summation order is fixed by the loop nest.
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
    const Envelope envelope,
    const Schedule schedule,
    const double carrier_hz,
    const double carrier_rate_hz) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_paths) {
    return;
  }

  // On the production route path_segment is DERIVED from the offsets table,
  // so it cannot disagree with it; the clamp is a memory-safety backstop.
  int64_t segment = path_segment[k];
  segment = segment < 0 ? 0 : segment;
  segment = segment >= num_segments ? num_segments - 1 : segment;
  const int seg = static_cast<int>(segment);

  const double base_tau = static_cast<double>(tau_rt[k]);
  const double rate = static_cast<double>(tau_rate[k]);
  const float w_re = weight_re[k];
  const float w_im = weight_im[k];
  const double fast_start = segment_fast_start(schedule, seg);

  double d_tau_rt = 0.0;
  double d_tau_rate = 0.0;
  double d_w_re = 0.0;
  double d_w_im = 0.0;

  for (int pulse = 0; pulse < num_pulses; ++pulse) {
    const double t_l = static_cast<double>(pulse) * schedule.pri_s;
    const double drift = rate * t_l;
    const double tau = base_tau + drift;
    const double f_l = segment_frequency(schedule, seg, pulse);
    const int64_t row_base = (static_cast<int64_t>(pulse) * num_segments + segment) * num_samples;
    for (int sample = 0; sample < num_samples; ++sample) {
      const double t_m = fast_start + static_cast<double>(sample) * schedule.sample_period_s;
      const PulseTerm term = pulse_term(tau, drift, t_l, t_m - tau, f_l, envelope, carrier_hz, carrier_rate_hz);
      const float g_re = grad_out_re[row_base + sample];
      const float g_im = grad_out_im[row_base + sample];
      const float re = term.amplitude * (w_re * term.cos_phi - w_im * term.sin_phi);
      const float im = term.amplitude * (w_re * term.sin_phi + w_im * term.cos_phi);

      d_w_re += term.amplitude * (static_cast<double>(g_re) * term.cos_phi + static_cast<double>(g_im) * term.sin_phi);
      d_w_im += term.amplitude * (-static_cast<double>(g_re) * term.sin_phi + static_cast<double>(g_im) * term.cos_phi);

      const double d_phi = -static_cast<double>(g_re) * im + static_cast<double>(g_im) * re;
      d_tau_rt += d_phi * term.dphi_dtau_rt;
      d_tau_rate += d_phi * term.dphi_dtau_rate;
    }
  }

  grad_tau_rt[k] = static_cast<float>(d_tau_rt);
  grad_tau_rate[k] = static_cast<float>(d_tau_rate);
  grad_weight_re[k] = static_cast<float>(d_w_re);
  grad_weight_im[k] = static_cast<float>(d_w_im);
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
  STD_TORCH_CHECK(out_re.sizes().equals(out_im.sizes()), "Pulsed echo output components must have the same shape.");
  STD_TORCH_CHECK(out_re.dim() == 3, "Pulsed echo output must have shape (pulses, segments, samples).");
  STD_TORCH_CHECK(
      out_re.size(0) == num_pulses && out_re.size(1) == num_segments && out_re.size(2) == num_samples,
      "Pulsed echo output shape disagrees with the declared grid.");
}

void check_cuda_double(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Double, name, " must have dtype torch.float64.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

// Validate the envelope table and pack the envelope the kernels read.
Envelope make_envelope(
    const torch::stable::Tensor& chips,
    int64_t pulse_kind,
    double pulse_width_s,
    double bandwidth_hz,
    double pulse_amplitude,
    int64_t periodic) {
  const int kind = checked_int(pulse_kind, "pulse_kind");
  STD_TORCH_CHECK(
      kind == kPulseRect || kind == kPulseLfm || kind == kPulseCoded, "pulse_kind must be 0 (rect), 1 (lfm) or 2 (coded).");
  STD_TORCH_CHECK(pulse_width_s > 0.0, "pulse_width_s must be positive.");
  STD_TORCH_CHECK(pulse_amplitude > 0.0, "pulse_amplitude must be positive.");
  check_cuda_double(chips, "chips");
  Envelope envelope{kind, pulse_width_s, bandwidth_hz, pulse_amplitude, nullptr, 0, periodic != 0};
  if (kind == kPulseCoded) {
    STD_TORCH_CHECK(chips.dim() == 2 && chips.size(1) == 3 && chips.size(0) > 0,
                    "a coded pulse needs a chips table of shape (num_chips, 3).");
    envelope.num_chips = checked_int(chips.size(0), "num_chips");
    envelope.chips = chips.const_data_ptr<double>();
  } else {
    STD_TORCH_CHECK(chips.numel() == 0, "only a coded pulse takes a chips table.");
  }
  return envelope;
}

Schedule make_schedule(
    const torch::stable::Tensor& segment_table,
    int segments,
    double range_gate_start_s,
    double sample_period_s,
    double pri_s,
    double frequency_step_hz) {
  check_cuda_double(segment_table, "segment_table");
  Schedule schedule{range_gate_start_s, sample_period_s, pri_s, frequency_step_hz, nullptr};
  if (segment_table.numel() != 0) {
    STD_TORCH_CHECK(
        segment_table.dim() == 2 && segment_table.size(0) == segments && segment_table.size(1) == 2,
        "segment_table must have shape (num_segments, 2).");
    schedule.segment_table = segment_table.const_data_ptr<double>();
  }
  return schedule;
}

void check_grid(int segments, int pulses, int samples) {
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(pulses > 0, "num_pulses must be positive.");
  STD_TORCH_CHECK(samples > 0, "num_samples must be positive.");
}

}  // namespace

void pulsed_echo_forward_cuda(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_offsets,
    const torch::stable::Tensor& chips,
    const torch::stable::Tensor& segment_table,
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
    double carrier_rate_hz,
    double frequency_step_hz,
    int64_t periodic) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int pulses = checked_int(num_pulses, "num_pulses");
  const int samples = checked_int(num_samples, "num_samples");
  check_grid(segments, pulses, samples);
  const Envelope envelope = make_envelope(chips, pulse_kind, pulse_width_s, bandwidth_hz, pulse_amplitude, periodic);
  const Schedule schedule =
      make_schedule(segment_table, segments, range_gate_start_s, sample_period_s, pri_s, frequency_step_hz);
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths);
  check_cuda_long(path_offsets, "path_offsets");
  STD_TORCH_CHECK(
      path_offsets.numel() == static_cast<int64_t>(segments) + 1, "path_offsets must hold num_segments + 1 values.");
  check_output(out_re, out_im, pulses, segments, samples, "out_re", "out_im");

  const torch::stable::accelerator::DeviceGuard device_guard(out_re.get_device_index());
  pulsed_echo_forward_kernel<<<
      sample_grid(samples, segments, pulses, kBlock),
      dim3(kBlock, 1, 1),
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
      envelope,
      schedule,
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
    const torch::stable::Tensor& chips,
    const torch::stable::Tensor& segment_table,
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
    double carrier_rate_hz,
    double frequency_step_hz,
    int64_t periodic) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int pulses = checked_int(num_pulses, "num_pulses");
  const int samples = checked_int(num_samples, "num_samples");
  check_grid(segments, pulses, samples);
  const Envelope envelope = make_envelope(chips, pulse_kind, pulse_width_s, bandwidth_hz, pulse_amplitude, periodic);
  const Schedule schedule =
      make_schedule(segment_table, segments, range_gate_start_s, sample_period_s, pri_s, frequency_step_hz);
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths);
  check_path_inputs(tan_tau_rt, tan_tau_rate, tan_weight_re, tan_weight_im, paths);
  check_cuda_long(path_offsets, "path_offsets");
  STD_TORCH_CHECK(
      path_offsets.numel() == static_cast<int64_t>(segments) + 1, "path_offsets must hold num_segments + 1 values.");
  check_output(tan_out_re, tan_out_im, pulses, segments, samples, "tan_out_re", "tan_out_im");

  const torch::stable::accelerator::DeviceGuard device_guard(tan_out_re.get_device_index());
  pulsed_echo_jvp_kernel<<<
      sample_grid(samples, segments, pulses, kBlock),
      dim3(kBlock, 1, 1),
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
      envelope,
      schedule,
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
    const torch::stable::Tensor& chips,
    const torch::stable::Tensor& segment_table,
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
    double carrier_rate_hz,
    double frequency_step_hz,
    int64_t periodic) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int pulses = checked_int(num_pulses, "num_pulses");
  const int samples = checked_int(num_samples, "num_samples");
  check_grid(segments, pulses, samples);
  const Envelope envelope = make_envelope(chips, pulse_kind, pulse_width_s, bandwidth_hz, pulse_amplitude, periodic);
  const Schedule schedule =
      make_schedule(segment_table, segments, range_gate_start_s, sample_period_s, pri_s, frequency_step_hz);
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths);
  check_cuda_long(path_segment, "path_segment");
  STD_TORCH_CHECK(path_segment.numel() == static_cast<int64_t>(paths), "path_segment must hold one segment index per path.");
  check_output(grad_out_re, grad_out_im, pulses, segments, samples, "grad_out_re", "grad_out_im");
  check_path_inputs(grad_tau_rt, grad_tau_rate, grad_weight_re, grad_weight_im, paths);

  if (paths == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(grad_tau_rt.get_device_index());
  pulsed_echo_backward_kernel<<<linear_grid(paths), dim3(kBlock, 1, 1), 0, current_cuda_stream(grad_tau_rt)>>>(
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
      envelope,
      schedule,
      carrier_hz,
      carrier_rate_hz);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("pulsed_echo_forward", TORCH_BOX(&pulsed_echo_forward_cuda));
  m.impl("pulsed_echo_backward", TORCH_BOX(&pulsed_echo_backward_cuda));
  m.impl("pulsed_echo_jvp", TORCH_BOX(&pulsed_echo_jvp_cuda));
}
