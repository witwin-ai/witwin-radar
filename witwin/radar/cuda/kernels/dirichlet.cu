// Dirichlet range-spectrum family.
//
// Closed form, verbatim, per path row i and output bin `bin`:
//
//   tau  = tau_is_seconds ? input[i] : 2 * input[i] / c0     (ROUND TRIP, s)
//   phi0 = 2 pi (fc * tau + slope * tau * (t_start - tau/2))
//   k0   = input[i] * k0_per_meter                = f_beat * n_fft / fs
//   x    = 2 pi (bin - k0) / n_fft
//   D(x) = [sin((n + 0.5) x) / sin(x/2)] * exp(-j n x)
//   spectrum[bin] = sum_i (a_re[i] + j a_im[i]) * D(x) * exp(+j phi0)
//
// With n = (N - 1)/2 this is algebraically the exact zero-padded n_fft-point
// DFT of the fmcw_beat.cu fast-time sample sequence at carrier_hz = fc,
// carrier_rate_hz = 0, tau_rate = 0. tests/test_phase6_real_compat_identity.py
// pins that identity in float64 on the CPU.
//
// Two switches, both additive and both defaulting to the legacy behaviour:
//
//   tau_is_seconds = 0  keeps the historical assumption that the input is a
//     ONE-WAY distance in metres and the round trip is monostatic. A caller
//     that already holds a round-trip delay passes 1 and supplies the matching
//     k0 scale, `slope * n_fft / fs`, instead of `(slope * 2 / c0) * n_fft / fs`.
//     The two are the same number, since tau = 2 d / c0.
//
//   fc is the carrier home, mirroring carrier_hz in the beat family. fc != 0
//     means this kernel owns the absolute phase 2 pi fc tau; fc == 0 means the
//     weight already carries it and the kernel applies none. A Channel
//     coefficient carries it, so pairing one with fc != 0 double counts the
//     carrier - which is why the Python contract refuses that combination
//     before any launch rather than leaving it to a reviewer.
//
// The complex weight is (a_re, a_im) as two real tensors, matching the beat and
// join families: no complex tensor crosses the autograd boundary, so the
// conjugate-Wirtinger convention cannot be got wrong at the seam. A real
// weight is the special case a_im = 0, and it is BIT-identical to the
// pre-complex kernel because every added term is a separate accumulation of
// exactly zero rather than a rewrite of the existing one.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

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

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
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

// The round-trip delay the phase terms are built from. The legacy branch keeps
// the literal expression `2.0f * value / kC0` so that a tau_is_seconds = 0
// caller gets exactly the float32 rounding it got before this flag existed.
__device__ __forceinline__ float round_trip_delay(const float value, const int tau_is_seconds) {
  return (tau_is_seconds != 0) ? value : 2.0f * value / kC0;
}

// 2 pi * d(tau)/d(input). Written as the two literal legacy/seconds constants
// rather than as one scaled expression, for the same rounding reason.
__device__ __forceinline__ float delay_phase_scale(const int tau_is_seconds) {
  return (tau_is_seconds != 0) ? kTwoPi : kTwoPi * 2.0f / kC0;
}

// The floor of the linear frame update's range-loss denominator, in whatever
// unit the path value is expressed in. It is a MICROMETRE, and the seconds
// branch is the same micrometre expressed as a round-trip delay.
//
// This has to follow the unit. The constant used to be the literal 1e-6f, which
// is a sensible guard for a distance in metres and is LARGER than every delay a
// radar ever sees: a 60 metre round trip is 4e-7 seconds, so a metres-scale
// clamp applied to a delay clamps every single row and turns the range-loss
// factor into `tau / 1e-6`, off by seven orders of magnitude. The legacy branch
// keeps the literal so a tau_is_seconds = 0 caller is unchanged.
__device__ __forceinline__ float min_path_value(const int tau_is_seconds) {
  return (tau_is_seconds != 0) ? (2.0f * 1e-6f / kC0) : 1e-6f;
}

__device__ __forceinline__ Complex path_response(
    const float distance,
    const float n,
    const float k0_per_meter,
    const int n_fft,
    const int bin,
    const float fc,
    const float slope,
    const float t_start,
    const int tau_is_seconds) {
  const float tau = round_trip_delay(distance, tau_is_seconds);
  const float phi0 = kTwoPi * (fc * tau + slope * tau * (t_start - 0.5f * tau));
  const float k0 = distance * k0_per_meter;
  const float x = kTwoPi * (static_cast<float>(bin) - k0) / static_cast<float>(n_fft);
  return cmul(dirichlet_kernel(x, n), cexp_f(phi0));
}

__global__ void forward_chunked_kernel(
    const float* __restrict__ d,
    const float* __restrict__ a,
    const float* __restrict__ a_im,
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
    const float t_start,
    const int tau_is_seconds) {
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
    const float amp_im = a_im[i];
    if (amp == 0.0f && amp_im == 0.0f) {
      continue;
    }
    const Complex response =
        path_response(d[i], n, k0_per_meter, n_fft, bin, fc, slope, t_start, tau_is_seconds);
    // The real part first, in exactly the statements the real-only kernel used,
    // then the imaginary contribution as a separate accumulation. A fused
    // `amp * re - amp_im * im` would be a different rounding, and the
    // real-compatibility criterion asks for bit equality, not closeness.
    sum_re += amp * response.re;
    sum_im += amp * response.im;
    sum_re -= amp_im * response.im;
    sum_im += amp_im * response.re;
  }

  const int out_idx = chunk_idx * num_bins + bin;
  output_re[out_idx] = sum_re;
  output_im[out_idx] = sum_im;
}

// JVP of forward_chunked. Differentiable in the path input (distance or delay,
// per tau_is_seconds) and in both weight components; every other argument is a
// configuration scalar.
//
//   d(out)/d(input) = (a_re + j a_im) * dR/d(input)
//   dR/d(input)     = D'(x) exp(+j phi0) dx/d(input)
//                   + j R * dphi0/d(input)
//
// with dx/d(input) = -2 pi k0_per_meter / n_fft, which is the same pair of
// terms the backward kernels contract against a cotangent.
__global__ void dirichlet_jvp_kernel(
    const float* __restrict__ d,
    const float* __restrict__ a,
    const float* __restrict__ a_im,
    const float* __restrict__ tan_d,
    const float* __restrict__ tan_a,
    const float* __restrict__ tan_a_im,
    float* __restrict__ tan_out_re,
    float* __restrict__ tan_out_im,
    const float n,
    const float k0_per_meter,
    const int num_bins,
    const int n_fft,
    const int num_targets,
    const int targets_per_chunk,
    const float fc,
    const float slope,
    const float t_start,
    const int tau_is_seconds) {
  const int bin = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunk_idx = blockIdx.y;
  if (bin >= num_bins) {
    return;
  }

  const int target_start = chunk_idx * targets_per_chunk;
  const int target_end = min(target_start + targets_per_chunk, num_targets);
  const float dx_dd = -kTwoPi * k0_per_meter / static_cast<float>(n_fft);

  float sum_re = 0.0f;
  float sum_im = 0.0f;
  for (int i = target_start; i < target_end; ++i) {
    const float dist = d[i];
    const float amp = a[i];
    const float amp_im = a_im[i];
    const float td = tan_d[i];
    const float ta = tan_a[i];
    const float ta_im = tan_a_im[i];

    const float tau = round_trip_delay(dist, tau_is_seconds);
    const float phi0 = kTwoPi * (fc * tau + slope * tau * (t_start - 0.5f * tau));
    const Complex phase_corr = cexp_f(phi0);
    const float dphi0_dd = delay_phase_scale(tau_is_seconds) * (fc + slope * t_start - slope * tau);
    const float k0 = dist * k0_per_meter;
    const float x = kTwoPi * (static_cast<float>(bin) - k0) / static_cast<float>(n_fft);
    const Complex result = cmul(dirichlet_kernel(x, n), phase_corr);
    const Complex result_grad = cmul(dirichlet_kernel_grad(x, n), phase_corr);

    // dR/d(input), assembled once and shared by both weight components.
    const Complex response_grad = {
        result_grad.re * dx_dd - result.im * dphi0_dd,
        result_grad.im * dx_dd + result.re * dphi0_dd};

    sum_re += ta * result.re - ta_im * result.im;
    sum_im += ta * result.im + ta_im * result.re;
    sum_re += td * (amp * response_grad.re - amp_im * response_grad.im);
    sum_im += td * (amp * response_grad.im + amp_im * response_grad.re);
  }

  const int out_idx = chunk_idx * num_bins + bin;
  tan_out_re[out_idx] = sum_re;
  tan_out_im[out_idx] = sum_im;
}

__global__ void forward_mimo_linear_chunked_kernel(
    const float* __restrict__ d0,
    const float* __restrict__ d_rate,
    const float* __restrict__ a0,
    const float* __restrict__ a0_im,
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
    const float t_start,
    const int tau_is_seconds) {
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
    float amp_im = a0_im[target_idx];
    if (amp == 0.0f && amp_im == 0.0f) {
      continue;
    }
    if (range_loss_update != 0) {
      const float range_loss = dist0 / fmaxf(dist, min_path_value(tau_is_seconds));
      amp *= range_loss;
      amp_im *= range_loss;
    }

    const Complex response =
        path_response(dist, n, k0_per_meter, n_fft, bin, fc, slope, t_start, tau_is_seconds);
    sum_re += amp * response.re;
    sum_im += amp * response.im;
    sum_re -= amp_im * response.im;
    sum_im += amp_im * response.re;
  }

  const int out_idx = (chirp_id * num_pairs + pair_id) * num_bins + bin;
  output_re[out_idx] = sum_re;
  output_im[out_idx] = sum_im;
}

// AD companions of forward_mimo_linear_chunked.
//
// The as-built forward, per target row i owned by pair p and per chirp c:
//
//   slot        = c * num_tx + tx(p)
//   t_slot      = slot * chirp_period
//   dist        = d0[i] + d_rate[i] * t_slot
//   g           = range_loss_update ? d0[i] / max(dist, 1e-6) : 1
//   out        += (a_re[i] + j a_im[i]) * g * R(dist)
//
// so, writing a = a_re + j a_im and R' = dR/d(dist),
//
//   d(dist)/d(d0)   = 1                d(dist)/d(d_rate) = t_slot
//   dg/d(d0)        = (dist - d0)/dist^2      (range_loss_update, dist > 1e-6)
//   dg/d(d_rate)    = -d0 t_slot / dist^2
//   d(out)/d(d0)    = a (dg/d(d0)     R + g R')
//   d(out)/d(d_rate)= a (dg/d(d_rate) R + g t_slot R')
//   d(out)/d(a_re)  = g R             d(out)/d(a_im)     = j g R
//
// The forward's `amp == 0 && amp_im == 0` skip is a value-preserving shortcut,
// not a branch of the function: d(out)/d(a_re) is g R whatever a is. Neither
// companion reproduces that skip. The `dist <= 0` skip IS a branch - the row
// contributes nothing at all there - and both companions do reproduce it.
//
// The backward owns one gradient slot per target row: a target belongs to
// exactly one pair, so one block per row reducing over (chirp, bin) needs no
// atomics and its summation order is a property of the frozen row set rather
// than of the schedule. The jvp keeps the forward's own (bin, pair, chirp)
// grid, so a tangent costs exactly one launch, like the primal.
__global__ void mimo_linear_backward_kernel(
    const float* __restrict__ d0,
    const float* __restrict__ d_rate,
    const float* __restrict__ a0,
    const float* __restrict__ a0_im,
    const float* __restrict__ grad_output_re,
    const float* __restrict__ grad_output_im,
    float* __restrict__ grad_d0,
    float* __restrict__ grad_d_rate,
    float* __restrict__ grad_a0,
    float* __restrict__ grad_a0_im,
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
    const float t_start,
    const int tau_is_seconds) {
  const int target_idx = blockIdx.x;
  if (target_idx >= num_pairs * targets_per_pair) {
    return;
  }

  const int pair_id = target_idx / targets_per_pair;
  const int num_rx = num_pairs / num_tx;
  const int tx_id = pair_id / num_rx;
  const float dist0 = d0[target_idx];
  const float rate = d_rate[target_idx];
  const float amp = a0[target_idx];
  const float amp_im = a0_im[target_idx];
  const float dx_dd = -kTwoPi * k0_per_meter / static_cast<float>(n_fft);

  float dL_dd0 = 0.0f;
  float dL_drate = 0.0f;
  float dL_da = 0.0f;
  float dL_da_im = 0.0f;

  for (int chirp_id = 0; chirp_id < chirp_per_frame; ++chirp_id) {
    const float slot =
        static_cast<float>(chirp_id) * static_cast<float>(num_tx) + static_cast<float>(tx_id);
    const float chirp_time = slot * chirp_period;
    const float dist = dist0 + rate * chirp_time;
    if (dist <= 0.0f) {
      continue;
    }

    float scale = 1.0f;
    float dscale_dd0 = 0.0f;
    float dscale_drate = 0.0f;
    if (range_loss_update != 0) {
      const float floor_value = min_path_value(tau_is_seconds);
      const float denom = fmaxf(dist, floor_value);
      scale = dist0 / denom;
      if (dist > floor_value) {
        dscale_dd0 = (denom - dist0) / (denom * denom);
        dscale_drate = -dist0 * chirp_time / (denom * denom);
      } else {
        // The clamp is active, so the denominator no longer moves with dist.
        dscale_dd0 = 1.0f / denom;
      }
    }

    const float k0 = dist * k0_per_meter;
    const float tau = round_trip_delay(dist, tau_is_seconds);
    const float phi0 = kTwoPi * (fc * tau + slope * tau * (t_start - 0.5f * tau));
    const Complex phase_corr = cexp_f(phi0);
    const float dphi0_dd = delay_phase_scale(tau_is_seconds) * (fc + slope * t_start - slope * tau);
    const int grad_offset = (chirp_id * num_pairs + pair_id) * num_bins;

    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
      const float gout_re = grad_output_re[grad_offset + bin];
      const float gout_im = grad_output_im[grad_offset + bin];
      const float x = kTwoPi * (static_cast<float>(bin) - k0) / static_cast<float>(n_fft);
      const Complex result = cmul(dirichlet_kernel(x, n), phase_corr);
      const Complex result_grad = cmul(dirichlet_kernel_grad(x, n), phase_corr);

      const float dot0_r = gout_re * result.re + gout_im * result.im;
      const float dot1_r = -gout_re * result.im + gout_im * result.re;
      const float dot0_g = gout_re * result_grad.re + gout_im * result_grad.im;
      const float dot1_g = -gout_re * result_grad.im + gout_im * result_grad.re;
      // <gout, R'> and <gout, j R'>, with R' = D'(x) e^{j phi0} dx/dd + j R dphi0/dd.
      const float dot0_p = dot0_g * dx_dd + dot1_r * dphi0_dd;
      const float dot1_p = dot1_g * dx_dd - dot0_r * dphi0_dd;

      dL_da += scale * dot0_r;
      dL_da_im += scale * dot1_r;
      const float weighted_r = amp * dot0_r + amp_im * dot1_r;
      const float weighted_p = amp * dot0_p + amp_im * dot1_p;
      dL_dd0 += dscale_dd0 * weighted_r + scale * weighted_p;
      dL_drate += dscale_drate * weighted_r + scale * chirp_time * weighted_p;
    }
  }

  // Four independent tree reductions, so each gradient slot has its own order.
  dL_dd0 = warp_sum(dL_dd0);
  dL_drate = warp_sum(dL_drate);
  dL_da = warp_sum(dL_da);
  dL_da_im = warp_sum(dL_da_im);
  __shared__ float warp_grad_d0[8];
  __shared__ float warp_grad_rate[8];
  __shared__ float warp_grad_a[8];
  __shared__ float warp_grad_a_im[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_grad_d0[warp] = dL_dd0;
    warp_grad_rate[warp] = dL_drate;
    warp_grad_a[warp] = dL_da;
    warp_grad_a_im[warp] = dL_da_im;
  }
  __syncthreads();

  if (warp == 0) {
    dL_dd0 = lane < 8 ? warp_grad_d0[lane] : 0.0f;
    dL_drate = lane < 8 ? warp_grad_rate[lane] : 0.0f;
    dL_da = lane < 8 ? warp_grad_a[lane] : 0.0f;
    dL_da_im = lane < 8 ? warp_grad_a_im[lane] : 0.0f;
    dL_dd0 = warp_sum(dL_dd0);
    dL_drate = warp_sum(dL_drate);
    dL_da = warp_sum(dL_da);
    dL_da_im = warp_sum(dL_da_im);
    if (lane == 0) {
      grad_d0[target_idx] = dL_dd0;
      grad_d_rate[target_idx] = dL_drate;
      grad_a0[target_idx] = dL_da;
      grad_a0_im[target_idx] = dL_da_im;
    }
  }
}

__global__ void mimo_linear_jvp_kernel(
    const float* __restrict__ d0,
    const float* __restrict__ d_rate,
    const float* __restrict__ a0,
    const float* __restrict__ a0_im,
    const float* __restrict__ tan_d0,
    const float* __restrict__ tan_d_rate,
    const float* __restrict__ tan_a0,
    const float* __restrict__ tan_a0_im,
    float* __restrict__ tan_out_re,
    float* __restrict__ tan_out_im,
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
    const float t_start,
    const int tau_is_seconds) {
  const int bin = blockIdx.x * blockDim.x + threadIdx.x;
  const int pair_id = blockIdx.y;
  const int chirp_id = blockIdx.z;
  if (bin >= num_bins || pair_id >= num_pairs || chirp_id >= chirp_per_frame) {
    return;
  }

  const int num_rx = num_pairs / num_tx;
  const int tx_id = pair_id / num_rx;
  const float slot =
      static_cast<float>(chirp_id) * static_cast<float>(num_tx) + static_cast<float>(tx_id);
  const float chirp_time = slot * chirp_period;
  const int target_start = pair_id * targets_per_pair;
  const float dx_dd = -kTwoPi * k0_per_meter / static_cast<float>(n_fft);

  float sum_re = 0.0f;
  float sum_im = 0.0f;
  for (int i = 0; i < targets_per_pair; ++i) {
    const int target_idx = target_start + i;
    const float dist0 = d0[target_idx];
    const float dist = dist0 + d_rate[target_idx] * chirp_time;
    if (dist <= 0.0f) {
      continue;
    }

    const float amp = a0[target_idx];
    const float amp_im = a0_im[target_idx];
    const float t_d0 = tan_d0[target_idx];
    const float t_rate = tan_d_rate[target_idx];
    const float t_dist = t_d0 + chirp_time * t_rate;

    float scale = 1.0f;
    float tan_scale = 0.0f;
    if (range_loss_update != 0) {
      const float floor_value = min_path_value(tau_is_seconds);
      const float denom = fmaxf(dist, floor_value);
      scale = dist0 / denom;
      if (dist > floor_value) {
        tan_scale = ((denom - dist0) * t_d0 - dist0 * chirp_time * t_rate) / (denom * denom);
      } else {
        tan_scale = t_d0 / denom;
      }
    }

    const float k0 = dist * k0_per_meter;
    const float tau = round_trip_delay(dist, tau_is_seconds);
    const float phi0 = kTwoPi * (fc * tau + slope * tau * (t_start - 0.5f * tau));
    const Complex phase_corr = cexp_f(phi0);
    const float dphi0_dd = delay_phase_scale(tau_is_seconds) * (fc + slope * t_start - slope * tau);
    const float x = kTwoPi * (static_cast<float>(bin) - k0) / static_cast<float>(n_fft);
    const Complex result = cmul(dirichlet_kernel(x, n), phase_corr);
    const Complex result_grad = cmul(dirichlet_kernel_grad(x, n), phase_corr);
    const Complex response_grad = {
        result_grad.re * dx_dd - result.im * dphi0_dd,
        result_grad.im * dx_dd + result.re * dphi0_dd};

    const float weight_re = scale * tan_a0[target_idx] + tan_scale * amp;
    const float weight_im = scale * tan_a0_im[target_idx] + tan_scale * amp_im;
    sum_re += weight_re * result.re - weight_im * result.im;
    sum_im += weight_re * result.im + weight_im * result.re;

    const float moved = scale * t_dist;
    sum_re += moved * (amp * response_grad.re - amp_im * response_grad.im);
    sum_im += moved * (amp * response_grad.im + amp_im * response_grad.re);
  }

  const int out_idx = (chirp_id * num_pairs + pair_id) * num_bins + bin;
  tan_out_re[out_idx] = sum_re;
  tan_out_im[out_idx] = sum_im;
}

// The imaginary-weight terms are appended as separate accumulations in every
// backward kernel below, for the same reason as in the forward: with a_im = 0
// they contribute exactly zero without perturbing the existing statements, so
// the legacy real path keeps its gradients unchanged.
//
//   dL/da_re = <gout, R>            = gout_re R.re + gout_im R.im
//   dL/da_im = <gout, j R>          = -gout_re R.im + gout_im R.re
//   dL/dd    = a_re <gout, dR/dd> + a_im <gout, j dR/dd>
__global__ void backward_kernel(
    const float* __restrict__ d,
    const float* __restrict__ a,
    const float* __restrict__ a_im,
    const float* __restrict__ grad_output_re,
    const float* __restrict__ grad_output_im,
    float* __restrict__ grad_d,
    float* __restrict__ grad_a,
    float* __restrict__ grad_a_im,
    const float n,
    const float k0_per_meter,
    const int num_bins,
    const int n_fft,
    const int num_targets,
    const float fc,
    const float slope,
    const float t_start,
    const int tau_is_seconds) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_targets) {
    return;
  }

  const float dist = d[i];
  const float amp = a[i];
  const float amp_im = a_im[i];
  const float k0 = dist * k0_per_meter;
  const float tau = round_trip_delay(dist, tau_is_seconds);
  const float phi0 = kTwoPi * (fc * tau + slope * tau * (t_start - 0.5f * tau));
  const Complex phase_corr = cexp_f(phi0);
  const float dphi0_dd = delay_phase_scale(tau_is_seconds) * (fc + slope * t_start - slope * tau);
  const float dx_dd = -kTwoPi * k0_per_meter / static_cast<float>(n_fft);

  float dL_dd = 0.0f;
  float dL_da = 0.0f;
  float dL_da_im = 0.0f;
  for (int bin = 0; bin < num_bins; ++bin) {
    const float gout_re = grad_output_re[bin];
    const float gout_im = grad_output_im[bin];
    const float x = kTwoPi * (static_cast<float>(bin) - k0) / static_cast<float>(n_fft);
    const Complex result = cmul(dirichlet_kernel(x, n), phase_corr);
    const Complex result_grad = cmul(dirichlet_kernel_grad(x, n), phase_corr);

    dL_da += gout_re * result.re + gout_im * result.im;
    dL_dd += amp * (gout_re * result_grad.re + gout_im * result_grad.im) * dx_dd;
    dL_dd += amp * (-gout_re * result.im + gout_im * result.re) * dphi0_dd;
    dL_da_im += -gout_re * result.im + gout_im * result.re;
    dL_dd += amp_im * (-gout_re * result_grad.im + gout_im * result_grad.re) * dx_dd;
    dL_dd += amp_im * (-gout_re * result.re - gout_im * result.im) * dphi0_dd;
  }

  grad_d[i] = dL_dd;
  grad_a[i] = dL_da;
  grad_a_im[i] = dL_da_im;
}

__global__ void backward_batched_kernel(
    const float* __restrict__ d,
    const float* __restrict__ a,
    const float* __restrict__ a_im,
    const float* __restrict__ grad_output_re,
    const float* __restrict__ grad_output_im,
    float* __restrict__ grad_d,
    float* __restrict__ grad_a,
    float* __restrict__ grad_a_im,
    const float n,
    const float k0_per_meter,
    const int num_bins,
    const int n_fft,
    const int num_targets,
    const int targets_per_spectrum,
    const float fc,
    const float slope,
    const float t_start,
    const int tau_is_seconds) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_targets) {
    return;
  }

  const int spectrum_idx = i / targets_per_spectrum;
  const int grad_offset = spectrum_idx * num_bins;
  const float dist = d[i];
  const float amp = a[i];
  const float amp_im = a_im[i];
  const float k0 = dist * k0_per_meter;
  const float tau = round_trip_delay(dist, tau_is_seconds);
  const float phi0 = kTwoPi * (fc * tau + slope * tau * (t_start - 0.5f * tau));
  const Complex phase_corr = cexp_f(phi0);
  const float dphi0_dd = delay_phase_scale(tau_is_seconds) * (fc + slope * t_start - slope * tau);
  const float dx_dd = -kTwoPi * k0_per_meter / static_cast<float>(n_fft);

  float dL_dd = 0.0f;
  float dL_da = 0.0f;
  float dL_da_im = 0.0f;
  for (int bin = 0; bin < num_bins; ++bin) {
    const float gout_re = grad_output_re[grad_offset + bin];
    const float gout_im = grad_output_im[grad_offset + bin];
    const float x = kTwoPi * (static_cast<float>(bin) - k0) / static_cast<float>(n_fft);
    const Complex result = cmul(dirichlet_kernel(x, n), phase_corr);
    const Complex result_grad = cmul(dirichlet_kernel_grad(x, n), phase_corr);

    dL_da += gout_re * result.re + gout_im * result.im;
    dL_dd += amp * (gout_re * result_grad.re + gout_im * result_grad.im) * dx_dd;
    dL_dd += amp * (-gout_re * result.im + gout_im * result.re) * dphi0_dd;
    dL_da_im += -gout_re * result.im + gout_im * result.re;
    dL_dd += amp_im * (-gout_re * result_grad.im + gout_im * result_grad.re) * dx_dd;
    dL_dd += amp_im * (-gout_re * result.re - gout_im * result.im) * dphi0_dd;
  }

  grad_d[i] = dL_dd;
  grad_a[i] = dL_da;
  grad_a_im[i] = dL_da_im;
}

__global__ void backward_parallel_bins_kernel(
    const float* __restrict__ d,
    const float* __restrict__ a,
    const float* __restrict__ a_im,
    const float* __restrict__ grad_output_re,
    const float* __restrict__ grad_output_im,
    float* __restrict__ grad_d,
    float* __restrict__ grad_a,
    float* __restrict__ grad_a_im,
    const float n,
    const float k0_per_meter,
    const int num_bins,
    const int n_fft,
    const int num_targets,
    const float fc,
    const float slope,
    const float t_start,
    const int tau_is_seconds) {
  const int target = blockIdx.x;
  if (target >= num_targets) {
    return;
  }

  const float dist = d[target];
  const float amp = a[target];
  const float amp_im = a_im[target];
  const float k0 = dist * k0_per_meter;
  const float tau = round_trip_delay(dist, tau_is_seconds);
  const float phi0 = kTwoPi * (fc * tau + slope * tau * (t_start - 0.5f * tau));
  const Complex phase_corr = cexp_f(phi0);
  const float dphi0_dd = delay_phase_scale(tau_is_seconds) * (fc + slope * t_start - slope * tau);
  const float dx_dd = -kTwoPi * k0_per_meter / static_cast<float>(n_fft);

  float dL_dd = 0.0f;
  float dL_da = 0.0f;
  float dL_da_im = 0.0f;
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
    const float gout_re = grad_output_re[bin];
    const float gout_im = grad_output_im[bin];
    const float x = kTwoPi * (static_cast<float>(bin) - k0) / static_cast<float>(n_fft);
    const Complex result = cmul(dirichlet_kernel(x, n), phase_corr);
    const Complex result_grad = cmul(dirichlet_kernel_grad(x, n), phase_corr);

    dL_da += gout_re * result.re + gout_im * result.im;
    dL_dd += amp * (gout_re * result_grad.re + gout_im * result_grad.im) * dx_dd;
    dL_dd += amp * (-gout_re * result.im + gout_im * result.re) * dphi0_dd;
    dL_da_im += -gout_re * result.im + gout_im * result.re;
    dL_dd += amp_im * (-gout_re * result_grad.im + gout_im * result_grad.re) * dx_dd;
    dL_dd += amp_im * (-gout_re * result.re - gout_im * result.im) * dphi0_dd;
  }

  // Each gradient slot keeps its own independent tree reduction, so adding the
  // third one changes neither the order nor the partitioning of the first two.
  dL_dd = warp_sum(dL_dd);
  dL_da = warp_sum(dL_da);
  dL_da_im = warp_sum(dL_da_im);
  __shared__ float warp_grad_d[8];
  __shared__ float warp_grad_a[8];
  __shared__ float warp_grad_a_im[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_grad_d[warp] = dL_dd;
    warp_grad_a[warp] = dL_da;
    warp_grad_a_im[warp] = dL_da_im;
  }
  __syncthreads();

  if (warp == 0) {
    dL_dd = lane < 8 ? warp_grad_d[lane] : 0.0f;
    dL_da = lane < 8 ? warp_grad_a[lane] : 0.0f;
    dL_da_im = lane < 8 ? warp_grad_a_im[lane] : 0.0f;
    dL_dd = warp_sum(dL_dd);
    dL_da = warp_sum(dL_da);
    dL_da_im = warp_sum(dL_da_im);
    if (lane == 0) {
      grad_d[target] = dL_dd;
      grad_a[target] = dL_da;
      grad_a_im[target] = dL_da_im;
    }
  }
}

__global__ void backward_per_bin_kernel(
    const float* __restrict__ d,
    const float* __restrict__ a,
    const float* __restrict__ a_im,
    const float* __restrict__ grad_output_re,
    const float* __restrict__ grad_output_im,
    float* __restrict__ grad_d,
    float* __restrict__ grad_a,
    float* __restrict__ grad_a_im,
    const float n,
    const float k0_per_meter,
    const int num_bins,
    const int n_fft,
    const int num_targets,
    const int bins_per_chunk,
    const float fc,
    const float slope,
    const float t_start,
    const int tau_is_seconds) {
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
    const float amp_im = a_im[i];
    const float k0 = dist * k0_per_meter;
    const float tau = round_trip_delay(dist, tau_is_seconds);
    const float phi0 = kTwoPi * (fc * tau + slope * tau * (t_start - 0.5f * tau));
    const Complex phase_corr = cexp_f(phi0);
    const float dphi0_dd = delay_phase_scale(tau_is_seconds) * (fc + slope * t_start - slope * tau);
    const float x = kTwoPi * (static_cast<float>(bin) - k0) / static_cast<float>(n_fft);
    const Complex result = cmul(dirichlet_kernel(x, n), phase_corr);
    const Complex result_grad = cmul(dirichlet_kernel_grad(x, n), phase_corr);
    const float dx_dd = -kTwoPi * k0_per_meter / static_cast<float>(n_fft);

    const float dL_da = gout_re * result.re + gout_im * result.im;
    float dL_dd = amp * (gout_re * result_grad.re + gout_im * result_grad.im) * dx_dd;
    dL_dd += amp * (-gout_re * result.im + gout_im * result.re) * dphi0_dd;
    const float dL_da_im = -gout_re * result.im + gout_im * result.re;
    dL_dd += amp_im * (-gout_re * result_grad.im + gout_im * result_grad.re) * dx_dd;
    dL_dd += amp_im * (-gout_re * result.re - gout_im * result.im) * dphi0_dd;

    const int out_idx = chunk_idx * num_targets + i;
    atomicAdd(grad_d + out_idx, dL_dd);
    atomicAdd(grad_a + out_idx, dL_da);
    atomicAdd(grad_a_im + out_idx, dL_da_im);
  }
}

void check_cuda_float(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Float,
      name,
      " must have dtype torch.float32.");
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

}  // namespace

void forward_chunked_cuda(
    const torch::stable::Tensor& d,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& a_im,
    torch::stable::Tensor& output_re,
    torch::stable::Tensor& output_im,
    double n,
    double k0_per_meter,
    int64_t num_bins,
    int64_t n_fft,
    int64_t num_targets,
    int64_t targets_per_chunk,
    double fc,
    double slope,
    double t_start,
    int64_t tau_is_seconds) {
  check_cuda_float(d, "d");
  check_cuda_float(a, "a");
  check_cuda_float(a_im, "a_im");
  check_cuda_float(output_re, "output_re");
  check_cuda_float(output_im, "output_im");
  STD_TORCH_CHECK(a.sizes().equals(a_im.sizes()), "a and a_im must have the same shape.");
  STD_TORCH_CHECK(
      output_re.sizes().equals(output_im.sizes()),
      "output_re and output_im must have the same shape.");
  STD_TORCH_CHECK(output_re.dim() == 2, "output tensors must have shape (chunks, bins).");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int targets = checked_int(num_targets, "num_targets");
  const int chunk_size = checked_int(targets_per_chunk, "targets_per_chunk");
  const int seconds = checked_int(tau_is_seconds, "tau_is_seconds");
  STD_TORCH_CHECK(chunk_size > 0, "targets_per_chunk must be positive.");

  const torch::stable::accelerator::DeviceGuard device_guard(d.get_device_index());
  constexpr int block_size = 256;
  const dim3 block(block_size, 1, 1);
  const dim3 grid((bins + block_size - 1) / block_size, output_re.size(0), 1);
  forward_chunked_kernel<<<grid, block, 0, current_cuda_stream(d)>>>(
      d.const_data_ptr<float>(),
      a.const_data_ptr<float>(),
      a_im.const_data_ptr<float>(),
      output_re.mutable_data_ptr<float>(),
      output_im.mutable_data_ptr<float>(),
      static_cast<float>(n),
      static_cast<float>(k0_per_meter),
      bins,
      fft,
      targets,
      chunk_size,
      static_cast<float>(fc),
      static_cast<float>(slope),
      static_cast<float>(t_start),
      seconds);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void dirichlet_jvp_cuda(
    const torch::stable::Tensor& d,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& a_im,
    const torch::stable::Tensor& tan_d,
    const torch::stable::Tensor& tan_a,
    const torch::stable::Tensor& tan_a_im,
    torch::stable::Tensor& tan_out_re,
    torch::stable::Tensor& tan_out_im,
    double n,
    double k0_per_meter,
    int64_t num_bins,
    int64_t n_fft,
    int64_t num_targets,
    int64_t targets_per_chunk,
    double fc,
    double slope,
    double t_start,
    int64_t tau_is_seconds) {
  check_cuda_float(d, "d");
  check_cuda_float(a, "a");
  check_cuda_float(a_im, "a_im");
  check_cuda_float(tan_d, "tan_d");
  check_cuda_float(tan_a, "tan_a");
  check_cuda_float(tan_a_im, "tan_a_im");
  check_cuda_float(tan_out_re, "tan_out_re");
  check_cuda_float(tan_out_im, "tan_out_im");
  STD_TORCH_CHECK(
      a.sizes().equals(a_im.sizes()) && d.sizes().equals(a.sizes()),
      "d, a, and a_im must have the same shape.");
  STD_TORCH_CHECK(
      tan_d.sizes().equals(d.sizes()) && tan_a.sizes().equals(d.sizes()) &&
          tan_a_im.sizes().equals(d.sizes()),
      "tangent inputs must match the primal shape.");
  STD_TORCH_CHECK(
      tan_out_re.sizes().equals(tan_out_im.sizes()),
      "tan_out_re and tan_out_im must have the same shape.");
  STD_TORCH_CHECK(tan_out_re.dim() == 2, "tangent outputs must have shape (chunks, bins).");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int targets = checked_int(num_targets, "num_targets");
  const int chunk_size = checked_int(targets_per_chunk, "targets_per_chunk");
  const int seconds = checked_int(tau_is_seconds, "tau_is_seconds");
  STD_TORCH_CHECK(chunk_size > 0, "targets_per_chunk must be positive.");

  const torch::stable::accelerator::DeviceGuard device_guard(d.get_device_index());
  constexpr int block_size = 256;
  const dim3 block(block_size, 1, 1);
  const dim3 grid((bins + block_size - 1) / block_size, tan_out_re.size(0), 1);
  dirichlet_jvp_kernel<<<grid, block, 0, current_cuda_stream(d)>>>(
      d.const_data_ptr<float>(),
      a.const_data_ptr<float>(),
      a_im.const_data_ptr<float>(),
      tan_d.const_data_ptr<float>(),
      tan_a.const_data_ptr<float>(),
      tan_a_im.const_data_ptr<float>(),
      tan_out_re.mutable_data_ptr<float>(),
      tan_out_im.mutable_data_ptr<float>(),
      static_cast<float>(n),
      static_cast<float>(k0_per_meter),
      bins,
      fft,
      targets,
      chunk_size,
      static_cast<float>(fc),
      static_cast<float>(slope),
      static_cast<float>(t_start),
      seconds);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void forward_mimo_linear_chunked_cuda(
    const torch::stable::Tensor& d0,
    const torch::stable::Tensor& d_rate,
    const torch::stable::Tensor& a0,
    const torch::stable::Tensor& a0_im,
    torch::stable::Tensor& output_re,
    torch::stable::Tensor& output_im,
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
    double t_start,
    int64_t tau_is_seconds) {
  check_cuda_float(d0, "d0");
  check_cuda_float(d_rate, "d_rate");
  check_cuda_float(a0, "a0");
  check_cuda_float(a0_im, "a0_im");
  check_cuda_float(output_re, "output_re");
  check_cuda_float(output_im, "output_im");
  STD_TORCH_CHECK(
      d0.sizes().equals(d_rate.sizes()) && d0.sizes().equals(a0.sizes()) &&
          d0.sizes().equals(a0_im.sizes()),
      "d0, d_rate, a0, and a0_im must have the same shape.");
  STD_TORCH_CHECK(
      output_re.sizes().equals(output_im.sizes()),
      "output_re and output_im must have the same shape.");
  STD_TORCH_CHECK(output_re.dim() == 3, "output tensors must have shape (chirps, pairs, bins).");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int per_pair = checked_int(targets_per_pair, "targets_per_pair");
  const int chirps = checked_int(chirp_per_frame, "chirp_per_frame");
  const int tx = checked_int(num_tx, "num_tx");
  const int update = checked_int(range_loss_update, "range_loss_update");
  const int pairs = checked_int(output_re.size(1), "num_pairs");
  const int seconds = checked_int(tau_is_seconds, "tau_is_seconds");
  STD_TORCH_CHECK(per_pair > 0, "targets_per_pair must be positive.");
  STD_TORCH_CHECK(tx > 0 && pairs % tx == 0, "num_pairs must be a positive multiple of num_tx.");

  const torch::stable::accelerator::DeviceGuard device_guard(d0.get_device_index());
  constexpr int block_size = 256;
  const dim3 block(block_size, 1, 1);
  const dim3 grid((bins + block_size - 1) / block_size, pairs, chirps);
  forward_mimo_linear_chunked_kernel<<<grid, block, 0, current_cuda_stream(d0)>>>(
      d0.const_data_ptr<float>(),
      d_rate.const_data_ptr<float>(),
      a0.const_data_ptr<float>(),
      a0_im.const_data_ptr<float>(),
      output_re.mutable_data_ptr<float>(),
      output_im.mutable_data_ptr<float>(),
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
      static_cast<float>(t_start),
      seconds);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void mimo_linear_backward_cuda(
    const torch::stable::Tensor& d0,
    const torch::stable::Tensor& d_rate,
    const torch::stable::Tensor& a0,
    const torch::stable::Tensor& a0_im,
    const torch::stable::Tensor& grad_output_re,
    const torch::stable::Tensor& grad_output_im,
    torch::stable::Tensor& grad_d0,
    torch::stable::Tensor& grad_d_rate,
    torch::stable::Tensor& grad_a0,
    torch::stable::Tensor& grad_a0_im,
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
    double t_start,
    int64_t tau_is_seconds) {
  check_cuda_float(d0, "d0");
  check_cuda_float(d_rate, "d_rate");
  check_cuda_float(a0, "a0");
  check_cuda_float(a0_im, "a0_im");
  check_cuda_float(grad_output_re, "grad_output_re");
  check_cuda_float(grad_output_im, "grad_output_im");
  check_cuda_float(grad_d0, "grad_d0");
  check_cuda_float(grad_d_rate, "grad_d_rate");
  check_cuda_float(grad_a0, "grad_a0");
  check_cuda_float(grad_a0_im, "grad_a0_im");
  STD_TORCH_CHECK(
      d0.sizes().equals(d_rate.sizes()) && d0.sizes().equals(a0.sizes()) &&
          d0.sizes().equals(a0_im.sizes()),
      "d0, d_rate, a0, and a0_im must have the same shape.");
  STD_TORCH_CHECK(
      grad_d0.sizes().equals(d0.sizes()) && grad_d_rate.sizes().equals(d0.sizes()) &&
          grad_a0.sizes().equals(d0.sizes()) && grad_a0_im.sizes().equals(d0.sizes()),
      "gradient outputs must match d0.");
  STD_TORCH_CHECK(
      grad_output_re.sizes().equals(grad_output_im.sizes()),
      "complex gradient components must have the same shape.");
  STD_TORCH_CHECK(
      grad_output_re.dim() == 3, "gradient tensors must have shape (chirps, pairs, bins).");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int per_pair = checked_int(targets_per_pair, "targets_per_pair");
  const int chirps = checked_int(chirp_per_frame, "chirp_per_frame");
  const int tx = checked_int(num_tx, "num_tx");
  const int update = checked_int(range_loss_update, "range_loss_update");
  const int pairs = checked_int(grad_output_re.size(1), "num_pairs");
  const int seconds = checked_int(tau_is_seconds, "tau_is_seconds");
  STD_TORCH_CHECK(per_pair > 0, "targets_per_pair must be positive.");
  STD_TORCH_CHECK(tx > 0 && pairs % tx == 0, "num_pairs must be a positive multiple of num_tx.");
  STD_TORCH_CHECK(
      grad_output_re.size(0) == chirps && grad_output_re.size(2) == bins,
      "gradient tensor shape does not match chirp_per_frame and num_bins.");
  STD_TORCH_CHECK(
      d0.numel() == static_cast<int64_t>(pairs) * static_cast<int64_t>(per_pair),
      "d0 must hold num_pairs * targets_per_pair rows.");

  const torch::stable::accelerator::DeviceGuard device_guard(d0.get_device_index());
  constexpr int block_size = 256;
  mimo_linear_backward_kernel<<<pairs * per_pair, block_size, 0, current_cuda_stream(d0)>>>(
      d0.const_data_ptr<float>(),
      d_rate.const_data_ptr<float>(),
      a0.const_data_ptr<float>(),
      a0_im.const_data_ptr<float>(),
      grad_output_re.const_data_ptr<float>(),
      grad_output_im.const_data_ptr<float>(),
      grad_d0.mutable_data_ptr<float>(),
      grad_d_rate.mutable_data_ptr<float>(),
      grad_a0.mutable_data_ptr<float>(),
      grad_a0_im.mutable_data_ptr<float>(),
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
      static_cast<float>(t_start),
      seconds);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void mimo_linear_jvp_cuda(
    const torch::stable::Tensor& d0,
    const torch::stable::Tensor& d_rate,
    const torch::stable::Tensor& a0,
    const torch::stable::Tensor& a0_im,
    const torch::stable::Tensor& tan_d0,
    const torch::stable::Tensor& tan_d_rate,
    const torch::stable::Tensor& tan_a0,
    const torch::stable::Tensor& tan_a0_im,
    torch::stable::Tensor& tan_out_re,
    torch::stable::Tensor& tan_out_im,
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
    double t_start,
    int64_t tau_is_seconds) {
  check_cuda_float(d0, "d0");
  check_cuda_float(d_rate, "d_rate");
  check_cuda_float(a0, "a0");
  check_cuda_float(a0_im, "a0_im");
  check_cuda_float(tan_d0, "tan_d0");
  check_cuda_float(tan_d_rate, "tan_d_rate");
  check_cuda_float(tan_a0, "tan_a0");
  check_cuda_float(tan_a0_im, "tan_a0_im");
  check_cuda_float(tan_out_re, "tan_out_re");
  check_cuda_float(tan_out_im, "tan_out_im");
  STD_TORCH_CHECK(
      d0.sizes().equals(d_rate.sizes()) && d0.sizes().equals(a0.sizes()) &&
          d0.sizes().equals(a0_im.sizes()),
      "d0, d_rate, a0, and a0_im must have the same shape.");
  STD_TORCH_CHECK(
      tan_d0.sizes().equals(d0.sizes()) && tan_d_rate.sizes().equals(d0.sizes()) &&
          tan_a0.sizes().equals(d0.sizes()) && tan_a0_im.sizes().equals(d0.sizes()),
      "tangent inputs must match the primal shape.");
  STD_TORCH_CHECK(
      tan_out_re.sizes().equals(tan_out_im.sizes()),
      "tan_out_re and tan_out_im must have the same shape.");
  STD_TORCH_CHECK(
      tan_out_re.dim() == 3, "tangent outputs must have shape (chirps, pairs, bins).");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int per_pair = checked_int(targets_per_pair, "targets_per_pair");
  const int chirps = checked_int(chirp_per_frame, "chirp_per_frame");
  const int tx = checked_int(num_tx, "num_tx");
  const int update = checked_int(range_loss_update, "range_loss_update");
  const int pairs = checked_int(tan_out_re.size(1), "num_pairs");
  const int seconds = checked_int(tau_is_seconds, "tau_is_seconds");
  STD_TORCH_CHECK(per_pair > 0, "targets_per_pair must be positive.");
  STD_TORCH_CHECK(tx > 0 && pairs % tx == 0, "num_pairs must be a positive multiple of num_tx.");

  const torch::stable::accelerator::DeviceGuard device_guard(d0.get_device_index());
  constexpr int block_size = 256;
  const dim3 block(block_size, 1, 1);
  const dim3 grid((bins + block_size - 1) / block_size, pairs, chirps);
  mimo_linear_jvp_kernel<<<grid, block, 0, current_cuda_stream(d0)>>>(
      d0.const_data_ptr<float>(),
      d_rate.const_data_ptr<float>(),
      a0.const_data_ptr<float>(),
      a0_im.const_data_ptr<float>(),
      tan_d0.const_data_ptr<float>(),
      tan_d_rate.const_data_ptr<float>(),
      tan_a0.const_data_ptr<float>(),
      tan_a0_im.const_data_ptr<float>(),
      tan_out_re.mutable_data_ptr<float>(),
      tan_out_im.mutable_data_ptr<float>(),
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
      static_cast<float>(t_start),
      seconds);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void backward_cuda(
    const torch::stable::Tensor& d,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& a_im,
    const torch::stable::Tensor& grad_output_re,
    const torch::stable::Tensor& grad_output_im,
    torch::stable::Tensor& grad_d,
    torch::stable::Tensor& grad_a,
    torch::stable::Tensor& grad_a_im,
    double n,
    double k0_per_meter,
    int64_t num_bins,
    int64_t n_fft,
    int64_t num_targets,
    double fc,
    double slope,
    double t_start,
    int64_t tau_is_seconds) {
  check_cuda_float(d, "d");
  check_cuda_float(a, "a");
  check_cuda_float(a_im, "a_im");
  check_cuda_float(grad_output_re, "grad_output_re");
  check_cuda_float(grad_output_im, "grad_output_im");
  check_cuda_float(grad_d, "grad_d");
  check_cuda_float(grad_a, "grad_a");
  check_cuda_float(grad_a_im, "grad_a_im");
  STD_TORCH_CHECK(
      d.sizes().equals(a.sizes()) && d.sizes().equals(a_im.sizes()),
      "d, a, and a_im must have the same shape.");
  STD_TORCH_CHECK(
      grad_d.sizes().equals(d.sizes()) && grad_a.sizes().equals(d.sizes()) &&
          grad_a_im.sizes().equals(d.sizes()),
      "gradient outputs must match d.");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int targets = checked_int(num_targets, "num_targets");
  const int seconds = checked_int(tau_is_seconds, "tau_is_seconds");
  STD_TORCH_CHECK(d.numel() == targets, "num_targets must match d.numel().");

  const torch::stable::accelerator::DeviceGuard device_guard(d.get_device_index());
  constexpr int block_size = 256;
  const dim3 block(block_size, 1, 1);
  const dim3 grid((targets + block_size - 1) / block_size, 1, 1);
  backward_kernel<<<grid, block, 0, current_cuda_stream(d)>>>(
      d.const_data_ptr<float>(),
      a.const_data_ptr<float>(),
      a_im.const_data_ptr<float>(),
      grad_output_re.const_data_ptr<float>(),
      grad_output_im.const_data_ptr<float>(),
      grad_d.mutable_data_ptr<float>(),
      grad_a.mutable_data_ptr<float>(),
      grad_a_im.mutable_data_ptr<float>(),
      static_cast<float>(n),
      static_cast<float>(k0_per_meter),
      bins,
      fft,
      targets,
      static_cast<float>(fc),
      static_cast<float>(slope),
      static_cast<float>(t_start),
      seconds);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void backward_batched_cuda(
    const torch::stable::Tensor& d,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& a_im,
    const torch::stable::Tensor& grad_output_re,
    const torch::stable::Tensor& grad_output_im,
    torch::stable::Tensor& grad_d,
    torch::stable::Tensor& grad_a,
    torch::stable::Tensor& grad_a_im,
    double n,
    double k0_per_meter,
    int64_t num_bins,
    int64_t n_fft,
    int64_t num_targets,
    int64_t targets_per_spectrum,
    double fc,
    double slope,
    double t_start,
    int64_t tau_is_seconds) {
  check_cuda_float(d, "d");
  check_cuda_float(a, "a");
  check_cuda_float(a_im, "a_im");
  check_cuda_float(grad_output_re, "grad_output_re");
  check_cuda_float(grad_output_im, "grad_output_im");
  check_cuda_float(grad_d, "grad_d");
  check_cuda_float(grad_a, "grad_a");
  check_cuda_float(grad_a_im, "grad_a_im");
  STD_TORCH_CHECK(
      d.sizes().equals(a.sizes()) && d.sizes().equals(a_im.sizes()),
      "d, a, and a_im must have the same shape.");
  STD_TORCH_CHECK(
      grad_d.sizes().equals(d.sizes()) && grad_a.sizes().equals(d.sizes()) &&
          grad_a_im.sizes().equals(d.sizes()),
      "gradient outputs must match d.");
  STD_TORCH_CHECK(
      grad_output_re.sizes().equals(grad_output_im.sizes()),
      "complex gradient components must have the same shape.");
  STD_TORCH_CHECK(grad_output_re.dim() == 2, "gradient tensors must have shape (spectra, bins).");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int targets = checked_int(num_targets, "num_targets");
  const int per_spectrum = checked_int(targets_per_spectrum, "targets_per_spectrum");
  const int seconds = checked_int(tau_is_seconds, "tau_is_seconds");
  STD_TORCH_CHECK(per_spectrum > 0, "targets_per_spectrum must be positive.");
  STD_TORCH_CHECK(
      grad_output_re.size(0) == (targets + per_spectrum - 1) / per_spectrum,
      "gradient spectrum count does not match num_targets and targets_per_spectrum.");
  STD_TORCH_CHECK(grad_output_re.size(1) == bins, "gradient bin count does not match num_bins.");

  const torch::stable::accelerator::DeviceGuard device_guard(d.get_device_index());
  constexpr int block_size = 256;
  const dim3 block(block_size, 1, 1);
  const dim3 grid((targets + block_size - 1) / block_size, 1, 1);
  backward_batched_kernel<<<grid, block, 0, current_cuda_stream(d)>>>(
      d.const_data_ptr<float>(),
      a.const_data_ptr<float>(),
      a_im.const_data_ptr<float>(),
      grad_output_re.const_data_ptr<float>(),
      grad_output_im.const_data_ptr<float>(),
      grad_d.mutable_data_ptr<float>(),
      grad_a.mutable_data_ptr<float>(),
      grad_a_im.mutable_data_ptr<float>(),
      static_cast<float>(n),
      static_cast<float>(k0_per_meter),
      bins,
      fft,
      targets,
      per_spectrum,
      static_cast<float>(fc),
      static_cast<float>(slope),
      static_cast<float>(t_start),
      seconds);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void backward_parallel_bins_cuda(
    const torch::stable::Tensor& d,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& a_im,
    const torch::stable::Tensor& grad_output_re,
    const torch::stable::Tensor& grad_output_im,
    torch::stable::Tensor& grad_d,
    torch::stable::Tensor& grad_a,
    torch::stable::Tensor& grad_a_im,
    double n,
    double k0_per_meter,
    int64_t num_bins,
    int64_t n_fft,
    int64_t num_targets,
    double fc,
    double slope,
    double t_start,
    int64_t tau_is_seconds) {
  check_cuda_float(d, "d");
  check_cuda_float(a, "a");
  check_cuda_float(a_im, "a_im");
  check_cuda_float(grad_output_re, "grad_output_re");
  check_cuda_float(grad_output_im, "grad_output_im");
  check_cuda_float(grad_d, "grad_d");
  check_cuda_float(grad_a, "grad_a");
  check_cuda_float(grad_a_im, "grad_a_im");
  STD_TORCH_CHECK(
      d.sizes().equals(a.sizes()) && d.sizes().equals(a_im.sizes()),
      "d, a, and a_im must have the same shape.");
  STD_TORCH_CHECK(
      grad_d.sizes().equals(d.sizes()) && grad_a.sizes().equals(d.sizes()) &&
          grad_a_im.sizes().equals(d.sizes()),
      "gradient outputs must match d.");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int targets = checked_int(num_targets, "num_targets");
  const int seconds = checked_int(tau_is_seconds, "tau_is_seconds");
  STD_TORCH_CHECK(d.numel() == targets, "num_targets must match d.numel().");
  STD_TORCH_CHECK(
      grad_output_re.numel() == bins && grad_output_im.numel() == bins,
      "gradient bin count mismatch.");

  const torch::stable::accelerator::DeviceGuard device_guard(d.get_device_index());
  constexpr int block_size = 256;
  backward_parallel_bins_kernel<<<targets, block_size, 0, current_cuda_stream(d)>>>(
      d.const_data_ptr<float>(),
      a.const_data_ptr<float>(),
      a_im.const_data_ptr<float>(),
      grad_output_re.const_data_ptr<float>(),
      grad_output_im.const_data_ptr<float>(),
      grad_d.mutable_data_ptr<float>(),
      grad_a.mutable_data_ptr<float>(),
      grad_a_im.mutable_data_ptr<float>(),
      static_cast<float>(n),
      static_cast<float>(k0_per_meter),
      bins,
      fft,
      targets,
      static_cast<float>(fc),
      static_cast<float>(slope),
      static_cast<float>(t_start),
      seconds);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void backward_per_bin_cuda(
    const torch::stable::Tensor& d,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& a_im,
    const torch::stable::Tensor& grad_output_re,
    const torch::stable::Tensor& grad_output_im,
    torch::stable::Tensor& grad_d,
    torch::stable::Tensor& grad_a,
    torch::stable::Tensor& grad_a_im,
    double n,
    double k0_per_meter,
    int64_t num_bins,
    int64_t n_fft,
    int64_t num_targets,
    int64_t bins_per_chunk,
    double fc,
    double slope,
    double t_start,
    int64_t tau_is_seconds) {
  check_cuda_float(d, "d");
  check_cuda_float(a, "a");
  check_cuda_float(a_im, "a_im");
  check_cuda_float(grad_output_re, "grad_output_re");
  check_cuda_float(grad_output_im, "grad_output_im");
  check_cuda_float(grad_d, "grad_d");
  check_cuda_float(grad_a, "grad_a");
  check_cuda_float(grad_a_im, "grad_a_im");
  STD_TORCH_CHECK(d.sizes().equals(a.sizes()) && d.sizes().equals(a_im.sizes()),
      "d, a, and a_im must have the same shape.");

  const int bins = checked_int(num_bins, "num_bins");
  const int fft = checked_int(n_fft, "n_fft");
  const int targets = checked_int(num_targets, "num_targets");
  const int chunk = checked_int(bins_per_chunk, "bins_per_chunk");
  const int seconds = checked_int(tau_is_seconds, "tau_is_seconds");
  STD_TORCH_CHECK(chunk > 0, "bins_per_chunk must be positive.");

  const torch::stable::accelerator::DeviceGuard device_guard(d.get_device_index());
  const dim3 block(chunk, 1, 1);
  const dim3 grid((bins + chunk - 1) / chunk, 1, 1);
  backward_per_bin_kernel<<<grid, block, 0, current_cuda_stream(d)>>>(
      d.const_data_ptr<float>(),
      a.const_data_ptr<float>(),
      a_im.const_data_ptr<float>(),
      grad_output_re.const_data_ptr<float>(),
      grad_output_im.const_data_ptr<float>(),
      grad_d.mutable_data_ptr<float>(),
      grad_a.mutable_data_ptr<float>(),
      grad_a_im.mutable_data_ptr<float>(),
      static_cast<float>(n),
      static_cast<float>(k0_per_meter),
      bins,
      fft,
      targets,
      chunk,
      static_cast<float>(fc),
      static_cast<float>(slope),
      static_cast<float>(t_start),
      seconds);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(witwin_radar_dirichlet_cuda, CUDA, m) {
  m.impl("forward_chunked", TORCH_BOX(&forward_chunked_cuda));
  m.impl("forward_mimo_linear_chunked", TORCH_BOX(&forward_mimo_linear_chunked_cuda));
  m.impl("mimo_linear_backward", TORCH_BOX(&mimo_linear_backward_cuda));
  m.impl("mimo_linear_jvp", TORCH_BOX(&mimo_linear_jvp_cuda));
  m.impl("dirichlet_jvp", TORCH_BOX(&dirichlet_jvp_cuda));
  m.impl("backward", TORCH_BOX(&backward_cuda));
  m.impl("backward_batched", TORCH_BOX(&backward_batched_cuda));
  m.impl("backward_parallel_bins", TORCH_BOX(&backward_parallel_bins_cuda));
  m.impl("backward_per_bin", TORCH_BOX(&backward_per_bin_cuda));
}
