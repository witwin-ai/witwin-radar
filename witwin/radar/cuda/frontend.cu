// Receiver frontend: oscillator and thermal noise, LNA, compression, baseband
// DC offset and flicker, AGC, and the ADC.
//
// Native owner of the receive chain. Registers in the single `_radar_native`
// library Radar ships (R-ADR-004).
//
// THE ORDER IS FIXED and the Python runtime (frontend.FrontendChain.apply)
// applies these operators in one sequence with no exceptions (R-ADR-031):
//
//   0. port        x <- x * sqrt(R)                sqrt(W) -> volts, exactly ONCE
//   0b. coupling   Y = C_tx^T H C_rx^T             cuBLAS, ahead of any active stage
//   1. phase       x <- x * exp(j theta)           supplied oscillator difference
//   2. thermal     x <- x + n,  n ~ CN(0, 2 s^2)   stage 1, INPUT-REFERRED
//   3. lna         x <- x * g_lna
//   4. compress    x <- x G(|x|^2)                 after the LNA, input-referred IIP3
//   5. iq          alpha x + beta conj(x)          Torch, two complex multiplies
//   6. baseband    x <- x + dc[rx] + flicker(t)    after the mixer
//   7. agc         x <- x * g (frame or per block)
//   8. adc         x <- mid-tread code             ALWAYS last
//
// Stages 1 to 3 are ONE operator (`frontend_noise`) rather than three, and that
// is the whole point. Two independently callable runtimes let the caller decide
// whether thermal noise lands before or after the LNA, and the answer is worth
// a factor of `g_lna^2` in output noise power. Thermal noise physically enters
// at the antenna and LNA input, so it is added BEFORE the gain, and there is no
// argument that reorders it.
//
//   y = ( x * exp(j theta) + n ) * g_lna
//
// COUNTER-BASED RNG, AND WHY IT MAY NOT BE A PER-THREAD STATE. Every draw here
// is Philox-4x32-10 keyed by `seed_base` and countered by
// `(stage_id, linear element index)`. The realisation therefore depends on the
// element's INDEX and on nothing else: not on the block size, not on the grid
// shape, not on how many threads a launch happened to use. A `curand`
// state-per-thread scheme keyed by thread id gives a different realisation the
// moment the launch configuration changes, which turns a scheduling decision
// into a numerical one. That is why this file carries its own Philox rather
// than calling curand_init.
//
// The `stage_id` in the counter is what makes toggling one stage leave every
// other stage BIT-IDENTICAL. A single generator threaded through the chain
// consumes draws as it goes, so enabling phase noise shifts the thermal
// realisation and a differential measurement compares two different noise
// realisations while believing it isolated one stage.
//
// DRAW ORDER IS PART OF THE CONTRACT, because a refactor that fuses two draws
// into one changes every realisation:
//
//   thermal: ONE Philox call per element, in linear index order; words 0 and 1
//            become the Box-Muller pair, real component FIRST then imaginary.
//   phase:   per integer-second oscillator block, keyed by seed XOR a block
//            hash, 41 Philox calls at stage 0: counter 0 draws the block
//            endpoint, then one call per dyadic level (40) at the visited node
//            index. Each call keeps the FIRST Box-Muller component and drops
//            the second, and no draw depends on which queries were made.
//   flicker: the same block and node walk at stage 2 (1/f^3 region).
//   floor:   one Philox call at stage 3 per hold cell floor(t * cell_rate),
//            FIRST component (white phase floor).
//
// THE OSCILLATOR HAS THREE REGIONS, ONE REALISATION OVER ABSOLUTE TIME:
//
//   phi(t) = sqrt(q) W(t) + c F(t) + s Z(floor(t f_cell))
//
//   W  the Brownian bridge below (Levy-Ciesielski: tent amplitude
//      0.5 sqrt(w) at width w), the exact white-FM / 1/f^2 region,
//      L(f) = q / (4 pi^2 f^2), q = 4 pi^2 f2^2 L(f2).
//   F  the same Schauder tents with amplitude w instead of 0.5 sqrt(w): the
//      derivative is a sum of Haar wavelets of equal height in every octave,
//      a Voss-McCartney flicker-FM process whose octave-averaged PSD is
//      L(f) = c^2 / (2 pi^2 ln2 f^3), c^2 = 2 pi^2 ln2 f3^3 L(f3). The octave
//      ripple of the discrete sum is within -0.36/+0.21 dB.
//   Z  i.i.d. unit Gaussian draws held over cells of 1/f_cell: the white floor,
//      L = s^2 / f_cell on the cell grid, s^2 = L_floor f_cell.
//
// All three are keyed by absolute time, so a delayed-transmit query and a
// receive query see the same oscillator and the range correlation
// phi(t) - phi(t - tau) is exact for W, a dyadic-scale model for F, and the
// zero-order-hold triangle 2 s^2 min(tau f_cell, 1) for Z. A zero coefficient
// skips its term, so the Wiener-only realisation is bit-identical to a run
// without the other two regions.
//
// INVALID QUERIES NEVER ASSERT ON THE DEVICE. A non-finite time or delay, or a
// negative delay, writes NaN phase and increments a device counter that the
// Python owner decodes on the host. A device-side assert would poison the CUDA
// context for the whole process; a counter cannot.
//
// Oscillator phase is evaluated on actual timestamps by oscillator_phase_forward.
// Its Brownian bridge is keyed by absolute dyadic time intervals, so the same
// oscillator can be queried at receive and delayed transmit times without
// inventing independent noise per path. The receive operator consumes that
// phase; phase generation and receive rotation have distinct physical owners.
//
// The accumulated phase is PUBLISHED as a device tensor. The backward and jvp
// operators consume it instead of regenerating it, which is what makes the
// derivative exactly consistent with the realisation it was taken at, and it is
// also what lets a test measure the phase-noise power spectrum without a second
// implementation of the generator.
//
// AGC IS DATA-DEPENDENT, AND ITS GAIN NEVER REACHES THE HOST. The gain and the
// measured RMS are published as device tensors for diagnostics; reading either
// to build a Python scalar would be a per-frame device-to-host transfer. The
// data dependence also means the frontend is NOT linear in the signal, so the
// cross-waveform linearity invariant holds only with AGC off - a tested fact
// rather than a footnote.
//
//   r    = sqrt(max(mean(|x|^2), 1e-24))
//   g    = clamp(target / r, g_min, g_max)
//   y    = g x
//   dg/dx_re = -target x_re / (N r^3)   when unclamped, 0 in the clamped region
//
// so the gradient is the rank-one update
//
//   dL/dx_re = g gy_re - (target / (N r^3)) x_re SUM_j (gy_re x_re + gy_im x_im)
//
// which is why the backward and jvp operators both take two passes: one group
// reduction, then one elementwise apply.
//
// THE QUANTIZER HAS NO BACKWARD AND NO JVP, ON PURPOSE. `round` is not
// differentiable, and a straight-through surrogate is a modelling decision
// rather than a detail this file may choose. Its Python owner
// raises on a grad-enabled or forward-dual input instead of silently detaching,
// which is the difference between an unsupported operation and a wrong
// gradient. This is the one deliberate exception to the three-per-family rule.
//
// The grid is MID-TREAD in two's complement: zero is a code, so an idle channel
// reads exactly zero rather than +/- step/2 of DC on each component.
//
//   step = 2 * full_scale / 2^bits
//   k    = clamp(rint(x / step), -2^(bits-1), 2^(bits-1) - 1)
//   y    = k * step                               in [-FS, FS - step]
//
// The code is formed in double, so a 30-bit grid's end codes are exact. A
// component whose unclamped code lies outside the range is CLIPPED; the
// no-overload input range is [-FS - step/2, FS - step/2). The clipped-component
// count is published as a device diagnostic. Suppressing it hides an AGC
// misconfiguration behind a signal that merely looks compressed.
//
// Numerics: the noise and quantiser passes are elementwise single precision,
// which is what the signal is. The two group reductions accumulate in double
// inside each thread and combine in double in shared memory, because a sum of
// millions of squared magnitudes in float32 loses the small terms entirely and
// the AGC gain is a square root of exactly that sum. Fast math stays off.

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

constexpr float kTwoPiF = 6.283185307179586476925286766559f;
constexpr double kRmsFloor = 1e-24;

// Philox-4x32-10. Written out here rather than pulled from curand because the
// contract is that the realisation depends on the element index alone.
constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;

__device__ __forceinline__ uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t* high) {
  const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
  *high = static_cast<uint32_t>(product >> 32);
  return static_cast<uint32_t>(product);
}

__device__ __forceinline__ void philox_round(uint32_t counter[4], const uint32_t key[2]) {
  uint32_t high0 = 0;
  uint32_t high1 = 0;
  const uint32_t low0 = mulhilo32(kPhiloxM0, counter[0], &high0);
  const uint32_t low1 = mulhilo32(kPhiloxM1, counter[2], &high1);
  const uint32_t out0 = high1 ^ counter[1] ^ key[0];
  const uint32_t out1 = low1;
  const uint32_t out2 = high0 ^ counter[3] ^ key[1];
  const uint32_t out3 = low0;
  counter[0] = out0;
  counter[1] = out1;
  counter[2] = out2;
  counter[3] = out3;
}

__device__ __forceinline__ void philox4x32_10(
    uint64_t seed, uint32_t stage_id, uint64_t index, uint32_t out[4]) {
  uint32_t counter[4] = {
      static_cast<uint32_t>(index & 0xFFFFFFFFull),
      static_cast<uint32_t>(index >> 32),
      stage_id,
      0u};
  uint32_t key[2] = {
      static_cast<uint32_t>(seed & 0xFFFFFFFFull),
      static_cast<uint32_t>(seed >> 32)};
  for (int round = 0; round < 10; ++round) {
    if (round > 0) {
      key[0] += kPhiloxW0;
      key[1] += kPhiloxW1;
    }
    philox_round(counter, key);
  }
  out[0] = counter[0];
  out[1] = counter[1];
  out[2] = counter[2];
  out[3] = counter[3];
}

// Open unit interval: the log below rejects an exact zero, and the shift keeps
// the mapping monotone in the raw word.
__device__ __forceinline__ float to_open_unit(uint32_t word) {
  return static_cast<float>(word) * 2.3283064365386963e-10f + 1.1641532182693481e-10f;
}

struct NormalPair {
  float first;
  float second;
};

// Box-Muller, real component FIRST then imaginary. The order is the contract.
__device__ __forceinline__ NormalPair standard_normal_pair(
    uint64_t seed, uint32_t stage_id, uint64_t index) {
  uint32_t words[4];
  philox4x32_10(seed, stage_id, index, words);
  const float u1 = to_open_unit(words[0]);
  const float u2 = to_open_unit(words[1]);
  const float radius = sqrtf(-2.0f * logf(u1));
  float sine = 0.0f;
  float cosine = 0.0f;
  sincosf(kTwoPiF * u2, &sine, &cosine);
  NormalPair pair;
  pair.first = radius * cosine;
  pair.second = radius * sine;
  return pair;
}

constexpr uint32_t kStageWiener = 0u;
constexpr uint32_t kStageFlicker = 2u;
constexpr uint32_t kStageFloor = 3u;

// A common continuous-time oscillator component, evaluated without a sample
// grid. Each integer-second interval has a Philox-keyed endpoint and a dyadic
// Schauder expansion. Forty levels resolve 2^-40 s (~0.91 ps). With
// `flicker == false` this is the Brownian bridge (tent amplitude 0.5 sqrt(w));
// the omitted bridge variance is at most diffusion*2^-42 rad^2. With
// `flicker == true` the tent amplitude is w, the flicker-FM expansion above.
// Refining/reordering queries cannot change the realization. Independent
// oracles: interval-overlap covariance and the Welch PSD tests.
__device__ double oscillator_block(double u, int64_t block, uint64_t seed, bool flicker) {
  const uint64_t key = seed ^ (static_cast<uint64_t>(block) * 0x9E3779B97F4A7C15ull);
  const uint32_t stage = flicker ? kStageFlicker : kStageWiener;
  double value = u * standard_normal_pair(key, stage, 0).first;
  double left = 0.0, width = 1.0;
  uint64_t node = 1;
  for (int level = 0; level < 40; ++level) {
    const double fraction = (u-left)/width;
    const double tent = 2.0 * fmin(fraction, 1.0-fraction);
    const double amplitude = flicker ? width : 0.5*sqrt(width);
    value += amplitude*tent*standard_normal_pair(key, stage, node).first;
    width *= 0.5;
    node *= 2;
    if (fraction >= 0.5) { left += width; ++node; }
  }
  return value;
}

// phi_component(t) - phi_component(s) for one Schauder component, summing the
// whole intervening integer-second blocks exactly as the bridge defines them.
__device__ double oscillator_component_difference(double t, double s, uint64_t seed, bool flicker) {
  const int64_t first=static_cast<int64_t>(floor(s)), last=static_cast<int64_t>(floor(t));
  double difference;
  if (first == last) {
    difference=oscillator_block(t-last,last,seed,flicker)-oscillator_block(s-first,first,seed,flicker);
  } else {
    difference=oscillator_block(1.0,first,seed,flicker)-oscillator_block(s-first,first,seed,flicker);
    for (int64_t block=first+1; block<last; ++block) difference+=oscillator_block(1.0,block,seed,flicker);
    difference+=oscillator_block(t-last,last,seed,flicker);
  }
  return difference;
}

__device__ double oscillator_floor(double t, double cell_rate, uint64_t seed) {
  const int64_t cell = static_cast<int64_t>(floor(t * cell_rate));
  return static_cast<double>(standard_normal_pair(seed, kStageFloor, static_cast<uint64_t>(cell)).first);
}

__global__ void oscillator_phase_kernel(
    const double* times, const double* delays, float* phase, int32_t* invalid, int64_t count,
    double diffusion, double flicker_scale, double floor_sigma, double floor_cell_rate,
    double sign, uint64_t seed) {
  const int64_t i = static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i >= count) return;
  const double t=times[i], delay=delays[i];
  if (!isfinite(t) || !isfinite(delay) || !(delay >= 0.0)) {
    phase[i] = __int_as_float(0x7fc00000);
    atomicAdd(invalid, 1);
    return;
  }
  const double s=t-delay;
  double total = 0.0;
  if (diffusion > 0.0) {
    total = sqrt(diffusion)*oscillator_component_difference(t, s, seed, false);
  }
  if (flicker_scale > 0.0) {
    total += flicker_scale*oscillator_component_difference(t, s, seed, true);
  }
  if (floor_sigma > 0.0) {
    total += floor_sigma*(oscillator_floor(t, floor_cell_rate, seed) - oscillator_floor(s, floor_cell_rate, seed));
  }
  phase[i]=static_cast<float>(sign*total);
}

__global__ void frontend_noise_forward_kernel(
    const float* __restrict__ x_re,
    const float* __restrict__ x_im,
    const float* __restrict__ phase_rad,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    int64_t num_elements,
    int num_phase,
    float thermal_sigma,
    float lna_gain,
    uint64_t seed,
    uint32_t stage_id) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  const int phase_index = static_cast<int>(index % num_phase);
  const float theta = phase_rad[phase_index];
  float sine = 0.0f;
  float cosine = 0.0f;
  sincosf(theta, &sine, &cosine);

  const float re = x_re[index];
  const float im = x_im[index];
  float rotated_re = re * cosine - im * sine;
  float rotated_im = re * sine + im * cosine;

  if (thermal_sigma > 0.0f) {
    const NormalPair pair = standard_normal_pair(
        seed, stage_id, static_cast<uint64_t>(index));
    rotated_re += thermal_sigma * pair.first;
    rotated_im += thermal_sigma * pair.second;
  }

  out_re[index] = rotated_re * lna_gain;
  out_im[index] = rotated_im * lna_gain;
}

// The noise draws are constants with respect to AD, so both derivative kernels
// are the same rotation and gain the primal applied - taken at the SAVED phase,
// never at a regenerated one.
__global__ void frontend_noise_linear_kernel(
    const float* __restrict__ phase_rad,
    const float* __restrict__ in_re,
    const float* __restrict__ in_im,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    int64_t num_elements,
    int num_phase,
    float lna_gain,
    int transpose) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  const int phase_index = static_cast<int>(index % num_phase);
  float sine = 0.0f;
  float cosine = 0.0f;
  sincosf(phase_rad[phase_index], &sine, &cosine);
  if (transpose != 0) {
    sine = -sine;
  }
  const float re = in_re[index];
  const float im = in_im[index];
  out_re[index] = (re * cosine - im * sine) * lna_gain;
  out_im[index] = (re * sine + im * cosine) * lna_gain;
}

// One block per group, accumulating in double. `num_groups` is 1 for a global
// AGC and the receiver count for a per-receiver one, so the block count is tiny
// and the loop inside each block is the long axis.
__global__ void frontend_agc_measure_kernel(
    const float* __restrict__ x_re,
    const float* __restrict__ x_im,
    float* __restrict__ gain,
    float* __restrict__ rms,
    int dim0,
    int num_groups,
    int dim2,
    float target_rms,
    float min_gain,
    float max_gain) {
  extern __shared__ double shared[];
  const int group = blockIdx.x;
  if (group >= num_groups) {
    return;
  }
  const int64_t span = static_cast<int64_t>(dim0) * dim2;
  double partial = 0.0;
  for (int64_t flat = threadIdx.x; flat < span; flat += blockDim.x) {
    const int64_t outer = flat / dim2;
    const int64_t inner = flat - outer * dim2;
    const int64_t index =
        (outer * num_groups + group) * static_cast<int64_t>(dim2) + inner;
    const double re = static_cast<double>(x_re[index]);
    const double im = static_cast<double>(x_im[index]);
    partial += re * re + im * im;
  }
  shared[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x != 0) {
    return;
  }
  const double mean_square = shared[0] / static_cast<double>(span);
  const double clamped = mean_square < kRmsFloor ? kRmsFloor : mean_square;
  const double measured = sqrt(clamped);
  double value = static_cast<double>(target_rms) / measured;
  value = value < min_gain ? min_gain : (value > max_gain ? max_gain : value);
  gain[group] = static_cast<float>(value);
  rms[group] = static_cast<float>(measured);
}

__global__ void frontend_agc_apply_kernel(
    const float* __restrict__ x_re,
    const float* __restrict__ x_im,
    const float* __restrict__ gain,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    int64_t num_elements,
    int num_groups,
    int dim2) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  const int group = static_cast<int>((index / dim2) % num_groups);
  const float g = gain[group];
  out_re[index] = x_re[index] * g;
  out_im[index] = x_im[index] * g;
}

// The reduction the rank-one term needs: SUM_j (a_re x_re + a_im x_im) over the
// group, where `a` is the output cotangent for backward and the input tangent
// for jvp. One expression, two callers.
__global__ void frontend_agc_inner_kernel(
    const float* __restrict__ x_re,
    const float* __restrict__ x_im,
    const float* __restrict__ a_re,
    const float* __restrict__ a_im,
    float* __restrict__ inner,
    int dim0,
    int num_groups,
    int dim2) {
  extern __shared__ double shared[];
  const int group = blockIdx.x;
  if (group >= num_groups) {
    return;
  }
  const int64_t span = static_cast<int64_t>(dim0) * dim2;
  double partial = 0.0;
  for (int64_t flat = threadIdx.x; flat < span; flat += blockDim.x) {
    const int64_t outer = flat / dim2;
    const int64_t inner_index = flat - outer * dim2;
    const int64_t index =
        (outer * num_groups + group) * static_cast<int64_t>(dim2) + inner_index;
    partial += static_cast<double>(a_re[index]) * static_cast<double>(x_re[index]) +
        static_cast<double>(a_im[index]) * static_cast<double>(x_im[index]);
  }
  shared[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    inner[group] = static_cast<float>(shared[0]);
  }
}

__global__ void frontend_agc_linear_kernel(
    const float* __restrict__ x_re,
    const float* __restrict__ x_im,
    const float* __restrict__ gain,
    const float* __restrict__ rms,
    const float* __restrict__ inner,
    const float* __restrict__ a_re,
    const float* __restrict__ a_im,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    int64_t num_elements,
    int dim0,
    int num_groups,
    int dim2,
    float target_rms,
    float min_gain,
    float max_gain) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  const int group = static_cast<int>((index / dim2) % num_groups);
  const float g = gain[group];
  const double measured = static_cast<double>(rms[group]);
  const double unclamped = static_cast<double>(target_rms) / measured;
  double coefficient = 0.0;
  if (unclamped > min_gain && unclamped < max_gain) {
    const double count = static_cast<double>(dim0) * static_cast<double>(dim2);
    coefficient = -static_cast<double>(target_rms) *
        static_cast<double>(inner[group]) /
        (count * measured * measured * measured);
  }
  out_re[index] = a_re[index] * g +
      static_cast<float>(coefficient * static_cast<double>(x_re[index]));
  out_im[index] = a_im[index] * g +
      static_cast<float>(coefficient * static_cast<double>(x_im[index]));
}

__global__ void frontend_quantize_kernel(
    const float* __restrict__ x_re,
    const float* __restrict__ x_im,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    int32_t* __restrict__ clipped_count,
    int64_t num_elements,
    double step,
    double lowest_code,
    double highest_code) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  int clipped = 0;
  double code_re = rint(static_cast<double>(x_re[index]) / step);
  double code_im = rint(static_cast<double>(x_im[index]) / step);
  if (code_re < lowest_code || code_re > highest_code) {
    ++clipped;
    code_re = code_re < lowest_code ? lowest_code : highest_code;
  }
  if (code_im < lowest_code || code_im > highest_code) {
    ++clipped;
    code_im = code_im < lowest_code ? lowest_code : highest_code;
  }
  out_re[index] = static_cast<float>(code_re * step);
  out_im[index] = static_cast<float>(code_im * step);
  if (clipped != 0) {
    atomicAdd(clipped_count, clipped);
  }
}

int checked_block(int64_t value) {
  const int block = checked_int(value, "block_size");
  STD_TORCH_CHECK(
      block >= 32 && block <= 1024 && (block & (block - 1)) == 0,
      "block_size must be a power of two between 32 and 1024.");
  return block;
}

void check_pair(
    const torch::stable::Tensor& re,
    const torch::stable::Tensor& im,
    int64_t num_elements,
    const char* name_re,
    const char* name_im) {
  check_cuda_float(re, name_re);
  check_cuda_float(im, name_im);
  STD_TORCH_CHECK(
      re.numel() == num_elements && im.numel() == num_elements,
      name_re,
      " and its imaginary partner must hold one value per signal element.");
}

void launch_agc_linear(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    const torch::stable::Tensor& gain,
    const torch::stable::Tensor& rms,
    const torch::stable::Tensor& a_re,
    const torch::stable::Tensor& a_im,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    torch::stable::Tensor& inner_buffer,
    int outer,
    int groups,
    int inner,
    double target_rms,
    double min_gain,
    double max_gain,
    int block) {
  const int64_t elements =
      static_cast<int64_t>(outer) * groups * static_cast<int64_t>(inner);
  check_pair(x_re, x_im, elements, "x_re", "x_im");
  check_pair(a_re, a_im, elements, "a_re", "a_im");
  check_pair(out_re, out_im, elements, "out_re", "out_im");
  check_cuda_float(gain, "gain");
  check_cuda_float(rms, "rms");
  check_cuda_float(inner_buffer, "inner");
  STD_TORCH_CHECK(
      gain.numel() == static_cast<int64_t>(groups) &&
          rms.numel() == static_cast<int64_t>(groups) &&
          inner_buffer.numel() == static_cast<int64_t>(groups),
      "gain, rms, and inner must hold one value per AGC group.");

  const torch::stable::accelerator::DeviceGuard device_guard(
      out_re.get_device_index());
  const cudaStream_t stream = current_cuda_stream(out_re);
  constexpr int reduce_block = 256;
  frontend_agc_inner_kernel<<<
      dim3(groups, 1, 1),
      dim3(reduce_block, 1, 1),
      reduce_block * sizeof(double),
      stream>>>(
      x_re.const_data_ptr<float>(),
      x_im.const_data_ptr<float>(),
      a_re.const_data_ptr<float>(),
      a_im.const_data_ptr<float>(),
      inner_buffer.mutable_data_ptr<float>(),
      outer,
      groups,
      inner);
  STD_CUDA_KERNEL_LAUNCH_CHECK();

  frontend_agc_linear_kernel<<<
      dim3(static_cast<unsigned int>((elements + block - 1) / block), 1, 1),
      dim3(block, 1, 1),
      0,
      stream>>>(
      x_re.const_data_ptr<float>(),
      x_im.const_data_ptr<float>(),
      gain.const_data_ptr<float>(),
      rms.const_data_ptr<float>(),
      inner_buffer.const_data_ptr<float>(),
      a_re.const_data_ptr<float>(),
      a_im.const_data_ptr<float>(),
      out_re.mutable_data_ptr<float>(),
      out_im.mutable_data_ptr<float>(),
      elements,
      outer,
      groups,
      inner,
      static_cast<float>(target_rms),
      static_cast<float>(min_gain),
      static_cast<float>(max_gain));
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}


// ---------------------------------------------------------------------------
// Receiver compression: a memoryless polynomial with AM/PM, saturating.
//
// u = scale |x|^2 is the stage input power over the input-referred IIP3
// (scale = 1 / (g_lna^2 P_IIP3 R), so the LNA gain that precedes the stage is
// removed and the intercept is the receiver's INPUT one). Per complex sample
//
//   G(u) = (1 - u + c5 u^2) exp(j k u)                    u <= u_sat
//   G(u) = G(u_sat) sqrt(u_sat / u)                       u >  u_sat
//   y    = x G(u)
//
// The cubic term alone puts the 1-dB point 9.64 dB below IIP3; c5 moves it to
// a declared P1dB. u_sat is the first stationary point of |y| = sqrt(u)|M(u)|,
// after which the output amplitude is held (a soft limiter), so the stage is
// monotone and its derivative continuous. The analytic Jacobian, with
// G'(u) = dG/du and w = x G'(u) scale:
//
//   dy = G dx + w * 2 (x_re dx_re + x_im dx_im)
// ---------------------------------------------------------------------------

struct CompressionGain {
  float g_re;
  float g_im;
  float w_re;  // x * dG/d|x|^2, real part
  float w_im;
};

__device__ __forceinline__ CompressionGain compression_gain(
    float re, float im, float scale, float c5, float k_pm, float u_sat) {
  const float magnitude_sq = re * re + im * im;
  const float u = scale * magnitude_sq;
  float g_re, g_im, gp_re, gp_im;
  if (u <= u_sat) {
    const float m = 1.0f - u + c5 * u * u;
    const float mp = -1.0f + 2.0f * c5 * u;
    float sine = 0.0f, cosine = 1.0f;
    if (k_pm != 0.0f) {
      sincosf(k_pm * u, &sine, &cosine);
    }
    g_re = m * cosine;
    g_im = m * sine;
    // (M' + j k M) exp(j k u)
    gp_re = mp * cosine - k_pm * m * sine;
    gp_im = mp * sine + k_pm * m * cosine;
  } else {
    const float m = 1.0f - u_sat + c5 * u_sat * u_sat;
    float sine = 0.0f, cosine = 1.0f;
    if (k_pm != 0.0f) {
      sincosf(k_pm * u_sat, &sine, &cosine);
    }
    const float hold = sqrtf(u_sat / u);
    g_re = m * cosine * hold;
    g_im = m * sine * hold;
    gp_re = -0.5f * g_re / u;
    gp_im = -0.5f * g_im / u;
  }
  CompressionGain out;
  out.g_re = g_re;
  out.g_im = g_im;
  out.w_re = (re * gp_re - im * gp_im) * scale;
  out.w_im = (re * gp_im + im * gp_re) * scale;
  return out;
}

// mode 0: forward  out = x G
// mode 1: jvp      out = G a + w 2 (x . a)
// mode 2: vjp      out = J^T a
__global__ void frontend_compress_kernel(
    const float* __restrict__ x_re,
    const float* __restrict__ x_im,
    const float* __restrict__ a_re,
    const float* __restrict__ a_im,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    int64_t num_elements,
    float scale,
    float c5,
    float k_pm,
    float u_sat,
    int mode) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  const float re = x_re[index];
  const float im = x_im[index];
  const CompressionGain g = compression_gain(re, im, scale, c5, k_pm, u_sat);
  if (mode == 0) {
    out_re[index] = re * g.g_re - im * g.g_im;
    out_im[index] = re * g.g_im + im * g.g_re;
    return;
  }
  const float ar = a_re[index];
  const float ai = a_im[index];
  if (mode == 1) {
    const float dot = 2.0f * (re * ar + im * ai);
    out_re[index] = g.g_re * ar - g.g_im * ai + g.w_re * dot;
    out_im[index] = g.g_re * ai + g.g_im * ar + g.w_im * dot;
    return;
  }
  // Columns of the real 2x2 Jacobian:
  //   d y / d x_re = (g_re + 2 w_re x_re, g_im + 2 w_im x_re)
  //   d y / d x_im = (-g_im + 2 w_re x_im, g_re + 2 w_im x_im)
  out_re[index] = ar * (g.g_re + 2.0f * g.w_re * re) + ai * (g.g_im + 2.0f * g.w_im * re);
  out_im[index] = ar * (-g.g_im + 2.0f * g.w_re * im) + ai * (g.g_re + 2.0f * g.w_im * im);
}

// ---------------------------------------------------------------------------
// Baseband additive impairments: a per-receiver DC offset and 1/f flicker.
//
// The signal is [outer, run] with outer = tx * num_rx + rx, so the receiver of
// an element is outer % num_rx: both impairments belong to the receive channel
// and are shared by every transmitter's slot of it. The flicker process is a
// Voss-McCartney sum of Haar wavelets of height h in every octave from 1 s down
// to the coarsest level not finer than two samples, evaluated at the element's
// absolute time t:
//
//   n(t) = h SUM_l s_l(t) xi_l(floor(t / w_l)),   w_l = 2^-l,
//   s_l  = +1 on the first half of the cell, -1 on the second,
//
// whose per-component two-sided density is h^2 / (2 ln2 f) to within the
// octave ripple. Each level draws one Philox call at stage 4, keyed by the
// receiver and the level, countered by the cell: words 0/1 are I/Q. The kernel
// writes the ADDITIVE TERM only; the Python owner adds it to the signal, so the
// stage's derivative is the identity carried by that addition and no
// companion kernel exists.
// ---------------------------------------------------------------------------

constexpr uint32_t kStageBasebandFlicker = 4u;

__global__ void frontend_baseband_kernel(
    const double* __restrict__ times,
    const float* __restrict__ dc_re,
    const float* __restrict__ dc_im,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    int64_t num_elements,
    int num_run,
    int num_rx,
    float flicker_level,
    int levels,
    uint64_t seed) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  const int64_t outer = index / num_run;
  const int run = static_cast<int>(index - outer * num_run);
  const int rx = static_cast<int>(outer % num_rx);
  float add_re = dc_re[rx];
  float add_im = dc_im[rx];
  if (flicker_level > 0.0f) {
    const double t = times[run];
    const uint64_t receiver_key = seed ^ (static_cast<uint64_t>(rx + 1) * 0xD1B54A32D192ED03ull);
    double sum_re = 0.0, sum_im = 0.0;
    double width = 1.0;
    for (int level = 0; level < levels; ++level) {
      const double scaled = t / width;
      const double cell = floor(scaled);
      const double sign = (scaled - cell) < 0.5 ? 1.0 : -1.0;
      const uint64_t key = receiver_key ^ (static_cast<uint64_t>(level + 1) * 0x9E3779B97F4A7C15ull);
      const NormalPair pair = standard_normal_pair(
          key, kStageBasebandFlicker, static_cast<uint64_t>(static_cast<int64_t>(cell)));
      sum_re += sign * pair.first;
      sum_im += sign * pair.second;
      width *= 0.5;
    }
    add_re += flicker_level * static_cast<float>(sum_re);
    add_im += flicker_level * static_cast<float>(sum_im);
  }
  out_re[index] = add_re;
  out_im[index] = add_im;
}

// ---------------------------------------------------------------------------
// Time-dynamic AGC: one gain per (group, slow-time block), a first-order loop.
//
// The signal is [outer, group, block, inner]. With r_k the RMS of block k
// (floored at sqrt(1e-24)) and z_k = ln(target / r_k) the gain that would
// settle block k exactly,
//
//   l~_0 = z_0                                   acquired on the first block
//   l~_k = (1 - beta) l_{k-1} + beta z_{k-1}     causal: block k-1 sets block k
//   l_k  = clamp(l~_k, ln g_min, ln g_max),  g_k = exp(l_k),  y = g_k x
//
// beta = 1 - exp(-T_block / tau) for a loop time constant tau. free_k is 1
// where the clamp is inactive. Derivatives, with a_k the cotangent of l_k:
//
//   reverse:  lam_k = a_k + (1 - beta) lamt_{k+1},  lamt_k = free_k lam_k,
//             dL/dz_k = beta lamt_{k+1} [+ lamt_0 for k = 0]
//   forward:  dl_0 = free_0 dz_0,  dl_k = free_k ((1-beta) dl_{k-1} + beta dz_{k-1})
//   dz_k / dx = -x / (N r_k^2)   (zero below the floor)
//
// a_k = g_k SUM_block (gy . x). Each pass is a reduction (the static AGC's
// group kernels over G*K groups), a scan with one thread per group, and one
// elementwise combine out = g a + coef x.
// ---------------------------------------------------------------------------

__global__ void frontend_agc_dynamic_scan_kernel(
    const float* __restrict__ rms,
    float* __restrict__ gain,
    float* __restrict__ free_mask,
    int num_groups,
    int num_blocks,
    double target_rms,
    double log_min,
    double log_max,
    double beta) {
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= num_groups) {
    return;
  }
  double previous_level = 0.0;
  double previous_ideal = 0.0;
  for (int k = 0; k < num_blocks; ++k) {
    const int slot = group * num_blocks + k;
    const double ideal = log(target_rms / static_cast<double>(rms[slot]));
    const double raw = k == 0 ? ideal : (1.0 - beta) * previous_level + beta * previous_ideal;
    const double level = raw < log_min ? log_min : (raw > log_max ? log_max : raw);
    gain[slot] = static_cast<float>(exp(level));
    free_mask[slot] = (raw > log_min && raw < log_max) ? 1.0f : 0.0f;
    previous_level = level;
    previous_ideal = ideal;
  }
}

// `inner` holds SUM_block (a . x) on entry and the combine coefficient on exit.
// reverse == 1: backward (a = output cotangent); reverse == 0: jvp (a = tangent).
__global__ void frontend_agc_dynamic_linear_scan_kernel(
    const float* __restrict__ gain,
    const float* __restrict__ rms,
    const float* __restrict__ free_mask,
    float* __restrict__ inner,
    int num_groups,
    int num_blocks,
    double count,
    double beta,
    int reverse) {
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= num_groups) {
    return;
  }
  const int base = group * num_blocks;
  if (reverse != 0) {
    double carried = 0.0;  // lamt_{k+1}
    for (int k = num_blocks - 1; k >= 0; --k) {
      const int slot = base + k;
      const double a = static_cast<double>(gain[slot]) * static_cast<double>(inner[slot]);
      const double lam = a + (1.0 - beta) * carried;
      const double lamt = static_cast<double>(free_mask[slot]) * lam;
      double dz = beta * carried;
      if (k == 0) {
        dz += lamt;
      }
      const double r = static_cast<double>(rms[slot]);
      const double r2 = r * r;
      inner[slot] = r2 > kRmsFloor ? static_cast<float>(-dz / (count * r2)) : 0.0f;
      carried = lamt;
    }
    return;
  }
  double previous_level = 0.0;
  double previous_ideal = 0.0;
  for (int k = 0; k < num_blocks; ++k) {
    const int slot = base + k;
    const double r = static_cast<double>(rms[slot]);
    const double r2 = r * r;
    const double dz = r2 > kRmsFloor ? -static_cast<double>(inner[slot]) / (count * r2) : 0.0;
    const double raw = k == 0 ? dz : (1.0 - beta) * previous_level + beta * previous_ideal;
    const double level = static_cast<double>(free_mask[slot]) * raw;
    inner[slot] = static_cast<float>(static_cast<double>(gain[slot]) * level);
    previous_level = level;
    previous_ideal = dz;
  }
}

// out = gain[g, k] a + coef[g, k] x over [outer, group * block, inner].
__global__ void frontend_agc_dynamic_combine_kernel(
    const float* __restrict__ x_re,
    const float* __restrict__ x_im,
    const float* __restrict__ a_re,
    const float* __restrict__ a_im,
    const float* __restrict__ gain,
    const float* __restrict__ coef,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    int64_t num_elements,
    int num_slots,
    int dim2) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  const int slot = static_cast<int>((index / dim2) % num_slots);
  const float g = gain[slot];
  const float c = coef == nullptr ? 0.0f : coef[slot];
  out_re[index] = a_re[index] * g + c * x_re[index];
  out_im[index] = a_im[index] * g + c * x_im[index];
}

}  // namespace

void oscillator_phase_forward_cuda(
    const torch::stable::Tensor& times, const torch::stable::Tensor& delays,
    torch::stable::Tensor& phase, torch::stable::Tensor& invalid, double diffusion,
    double flicker_scale, double floor_sigma, double floor_cell_rate, double sign,
    int64_t seed_base, int64_t block_size) {
  STD_TORCH_CHECK(times.is_cuda() && delays.is_cuda(), "oscillator times/delays must be CUDA");
  STD_TORCH_CHECK(times.scalar_type()==torch::headeronly::ScalarType::Double &&
                  delays.scalar_type()==torch::headeronly::ScalarType::Double,
                  "oscillator times/delays must be float64 seconds");
  STD_TORCH_CHECK(times.is_contiguous() && delays.is_contiguous(), "oscillator grids must be contiguous");
  check_cuda_float(phase,"phase");
  STD_TORCH_CHECK(times.numel()==delays.numel() && times.numel()==phase.numel(), "oscillator grid shape mismatch");
  check_cuda_int(invalid, "invalid");
  STD_TORCH_CHECK(invalid.numel()==1, "invalid must hold exactly one counter");
  STD_TORCH_CHECK(diffusion>=0 && flicker_scale>=0 && floor_sigma>=0 && seed_base>=0 && (sign==1.0 || sign==-1.0),
                  "invalid oscillator parameters");
  STD_TORCH_CHECK(floor_sigma==0.0 || floor_cell_rate>0.0, "a phase floor needs a positive hold-cell rate");
  // Bridge evaluation uses more registers than the elementwise receiver pass.
  const int requested=checked_block(block_size);
  const int block=requested > 256 ? 256 : requested;
  if (times.numel()==0) return;
  const torch::stable::accelerator::DeviceGuard guard(times.get_device_index());
  oscillator_phase_kernel<<<(times.numel()+block-1)/block,block,0,current_cuda_stream(times)>>>(
      times.const_data_ptr<double>(),delays.const_data_ptr<double>(),phase.mutable_data_ptr<float>(),
      invalid.mutable_data_ptr<int32_t>(),times.numel(),diffusion,flicker_scale,floor_sigma,floor_cell_rate,
      sign,static_cast<uint64_t>(seed_base));
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void frontend_noise_forward_cuda(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    const torch::stable::Tensor& phase_rad,
    int64_t num_outer,
    int64_t num_phase,
    double thermal_sigma,
    double lna_gain,
    int64_t seed_base,
    int64_t thermal_stage_id,
    int64_t block_size) {
  const int outer = checked_int(num_outer, "num_outer");
  const int phase = checked_int(num_phase, "num_phase");
  STD_TORCH_CHECK(outer > 0, "num_outer must be positive.");
  STD_TORCH_CHECK(phase > 0, "num_phase must be positive.");
  STD_TORCH_CHECK(thermal_sigma >= 0.0, "thermal_sigma must be non-negative.");
  STD_TORCH_CHECK(seed_base >= 0, "seed_base must be non-negative.");
  const int block = checked_block(block_size);
  const int64_t elements = static_cast<int64_t>(outer) * phase;
  check_pair(x_re, x_im, elements, "x_re", "x_im");
  check_pair(out_re, out_im, elements, "out_re", "out_im");
  check_cuda_float(phase_rad, "phase_rad");
  STD_TORCH_CHECK(
      phase_rad.numel() == static_cast<int64_t>(phase),
      "phase_rad must hold one accumulated phase per slow-time sample.");

  const torch::stable::accelerator::DeviceGuard device_guard(
      out_re.get_device_index());
  const cudaStream_t stream = current_cuda_stream(out_re);
  frontend_noise_forward_kernel<<<
      dim3(static_cast<unsigned int>((elements + block - 1) / block), 1, 1),
      dim3(block, 1, 1),
      0,
      stream>>>(
      x_re.const_data_ptr<float>(),
      x_im.const_data_ptr<float>(),
      phase_rad.const_data_ptr<float>(),
      out_re.mutable_data_ptr<float>(),
      out_im.mutable_data_ptr<float>(),
      elements,
      phase,
      static_cast<float>(thermal_sigma),
      static_cast<float>(lna_gain),
      static_cast<uint64_t>(seed_base),
      static_cast<uint32_t>(thermal_stage_id));
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void frontend_noise_backward_cuda(
    const torch::stable::Tensor& phase_rad,
    const torch::stable::Tensor& grad_out_re,
    const torch::stable::Tensor& grad_out_im,
    torch::stable::Tensor& grad_x_re,
    torch::stable::Tensor& grad_x_im,
    int64_t num_outer,
    int64_t num_phase,
    double lna_gain,
    int64_t block_size) {
  const int outer = checked_int(num_outer, "num_outer");
  const int phase = checked_int(num_phase, "num_phase");
  STD_TORCH_CHECK(outer > 0 && phase > 0, "the signal grid must be non-empty.");
  const int block = checked_block(block_size);
  const int64_t elements = static_cast<int64_t>(outer) * phase;
  check_cuda_float(phase_rad, "phase_rad");
  STD_TORCH_CHECK(
      phase_rad.numel() == static_cast<int64_t>(phase),
      "phase_rad must hold one accumulated phase per slow-time sample.");
  check_pair(grad_out_re, grad_out_im, elements, "grad_out_re", "grad_out_im");
  check_pair(grad_x_re, grad_x_im, elements, "grad_x_re", "grad_x_im");

  const torch::stable::accelerator::DeviceGuard device_guard(
      grad_x_re.get_device_index());
  frontend_noise_linear_kernel<<<
      dim3(static_cast<unsigned int>((elements + block - 1) / block), 1, 1),
      dim3(block, 1, 1),
      0,
      current_cuda_stream(grad_x_re)>>>(
      phase_rad.const_data_ptr<float>(),
      grad_out_re.const_data_ptr<float>(),
      grad_out_im.const_data_ptr<float>(),
      grad_x_re.mutable_data_ptr<float>(),
      grad_x_im.mutable_data_ptr<float>(),
      elements,
      phase,
      static_cast<float>(lna_gain),
      1);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void frontend_noise_jvp_cuda(
    const torch::stable::Tensor& phase_rad,
    const torch::stable::Tensor& tan_x_re,
    const torch::stable::Tensor& tan_x_im,
    torch::stable::Tensor& tan_out_re,
    torch::stable::Tensor& tan_out_im,
    int64_t num_outer,
    int64_t num_phase,
    double lna_gain,
    int64_t block_size) {
  const int outer = checked_int(num_outer, "num_outer");
  const int phase = checked_int(num_phase, "num_phase");
  STD_TORCH_CHECK(outer > 0 && phase > 0, "the signal grid must be non-empty.");
  const int block = checked_block(block_size);
  const int64_t elements = static_cast<int64_t>(outer) * phase;
  check_cuda_float(phase_rad, "phase_rad");
  STD_TORCH_CHECK(
      phase_rad.numel() == static_cast<int64_t>(phase),
      "phase_rad must hold one accumulated phase per slow-time sample.");
  check_pair(tan_x_re, tan_x_im, elements, "tan_x_re", "tan_x_im");
  check_pair(tan_out_re, tan_out_im, elements, "tan_out_re", "tan_out_im");

  const torch::stable::accelerator::DeviceGuard device_guard(
      tan_out_re.get_device_index());
  frontend_noise_linear_kernel<<<
      dim3(static_cast<unsigned int>((elements + block - 1) / block), 1, 1),
      dim3(block, 1, 1),
      0,
      current_cuda_stream(tan_out_re)>>>(
      phase_rad.const_data_ptr<float>(),
      tan_x_re.const_data_ptr<float>(),
      tan_x_im.const_data_ptr<float>(),
      tan_out_re.mutable_data_ptr<float>(),
      tan_out_im.mutable_data_ptr<float>(),
      elements,
      phase,
      static_cast<float>(lna_gain),
      0);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void frontend_agc_forward_cuda(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    torch::stable::Tensor& gain,
    torch::stable::Tensor& rms,
    int64_t dim0,
    int64_t num_groups,
    int64_t dim2,
    double target_rms,
    double min_gain,
    double max_gain,
    int64_t block_size) {
  const int outer = checked_int(dim0, "dim0");
  const int groups = checked_int(num_groups, "num_groups");
  const int inner = checked_int(dim2, "dim2");
  STD_TORCH_CHECK(
      outer > 0 && groups > 0 && inner > 0, "the AGC grid must be non-empty.");
  STD_TORCH_CHECK(target_rms > 0.0, "target_rms must be positive.");
  STD_TORCH_CHECK(min_gain <= max_gain, "min_gain must not exceed max_gain.");
  const int block = checked_block(block_size);
  const int64_t elements =
      static_cast<int64_t>(outer) * groups * static_cast<int64_t>(inner);
  check_pair(x_re, x_im, elements, "x_re", "x_im");
  check_pair(out_re, out_im, elements, "out_re", "out_im");
  check_cuda_float(gain, "gain");
  check_cuda_float(rms, "rms");
  STD_TORCH_CHECK(
      gain.numel() == static_cast<int64_t>(groups) &&
          rms.numel() == static_cast<int64_t>(groups),
      "gain and rms must hold one value per AGC group.");

  const torch::stable::accelerator::DeviceGuard device_guard(
      out_re.get_device_index());
  const cudaStream_t stream = current_cuda_stream(out_re);
  constexpr int reduce_block = 256;
  frontend_agc_measure_kernel<<<
      dim3(groups, 1, 1),
      dim3(reduce_block, 1, 1),
      reduce_block * sizeof(double),
      stream>>>(
      x_re.const_data_ptr<float>(),
      x_im.const_data_ptr<float>(),
      gain.mutable_data_ptr<float>(),
      rms.mutable_data_ptr<float>(),
      outer,
      groups,
      inner,
      static_cast<float>(target_rms),
      static_cast<float>(min_gain),
      static_cast<float>(max_gain));
  STD_CUDA_KERNEL_LAUNCH_CHECK();

  frontend_agc_apply_kernel<<<
      dim3(static_cast<unsigned int>((elements + block - 1) / block), 1, 1),
      dim3(block, 1, 1),
      0,
      stream>>>(
      x_re.const_data_ptr<float>(),
      x_im.const_data_ptr<float>(),
      gain.const_data_ptr<float>(),
      out_re.mutable_data_ptr<float>(),
      out_im.mutable_data_ptr<float>(),
      elements,
      groups,
      inner);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void frontend_agc_backward_cuda(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    const torch::stable::Tensor& gain,
    const torch::stable::Tensor& rms,
    const torch::stable::Tensor& grad_out_re,
    const torch::stable::Tensor& grad_out_im,
    torch::stable::Tensor& grad_x_re,
    torch::stable::Tensor& grad_x_im,
    torch::stable::Tensor& inner,
    int64_t dim0,
    int64_t num_groups,
    int64_t dim2,
    double target_rms,
    double min_gain,
    double max_gain,
    int64_t block_size) {
  launch_agc_linear(
      x_re,
      x_im,
      gain,
      rms,
      grad_out_re,
      grad_out_im,
      grad_x_re,
      grad_x_im,
      inner,
      checked_int(dim0, "dim0"),
      checked_int(num_groups, "num_groups"),
      checked_int(dim2, "dim2"),
      target_rms,
      min_gain,
      max_gain,
      checked_block(block_size));
}

void frontend_agc_jvp_cuda(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    const torch::stable::Tensor& gain,
    const torch::stable::Tensor& rms,
    const torch::stable::Tensor& tan_x_re,
    const torch::stable::Tensor& tan_x_im,
    torch::stable::Tensor& tan_out_re,
    torch::stable::Tensor& tan_out_im,
    torch::stable::Tensor& inner,
    int64_t dim0,
    int64_t num_groups,
    int64_t dim2,
    double target_rms,
    double min_gain,
    double max_gain,
    int64_t block_size) {
  launch_agc_linear(
      x_re,
      x_im,
      gain,
      rms,
      tan_x_re,
      tan_x_im,
      tan_out_re,
      tan_out_im,
      inner,
      checked_int(dim0, "dim0"),
      checked_int(num_groups, "num_groups"),
      checked_int(dim2, "dim2"),
      target_rms,
      min_gain,
      max_gain,
      checked_block(block_size));
}

void frontend_quantize_forward_cuda(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    torch::stable::Tensor& clipped_count,
    int64_t num_elements,
    int64_t bits,
    double full_scale,
    int64_t block_size) {
  const int64_t elements = num_elements;
  STD_TORCH_CHECK(elements > 0, "num_elements must be positive.");
  STD_TORCH_CHECK(bits >= 2 && bits <= 30, "bits must lie in [2, 30].");
  STD_TORCH_CHECK(full_scale > 0.0, "full_scale must be positive.");
  const int block = checked_block(block_size);
  check_pair(x_re, x_im, elements, "x_re", "x_im");
  check_pair(out_re, out_im, elements, "out_re", "out_im");
  check_cuda_int(clipped_count, "clipped_count");
  STD_TORCH_CHECK(
      clipped_count.numel() == 1, "clipped_count must hold exactly one value.");

  const double levels = static_cast<double>(int64_t{1} << bits);
  const double step = 2.0 * full_scale / levels;
  const double half_levels = 0.5 * levels;

  const torch::stable::accelerator::DeviceGuard device_guard(
      out_re.get_device_index());
  frontend_quantize_kernel<<<
      dim3(static_cast<unsigned int>((elements + block - 1) / block), 1, 1),
      dim3(block, 1, 1),
      0,
      current_cuda_stream(out_re)>>>(
      x_re.const_data_ptr<float>(),
      x_im.const_data_ptr<float>(),
      out_re.mutable_data_ptr<float>(),
      out_im.mutable_data_ptr<float>(),
      clipped_count.mutable_data_ptr<int32_t>(),
      elements,
      step,
      -half_levels,
      half_levels - 1.0);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}


void frontend_compress_launch(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    const torch::stable::Tensor& a_re,
    const torch::stable::Tensor& a_im,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    double scale,
    double c5,
    double k_pm,
    double u_sat,
    int64_t block_size,
    int mode) {
  const int64_t elements = x_re.numel();
  STD_TORCH_CHECK(elements > 0, "the compression stage needs a non-empty signal.");
  STD_TORCH_CHECK(scale > 0.0 && u_sat > 0.0, "compression scale and saturation point must be positive.");
  const int block = checked_block(block_size);
  check_pair(x_re, x_im, elements, "x_re", "x_im");
  check_pair(a_re, a_im, elements, "a_re", "a_im");
  check_pair(out_re, out_im, elements, "out_re", "out_im");
  const torch::stable::accelerator::DeviceGuard device_guard(out_re.get_device_index());
  frontend_compress_kernel<<<
      dim3(static_cast<unsigned int>((elements + block - 1) / block), 1, 1),
      dim3(block, 1, 1),
      0,
      current_cuda_stream(out_re)>>>(
      x_re.const_data_ptr<float>(),
      x_im.const_data_ptr<float>(),
      a_re.const_data_ptr<float>(),
      a_im.const_data_ptr<float>(),
      out_re.mutable_data_ptr<float>(),
      out_im.mutable_data_ptr<float>(),
      elements,
      static_cast<float>(scale),
      static_cast<float>(c5),
      static_cast<float>(k_pm),
      static_cast<float>(u_sat),
      mode);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void frontend_compress_forward_cuda(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    double scale,
    double c5,
    double k_pm,
    double u_sat,
    int64_t block_size) {
  frontend_compress_launch(x_re, x_im, x_re, x_im, out_re, out_im, scale, c5, k_pm, u_sat, block_size, 0);
}

void frontend_compress_backward_cuda(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    const torch::stable::Tensor& grad_out_re,
    const torch::stable::Tensor& grad_out_im,
    torch::stable::Tensor& grad_x_re,
    torch::stable::Tensor& grad_x_im,
    double scale,
    double c5,
    double k_pm,
    double u_sat,
    int64_t block_size) {
  frontend_compress_launch(
      x_re, x_im, grad_out_re, grad_out_im, grad_x_re, grad_x_im, scale, c5, k_pm, u_sat, block_size, 2);
}

void frontend_compress_jvp_cuda(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    const torch::stable::Tensor& tan_x_re,
    const torch::stable::Tensor& tan_x_im,
    torch::stable::Tensor& tan_out_re,
    torch::stable::Tensor& tan_out_im,
    double scale,
    double c5,
    double k_pm,
    double u_sat,
    int64_t block_size) {
  frontend_compress_launch(
      x_re, x_im, tan_x_re, tan_x_im, tan_out_re, tan_out_im, scale, c5, k_pm, u_sat, block_size, 1);
}

void frontend_baseband_forward_cuda(
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    const torch::stable::Tensor& times,
    const torch::stable::Tensor& dc_re,
    const torch::stable::Tensor& dc_im,
    int64_t num_outer,
    int64_t num_run,
    int64_t num_rx,
    double flicker_level,
    int64_t levels,
    int64_t seed_base,
    int64_t block_size) {
  const int outer = checked_int(num_outer, "num_outer");
  const int run = checked_int(num_run, "num_run");
  const int receivers = checked_int(num_rx, "num_rx");
  STD_TORCH_CHECK(outer > 0 && run > 0 && receivers > 0, "the baseband grid must be non-empty.");
  STD_TORCH_CHECK(flicker_level >= 0.0 && levels >= 0 && levels <= 62, "invalid flicker parameters.");
  STD_TORCH_CHECK(seed_base >= 0, "seed_base must be non-negative.");
  const int block = checked_block(block_size);
  const int64_t elements = static_cast<int64_t>(outer) * run;
  check_pair(out_re, out_im, elements, "out_re", "out_im");
  check_pair(dc_re, dc_im, receivers, "dc_re", "dc_im");
  STD_TORCH_CHECK(times.is_cuda() && times.scalar_type() == torch::headeronly::ScalarType::Double &&
                      times.is_contiguous() && times.numel() == run,
                  "times must be contiguous CUDA float64 seconds, one per run element.");
  const torch::stable::accelerator::DeviceGuard device_guard(out_re.get_device_index());
  frontend_baseband_kernel<<<
      dim3(static_cast<unsigned int>((elements + block - 1) / block), 1, 1),
      dim3(block, 1, 1),
      0,
      current_cuda_stream(out_re)>>>(
      times.const_data_ptr<double>(),
      dc_re.const_data_ptr<float>(),
      dc_im.const_data_ptr<float>(),
      out_re.mutable_data_ptr<float>(),
      out_im.mutable_data_ptr<float>(),
      elements,
      run,
      receivers,
      static_cast<float>(flicker_level),
      static_cast<int>(levels),
      static_cast<uint64_t>(seed_base));
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

struct DynamicAgcGrid {
  int outer;
  int groups;
  int blocks;
  int inner;
  int64_t elements;
  int slots;
};

DynamicAgcGrid dynamic_agc_grid(int64_t dim0, int64_t num_groups, int64_t num_blocks, int64_t dim2) {
  DynamicAgcGrid grid;
  grid.outer = checked_int(dim0, "dim0");
  grid.groups = checked_int(num_groups, "num_groups");
  grid.blocks = checked_int(num_blocks, "num_blocks");
  grid.inner = checked_int(dim2, "dim2");
  STD_TORCH_CHECK(grid.outer > 0 && grid.groups > 0 && grid.blocks > 0 && grid.inner > 0,
                  "the AGC grid must be non-empty.");
  grid.slots = checked_int(static_cast<int64_t>(grid.groups) * grid.blocks, "num_groups * num_blocks");
  grid.elements = static_cast<int64_t>(grid.outer) * grid.slots * grid.inner;
  return grid;
}

void check_slots(const torch::stable::Tensor& tensor, int slots, const char* name) {
  check_cuda_float(tensor, name);
  STD_TORCH_CHECK(tensor.numel() == static_cast<int64_t>(slots), name, " must hold one value per (group, block).");
}

void launch_slot_reduction(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    const torch::stable::Tensor& a_re,
    const torch::stable::Tensor& a_im,
    torch::stable::Tensor& out,
    const DynamicAgcGrid& grid,
    cudaStream_t stream) {
  constexpr int reduce_block = 256;
  frontend_agc_inner_kernel<<<dim3(grid.slots, 1, 1), dim3(reduce_block, 1, 1), reduce_block * sizeof(double), stream>>>(
      x_re.const_data_ptr<float>(),
      x_im.const_data_ptr<float>(),
      a_re.const_data_ptr<float>(),
      a_im.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      grid.outer,
      grid.slots,
      grid.inner);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_dynamic_combine(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    const torch::stable::Tensor& a_re,
    const torch::stable::Tensor& a_im,
    const torch::stable::Tensor& gain,
    const float* coef,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    const DynamicAgcGrid& grid,
    int block,
    cudaStream_t stream) {
  frontend_agc_dynamic_combine_kernel<<<
      dim3(static_cast<unsigned int>((grid.elements + block - 1) / block), 1, 1),
      dim3(block, 1, 1),
      0,
      stream>>>(
      x_re.const_data_ptr<float>(),
      x_im.const_data_ptr<float>(),
      a_re.const_data_ptr<float>(),
      a_im.const_data_ptr<float>(),
      gain.const_data_ptr<float>(),
      coef,
      out_re.mutable_data_ptr<float>(),
      out_im.mutable_data_ptr<float>(),
      grid.elements,
      grid.slots,
      grid.inner);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void frontend_agc_dynamic_forward_cuda(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    torch::stable::Tensor& gain,
    torch::stable::Tensor& rms,
    torch::stable::Tensor& free_mask,
    int64_t dim0,
    int64_t num_groups,
    int64_t num_blocks,
    int64_t dim2,
    double target_rms,
    double min_gain,
    double max_gain,
    double beta,
    int64_t block_size) {
  const DynamicAgcGrid grid = dynamic_agc_grid(dim0, num_groups, num_blocks, dim2);
  STD_TORCH_CHECK(target_rms > 0.0 && min_gain > 0.0 && min_gain <= max_gain, "invalid AGC limits.");
  STD_TORCH_CHECK(beta > 0.0 && beta <= 1.0, "the AGC loop coefficient must lie in (0, 1].");
  const int block = checked_block(block_size);
  check_pair(x_re, x_im, grid.elements, "x_re", "x_im");
  check_pair(out_re, out_im, grid.elements, "out_re", "out_im");
  check_slots(gain, grid.slots, "gain");
  check_slots(rms, grid.slots, "rms");
  check_slots(free_mask, grid.slots, "free_mask");
  const torch::stable::accelerator::DeviceGuard device_guard(out_re.get_device_index());
  const cudaStream_t stream = current_cuda_stream(out_re);
  constexpr int reduce_block = 256;
  // The static measure kernel over G*K groups writes each block's RMS; the
  // static gain it also writes is overwritten by the scan.
  frontend_agc_measure_kernel<<<dim3(grid.slots, 1, 1), dim3(reduce_block, 1, 1), reduce_block * sizeof(double), stream>>>(
      x_re.const_data_ptr<float>(),
      x_im.const_data_ptr<float>(),
      gain.mutable_data_ptr<float>(),
      rms.mutable_data_ptr<float>(),
      grid.outer,
      grid.slots,
      grid.inner,
      static_cast<float>(target_rms),
      static_cast<float>(min_gain),
      static_cast<float>(max_gain));
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  frontend_agc_dynamic_scan_kernel<<<dim3((grid.groups + 63) / 64, 1, 1), dim3(64, 1, 1), 0, stream>>>(
      rms.const_data_ptr<float>(),
      gain.mutable_data_ptr<float>(),
      free_mask.mutable_data_ptr<float>(),
      grid.groups,
      grid.blocks,
      target_rms,
      log(min_gain),
      log(max_gain),
      beta);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  launch_dynamic_combine(x_re, x_im, x_re, x_im, gain, nullptr, out_re, out_im, grid, block, stream);
}

void frontend_agc_dynamic_linear(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    const torch::stable::Tensor& gain,
    const torch::stable::Tensor& rms,
    const torch::stable::Tensor& free_mask,
    const torch::stable::Tensor& a_re,
    const torch::stable::Tensor& a_im,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    torch::stable::Tensor& scratch,
    int64_t dim0,
    int64_t num_groups,
    int64_t num_blocks,
    int64_t dim2,
    double beta,
    int64_t block_size,
    int reverse) {
  const DynamicAgcGrid grid = dynamic_agc_grid(dim0, num_groups, num_blocks, dim2);
  STD_TORCH_CHECK(beta > 0.0 && beta <= 1.0, "the AGC loop coefficient must lie in (0, 1].");
  const int block = checked_block(block_size);
  check_pair(x_re, x_im, grid.elements, "x_re", "x_im");
  check_pair(a_re, a_im, grid.elements, "a_re", "a_im");
  check_pair(out_re, out_im, grid.elements, "out_re", "out_im");
  check_slots(gain, grid.slots, "gain");
  check_slots(rms, grid.slots, "rms");
  check_slots(free_mask, grid.slots, "free_mask");
  check_slots(scratch, grid.slots, "scratch");
  const torch::stable::accelerator::DeviceGuard device_guard(out_re.get_device_index());
  const cudaStream_t stream = current_cuda_stream(out_re);
  launch_slot_reduction(x_re, x_im, a_re, a_im, scratch, grid, stream);
  frontend_agc_dynamic_linear_scan_kernel<<<dim3((grid.groups + 63) / 64, 1, 1), dim3(64, 1, 1), 0, stream>>>(
      gain.const_data_ptr<float>(),
      rms.const_data_ptr<float>(),
      free_mask.const_data_ptr<float>(),
      scratch.mutable_data_ptr<float>(),
      grid.groups,
      grid.blocks,
      static_cast<double>(grid.outer) * static_cast<double>(grid.inner),
      beta,
      reverse);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  launch_dynamic_combine(
      x_re, x_im, a_re, a_im, gain, scratch.const_data_ptr<float>(), out_re, out_im, grid, block, stream);
}

void frontend_agc_dynamic_backward_cuda(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    const torch::stable::Tensor& gain,
    const torch::stable::Tensor& rms,
    const torch::stable::Tensor& free_mask,
    const torch::stable::Tensor& grad_out_re,
    const torch::stable::Tensor& grad_out_im,
    torch::stable::Tensor& grad_x_re,
    torch::stable::Tensor& grad_x_im,
    torch::stable::Tensor& scratch,
    int64_t dim0,
    int64_t num_groups,
    int64_t num_blocks,
    int64_t dim2,
    double beta,
    int64_t block_size) {
  frontend_agc_dynamic_linear(
      x_re, x_im, gain, rms, free_mask, grad_out_re, grad_out_im, grad_x_re, grad_x_im, scratch, dim0, num_groups,
      num_blocks, dim2, beta, block_size, 1);
}

void frontend_agc_dynamic_jvp_cuda(
    const torch::stable::Tensor& x_re,
    const torch::stable::Tensor& x_im,
    const torch::stable::Tensor& gain,
    const torch::stable::Tensor& rms,
    const torch::stable::Tensor& free_mask,
    const torch::stable::Tensor& tan_x_re,
    const torch::stable::Tensor& tan_x_im,
    torch::stable::Tensor& tan_out_re,
    torch::stable::Tensor& tan_out_im,
    torch::stable::Tensor& scratch,
    int64_t dim0,
    int64_t num_groups,
    int64_t num_blocks,
    int64_t dim2,
    double beta,
    int64_t block_size) {
  frontend_agc_dynamic_linear(
      x_re, x_im, gain, rms, free_mask, tan_x_re, tan_x_im, tan_out_re, tan_out_im, scratch, dim0, num_groups,
      num_blocks, dim2, beta, block_size, 0);
}

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("oscillator_phase_forward", TORCH_BOX(&oscillator_phase_forward_cuda));
  m.impl("frontend_noise_forward", TORCH_BOX(&frontend_noise_forward_cuda));
  m.impl("frontend_noise_backward", TORCH_BOX(&frontend_noise_backward_cuda));
  m.impl("frontend_noise_jvp", TORCH_BOX(&frontend_noise_jvp_cuda));
  m.impl("frontend_agc_forward", TORCH_BOX(&frontend_agc_forward_cuda));
  m.impl("frontend_agc_backward", TORCH_BOX(&frontend_agc_backward_cuda));
  m.impl("frontend_agc_jvp", TORCH_BOX(&frontend_agc_jvp_cuda));
  m.impl("frontend_quantize_forward", TORCH_BOX(&frontend_quantize_forward_cuda));
  m.impl("frontend_compress_forward", TORCH_BOX(&frontend_compress_forward_cuda));
  m.impl("frontend_compress_backward", TORCH_BOX(&frontend_compress_backward_cuda));
  m.impl("frontend_compress_jvp", TORCH_BOX(&frontend_compress_jvp_cuda));
  m.impl("frontend_baseband_forward", TORCH_BOX(&frontend_baseband_forward_cuda));
  m.impl("frontend_agc_dynamic_forward", TORCH_BOX(&frontend_agc_dynamic_forward_cuda));
  m.impl("frontend_agc_dynamic_backward", TORCH_BOX(&frontend_agc_dynamic_backward_cuda));
  m.impl("frontend_agc_dynamic_jvp", TORCH_BOX(&frontend_agc_dynamic_jvp_cuda));
}
