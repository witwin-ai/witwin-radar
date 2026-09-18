// Receiver frontend: phase noise, thermal noise, LNA, AGC, and the ADC.
//
// Native owner of the receive chain. Registers in the single `_radar_native`
// library Radar ships (R-ADR-004).
//
// THE ORDER IS FIXED HERE AND NOWHERE ELSE. The Python runtime applies these
// operators in one sequence with no exceptions:
//
//   0. port      x <- x * sqrt(R)                sqrt(W) -> volts, exactly ONCE
//   1. phase     x <- x * exp(j theta)           supplied oscillator difference
//   2. thermal   x <- x + n,  n ~ CN(0, 2 s^2)   stage 1, INPUT-REFERRED
//   3. lna       x <- x * g_lna
//   4. agc       x <- x * clamp(target/rms, ...)
//   5. adc       x <- clip and round             ALWAYS last
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
//   step = 2 * full_scale / (2^bits - 1)
//   y    = round((clamp(x, -FS, FS) + FS) / step) * step - FS
//
// The clipped-sample count is published as a device diagnostic. Suppressing it
// hides an AGC misconfiguration behind a signal that merely looks compressed.
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

// A common continuous-time Wiener oscillator, evaluated without a sample grid.
// Each integer-second interval has a Philox-keyed endpoint and Brownian bridge.
// Forty dyadic levels resolve 2^-40 s (~0.91 ps). The omitted bridge variance
// is at most diffusion*2^-42 rad^2. Refining/reordering queries cannot change
// the realization. Independent oracle: interval-overlap covariance tests.
__device__ double oscillator_block(double u, int64_t block, uint64_t seed) {
  const uint64_t key = seed ^ (static_cast<uint64_t>(block) * 0x9E3779B97F4A7C15ull);
  double value = u * standard_normal_pair(key, 0, 0).first;
  double left = 0.0, width = 1.0;
  uint64_t node = 1;
  for (int level = 0; level < 40; ++level) {
    const double fraction = (u-left)/width;
    const double tent = 2.0 * fmin(fraction, 1.0-fraction);
    value += 0.5*sqrt(width)*tent*standard_normal_pair(key, 0, node).first;
    width *= 0.5;
    node *= 2;
    if (fraction >= 0.5) { left += width; ++node; }
  }
  return value;
}

__global__ void oscillator_phase_kernel(
    const double* times, const double* delays, float* phase, int64_t count,
    double diffusion, double sign, uint64_t seed) {
  const int64_t i = static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i >= count) return;
  const double t=times[i], s=t-delays[i];
  const int64_t first=static_cast<int64_t>(floor(s)), last=static_cast<int64_t>(floor(t));
  double difference;
  if (first == last) {
    difference=oscillator_block(t-last,last,seed)-oscillator_block(s-first,first,seed);
  } else {
    difference=oscillator_block(1.0,first,seed)-oscillator_block(s-first,first,seed);
    for (int64_t block=first+1; block<last; ++block) difference+=oscillator_block(1.0,block,seed);
    difference+=oscillator_block(t-last,last,seed);
  }
  phase[i]=static_cast<float>(sign*sqrt(diffusion)*difference);
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
    float full_scale,
    float step) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  int clipped = 0;
  const float re = x_re[index];
  const float im = x_im[index];
  if (re < -full_scale || re > full_scale) {
    ++clipped;
  }
  if (im < -full_scale || im > full_scale) {
    ++clipped;
  }
  const float clipped_re = re < -full_scale ? -full_scale : (re > full_scale ? full_scale : re);
  const float clipped_im = im < -full_scale ? -full_scale : (im > full_scale ? full_scale : im);
  out_re[index] = rintf((clipped_re + full_scale) / step) * step - full_scale;
  out_im[index] = rintf((clipped_im + full_scale) / step) * step - full_scale;
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

}  // namespace

void oscillator_phase_forward_cuda(
    const torch::stable::Tensor& times, const torch::stable::Tensor& delays,
    torch::stable::Tensor& phase, double diffusion, double sign,
    int64_t seed_base, int64_t block_size) {
  STD_TORCH_CHECK(times.is_cuda() && delays.is_cuda(), "oscillator times/delays must be CUDA");
  STD_TORCH_CHECK(times.scalar_type()==torch::headeronly::ScalarType::Double &&
                  delays.scalar_type()==torch::headeronly::ScalarType::Double,
                  "oscillator times/delays must be float64 seconds");
  STD_TORCH_CHECK(times.is_contiguous() && delays.is_contiguous(), "oscillator grids must be contiguous");
  check_cuda_float(phase,"phase");
  STD_TORCH_CHECK(times.numel()==delays.numel() && times.numel()==phase.numel(), "oscillator grid shape mismatch");
  STD_TORCH_CHECK(diffusion>=0 && seed_base>=0 && (sign==1.0 || sign==-1.0), "invalid oscillator parameters");
  // Bridge evaluation uses more registers than the elementwise receiver pass.
  const int requested=checked_block(block_size);
  const int block=requested > 256 ? 256 : requested;
  if (times.numel()==0) return;
  const torch::stable::accelerator::DeviceGuard guard(times.get_device_index());
  oscillator_phase_kernel<<<(times.numel()+block-1)/block,block,0,current_cuda_stream(times)>>>(
      times.const_data_ptr<double>(),delays.const_data_ptr<double>(),phase.mutable_data_ptr<float>(),
      times.numel(),diffusion,sign,static_cast<uint64_t>(seed_base));
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
  STD_TORCH_CHECK(bits >= 1 && bits <= 30, "bits must lie in [1, 30].");
  STD_TORCH_CHECK(full_scale > 0.0, "full_scale must be positive.");
  const int block = checked_block(block_size);
  check_pair(x_re, x_im, elements, "x_re", "x_im");
  check_pair(out_re, out_im, elements, "out_re", "out_im");
  check_cuda_int(clipped_count, "clipped_count");
  STD_TORCH_CHECK(
      clipped_count.numel() == 1, "clipped_count must hold exactly one value.");

  const double levels = static_cast<double>(int64_t{1} << bits);
  const double step = 2.0 * full_scale / (levels - 1.0);

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
      static_cast<float>(full_scale),
      static_cast<float>(step));
  STD_CUDA_KERNEL_LAUNCH_CHECK();
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
}
