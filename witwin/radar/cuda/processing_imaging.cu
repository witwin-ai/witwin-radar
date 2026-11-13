// Time-domain back-projection of range-compressed samples onto image pixels.
//
// One family, linear in the samples, so the forward is also the forward-mode
// companion (the tangent image is the same map applied to the tangent
// samples) and the backward is the adjoint of the same map:
//
//   L_cv(p)  = |p - tx[c][v]| + |p - rx[c][v]|          two-way path [m]
//   tau      = L_cv(p) / c0                             round-trip delay [s]
//   k        = (L_cv(p) / 2 - range_origin) / range_bin fractional range bin
//   X(k)     = (1 - f) X[i0] + f X[i0 + 1],  i0 = floor(k), f = k - i0
//   cycles   = carrier * tau - curvature * tau^2        compressed-sample phase
//   w        = exp(-j * sign * 2 pi * cycles)           its compensation
//   I[b][g][p] = scale * sum_{c, v in group g} w * X[b][v][c](k)
//
// X is [batch, elements, slow, range] complex (a real view with a trailing
// pair), tx and rx are [slow, elements, 3] float64 metres, pixels [P, 3]
// float64 metres. `groups` is 1 (every element sums into one image) or the
// element count (one image per element, which is what array calibration reads
// at a reflector). A k outside [0, range - 1] contributes nothing; k = range - 1
// reads the last bin with f = 0.
//
// Precision: the path, the delay and the phase are double. A 77 GHz carrier
// makes thousands of cycles of phase at a few metres, and a float32 path at
// 10 m is a 1 um ulp, a twentieth of a degree per ulp; the phase is wrapped to
// its fractional cycle in double before the trigonometric call. The samples
// are interpolated and accumulated in their own precision.
//
// Adjoint (VJP). For a complex linear map y = a x, Torch's cotangent is
// conj(a) * dL/dy*, so
//
//   dX[b][v][c][i0]     += scale * (1 - f) * conj(w) * G[b][g][p]
//   dX[b][v][c][i0 + 1] += scale * f * conj(w) * G[b][g][p]
//
// accumulated with atomics: many pixels read one bin. The geometry (tx, rx,
// pixels) is a host constant of the image and carries no derivative; the Python
// facade refuses a differentiable one before this family runs.

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

struct ImagingLaw {
  double range_origin_m;
  double inverse_range_bin;
  double inverse_speed_s_per_m;
  double carrier_hz;
  double curvature_hz_per_s;
  double phase_sign;
  double scale;
};

struct Tap {
  int64_t index;  // flat offset of bin i0 in the sample array, in complex elements
  int step;       // 1, or 0 on the last bin
  double fraction;
  double cos_w;
  double sin_w;
  bool valid;
};

__device__ __forceinline__ Tap tap(
    const double* __restrict__ tx, const double* __restrict__ rx, double px, double py, double pz, int64_t cv,
    int64_t row, int ranges, const ImagingLaw& law) {
  const double* t = tx + 3 * cv;
  const double* r = rx + 3 * cv;
  const double tdx = px - __ldg(t), tdy = py - __ldg(t + 1), tdz = pz - __ldg(t + 2);
  const double rdx = px - __ldg(r), rdy = py - __ldg(r + 1), rdz = pz - __ldg(r + 2);
  const double path = sqrt(tdx * tdx + tdy * tdy + tdz * tdz) + sqrt(rdx * rdx + rdy * rdy + rdz * rdz);
  const double k = (0.5 * path - law.range_origin_m) * law.inverse_range_bin;
  Tap out{0, 0, 0.0, 1.0, 0.0, false};
  if (!(k >= 0.0 && k <= static_cast<double>(ranges - 1))) return out;
  const double base = floor(k);
  const int i0 = static_cast<int>(base);
  out.index = row * ranges + i0;
  out.step = i0 + 1 < ranges ? 1 : 0;
  out.fraction = k - base;
  const double tau = path * law.inverse_speed_s_per_m;
  const double cycles = law.carrier_hz * tau - law.curvature_hz_per_s * tau * tau;
  double s, c;
  sincospi(-2.0 * law.phase_sign * (cycles - floor(cycles)), &s, &c);
  out.cos_w = c;
  out.sin_w = s;
  out.valid = true;
  return out;
}

// Thread (b, g, p): one output pixel of one image. The pixel index runs
// fastest, so neighbouring threads read neighbouring range bins.
template <typename T>
__global__ void backprojection_kernel(
    const T* __restrict__ samples, const double* __restrict__ tx, const double* __restrict__ rx,
    const double* __restrict__ pixels, T* __restrict__ image, int batch, int elements, int slow, int ranges,
    int count, int groups, ImagingLaw law) {
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(batch) * groups * count;
  if (thread >= total) return;
  const int p = static_cast<int>(thread % count);
  const int g = static_cast<int>((thread / count) % groups);
  const int b = static_cast<int>(thread / (static_cast<int64_t>(count) * groups));
  const double px = pixels[3 * p], py = pixels[3 * p + 1], pz = pixels[3 * p + 2];
  const int first = groups == 1 ? 0 : g;
  const int last = groups == 1 ? elements : g + 1;
  T sum_re = 0, sum_im = 0;
  for (int c = 0; c < slow; ++c) {
    for (int v = first; v < last; ++v) {
      const int64_t row = (static_cast<int64_t>(b) * elements + v) * slow + c;
      const Tap h = tap(tx, rx, px, py, pz, static_cast<int64_t>(c) * elements + v, row, ranges, law);
      if (!h.valid) continue;
      const T* a = samples + 2 * h.index;
      const T* z = a + 2 * h.step;
      const T f = static_cast<T>(h.fraction);
      const T re = a[0] + f * (z[0] - a[0]);
      const T im = a[1] + f * (z[1] - a[1]);
      const T cw = static_cast<T>(h.cos_w), sw = static_cast<T>(h.sin_w);
      sum_re += re * cw - im * sw;
      sum_im += re * sw + im * cw;
    }
  }
  const T scale = static_cast<T>(law.scale);
  image[2 * thread] = sum_re * scale;
  image[2 * thread + 1] = sum_im * scale;
}

template <typename T>
__global__ void backprojection_adjoint_kernel(
    const T* __restrict__ grad_image, const double* __restrict__ tx, const double* __restrict__ rx,
    const double* __restrict__ pixels, T* __restrict__ grad_samples, int batch, int elements, int slow, int ranges,
    int count, int groups, ImagingLaw law) {
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(batch) * groups * count;
  if (thread >= total) return;
  const int p = static_cast<int>(thread % count);
  const int g = static_cast<int>((thread / count) % groups);
  const int b = static_cast<int>(thread / (static_cast<int64_t>(count) * groups));
  const T scale = static_cast<T>(law.scale);
  const T g_re = grad_image[2 * thread] * scale;
  const T g_im = grad_image[2 * thread + 1] * scale;
  if (g_re == T(0) && g_im == T(0)) return;
  const double px = pixels[3 * p], py = pixels[3 * p + 1], pz = pixels[3 * p + 2];
  const int first = groups == 1 ? 0 : g;
  const int last = groups == 1 ? elements : g + 1;
  for (int c = 0; c < slow; ++c) {
    for (int v = first; v < last; ++v) {
      const int64_t row = (static_cast<int64_t>(b) * elements + v) * slow + c;
      const Tap h = tap(tx, rx, px, py, pz, static_cast<int64_t>(c) * elements + v, row, ranges, law);
      if (!h.valid) continue;
      // conj(w) * G
      const T cw = static_cast<T>(h.cos_w), sw = static_cast<T>(h.sin_w);
      const T re = g_re * cw + g_im * sw;
      const T im = g_im * cw - g_re * sw;
      const T f = static_cast<T>(h.fraction);
      T* a = grad_samples + 2 * h.index;
      atomicAdd(a, (T(1) - f) * re);
      atomicAdd(a + 1, (T(1) - f) * im);
      if (h.step != 0 && f != T(0)) {
        atomicAdd(a + 2, f * re);
        atomicAdd(a + 3, f * im);
      }
    }
  }
}

struct Shape {
  int batch, elements, slow, ranges, count, groups;
};

Shape check_inputs(
    const torch::stable::Tensor& samples, const torch::stable::Tensor& tx, const torch::stable::Tensor& rx,
    const torch::stable::Tensor& pixels, const torch::stable::Tensor& image, int64_t groups) {
  const auto dtype = samples.scalar_type();
  STD_TORCH_CHECK(
      dtype == torch::headeronly::ScalarType::Float || dtype == torch::headeronly::ScalarType::Double,
      "samples must be a float32 or float64 real view of complex samples.");
  STD_TORCH_CHECK(samples.is_cuda() && samples.is_contiguous() && samples.dim() == 5 && samples.size(4) == 2,
      "samples must be a contiguous [batch, elements, slow, range, 2] CUDA tensor.");
  STD_TORCH_CHECK(image.is_cuda() && image.is_contiguous() && image.dim() == 4 && image.size(3) == 2,
      "image must be a contiguous [batch, groups, pixels, 2] CUDA tensor.");
  STD_TORCH_CHECK(image.scalar_type() == dtype, "image must share the samples' dtype.");
  for (const auto* tensor : {&tx, &rx, &pixels}) {
    STD_TORCH_CHECK(tensor->is_cuda() && tensor->is_contiguous(), "geometry must be contiguous CUDA tensors.");
    STD_TORCH_CHECK(tensor->scalar_type() == torch::headeronly::ScalarType::Double, "geometry must be float64.");
  }
  Shape shape{};
  shape.batch = checked_int(samples.size(0), "batch");
  shape.elements = checked_int(samples.size(1), "elements");
  shape.slow = checked_int(samples.size(2), "slow-time samples");
  shape.ranges = checked_int(samples.size(3), "range bins");
  shape.count = checked_int(pixels.numel() / 3, "pixels");
  shape.groups = checked_int(groups, "groups");
  STD_TORCH_CHECK(pixels.dim() == 2 && pixels.size(1) == 3, "pixels must be [P, 3].");
  STD_TORCH_CHECK(tx.dim() == 3 && tx.size(0) == shape.slow && tx.size(1) == shape.elements && tx.size(2) == 3,
      "tx must be [slow, elements, 3].");
  STD_TORCH_CHECK(rx.dim() == 3 && rx.size(0) == shape.slow && rx.size(1) == shape.elements && rx.size(2) == 3,
      "rx must be [slow, elements, 3].");
  STD_TORCH_CHECK(shape.groups == 1 || shape.groups == shape.elements, "groups must be 1 or the element count.");
  STD_TORCH_CHECK(image.size(0) == shape.batch && image.size(1) == shape.groups && image.size(2) == shape.count,
      "image must be [batch, groups, pixels, 2].");
  STD_TORCH_CHECK(shape.ranges >= 1, "the range axis must hold at least one bin.");
  const auto device = samples.get_device_index();
  STD_TORCH_CHECK(device == image.get_device_index() && device == tx.get_device_index() &&
      device == rx.get_device_index() && device == pixels.get_device_index(),
      "back-projection tensors must share a CUDA device.");
  return shape;
}

ImagingLaw make_law(
    double range_origin_m, double range_bin_m, double inverse_speed_s_per_m, double carrier_hz,
    double curvature_hz_per_s, double phase_sign, double scale) {
  STD_TORCH_CHECK(range_bin_m > 0.0, "range_bin_m must be positive.");
  STD_TORCH_CHECK(phase_sign == 1.0 || phase_sign == -1.0, "phase_sign must be +1 or -1.");
  return {range_origin_m, 1.0 / range_bin_m, inverse_speed_s_per_m, carrier_hz, curvature_hz_per_s, phase_sign,
          scale};
}

template <bool Adjoint>
void launch(
    const torch::stable::Tensor& input, const torch::stable::Tensor& tx, const torch::stable::Tensor& rx,
    const torch::stable::Tensor& pixels, torch::stable::Tensor output, const Shape& shape, const ImagingLaw& law) {
  const int64_t total = static_cast<int64_t>(shape.batch) * shape.groups * shape.count;
  if (total == 0) return;
  STD_TORCH_CHECK(total <= static_cast<int64_t>(kBlock) * 2147483647LL, "the image is too large for one launch.");
  const torch::stable::accelerator::DeviceGuard guard(input.get_device_index());
  const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock), 1, 1);
  const cudaStream_t stream = current_cuda_stream(output);
  const double* t = tx.const_data_ptr<double>();
  const double* r = rx.const_data_ptr<double>();
  const double* p = pixels.const_data_ptr<double>();
  if (input.scalar_type() == torch::headeronly::ScalarType::Float) {
    if (Adjoint) {
      backprojection_adjoint_kernel<float><<<grid, dim3(kBlock, 1, 1), 0, stream>>>(
          input.const_data_ptr<float>(), t, r, p, output.mutable_data_ptr<float>(), shape.batch, shape.elements,
          shape.slow, shape.ranges, shape.count, shape.groups, law);
    } else {
      backprojection_kernel<float><<<grid, dim3(kBlock, 1, 1), 0, stream>>>(
          input.const_data_ptr<float>(), t, r, p, output.mutable_data_ptr<float>(), shape.batch, shape.elements,
          shape.slow, shape.ranges, shape.count, shape.groups, law);
    }
  } else {
    if (Adjoint) {
      backprojection_adjoint_kernel<double><<<grid, dim3(kBlock, 1, 1), 0, stream>>>(
          input.const_data_ptr<double>(), t, r, p, output.mutable_data_ptr<double>(), shape.batch, shape.elements,
          shape.slow, shape.ranges, shape.count, shape.groups, law);
    } else {
      backprojection_kernel<double><<<grid, dim3(kBlock, 1, 1), 0, stream>>>(
          input.const_data_ptr<double>(), t, r, p, output.mutable_data_ptr<double>(), shape.batch, shape.elements,
          shape.slow, shape.ranges, shape.count, shape.groups, law);
    }
  }
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void backprojection_forward_cuda(
    const torch::stable::Tensor& samples, const torch::stable::Tensor& tx, const torch::stable::Tensor& rx,
    const torch::stable::Tensor& pixels, torch::stable::Tensor image, int64_t groups, double range_origin_m,
    double range_bin_m, double inverse_speed_s_per_m, double carrier_hz, double curvature_hz_per_s,
    double phase_sign, double scale) {
  const Shape shape = check_inputs(samples, tx, rx, pixels, image, groups);
  const ImagingLaw law =
      make_law(range_origin_m, range_bin_m, inverse_speed_s_per_m, carrier_hz, curvature_hz_per_s, phase_sign, scale);
  launch<false>(samples, tx, rx, pixels, image, shape, law);
}

// The map is linear in the samples, so the tangent image is the forward map of
// the tangent samples: the same kernel, registered under its own role.
void backprojection_jvp_cuda(
    const torch::stable::Tensor& tangent_samples, const torch::stable::Tensor& tx, const torch::stable::Tensor& rx,
    const torch::stable::Tensor& pixels, torch::stable::Tensor tangent_image, int64_t groups,
    double range_origin_m, double range_bin_m, double inverse_speed_s_per_m, double carrier_hz,
    double curvature_hz_per_s, double phase_sign, double scale) {
  backprojection_forward_cuda(tangent_samples, tx, rx, pixels, tangent_image, groups, range_origin_m, range_bin_m,
      inverse_speed_s_per_m, carrier_hz, curvature_hz_per_s, phase_sign, scale);
}

// `grad_samples` must arrive zeroed: the adjoint accumulates into it.
void backprojection_backward_cuda(
    const torch::stable::Tensor& grad_image, const torch::stable::Tensor& tx, const torch::stable::Tensor& rx,
    const torch::stable::Tensor& pixels, torch::stable::Tensor grad_samples, int64_t groups,
    double range_origin_m, double range_bin_m, double inverse_speed_s_per_m, double carrier_hz,
    double curvature_hz_per_s, double phase_sign, double scale) {
  const Shape shape = check_inputs(grad_samples, tx, rx, pixels, grad_image, groups);
  const ImagingLaw law =
      make_law(range_origin_m, range_bin_m, inverse_speed_s_per_m, carrier_hz, curvature_hz_per_s, phase_sign, scale);
  launch<true>(grad_image, tx, rx, pixels, grad_samples, shape, law);
}

}  // namespace

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("backprojection_forward", TORCH_BOX(&backprojection_forward_cuda));
  m.impl("backprojection_backward", TORCH_BOX(&backprojection_backward_cuda));
  m.impl("backprojection_jvp", TORCH_BOX(&backprojection_jvp_cuda));
}
