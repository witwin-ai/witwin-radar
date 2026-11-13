// Sensor pattern weighting and geometric delay for Channel-composed paths.
//
// Channel exclusively owns propagation spreading, transmit power, reference
// phase, and endpoint polarization projection. This operator applies only the
// transmit and receive ELEMENT patterns (each with its own absolute peak gain)
// and sqrt(intensity), while publishing geometric delay and delay rate for
// diagnostics and differentiation.
//
// Per row k, with a_t, a_r the elements' peak voltage gains 10^(G_dBi/20):
//   tau_rt = L / c0
//   power x power:  pattern_gain = G_t * G_r
//                   scale = sqrt(max(intensity, 0)) * sqrt(max(G_t * G_r, 0)) * a_t a_r
//   any field:      A = F (complex field pattern) or sqrt(max(G, 0)) (power)
//                   pattern_gain = |A_t|^2 |A_r|^2
//                   scale = sqrt(max(intensity, 0)) * A_t A_r * a_t a_r   (complex)
//   out = weight_in * scale                                               (complex)
//
// The absolute gain is applied HERE and only here: the tables are normalised
// (power at most 1, |field| at most 1) and Channel's transport carries no
// antenna gain. The power-by-power branch is the original real expression,
// kept verbatim, so a radar without field patterns is bit-identical to one
// built before field patterns existed.
//
// PACKED ELEMENT PATTERNS. Each element pattern is one float32 buffer:
//   [kind, quantity, num_x, num_y, peak_amplitude, 0, 0, 0,
//    x_axis[num_x], y_axis[num_y], values...]
// kind 0 (separable): x_re[num_x], y_re[num_y], x_im[num_x], y_im[num_y]
// kind 1 (map):       re[num_y*num_x], im[num_y*num_x]   (row-major in y)
// quantity 0 is a POWER pattern (imaginary parts zero, unused), 1 a complex
// FIELD pattern. A future polarimetric quantity appends its second component
// after these and takes a new quantity code; the header does not move.
//
// Pattern lookup uses a world-to-pattern frame. Endpoint positions, intensity,
// complex input weight, the pattern frame and the packed tables (values and
// peak amplitudes) are differentiable; velocities and row topology are frozen
// constants. The parameter derivatives are below evaluate_row.
//
// THE ANTENNA PATTERN IS INTERPOLATED, NOT TABULATED IN THE PHYSICS. The
// DIRECTION is differentiable, and the interpolation is piecewise linear in the
// two angles, so the gain has an exact almost-everywhere derivative that this
// kernel carries:
//
//   w      = local(v)             w_j = v . pattern_frame[j]
//   fwd    = -w_2
//   X      = (180/pi) atan2(w_0, fwd)      Y = (180/pi) atan2(w_1, fwd)
//   dX/dw0 = (180/pi) fwd / (w_0^2 + fwd^2)   dX/dw2 = (180/pi) w_0 / (w_0^2 + fwd^2)
//   dY/dw1 = (180/pi) fwd / (w_1^2 + fwd^2)   dY/dw2 = (180/pi) w_1 / (w_1^2 + fwd^2)
//
// A knot of the interpolation table and the two edges of its support are
// genuine non-differentiabilities. This kernel returns the ALMOST-EVERYWHERE
// derivative there - zero outside the support, and the left segment's slope at
// a knot - exactly as the Torch expression it replaces does, because both take
// the same `bucketize` branch. A finite difference that straddles a knot
// disagrees with both, and that disagreement is correct behaviour rather than a
// defect.
//
// Differentiability. The position-only operators differentiate the four
// endpoint positions (tx_pos, rx_pos, site_in, site_out), the intensity and the
// incoming complex weight; the jvp also takes optional table and frame
// tangents and the backward optional keyed table and frame contributions. The
// velocities are constants. `tau_rate` depends on the
// positions through the unit directions and that dependence is carried too:
//
//   d(u . p)/d(head) = ( p - u (u . p) ) / |d|
//
// THE BACKWARD PASS USES NO ATOMICS. Many rows share one transmitter and one
// receiver, so the antenna-position gradient is a real reduction. It is done in
// two kernels with an explicit per-row scratch buffer rather than with
// atomicAdd, so the summation order is ascending row order and is a property of
// the frozen row set rather than of the schedule - the same choice
// `two_way_join.cu` makes for the same reason.
//
// Numerics: lengths, dot products, and the interpolation are single precision
// like the Torch expression they replace, because this family produces an
// AMPLITUDE rather than a phase. There is no cycle count here and no `sincosf`;
// the phase lives entirely in the waveform families, which is why fast math is
// off there and irrelevant here. Fast math is nevertheless left off in this
// translation unit too, so that the clamp and support comparisons mean what
// they say.

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

constexpr float kPiF = 3.14159265358979323846f;
constexpr float kRadToDeg = 57.295779513082320876798f;
constexpr float kMinDistance = 1e-6f;
constexpr float kMinNorm = 1e-12f;
constexpr float kMinDenom = 1e-12f;

constexpr int kRowVia = 0;
constexpr int kRowDirect = 1;

constexpr int kPatternSeparable = 0;
constexpr int kPatternMap = 1;

struct Vec3 {
  float x;
  float y;
  float z;
};

__device__ __forceinline__ Vec3 make_vec3(float x, float y, float z) {
  Vec3 v;
  v.x = x;
  v.y = y;
  v.z = z;
  return v;
}

__device__ __forceinline__ Vec3 load3(const float* __restrict__ base, int64_t index) {
  const int64_t offset = index * 3;
  return make_vec3(base[offset], base[offset + 1], base[offset + 2]);
}

__device__ __forceinline__ Vec3 sub3(const Vec3 a, const Vec3 b) {
  return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ Vec3 add3(const Vec3 a, const Vec3 b) {
  return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ Vec3 scale3(const Vec3 a, float s) {
  return make_vec3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot3(const Vec3 a, const Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float norm3(const Vec3 a) {
  return sqrtf(dot3(a, a));
}

__device__ __forceinline__ Vec3 normalize3(const Vec3 a) {
  const float n = norm3(a);
  return scale3(a, 1.0f / (n < kMinNorm ? kMinNorm : n));
}

// Mirrors torch.bucketize(query, axis) with right=False: the number of axis
// entries strictly less than `query`. Reproduced rather than approximated,
// because the branch it selects is what decides which interpolation segment -
// and therefore which almost-everywhere derivative - this row gets.
__device__ __forceinline__ int bucketize_left(
    const float* __restrict__ axis, int count, float query) {
  int low = 0;
  int high = count;
  while (low < high) {
    const int mid = (low + high) >> 1;
    if (axis[mid] < query) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

struct Interp1d {
  float value;
  float derivative;
};

// Exactly `sensors.interp1d_zero_outside`, including its two edge
// conventions: a query outside the axis returns 0, and a query that lands on
// the first knot takes the degenerate left == right branch whose weight is 0.
__device__ __forceinline__ Interp1d interp1d_zero_outside(
    const float* __restrict__ axis,
    const float* __restrict__ values,
    int count,
    float query) {
  Interp1d result;
  result.value = 0.0f;
  result.derivative = 0.0f;
  if (count <= 0) {
    return result;
  }
  const int upper = bucketize_left(axis, count, query);
  int left = upper - 1;
  left = left < 0 ? 0 : (left > count - 1 ? count - 1 : left);
  int right = upper;
  right = right < 0 ? 0 : (right > count - 1 ? count - 1 : right);

  const float x0 = axis[left];
  const float x1 = axis[right];
  const float y0 = values[left];
  const float y1 = values[right];
  const float span = x1 - x0;
  const float denom = span < kMinDenom ? kMinDenom : span;
  const float weight = (left == right) ? 0.0f : (query - x0) / denom;
  const bool inside = (query >= axis[0]) && (query <= axis[count - 1]);
  if (!inside) {
    return result;
  }
  result.value = y0 + weight * (y1 - y0);
  if (left != right) {
    result.derivative = (y1 - y0) / denom;
  }
  return result;
}

struct PatternGain {
  float value;
  float d_x;  // d(gain) / d(x angle, degrees)
  float d_y;  // d(gain) / d(y angle, degrees)
};

__device__ __forceinline__ PatternGain evaluate_pattern_xy(
    int pattern_kind,
    const float* __restrict__ x_axis,
    const float* __restrict__ y_axis,
    const float* __restrict__ x_values,
    const float* __restrict__ y_values,
    const float* __restrict__ map_values,
    int num_x,
    int num_y,
    float x_deg,
    float y_deg) {
  PatternGain gain;
  gain.value = 0.0f;
  gain.d_x = 0.0f;
  gain.d_y = 0.0f;

  if (pattern_kind == kPatternSeparable) {
    const Interp1d fx = interp1d_zero_outside(x_axis, x_values, num_x, x_deg);
    const Interp1d fy = interp1d_zero_outside(y_axis, y_values, num_y, y_deg);
    gain.value = fx.value * fy.value;
    gain.d_x = fx.derivative * fy.value;
    gain.d_y = fx.value * fy.derivative;
    return gain;
  }

  // The `map` kind: bilinear on a (num_y, num_x) row-major grid, zero outside.
  if (num_x <= 0 || num_y <= 0) {
    return gain;
  }
  const int x_upper = bucketize_left(x_axis, num_x, x_deg);
  const int y_upper = bucketize_left(y_axis, num_y, y_deg);
  int x_left = x_upper - 1;
  x_left = x_left < 0 ? 0 : (x_left > num_x - 1 ? num_x - 1 : x_left);
  int x_right = x_upper;
  x_right = x_right < 0 ? 0 : (x_right > num_x - 1 ? num_x - 1 : x_right);
  int y_low = y_upper - 1;
  y_low = y_low < 0 ? 0 : (y_low > num_y - 1 ? num_y - 1 : y_low);
  int y_high = y_upper;
  y_high = y_high < 0 ? 0 : (y_high > num_y - 1 ? num_y - 1 : y_high);

  const float x0 = x_axis[x_left];
  const float x1 = x_axis[x_right];
  const float y0 = y_axis[y_low];
  const float y1 = y_axis[y_high];
  const float span_x = x1 - x0;
  const float span_y = y1 - y0;
  const float denom_x = span_x < kMinDenom ? kMinDenom : span_x;
  const float denom_y = span_y < kMinDenom ? kMinDenom : span_y;
  const float tx = (x_left == x_right) ? 0.0f : (x_deg - x0) / denom_x;
  const float ty = (y_low == y_high) ? 0.0f : (y_deg - y0) / denom_y;

  const float v00 = map_values[static_cast<int64_t>(y_low) * num_x + x_left];
  const float v10 = map_values[static_cast<int64_t>(y_low) * num_x + x_right];
  const float v01 = map_values[static_cast<int64_t>(y_high) * num_x + x_left];
  const float v11 = map_values[static_cast<int64_t>(y_high) * num_x + x_right];

  const bool inside = (x_deg >= x_axis[0]) && (x_deg <= x_axis[num_x - 1]) &&
      (y_deg >= y_axis[0]) && (y_deg <= y_axis[num_y - 1]);
  if (!inside) {
    return gain;
  }
  gain.value = (1.0f - tx) * (1.0f - ty) * v00 + tx * (1.0f - ty) * v10 +
      (1.0f - tx) * ty * v01 + tx * ty * v11;
  if (x_left != x_right) {
    gain.d_x = ((1.0f - ty) * (v10 - v00) + ty * (v11 - v01)) / denom_x;
  }
  if (y_low != y_high) {
    gain.d_y = ((1.0f - tx) * (v01 - v00) + tx * (v11 - v10)) / denom_y;
  }
  return gain;
}

struct DirectionalGain {
  float value;
  Vec3 gradient;  // d(gain) / d(world direction vector)
};

__device__ __forceinline__ DirectionalGain evaluate_pattern_vector(
    int pattern_kind,
    const float* __restrict__ x_axis,
    const float* __restrict__ y_axis,
    const float* __restrict__ x_values,
    const float* __restrict__ y_values,
    const float* __restrict__ map_values,
    int num_x,
    int num_y,
    const Vec3 axis_right,
    const Vec3 axis_up,
    const Vec3 axis_back,
    const Vec3 world_vector) {
  const float w0 = dot3(world_vector, axis_right);
  const float w1 = dot3(world_vector, axis_up);
  const float w2 = dot3(world_vector, axis_back);
  const float fwd = -w2;

  const float den_x = w0 * w0 + fwd * fwd;
  const float den_y = w1 * w1 + fwd * fwd;
  const float x_deg = kRadToDeg * atan2f(w0, fwd);
  const float y_deg = kRadToDeg * atan2f(w1, fwd);

  const PatternGain gain = evaluate_pattern_xy(
      pattern_kind,
      x_axis,
      y_axis,
      x_values,
      y_values,
      map_values,
      num_x,
      num_y,
      x_deg,
      y_deg);

  const float dx_dw0 = den_x > 0.0f ? kRadToDeg * fwd / den_x : 0.0f;
  const float dx_dw2 = den_x > 0.0f ? kRadToDeg * w0 / den_x : 0.0f;
  const float dy_dw1 = den_y > 0.0f ? kRadToDeg * fwd / den_y : 0.0f;
  const float dy_dw2 = den_y > 0.0f ? kRadToDeg * w1 / den_y : 0.0f;

  const float dg_dw0 = gain.d_x * dx_dw0;
  const float dg_dw1 = gain.d_y * dy_dw1;
  const float dg_dw2 = gain.d_x * dx_dw2 + gain.d_y * dy_dw2;

  DirectionalGain result;
  result.value = gain.value;
  result.gradient = add3(
      add3(scale3(axis_right, dg_dw0), scale3(axis_up, dg_dw1)),
      scale3(axis_back, dg_dw2));
  return result;
}

constexpr int kQuantityPower = 0;
constexpr int kQuantityField = 1;
constexpr int kPatternHeader = 8;

// One element pattern evaluated in a world direction: a complex value and its
// complex gradient with respect to the direction vector. A power pattern
// returns its gain in `re` with `im == 0`.
struct ElementValue {
  int quantity;
  float amplitude;  // the element's peak voltage gain a = 10^(G_dBi/20)
  float re;
  float im;
  Vec3 grad_re;
  Vec3 grad_im;
};

__device__ __forceinline__ ElementValue evaluate_element(
    const float* __restrict__ packed,
    const Vec3 axis_right,
    const Vec3 axis_up,
    const Vec3 axis_back,
    const Vec3 world_vector) {
  const int kind = static_cast<int>(packed[0]);
  const int num_x = static_cast<int>(packed[2]);
  const int num_y = static_cast<int>(packed[3]);
  const float* x_axis = packed + kPatternHeader;
  const float* y_axis = x_axis + num_x;
  const float* values = y_axis + num_y;
  ElementValue out;
  out.quantity = static_cast<int>(packed[1]);
  out.amplitude = packed[4];

  if (kind == kPatternMap) {
    const int cells = num_x * num_y;
    const DirectionalGain real_part = evaluate_pattern_vector(
        kPatternMap, x_axis, y_axis, values, values, values, num_x, num_y, axis_right, axis_up, axis_back,
        world_vector);
    out.re = real_part.value;
    out.grad_re = real_part.gradient;
    if (out.quantity == kQuantityField) {
      const DirectionalGain imaginary_part = evaluate_pattern_vector(
          kPatternMap, x_axis, y_axis, values + cells, values + cells, values + cells, num_x, num_y, axis_right,
          axis_up, axis_back, world_vector);
      out.im = imaginary_part.value;
      out.grad_im = imaginary_part.gradient;
    } else {
      out.im = 0.0f;
      out.grad_im = make_vec3(0.0f, 0.0f, 0.0f);
    }
    return out;
  }

  const float* x_re = values;
  const float* y_re = x_re + num_x;
  if (out.quantity != kQuantityField) {
    const DirectionalGain gain = evaluate_pattern_vector(
        kPatternSeparable, x_axis, y_axis, x_re, y_re, x_re, num_x, num_y, axis_right, axis_up, axis_back,
        world_vector);
    out.re = gain.value;
    out.grad_re = gain.gradient;
    out.im = 0.0f;
    out.grad_im = make_vec3(0.0f, 0.0f, 0.0f);
    return out;
  }
  // Complex separable: F = Fx(x) Fy(y), with the same angle map and the same
  // a.e. interpolation derivatives as the power branch.
  const float* x_im = y_re + num_y;
  const float* y_im = x_im + num_x;
  const float w0 = dot3(world_vector, axis_right);
  const float w1 = dot3(world_vector, axis_up);
  const float w2 = dot3(world_vector, axis_back);
  const float fwd = -w2;
  const float den_x = w0 * w0 + fwd * fwd;
  const float den_y = w1 * w1 + fwd * fwd;
  const float x_deg = kRadToDeg * atan2f(w0, fwd);
  const float y_deg = kRadToDeg * atan2f(w1, fwd);
  const Interp1d fxr = interp1d_zero_outside(x_axis, x_re, num_x, x_deg);
  const Interp1d fxi = interp1d_zero_outside(x_axis, x_im, num_x, x_deg);
  const Interp1d fyr = interp1d_zero_outside(y_axis, y_re, num_y, y_deg);
  const Interp1d fyi = interp1d_zero_outside(y_axis, y_im, num_y, y_deg);
  out.re = fxr.value * fyr.value - fxi.value * fyi.value;
  out.im = fxr.value * fyi.value + fxi.value * fyr.value;
  const float dre_dx = fxr.derivative * fyr.value - fxi.derivative * fyi.value;
  const float dim_dx = fxr.derivative * fyi.value + fxi.derivative * fyr.value;
  const float dre_dy = fxr.value * fyr.derivative - fxi.value * fyi.derivative;
  const float dim_dy = fxr.value * fyi.derivative + fxi.value * fyr.derivative;
  const float dx_dw0 = den_x > 0.0f ? kRadToDeg * fwd / den_x : 0.0f;
  const float dx_dw2 = den_x > 0.0f ? kRadToDeg * w0 / den_x : 0.0f;
  const float dy_dw1 = den_y > 0.0f ? kRadToDeg * fwd / den_y : 0.0f;
  const float dy_dw2 = den_y > 0.0f ? kRadToDeg * w1 / den_y : 0.0f;
  out.grad_re = add3(
      add3(scale3(axis_right, dre_dx * dx_dw0), scale3(axis_up, dre_dy * dy_dw1)),
      scale3(axis_back, dre_dx * dx_dw2 + dre_dy * dy_dw2));
  out.grad_im = add3(
      add3(scale3(axis_right, dim_dx * dx_dw0), scale3(axis_up, dim_dy * dy_dw1)),
      scale3(axis_back, dim_dx * dx_dw2 + dim_dy * dy_dw2));
  return out;
}

// A power element as a voltage amplitude sqrt(max(G, 0)), with the a.e.
// derivative 0.5 / sqrt(G) dG (zero at and below the clamp).
__device__ __forceinline__ void element_amplitude(
    const ElementValue& element, float* re, float* im, Vec3* grad_re, Vec3* grad_im) {
  if (element.quantity == kQuantityField) {
    *re = element.re;
    *im = element.im;
    *grad_re = element.grad_re;
    *grad_im = element.grad_im;
    return;
  }
  const float amplitude = element.re > 0.0f ? sqrtf(element.re) : 0.0f;
  *re = amplitude;
  *im = 0.0f;
  *grad_re = element.re > 0.0f ? scale3(element.grad_re, 0.5f / amplitude) : make_vec3(0.0f, 0.0f, 0.0f);
  *grad_im = make_vec3(0.0f, 0.0f, 0.0f);
}

// Everything one row needs, primal and partial derivatives together, so that
// the three operators cannot drift apart: forward reads the values, jvp
// contracts the partials with the input tangents, and backward contracts them
// with the output cotangents. There is exactly one expression for each
// quantity in this file.
struct RowTerm {
  float tau_rt;
  float tau_rate;
  float pattern_gain;  // G_t * G_r, the published normalised power product
  float scale;         // the factor the weight is multiplied by (real part)
  float scale_im;      // its imaginary part; zero unless a field pattern is used
  bool complex_scale;  // false: the power-by-power branch, scale is real

  // d(scale) and d(tau_*) with respect to the four differentiable positions.
  Vec3 dscale_dsite_in;
  Vec3 dscale_dsite_out;
  Vec3 dscale_dtx;
  Vec3 dscale_drx;
  Vec3 dtau_rt_dsite_in;
  Vec3 dtau_rt_dsite_out;
  Vec3 dtau_rt_dtx;
  Vec3 dtau_rt_drx;
  Vec3 dtau_rate_dsite_in;
  Vec3 dtau_rate_dsite_out;
  Vec3 dtau_rate_dtx;
  Vec3 dtau_rate_drx;

  float dscale_dintensity;

  // Imaginary parts of d(scale), populated only when complex_scale is true.
  Vec3 dscale_im_dsite_in;
  Vec3 dscale_im_dsite_out;
  Vec3 dscale_im_dtx;
  Vec3 dscale_im_drx;
  float dscale_im_dintensity;
};

struct RowInputs {
  const float* tx_pos;
  const float* rx_pos;
  const float* tx_velocity;
  const float* rx_velocity;
  const float* site_in;
  const float* site_out;
  const float* site_velocity;
  const float* fixed_length_m;
  const int64_t* tx_index;
  const int64_t* rx_index;
  const int32_t* row_kind;
  const float* intensity;
  const float* pattern_frame;
  const float* tx_pattern;
  const float* rx_pattern;
  int num_tx;
  int num_rx;
  float c0;
  // Slot stride: the index and per-row tables describe ONE slot of
  // rows_per_slot rows; row k of a slot-major batch reads table row
  // k % rows_per_slot and its antennas are offset by the slot times the
  // per-slot antenna counts. With one slot this is the plain launch.
  int64_t rows_per_slot;
  int tx_per_slot;
  int rx_per_slot;
};

// The strided antenna of row k: table row k % rows_per_slot, clamped to one
// slot's antennas, offset by its slot.
__device__ __forceinline__ int64_t strided_antenna(
    const int64_t* __restrict__ index, int64_t k, int64_t rows_per_slot, int per_slot) {
  int64_t a = index[k % rows_per_slot];
  a = a < 0 ? 0 : (a >= per_slot ? per_slot - 1 : a);
  return a + (k / rows_per_slot) * per_slot;
}

__device__ __forceinline__ RowTerm evaluate_row(const RowInputs& in, int64_t k) {
  RowTerm term;
  const Vec3 zero = make_vec3(0.0f, 0.0f, 0.0f);
  term.tau_rt = 0.0f;
  term.tau_rate = 0.0f;
  term.pattern_gain = 0.0f;
  term.scale = 0.0f;
  term.dscale_dsite_in = zero;
  term.dscale_dsite_out = zero;
  term.dscale_dtx = zero;
  term.dscale_drx = zero;
  term.dtau_rt_dsite_in = zero;
  term.dtau_rt_dsite_out = zero;
  term.dtau_rt_dtx = zero;
  term.dtau_rt_drx = zero;
  term.dtau_rate_dsite_in = zero;
  term.dtau_rate_dsite_out = zero;
  term.dtau_rate_dtx = zero;
  term.dtau_rate_drx = zero;
  term.dscale_dintensity = 0.0f;
  term.scale_im = 0.0f;
  term.complex_scale = false;
  term.dscale_im_dsite_in = zero;
  term.dscale_im_dsite_out = zero;
  term.dscale_im_dtx = zero;
  term.dscale_im_drx = zero;
  term.dscale_im_dintensity = 0.0f;

  const int64_t r = k % in.rows_per_slot;
  const int64_t ti = strided_antenna(in.tx_index, k, in.rows_per_slot, in.tx_per_slot);
  const int64_t ri = strided_antenna(in.rx_index, k, in.rows_per_slot, in.rx_per_slot);
  const int kind = in.row_kind[r];

  const Vec3 tx = load3(in.tx_pos, ti);
  const Vec3 rx = load3(in.rx_pos, ri);
  const Vec3 v_tx = load3(in.tx_velocity, ti % in.tx_per_slot);
  const Vec3 v_rx = load3(in.rx_velocity, ri % in.rx_per_slot);

  const Vec3 axis_right = load3(in.pattern_frame, 0);
  const Vec3 axis_up = load3(in.pattern_frame, 1);
  const Vec3 axis_back = load3(in.pattern_frame, 2);

  Vec3 direction_tx;
  Vec3 direction_rx;
  float length;
  Vec3 dlen_dsite_in = zero;
  Vec3 dlen_dsite_out = zero;
  Vec3 dlen_dtx = zero;
  Vec3 dlen_drx = zero;
  Vec3 drate_dsite_in = zero;
  Vec3 drate_dsite_out = zero;
  Vec3 drate_dtx = zero;
  Vec3 drate_drx = zero;
  float rate = 0.0f;

  // Which position each pattern direction differentiates against, so the
  // chain rule below is written once for both row kinds.
  Vec3* dtx_dir_head;
  Vec3* dtx_dir_tail;
  Vec3* drx_dir_head;
  Vec3* drx_dir_tail;
  Vec3 grad_site_in_pattern = zero;
  Vec3 grad_site_out_pattern = zero;
  Vec3 grad_tx_pattern = zero;
  Vec3 grad_rx_pattern = zero;

  if (kind == kRowDirect) {
    const Vec3 d_in = sub3(rx, tx);
    const float raw = norm3(d_in);
    const float a = raw < kMinDistance ? kMinDistance : raw;
    const Vec3 u_in = scale3(d_in, 1.0f / a);
    // The LENGTH uses the raw norm and the DIRECTION uses the clamped one:
    // the length is a distance, and the clamp exists only to keep a unit
    // vector finite at a coincident pair.
    length = raw;
    dlen_drx = u_in;
    dlen_dtx = scale3(u_in, -1.0f);

    const Vec3 p = sub3(v_rx, v_tx);
    const float u_dot_p = dot3(u_in, p);
    rate = u_dot_p;
    const Vec3 transverse = scale3(sub3(p, scale3(u_in, u_dot_p)), 1.0f / a);
    drate_drx = transverse;
    drate_dtx = scale3(transverse, -1.0f);

    direction_tx = sub3(rx, tx);
    direction_rx = sub3(tx, rx);
    dtx_dir_head = &grad_rx_pattern;
    dtx_dir_tail = &grad_tx_pattern;
    drx_dir_head = &grad_tx_pattern;
    drx_dir_tail = &grad_rx_pattern;
  } else {
    const Vec3 site_in = load3(in.site_in, k);
    const Vec3 site_out = load3(in.site_out, k);
    const Vec3 v_site = load3(in.site_velocity, r);
    const Vec3 d_in = sub3(site_in, tx);
    const Vec3 d_out = sub3(rx, site_out);
    const float raw_a = norm3(d_in);
    const float raw_b = norm3(d_out);
    const float a = raw_a < kMinDistance ? kMinDistance : raw_a;
    const float b = raw_b < kMinDistance ? kMinDistance : raw_b;
    const Vec3 u_in = scale3(d_in, 1.0f / a);
    const Vec3 u_out = scale3(d_out, 1.0f / b);

    length = raw_a + in.fixed_length_m[r] + raw_b;
    dlen_dsite_in = u_in;
    dlen_dtx = scale3(u_in, -1.0f);
    dlen_dsite_out = scale3(u_out, -1.0f);
    dlen_drx = u_out;

    const Vec3 p = sub3(v_site, v_tx);
    const Vec3 q = sub3(v_rx, v_site);
    const float u_dot_p = dot3(u_in, p);
    const float u_dot_q = dot3(u_out, q);
    rate = u_dot_p + u_dot_q;
    const Vec3 transverse_in = scale3(sub3(p, scale3(u_in, u_dot_p)), 1.0f / a);
    const Vec3 transverse_out = scale3(sub3(q, scale3(u_out, u_dot_q)), 1.0f / b);
    drate_dsite_in = transverse_in;
    drate_dtx = scale3(transverse_in, -1.0f);
    drate_drx = transverse_out;
    drate_dsite_out = scale3(transverse_out, -1.0f);

    direction_tx = sub3(site_in, tx);
    direction_rx = sub3(site_out, rx);
    dtx_dir_head = &grad_site_in_pattern;
    dtx_dir_tail = &grad_tx_pattern;
    drx_dir_head = &grad_site_out_pattern;
    drx_dir_tail = &grad_rx_pattern;
  }

  const ElementValue element_tx =
      evaluate_element(in.tx_pattern, axis_right, axis_up, axis_back, direction_tx);
  const ElementValue element_rx =
      evaluate_element(in.rx_pattern, axis_right, axis_up, axis_back, direction_rx);
  const float peak = element_tx.amplitude * element_rx.amplitude;

  const float raw_intensity = in.intensity[r];
  const float clamped_intensity = raw_intensity > 0.0f ? raw_intensity : 0.0f;
  const float intensity_amplitude = sqrtf(clamped_intensity);

  if (element_tx.quantity != kQuantityField && element_rx.quantity != kQuantityField) {
    const float gain_product = element_tx.re * element_rx.re;
    term.pattern_gain = gain_product;
    const float clamped_gain = gain_product > 0.0f ? gain_product : 0.0f;
    const float gain_amplitude = sqrtf(clamped_gain);

    const float scale = intensity_amplitude * gain_amplitude * peak;
    term.scale = scale;

    // d(scale)/d(intensity): the derivative of the square root, taken to be
    // zero at and below the clamp, which is the almost-everywhere derivative
    // of sqrt(clamp(x, 0)).
    if (raw_intensity > 0.0f) {
      term.dscale_dintensity = 0.5f / intensity_amplitude * gain_amplitude * peak;
    }

    // d(scale)/d(direction), through sqrt(clamp(G_t G_r, 0)).
    if (gain_product > 0.0f) {
      const float common = 0.5f / gain_amplitude * intensity_amplitude * peak;
      const Vec3 dscale_dtx_dir = scale3(element_tx.grad_re, common * element_rx.re);
      const Vec3 dscale_drx_dir = scale3(element_rx.grad_re, common * element_tx.re);
      *dtx_dir_head = add3(*dtx_dir_head, dscale_dtx_dir);
      *dtx_dir_tail = sub3(*dtx_dir_tail, dscale_dtx_dir);
      *drx_dir_head = add3(*drx_dir_head, dscale_drx_dir);
      *drx_dir_tail = sub3(*drx_dir_tail, dscale_drx_dir);
    }
  } else {
    // A field pattern on either side: the scale is the complex product of
    // the two element amplitudes.
    float at_re, at_im, ar_re, ar_im;
    Vec3 gt_re, gt_im, gr_re, gr_im;
    element_amplitude(element_tx, &at_re, &at_im, &gt_re, &gt_im);
    element_amplitude(element_rx, &ar_re, &ar_im, &gr_re, &gr_im);
    const float product_re = at_re * ar_re - at_im * ar_im;
    const float product_im = at_re * ar_im + at_im * ar_re;
    term.pattern_gain = (at_re * at_re + at_im * at_im) * (ar_re * ar_re + ar_im * ar_im);
    term.complex_scale = true;
    const float common = intensity_amplitude * peak;
    term.scale = common * product_re;
    term.scale_im = common * product_im;
    if (raw_intensity > 0.0f) {
      const float d = 0.5f / intensity_amplitude * peak;
      term.dscale_dintensity = d * product_re;
      term.dscale_im_dintensity = d * product_im;
    }
    // d(A_t A_r) = dA_t A_r (transmit direction) + A_t dA_r (receive direction).
    const Vec3 dtx_re = scale3(sub3(scale3(gt_re, ar_re), scale3(gt_im, ar_im)), common);
    const Vec3 dtx_im = scale3(add3(scale3(gt_re, ar_im), scale3(gt_im, ar_re)), common);
    const Vec3 drx_re = scale3(sub3(scale3(gr_re, at_re), scale3(gr_im, at_im)), common);
    const Vec3 drx_im = scale3(add3(scale3(gr_im, at_re), scale3(gr_re, at_im)), common);
    *dtx_dir_head = add3(*dtx_dir_head, dtx_re);
    *dtx_dir_tail = sub3(*dtx_dir_tail, dtx_re);
    *drx_dir_head = add3(*drx_dir_head, drx_re);
    *drx_dir_tail = sub3(*drx_dir_tail, drx_re);
    // The imaginary parts follow the same head/tail routing.
    Vec3* im_head_tx = dtx_dir_head == &grad_rx_pattern ? &term.dscale_im_drx : &term.dscale_im_dsite_in;
    Vec3* im_tail_tx = &term.dscale_im_dtx;
    Vec3* im_head_rx = drx_dir_head == &grad_tx_pattern ? &term.dscale_im_dtx : &term.dscale_im_dsite_out;
    Vec3* im_tail_rx = &term.dscale_im_drx;
    *im_head_tx = add3(*im_head_tx, dtx_im);
    *im_tail_tx = sub3(*im_tail_tx, dtx_im);
    *im_head_rx = add3(*im_head_rx, drx_im);
    *im_tail_rx = sub3(*im_tail_rx, drx_im);
  }
  term.tau_rt = length / in.c0;
  term.tau_rate = rate / in.c0;

  // The scale depends on the positions only through the pattern directions:
  // Channel owns spreading, so the path length carries no amplitude here.
  term.dscale_dsite_in = grad_site_in_pattern;
  term.dscale_dsite_out = grad_site_out_pattern;
  term.dscale_dtx = grad_tx_pattern;
  term.dscale_drx = grad_rx_pattern;

  const float inv_c0 = 1.0f / in.c0;
  term.dtau_rt_dsite_in = scale3(dlen_dsite_in, inv_c0);
  term.dtau_rt_dsite_out = scale3(dlen_dsite_out, inv_c0);
  term.dtau_rt_dtx = scale3(dlen_dtx, inv_c0);
  term.dtau_rt_drx = scale3(dlen_drx, inv_c0);
  term.dtau_rate_dsite_in = scale3(drate_dsite_in, inv_c0);
  term.dtau_rate_dsite_out = scale3(drate_dsite_out, inv_c0);
  term.dtau_rate_dtx = scale3(drate_dtx, inv_c0);
  term.dtau_rate_drx = scale3(drate_drx, inv_c0);
  return term;
}

// ---------------------------------------------------------------------------
// The element patterns' own parameters: table values, peak gain and frame.
// ---------------------------------------------------------------------------
//
// With each element written as a complex amplitude A (the field F, or
// sqrt(max(G, 0)) for a power pattern) the scale of every row is
//
//   scale = q A_t A_r,   q = sqrt(max(I, 0)) a_t a_r,
//
// which is the power-by-power expression above as well (sqrt(G_t G_r) =
// sqrt(G_t) sqrt(G_r) for the non-negative tables a power pattern holds). The
// table VALUES enter A linearly through the interpolation, whose weights are
// constants of the row's direction:
//
//   map:        A = sum_c w_c T[c]            (bilinear, four cells)
//   separable:  A = Fx Fy,  Fx = sum_c wx_c Tx[c],  Fy = sum_c wy_c Ty[c]
//
// a power element then takes dA = dG / (2 sqrt(G)) for G > 0 and 0 at the
// clamp, the same almost-everywhere convention the direction derivative uses.
// The peak amplitude a (packed slot 4) enters q. The pattern frame enters
// through w_j = v . axis_j: dA/daxis_j = v (dA/dw_j), and at the orthonormal
// frame the radar builds dA/dw_j = (dA/dv) . axis_j.
//
// A real loss L sees a row through the cotangent of its complex scale,
// Gs = g_scale + j g_scale_im (the same pair backward_rows forms), as
// dL = Re(conj(Gs) dscale), so the cotangent of A_t is q conj(A_r) Gs and that
// of a_t is sqrt(I) a_r Re(conj(Gs) A_t A_r).
//
// THE VJP IS A KEYED LIST, NOT AN ATOMIC SCATTER. Many rows interpolate the
// same knots. Each row writes its contributions as (key, value) pairs - key
// the flat index into [tx_pattern | rx_pattern | frame] - and the caller
// stable-sorts the keys and reduces each key's run in ascending order, so the
// summation order is a property of the row set and the reduction is
// deterministic, the same guarantee the antenna reduction above gives.

// One axis of the linear interpolation: the two knots and their weights, with
// interp1d_zero_outside's edge conventions (outside: no knot carries weight;
// a degenerate segment puts all of it on its one knot).
struct Stencil1d {
  int left;
  int right;
  float w_left;
  float w_right;
  bool inside;
};

__device__ __forceinline__ Stencil1d stencil1d(const float* __restrict__ axis, int count, float query) {
  const int upper = bucketize_left(axis, count, query);
  int left = upper - 1;
  left = left < 0 ? 0 : (left > count - 1 ? count - 1 : left);
  int right = upper;
  right = right < 0 ? 0 : (right > count - 1 ? count - 1 : right);
  const float x0 = axis[left];
  const float span = axis[right] - x0;
  const float denom = span < kMinDenom ? kMinDenom : span;
  const float weight = (left == right) ? 0.0f : (query - x0) / denom;
  Stencil1d s;
  s.left = left;
  s.right = right;
  s.inside = (query >= axis[0]) && (query <= axis[count - 1]);
  s.w_left = s.inside ? 1.0f - weight : 0.0f;
  s.w_right = s.inside ? weight : 0.0f;
  return s;
}

// Where one element's value in one direction reads its table: the knots,
// their weights, and the offsets of the real and imaginary blocks in the
// packed buffer. Map: cell[0..3] are flat (y, x) cells. Separable: cell[0..1]
// index the x cut, cell[2..3] the y cut.
struct Stencil {
  int kind;
  int quantity;
  int num_x;
  int num_y;
  int64_t base;   // first value of the real block
  int64_t block;  // length of the real block (offset of the imaginary one)
  int cell[4];
  float weight[4];
};

__device__ __forceinline__ Stencil pattern_stencil(
    const float* __restrict__ packed, const Vec3 axis_right, const Vec3 axis_up, const Vec3 axis_back,
    const Vec3 world_vector) {
  Stencil s;
  s.kind = static_cast<int>(packed[0]);
  s.quantity = static_cast<int>(packed[1]);
  s.num_x = static_cast<int>(packed[2]);
  s.num_y = static_cast<int>(packed[3]);
  s.base = kPatternHeader + s.num_x + s.num_y;
  s.block = s.kind == kPatternMap ? static_cast<int64_t>(s.num_x) * s.num_y : s.num_x + s.num_y;
  const float* x_axis = packed + kPatternHeader;
  const float* y_axis = x_axis + s.num_x;
  const float w0 = dot3(world_vector, axis_right);
  const float w1 = dot3(world_vector, axis_up);
  const float fwd = -dot3(world_vector, axis_back);
  const float x_deg = kRadToDeg * atan2f(w0, fwd);
  const float y_deg = kRadToDeg * atan2f(w1, fwd);
  const Stencil1d sx = stencil1d(x_axis, s.num_x, x_deg);
  const Stencil1d sy = stencil1d(y_axis, s.num_y, y_deg);
  if (s.kind == kPatternMap) {
    const bool inside = sx.inside && sy.inside;
    // Bilinear: tx and ty are the right/high weights of the two axes.
    const float tx = sx.w_right;
    const float ty = sy.w_right;
    s.cell[0] = sy.left * s.num_x + sx.left;
    s.cell[1] = sy.left * s.num_x + sx.right;
    s.cell[2] = sy.right * s.num_x + sx.left;
    s.cell[3] = sy.right * s.num_x + sx.right;
    s.weight[0] = inside ? (1.0f - tx) * (1.0f - ty) : 0.0f;
    s.weight[1] = inside ? tx * (1.0f - ty) : 0.0f;
    s.weight[2] = inside ? (1.0f - tx) * ty : 0.0f;
    s.weight[3] = inside ? tx * ty : 0.0f;
    return s;
  }
  s.cell[0] = sx.left;
  s.cell[1] = sx.right;
  s.cell[2] = sy.left;
  s.cell[3] = sy.right;
  s.weight[0] = sx.w_left;
  s.weight[1] = sx.w_right;
  s.weight[2] = sy.w_left;
  s.weight[3] = sy.w_right;
  return s;
}

// The separable cuts' values at the query, read from `values` (the primal or
// a tangent buffer laid out like it): sum over each axis's two knots.
__device__ __forceinline__ void separable_cuts(
    const Stencil& s, const float* __restrict__ values, float* x_re, float* x_im, float* y_re, float* y_im) {
  const float* xr = values + s.base;
  const float* yr = xr + s.num_x;
  const float* xi = values + s.base + s.block;
  const float* yi = xi + s.num_x;
  const bool field = s.quantity == kQuantityField;
  *x_re = s.weight[0] * xr[s.cell[0]] + s.weight[1] * xr[s.cell[1]];
  *y_re = s.weight[2] * yr[s.cell[2]] + s.weight[3] * yr[s.cell[3]];
  *x_im = field ? s.weight[0] * xi[s.cell[0]] + s.weight[1] * xi[s.cell[1]] : 0.0f;
  *y_im = field ? s.weight[2] * yi[s.cell[2]] + s.weight[3] * yi[s.cell[3]] : 0.0f;
}

// The element's raw table value (G for power, F for field) as a linear
// function of `values`, for the tangent: the same stencil, the tangent's
// numbers. `primal` supplies the other cut of a separable product.
__device__ __forceinline__ void table_tangent(
    const Stencil& s, const float* __restrict__ primal, const float* __restrict__ tangent, float* d_re, float* d_im) {
  if (s.kind == kPatternMap) {
    float re = 0.0f;
    float im = 0.0f;
    for (int c = 0; c < 4; ++c) {
      re += s.weight[c] * tangent[s.base + s.cell[c]];
      if (s.quantity == kQuantityField) im += s.weight[c] * tangent[s.base + s.block + s.cell[c]];
    }
    *d_re = re;
    *d_im = im;
    return;
  }
  float pxr, pxi, pyr, pyi, txr, txi, tyr, tyi;
  separable_cuts(s, primal, &pxr, &pxi, &pyr, &pyi);
  separable_cuts(s, tangent, &txr, &txi, &tyr, &tyi);
  // d(Fx Fy) = dFx Fy + Fx dFy, complex.
  *d_re = (txr * pyr - txi * pyi) + (pxr * tyr - pxi * tyi);
  *d_im = (txr * pyi + txi * pyr) + (pxr * tyi + pxi * tyr);
}

// Everything the parameter derivatives of one row need, per element side.
struct ElementSide {
  Stencil stencil;
  Vec3 direction;  // the world vector the element was evaluated in
  float raw_re;    // G (power) or Re F (field)
  float a_re;      // the amplitude A
  float a_im;
  Vec3 ga_re;      // dA / d(world vector)
  Vec3 ga_im;
  float da_dg;     // dA/dG of a power element (0.5 / sqrt(G), 0 at the clamp); 1 for a field
  float peak;      // a = 10^(G_dBi / 20)
};

struct RowParameters {
  ElementSide tx;
  ElementSide rx;
  float root_intensity;  // sqrt(max(I, 0))
};

__device__ __forceinline__ ElementSide element_side(
    const float* __restrict__ packed, const Vec3 axis_right, const Vec3 axis_up, const Vec3 axis_back,
    const Vec3 direction) {
  ElementSide side;
  const ElementValue value = evaluate_element(packed, axis_right, axis_up, axis_back, direction);
  side.stencil = pattern_stencil(packed, axis_right, axis_up, axis_back, direction);
  side.direction = direction;
  side.raw_re = value.re;
  side.peak = value.amplitude;
  element_amplitude(value, &side.a_re, &side.a_im, &side.ga_re, &side.ga_im);
  side.da_dg = value.quantity == kQuantityField ? 1.0f : (value.re > 0.0f ? 0.5f / side.a_re : 0.0f);
  return side;
}

__device__ __forceinline__ RowParameters row_parameters(const RowInputs& in, int64_t k) {
  const int64_t r = k % in.rows_per_slot;
  const int64_t ti = strided_antenna(in.tx_index, k, in.rows_per_slot, in.tx_per_slot);
  const int64_t ri = strided_antenna(in.rx_index, k, in.rows_per_slot, in.rx_per_slot);
  const Vec3 tx = load3(in.tx_pos, ti);
  const Vec3 rx = load3(in.rx_pos, ri);
  Vec3 direction_tx;
  Vec3 direction_rx;
  // The two directions evaluate_row looks the elements up in.
  if (in.row_kind[r] == kRowDirect) {
    direction_tx = sub3(rx, tx);
    direction_rx = sub3(tx, rx);
  } else {
    direction_tx = sub3(load3(in.site_in, k), tx);
    direction_rx = sub3(load3(in.site_out, k), rx);
  }
  const Vec3 axis_right = load3(in.pattern_frame, 0);
  const Vec3 axis_up = load3(in.pattern_frame, 1);
  const Vec3 axis_back = load3(in.pattern_frame, 2);
  RowParameters p;
  p.tx = element_side(in.tx_pattern, axis_right, axis_up, axis_back, direction_tx);
  p.rx = element_side(in.rx_pattern, axis_right, axis_up, axis_back, direction_rx);
  const float raw_intensity = in.intensity[r];
  p.root_intensity = sqrtf(raw_intensity > 0.0f ? raw_intensity : 0.0f);
  return p;
}

// dA of one side: the table tangent, the frame tangent, and nothing else.
__device__ __forceinline__ void side_tangent(
    const ElementSide& side, const float* __restrict__ primal, const float* __restrict__ tangent,
    const float* __restrict__ pattern_frame, const float* __restrict__ tan_frame, float* d_re, float* d_im) {
  float re = 0.0f;
  float im = 0.0f;
  if (tangent != nullptr) {
    float g_re, g_im;
    table_tangent(side.stencil, primal, tangent, &g_re, &g_im);
    re += side.da_dg * g_re;
    im += side.stencil.quantity == kQuantityField ? g_im : 0.0f;
  }
  if (tan_frame != nullptr) {
    for (int j = 0; j < 3; ++j) {
      const Vec3 axis = load3(pattern_frame, j);
      const float along = dot3(side.direction, load3(tan_frame, j));
      re += dot3(side.ga_re, axis) * along;
      im += dot3(side.ga_im, axis) * along;
    }
  }
  *d_re = re;
  *d_im = im;
}

// d(scale) from the parameter tangents: q (dA_t A_r + A_t dA_r) + dq A_t A_r.
__device__ __forceinline__ void parameter_scale_tangent(
    const RowInputs& in, int64_t k, const float* __restrict__ tan_tx_pattern,
    const float* __restrict__ tan_rx_pattern, const float* __restrict__ tan_frame, float* d_scale,
    float* d_scale_im) {
  const RowParameters p = row_parameters(in, k);
  float dt_re, dt_im, dr_re, dr_im;
  side_tangent(p.tx, in.tx_pattern, tan_tx_pattern, in.pattern_frame, tan_frame, &dt_re, &dt_im);
  side_tangent(p.rx, in.rx_pattern, tan_rx_pattern, in.pattern_frame, tan_frame, &dr_re, &dr_im);
  const float q = p.root_intensity * p.tx.peak * p.rx.peak;
  float re = q * ((dt_re * p.rx.a_re - dt_im * p.rx.a_im) + (p.tx.a_re * dr_re - p.tx.a_im * dr_im));
  float im = q * ((dt_re * p.rx.a_im + dt_im * p.rx.a_re) + (p.tx.a_re * dr_im + p.tx.a_im * dr_re));
  const float d_peak_t = tan_tx_pattern != nullptr ? tan_tx_pattern[4] : 0.0f;
  const float d_peak_r = tan_rx_pattern != nullptr ? tan_rx_pattern[4] : 0.0f;
  const float dq = p.root_intensity * (d_peak_t * p.rx.peak + p.tx.peak * d_peak_r);
  re += dq * (p.tx.a_re * p.rx.a_re - p.tx.a_im * p.rx.a_im);
  im += dq * (p.tx.a_re * p.rx.a_im + p.tx.a_im * p.rx.a_re);
  *d_scale += re;
  *d_scale_im += im;
}

// The keyed contributions of one element side: up to eight table entries and
// its peak, at `key_base` (the side's offset in the key space). `g_re`, `g_im`
// is the cotangent of the side's amplitude A. Every entry is written: a knot
// the row does not reach carries weight zero, so the layout is fixed.
__device__ __forceinline__ void side_contributions(
    const ElementSide& side, const float* __restrict__ packed, float g_re, float g_im, float g_peak,
    int64_t key_base, int64_t* __restrict__ keys, float* __restrict__ values) {
  const Stencil& s = side.stencil;
  const bool field = s.quantity == kQuantityField;
  // The cotangent of the raw table value: F itself, or G = A^2 through dA/dG.
  const float r_re = field ? g_re : g_re * side.da_dg;
  const float r_im = field ? g_im : 0.0f;
  if (s.kind == kPatternMap) {
    for (int c = 0; c < 4; ++c) {
      keys[c] = key_base + s.base + s.cell[c];
      values[c] = s.weight[c] * r_re;
      keys[4 + c] = key_base + s.base + s.block + s.cell[c];
      values[4 + c] = field ? s.weight[c] * r_im : 0.0f;
    }
  } else {
    float xr, xi, yr, yi;
    separable_cuts(s, packed, &xr, &xi, &yr, &yi);
    // Raw = Fx Fy: the cotangent of Fx is conj(Fy) R, of Fy conj(Fx) R.
    const float gx_re = r_re * yr + r_im * yi;
    const float gx_im = r_im * yr - r_re * yi;
    const float gy_re = r_re * xr + r_im * xi;
    const float gy_im = r_im * xr - r_re * xi;
    const int64_t x_re = key_base + s.base;
    const int64_t y_re = x_re + s.num_x;
    const int64_t x_im = key_base + s.base + s.block;
    const int64_t y_im = x_im + s.num_x;
    keys[0] = x_re + s.cell[0];
    values[0] = s.weight[0] * gx_re;
    keys[1] = x_re + s.cell[1];
    values[1] = s.weight[1] * gx_re;
    keys[2] = y_re + s.cell[2];
    values[2] = s.weight[2] * gy_re;
    keys[3] = y_re + s.cell[3];
    values[3] = s.weight[3] * gy_re;
    keys[4] = x_im + s.cell[0];
    values[4] = field ? s.weight[0] * gx_im : 0.0f;
    keys[5] = x_im + s.cell[1];
    values[5] = field ? s.weight[1] * gx_im : 0.0f;
    keys[6] = y_im + s.cell[2];
    values[6] = field ? s.weight[2] * gy_im : 0.0f;
    keys[7] = y_im + s.cell[3];
    values[7] = field ? s.weight[3] * gy_im : 0.0f;
  }
  keys[8] = key_base + 4;
  values[8] = g_peak;
}

__global__ void sensor_weight_forward_kernel(
    RowInputs in,
    const float* __restrict__ weight_in_re,
    const float* __restrict__ weight_in_im,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    float* __restrict__ tau_rt,
    float* __restrict__ tau_rate,
    float* __restrict__ pattern_gain,
    int num_paths) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_paths) {
    return;
  }
  const RowTerm term = evaluate_row(in, k);
  if (term.complex_scale) {
    const float w_re = weight_in_re[k];
    const float w_im = weight_in_im[k];
    out_re[k] = w_re * term.scale - w_im * term.scale_im;
    out_im[k] = w_re * term.scale_im + w_im * term.scale;
  } else {
    out_re[k] = weight_in_re[k] * term.scale;
    out_im[k] = weight_in_im[k] * term.scale;
  }
  tau_rt[k] = term.tau_rt;
  tau_rate[k] = term.tau_rate;
  pattern_gain[k] = term.pattern_gain;
}

__global__ void sensor_weight_jvp_kernel(
    RowInputs in,
    const float* __restrict__ weight_in_re,
    const float* __restrict__ weight_in_im,
    const float* __restrict__ tan_tx_pos,
    const float* __restrict__ tan_rx_pos,
    const float* __restrict__ tan_site_in,
    const float* __restrict__ tan_site_out,
    const float* __restrict__ tan_intensity,
    const float* __restrict__ tan_weight_re,
    const float* __restrict__ tan_weight_im,
    float* __restrict__ tan_out_re,
    float* __restrict__ tan_out_im,
    float* __restrict__ tan_tau_rt,
    float* __restrict__ tan_tau_rate,
    const float* __restrict__ tan_tx_pattern,
    const float* __restrict__ tan_rx_pattern,
    const float* __restrict__ tan_frame,
    int num_paths) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_paths) {
    return;
  }
  const RowTerm term = evaluate_row(in, k);

  const int64_t r = k % in.rows_per_slot;
  const int64_t ti = strided_antenna(in.tx_index, k, in.rows_per_slot, in.tx_per_slot);
  const int64_t ri = strided_antenna(in.rx_index, k, in.rows_per_slot, in.rx_per_slot);

  const Vec3 d_tx = load3(tan_tx_pos, ti);
  const Vec3 d_rx = load3(tan_rx_pos, ri);
  const Vec3 d_site_in = load3(tan_site_in, k);
  const Vec3 d_site_out = load3(tan_site_out, k);

  float d_scale = dot3(term.dscale_dsite_in, d_site_in) +
      dot3(term.dscale_dsite_out, d_site_out) + dot3(term.dscale_dtx, d_tx) +
      dot3(term.dscale_drx, d_rx) + term.dscale_dintensity * tan_intensity[r];
  // The element parameters' tangent, only when one was supplied: a launch
  // without one is the position-only operator, unchanged.
  float d_scale_param_im = 0.0f;
  if (tan_tx_pattern != nullptr || tan_rx_pattern != nullptr || tan_frame != nullptr) {
    parameter_scale_tangent(in, k, tan_tx_pattern, tan_rx_pattern, tan_frame, &d_scale, &d_scale_param_im);
  }

  if (term.complex_scale) {
    const float d_scale_im = dot3(term.dscale_im_dsite_in, d_site_in) +
        dot3(term.dscale_im_dsite_out, d_site_out) + dot3(term.dscale_im_dtx, d_tx) +
        dot3(term.dscale_im_drx, d_rx) + term.dscale_im_dintensity * tan_intensity[r] + d_scale_param_im;
    const float w_re = weight_in_re[k];
    const float w_im = weight_in_im[k];
    const float t_re = tan_weight_re[k];
    const float t_im = tan_weight_im[k];
    tan_out_re[k] = t_re * term.scale - t_im * term.scale_im + w_re * d_scale - w_im * d_scale_im;
    tan_out_im[k] = t_re * term.scale_im + t_im * term.scale + w_re * d_scale_im + w_im * d_scale;
  } else {
    tan_out_re[k] = tan_weight_re[k] * term.scale + weight_in_re[k] * d_scale;
    tan_out_im[k] = tan_weight_im[k] * term.scale + weight_in_im[k] * d_scale;
  }
  tan_tau_rt[k] = dot3(term.dtau_rt_dsite_in, d_site_in) +
      dot3(term.dtau_rt_dsite_out, d_site_out) + dot3(term.dtau_rt_dtx, d_tx) +
      dot3(term.dtau_rt_drx, d_rx);
  tan_tau_rate[k] = dot3(term.dtau_rate_dsite_in, d_site_in) +
      dot3(term.dtau_rate_dsite_out, d_site_out) +
      dot3(term.dtau_rate_dtx, d_tx) + dot3(term.dtau_rate_drx, d_rx);
}

__global__ void sensor_weight_backward_rows_kernel(
    RowInputs in,
    const float* __restrict__ weight_in_re,
    const float* __restrict__ weight_in_im,
    const float* __restrict__ grad_out_re,
    const float* __restrict__ grad_out_im,
    const float* __restrict__ grad_tau_rt,
    const float* __restrict__ grad_tau_rate,
    float* __restrict__ grad_site_in,
    float* __restrict__ grad_site_out,
    float* __restrict__ grad_intensity,
    float* __restrict__ grad_weight_re,
    float* __restrict__ grad_weight_im,
    float* __restrict__ tx_row_scratch,
    float* __restrict__ rx_row_scratch,
    int num_paths) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_paths) {
    return;
  }
  const RowTerm term = evaluate_row(in, k);

  const float g_re = grad_out_re[k];
  const float g_im = grad_out_im[k];
  const float g_tau = grad_tau_rt[k];
  const float g_rate = grad_tau_rate[k];

  // The cotangent that reaches `scale` from both output components.
  const float g_scale = g_re * weight_in_re[k] + g_im * weight_in_im[k];
  Vec3 site_in_grad = add3(
      add3(
          scale3(term.dscale_dsite_in, g_scale),
          scale3(term.dtau_rt_dsite_in, g_tau)),
      scale3(term.dtau_rate_dsite_in, g_rate));
  Vec3 site_out_grad = add3(
      add3(
          scale3(term.dscale_dsite_out, g_scale),
          scale3(term.dtau_rt_dsite_out, g_tau)),
      scale3(term.dtau_rate_dsite_out, g_rate));
  Vec3 tx_grad = add3(
      add3(scale3(term.dscale_dtx, g_scale), scale3(term.dtau_rt_dtx, g_tau)),
      scale3(term.dtau_rate_dtx, g_rate));
  Vec3 rx_grad = add3(
      add3(scale3(term.dscale_drx, g_scale), scale3(term.dtau_rt_drx, g_tau)),
      scale3(term.dtau_rate_drx, g_rate));
  if (term.complex_scale) {
    // out = w s: dL/dw = g conj(s), and a real parameter p sees
    // Re(conj(g) w ds/dp) = ds_re g_scale + ds_im g_scale_im.
    const float g_scale_im = g_im * weight_in_re[k] - g_re * weight_in_im[k];
    grad_weight_re[k] = g_re * term.scale + g_im * term.scale_im;
    grad_weight_im[k] = g_im * term.scale - g_re * term.scale_im;
    grad_intensity[k] = g_scale * term.dscale_dintensity + g_scale_im * term.dscale_im_dintensity;
    site_in_grad = add3(site_in_grad, scale3(term.dscale_im_dsite_in, g_scale_im));
    site_out_grad = add3(site_out_grad, scale3(term.dscale_im_dsite_out, g_scale_im));
    tx_grad = add3(tx_grad, scale3(term.dscale_im_dtx, g_scale_im));
    rx_grad = add3(rx_grad, scale3(term.dscale_im_drx, g_scale_im));
  } else {
    grad_weight_re[k] = g_re * term.scale;
    grad_weight_im[k] = g_im * term.scale;
    grad_intensity[k] = g_scale * term.dscale_dintensity;
  }

  const int64_t base = static_cast<int64_t>(k) * 3;
  grad_site_in[base] = site_in_grad.x;
  grad_site_in[base + 1] = site_in_grad.y;
  grad_site_in[base + 2] = site_in_grad.z;
  grad_site_out[base] = site_out_grad.x;
  grad_site_out[base + 1] = site_out_grad.y;
  grad_site_out[base + 2] = site_out_grad.z;
  tx_row_scratch[base] = tx_grad.x;
  tx_row_scratch[base + 1] = tx_grad.y;
  tx_row_scratch[base + 2] = tx_grad.z;
  rx_row_scratch[base] = rx_grad.x;
  rx_row_scratch[base + 1] = rx_grad.y;
  rx_row_scratch[base + 2] = rx_grad.z;
}

// The antenna reduction. One thread per (antenna, component), walking the rows
// in ASCENDING order, so the summation order is a property of the frozen row
// set rather than of the schedule. This is why the row pass writes a scratch
// buffer instead of calling atomicAdd.
__global__ void sensor_weight_backward_antennas_kernel(
    const int64_t* __restrict__ tx_index,
    const int64_t* __restrict__ rx_index,
    const float* __restrict__ tx_row_scratch,
    const float* __restrict__ rx_row_scratch,
    float* __restrict__ grad_tx_pos,
    float* __restrict__ grad_rx_pos,
    int num_paths,
    int num_tx,
    int num_rx,
    int64_t rows_per_slot,
    int tx_per_slot,
    int rx_per_slot) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = (num_tx + num_rx) * 3;
  if (slot >= total) {
    return;
  }
  const bool is_tx = slot < num_tx * 3;
  const int local = is_tx ? slot : slot - num_tx * 3;
  const int antenna = local / 3;
  const int component = local % 3;
  const int64_t* index = is_tx ? tx_index : rx_index;
  const float* scratch = is_tx ? tx_row_scratch : rx_row_scratch;
  const int per_slot = is_tx ? tx_per_slot : rx_per_slot;

  double total_grad = 0.0;
  for (int k = 0; k < num_paths; ++k) {
    const int64_t a = strided_antenna(index, k, rows_per_slot, per_slot);
    if (static_cast<int>(a) != antenna) {
      continue;
    }
    total_grad += static_cast<double>(scratch[static_cast<int64_t>(k) * 3 + component]);
  }
  float* out = is_tx ? grad_tx_pos : grad_rx_pos;
  out[antenna * 3 + component] = static_cast<float>(total_grad);
}

// Per row: the keyed contributions to the element parameters, laid out as
// [tx side (9) | rx side (9)] when the tables are wanted and then the nine
// frame entries when the frame is wanted. Keys index [tx_pattern | rx_pattern
// | frame]; a knot outside the row's stencil carries the value zero.
__global__ void sensor_weight_parameter_rows_kernel(
    RowInputs in,
    const float* __restrict__ weight_in_re,
    const float* __restrict__ weight_in_im,
    const float* __restrict__ grad_out_re,
    const float* __restrict__ grad_out_im,
    int64_t* __restrict__ keys,
    float* __restrict__ values,
    int64_t tx_length,
    int64_t rx_length,
    int want_tables,
    int want_frame,
    int stride,
    int num_paths) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_paths) {
    return;
  }
  const RowParameters p = row_parameters(in, k);
  const float g_re = grad_out_re[k];
  const float g_im = grad_out_im[k];
  // Gs = conj-cotangent of the complex scale: out = w scale.
  const float gs_re = g_re * weight_in_re[k] + g_im * weight_in_im[k];
  const float gs_im = g_im * weight_in_re[k] - g_re * weight_in_im[k];
  const float q = p.root_intensity * p.tx.peak * p.rx.peak;
  // G_At = q conj(A_r) Gs and G_Ar = q conj(A_t) Gs.
  const float gt_re = q * (p.rx.a_re * gs_re + p.rx.a_im * gs_im);
  const float gt_im = q * (p.rx.a_re * gs_im - p.rx.a_im * gs_re);
  const float gr_re = q * (p.tx.a_re * gs_re + p.tx.a_im * gs_im);
  const float gr_im = q * (p.tx.a_re * gs_im - p.tx.a_im * gs_re);
  int64_t* row_keys = keys + static_cast<int64_t>(k) * stride;
  float* row_values = values + static_cast<int64_t>(k) * stride;
  int offset = 0;
  if (want_tables) {
    // Re(conj(Gs) A_t A_r): the cotangent of q / q, shared by both peaks.
    const float product_re = p.tx.a_re * p.rx.a_re - p.tx.a_im * p.rx.a_im;
    const float product_im = p.tx.a_re * p.rx.a_im + p.tx.a_im * p.rx.a_re;
    const float along = gs_re * product_re + gs_im * product_im;
    side_contributions(
        p.tx, in.tx_pattern, gt_re, gt_im, p.root_intensity * p.rx.peak * along, 0, row_keys, row_values);
    side_contributions(
        p.rx, in.rx_pattern, gr_re, gr_im, p.root_intensity * p.tx.peak * along, tx_length, row_keys + 9,
        row_values + 9);
    offset = 18;
  }
  if (want_frame) {
    const int64_t frame_base = tx_length + rx_length;
    for (int j = 0; j < 3; ++j) {
      const Vec3 axis = load3(in.pattern_frame, j);
      // dL/dw_j of each side, then dw_j/daxis_j = the side's direction.
      const float t_local = gt_re * dot3(p.tx.ga_re, axis) + gt_im * dot3(p.tx.ga_im, axis);
      const float r_local = gr_re * dot3(p.rx.ga_re, axis) + gr_im * dot3(p.rx.ga_im, axis);
      const Vec3 g = add3(scale3(p.tx.direction, t_local), scale3(p.rx.direction, r_local));
      row_keys[offset + 3 * j] = frame_base + 3 * j;
      row_keys[offset + 3 * j + 1] = frame_base + 3 * j + 1;
      row_keys[offset + 3 * j + 2] = frame_base + 3 * j + 2;
      row_values[offset + 3 * j] = g.x;
      row_values[offset + 3 * j + 1] = g.y;
      row_values[offset + 3 * j + 2] = g.z;
    }
  }
}

// One thread per parameter: the run of its key in the stable-sorted list,
// summed in ascending order in double.
__global__ void keyed_sum_kernel(
    const int64_t* __restrict__ sorted_keys,
    const float* __restrict__ sorted_values,
    int64_t count,
    float* __restrict__ grad,
    int64_t total) {
  const int64_t slot = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= total) {
    return;
  }
  int64_t low = 0;
  int64_t high = count;
  while (low < high) {
    const int64_t mid = (low + high) >> 1;
    if (sorted_keys[mid] < slot) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  double sum = 0.0;
  for (int64_t entry = low; entry < count && sorted_keys[entry] == slot; ++entry) {
    sum += static_cast<double>(sorted_values[entry]);
  }
  grad[slot] = static_cast<float>(sum);
}

void check_row_vector(
    const torch::stable::Tensor& tensor, int num_paths, const char* name) {
  check_cuda_float(tensor, name);
  STD_TORCH_CHECK(
      tensor.numel() == static_cast<int64_t>(num_paths) * 3,
      name,
      " must hold three components per path.");
}

void check_row_scalar(
    const torch::stable::Tensor& tensor, int num_paths, const char* name) {
  check_cuda_float(tensor, name);
  STD_TORCH_CHECK(
      tensor.numel() == static_cast<int64_t>(num_paths),
      name,
      " must hold one value per path.");
}

// The packed layout is validated against its own header ON THE HOST, from the
// buffer's length alone: the header lives on the device, so the host checks
// the one property it can (a non-empty CUDA float buffer) and the Python owner
// that packed it (sensors.Pattern.packed) is the layout's single writer.
void check_packed_pattern(const torch::stable::Tensor& tensor, const char* name) {
  check_cuda_float(tensor, name);
  STD_TORCH_CHECK(tensor.numel() >= kPatternHeader + 4, name, " must be a packed element pattern.");
}

// A parameter tangent is either empty (zero, skipped) or shaped like its primal.
const float* optional_tangent(
    const torch::stable::Tensor& tangent, const torch::stable::Tensor& primal, const char* name) {
  if (tangent.numel() == 0) {
    return nullptr;
  }
  check_cuda_float(tangent, name);
  STD_TORCH_CHECK(tangent.numel() == primal.numel(), name, " must match its primal or be empty.");
  return tangent.const_data_ptr<float>();
}

struct GeometryTensors {
  const torch::stable::Tensor& tx_pos;
  const torch::stable::Tensor& rx_pos;
  const torch::stable::Tensor& tx_velocity;
  const torch::stable::Tensor& rx_velocity;
  const torch::stable::Tensor& site_in;
  const torch::stable::Tensor& site_out;
  const torch::stable::Tensor& site_velocity;
  const torch::stable::Tensor& fixed_length_m;
  const torch::stable::Tensor& tx_index;
  const torch::stable::Tensor& rx_index;
  const torch::stable::Tensor& row_kind;
  const torch::stable::Tensor& intensity;
  const torch::stable::Tensor& pattern_frame;
  const torch::stable::Tensor& tx_pattern;
  const torch::stable::Tensor& rx_pattern;
};

RowInputs build_inputs(
    const GeometryTensors& t,
    int num_paths,
    int num_tx,
    int num_rx,
    double c0,
    int64_t slots) {
  STD_TORCH_CHECK(num_tx > 0, "num_tx must be positive.");
  STD_TORCH_CHECK(num_rx > 0, "num_rx must be positive.");
  STD_TORCH_CHECK(c0 > 0.0, "c0 must be positive.");
  STD_TORCH_CHECK(slots >= 1, "slots must be positive.");
  STD_TORCH_CHECK(
      num_paths % slots == 0 && num_tx % slots == 0 && num_rx % slots == 0,
      "a slot-major batch carries the same rows and antennas in every slot.");
  const int64_t slot_rows = num_paths / slots;
  const int tx_per_slot = static_cast<int>(num_tx / slots);
  const int rx_per_slot = static_cast<int>(num_rx / slots);

  check_cuda_float(t.tx_pos, "tx_pos");
  check_cuda_float(t.rx_pos, "rx_pos");
  check_cuda_float(t.tx_velocity, "tx_velocity");
  check_cuda_float(t.rx_velocity, "rx_velocity");
  STD_TORCH_CHECK(
      t.tx_pos.numel() == static_cast<int64_t>(num_tx) * 3 &&
          t.tx_velocity.numel() == static_cast<int64_t>(tx_per_slot) * 3,
      "tx_pos must hold three components per transmitter and tx_velocity per transmitter of one slot.");
  STD_TORCH_CHECK(
      t.rx_pos.numel() == static_cast<int64_t>(num_rx) * 3 &&
          t.rx_velocity.numel() == static_cast<int64_t>(rx_per_slot) * 3,
      "rx_pos must hold three components per receiver and rx_velocity per receiver of one slot.");

  const int one_slot = static_cast<int>(slot_rows);
  check_row_vector(t.site_in, num_paths, "site_in");
  check_row_vector(t.site_out, num_paths, "site_out");
  check_row_vector(t.site_velocity, one_slot, "site_velocity");
  check_row_scalar(t.fixed_length_m, one_slot, "fixed_length_m");
  check_row_scalar(t.intensity, one_slot, "intensity");
  check_cuda_long(t.tx_index, "tx_index");
  check_cuda_long(t.rx_index, "rx_index");
  check_cuda_int(t.row_kind, "row_kind");
  STD_TORCH_CHECK(
      t.tx_index.numel() == slot_rows && t.rx_index.numel() == slot_rows && t.row_kind.numel() == slot_rows,
      "tx_index, rx_index, and row_kind must hold one entry per path of one slot.");

  check_cuda_float(t.pattern_frame, "pattern_frame");
  STD_TORCH_CHECK(
      t.pattern_frame.numel() == 9,
      "pattern_frame must hold the three world-space pattern frame axes.");

  check_packed_pattern(t.tx_pattern, "tx_pattern");
  check_packed_pattern(t.rx_pattern, "rx_pattern");

  RowInputs in;
  in.tx_pos = t.tx_pos.const_data_ptr<float>();
  in.rx_pos = t.rx_pos.const_data_ptr<float>();
  in.tx_velocity = t.tx_velocity.const_data_ptr<float>();
  in.rx_velocity = t.rx_velocity.const_data_ptr<float>();
  in.site_in = t.site_in.const_data_ptr<float>();
  in.site_out = t.site_out.const_data_ptr<float>();
  in.site_velocity = t.site_velocity.const_data_ptr<float>();
  in.fixed_length_m = t.fixed_length_m.const_data_ptr<float>();
  in.tx_index = t.tx_index.const_data_ptr<int64_t>();
  in.rx_index = t.rx_index.const_data_ptr<int64_t>();
  in.row_kind = t.row_kind.const_data_ptr<int32_t>();
  in.intensity = t.intensity.const_data_ptr<float>();
  in.pattern_frame = t.pattern_frame.const_data_ptr<float>();
  in.tx_pattern = t.tx_pattern.const_data_ptr<float>();
  in.rx_pattern = t.rx_pattern.const_data_ptr<float>();
  in.num_tx = num_tx;
  in.num_rx = num_rx;
  in.c0 = static_cast<float>(c0);
  in.rows_per_slot = slot_rows > 0 ? slot_rows : 1;
  in.tx_per_slot = tx_per_slot;
  in.rx_per_slot = rx_per_slot;
  return in;
}

}  // namespace

void sensor_weight_forward_cuda(
    const torch::stable::Tensor& tx_pos,
    const torch::stable::Tensor& rx_pos,
    const torch::stable::Tensor& tx_velocity,
    const torch::stable::Tensor& rx_velocity,
    const torch::stable::Tensor& site_in,
    const torch::stable::Tensor& site_out,
    const torch::stable::Tensor& site_velocity,
    const torch::stable::Tensor& fixed_length_m,
    const torch::stable::Tensor& tx_index,
    const torch::stable::Tensor& rx_index,
    const torch::stable::Tensor& row_kind,
    const torch::stable::Tensor& intensity,
    const torch::stable::Tensor& weight_in_re,
    const torch::stable::Tensor& weight_in_im,
    const torch::stable::Tensor& pattern_frame,
    const torch::stable::Tensor& tx_pattern,
    const torch::stable::Tensor& rx_pattern,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    torch::stable::Tensor& tau_rt,
    torch::stable::Tensor& tau_rate,
    torch::stable::Tensor& pattern_gain,
    int64_t num_paths,
    int64_t num_tx,
    int64_t num_rx,
    double c0,
    int64_t slots) {
  const int paths = checked_int(num_paths, "num_paths");
  const GeometryTensors tensors{
      tx_pos,
      rx_pos,
      tx_velocity,
      rx_velocity,
      site_in,
      site_out,
      site_velocity,
      fixed_length_m,
      tx_index,
      rx_index,
      row_kind,
      intensity,
      pattern_frame,
      tx_pattern,
      rx_pattern};
  RowInputs in = build_inputs(
      tensors,
      paths,
      checked_int(num_tx, "num_tx"),
      checked_int(num_rx, "num_rx"),
      c0,
      slots);
  check_row_scalar(weight_in_re, paths, "weight_in_re");
  check_row_scalar(weight_in_im, paths, "weight_in_im");
  check_row_scalar(out_re, paths, "out_re");
  check_row_scalar(out_im, paths, "out_im");
  check_row_scalar(tau_rt, paths, "tau_rt");
  check_row_scalar(tau_rate, paths, "tau_rate");
  check_row_scalar(pattern_gain, paths, "pattern_gain");
  if (paths == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      out_re.get_device_index());
  constexpr int block_size = 256;
  sensor_weight_forward_kernel<<<
      dim3((paths + block_size - 1) / block_size, 1, 1),
      dim3(block_size, 1, 1),
      0,
      current_cuda_stream(out_re)>>>(
      in,
      weight_in_re.const_data_ptr<float>(),
      weight_in_im.const_data_ptr<float>(),
      out_re.mutable_data_ptr<float>(),
      out_im.mutable_data_ptr<float>(),
      tau_rt.mutable_data_ptr<float>(),
      tau_rate.mutable_data_ptr<float>(),
      pattern_gain.mutable_data_ptr<float>(),
      paths);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void sensor_weight_jvp_cuda(
    const torch::stable::Tensor& tx_pos,
    const torch::stable::Tensor& rx_pos,
    const torch::stable::Tensor& tx_velocity,
    const torch::stable::Tensor& rx_velocity,
    const torch::stable::Tensor& site_in,
    const torch::stable::Tensor& site_out,
    const torch::stable::Tensor& site_velocity,
    const torch::stable::Tensor& fixed_length_m,
    const torch::stable::Tensor& tx_index,
    const torch::stable::Tensor& rx_index,
    const torch::stable::Tensor& row_kind,
    const torch::stable::Tensor& intensity,
    const torch::stable::Tensor& weight_in_re,
    const torch::stable::Tensor& weight_in_im,
    const torch::stable::Tensor& pattern_frame,
    const torch::stable::Tensor& tx_pattern,
    const torch::stable::Tensor& rx_pattern,
    const torch::stable::Tensor& tan_tx_pos,
    const torch::stable::Tensor& tan_rx_pos,
    const torch::stable::Tensor& tan_site_in,
    const torch::stable::Tensor& tan_site_out,
    const torch::stable::Tensor& tan_intensity,
    const torch::stable::Tensor& tan_weight_re,
    const torch::stable::Tensor& tan_weight_im,
    const torch::stable::Tensor& tan_tx_pattern,
    const torch::stable::Tensor& tan_rx_pattern,
    const torch::stable::Tensor& tan_pattern_frame,
    torch::stable::Tensor& tan_out_re,
    torch::stable::Tensor& tan_out_im,
    torch::stable::Tensor& tan_tau_rt,
    torch::stable::Tensor& tan_tau_rate,
    int64_t num_paths,
    int64_t num_tx,
    int64_t num_rx,
    double c0,
    int64_t slots) {
  const int paths = checked_int(num_paths, "num_paths");
  const int transmitters = checked_int(num_tx, "num_tx");
  const int receivers = checked_int(num_rx, "num_rx");
  const GeometryTensors tensors{
      tx_pos,
      rx_pos,
      tx_velocity,
      rx_velocity,
      site_in,
      site_out,
      site_velocity,
      fixed_length_m,
      tx_index,
      rx_index,
      row_kind,
      intensity,
      pattern_frame,
      tx_pattern,
      rx_pattern};
  RowInputs in = build_inputs(
      tensors,
      paths,
      transmitters,
      receivers,
      c0,
      slots);
  check_row_scalar(weight_in_re, paths, "weight_in_re");
  check_row_scalar(weight_in_im, paths, "weight_in_im");
  check_cuda_float(tan_tx_pos, "tan_tx_pos");
  check_cuda_float(tan_rx_pos, "tan_rx_pos");
  STD_TORCH_CHECK(
      tan_tx_pos.numel() == static_cast<int64_t>(transmitters) * 3,
      "tan_tx_pos must match tx_pos.");
  STD_TORCH_CHECK(
      tan_rx_pos.numel() == static_cast<int64_t>(receivers) * 3,
      "tan_rx_pos must match rx_pos.");
  check_row_vector(tan_site_in, paths, "tan_site_in");
  check_row_vector(tan_site_out, paths, "tan_site_out");
  check_row_scalar(tan_intensity, static_cast<int>(in.rows_per_slot), "tan_intensity");
  check_row_scalar(tan_weight_re, paths, "tan_weight_re");
  check_row_scalar(tan_weight_im, paths, "tan_weight_im");
  check_row_scalar(tan_out_re, paths, "tan_out_re");
  check_row_scalar(tan_out_im, paths, "tan_out_im");
  check_row_scalar(tan_tau_rt, paths, "tan_tau_rt");
  check_row_scalar(tan_tau_rate, paths, "tan_tau_rate");
  // An empty parameter tangent is a zero one and skips that work entirely.
  const float* tan_tx_table = optional_tangent(tan_tx_pattern, tx_pattern, "tan_tx_pattern");
  const float* tan_rx_table = optional_tangent(tan_rx_pattern, rx_pattern, "tan_rx_pattern");
  const float* tan_frame = optional_tangent(tan_pattern_frame, pattern_frame, "tan_pattern_frame");
  if (paths == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      tan_out_re.get_device_index());
  constexpr int block_size = 256;
  sensor_weight_jvp_kernel<<<
      dim3((paths + block_size - 1) / block_size, 1, 1),
      dim3(block_size, 1, 1),
      0,
      current_cuda_stream(tan_out_re)>>>(
      in,
      weight_in_re.const_data_ptr<float>(),
      weight_in_im.const_data_ptr<float>(),
      tan_tx_pos.const_data_ptr<float>(),
      tan_rx_pos.const_data_ptr<float>(),
      tan_site_in.const_data_ptr<float>(),
      tan_site_out.const_data_ptr<float>(),
      tan_intensity.const_data_ptr<float>(),
      tan_weight_re.const_data_ptr<float>(),
      tan_weight_im.const_data_ptr<float>(),
      tan_out_re.mutable_data_ptr<float>(),
      tan_out_im.mutable_data_ptr<float>(),
      tan_tau_rt.mutable_data_ptr<float>(),
      tan_tau_rate.mutable_data_ptr<float>(),
      tan_tx_table,
      tan_rx_table,
      tan_frame,
      paths);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void sensor_weight_backward_cuda(
    const torch::stable::Tensor& tx_pos,
    const torch::stable::Tensor& rx_pos,
    const torch::stable::Tensor& tx_velocity,
    const torch::stable::Tensor& rx_velocity,
    const torch::stable::Tensor& site_in,
    const torch::stable::Tensor& site_out,
    const torch::stable::Tensor& site_velocity,
    const torch::stable::Tensor& fixed_length_m,
    const torch::stable::Tensor& tx_index,
    const torch::stable::Tensor& rx_index,
    const torch::stable::Tensor& row_kind,
    const torch::stable::Tensor& intensity,
    const torch::stable::Tensor& weight_in_re,
    const torch::stable::Tensor& weight_in_im,
    const torch::stable::Tensor& pattern_frame,
    const torch::stable::Tensor& tx_pattern,
    const torch::stable::Tensor& rx_pattern,
    const torch::stable::Tensor& grad_out_re,
    const torch::stable::Tensor& grad_out_im,
    const torch::stable::Tensor& grad_tau_rt,
    const torch::stable::Tensor& grad_tau_rate,
    torch::stable::Tensor& grad_tx_pos,
    torch::stable::Tensor& grad_rx_pos,
    torch::stable::Tensor& grad_site_in,
    torch::stable::Tensor& grad_site_out,
    torch::stable::Tensor& grad_intensity,
    torch::stable::Tensor& grad_weight_re,
    torch::stable::Tensor& grad_weight_im,
    torch::stable::Tensor& tx_row_scratch,
    torch::stable::Tensor& rx_row_scratch,
    torch::stable::Tensor& parameter_keys,
    torch::stable::Tensor& parameter_values,
    int64_t num_paths,
    int64_t num_tx,
    int64_t num_rx,
    double c0,
    int64_t slots,
    int64_t want_tables,
    int64_t want_frame) {
  const int paths = checked_int(num_paths, "num_paths");
  const int transmitters = checked_int(num_tx, "num_tx");
  const int receivers = checked_int(num_rx, "num_rx");
  const GeometryTensors tensors{
      tx_pos,
      rx_pos,
      tx_velocity,
      rx_velocity,
      site_in,
      site_out,
      site_velocity,
      fixed_length_m,
      tx_index,
      rx_index,
      row_kind,
      intensity,
      pattern_frame,
      tx_pattern,
      rx_pattern};
  RowInputs in = build_inputs(
      tensors,
      paths,
      transmitters,
      receivers,
      c0,
      slots);
  check_row_scalar(weight_in_re, paths, "weight_in_re");
  check_row_scalar(weight_in_im, paths, "weight_in_im");
  check_row_scalar(grad_out_re, paths, "grad_out_re");
  check_row_scalar(grad_out_im, paths, "grad_out_im");
  check_row_scalar(grad_tau_rt, paths, "grad_tau_rt");
  check_row_scalar(grad_tau_rate, paths, "grad_tau_rate");
  check_row_vector(grad_site_in, paths, "grad_site_in");
  check_row_vector(grad_site_out, paths, "grad_site_out");
  check_row_scalar(grad_intensity, paths, "grad_intensity");
  check_row_scalar(grad_weight_re, paths, "grad_weight_re");
  check_row_scalar(grad_weight_im, paths, "grad_weight_im");
  check_row_vector(tx_row_scratch, paths, "tx_row_scratch");
  check_row_vector(rx_row_scratch, paths, "rx_row_scratch");
  check_cuda_float(grad_tx_pos, "grad_tx_pos");
  check_cuda_float(grad_rx_pos, "grad_rx_pos");
  STD_TORCH_CHECK(
      grad_tx_pos.numel() == static_cast<int64_t>(transmitters) * 3,
      "grad_tx_pos must match tx_pos.");
  STD_TORCH_CHECK(
      grad_rx_pos.numel() == static_cast<int64_t>(receivers) * 3,
      "grad_rx_pos must match rx_pos.");
  const int parameter_stride = (want_tables != 0 ? 18 : 0) + (want_frame != 0 ? 9 : 0);
  check_cuda_long(parameter_keys, "parameter_keys");
  check_cuda_float(parameter_values, "parameter_values");
  STD_TORCH_CHECK(
      parameter_keys.numel() == static_cast<int64_t>(paths) * parameter_stride &&
          parameter_values.numel() == parameter_keys.numel(),
      "parameter_keys and parameter_values must hold one entry per parameter contribution of every path.");

  const torch::stable::accelerator::DeviceGuard device_guard(
      grad_tx_pos.get_device_index());
  const cudaStream_t stream = current_cuda_stream(grad_tx_pos);
  constexpr int block_size = 256;
  if (paths > 0) {
    sensor_weight_backward_rows_kernel<<<
        dim3((paths + block_size - 1) / block_size, 1, 1),
        dim3(block_size, 1, 1),
        0,
        stream>>>(
        in,
        weight_in_re.const_data_ptr<float>(),
        weight_in_im.const_data_ptr<float>(),
        grad_out_re.const_data_ptr<float>(),
        grad_out_im.const_data_ptr<float>(),
        grad_tau_rt.const_data_ptr<float>(),
        grad_tau_rate.const_data_ptr<float>(),
        grad_site_in.mutable_data_ptr<float>(),
        grad_site_out.mutable_data_ptr<float>(),
        grad_intensity.mutable_data_ptr<float>(),
        grad_weight_re.mutable_data_ptr<float>(),
        grad_weight_im.mutable_data_ptr<float>(),
        tx_row_scratch.mutable_data_ptr<float>(),
        rx_row_scratch.mutable_data_ptr<float>(),
        paths);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (paths > 0 && parameter_stride > 0) {
    sensor_weight_parameter_rows_kernel<<<
        dim3((paths + block_size - 1) / block_size, 1, 1),
        dim3(block_size, 1, 1),
        0,
        stream>>>(
        in,
        weight_in_re.const_data_ptr<float>(),
        weight_in_im.const_data_ptr<float>(),
        grad_out_re.const_data_ptr<float>(),
        grad_out_im.const_data_ptr<float>(),
        parameter_keys.mutable_data_ptr<int64_t>(),
        parameter_values.mutable_data_ptr<float>(),
        tx_pattern.numel(),
        rx_pattern.numel(),
        want_tables != 0 ? 1 : 0,
        want_frame != 0 ? 1 : 0,
        parameter_stride,
        paths);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int antenna_slots = (transmitters + receivers) * 3;
  sensor_weight_backward_antennas_kernel<<<
      dim3((antenna_slots + block_size - 1) / block_size, 1, 1),
      dim3(block_size, 1, 1),
      0,
      stream>>>(
      tx_index.const_data_ptr<int64_t>(),
      rx_index.const_data_ptr<int64_t>(),
      tx_row_scratch.const_data_ptr<float>(),
      rx_row_scratch.const_data_ptr<float>(),
      grad_tx_pos.mutable_data_ptr<float>(),
      grad_rx_pos.mutable_data_ptr<float>(),
      paths,
      transmitters,
      receivers,
      in.rows_per_slot,
      in.tx_per_slot,
      in.rx_per_slot);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void keyed_sum_forward_cuda(
    const torch::stable::Tensor& sorted_keys,
    const torch::stable::Tensor& sorted_values,
    torch::stable::Tensor& total_out) {
  check_cuda_long(sorted_keys, "sorted_keys");
  check_cuda_float(sorted_values, "sorted_values");
  check_cuda_float(total_out, "total");
  STD_TORCH_CHECK(sorted_values.numel() == sorted_keys.numel(), "sorted_values must match sorted_keys.");
  const int64_t total = total_out.numel();
  if (total == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(total_out.get_device_index());
  constexpr int block_size = 256;
  keyed_sum_kernel<<<
      dim3(static_cast<unsigned>((total + block_size - 1) / block_size), 1, 1),
      dim3(block_size, 1, 1),
      0,
      current_cuda_stream(total_out)>>>(
      sorted_keys.const_data_ptr<int64_t>(),
      sorted_values.const_data_ptr<float>(),
      sorted_keys.numel(),
      total_out.mutable_data_ptr<float>(),
      total);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("keyed_sum_forward", TORCH_BOX(&keyed_sum_forward_cuda));
  m.impl("sensor_weight_forward", TORCH_BOX(&sensor_weight_forward_cuda));
  m.impl("sensor_weight_backward", TORCH_BOX(&sensor_weight_backward_cuda));
  m.impl("sensor_weight_jvp", TORCH_BOX(&sensor_weight_jvp_cuda));
}
