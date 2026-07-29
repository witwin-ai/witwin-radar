// Sensor pattern weighting and geometric delay for Channel-composed paths.
//
// Channel exclusively owns propagation spreading, transmit power, reference
// phase, and endpoint polarization projection. This operator applies only the
// transmit/receive antenna power patterns and sqrt(intensity), while publishing
// geometric delay and delay rate for diagnostics and differentiation.
//
// Per row k:
//   tau_rt = L / c0
//   pattern_gain = G_t * G_r
//   scale = sqrt(max(intensity, 0)) * sqrt(max(G_t * G_r, 0))
//   out = weight_in * scale
//
// Pattern lookup uses a fixed world-to-pattern frame. Endpoint positions,
// intensity, and complex input weight are differentiable; velocities, row
// topology, the pattern frame, and lookup tables are frozen constants.
//// THE ANTENNA PATTERN IS INTERPOLATED, NOT TABULATED IN THE PHYSICS. The table
// is a constant; the DIRECTION is differentiable, and the interpolation is
// piecewise linear in the two angles, so the gain has an exact almost-everywhere
// derivative that this kernel carries:
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
// Differentiability. The differentiable inputs are the four endpoint positions
// (tx_pos, rx_pos, site_in, site_out), the intensity, and the incoming complex
// weight. The velocities, pattern frame, and lookup tables are constants. `tau_rate` depends on the
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

#include <cstdint>
#include <limits>

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

// Exactly `utils/antenna.interp1d_zero_outside`, including its two edge
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

// Everything one row needs, primal and partial derivatives together, so that
// the three operators cannot drift apart: forward reads the values, jvp
// contracts the partials with the input tangents, and backward contracts them
// with the output cotangents. There is exactly one expression for each
// quantity in this file.
struct RowTerm {
  float tau_rt;
  float tau_rate;
  float pattern_gain;  // G_t * G_r, the published power product
  float scale;         // the real signed factor the weight is multiplied by

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
  const float* pattern_x_axis;
  const float* pattern_y_axis;
  const float* pattern_x_values;
  const float* pattern_y_values;
  const float* pattern_values;
  int num_tx;
  int num_rx;
  int pattern_kind;
  int num_x;
  int num_y;
  float c0;
};

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

  int64_t ti = in.tx_index[k];
  ti = ti < 0 ? 0 : (ti >= in.num_tx ? in.num_tx - 1 : ti);
  int64_t ri = in.rx_index[k];
  ri = ri < 0 ? 0 : (ri >= in.num_rx ? in.num_rx - 1 : ri);
  const int kind = in.row_kind[k];

  const Vec3 tx = load3(in.tx_pos, ti);
  const Vec3 rx = load3(in.rx_pos, ri);
  const Vec3 v_tx = load3(in.tx_velocity, ti);
  const Vec3 v_rx = load3(in.rx_velocity, ri);

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
    // The LENGTH uses the raw norm and the DIRECTION uses the clamped one,
    // exactly as `compute_total_path_lengths` and `_total_path_length_rates`
    // split it today: the length is a distance and the clamp exists only to
    // keep a unit vector finite at a coincident pair.
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
    const Vec3 v_site = load3(in.site_velocity, k);
    const Vec3 d_in = sub3(site_in, tx);
    const Vec3 d_out = sub3(rx, site_out);
    const float raw_a = norm3(d_in);
    const float raw_b = norm3(d_out);
    const float a = raw_a < kMinDistance ? kMinDistance : raw_a;
    const float b = raw_b < kMinDistance ? kMinDistance : raw_b;
    const Vec3 u_in = scale3(d_in, 1.0f / a);
    const Vec3 u_out = scale3(d_out, 1.0f / b);

    length = raw_a + in.fixed_length_m[k] + raw_b;
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

  const DirectionalGain gain_tx = evaluate_pattern_vector(
      in.pattern_kind,
      in.pattern_x_axis,
      in.pattern_y_axis,
      in.pattern_x_values,
      in.pattern_y_values,
      in.pattern_values,
      in.num_x,
      in.num_y,
      axis_right,
      axis_up,
      axis_back,
      direction_tx);
  const DirectionalGain gain_rx = evaluate_pattern_vector(
      in.pattern_kind,
      in.pattern_x_axis,
      in.pattern_y_axis,
      in.pattern_x_values,
      in.pattern_y_values,
      in.pattern_values,
      in.num_x,
      in.num_y,
      axis_right,
      axis_up,
      axis_back,
      direction_rx);

  const float gain_product = gain_tx.value * gain_rx.value;
  term.pattern_gain = gain_product;
  const float clamped_gain = gain_product > 0.0f ? gain_product : 0.0f;
  const float gain_amplitude = sqrtf(clamped_gain);

  const float raw_intensity = in.intensity[k];
  const float clamped_intensity = raw_intensity > 0.0f ? raw_intensity : 0.0f;
  const float intensity_amplitude = sqrtf(clamped_intensity);

  const float scale = intensity_amplitude * gain_amplitude;
  term.scale = scale;
  term.tau_rt = length / in.c0;
  term.tau_rate = rate / in.c0;

  // d(scale)/d(intensity): the derivative of the square root, taken to be zero
  // at and below the clamp, which is the almost-everywhere derivative of
  // sqrt(clamp(x, 0)).
  if (raw_intensity > 0.0f) {
    term.dscale_dintensity =
        0.5f / intensity_amplitude * gain_amplitude;
  }

  // d(scale)/d(direction), through sqrt(clamp(G_t G_r, 0)).
  if (gain_product > 0.0f) {
    const float common =
        0.5f / gain_amplitude * intensity_amplitude;
    const Vec3 dscale_dtx_dir = scale3(gain_tx.gradient, common * gain_rx.value);
    const Vec3 dscale_drx_dir = scale3(gain_rx.gradient, common * gain_tx.value);
    *dtx_dir_head = add3(*dtx_dir_head, dscale_dtx_dir);
    *dtx_dir_tail = sub3(*dtx_dir_tail, dscale_dtx_dir);
    *drx_dir_head = add3(*drx_dir_head, dscale_drx_dir);
    *drx_dir_tail = sub3(*drx_dir_tail, dscale_drx_dir);
  }

  const float dscale_dlength = 0.0f;

  term.dscale_dsite_in =
      add3(grad_site_in_pattern, scale3(dlen_dsite_in, dscale_dlength));
  term.dscale_dsite_out =
      add3(grad_site_out_pattern, scale3(dlen_dsite_out, dscale_dlength));
  term.dscale_dtx = add3(grad_tx_pattern, scale3(dlen_dtx, dscale_dlength));
  term.dscale_drx = add3(grad_rx_pattern, scale3(dlen_drx, dscale_dlength));

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
  out_re[k] = weight_in_re[k] * term.scale;
  out_im[k] = weight_in_im[k] * term.scale;
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
    int num_paths) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_paths) {
    return;
  }
  const RowTerm term = evaluate_row(in, k);

  int64_t ti = in.tx_index[k];
  ti = ti < 0 ? 0 : (ti >= in.num_tx ? in.num_tx - 1 : ti);
  int64_t ri = in.rx_index[k];
  ri = ri < 0 ? 0 : (ri >= in.num_rx ? in.num_rx - 1 : ri);

  const Vec3 d_tx = load3(tan_tx_pos, ti);
  const Vec3 d_rx = load3(tan_rx_pos, ri);
  const Vec3 d_site_in = load3(tan_site_in, k);
  const Vec3 d_site_out = load3(tan_site_out, k);

  const float d_scale = dot3(term.dscale_dsite_in, d_site_in) +
      dot3(term.dscale_dsite_out, d_site_out) + dot3(term.dscale_dtx, d_tx) +
      dot3(term.dscale_drx, d_rx) + term.dscale_dintensity * tan_intensity[k];

  tan_out_re[k] = tan_weight_re[k] * term.scale + weight_in_re[k] * d_scale;
  tan_out_im[k] = tan_weight_im[k] * term.scale + weight_in_im[k] * d_scale;
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

  grad_weight_re[k] = g_re * term.scale;
  grad_weight_im[k] = g_im * term.scale;

  // The cotangent that reaches `scale` from both output components.
  const float g_scale = g_re * weight_in_re[k] + g_im * weight_in_im[k];
  grad_intensity[k] = g_scale * term.dscale_dintensity;

  const Vec3 site_in_grad = add3(
      add3(
          scale3(term.dscale_dsite_in, g_scale),
          scale3(term.dtau_rt_dsite_in, g_tau)),
      scale3(term.dtau_rate_dsite_in, g_rate));
  const Vec3 site_out_grad = add3(
      add3(
          scale3(term.dscale_dsite_out, g_scale),
          scale3(term.dtau_rt_dsite_out, g_tau)),
      scale3(term.dtau_rate_dsite_out, g_rate));
  const Vec3 tx_grad = add3(
      add3(scale3(term.dscale_dtx, g_scale), scale3(term.dtau_rt_dtx, g_tau)),
      scale3(term.dtau_rate_dtx, g_rate));
  const Vec3 rx_grad = add3(
      add3(scale3(term.dscale_drx, g_scale), scale3(term.dtau_rt_drx, g_tau)),
      scale3(term.dtau_rate_drx, g_rate));

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
    int num_rx) {
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
  const int bound = is_tx ? num_tx : num_rx;

  double total_grad = 0.0;
  for (int k = 0; k < num_paths; ++k) {
    int64_t a = index[k];
    a = a < 0 ? 0 : (a >= bound ? bound - 1 : a);
    if (static_cast<int>(a) != antenna) {
      continue;
    }
    total_grad += static_cast<double>(scratch[static_cast<int64_t>(k) * 3 + component]);
  }
  float* out = is_tx ? grad_tx_pos : grad_rx_pos;
  out[antenna * 3 + component] = static_cast<float>(total_grad);
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

void check_cuda_int(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Int,
      name,
      " must have dtype torch.int32.");
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
  const torch::stable::Tensor& pattern_x_axis;
  const torch::stable::Tensor& pattern_y_axis;
  const torch::stable::Tensor& pattern_x_values;
  const torch::stable::Tensor& pattern_y_values;
  const torch::stable::Tensor& pattern_values;
};

RowInputs build_inputs(
    const GeometryTensors& t,
    int num_paths,
    int num_tx,
    int num_rx,
    int pattern_kind,
    double c0) {
  STD_TORCH_CHECK(num_tx > 0, "num_tx must be positive.");
  STD_TORCH_CHECK(num_rx > 0, "num_rx must be positive.");
  STD_TORCH_CHECK(c0 > 0.0, "c0 must be positive.");
  STD_TORCH_CHECK(
      pattern_kind == kPatternSeparable || pattern_kind == kPatternMap,
      "pattern_kind must be 0 (separable) or 1 (map).");

  check_cuda_float(t.tx_pos, "tx_pos");
  check_cuda_float(t.rx_pos, "rx_pos");
  check_cuda_float(t.tx_velocity, "tx_velocity");
  check_cuda_float(t.rx_velocity, "rx_velocity");
  STD_TORCH_CHECK(
      t.tx_pos.numel() == static_cast<int64_t>(num_tx) * 3 &&
          t.tx_velocity.numel() == static_cast<int64_t>(num_tx) * 3,
      "tx_pos and tx_velocity must hold three components per transmitter.");
  STD_TORCH_CHECK(
      t.rx_pos.numel() == static_cast<int64_t>(num_rx) * 3 &&
          t.rx_velocity.numel() == static_cast<int64_t>(num_rx) * 3,
      "rx_pos and rx_velocity must hold three components per receiver.");

  check_row_vector(t.site_in, num_paths, "site_in");
  check_row_vector(t.site_out, num_paths, "site_out");
  check_row_vector(t.site_velocity, num_paths, "site_velocity");
  check_row_scalar(t.fixed_length_m, num_paths, "fixed_length_m");
  check_row_scalar(t.intensity, num_paths, "intensity");
  check_cuda_long(t.tx_index, "tx_index");
  check_cuda_long(t.rx_index, "rx_index");
  check_cuda_int(t.row_kind, "row_kind");
  STD_TORCH_CHECK(
      t.tx_index.numel() == static_cast<int64_t>(num_paths) &&
          t.rx_index.numel() == static_cast<int64_t>(num_paths) &&
          t.row_kind.numel() == static_cast<int64_t>(num_paths),
      "tx_index, rx_index, and row_kind must hold one entry per path.");

  check_cuda_float(t.pattern_frame, "pattern_frame");
  STD_TORCH_CHECK(
      t.pattern_frame.numel() == 9,
      "pattern_frame must hold the three world-space pattern frame axes.");

  check_cuda_float(t.pattern_x_axis, "pattern_x_axis");
  check_cuda_float(t.pattern_y_axis, "pattern_y_axis");
  check_cuda_float(t.pattern_x_values, "pattern_x_values");
  check_cuda_float(t.pattern_y_values, "pattern_y_values");
  check_cuda_float(t.pattern_values, "pattern_values");
  const int num_x = checked_int(t.pattern_x_axis.numel(), "pattern_x_axis");
  const int num_y = checked_int(t.pattern_y_axis.numel(), "pattern_y_axis");
  STD_TORCH_CHECK(num_x >= 2 && num_y >= 2, "the pattern axes need two samples.");
  if (pattern_kind == kPatternSeparable) {
    STD_TORCH_CHECK(
        t.pattern_x_values.numel() == num_x && t.pattern_y_values.numel() == num_y,
        "a separable pattern needs one value per axis sample.");
  } else {
    STD_TORCH_CHECK(
        t.pattern_values.numel() == static_cast<int64_t>(num_x) * num_y,
        "a map pattern needs one value per (y, x) grid cell.");
  }

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
  in.pattern_x_axis = t.pattern_x_axis.const_data_ptr<float>();
  in.pattern_y_axis = t.pattern_y_axis.const_data_ptr<float>();
  in.pattern_x_values = t.pattern_x_values.const_data_ptr<float>();
  in.pattern_y_values = t.pattern_y_values.const_data_ptr<float>();
  in.pattern_values = t.pattern_values.const_data_ptr<float>();
  in.num_tx = num_tx;
  in.num_rx = num_rx;
  in.pattern_kind = pattern_kind;
  in.num_x = num_x;
  in.num_y = num_y;
  in.c0 = static_cast<float>(c0);
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
    const torch::stable::Tensor& pattern_x_axis,
    const torch::stable::Tensor& pattern_y_axis,
    const torch::stable::Tensor& pattern_x_values,
    const torch::stable::Tensor& pattern_y_values,
    const torch::stable::Tensor& pattern_values,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    torch::stable::Tensor& tau_rt,
    torch::stable::Tensor& tau_rate,
    torch::stable::Tensor& pattern_gain,
    int64_t num_paths,
    int64_t num_tx,
    int64_t num_rx,
    int64_t pattern_kind,
    double c0) {
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
      pattern_x_axis,
      pattern_y_axis,
      pattern_x_values,
      pattern_y_values,
      pattern_values};
  RowInputs in = build_inputs(
      tensors,
      paths,
      checked_int(num_tx, "num_tx"),
      checked_int(num_rx, "num_rx"),
      checked_int(pattern_kind, "pattern_kind"),
      c0);
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
    const torch::stable::Tensor& pattern_x_axis,
    const torch::stable::Tensor& pattern_y_axis,
    const torch::stable::Tensor& pattern_x_values,
    const torch::stable::Tensor& pattern_y_values,
    const torch::stable::Tensor& pattern_values,
    const torch::stable::Tensor& tan_tx_pos,
    const torch::stable::Tensor& tan_rx_pos,
    const torch::stable::Tensor& tan_site_in,
    const torch::stable::Tensor& tan_site_out,
    const torch::stable::Tensor& tan_intensity,
    const torch::stable::Tensor& tan_weight_re,
    const torch::stable::Tensor& tan_weight_im,
    torch::stable::Tensor& tan_out_re,
    torch::stable::Tensor& tan_out_im,
    torch::stable::Tensor& tan_tau_rt,
    torch::stable::Tensor& tan_tau_rate,
    int64_t num_paths,
    int64_t num_tx,
    int64_t num_rx,
    int64_t pattern_kind,
    double c0) {
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
      pattern_x_axis,
      pattern_y_axis,
      pattern_x_values,
      pattern_y_values,
      pattern_values};
  RowInputs in = build_inputs(
      tensors,
      paths,
      transmitters,
      receivers,
      checked_int(pattern_kind, "pattern_kind"),
      c0);
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
  check_row_scalar(tan_intensity, paths, "tan_intensity");
  check_row_scalar(tan_weight_re, paths, "tan_weight_re");
  check_row_scalar(tan_weight_im, paths, "tan_weight_im");
  check_row_scalar(tan_out_re, paths, "tan_out_re");
  check_row_scalar(tan_out_im, paths, "tan_out_im");
  check_row_scalar(tan_tau_rt, paths, "tan_tau_rt");
  check_row_scalar(tan_tau_rate, paths, "tan_tau_rate");
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
    const torch::stable::Tensor& pattern_x_axis,
    const torch::stable::Tensor& pattern_y_axis,
    const torch::stable::Tensor& pattern_x_values,
    const torch::stable::Tensor& pattern_y_values,
    const torch::stable::Tensor& pattern_values,
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
    int64_t num_paths,
    int64_t num_tx,
    int64_t num_rx,
    int64_t pattern_kind,
    double c0) {
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
      pattern_x_axis,
      pattern_y_axis,
      pattern_x_values,
      pattern_y_values,
      pattern_values};
  RowInputs in = build_inputs(
      tensors,
      paths,
      transmitters,
      receivers,
      checked_int(pattern_kind, "pattern_kind"),
      c0);
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
      receivers);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("sensor_weight_forward", TORCH_BOX(&sensor_weight_forward_cuda));
  m.impl("sensor_weight_backward", TORCH_BOX(&sensor_weight_backward_cuda));
  m.impl("sensor_weight_jvp", TORCH_BOX(&sensor_weight_jvp_cuda));
}
