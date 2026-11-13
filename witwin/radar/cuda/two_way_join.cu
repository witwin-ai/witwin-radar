// The two-way join: inbound leg x outbound leg -> radar round trip.
//
// Both legs run FROM a radar element TOWARD the site (R-ADR-039) and each
// publishes Channel's complex3_transport: the field vector AT THE SITE,
// unprojected, in world Cartesian components, in Channel's phasor convention
// exp(-j k d) at the reference frequency. Per composed row k, with
// i = idx_in[k], o = idx_out[k], s = idx_s[k]:
//
//   E = field_in[i]     incident field at the site, launched by the transmitter
//   H = field_out[o]    field at the site launched by the receiver; by
//                       reciprocity H . X is what the receiver reads of a
//                       field X re-radiated by the site
//   V0 = H^T M_s E      M_s the site's 3x3 scattering dyadic (below)
//   V  = a_t a_r V0 exp(j psi)
//   tau = L64 / c0
//
// THE SCATTERING DYADIC (R-ADR-041). A response of width 1 is a scalar s and
// M = s I: the site re-radiates s times the incident field, which the receiver
// reads through H's transverse field, so V0 = s (H . E). A response of width 4
// is the 2x2 matrix [HH, HV, VH, VV] (receive index first) in the site's local
// backscatter-alignment basis: for a propagation direction k into the site and
// the site's up axis u, h = (u x k)/|u x k| and v = h x k. The incident basis
// uses k_i (the inbound leg's final-segment direction) and the receive basis
// uses k_r (the outbound leg's), so
//   V0 = sum_ab S_ab (H . b_r,a) (b_i,b . E)
// and a monostatic S = I equals the width-1 s = 1. Where u is parallel to k the
// basis falls back to u' = x (or y when k is along x); the derivative is that
// branch's.
//
// THE ELEMENT FACTOR (R-ADR-041). Channel launches a leg from a short dipole
// p: its field carries |P_perp(d) p| = sqrt(|p|^2 - (d.p)^2), d the unit
// departure direction from the element. Radar owns the element pattern, so the
// join divides that magnitude out: a = 1 / |P_perp(d) p|, which makes the
// launched field a unit co-polar vector times the free-space transport, and the
// pattern stage applies the declared element pattern once. Along the element's
// own polarization axis the magnitude is zero and so is Channel's field; a is
// published as 0 there (a linearly polarized element has a null on its axis).
// p is each leg row's source polarization (one table [in | out | direct
// source | direct sink], rows of each leg): the vector
// Channel launched that row with, which turns with a radar pose. Empty tables
// disable the factor (a = 1).
//
// SITE-LESS ROWS. A transmitter-to-receiver leg (field_d, at the receiver) is
// a composed row of its own: V = a_t a_r (p_r . E) exp(j psi), with a_t on the
// leg's first segment, a_r = 1/|P_perp(d) p_r| on its last and p_r the
// receiver's polarization, so the receive projection and both element factors
// are this kernel's, in the same launch. ``place`` is each site row's and
// each site-less row's position in the composed order of one slot.
//
// THE FLOAT64 PATH LENGTH (R-ADR-040). Channel computes a leg's phase as
// -k32 L32 from its float32 unfolded length L32 (path_length_m) and float32
// wave number k32 = float32(2 pi f / c). The join recomputes the round-trip
// length in double, L64 = sum of |P_{j+1} - P_j| over the element, Channel's
// float32 interaction points and the site, and replaces Channel's phase:
//   psi = -(k64 (L64 - L32) + (k64 - k32) L32),   k64 = 2 pi f / c0,
// with c0 = 299792458 m/s exactly. (k64, k32) come from a per-frequency-column
// table: a wideband band lays its columns out as the outermost row blocks of
// each family, and each column's phase is Channel's at its own float32
// frequency. Line-of-sight rows are exact. A specular
// length is stationary for a move of its interaction point ALONG the surface
// (Fermat), so only the float32 point's error ACROSS the surface remains, up
// to (cos theta_i + cos theta_r) ulp32(x) / 2; a straight-through point enters
// at second order; a diffraction point is stationary only along its edge.
// tau = L64 / c0 is published as the float32 pair
// tau_hi = float(tau), tau_lo = float(tau - tau_hi); tau_lo is primal only.
// An empty wave table disables the correction: psi = 0, tau = tau_in + tau_out.
//
// Rules each pinned by a test:
//  1. Association. r = H^T M first (r = s H for a scalar), then V0 = r . E,
//     which for one non-zero component is (C_out S) C_in exactly.
//  2. A dead row's payload is exactly zero.
//  3. The VJP uses the frozen CSR tables: one thread owns one gradient slot
//     and loops its own segment, so there are no atomics and the summation
//     order is a property of the frozen join. Leg geometry gradients are owned
//     by leg rows; the per-endpoint reduction happens in Torch's own gather.
//  4. Sums accumulate in double and store float32.
//
// Slot stride: a slot-major replay of G instants carries G copies of one join.
// The index and CSR tables describe ONE slot; row k of the batch is join row
// k % rows_per_slot of slot g = k / rows_per_slot, and its leg rows are offset
// by g times the legs' per-slot row counts. Sites are shared across slots
// (sites_per_slot == 0) or slot-local (sites_per_slot > 0, a per-row
// response). The VJP walks the per-slot CSR once per slot in slot order.
//
// Validity is computed in Torch, not here, and enters as an int32 mask.

#include <torch/csrc/stable/accelerator.h>
#include <cmath>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cuda_runtime.h>
#include "radar_checks.cuh"
#include "path_interpolation.cuh"

#include <cstdint>

namespace {

constexpr double kSpeedOfLight = 299792458.0;

struct Complex {
  double re;
  double im;
};

__device__ __forceinline__ Complex mul(const Complex a, const Complex b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
__device__ __forceinline__ Complex add(const Complex a, const Complex b) { return {a.re + b.re, a.im + b.im}; }
__device__ __forceinline__ Complex scale(const Complex a, const double b) { return {a.re * b, a.im * b}; }

struct Vec3 {
  double x, y, z;
};
__device__ __forceinline__ Vec3 vsub(const Vec3 a, const Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ __forceinline__ Vec3 vadd(const Vec3 a, const Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ __forceinline__ Vec3 vscale(const Vec3 a, const double b) { return {a.x * b, a.y * b, a.z * b}; }
__device__ __forceinline__ double vdot(const Vec3 a, const Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ Vec3 vcross(const Vec3 a, const Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ __forceinline__ double vnorm(const Vec3 a) { return sqrt(vdot(a, a)); }
__device__ __forceinline__ double vget(const Vec3 a, const int j) { return j == 0 ? a.x : (j == 1 ? a.y : a.z); }
__device__ __forceinline__ Vec3 unit_axis(const int j) {
  return {j == 0 ? 1.0 : 0.0, j == 1 ? 1.0 : 0.0, j == 2 ? 1.0 : 0.0};
}

// A complex 3-vector: three complex components.
struct CVec3 {
  Complex c[3];
};

__device__ __forceinline__ CVec3 load_field(const float* __restrict__ field, const int64_t row) {
  const float* f = field + 6 * row;
  return {{{f[0], f[1]}, {f[2], f[3]}, {f[4], f[5]}}};
}
// sum_j a_j b_j with complex a and complex b, in component order.
__device__ __forceinline__ Complex cdot(const CVec3 a, const CVec3 b) {
  Complex sum = mul(a.c[0], b.c[0]);
  sum = add(sum, mul(a.c[1], b.c[1]));
  return add(sum, mul(a.c[2], b.c[2]));
}
// sum_j a_j v_j with complex a and real v.
__device__ __forceinline__ Complex cdot_real(const CVec3 a, const Vec3 v) {
  return add(add(scale(a.c[0], v.x), scale(a.c[1], v.y)), scale(a.c[2], v.z));
}

__device__ __forceinline__ Complex load(const float* __restrict__ values, const int64_t index) {
  return {static_cast<double>(values[2 * index]), static_cast<double>(values[2 * index + 1])};
}

template <typename T>
__device__ __forceinline__ Vec3 load3(const T* __restrict__ values, const int64_t row) {
  return {static_cast<double>(values[3 * row]), static_cast<double>(values[3 * row + 1]),
          static_cast<double>(values[3 * row + 2])};
}

// One slot's layout inside a slot-major batch.
struct SlotStride {
  int64_t rows;   // composed rows per slot
  int64_t in;     // inbound leg rows per slot
  int64_t out;    // outbound leg rows per slot
  int64_t sites;  // response slots per slot, 0 when shared by every slot
};

__device__ __forceinline__ int64_t strided(
    const int64_t* __restrict__ table, const int64_t k, const SlotStride stride, const int64_t per_slot) {
  return table[k % stride.rows] + (k / stride.rows) * per_slot;
}

// One leg's geometry: source element P_0, interaction points P_1..P_d and the
// site P_{d+1}. ``on`` is false when the leg carries no geometry.
template <typename SinkT>
struct LegGeometry {
  const float* src;
  const SinkT* sink;
  const float* points;
  const int32_t* depth;
  const float* length32;
  int64_t width;
  bool on;

  __device__ __forceinline__ int count(const int64_t row) const { return depth[row] + 2; }
  __device__ __forceinline__ Vec3 point(const int64_t row, const int j, const int last) const {
    if (j == 0) return load3(src, row);
    if (j == last) return load3(sink, row);
    return load3(points, row * width + (j - 1));
  }
};

// The leg's float64 unfolded length [m].
template <typename SinkT>
__device__ __forceinline__ double leg_length(const LegGeometry<SinkT>& leg, const int64_t row) {
  const int last = leg.count(row) - 1;
  double length = 0.0;
  Vec3 previous = leg.point(row, 0, last);
  for (int j = 1; j <= last; ++j) {
    const Vec3 next = leg.point(row, j, last);
    length += vnorm(vsub(next, previous));
    previous = next;
  }
  return length;
}

// The element factor a = 1 / |P_perp(d) p| of one segment of a leg - the first
// (the element launched it) or the last (an element receives it) - with its
// gradient with respect to that segment's unit direction d. ``head`` is the
// segment's start point index j0; the segment is P_{j0} -> P_{j0+1}.
struct ElementFactor {
  double value;
  Vec3 d_direction;  // da / dd
  Vec3 direction;    // d
  double segment;    // |P_{j0+1} - P_{j0}|
  int head;          // j0
};

__device__ __forceinline__ ElementFactor unit_factor() { return {1.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 1.0, 0}; }

template <typename SinkT>
__device__ __forceinline__ ElementFactor element_factor(
    const LegGeometry<SinkT>& leg, const int64_t row, const float* polarization, const bool last_segment) {
  if (polarization == nullptr || !leg.on) return unit_factor();
  const Vec3 p = load3(polarization, row);
  const int last = leg.count(row) - 1;
  const int head = last_segment ? last - 1 : 0;
  const Vec3 span = vsub(leg.point(row, head + 1, last), leg.point(row, head, last));
  const double segment = vnorm(span);
  const Vec3 d = segment > 0.0 ? vscale(span, 1.0 / segment) : Vec3{0.0, 0.0, 0.0};
  const double along = vdot(d, p);
  const double squared = vdot(p, p) - along * along;
  if (!(squared > 1e-12 * vdot(p, p)) || !(segment > 0.0)) {
    return {0.0, {0.0, 0.0, 0.0}, d, segment > 0.0 ? segment : 1.0, head};
  }
  const double a = 1.0 / sqrt(squared);
  return {a, vscale(p, a * a * a * along), d, segment, head};
}

// The site basis (h, v) for propagation direction k and up axis u, and its
// tangent along dk. h = (u x k)/|u x k|, v = h x k.
struct Basis {
  Vec3 h, v, dh, dv;
};

__device__ __forceinline__ Basis site_basis(const Vec3 k, const Vec3 up, const Vec3 dk) {
  Vec3 u = up;
  Vec3 w = vcross(u, k);
  double n = vnorm(w);
  if (!(n > 1e-6 * vnorm(u))) {
    u = fabs(k.x) < 0.9 ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
    w = vcross(u, k);
    n = vnorm(w);
  }
  const Vec3 h = vscale(w, 1.0 / n);
  const Vec3 v = vcross(h, k);
  const Vec3 dw = vcross(u, dk);
  const Vec3 dh = vscale(vsub(dw, vscale(h, vdot(h, dw))), 1.0 / n);
  const Vec3 dv = vadd(vcross(dh, k), vcross(h, dk));
  return {h, v, dh, dv};
}

// Everything one composed row needs, evaluated once per row.
struct RowState {
  CVec3 e;        // incident field E
  CVec3 hfield;   // receiver-launched field H
  CVec3 r;        // H^T M
  CVec3 m;        // M E
  Complex v0;     // H^T M E
  Complex phase;  // exp(j psi)
  Complex q;      // a_t a_r exp(j psi)
  Complex value;  // V
  ElementFactor at, ar;
  double tau;
};

struct ResponseView {
  const float* s;    // [owners, width] complex pairs
  const float* up;   // [owners, 3] when width == 4
  int width;
};

// r = H^T M and m = M E for one row; c_a and u_a are the matrix terms.
__device__ __forceinline__ void apply_response(
    const ResponseView& response, const int64_t s, const Vec3 k_in, const Vec3 k_out, const CVec3& h,
    const CVec3& e, CVec3& r, CVec3& m) {
  if (response.width == 1) {
    const Complex value = load(response.s, s);
    for (int j = 0; j < 3; ++j) {
      r.c[j] = mul(h.c[j], value);
      m.c[j] = mul(value, e.c[j]);
    }
    return;
  }
  const Vec3 up = load3(response.up, s);
  const Vec3 zero{0.0, 0.0, 0.0};
  const Basis bi = site_basis(k_in, up, zero);
  const Basis br = site_basis(k_out, up, zero);
  const Vec3 basis_i[2] = {bi.h, bi.v};
  const Vec3 basis_r[2] = {br.h, br.v};
  Complex c[2], t[2];
  for (int a = 0; a < 2; ++a) c[a] = cdot_real(h, basis_r[a]);
  for (int b = 0; b < 2; ++b) t[b] = cdot_real(e, basis_i[b]);
  Complex rb[2], ma[2];
  for (int b = 0; b < 2; ++b) {
    rb[b] = add(mul(load(response.s, 4 * s + b), c[0]), mul(load(response.s, 4 * s + 2 + b), c[1]));
  }
  for (int a = 0; a < 2; ++a) {
    ma[a] = add(mul(load(response.s, 4 * s + 2 * a), t[0]), mul(load(response.s, 4 * s + 2 * a + 1), t[1]));
  }
  for (int j = 0; j < 3; ++j) {
    r.c[j] = add(scale(rb[0], vget(basis_i[0], j)), scale(rb[1], vget(basis_i[1], j)));
    m.c[j] = add(scale(ma[0], vget(basis_r[0], j)), scale(ma[1], vget(basis_r[1], j)));
  }
}

// dV0 for a tangent of k_in (incident basis) and of k_out (receive basis).
__device__ __forceinline__ Complex basis_tangent(
    const ResponseView& response, const int64_t s, const Vec3 k_in, const Vec3 k_out, const Vec3 dk_in,
    const Vec3 dk_out, const CVec3& h, const CVec3& e) {
  const Vec3 up = load3(response.up, s);
  const Basis bi = site_basis(k_in, up, dk_in);
  const Basis br = site_basis(k_out, up, dk_out);
  const Vec3 basis_i[2] = {bi.h, bi.v}, dbasis_i[2] = {bi.dh, bi.dv};
  const Vec3 basis_r[2] = {br.h, br.v}, dbasis_r[2] = {br.dh, br.dv};
  Complex total{0.0, 0.0};
  for (int a = 0; a < 2; ++a) {
    const Complex ca = cdot_real(h, basis_r[a]);
    const Complex dca = cdot_real(h, dbasis_r[a]);
    for (int b = 0; b < 2; ++b) {
      const Complex sab = load(response.s, 4 * s + 2 * a + b);
      const Complex tb = cdot_real(e, basis_i[b]);
      const Complex dtb = cdot_real(e, dbasis_i[b]);
      total = add(total, mul(sab, add(mul(dca, tb), mul(ca, dtb))));
    }
  }
  return total;
}


template <typename SinkT>
struct JoinInputs {
  const float* tau_in;
  const float* tau_out;
  const float* field_in;
  const float* field_out;
  const float* k_in;
  const float* k_out;
  LegGeometry<SinkT> geo_in, geo_out;
  ResponseView response;
  const int32_t* row_valid;
  const int64_t* idx_in;
  const int64_t* idx_out;
  const int64_t* idx_s;
  // Each leg row's source polarization (nullptr: no element factor).
  const float* pol_in;
  const float* pol_out;
  // (k64, k32) per frequency column. A band lays its columns out as the
  // outermost blocks of both row families: site row k is in column
  // k / (site_total / columns), site-less row d in d / (direct_total / columns).
  const double* waves;
  int64_t columns;
  SlotStride stride;
  // Where each row lands in the composed order: composed row of site row k is
  // slot * paths + place[k % rows_per_slot], of direct row d is
  // slot * paths + place[rows_per_slot + d % direct_per_slot]. nullptr: the
  // site rows are the composed order.
  const int64_t* place;
  int64_t paths;
  // The site-less transmitter-to-receiver leg, one composed row per leg row.
  LegGeometry<float> geo_d;
  const float* field_d;
  const float* pol_d_src;
  const float* pol_d_sink;
  const int32_t* valid_d;
  int64_t direct_per_slot;
  int64_t direct_total;
  int64_t site_total;
  // Whether the legs carry Channel's geometry: the float64 path length, the
  // element factors and the float32 delay pair. Decided by the wave table.
  bool geometric;
};

struct Waves {
  double k64, k32;
};

__device__ __forceinline__ Waves waves_of_column(const double* waves, const int64_t column) {
  if (waves == nullptr) return {0.0, 0.0};
  return {waves[2 * column], waves[2 * column + 1]};
}

template <typename SinkT>
__device__ __forceinline__ Waves site_waves(const JoinInputs<SinkT>& in, const int64_t k) {
  return waves_of_column(in.waves, k / (in.site_total / in.columns));
}

template <typename SinkT>
__device__ __forceinline__ Waves direct_waves(const JoinInputs<SinkT>& in, const int64_t d) {
  return waves_of_column(in.waves, d / (in.direct_total / in.columns));
}

template <typename SinkT>
__device__ __forceinline__ int64_t site_position(const JoinInputs<SinkT>& in, const int64_t k) {
  if (in.place == nullptr) return k;
  return (k / in.stride.rows) * in.paths + in.place[k % in.stride.rows];
}

template <typename SinkT>
__device__ __forceinline__ int64_t direct_position(const JoinInputs<SinkT>& in, const int64_t d) {
  return (d / in.direct_per_slot) * in.paths + in.place[in.stride.rows + d % in.direct_per_slot];
}

template <typename SinkT>
__device__ __forceinline__ Vec3 direction_or_zero(const float* k, const int64_t row) {
  return k == nullptr ? Vec3{0.0, 0.0, 0.0} : load3(k, row);
}

// The carrier rotation exp(j psi), psi = -(k64 (L64 - L32) + (k64 - k32) L32).
__device__ __forceinline__ Complex carrier_rotation(const Waves w, const double l64, const double l32) {
  const double psi = -(w.k64 * (l64 - l32) + (w.k64 - w.k32) * l32);
  double sine, cosine;
  sincos(psi, &sine, &cosine);
  return {cosine, sine};
}

template <typename SinkT>
__device__ RowState row_state(
    const JoinInputs<SinkT>& in, const int64_t k, const int64_t i, const int64_t o, const int64_t s) {
  RowState state;
  state.e = load_field(in.field_in, i);
  state.hfield = load_field(in.field_out, o);
  const Vec3 k_in = direction_or_zero<SinkT>(in.k_in, i);
  const Vec3 k_out = direction_or_zero<SinkT>(in.k_out, o);
  apply_response(in.response, s, k_in, k_out, state.hfield, state.e, state.r, state.m);
  state.v0 = cdot(state.r, state.e);
  state.at = element_factor(in.geo_in, i, in.pol_in, false);
  state.ar = element_factor(in.geo_out, o, in.pol_out, false);
  if (in.geometric) {
    const double l64 = leg_length(in.geo_in, i) + leg_length(in.geo_out, o);
    const double l32 = static_cast<double>(in.geo_in.length32[i]) + static_cast<double>(in.geo_out.length32[o]);
    state.phase = carrier_rotation(site_waves(in, k), l64, l32);
    state.q = scale(state.phase, state.at.value * state.ar.value);
    state.value = mul(state.v0, state.q);
    state.tau = l64 / kSpeedOfLight;
  } else {
    state.phase = {1.0, 0.0};
    state.q = {1.0, 0.0};
    state.value = state.v0;
    state.tau = static_cast<double>(in.tau_in[i]) + static_cast<double>(in.tau_out[o]);
  }
  return state;
}

// A site-less row: the transmitter's field at the receiver, read along the
// receiver's polarization, V = a_t a_r (p_r . E) exp(j psi), with a_r the
// receive element's factor on the leg's LAST segment.
struct DirectState {
  CVec3 e;
  Vec3 p;  // receive polarization
  ElementFactor at, ar;
  Complex v0, phase, value;
  double tau;
};

template <typename SinkT>
__device__ DirectState direct_state(const JoinInputs<SinkT>& in, const int64_t d) {
  DirectState state;
  state.e = load_field(in.field_d, d);
  state.p = load3(in.pol_d_sink, d);
  state.at = element_factor(in.geo_d, d, in.pol_d_src, false);
  state.ar = element_factor(in.geo_d, d, in.pol_d_sink, true);
  state.v0 = cdot_real(state.e, state.p);
  const double l64 = leg_length(in.geo_d, d);
  state.phase = carrier_rotation(direct_waves(in, d), l64, static_cast<double>(in.geo_d.length32[d]));
  state.value = mul(state.v0, scale(state.phase, state.at.value * state.ar.value));
  state.tau = l64 / kSpeedOfLight;
  return state;
}

__device__ __forceinline__ void write_row(
    float* __restrict__ tau_hi, float* __restrict__ tau_lo, float* __restrict__ value, const int64_t at,
    const double tau, const bool pair, const Complex v) {
  const float high = static_cast<float>(tau);
  tau_hi[at] = high;
  tau_lo[at] = pair ? static_cast<float>(tau - static_cast<double>(high)) : 0.0f;
  value[2 * at] = static_cast<float>(v.re);
  value[2 * at + 1] = static_cast<float>(v.im);
}

template <typename SinkT>
__global__ void two_way_join_forward_kernel(
    const JoinInputs<SinkT> in, float* __restrict__ tau_hi, float* __restrict__ tau_lo,
    float* __restrict__ value) {
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread < in.site_total) {
    const int64_t k = thread;
    const int64_t at = site_position(in, k);
    if (in.row_valid[k] == 0) {
      write_row(tau_hi, tau_lo, value, at, 0.0, false, {0.0, 0.0});
      return;
    }
    const int64_t i = strided(in.idx_in, k, in.stride, in.stride.in);
    const int64_t o = strided(in.idx_out, k, in.stride, in.stride.out);
    const int64_t s = strided(in.idx_s, k, in.stride, in.stride.sites);
    const RowState state = row_state(in, k, i, o, s);
    write_row(tau_hi, tau_lo, value, at, state.tau, in.geometric, state.value);
    return;
  }
  const int64_t d = thread - in.site_total;
  if (d >= in.direct_total) return;
  const int64_t at = direct_position(in, d);
  if (in.valid_d[d] == 0) {
    write_row(tau_hi, tau_lo, value, at, 0.0, false, {0.0, 0.0});
    return;
  }
  const DirectState state = direct_state(in, d);
  write_row(tau_hi, tau_lo, value, at, state.tau, true, state.value);
}

// Tangents of one leg's geometry.
template <typename SinkT>
struct LegTangent {
  const float* src;
  const SinkT* sink;
  const float* points;
  const float* length32;
};

template <typename SinkT>
__device__ __forceinline__ Vec3 tangent_point(
    const LegGeometry<SinkT>& leg, const LegTangent<SinkT>& tangent, const int64_t row, const int j, const int last) {
  if (j == 0) return load3(tangent.src, row);
  if (j == last) return load3(tangent.sink, row);
  return load3(tangent.points, row * leg.width + (j - 1));
}

// d L64 of one leg along a geometry tangent.
template <typename SinkT>
__device__ __forceinline__ double length_tangent(
    const LegGeometry<SinkT>& leg, const LegTangent<SinkT>& tangent, const int64_t row) {
  const int last = leg.count(row) - 1;
  double d_length = 0.0;
  Vec3 previous = leg.point(row, 0, last);
  Vec3 d_previous = tangent_point(leg, tangent, row, 0, last);
  for (int j = 1; j <= last; ++j) {
    const Vec3 next = leg.point(row, j, last);
    const Vec3 d_next = tangent_point(leg, tangent, row, j, last);
    const Vec3 segment = vsub(next, previous);
    const double length = vnorm(segment);
    if (length > 0.0) d_length += vdot(vscale(segment, 1.0 / length), vsub(d_next, d_previous));
    previous = next;
    d_previous = d_next;
  }
  return d_length;
}

// d a of an element factor along a geometry tangent:
// d(d) = (I - d d^T) d(P_{j0+1} - P_{j0}) / |P_{j0+1} - P_{j0}|.
template <typename SinkT>
__device__ __forceinline__ double factor_tangent(
    const LegGeometry<SinkT>& leg, const LegTangent<SinkT>& tangent, const int64_t row, const ElementFactor& factor) {
  if (!(factor.value > 0.0) || !leg.on) return 0.0;
  const int last = leg.count(row) - 1;
  const Vec3 d_span = vsub(
      tangent_point(leg, tangent, row, factor.head + 1, last), tangent_point(leg, tangent, row, factor.head, last));
  const Vec3 dd =
      vscale(vsub(d_span, vscale(factor.direction, vdot(factor.direction, d_span))), 1.0 / factor.segment);
  return vdot(factor.d_direction, dd);
}

template <typename SinkT>
__global__ void two_way_join_jvp_kernel(
    const JoinInputs<SinkT> in, const float* __restrict__ tan_tau_in, const float* __restrict__ tan_tau_out,
    const float* __restrict__ tan_field_in, const float* __restrict__ tan_field_out,
    const float* __restrict__ tan_k_in, const float* __restrict__ tan_k_out, const LegTangent<SinkT> tan_in,
    const LegTangent<SinkT> tan_out, const float* __restrict__ tan_s, const float* __restrict__ tan_field_d,
    const LegTangent<float> tan_d, float* __restrict__ tan_tau, float* __restrict__ tan_value) {
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread >= in.site_total) {
    const int64_t d = thread - in.site_total;
    if (d >= in.direct_total) return;
    const int64_t at = direct_position(in, d);
    if (in.valid_d[d] == 0) {
      tan_tau[at] = 0.0f;
      tan_value[2 * at] = 0.0f;
      tan_value[2 * at + 1] = 0.0f;
      return;
    }
    const DirectState state = direct_state(in, d);
    const Waves w = direct_waves(in, d);
    const double dl64 = length_tangent(in.geo_d, tan_d, d);
    const double dpsi = -(w.k64 * dl64 - w.k32 * static_cast<double>(tan_d.length32[d]));
    const double da_t = factor_tangent(in.geo_d, tan_d, d, state.at);
    const double da_r = factor_tangent(in.geo_d, tan_d, d, state.ar);
    const Complex rotated = mul(state.v0, state.phase);
    Complex dv = mul(cdot_real(load_field(tan_field_d, d), state.p), scale(state.phase, state.at.value * state.ar.value));
    dv = add(dv, mul(Complex{0.0, dpsi}, state.value));
    dv = add(dv, scale(rotated, da_t * state.ar.value + state.at.value * da_r));
    tan_tau[at] = static_cast<float>(dl64 / kSpeedOfLight);
    tan_value[2 * at] = static_cast<float>(dv.re);
    tan_value[2 * at + 1] = static_cast<float>(dv.im);
    return;
  }
  const int64_t k = thread;
  const int64_t at = site_position(in, k);
  if (in.row_valid[k] == 0) {
    tan_tau[at] = 0.0f;
    tan_value[2 * at] = 0.0f;
    tan_value[2 * at + 1] = 0.0f;
    return;
  }
  const int64_t i = strided(in.idx_in, k, in.stride, in.stride.in);
  const int64_t o = strided(in.idx_out, k, in.stride, in.stride.out);
  const int64_t s = strided(in.idx_s, k, in.stride, in.stride.sites);
  const RowState state = row_state(in, k, i, o, s);
  const CVec3 de = load_field(tan_field_in, i);
  const CVec3 dh = load_field(tan_field_out, o);

  // dV0, each term keeping the primal association: (dH M) E, (H dM) E, (H M) dE.
  CVec3 dr, unused;
  apply_response(in.response, s, direction_or_zero<SinkT>(in.k_in, i), direction_or_zero<SinkT>(in.k_out, o), dh,
                 state.e, dr, unused);
  Complex first = cdot(dr, state.e);
  Complex second{0.0, 0.0};
  if (in.response.width == 1) {
    const Complex ds = load(tan_s, s);
    CVec3 hs;
    for (int j = 0; j < 3; ++j) hs.c[j] = mul(state.hfield.c[j], ds);
    second = cdot(hs, state.e);
  } else {
    const ResponseView tangent_response{tan_s, in.response.up, 4};
    CVec3 hs;
    apply_response(tangent_response, s, direction_or_zero<SinkT>(in.k_in, i),
                   direction_or_zero<SinkT>(in.k_out, o), state.hfield, state.e, hs, unused);
    second = cdot(hs, state.e);
    second = add(second, basis_tangent(in.response, s, load3(in.k_in, i), load3(in.k_out, o), load3(tan_k_in, i),
                                       load3(tan_k_out, o), state.hfield, state.e));
  }
  const Complex third = cdot(state.r, de);
  const Complex dv0 = add(add(first, second), third);

  if (!in.geometric) {
    tan_tau[at] = static_cast<float>(static_cast<double>(tan_tau_in[i]) + static_cast<double>(tan_tau_out[o]));
    tan_value[2 * at] = static_cast<float>(dv0.re);
    tan_value[2 * at + 1] = static_cast<float>(dv0.im);
    return;
  }
  const Waves w = site_waves(in, k);
  const double dl64 = length_tangent(in.geo_in, tan_in, i) + length_tangent(in.geo_out, tan_out, o);
  const double dl32 = static_cast<double>(tan_in.length32[i]) + static_cast<double>(tan_out.length32[o]);
  const double dpsi = -(w.k64 * dl64 - w.k32 * dl32);
  const double da_t = factor_tangent(in.geo_in, tan_in, i, state.at);
  const double da_r = factor_tangent(in.geo_out, tan_out, o, state.ar);
  // V = a_t a_r exp(j psi) V0.
  Complex dv = mul(dv0, state.q);
  dv = add(dv, mul(Complex{0.0, dpsi}, state.value));
  dv = add(dv, scale(mul(state.v0, state.phase), da_t * state.ar.value + state.at.value * da_r));
  tan_tau[at] = static_cast<float>(dl64 / kSpeedOfLight);
  tan_value[2 * at] = static_cast<float>(dv.re);
  tan_value[2 * at + 1] = static_cast<float>(dv.im);
}

// Reverse-mode outputs of one leg family.
template <typename SinkT>
struct LegGradients {
  float* tau;
  float* field;
  float* k;
  float* src;
  SinkT* sink;
  float* points;
  float* length32;
};

// The projection of one element factor's cotangent onto its segment's end
// points: d a / d P_{j0+1} = (I - d d^T)(da/dd)/|span| and the negative at P_{j0}.
__device__ __forceinline__ Vec3 factor_push(const ElementFactor& factor, const double g_factor) {
  return vscale(vsub(factor.d_direction, vscale(factor.direction, vdot(factor.direction, factor.d_direction))),
                g_factor / factor.segment);
}

// The geometry gradient of one leg row from the accumulated dLoss/dL64,
// dLoss/dL32 and the cotangents of its first- and last-segment element factors.
template <typename SinkT>
__device__ __forceinline__ void write_leg_geometry(
    const LegGeometry<SinkT>& leg, const LegGradients<SinkT>& out, const int64_t row, const double g_length,
    const double g_length32, const ElementFactor& first, const double g_first, const ElementFactor& last_factor,
    const double g_last) {
  const int last = leg.count(row) - 1;
  for (int64_t j = 0; j < leg.width; ++j) {
    for (int c = 0; c < 3; ++c) out.points[3 * (row * leg.width + j) + c] = 0.0f;
  }
  const bool use_first = g_first != 0.0 && first.value > 0.0;
  const bool use_last = g_last != 0.0 && last_factor.value > 0.0;
  const Vec3 push_first = use_first ? factor_push(first, g_first) : Vec3{0.0, 0.0, 0.0};
  const Vec3 push_last = use_last ? factor_push(last_factor, g_last) : Vec3{0.0, 0.0, 0.0};
  Vec3 incoming{0.0, 0.0, 0.0};  // unit direction of the segment ending at P_j
  for (int j = 0; j <= last; ++j) {
    Vec3 outgoing{0.0, 0.0, 0.0};
    if (j < last) {
      const Vec3 segment = vsub(leg.point(row, j + 1, last), leg.point(row, j, last));
      const double length = vnorm(segment);
      if (length > 0.0) outgoing = vscale(segment, 1.0 / length);
    }
    Vec3 gradient = vscale(vsub(incoming, outgoing), g_length);
    if (use_first && j == first.head) gradient = vsub(gradient, push_first);
    if (use_first && j == first.head + 1) gradient = vadd(gradient, push_first);
    if (use_last && j == last_factor.head) gradient = vsub(gradient, push_last);
    if (use_last && j == last_factor.head + 1) gradient = vadd(gradient, push_last);
    if (j == 0) {
      out.src[3 * row] = static_cast<float>(gradient.x);
      out.src[3 * row + 1] = static_cast<float>(gradient.y);
      out.src[3 * row + 2] = static_cast<float>(gradient.z);
    } else if (j == last) {
      out.sink[3 * row] = static_cast<SinkT>(gradient.x);
      out.sink[3 * row + 1] = static_cast<SinkT>(gradient.y);
      out.sink[3 * row + 2] = static_cast<SinkT>(gradient.z);
    } else {
      float* point = out.points + 3 * (row * leg.width + (j - 1));
      point[0] = static_cast<float>(gradient.x);
      point[1] = static_cast<float>(gradient.y);
      point[2] = static_cast<float>(gradient.z);
    }
    incoming = outgoing;
  }
  out.length32[row] = static_cast<float>(g_length32);
}

// Real-pair backpropagation through z = w * cofactor.
__device__ __forceinline__ void accumulate(const double g_re, const double g_im, const Complex w, double& d_re,
                                           double& d_im) {
  d_re += g_re * w.re + g_im * w.im;
  d_im += -g_re * w.im + g_im * w.re;
}

// The cotangent of a real scalar x with dV/dx = w: Re(g conj-pair w).
__device__ __forceinline__ double real_cotangent(const double g_re, const double g_im, const Complex w) {
  return g_re * w.re + g_im * w.im;
}

// The site-less leg's VJP: one thread per direct row, which is its only
// composed row, so it owns its gradient slot outright.
template <typename SinkT>
__device__ void direct_backward(
    const JoinInputs<SinkT>& in, const int64_t d, const float* __restrict__ grad_tau,
    const float* __restrict__ grad_value, const LegGradients<float>& out) {
  if (in.valid_d[d] == 0) {
    for (int c = 0; c < 6; ++c) out.field[6 * d + c] = 0.0f;
    const ElementFactor none = unit_factor();
    write_leg_geometry(in.geo_d, out, d, 0.0, 0.0, none, 0.0, none, 0.0);
    return;
  }
  const int64_t at = direct_position(in, d);
  const DirectState state = direct_state(in, d);
  const Waves w = direct_waves(in, d);
  const double g_re = static_cast<double>(grad_value[2 * at]);
  const double g_im = static_cast<double>(grad_value[2 * at + 1]);
  const Complex gain = scale(state.phase, state.at.value * state.ar.value);
  double d_field[6] = {0, 0, 0, 0, 0, 0};
  for (int j = 0; j < 3; ++j) accumulate(g_re, g_im, scale(gain, vget(state.p, j)), d_field[2 * j], d_field[2 * j + 1]);
  for (int c = 0; c < 6; ++c) out.field[6 * d + c] = static_cast<float>(d_field[c]);
  const double rotation = g_im * state.value.re - g_re * state.value.im;
  const double g_length = -w.k64 * rotation + static_cast<double>(grad_tau[at]) / kSpeedOfLight;
  const double g_length32 = w.k32 * rotation;
  const Complex rotated = mul(state.v0, state.phase);
  const double g_at = real_cotangent(g_re, g_im, scale(rotated, state.ar.value));
  const double g_ar = real_cotangent(g_re, g_im, scale(rotated, state.at.value));
  write_leg_geometry(in.geo_d, out, d, g_length, g_length32, state.at, g_at, state.ar, g_ar);
}

// One thread owns one gradient slot. The four owner families - inbound rows,
// outbound rows, response owners, site-less rows - are laid end to end on a
// single grid so the whole VJP is one launch, and each thread walks only its
// own CSR segment (a site-less row owns exactly one composed row).
template <typename SinkT>
__global__ void two_way_join_backward_kernel(
    const JoinInputs<SinkT> in, const int64_t* __restrict__ by_in_offsets, const int64_t* __restrict__ by_in_rows,
    const int64_t* __restrict__ by_out_offsets, const int64_t* __restrict__ by_out_rows,
    const int64_t* __restrict__ by_s_offsets, const int64_t* __restrict__ by_s_rows,
    const float* __restrict__ grad_tau, const float* __restrict__ grad_value, const LegGradients<SinkT> grad_in,
    const LegGradients<SinkT> grad_out, float* __restrict__ grad_s, const LegGradients<float> grad_d,
    const int num_in, const int num_out, const int num_owners, const int64_t num_slots) {
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(num_in) + num_out + num_owners;
  if (thread >= total) {
    const int64_t d = thread - total;
    if (d < in.direct_total) direct_backward(in, d, grad_tau, grad_value, grad_d);
    return;
  }

  int family;
  int64_t slot;
  int64_t per_slot;
  const int64_t* offsets;
  const int64_t* segment_rows;
  if (thread < num_in) {
    family = 0;
    slot = thread;
    per_slot = in.stride.in;
    offsets = by_in_offsets;
    segment_rows = by_in_rows;
  } else if (thread < num_in + num_out) {
    family = 1;
    slot = thread - num_in;
    per_slot = in.stride.out;
    offsets = by_out_offsets;
    segment_rows = by_out_rows;
  } else {
    family = 2;
    slot = thread - num_in - num_out;
    per_slot = in.stride.sites;
    offsets = by_s_offsets;
    segment_rows = by_s_rows;
  }
  const bool shared = per_slot == 0;
  const int64_t owner = shared ? slot : slot % per_slot;
  const int64_t first_slot = shared ? 0 : slot / per_slot;
  const int64_t last_slot = shared ? num_slots : first_slot + 1;
  const bool geometric = in.geometric;
  const int width = in.response.width;

  double d_tau = 0.0, g_length = 0.0, g_length32 = 0.0, g_factor = 0.0;
  double d_field[6] = {0, 0, 0, 0, 0, 0};
  double d_k[3] = {0, 0, 0};
  double d_s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const int64_t begin = offsets[owner];
  const int64_t end = offsets[owner + 1];
  for (int64_t g = first_slot; g < last_slot; ++g)
    for (int64_t entry = begin; entry < end; ++entry) {
      const int64_t k = segment_rows[entry] + g * in.stride.rows;
      if (in.row_valid[k] == 0) continue;
      const int64_t i = strided(in.idx_in, k, in.stride, in.stride.in);
      const int64_t o = strided(in.idx_out, k, in.stride, in.stride.out);
      const int64_t s = strided(in.idx_s, k, in.stride, in.stride.sites);
      const RowState state = row_state(in, k, i, o, s);
      const int64_t at = site_position(in, k);
      const double g_re = static_cast<double>(grad_value[2 * at]);
      const double g_im = static_cast<double>(grad_value[2 * at + 1]);
      const double g_tau = static_cast<double>(grad_tau[at]);
      if (family == 2) {
        if (width == 1) {
          accumulate(g_re, g_im, mul(cdot(state.hfield, state.e), state.q), d_s[0], d_s[1]);
        } else {
          const Vec3 up = load3(in.response.up, s);
          const Vec3 zero{0.0, 0.0, 0.0};
          const Basis bi = site_basis(load3(in.k_in, i), up, zero);
          const Basis br = site_basis(load3(in.k_out, o), up, zero);
          const Vec3 basis_i[2] = {bi.h, bi.v};
          const Vec3 basis_r[2] = {br.h, br.v};
          for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
              const Complex w =
                  mul(mul(cdot_real(state.hfield, basis_r[a]), cdot_real(state.e, basis_i[b])), state.q);
              accumulate(g_re, g_im, w, d_s[2 * (2 * a + b)], d_s[2 * (2 * a + b) + 1]);
            }
        }
        continue;
      }
      const CVec3& cofactor = family == 0 ? state.r : state.m;
      for (int j = 0; j < 3; ++j) {
        const Complex w = geometric ? mul(cofactor.c[j], state.q) : cofactor.c[j];
        accumulate(g_re, g_im, w, d_field[2 * j], d_field[2 * j + 1]);
      }
      if (width == 4) {
        for (int axis = 0; axis < 3; ++axis) {
          const Vec3 zero{0.0, 0.0, 0.0};
          const Vec3 tangent = unit_axis(axis);
          const Complex dv0 = family == 0
              ? basis_tangent(in.response, s, load3(in.k_in, i), load3(in.k_out, o), tangent, zero, state.hfield,
                              state.e)
              : basis_tangent(in.response, s, load3(in.k_in, i), load3(in.k_out, o), zero, tangent, state.hfield,
                              state.e);
          const Complex dv = mul(dv0, state.q);
          d_k[axis] += g_re * dv.re + g_im * dv.im;
        }
      }
      if (!geometric) {
        d_tau += g_tau;
        continue;
      }
      const Waves w = site_waves(in, k);
      // Q = g_im Re V - g_re Im V: the cotangent of a phase rotation.
      const double rotation = g_im * state.value.re - g_re * state.value.im;
      g_length += -w.k64 * rotation + g_tau / kSpeedOfLight;
      g_length32 += w.k32 * rotation;
      // dV/da_own = a_other exp(j psi) V0.
      const double other = family == 0 ? state.ar.value : state.at.value;
      g_factor += real_cotangent(g_re, g_im, scale(mul(state.v0, state.phase), other));
    }

  if (family == 2) {
    for (int c = 0; c < 2 * width; ++c) grad_s[2 * width * slot + c] = static_cast<float>(d_s[c]);
    return;
  }
  const LegGradients<SinkT>& out = family == 0 ? grad_in : grad_out;
  const LegGeometry<SinkT>& leg = family == 0 ? in.geo_in : in.geo_out;
  out.tau[slot] = static_cast<float>(d_tau);
  for (int c = 0; c < 6; ++c) out.field[6 * slot + c] = static_cast<float>(d_field[c]);
  if (out.k != nullptr) {
    for (int c = 0; c < 3; ++c) out.k[3 * slot + c] = static_cast<float>(d_k[c]);
  }
  if (geometric) {
    const ElementFactor own = element_factor(leg, slot, family == 0 ? in.pol_in : in.pol_out, false);
    write_leg_geometry(leg, out, slot, g_length, g_length32, own, g_factor, unit_factor(), 0.0);
  }
}

void check_float_len(const torch::stable::Tensor& tensor, int64_t expected, const char* name) {
  check_cuda_float(tensor, name);
  STD_TORCH_CHECK(tensor.numel() == expected, name, " must hold ", expected, " values.");
}

// The per-slot layout of a slot-major batch, checked against the totals.
SlotStride slot_stride(
    int64_t rows, int64_t legs_in, int64_t legs_out, int64_t owners, int64_t slots, int64_t slot_local_sites) {
  STD_TORCH_CHECK(slots >= 1, "slots must be positive.");
  STD_TORCH_CHECK(
      rows % slots == 0 && legs_in % slots == 0 && legs_out % slots == 0,
      "a slot-major join carries the same rows and legs in every slot.");
  STD_TORCH_CHECK(
      slot_local_sites == 0 || owners % slots == 0, "slot-local responses must hold the same count in every slot.");
  return {rows / slots, legs_in / slots, legs_out / slots, slot_local_sites == 0 ? 0 : owners / slots};
}

bool sink_is_double(const torch::stable::Tensor& sink) {
  return sink.scalar_type() == torch::headeronly::ScalarType::Double;
}

void check_sink(const torch::stable::Tensor& sink, int64_t rows, const char* name) {
  STD_TORCH_CHECK(sink.is_cuda() && sink.is_contiguous(), name, " must be a contiguous CUDA tensor.");
  STD_TORCH_CHECK(
      sink.scalar_type() == torch::headeronly::ScalarType::Float ||
          sink.scalar_type() == torch::headeronly::ScalarType::Double,
      name, " must be float32 or float64.");
  STD_TORCH_CHECK(sink.numel() == 3 * rows, name, " must hold one point per leg row.");
}

// The geometry of one leg, checked; it is used exactly when the join is
// geometric, and a leg with no rows has none to check.
template <typename SinkT>
LegGeometry<SinkT> leg_geometry(
    const torch::stable::Tensor& src, const torch::stable::Tensor& sink, const torch::stable::Tensor& points,
    const torch::stable::Tensor& depth, const torch::stable::Tensor& length32, int64_t legs, const char* name,
    bool geometric) {
  if (!geometric || legs == 0) {
    STD_TORCH_CHECK(geometric || length32.numel() == 0, name, " carries geometry but the join has no wave numbers.");
    return {nullptr, nullptr, nullptr, nullptr, nullptr, 0, false};
  }
  check_float_len(length32, legs, "path_length_m");
  check_float_len(src, 3 * legs, "departure_origin_m");
  check_sink(sink, legs, "arrival_target_m");
  STD_TORCH_CHECK(
      sink.scalar_type() == (sizeof(SinkT) == 8 ? torch::headeronly::ScalarType::Double
                                                 : torch::headeronly::ScalarType::Float),
      name, " site precision differs from the join's.");
  check_cuda_float(points, "interaction_positions_m");
  check_cuda_int(depth, "depth");
  STD_TORCH_CHECK(depth.numel() == legs, name, " depth must hold one value per leg row.");
  STD_TORCH_CHECK(points.numel() % (3 * legs) == 0, name, " interaction table is not [rows, W, 3].");
  return {src.const_data_ptr<float>(), sink.const_data_ptr<SinkT>(), points.const_data_ptr<float>(),
          depth.const_data_ptr<int32_t>(), length32.const_data_ptr<float>(), points.numel() / (3 * legs), true};
}

const float* polarization_rows(const torch::stable::Tensor& polarization, int64_t legs, const char* name) {
  if (polarization.numel() == 0) return nullptr;
  check_float_len(polarization, 3 * legs, name);
  return polarization.const_data_ptr<float>();
}

#define JOIN_INPUT_PARAMS                                                                                      \
  const torch::stable::Tensor &tau_in, const torch::stable::Tensor &tau_out,                                  \
      const torch::stable::Tensor &field_in, const torch::stable::Tensor &field_out,                          \
      const torch::stable::Tensor &k_in, const torch::stable::Tensor &k_out, const torch::stable::Tensor &src_in, \
      const torch::stable::Tensor &sink_in, const torch::stable::Tensor &points_in,                           \
      const torch::stable::Tensor &depth_in, const torch::stable::Tensor &length_in,                          \
      const torch::stable::Tensor &src_out, const torch::stable::Tensor &sink_out,                            \
      const torch::stable::Tensor &points_out, const torch::stable::Tensor &depth_out,                        \
      const torch::stable::Tensor &length_out, const torch::stable::Tensor &response,                         \
      const torch::stable::Tensor &up, const torch::stable::Tensor &polarization,                             \
      const torch::stable::Tensor &field_d, const torch::stable::Tensor &src_d,                               \
      const torch::stable::Tensor &sink_d, const torch::stable::Tensor &points_d,                             \
      const torch::stable::Tensor &depth_d, const torch::stable::Tensor &length_d,                            \
      const torch::stable::Tensor &valid_d, const torch::stable::Tensor &waves,                               \
      const torch::stable::Tensor &place, const torch::stable::Tensor &row_valid,                             \
      const torch::stable::Tensor &idx_in, const torch::stable::Tensor &idx_out,                              \
      const torch::stable::Tensor &idx_s

#define JOIN_SCALAR_PARAMS int64_t response_width, int64_t slots, int64_t slot_local, int64_t columns

#define JOIN_INPUT_ARGS                                                                                        \
  tau_in, tau_out, field_in, field_out, k_in, k_out, src_in, sink_in, points_in, depth_in, length_in, src_out, \
      sink_out, points_out, depth_out, length_out, response, up, polarization, field_d, src_d, sink_d,          \
      points_d, depth_d, length_d, valid_d, waves, place, row_valid, idx_in, idx_out, idx_s, num_rows,          \
      response_width, slots, slot_local, columns

// Everything forward, JVP and VJP share, checked once. ``num_rows`` is the
// composed output length (site rows and site-less rows of every slot).
template <typename SinkT>
JoinInputs<SinkT> join_inputs(JOIN_INPUT_PARAMS, int64_t num_rows, JOIN_SCALAR_PARAMS) {
  check_cuda_float(tau_in, "tau_in");
  check_cuda_float(tau_out, "tau_out");
  const int64_t legs_in = tau_in.numel();
  const int64_t legs_out = tau_out.numel();
  check_float_len(field_in, 6 * legs_in, "field_in");
  check_float_len(field_out, 6 * legs_out, "field_out");
  const int64_t width = response_width;
  STD_TORCH_CHECK(width == 1 || width == 4, "a scatter response has width 1 (scalar) or 4 (2x2 matrix).");
  check_cuda_float(response, "response");
  STD_TORCH_CHECK(response.numel() % (2 * width) == 0, "response must hold whole complex values.");
  const int64_t owners = response.numel() / (2 * width);
  if (width == 4) {
    check_float_len(up, 3 * owners, "up");
    check_float_len(k_in, 3 * legs_in, "k_in");
    check_float_len(k_out, 3 * legs_out, "k_out");
  }
  check_cuda_int(row_valid, "row_valid");
  const int64_t site_rows = row_valid.numel();
  const SlotStride stride = slot_stride(site_rows, legs_in, legs_out, owners, slots, slot_local);
  const int slot_rows = checked_int(stride.rows, "rows_per_slot");
  check_index(idx_in, slot_rows, "idx_in");
  check_index(idx_out, slot_rows, "idx_out");
  check_index(idx_s, slot_rows, "idx_s");
  JoinInputs<SinkT> in{};
  in.tau_in = tau_in.const_data_ptr<float>();
  in.tau_out = tau_out.const_data_ptr<float>();
  in.field_in = field_in.const_data_ptr<float>();
  in.field_out = field_out.const_data_ptr<float>();
  in.k_in = width == 4 ? k_in.const_data_ptr<float>() : nullptr;
  in.k_out = width == 4 ? k_out.const_data_ptr<float>() : nullptr;
  in.geometric = waves.numel() != 0;
  in.geo_in = leg_geometry<SinkT>(src_in, sink_in, points_in, depth_in, length_in, legs_in, "inbound", in.geometric);
  in.geo_out =
      leg_geometry<SinkT>(src_out, sink_out, points_out, depth_out, length_out, legs_out, "outbound", in.geometric);
  in.response = {response.const_data_ptr<float>(), width == 4 ? up.const_data_ptr<float>() : nullptr,
                 static_cast<int>(width)};
  in.row_valid = row_valid.const_data_ptr<int32_t>();
  in.idx_in = idx_in.const_data_ptr<int64_t>();
  in.idx_out = idx_out.const_data_ptr<int64_t>();
  in.idx_s = idx_s.const_data_ptr<int64_t>();
  // One polarization table, [pol_in | pol_out | pol_d_src | pol_d_sink], or empty.
  const int64_t direct_rows = field_d.numel() / 6;
  const float* pol = polarization_rows(polarization, legs_in + legs_out + 2 * direct_rows, "polarization");
  in.pol_in = in.geometric && pol != nullptr ? pol : nullptr;
  in.pol_out = in.geometric && pol != nullptr ? pol + 3 * legs_in : nullptr;
  STD_TORCH_CHECK(columns >= 1 && site_rows % columns == 0, "the site rows split into whole frequency columns.");
  in.columns = columns;
  if (waves.numel() != 0) {
    STD_TORCH_CHECK(
        waves.is_cuda() && waves.is_contiguous() && waves.scalar_type() == torch::headeronly::ScalarType::Double &&
            waves.numel() == 2 * columns,
        "waves must be a float64 CUDA table of (k64, k32) per frequency column.");
    in.waves = waves.const_data_ptr<double>();
  } else {
    in.waves = nullptr;
  }
  in.stride = stride;
  // The site-less leg.
  const int64_t direct = field_d.numel() / 6;
  check_float_len(field_d, 6 * direct, "field_d");
  in.direct_total = direct;
  in.site_total = site_rows;
  if (direct != 0) {
    STD_TORCH_CHECK(
        direct % slots == 0 && direct % columns == 0,
        "a slot-major site-less leg carries the same rows in every slot and frequency column.");
    in.direct_per_slot = direct / slots;
    in.field_d = field_d.const_data_ptr<float>();
    STD_TORCH_CHECK(in.geometric, "a site-less leg is composed from Channel's geometry and needs wave numbers.");
    in.geo_d = leg_geometry<float>(src_d, sink_d, points_d, depth_d, length_d, direct, "direct", true);
    STD_TORCH_CHECK(pol != nullptr, "a site-less row is read along the receiver's polarization.");
    in.pol_d_src = pol + 3 * (legs_in + legs_out);
    in.pol_d_sink = in.pol_d_src + 3 * direct;
    check_cuda_int(valid_d, "valid_d");
    STD_TORCH_CHECK(valid_d.numel() == direct, "valid_d must hold one flag per site-less row.");
    in.valid_d = valid_d.const_data_ptr<int32_t>();
  } else {
    in.direct_per_slot = 1;
    in.geo_d = {nullptr, nullptr, nullptr, nullptr, nullptr, 0, false};
  }
  const int64_t composed_per_slot = stride.rows + in.direct_per_slot * (direct != 0 ? 1 : 0);
  in.paths = composed_per_slot;
  STD_TORCH_CHECK(num_rows == composed_per_slot * slots, "num_rows must be every slot's composed rows.");
  if (place.numel() != 0) {
    check_cuda_long(place, "place");
    STD_TORCH_CHECK(place.numel() == composed_per_slot, "place must hold one slot's composed positions.");
    in.place = place.const_data_ptr<int64_t>();
  } else {
    STD_TORCH_CHECK(direct == 0, "a join with site-less rows needs its placement table.");
    in.place = nullptr;
  }
  return in;
}

template <typename SinkT>
void forward_impl(JOIN_INPUT_PARAMS, torch::stable::Tensor& tau_hi, torch::stable::Tensor& tau_lo,
                  torch::stable::Tensor& value, JOIN_SCALAR_PARAMS) {
  const int64_t num_rows = tau_hi.numel();
  const JoinInputs<SinkT> in = join_inputs<SinkT>(JOIN_INPUT_ARGS);
  check_float_len(tau_hi, num_rows, "tau_hi");
  check_float_len(tau_lo, num_rows, "tau_lo");
  check_float_len(value, 2 * num_rows, "value");
  const int threads = checked_int(in.site_total + in.direct_total, "join threads");
  if (threads == 0) return;
  const torch::stable::accelerator::DeviceGuard device_guard(tau_hi.get_device_index());
  two_way_join_forward_kernel<SinkT><<<linear_grid(threads), dim3(kBlock, 1, 1), 0, current_cuda_stream(tau_hi)>>>(
      in, tau_hi.mutable_data_ptr<float>(), tau_lo.mutable_data_ptr<float>(), value.mutable_data_ptr<float>());
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SinkT>
LegTangent<SinkT> leg_tangent_view(
    const LegGeometry<SinkT>& leg, const torch::stable::Tensor& src, const torch::stable::Tensor& sink,
    const torch::stable::Tensor& points, const torch::stable::Tensor& length32, int64_t legs) {
  if (!leg.on || legs == 0) return {nullptr, nullptr, nullptr, nullptr};
  check_float_len(src, 3 * legs, "tangent departure_origin_m");
  check_sink(sink, legs, "tangent arrival_target_m");
  STD_TORCH_CHECK(sink.scalar_type() == (sizeof(SinkT) == 8 ? torch::headeronly::ScalarType::Double
                                                              : torch::headeronly::ScalarType::Float),
                  "a sink tangent has its primal's dtype.");
  check_float_len(points, 3 * legs * leg.width, "tangent interaction_positions_m");
  check_float_len(length32, legs, "tangent path_length_m");
  return {src.const_data_ptr<float>(), sink.const_data_ptr<SinkT>(), points.const_data_ptr<float>(),
          length32.const_data_ptr<float>()};
}

template <typename SinkT>
void jvp_impl(JOIN_INPUT_PARAMS, const torch::stable::Tensor& tan_tau_in, const torch::stable::Tensor& tan_tau_out,
              const torch::stable::Tensor& tan_field_in, const torch::stable::Tensor& tan_field_out,
              const torch::stable::Tensor& tan_k_in, const torch::stable::Tensor& tan_k_out,
              const torch::stable::Tensor& tan_src_in, const torch::stable::Tensor& tan_sink_in,
              const torch::stable::Tensor& tan_points_in, const torch::stable::Tensor& tan_length_in,
              const torch::stable::Tensor& tan_src_out, const torch::stable::Tensor& tan_sink_out,
              const torch::stable::Tensor& tan_points_out, const torch::stable::Tensor& tan_length_out,
              const torch::stable::Tensor& tan_response, const torch::stable::Tensor& tan_field_d,
              const torch::stable::Tensor& tan_src_d, const torch::stable::Tensor& tan_sink_d,
              const torch::stable::Tensor& tan_points_d, const torch::stable::Tensor& tan_length_d,
              torch::stable::Tensor& tan_tau, torch::stable::Tensor& tan_value, JOIN_SCALAR_PARAMS) {
  const int64_t num_rows = tan_tau.numel();
  const JoinInputs<SinkT> in = join_inputs<SinkT>(JOIN_INPUT_ARGS);
  check_float_len(tan_tau_in, tau_in.numel(), "tan_tau_in");
  check_float_len(tan_tau_out, tau_out.numel(), "tan_tau_out");
  check_float_len(tan_field_in, field_in.numel(), "tan_field_in");
  check_float_len(tan_field_out, field_out.numel(), "tan_field_out");
  check_float_len(tan_response, response.numel(), "tan_response");
  if (response_width == 4) {
    check_float_len(tan_k_in, k_in.numel(), "tan_k_in");
    check_float_len(tan_k_out, k_out.numel(), "tan_k_out");
  }
  const LegTangent<SinkT> t_in =
      leg_tangent_view(in.geo_in, tan_src_in, tan_sink_in, tan_points_in, tan_length_in, tau_in.numel());
  const LegTangent<SinkT> t_out =
      leg_tangent_view(in.geo_out, tan_src_out, tan_sink_out, tan_points_out, tan_length_out, tau_out.numel());
  check_float_len(tan_field_d, field_d.numel(), "tan_field_d");
  const LegTangent<float> t_d =
      leg_tangent_view(in.geo_d, tan_src_d, tan_sink_d, tan_points_d, tan_length_d, in.direct_total);
  check_float_len(tan_tau, num_rows, "tan_tau");
  check_float_len(tan_value, 2 * num_rows, "tan_value");
  const int threads = checked_int(in.site_total + in.direct_total, "join threads");
  if (threads == 0) return;
  const torch::stable::accelerator::DeviceGuard device_guard(tan_tau.get_device_index());
  two_way_join_jvp_kernel<SinkT><<<linear_grid(threads), dim3(kBlock, 1, 1), 0, current_cuda_stream(tan_tau)>>>(
      in, tan_tau_in.const_data_ptr<float>(), tan_tau_out.const_data_ptr<float>(),
      tan_field_in.const_data_ptr<float>(), tan_field_out.const_data_ptr<float>(),
      response_width == 4 ? tan_k_in.const_data_ptr<float>() : nullptr,
      response_width == 4 ? tan_k_out.const_data_ptr<float>() : nullptr, t_in, t_out,
      tan_response.const_data_ptr<float>(), tan_field_d.const_data_ptr<float>(), t_d,
      tan_tau.mutable_data_ptr<float>(), tan_value.mutable_data_ptr<float>());
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SinkT>
LegGradients<SinkT> leg_gradients(
    const LegGeometry<SinkT>& leg, torch::stable::Tensor& tau, torch::stable::Tensor& field,
    torch::stable::Tensor& k, torch::stable::Tensor& src, torch::stable::Tensor& sink,
    torch::stable::Tensor& points, torch::stable::Tensor& length32, int64_t legs, int64_t width) {
  check_float_len(tau, legs, "grad_tau");
  check_float_len(field, 6 * legs, "grad_field");
  LegGradients<SinkT> out{tau.mutable_data_ptr<float>(), field.mutable_data_ptr<float>(), nullptr, nullptr,
                          nullptr, nullptr, nullptr};
  if (width == 4) {
    check_float_len(k, 3 * legs, "grad_k");
    out.k = k.mutable_data_ptr<float>();
  }
  if (leg.on && legs != 0) {
    check_float_len(src, 3 * legs, "grad_departure_origin_m");
    check_sink(sink, legs, "grad_arrival_target_m");
    check_float_len(points, 3 * legs * leg.width, "grad_interaction_positions_m");
    check_float_len(length32, legs, "grad_path_length_m");
    out.src = src.mutable_data_ptr<float>();
    out.sink = sink.mutable_data_ptr<SinkT>();
    out.points = points.mutable_data_ptr<float>();
    out.length32 = length32.mutable_data_ptr<float>();
  }
  return out;
}

template <typename SinkT>
void backward_impl(JOIN_INPUT_PARAMS, const torch::stable::Tensor& by_in_offsets,
                   const torch::stable::Tensor& by_in_rows, const torch::stable::Tensor& by_out_offsets,
                   const torch::stable::Tensor& by_out_rows, const torch::stable::Tensor& by_s_offsets,
                   const torch::stable::Tensor& by_s_rows, const torch::stable::Tensor& grad_tau,
                   const torch::stable::Tensor& grad_value, torch::stable::Tensor& grad_tau_in,
                   torch::stable::Tensor& grad_tau_out, torch::stable::Tensor& grad_field_in,
                   torch::stable::Tensor& grad_field_out, torch::stable::Tensor& grad_k_in,
                   torch::stable::Tensor& grad_k_out, torch::stable::Tensor& grad_src_in,
                   torch::stable::Tensor& grad_sink_in, torch::stable::Tensor& grad_points_in,
                   torch::stable::Tensor& grad_length_in, torch::stable::Tensor& grad_src_out,
                   torch::stable::Tensor& grad_sink_out, torch::stable::Tensor& grad_points_out,
                   torch::stable::Tensor& grad_length_out, torch::stable::Tensor& grad_response,
                   torch::stable::Tensor& grad_field_d, torch::stable::Tensor& grad_src_d,
                   torch::stable::Tensor& grad_sink_d, torch::stable::Tensor& grad_points_d,
                   torch::stable::Tensor& grad_length_d, JOIN_SCALAR_PARAMS) {
  const int64_t num_rows = grad_tau.numel();
  const JoinInputs<SinkT> in = join_inputs<SinkT>(JOIN_INPUT_ARGS);
  const int legs_in = checked_int(tau_in.numel(), "num_in");
  const int legs_out = checked_int(tau_out.numel(), "num_out");
  const int owners = checked_int(response.numel() / (2 * response_width), "num_owners");
  check_cuda_long(by_in_offsets, "by_in_offsets");
  check_cuda_long(by_in_rows, "by_in_rows");
  check_cuda_long(by_out_offsets, "by_out_offsets");
  check_cuda_long(by_out_rows, "by_out_rows");
  check_cuda_long(by_s_offsets, "by_s_offsets");
  check_cuda_long(by_s_rows, "by_s_rows");
  const SlotStride stride = in.stride;
  const int64_t site_owners = stride.sites == 0 ? owners : stride.sites;
  STD_TORCH_CHECK(
      by_in_offsets.numel() == stride.in + 1 && by_out_offsets.numel() == stride.out + 1 &&
          by_s_offsets.numel() == site_owners + 1,
      "each CSR offsets table must hold one entry per owner of one slot plus one.");
  STD_TORCH_CHECK(
      by_in_rows.numel() == stride.rows && by_out_rows.numel() == stride.rows && by_s_rows.numel() == stride.rows,
      "each CSR row table must permute one slot's composed site rows exactly once.");
  check_float_len(grad_tau, num_rows, "grad_tau");
  check_float_len(grad_value, 2 * num_rows, "grad_value");
  check_float_len(grad_response, response.numel(), "grad_response");
  const LegGradients<SinkT> g_in = leg_gradients(in.geo_in, grad_tau_in, grad_field_in, grad_k_in, grad_src_in,
                                                 grad_sink_in, grad_points_in, grad_length_in, legs_in,
                                                 response_width);
  const LegGradients<SinkT> g_out = leg_gradients(in.geo_out, grad_tau_out, grad_field_out, grad_k_out,
                                                  grad_src_out, grad_sink_out, grad_points_out, grad_length_out,
                                                  legs_out, response_width);
  LegGradients<float> g_d{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  if (in.direct_total != 0) {
    check_float_len(grad_field_d, 6 * in.direct_total, "grad_field_d");
    check_float_len(grad_src_d, 3 * in.direct_total, "grad_src_d");
    check_float_len(grad_sink_d, 3 * in.direct_total, "grad_sink_d");
    check_float_len(grad_points_d, 3 * in.direct_total * in.geo_d.width, "grad_points_d");
    check_float_len(grad_length_d, in.direct_total, "grad_length_d");
    g_d = {nullptr, grad_field_d.mutable_data_ptr<float>(), nullptr,
           grad_src_d.mutable_data_ptr<float>(), grad_sink_d.mutable_data_ptr<float>(),
           grad_points_d.mutable_data_ptr<float>(), grad_length_d.mutable_data_ptr<float>()};
  }
  const int total = checked_int(static_cast<int64_t>(legs_in) + legs_out + owners + in.direct_total, "join threads");
  if (total == 0) return;
  const torch::stable::accelerator::DeviceGuard device_guard(grad_response.get_device_index());
  two_way_join_backward_kernel<SinkT>
      <<<linear_grid(total), dim3(kBlock, 1, 1), 0, current_cuda_stream(grad_response)>>>(
          in, by_in_offsets.const_data_ptr<int64_t>(), by_in_rows.const_data_ptr<int64_t>(),
          by_out_offsets.const_data_ptr<int64_t>(), by_out_rows.const_data_ptr<int64_t>(),
          by_s_offsets.const_data_ptr<int64_t>(), by_s_rows.const_data_ptr<int64_t>(),
          grad_tau.const_data_ptr<float>(), grad_value.const_data_ptr<float>(), g_in, g_out,
          grad_response.mutable_data_ptr<float>(), g_d, legs_in, legs_out, owners, slots);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// The site precision of the join: float64 when the binding authored float64
// sites, float32 otherwise (a leg with no rows carries an empty site table).
#define JOIN_DISPATCH(impl, ...)                                                      \
  if ((sink_in.numel() != 0 && sink_is_double(sink_in)) ||                            \
      (sink_out.numel() != 0 && sink_is_double(sink_out))) {                          \
    impl<double>(__VA_ARGS__);                                                        \
  } else {                                                                            \
    impl<float>(__VA_ARGS__);                                                         \
  }

#define JOIN_TENSORS                                                                                           \
  tau_in, tau_out, field_in, field_out, k_in, k_out, src_in, sink_in, points_in, depth_in, length_in, src_out, \
      sink_out, points_out, depth_out, length_out, response, up, polarization, field_d, src_d, sink_d,          \
      points_d, depth_d, length_d, valid_d, waves, place, row_valid, idx_in, idx_out, idx_s

#define JOIN_SCALARS response_width, slots, slot_local, columns

void two_way_join_forward_cuda(JOIN_INPUT_PARAMS, torch::stable::Tensor& tau_hi, torch::stable::Tensor& tau_lo,
                               torch::stable::Tensor& value, JOIN_SCALAR_PARAMS) {
  JOIN_DISPATCH(forward_impl, JOIN_TENSORS, tau_hi, tau_lo, value, JOIN_SCALARS)
}

void two_way_join_jvp_cuda(
    JOIN_INPUT_PARAMS, const torch::stable::Tensor& tan_tau_in, const torch::stable::Tensor& tan_tau_out,
    const torch::stable::Tensor& tan_field_in, const torch::stable::Tensor& tan_field_out,
    const torch::stable::Tensor& tan_k_in, const torch::stable::Tensor& tan_k_out,
    const torch::stable::Tensor& tan_src_in, const torch::stable::Tensor& tan_sink_in,
    const torch::stable::Tensor& tan_points_in, const torch::stable::Tensor& tan_length_in,
    const torch::stable::Tensor& tan_src_out, const torch::stable::Tensor& tan_sink_out,
    const torch::stable::Tensor& tan_points_out, const torch::stable::Tensor& tan_length_out,
    const torch::stable::Tensor& tan_response, const torch::stable::Tensor& tan_field_d,
    const torch::stable::Tensor& tan_src_d, const torch::stable::Tensor& tan_sink_d,
    const torch::stable::Tensor& tan_points_d, const torch::stable::Tensor& tan_length_d,
    torch::stable::Tensor& tan_tau, torch::stable::Tensor& tan_value, JOIN_SCALAR_PARAMS) {
  JOIN_DISPATCH(jvp_impl, JOIN_TENSORS, tan_tau_in, tan_tau_out, tan_field_in, tan_field_out, tan_k_in, tan_k_out,
                tan_src_in, tan_sink_in, tan_points_in, tan_length_in, tan_src_out, tan_sink_out, tan_points_out,
                tan_length_out, tan_response, tan_field_d, tan_src_d, tan_sink_d, tan_points_d, tan_length_d, tan_tau,
                tan_value, JOIN_SCALARS)
}

void two_way_join_backward_cuda(
    JOIN_INPUT_PARAMS, const torch::stable::Tensor& by_in_offsets, const torch::stable::Tensor& by_in_rows,
    const torch::stable::Tensor& by_out_offsets, const torch::stable::Tensor& by_out_rows,
    const torch::stable::Tensor& by_s_offsets, const torch::stable::Tensor& by_s_rows,
    const torch::stable::Tensor& grad_tau, const torch::stable::Tensor& grad_value, torch::stable::Tensor& grad_tau_in,
    torch::stable::Tensor& grad_tau_out, torch::stable::Tensor& grad_field_in, torch::stable::Tensor& grad_field_out,
    torch::stable::Tensor& grad_k_in, torch::stable::Tensor& grad_k_out, torch::stable::Tensor& grad_src_in,
    torch::stable::Tensor& grad_sink_in, torch::stable::Tensor& grad_points_in, torch::stable::Tensor& grad_length_in,
    torch::stable::Tensor& grad_src_out, torch::stable::Tensor& grad_sink_out, torch::stable::Tensor& grad_points_out,
    torch::stable::Tensor& grad_length_out, torch::stable::Tensor& grad_response, torch::stable::Tensor& grad_field_d,
    torch::stable::Tensor& grad_src_d, torch::stable::Tensor& grad_sink_d, torch::stable::Tensor& grad_points_d,
    torch::stable::Tensor& grad_length_d, JOIN_SCALAR_PARAMS) {
  JOIN_DISPATCH(backward_impl, JOIN_TENSORS, by_in_offsets, by_in_rows, by_out_offsets, by_out_rows, by_s_offsets,
                by_s_rows, grad_tau, grad_value, grad_tau_in, grad_tau_out, grad_field_in, grad_field_out, grad_k_in,
                grad_k_out, grad_src_in, grad_sink_in, grad_points_in, grad_length_in, grad_src_out, grad_sink_out,
                grad_points_out, grad_length_out, grad_response, grad_field_d, grad_src_d, grad_sink_d, grad_points_d,
                grad_length_d, JOIN_SCALARS)
}

// Carrier-aware interpolation of a single IDENTICAL round-trip path through K
// sampled nodes. Row i of x is K groups of four doubles: (tau_j [s], Re(C_j),
// Im(C_j), w_j). fc is Hz. Channel convention C = A exp(-j 2 pi fc tau).
//
//   tau   = sum_j w_j tau_j
//   theta = -2 pi fc (tau - tau_j)
//   C     = sum_j w_j C_j exp(j theta_j)
//
// Each node coefficient is transported to the query delay BEFORE the envelopes
// are blended, which is what avoids cancellation across carrier wraps; the
// weights blend envelopes only. w is the caller's Lagrange basis at the query
// time, so K=2 with w=(1-alpha, alpha) is the linear rule and the polynomial
// order is the caller's choice. The weights must sum to one - a partition of
// unity is what makes an exactly-sampled query reproduce its node - and they
// are a discrete time-grid decision, so the kernel publishes NO derivative for
// them. Valid only within a topology-stable, error-tested interval. No
// spreading/scattering is added here. Oracle: test_path_interpolation.py.
template<bool Indexed>
__device__ InterpolationNode interpolation_node(const double* x, const int64_t* indices,
    const double* weights, int64_t i, int64_t j, int64_t nodes, int64_t samples) {
  if constexpr(Indexed) {
    const int64_t row=indices[i*nodes+j];
    // Routing is discrete metadata. Do not let a malformed native caller read
    // outside the compact table; Python's partition builder supplies valid IDs.
    if(row<0 || row>=samples) return {0,0,0,0};
    const double* a=x+3*row;
    return {a[0],a[1],a[2],weights[i*nodes+j]};
  } else {
    const double* a=x+4*(nodes*i+j);
    return {a[0],a[1],a[2],a[3]};
  }
}

template<bool Indexed>
struct PackedPathNodes {
  const double* x;
  const int64_t* indices;
  const double* weights;
  int64_t i,nodes,samples;
  __device__ InterpolationNode operator()(int64_t j) const {
    return interpolation_node<Indexed>(x,indices,weights,i,j,nodes,samples);
  }
};

template<bool Indexed>
__global__ void path_interpolate_kernel(const double* x, const int64_t* indices,
    const double* weights, const double* v, double* out, int64_t n,
    int64_t nodes, int64_t samples, double fc, int mode) {
  const int64_t i = static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i >= n) return;
  const PackedPathNodes<Indexed> read{x,indices,weights,i,nodes,samples};
  const auto value=interpolate_path(read,nodes,fc);
  if (mode==0) { out[3*i]=value.tau; out[3*i+1]=value.re; out[3*i+2]=value.im; }
  else {
    // Column m of the Jacobian, for node m. d(theta_j)/d(tau_m) = -k(w_m -
    // delta_jm) collapses the transported sum into the published re/im, so a
    // delay column needs only this node's own transported value.
    double* row = out+(mode==1 ? 4*nodes*i : 3*i);
    const double* vec = mode==1 ? v+3*i : v;
    double jvp[3] = {0,0,0};
    for (int64_t j=0;j<nodes;++j) {
      double jac[3][3];
      interpolation_jacobian(read(j),value,fc,jac);
      if (mode==1) {
        for(int col=0;col<3;++col) {
          double sum=0; for(int r=0;r<3;++r) sum+=jac[r][col]*vec[r];
          row[4*j+col]=sum;
        }
        row[4*j+3]=0;
      } else {
        const auto d=interpolation_node<Indexed>(vec,indices,weights,i,j,nodes,samples);
        const double direction[3]={d.tau,d.re,d.im};
        for(int r=0;r<3;++r) for(int col=0;col<3;++col) jvp[r]+=jac[r][col]*direction[col];
      }
    }
    if (mode==2) for(int r=0;r<3;++r) row[r]=jvp[r];
  }
}

void path_interpolate_run(const torch::stable::Tensor& x, const torch::stable::Tensor& indices,
    const torch::stable::Tensor& weights, const torch::stable::Tensor& v,
    torch::stable::Tensor& out, double fc, int mode) {
  const torch::stable::Tensor* tensors[4] = {&x,&weights,&v,&out};
  for (const auto* tensor : tensors) {
    STD_TORCH_CHECK(tensor->is_cuda() && tensor->is_contiguous() &&
      tensor->scalar_type()==torch::headeronly::ScalarType::Double,
      "path interpolation requires contiguous CUDA doubles");
    STD_TORCH_CHECK(tensor->get_device_index()==x.get_device_index(), "interpolation devices differ");
  }
  check_cuda_long(indices,"interpolation indices");
  STD_TORCH_CHECK(indices.get_device_index()==x.get_device_index(), "interpolation index device differs");
  const bool indexed=indices.dim()==2;
  STD_TORCH_CHECK(x.dim()==2, "interpolation x must be a matrix");
  STD_TORCH_CHECK(indexed ? x.size(1)==3 && indices.size(1)>=2 :
    indices.dim()==1 && indices.numel()==0 && x.size(1)>=8 && x.size(1)%4==0,
    "interpolation needs compact [S,3] with [N,K] indices, or packed [N,4K], K>=2");
  const int64_t n=indexed ? indices.size(0) : x.size(0);
  const int64_t nodes=indexed ? indices.size(1) : x.size(1)/4, wide=4*nodes;
  STD_TORCH_CHECK(indexed ? weights.dim()==2 && weights.size(0)==n && weights.size(1)==nodes : weights.numel()==0,
    "interpolation basis shape differs from indices");
  STD_TORCH_CHECK(out.numel()==n*(mode==1?wide:3), "interpolation output shape");
  STD_TORCH_CHECK(v.numel()==(mode==0?0:mode==1?n*3:x.numel()), "interpolation vector shape");
  STD_TORCH_CHECK(std::isfinite(fc) && fc>0, "interpolation carrier must be positive");
  const torch::stable::accelerator::DeviceGuard guard(x.get_device_index());
  if(n) {
    const auto stream=current_cuda_stream(x);
    if(indexed) path_interpolate_kernel<true><<<(n+255)/256,256,0,stream>>>(
      x.const_data_ptr<double>(),indices.const_data_ptr<int64_t>(),weights.const_data_ptr<double>(),
      v.const_data_ptr<double>(),out.mutable_data_ptr<double>(),n,nodes,x.size(0),fc,mode);
    else path_interpolate_kernel<false><<<(n+255)/256,256,0,stream>>>(
      x.const_data_ptr<double>(),indices.const_data_ptr<int64_t>(),weights.const_data_ptr<double>(),
      v.const_data_ptr<double>(),out.mutable_data_ptr<double>(),n,nodes,x.size(0),fc,mode);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
void path_interpolate_forward(const torch::stable::Tensor& x, const torch::stable::Tensor& indices,
    const torch::stable::Tensor& weights, const torch::stable::Tensor& v,
    torch::stable::Tensor& out, double fc) { path_interpolate_run(x,indices,weights,v,out,fc,0); }
void path_interpolate_backward(const torch::stable::Tensor& x, const torch::stable::Tensor& indices,
    const torch::stable::Tensor& weights, const torch::stable::Tensor& v,
    torch::stable::Tensor& out, double fc) { path_interpolate_run(x,indices,weights,v,out,fc,1); }
void path_interpolate_jvp(const torch::stable::Tensor& x, const torch::stable::Tensor& indices,
    const torch::stable::Tensor& weights, const torch::stable::Tensor& v,
    torch::stable::Tensor& out, double fc) { path_interpolate_run(x,indices,weights,v,out,fc,2); }

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("path_interpolate_forward", TORCH_BOX(&path_interpolate_forward));
  m.impl("path_interpolate_backward", TORCH_BOX(&path_interpolate_backward));
  m.impl("path_interpolate_jvp", TORCH_BOX(&path_interpolate_jvp));
  m.impl("two_way_join_forward", TORCH_BOX(&two_way_join_forward_cuda));
  m.impl("two_way_join_backward", TORCH_BOX(&two_way_join_backward_cuda));
  m.impl("two_way_join_jvp", TORCH_BOX(&two_way_join_jvp_cuda));
}
