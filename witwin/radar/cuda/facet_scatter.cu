// Physical-optics facet scattering: a mesh target's facets as scatter sites.
//
// Two operator families.
//
// facet_geometry gathers one frame's facet corners from the deformed world
// vertices and places each scatter site, in float64, at the centroid of its
// anchor facet, the first of its members (the host orders each site's facets so
// that the anchor is the member nearest their mean, which keeps the site ON the
// surface: a mean of a curved patch's centroids lies inside a closed body,
// where every leg to it is occluded). Both outputs are linear in the vertices,
// so the forward-mode companion is the forward operator applied to the vertex
// tangent and the reverse companion is its transpose, reduced per vertex
// through a frozen vertex-incidence CSR: one thread owns one vertex, no atomics.
//
// facet_response evaluates one 2x2 scattering matrix per COMPOSED row, in the
// join's site basis (two_way_join.cu, R-ADR-041): [HH, HV, VH, VV], receive
// index first, for the up axis it also publishes per row. The matrix is the
// projection S_ab = b_r,a . D . b_i,b of the coherent sum over the row's
// site's facets f of their scattering dyadics
//
//   D_f = j (k^2 / pi) A_f chi_f F_f G_f exp(j k q . (c_f - x_s))
//       + k sqrt(sigma_d,f / pi) exp(-j psi_f) exp(j k q . (c_f - x_s)) I
//
// with, all lengths in metres and k = 2 pi / lambda the carrier wavenumber:
//
//   k_i, k_r                   the inbound and outbound legs' final-segment
//                              directions INTO the site (both legs run from a
//                              radar element toward the site, R-ADR-039)
//   n, A_f                     unit normal (right-handed corner order) and area
//   ci = -n.k_i, co = -n.k_r   both must be positive, otherwise D_f = 0: a facet
//                              is lit and seen from its normal side only
//   q  = -(k_i + k_r)          the bistatic phase vector
//   F  = (1/A) int_T exp(j k q.(r - c_f)) dA, the exact flat-facet form factor
//   G  = -(r_TE R te te^T + r_TM R tm tm^T), the facet's polarimetric law:
//        te = (n x k_i)/|n x k_i| and tm = te x k_i the incident TE and TM
//        directions, r_TE and r_TM the material's tangential-field reflection
//        coefficients at the incidence angle (stack_reflection; both -1 on a
//        conductor), and R x = (n . k_i) x - k_i (n . x) the physical-optics
//        radiation of the surface current 2 n x H a field x induces, taken
//        transverse to the scattered direction by the basis projection. For a
//        conductor G = R on the incident transverse plane, the vector PO
//        dyadic; monostatic it is (n . k_i) I = -cos(theta) I.
//   chi = exp(-(k h (ci + co))^2 / 2), the coherent (Ament) roughness factor of
//                              a Gaussian surface of rms height h
//   sigma_d = 4 A f_d ci co,   f_d = (1 - exp(-(2 k h ci)^2)) (|r_TE|^2 + |r_TM|^2) / 2,
//                              the Lambertian diffuse cross section that carries
//                              the power the coherent term loses, scattered as
//                              the identity dyadic
//   psi_f                      a uniform phase drawn once per facet from the
//                              declared seed: a frozen surface realisation
//
// The normalisation is the two-way join's: a scalar response s is s I, and a
// broadside conductor plate reads D = -j (4 pi A / lambda^2) I, so
// sigma = 4 pi A^2 / lambda^2. The factor j is the Huygens kernel's phase in the
// exp(+j omega t), exp(-j k d) convention: a stationary-phase evaluation over a
// convex surface contributes -j, so a PEC sphere returns -sqrt(pi) a with the
// phase of its front point. The phase reference of every facet is the site
// position the legs used; exp(j k q . (c_f - x_s)) moves each facet's
// reference to its centroid under the far-field-of-the-cluster approximation
// and is exactly 1 for a site of one facet.
//
// The basis is the join's own: b = (h, v), h = (u x k)/|u x k|, v = h x k for
// k = k_i (incident) and k = k_r (receive), with the published up axis
// u = z and the join's fallback u' = x (or y when k is along x) where u is
// parallel to k. The join rebuilds sum_ab S_ab b_r,a b_i,b^T, the projection of
// D onto the two transverse planes, which does not depend on u: the up axis is a
// bookkeeping choice, not physics, and takes no derivative.
//
// SLOT STRIDE. A slot-major replay of G instants composes G copies of one
// join. idx_in, idx_out and idx_site describe ONE slot of rows_per_slot rows;
// row k of the batch is join row k % rows_per_slot of slot g, its leg rows are
// offset by g times the legs' per-slot row counts, and the facets and sites
// are shared by every slot. The VJP walks each owner's per-slot CSR once per
// slot in slot order, which is the summation order of the per-slot table
// copies, so the strided route is bit-identical to them.
//
// Derivatives are exact forward-mode jets (value plus N directional
// derivatives) carried through the same expression the forward evaluates. The
// two special functions - the form factor F(a, b) and the layer-stack
// reflection - are evaluated once on a small inner jet over their own
// arguments and chained into the outer jet. The VJP owns one gradient slot per
// thread over four owner families laid end to end: inbound leg rows (k_i),
// outbound leg rows (k_r), facets (corners) and sites (the float64 site
// position in the phase offset), reduced through the join's own frozen CSR.
//
// Everything accumulates in double and stores float32 (the site gradient in
// float64). A dead row publishes exactly zero. The material table, the
// wavenumber and the seed take no derivative: they select the law.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cuda_runtime.h>
#include "radar_checks.cuh"
#include "scatter_jets.cuh"

#include <cstdint>

namespace {

using namespace radar_jets;

constexpr double kPi = kJetPi;

// ---------------------------------------------------------------------------
// The flat-triangle form factor.
// ---------------------------------------------------------------------------
//
// With r = v0 + s e1 + t e2 over the unit simplex and a = k q.e1, b = k q.e2,
// (1/A) int_T exp(j k q.(r - c)) dA = 2 g[p0, p1, p2], the second divided
// difference of g(x) = -exp(j x) at the centred nodes p0 = -(a + b)/3,
// p1 = p0 + a, p2 = p0 + b (Hermite-Genocchi). It is 1 at a = b = 0. Close
// nodes take the Taylor series sum_m j^m h_m(p) / (m + 2)!, h_m the complete
// homogeneous symmetric polynomial; well separated nodes take the recursive
// difference over the widest pair, which divides by at least 1.5.

// g[x, y] = -j exp(j (x + y)/2) sinc((y - x)/2).
template <int N>
__device__ __forceinline__ C<N> first_difference(const J<N>& x, const J<N>& y) {
  const C<N> phase = cis((x + y) * 0.5);
  return times_j(phase * jsinc((y - x) * 0.5)) * -1.0;
}

__device__ __forceinline__ void form_factor(
    double a, double b, double* f_re, double* f_im, double* fa_re, double* fa_im, double* fb_re,
    double* fb_im) {
  J<2> ja = jconst<2>(a);
  J<2> jb = jconst<2>(b);
  ja.d[0] = 1.0;
  jb.d[1] = 1.0;
  const J<2> p0 = (ja + jb) * (-1.0 / 3.0);
  const J<2> p1 = p0 + ja;
  const J<2> p2 = p0 + jb;
  const double s01 = fabs(a);
  const double s02 = fabs(b);
  const double s12 = fabs(b - a);
  const double widest = fmax(s01, fmax(s02, s12));
  C<2> g;
  if (widest < 1.5) {
    // h_m over three nodes by the recurrence h_m(x..z) = h_m(x..y) + z h_{m-1}(x..z).
    J<2> h0 = jconst<2>(1.0);
    J<2> h1 = jconst<2>(1.0);
    J<2> h2 = jconst<2>(1.0);
    g = cconst<2>(0.5, 0.0);
    double factorial = 2.0;
    for (int m = 1; m <= 18; ++m) {
      h0 = h0 * p0;
      h1 = h0 + p1 * h1;
      h2 = h1 + p2 * h2;
      factorial *= static_cast<double>(m + 2);
      const J<2> term = h2 * (1.0 / factorial);
      switch (m & 3) {
        case 0: g.re = g.re + term; break;
        case 1: g.im = g.im + term; break;
        case 2: g.re = g.re - term; break;
        default: g.im = g.im - term; break;
      }
    }
  } else {
    J<2> x0 = p0;
    J<2> x1 = p1;
    J<2> x2 = p2;
    if (widest == s01) {
      x1 = p2;
      x2 = p1;
    } else if (widest == s12) {
      x0 = p1;
      x1 = p0;
    }
    const C<2> upper = first_difference(x1, x2);
    const C<2> lower = first_difference(x0, x1);
    const J<2> span = x2 - x0;
    g = {(upper.re - lower.re) / span, (upper.im - lower.im) / span};
  }
  *f_re = 2.0 * g.re.v;
  *f_im = 2.0 * g.im.v;
  *fa_re = 2.0 * g.re.d[0];
  *fa_im = 2.0 * g.im.d[0];
  *fb_re = 2.0 * g.re.d[1];
  *fb_im = 2.0 * g.im.d[1];
}

// ---------------------------------------------------------------------------
// Material reflection: a planar layer stack in vacuum, or a perfect conductor.
// ---------------------------------------------------------------------------
//
// The characteristic-matrix form Channel's production layer stack uses, in
// relative units: k_z = k0 sqrt(eps mu - sin^2), y_TE = k_z / mu,
// y_TM = eps / k_z (the common factors cancel in r), one-way phase
// delta = k0 sqrt(eps mu - sin^2) d. Each layer matrix is scaled by
// exp(Im delta) so a thick lossy layer cannot overflow; r is invariant to that
// scale. Both coefficients are in the tangential-field convention the
// admittances define, equal at normal incidence; r_TM is the negative of the
// textbook magnetic-field TM coefficient. material_params row:
// (model, rms_height_m); model 1 is a perfect conductor (r_TE = r_TM = -1). layers row: (thickness_m, eps_re, eps_im, mu_r) with
// eps_im <= 0.

struct Reflection {
  double te_re, te_im, te_slope_re, te_slope_im;
  double tm_re, tm_im, tm_slope_re, tm_slope_im;
};

__device__ __forceinline__ Reflection stack_reflection(
    double cosine, int model, const float* __restrict__ layers, int64_t begin, int64_t end,
    double k0) {
  Reflection out;
  if (model == 1) {
    out = {-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0};
    return out;
  }
  J<1> c = jconst<1>(cosine);
  c.d[0] = 1.0;
  const J<1> sin2 = jconst<1>(1.0) - c * c;
  // Vacuum outside and behind: k_z = cos, y_TE = cos, y_TM = 1 / cos.
  const C<1> kz_out = {c, jconst<1>(0.0)};
  const C<1> one = cconst<1>(1.0, 0.0);
  C<1> results[2];
  for (int pol = 0; pol < 2; ++pol) {
    const C<1> y_out = pol == 0 ? kz_out : one / kz_out;
    C<1> m11 = one;
    C<1> m12 = cconst<1>(0.0, 0.0);
    C<1> m21 = cconst<1>(0.0, 0.0);
    C<1> m22 = one;
    for (int64_t layer = begin; layer < end; ++layer) {
      const double thickness = layers[layer * 4];
      const C<1> eps = cconst<1>(layers[layer * 4 + 1], layers[layer * 4 + 2]);
      const double mu = layers[layer * 4 + 3];
      C<1> arg = eps * mu;
      arg.re = arg.re - sin2;
      const C<1> kz = csqrt_passive(arg);
      const C<1> y = pol == 0 ? kz * (1.0 / mu) : eps / kz;
      const C<1> delta = kz * (k0 * thickness);
      // cos and sin of delta, both scaled by exp(Im delta).
      const C<1> e_plus = cis(delta.re);
      const C<1> e_minus = cexp(C<1>{delta.im * 2.0, -delta.re});
      const C<1> cos_s = (e_plus + e_minus) * 0.5;
      const C<1> sin_s = times_j(e_minus - e_plus) * 0.5;  // (e+ - e-) / (2 j)
      const C<1> j_sin = times_j(sin_s);
      const C<1> l12 = j_sin / y;
      const C<1> l21 = j_sin * y;
      const C<1> n11 = m11 * cos_s + m12 * l21;
      const C<1> n12 = m11 * l12 + m12 * cos_s;
      const C<1> n21 = m21 * cos_s + m22 * l21;
      const C<1> n22 = m21 * l12 + m22 * cos_s;
      m11 = n11;
      m12 = n12;
      m21 = n21;
      m22 = n22;
    }
    const C<1> y_back = y_out;
    const C<1> b = m11 + m12 * y_back;
    const C<1> cc = m21 + m22 * y_back;
    results[pol] = (y_out * b - cc) / (y_out * b + cc);
  }
  out.te_re = results[0].re.v;
  out.te_im = results[0].im.v;
  out.te_slope_re = results[0].re.d[0];
  out.te_slope_im = results[0].im.d[0];
  out.tm_re = results[1].re.v;
  out.tm_im = results[1].im.v;
  out.tm_slope_re = results[1].re.d[0];
  out.tm_slope_im = results[1].im.d[0];
  return out;
}

// r_TE and r_TM at cosine x, chained into the outer jet.
template <int N>
__device__ __forceinline__ void fresnel(
    const J<N>& x, int model, const float* __restrict__ layers, int64_t begin, int64_t end, double k0, C<N>* te,
    C<N>* tm) {
  const Reflection r = stack_reflection(x.v, model, layers, begin, end, k0);
  *te = chain(x, r.te_re, r.te_im, r.te_slope_re, r.te_slope_im);
  *tm = chain(x, r.tm_re, r.tm_im, r.tm_slope_re, r.tm_slope_im);
}

// A uniform phase in [0, 2 pi) per (seed, facet): splitmix64 of the pair.
__device__ __forceinline__ double facet_phase(uint64_t seed, int64_t facet) {
  uint64_t z = seed * 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(facet) + 0x632BE59BD9B4E019ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  return kTwoPiD * static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
}

struct Law {
  const float* material_params;
  const float* layers;
  const int64_t* layer_offsets;
  const int64_t* facet_material;
  double k;
  uint64_t seed;
};

template <int N>
__device__ __forceinline__ V3<N> jnormalize(const V3<N>& a) {
  return scale(a, jconst<N>(1.0) / jsqrt(dot(a, a)));
}

// The join's site basis for propagation direction k and the published up axis
// z, with its fallback where z is parallel to k (two_way_join.cu site_basis).
template <int N>
__device__ __forceinline__ void site_basis(const V3<N>& k, V3<N>* h, V3<N>* v) {
  V3<N> u = {jconst<N>(0.0), jconst<N>(0.0), jconst<N>(1.0)};
  V3<N> w = cross(u, k);
  if (!(sqrt(dot(w, w).v) > 1e-6)) {
    u = fabs(k.x.v) < 0.9 ? V3<N>{jconst<N>(1.0), jconst<N>(0.0), jconst<N>(0.0)}
                          : V3<N>{jconst<N>(0.0), jconst<N>(1.0), jconst<N>(0.0)};
    w = cross(u, k);
  }
  *h = jnormalize(w);
  *v = cross(*h, k);
}

template <int N>
struct M4 {
  C<N> s[4];
};

template <int N>
__device__ __forceinline__ M4<N> m4_zero() {
  M4<N> m;
  for (int e = 0; e < 4; ++e) m.s[e] = cconst<N>(0.0, 0.0);
  return m;
}

// One facet's contribution to one row, as [HH, HV, VH, VV] in the row's site
// basis, over whatever lanes the caller seeded in its inputs.
template <int N>
__device__ __forceinline__ M4<N> facet_matrix(
    const V3<N>& k_in, const V3<N>& k_out, const V3<N>& v0, const V3<N>& v1, const V3<N>& v2,
    const V3<N>& site, int64_t facet, const Law& law) {
  const V3<N> e1 = v1 - v0;
  const V3<N> e2 = v2 - v0;
  const V3<N> normal_area = cross(e1, e2);
  const J<N> twice_area = jsqrt(dot(normal_area, normal_area));
  if (!(twice_area.v > 0.0)) return m4_zero<N>();
  const V3<N> n = scale(normal_area, jconst<N>(1.0) / twice_area);
  const J<N> area = twice_area * 0.5;
  const J<N> nk = dot(n, k_in);
  const J<N> ci = -nk;
  const J<N> co = -dot(n, k_out);
  if (!(ci.v > 0.0) || !(co.v > 0.0)) return m4_zero<N>();

  const double k = law.k;
  const V3<N> q = (k_in + k_out) * -1.0;
  const J<N> sum_cos = ci + co;

  // Form factor about the centroid, and the centroid's offset from the site.
  const J<N> a = dot(q, e1) * k;
  const J<N> b = dot(q, e2) * k;
  double fr, fi, far, fai, fbr, fbi;
  form_factor(a.v, b.v, &fr, &fi, &far, &fai, &fbr, &fbi);
  C<N> form;
  form.re = jconst<N>(fr);
  form.im = jconst<N>(fi);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    form.re.d[i] = far * a.d[i] + fbr * b.d[i];
    form.im.d[i] = fai * a.d[i] + fbi * b.d[i];
  }
  const V3<N> centroid = v0 + (e1 + e2) * (1.0 / 3.0);
  const C<N> offset = cis(dot(q, centroid - site) * k);

  const int64_t material = law.facet_material[facet];
  const int model = static_cast<int>(law.material_params[material * 2]);
  const double rms = static_cast<double>(law.material_params[material * 2 + 1]);
  const int64_t begin = law.layer_offsets[material];
  const int64_t end = law.layer_offsets[material + 1];
  C<N> r_te, r_tm;
  fresnel(ci, model, law.layers, begin, end, k, &r_te, &r_tm);

  // The coherent amplitude: j (k^2/pi) A chi F exp(j k q.(c - x_s)).
  J<N> magnitude = area * (k * k / kPi);
  if (rms > 0.0) {
    const J<N> kh = sum_cos * (k * rms);
    magnitude = magnitude * jexp(kh * kh * -0.5);
  }
  const C<N> coherent = times_j(form * offset * magnitude);

  // The incident TE / TM directions; at normal incidence any transverse pair
  // serves, since there r_TE = r_TM.
  V3<N> te_raw = cross(n, k_in);
  if (!(sqrt(dot(te_raw, te_raw).v) > 1e-9)) {
    V3<N> h, v;
    site_basis(k_in, &h, &v);
    te_raw = h;
  }
  const V3<N> te = jnormalize(te_raw);
  const V3<N> tm = cross(te, k_in);

  V3<N> bi[2], br[2];
  site_basis(k_in, &bi[0], &bi[1]);
  site_basis(k_out, &br[0], &br[1]);

  // e . R x = (n.k_i)(e.x) - (e.k_i)(n.x): R's leading transverse projection
  // is absorbed by the receive basis.
  J<N> r_te_proj[2], r_tm_proj[2];
  const J<N> n_te = dot(n, te);
  const J<N> n_tm = dot(n, tm);
  for (int a_index = 0; a_index < 2; ++a_index) {
    const J<N> e_k = dot(br[a_index], k_in);
    r_te_proj[a_index] = nk * dot(br[a_index], te) - e_k * n_te;
    r_tm_proj[a_index] = nk * dot(br[a_index], tm) - e_k * n_tm;
  }

  C<N> diffuse = cconst<N>(0.0, 0.0);
  bool rough = false;
  if (rms > 0.0) {
    const J<N> reflectance =
        (r_te.re * r_te.re + r_te.im * r_te.im + r_tm.re * r_tm.re + r_tm.im * r_tm.im) * 0.5;
    const J<N> kh = ci * (2.0 * k * rms);
    const J<N> fraction = (jconst<N>(1.0) - jexp(kh * kh * -1.0)) * reflectance;
    const J<N> sigma_d = fraction * ci * co * area * 4.0;
    if (sigma_d.v > 1.0e-300) {
      const J<N> amplitude = jsqrt(sigma_d * (1.0 / kPi)) * k;
      const double psi = facet_phase(law.seed, facet);
      diffuse = cconst<N>(cos(psi), -sin(psi)) * offset * amplitude;
      rough = true;
    }
  }

  M4<N> m;
  for (int b_index = 0; b_index < 2; ++b_index) {
    const J<N> p_te = dot(te, bi[b_index]);
    const J<N> p_tm = dot(tm, bi[b_index]);
    for (int a_index = 0; a_index < 2; ++a_index) {
      // e_a . G x_b = -(r_TE p_te e_a.R te + r_TM p_tm e_a.R tm)
      const C<N> law_ab = (r_te * (p_te * r_te_proj[a_index]) + r_tm * (p_tm * r_tm_proj[a_index])) * -1.0;
      C<N> value = coherent * law_ab;
      if (rough) value = value + diffuse * dot(br[a_index], bi[b_index]);
      m.s[2 * a_index + b_index] = value;
    }
  }
  return m;
}

template <int N>
__device__ __forceinline__ V3<N> jload64(const double* __restrict__ data, int64_t index) {
  const int64_t base = index * 3;
  return {jconst<N>(data[base]), jconst<N>(data[base + 1]), jconst<N>(data[base + 2])};
}

template <int N>
__device__ __forceinline__ void jtangent64(V3<N>& vector, const double* __restrict__ data, int64_t index) {
  vector.x.d[0] = data[index * 3];
  vector.y.d[0] = data[index * 3 + 1];
  vector.z.d[0] = data[index * 3 + 2];
}

// One slot's tables and the stride that lays the batch's slots over them.
struct Rows {
  const float* origin_in;
  const float* k_in;
  const float* k_out;
  const int64_t* idx_in;
  const int64_t* idx_out;
  const int64_t* idx_site;
  const float* corners;
  const double* sites;
  const int64_t* site_offsets;
  const int64_t* site_facets;
  const int32_t* row_valid;
  int64_t rows_per_slot;
  int64_t in_per_slot;
  int64_t out_per_slot;
};

struct RowIndex {
  int64_t i, o, s;
};

__device__ __forceinline__ RowIndex row_index(const Rows& rows, int64_t k) {
  const int64_t g = k / rows.rows_per_slot;
  const int64_t r = k - g * rows.rows_per_slot;
  return {rows.idx_in[r] + g * rows.in_per_slot, rows.idx_out[r] + g * rows.out_per_slot, rows.idx_site[r]};
}

// An inbound row whose last segment starts AT the site is the facet's own
// surface reflecting the wave onto itself: a specular interaction that
// coincides with the sink, which the physical-optics facet already is. It is
// not a path to compose. The tolerance is float32 resolution at the site's
// range plus a micrometre; a genuine interaction that close to a site is a
// set of measure zero.
__device__ __forceinline__ bool self_illuminated(const Rows& rows, const RowIndex& index) {
  double d2 = 0.0;
  double r2 = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double site = rows.sites[index.s * 3 + axis];
    const double delta = static_cast<double>(rows.origin_in[index.i * 3 + axis]) - site;
    d2 += delta * delta;
    r2 += site * site;
  }
  const double tolerance = 1.0e-6 + 1.0e-6 * sqrt(r2);
  return d2 < tolerance * tolerance;
}

__device__ __forceinline__ void store_up(float* up, int64_t k) {
  up[k * 3] = 0.0f;
  up[k * 3 + 1] = 0.0f;
  up[k * 3 + 2] = 1.0f;
}

__global__ void facet_response_forward_kernel(Rows rows, Law law, float* s, float* up, int64_t num_rows) {
  const int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= num_rows) return;
  store_up(up, k);
  if (rows.row_valid[k] == 0) {
    for (int e = 0; e < 8; ++e) s[k * 8 + e] = 0.0f;
    return;
  }
  const RowIndex index = row_index(rows, k);
  if (self_illuminated(rows, index)) {
    for (int e = 0; e < 8; ++e) s[k * 8 + e] = 0.0f;
    return;
  }
  const V3<0> k_in = jload<0>(rows.k_in, index.i);
  const V3<0> k_out = jload<0>(rows.k_out, index.o);
  const V3<0> site = jload64<0>(rows.sites, index.s);
  double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int64_t entry = rows.site_offsets[index.s]; entry < rows.site_offsets[index.s + 1]; ++entry) {
    const int64_t f = rows.site_facets[entry];
    const M4<0> m = facet_matrix<0>(
        k_in, k_out, jload<0>(rows.corners, 3 * f), jload<0>(rows.corners, 3 * f + 1),
        jload<0>(rows.corners, 3 * f + 2), site, f, law);
    for (int e = 0; e < 4; ++e) {
      acc[2 * e] += m.s[e].re.v;
      acc[2 * e + 1] += m.s[e].im.v;
    }
  }
  for (int e = 0; e < 8; ++e) s[k * 8 + e] = static_cast<float>(acc[e]);
}

__global__ void facet_response_jvp_kernel(
    Rows rows, Law law, const float* tan_k_in, const float* tan_k_out, const float* tan_corners,
    const double* tan_sites, float* tan_s, int64_t num_rows) {
  const int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= num_rows) return;
  if (rows.row_valid[k] == 0) {
    for (int e = 0; e < 8; ++e) tan_s[k * 8 + e] = 0.0f;
    return;
  }
  const RowIndex index = row_index(rows, k);
  if (self_illuminated(rows, index)) {
    for (int e = 0; e < 8; ++e) tan_s[k * 8 + e] = 0.0f;
    return;
  }
  V3<1> k_in = jload<1>(rows.k_in, index.i);
  V3<1> k_out = jload<1>(rows.k_out, index.o);
  V3<1> site = jload64<1>(rows.sites, index.s);
  jtangent(k_in, tan_k_in, index.i);
  jtangent(k_out, tan_k_out, index.o);
  jtangent64(site, tan_sites, index.s);
  double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int64_t entry = rows.site_offsets[index.s]; entry < rows.site_offsets[index.s + 1]; ++entry) {
    const int64_t f = rows.site_facets[entry];
    V3<1> v[3];
    for (int corner = 0; corner < 3; ++corner) {
      v[corner] = jload<1>(rows.corners, 3 * f + corner);
      jtangent(v[corner], tan_corners, 3 * f + corner);
    }
    const M4<1> m = facet_matrix<1>(k_in, k_out, v[0], v[1], v[2], site, f, law);
    for (int e = 0; e < 4; ++e) {
      acc[2 * e] += m.s[e].re.d[0];
      acc[2 * e + 1] += m.s[e].im.d[0];
    }
  }
  for (int e = 0; e < 8; ++e) tan_s[k * 8 + e] = static_cast<float>(acc[e]);
}

struct Owners {
  const int64_t* by_in_offsets;
  const int64_t* by_in_rows;
  const int64_t* by_out_offsets;
  const int64_t* by_out_rows;
  const int64_t* by_site_offsets;
  const int64_t* by_site_rows;
  const int64_t* facet_site;
  int64_t slots;
};

template <int N>
__device__ __forceinline__ void contract(const M4<N>& m, const float* grad, int64_t k, double* g) {
  for (int e = 0; e < 4; ++e) {
    const double gr = grad[k * 8 + 2 * e];
    const double gi = grad[k * 8 + 2 * e + 1];
    for (int lane = 0; lane < N; ++lane) g[lane] += gr * m.s[e].re.d[lane] + gi * m.s[e].im.d[lane];
  }
}

// One owner slot per thread. Families, end to end: inbound leg rows and
// outbound leg rows of every slot, then facets and sites (shared by every
// slot). A leg-row owner of slot g walks its per-slot CSR in slot g; a shared
// owner walks its per-slot CSR once per slot, in slot order.
__global__ void facet_response_backward_kernel(
    Rows rows, Law law, Owners owners, const float* grad, float* grad_k_in, float* grad_k_out, float* grad_corners,
    double* grad_sites, int64_t num_in, int64_t num_out, int64_t num_facets, int64_t num_sites) {
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread >= num_in + num_out + num_facets + num_sites) return;

  if (thread < num_in + num_out || thread >= num_in + num_out + num_facets) {
    int family;
    int64_t slot;
    const int64_t* offsets;
    const int64_t* members;
    int64_t first_slot = 0;
    int64_t last_slot = owners.slots;
    if (thread < num_in) {
      family = 0;
      first_slot = thread / rows.in_per_slot;
      last_slot = first_slot + 1;
      slot = thread - first_slot * rows.in_per_slot;
      offsets = owners.by_in_offsets;
      members = owners.by_in_rows;
    } else if (thread < num_in + num_out) {
      family = 1;
      first_slot = (thread - num_in) / rows.out_per_slot;
      last_slot = first_slot + 1;
      slot = thread - num_in - first_slot * rows.out_per_slot;
      offsets = owners.by_out_offsets;
      members = owners.by_out_rows;
    } else {
      family = 3;
      slot = thread - num_in - num_out - num_facets;
      offsets = owners.by_site_offsets;
      members = owners.by_site_rows;
    }
    double g[3] = {0.0, 0.0, 0.0};
    for (int64_t gslot = first_slot; gslot < last_slot; ++gslot) {
      for (int64_t entry = offsets[slot]; entry < offsets[slot + 1]; ++entry) {
        const int64_t k = gslot * rows.rows_per_slot + members[entry];
        if (rows.row_valid[k] == 0) continue;
        const RowIndex index = row_index(rows, k);
        if (self_illuminated(rows, index)) continue;
        V3<3> k_in = jload<3>(rows.k_in, index.i);
        V3<3> k_out = jload<3>(rows.k_out, index.o);
        V3<3> site = jload64<3>(rows.sites, index.s);
        if (family == 0) jseed(k_in, 0);
        if (family == 1) jseed(k_out, 0);
        if (family == 3) jseed(site, 0);
        for (int64_t member = rows.site_offsets[index.s]; member < rows.site_offsets[index.s + 1]; ++member) {
          const int64_t f = rows.site_facets[member];
          const M4<3> m = facet_matrix<3>(
              k_in, k_out, jload<3>(rows.corners, 3 * f), jload<3>(rows.corners, 3 * f + 1),
              jload<3>(rows.corners, 3 * f + 2), site, f, law);
          contract(m, grad, k, g);
        }
      }
    }
    if (family == 3) {
      for (int lane = 0; lane < 3; ++lane) grad_sites[slot * 3 + lane] = g[lane];
      return;
    }
    float* out = family == 0 ? grad_k_in : grad_k_out;
    for (int lane = 0; lane < 3; ++lane) out[thread * 3 - (family == 0 ? 0 : num_in * 3) + lane] = static_cast<float>(g[lane]);
    return;
  }

  // A facet: nine lanes over its own corners, summed over its site's rows in every slot.
  const int64_t f = thread - num_in - num_out;
  const int64_t s = owners.facet_site[f];
  double g[9];
  for (int lane = 0; lane < 9; ++lane) g[lane] = 0.0;
  for (int64_t gslot = 0; gslot < owners.slots; ++gslot) {
    for (int64_t entry = owners.by_site_offsets[s]; entry < owners.by_site_offsets[s + 1]; ++entry) {
      const int64_t k = gslot * rows.rows_per_slot + owners.by_site_rows[entry];
      if (rows.row_valid[k] == 0) continue;
      const RowIndex index = row_index(rows, k);
      if (self_illuminated(rows, index)) continue;
      V3<9> v[3];
      for (int corner = 0; corner < 3; ++corner) {
        v[corner] = jload<9>(rows.corners, 3 * f + corner);
        jseed(v[corner], 3 * corner);
      }
      const M4<9> m = facet_matrix<9>(
          jload<9>(rows.k_in, index.i), jload<9>(rows.k_out, index.o), v[0], v[1], v[2],
          jload64<9>(rows.sites, s), f, law);
      contract(m, grad, k, g);
    }
  }
  for (int lane = 0; lane < 9; ++lane) grad_corners[f * 9 + lane] = static_cast<float>(g[lane]);
}

// ---------------------------------------------------------------------------
// Facet geometry.
// ---------------------------------------------------------------------------

// Threads [0, F): corners of facet t. Threads [F, F + S): site s, the
// centroid of its anchor facet site_facets[site_offsets[s]], in float64 (the
// precise site the join's path length is computed from, R-ADR-040). Linear in
// the vertices, so this kernel run on a vertex tangent is the forward-mode
// companion.
__global__ void facet_geometry_kernel(
    const float* __restrict__ vertices, const int64_t* __restrict__ faces,
    const int64_t* __restrict__ site_offsets, const int64_t* __restrict__ site_facets,
    float* __restrict__ corners, double* __restrict__ sites, int num_facets, int num_sites) {
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread < num_facets) {
    for (int corner = 0; corner < 3; ++corner) {
      const int64_t vertex = faces[static_cast<int64_t>(thread) * 3 + corner];
      for (int axis = 0; axis < 3; ++axis) {
        corners[static_cast<int64_t>(thread) * 9 + corner * 3 + axis] = vertices[vertex * 3 + axis];
      }
    }
    return;
  }
  const int s = thread - num_facets;
  if (s >= num_sites) return;
  const int64_t anchor = site_facets[site_offsets[s]];
  double sum[3] = {0.0, 0.0, 0.0};
  for (int corner = 0; corner < 3; ++corner) {
    const int64_t vertex = faces[anchor * 3 + corner];
    for (int axis = 0; axis < 3; ++axis) sum[axis] += vertices[vertex * 3 + axis];
  }
  for (int axis = 0; axis < 3; ++axis) sites[static_cast<int64_t>(s) * 3 + axis] = sum[axis] / 3.0;
}

// One thread per vertex: the transpose of facet_geometry_kernel, reduced over
// the vertex's (facet, corner) incidences in frozen CSR order.
__global__ void facet_geometry_backward_kernel(
    const int64_t* __restrict__ vertex_offsets, const int64_t* __restrict__ vertex_entries,
    const int64_t* __restrict__ facet_site, const int64_t* __restrict__ site_offsets,
    const int64_t* __restrict__ site_facets, const float* __restrict__ grad_corners,
    const double* __restrict__ grad_sites, float* __restrict__ grad_vertices, int num_vertices) {
  const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex >= num_vertices) return;
  double g[3] = {0.0, 0.0, 0.0};
  for (int64_t entry = vertex_offsets[vertex]; entry < vertex_offsets[vertex + 1]; ++entry) {
    const int64_t incidence = vertex_entries[entry];
    const int64_t f = incidence / 3;
    const int64_t corner = incidence % 3;
    const int64_t s = facet_site[f];
    const double weight = site_facets[site_offsets[s]] == f ? 1.0 / 3.0 : 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      g[axis] += grad_corners[f * 9 + corner * 3 + axis] + weight * grad_sites[s * 3 + axis];
    }
  }
  for (int axis = 0; axis < 3; ++axis) grad_vertices[static_cast<int64_t>(vertex) * 3 + axis] = static_cast<float>(g[axis]);
}

void check_count(const torch::stable::Tensor& tensor, int64_t expected, const char* name) {
  check_cuda_float(tensor, name);
  STD_TORCH_CHECK(tensor.numel() == expected, name, " has the wrong number of values.");
}

void check_double_count(const torch::stable::Tensor& tensor, int64_t expected, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Double, name, " must have dtype torch.float64.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  STD_TORCH_CHECK(tensor.numel() == expected, name, " has the wrong number of values.");
}

void check_long_count(const torch::stable::Tensor& tensor, int64_t expected, const char* name) {
  check_cuda_long(tensor, name);
  STD_TORCH_CHECK(tensor.numel() == expected, name, " has the wrong number of entries.");
}

inline int check_origin(const torch::stable::Tensor& origin_in, const torch::stable::Tensor& k_in) {
  check_cuda_float(origin_in, "origin_in");
  STD_TORCH_CHECK(origin_in.numel() == k_in.numel(), "origin_in must hold one point per inbound leg row.");
  return 0;
}

struct Shapes {
  int64_t rows;
  int64_t facets;
  int64_t sites;
};

Shapes check_rows(
    const torch::stable::Tensor& k_in, const torch::stable::Tensor& k_out, const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out, const torch::stable::Tensor& idx_site, const torch::stable::Tensor& corners,
    const torch::stable::Tensor& sites, const torch::stable::Tensor& site_offsets,
    const torch::stable::Tensor& site_facets, const torch::stable::Tensor& facet_material,
    const torch::stable::Tensor& material_params, const torch::stable::Tensor& layer_offsets,
    const torch::stable::Tensor& layers, const torch::stable::Tensor& row_valid, int64_t slots) {
  STD_TORCH_CHECK(slots >= 1, "slots must be at least 1.");
  check_cuda_float(k_in, "k_in");
  check_cuda_float(k_out, "k_out");
  STD_TORCH_CHECK(
      k_in.numel() % (3 * slots) == 0 && k_out.numel() % (3 * slots) == 0,
      "k_in and k_out must hold three values per leg row of every slot.");
  const int per_slot = checked_int(idx_in.numel(), "rows_per_slot");
  check_index(idx_in, per_slot, "idx_in");
  check_index(idx_out, per_slot, "idx_out");
  check_index(idx_site, per_slot, "idx_site");
  check_cuda_float(corners, "corners");
  STD_TORCH_CHECK(corners.numel() % 9 == 0, "corners must hold nine values per facet.");
  const int64_t facets = corners.numel() / 9;
  STD_TORCH_CHECK(sites.numel() % 3 == 0, "sites must hold three values per site.");
  const int64_t num_sites = sites.numel() / 3;
  check_double_count(sites, num_sites * 3, "sites");
  check_long_count(site_offsets, num_sites + 1, "site_offsets");
  check_long_count(site_facets, facets, "site_facets");
  check_long_count(facet_material, facets, "facet_material");
  check_cuda_float(material_params, "material_params");
  STD_TORCH_CHECK(material_params.numel() % 2 == 0, "material_params must hold two values per material.");
  check_long_count(layer_offsets, material_params.numel() / 2 + 1, "layer_offsets");
  check_cuda_float(layers, "layers");
  STD_TORCH_CHECK(layers.numel() % 4 == 0, "layers must hold four values per layer.");
  check_cuda_int(row_valid, "row_valid");
  STD_TORCH_CHECK(row_valid.numel() == static_cast<int64_t>(per_slot) * slots, "row_valid must hold one flag per composed row.");
  return {static_cast<int64_t>(per_slot) * slots, facets, num_sites};
}

Rows make_rows(
    const torch::stable::Tensor& origin_in, const torch::stable::Tensor& k_in, const torch::stable::Tensor& k_out, const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out, const torch::stable::Tensor& idx_site, const torch::stable::Tensor& corners,
    const torch::stable::Tensor& sites, const torch::stable::Tensor& site_offsets,
    const torch::stable::Tensor& site_facets, const torch::stable::Tensor& row_valid, int64_t slots) {
  return {
      origin_in.const_data_ptr<float>(),
      k_in.const_data_ptr<float>(),          k_out.const_data_ptr<float>(),
      idx_in.const_data_ptr<int64_t>(),      idx_out.const_data_ptr<int64_t>(),
      idx_site.const_data_ptr<int64_t>(),    corners.const_data_ptr<float>(),
      sites.const_data_ptr<double>(),        site_offsets.const_data_ptr<int64_t>(),
      site_facets.const_data_ptr<int64_t>(), row_valid.const_data_ptr<int32_t>(),
      idx_in.numel(),                        k_in.numel() / (3 * slots),
      k_out.numel() / (3 * slots)};
}

Law make_law(
    const torch::stable::Tensor& facet_material, const torch::stable::Tensor& material_params,
    const torch::stable::Tensor& layer_offsets, const torch::stable::Tensor& layers, double wavenumber,
    int64_t seed) {
  STD_TORCH_CHECK(wavenumber > 0.0, "wavenumber must be positive.");
  return {
      material_params.const_data_ptr<float>(),
      layers.const_data_ptr<float>(),
      layer_offsets.const_data_ptr<int64_t>(),
      facet_material.const_data_ptr<int64_t>(),
      wavenumber,
      static_cast<uint64_t>(seed)};
}

inline dim3 wide_grid(int64_t count) {
  return dim3(static_cast<unsigned>((count + kBlock - 1) / kBlock), 1, 1);
}

}  // namespace

void facet_geometry_forward_cuda(
    const torch::stable::Tensor& vertices, const torch::stable::Tensor& faces,
    const torch::stable::Tensor& site_offsets, const torch::stable::Tensor& site_facets,
    torch::stable::Tensor corners, torch::stable::Tensor sites) {
  check_cuda_float(vertices, "vertices");
  STD_TORCH_CHECK(vertices.numel() % 3 == 0, "vertices must hold three values per vertex.");
  check_cuda_long(faces, "faces");
  STD_TORCH_CHECK(faces.numel() % 3 == 0, "faces must hold three vertex indices per facet.");
  const int facets = checked_int(faces.numel() / 3, "facets");
  check_cuda_long(site_offsets, "site_offsets");
  const int num_sites = checked_int(site_offsets.numel() - 1, "sites");
  check_long_count(site_facets, facets, "site_facets");
  check_count(corners, static_cast<int64_t>(facets) * 9, "corners");
  check_double_count(sites, static_cast<int64_t>(num_sites) * 3, "sites");
  const int total = facets + num_sites;
  if (total == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(corners.get_device_index());
  facet_geometry_kernel<<<linear_grid(total), dim3(kBlock, 1, 1), 0, current_cuda_stream(corners)>>>(
      vertices.const_data_ptr<float>(), faces.const_data_ptr<int64_t>(), site_offsets.const_data_ptr<int64_t>(),
      site_facets.const_data_ptr<int64_t>(), corners.mutable_data_ptr<float>(), sites.mutable_data_ptr<double>(),
      facets, num_sites);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void facet_geometry_backward_cuda(
    const torch::stable::Tensor& vertex_offsets, const torch::stable::Tensor& vertex_entries,
    const torch::stable::Tensor& facet_site, const torch::stable::Tensor& site_offsets,
    const torch::stable::Tensor& site_facets, const torch::stable::Tensor& grad_corners,
    const torch::stable::Tensor& grad_sites, torch::stable::Tensor grad_vertices) {
  check_cuda_long(vertex_offsets, "vertex_offsets");
  const int vertices = checked_int(vertex_offsets.numel() - 1, "vertices");
  check_cuda_long(facet_site, "facet_site");
  const int64_t facets = facet_site.numel();
  check_long_count(vertex_entries, facets * 3, "vertex_entries");
  check_cuda_long(site_offsets, "site_offsets");
  check_long_count(site_facets, facets, "site_facets");
  check_count(grad_corners, facets * 9, "grad_corners");
  check_double_count(grad_sites, (site_offsets.numel() - 1) * 3, "grad_sites");
  check_count(grad_vertices, static_cast<int64_t>(vertices) * 3, "grad_vertices");
  if (vertices == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(grad_vertices.get_device_index());
  facet_geometry_backward_kernel<<<
      linear_grid(vertices), dim3(kBlock, 1, 1), 0, current_cuda_stream(grad_vertices)>>>(
      vertex_offsets.const_data_ptr<int64_t>(), vertex_entries.const_data_ptr<int64_t>(),
      facet_site.const_data_ptr<int64_t>(), site_offsets.const_data_ptr<int64_t>(),
      site_facets.const_data_ptr<int64_t>(), grad_corners.const_data_ptr<float>(), grad_sites.const_data_ptr<double>(),
      grad_vertices.mutable_data_ptr<float>(), vertices);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}


#define FACET_ROW_ARGS                                                                                         \
  const torch::stable::Tensor &origin_in, const torch::stable::Tensor &k_in, const torch::stable::Tensor &k_out, const torch::stable::Tensor &idx_in,  \
      const torch::stable::Tensor &idx_out, const torch::stable::Tensor &idx_site,                             \
      const torch::stable::Tensor &corners, const torch::stable::Tensor &sites,                                \
      const torch::stable::Tensor &site_offsets, const torch::stable::Tensor &site_facets,                     \
      const torch::stable::Tensor &facet_material, const torch::stable::Tensor &material_params,               \
      const torch::stable::Tensor &layer_offsets, const torch::stable::Tensor &layers,                         \
      const torch::stable::Tensor &row_valid
#define FACET_CHECK_ROWS                                                                                       \
  (check_origin(origin_in, k_in),                                                                               \
   check_rows(k_in, k_out, idx_in, idx_out, idx_site, corners, sites, site_offsets, site_facets, facet_material, \
              material_params, layer_offsets, layers, row_valid, slots))
#define FACET_MAKE_ROWS \
  make_rows(origin_in, k_in, k_out, idx_in, idx_out, idx_site, corners, sites, site_offsets, site_facets, row_valid, slots)

void facet_response_forward_cuda(
    FACET_ROW_ARGS, torch::stable::Tensor s, torch::stable::Tensor up, double wavenumber, int64_t seed,
    int64_t slots) {
  const Shapes shapes = FACET_CHECK_ROWS;
  check_count(s, shapes.rows * 8, "s");
  check_count(up, shapes.rows * 3, "up");
  const Law law = make_law(facet_material, material_params, layer_offsets, layers, wavenumber, seed);
  if (shapes.rows == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(s.get_device_index());
  facet_response_forward_kernel<<<wide_grid(shapes.rows), dim3(kBlock, 1, 1), 0, current_cuda_stream(s)>>>(
      FACET_MAKE_ROWS, law, s.mutable_data_ptr<float>(), up.mutable_data_ptr<float>(), shapes.rows);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void facet_response_jvp_cuda(
    FACET_ROW_ARGS, const torch::stable::Tensor& tan_k_in, const torch::stable::Tensor& tan_k_out,
    const torch::stable::Tensor& tan_corners, const torch::stable::Tensor& tan_sites, torch::stable::Tensor tan_s,
    double wavenumber, int64_t seed, int64_t slots) {
  const Shapes shapes = FACET_CHECK_ROWS;
  check_count(tan_k_in, k_in.numel(), "tan_k_in");
  check_count(tan_k_out, k_out.numel(), "tan_k_out");
  check_count(tan_corners, corners.numel(), "tan_corners");
  check_double_count(tan_sites, sites.numel(), "tan_sites");
  check_count(tan_s, shapes.rows * 8, "tan_s");
  const Law law = make_law(facet_material, material_params, layer_offsets, layers, wavenumber, seed);
  if (shapes.rows == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(tan_s.get_device_index());
  facet_response_jvp_kernel<<<wide_grid(shapes.rows), dim3(kBlock, 1, 1), 0, current_cuda_stream(tan_s)>>>(
      FACET_MAKE_ROWS, law, tan_k_in.const_data_ptr<float>(), tan_k_out.const_data_ptr<float>(),
      tan_corners.const_data_ptr<float>(), tan_sites.const_data_ptr<double>(), tan_s.mutable_data_ptr<float>(),
      shapes.rows);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void facet_response_backward_cuda(
    FACET_ROW_ARGS, const torch::stable::Tensor& by_in_offsets, const torch::stable::Tensor& by_in_rows,
    const torch::stable::Tensor& by_out_offsets, const torch::stable::Tensor& by_out_rows,
    const torch::stable::Tensor& by_site_offsets, const torch::stable::Tensor& by_site_rows,
    const torch::stable::Tensor& facet_site, const torch::stable::Tensor& grad_s, torch::stable::Tensor grad_k_in,
    torch::stable::Tensor grad_k_out, torch::stable::Tensor grad_corners, torch::stable::Tensor grad_sites,
    double wavenumber, int64_t seed, int64_t slots) {
  const Shapes shapes = FACET_CHECK_ROWS;
  const int64_t legs_in = k_in.numel() / 3;
  const int64_t legs_out = k_out.numel() / 3;
  const int64_t per_slot = idx_in.numel();
  check_long_count(by_in_offsets, legs_in / slots + 1, "by_in_offsets");
  check_long_count(by_out_offsets, legs_out / slots + 1, "by_out_offsets");
  check_long_count(by_site_offsets, shapes.sites + 1, "by_site_offsets");
  check_long_count(by_in_rows, per_slot, "by_in_rows");
  check_long_count(by_out_rows, per_slot, "by_out_rows");
  check_long_count(by_site_rows, per_slot, "by_site_rows");
  check_long_count(facet_site, shapes.facets, "facet_site");
  check_count(grad_s, shapes.rows * 8, "grad_s");
  check_count(grad_k_in, k_in.numel(), "grad_k_in");
  check_count(grad_k_out, k_out.numel(), "grad_k_out");
  check_count(grad_corners, corners.numel(), "grad_corners");
  check_double_count(grad_sites, sites.numel(), "grad_sites");
  const Law law = make_law(facet_material, material_params, layer_offsets, layers, wavenumber, seed);
  const int64_t total = legs_in + legs_out + shapes.facets + shapes.sites;
  if (total == 0) return;
  const Owners owners = {
      by_in_offsets.const_data_ptr<int64_t>(),  by_in_rows.const_data_ptr<int64_t>(),
      by_out_offsets.const_data_ptr<int64_t>(), by_out_rows.const_data_ptr<int64_t>(),
      by_site_offsets.const_data_ptr<int64_t>(), by_site_rows.const_data_ptr<int64_t>(),
      facet_site.const_data_ptr<int64_t>(),     slots};
  const torch::stable::accelerator::DeviceGuard guard(grad_corners.get_device_index());
  facet_response_backward_kernel<<<wide_grid(total), dim3(kBlock, 1, 1), 0, current_cuda_stream(grad_corners)>>>(
      FACET_MAKE_ROWS, law, owners, grad_s.const_data_ptr<float>(), grad_k_in.mutable_data_ptr<float>(),
      grad_k_out.mutable_data_ptr<float>(), grad_corners.mutable_data_ptr<float>(),
      grad_sites.mutable_data_ptr<double>(), legs_in, legs_out, shapes.facets, shapes.sites);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("facet_geometry_forward", TORCH_BOX(&facet_geometry_forward_cuda));
  m.impl("facet_geometry_jvp", TORCH_BOX(&facet_geometry_forward_cuda));
  m.impl("facet_geometry_backward", TORCH_BOX(&facet_geometry_backward_cuda));
  m.impl("facet_response_forward", TORCH_BOX(&facet_response_forward_cuda));
  m.impl("facet_response_jvp", TORCH_BOX(&facet_response_jvp_cuda));
  m.impl("facet_response_backward", TORCH_BOX(&facet_response_backward_cuda));
}
