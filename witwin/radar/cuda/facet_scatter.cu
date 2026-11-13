// Physical-optics facet scattering: a mesh target's facets as scatter sites.
//
// Two operator families.
//
// facet_geometry gathers one frame's facet corners from the deformed world
// vertices and places each scatter site at the mean of its member facets'
// centroids. Both outputs are linear in the vertices, so the forward-mode
// companion is the forward operator applied to the vertex tangent and the
// reverse companion is its transpose, reduced per vertex through a frozen
// vertex-incidence CSR: one thread owns one vertex, no atomics.
//
// facet_response evaluates one complex factor per COMPOSED row: the coherent
// sum over the facets f of the row's site s of
//
//   S_f = j (k^2 / pi) A_f ob_f chi_f Gamma_f F_f exp(j k q . (c_f - x_s))
//       + k sqrt(sigma_d,f / pi) exp(-j psi_f) exp(j k q . (c_f - x_s))
//
// with, all lengths in metres and k = 2 pi / lambda the carrier wavenumber:
//
//   u_in  = -dir_in            unit vector from the facet toward the wave's source
//   u_out =  dir_out           unit vector from the facet toward the receiver side
//   n, A_f                     unit normal (right-handed corner order) and area
//   ci = n.u_in, co = n.u_out  both must be positive, otherwise S_f = 0: a facet
//                              is lit and seen from its normal side only
//   q  = u_in + u_out          the bistatic phase vector
//   ob = (ci + co) / 2         the scalar Kirchhoff obliquity; ci when monostatic
//   F  = (1/A) int_T exp(j k q.(r - c_f)) dA, the exact flat-facet form factor
//   Gamma                      co-polar reflection coefficient of the facet's
//                              material at the incidence angle ci, the
//                              tangent-plane (Kirchhoff) approximation; see
//                              stack_reflection
//   chi = exp(-(k h (ci + co))^2 / 2), the coherent (Ament) roughness factor of
//                              a Gaussian surface of rms height h
//   sigma_d = 4 A f_d ci co,   f_d = (1 - exp(-(2 k h ci)^2)) |Gamma(ci)|^2,
//                              the Lambertian diffuse cross section that carries
//                              exactly the power the coherent term loses
//   psi_f                      a uniform phase drawn once per facet from the
//                              declared seed: a frozen surface realisation
//
// The normalisation is the two-way join's. A site's legs carry
// lambda / (4 pi d) exp(-j k d) each with the site excited at 1 W, so a flat
// plate at broadside reads S = 4 pi A / lambda^2 = (k^2 / pi) A and
// sigma = 4 pi A^2 / lambda^2. The factor j is the Huygens kernel's phase in the
// exp(+j omega t), exp(-j k d) convention: a stationary-phase evaluation over a
// convex surface contributes -j, so a PEC sphere returns -sqrt(pi) a with the
// phase of its front point, and a broadside PEC plate -j (4 pi A / lambda^2).
// The phase reference of every facet is the site position the legs used; the
// exp(j k q . (c_f - x_s)) factor moves each facet's own reference to its
// centroid under the far-field-of-the-cluster approximation, and is exactly 1
// for a site of one facet.
//
// The co-polar coefficient weights the TE (E perpendicular to the plane of
// incidence) and TM coefficients by the declared world polarization p:
// Gamma = w r_TE + (1 - w) r_TM with w = (p.s)^2 / |p_perp|^2, s the unit
// normal of the plane of incidence n x u_in and p_perp the part of p across
// u_in. Both are tangential-field coefficients of the characteristic-matrix
// stack (y_TE = k_z / mu, y_TM = eps / k_z), so they are equal at normal
// incidence and both -1 on a conductor.
//
// Derivatives are exact forward-mode jets (value plus N directional
// derivatives) carried through the same expression the forward evaluates. The
// two special functions - the form factor F(a, b) and the layer-stack
// reflection Gamma(cos) - are evaluated once on a small inner jet over their
// own arguments and chained into the outer jet. The VJP owns one gradient slot
// per thread over four owner families laid end to end: inbound leg rows
// (dir_in), outbound leg rows (dir_out), facets (corners) and sites (the site
// position in the phase offset), reduced through the join's own frozen CSR.
//
// Everything accumulates in double and stores float32. A dead row publishes
// exactly zero. The material table, the polarization, the wavenumber and the
// seed take no derivative: they select the law.

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

// The co-polar coefficient Gamma = w r_TE + (1 - w) r_TM at cosine x.
template <int N>
__device__ __forceinline__ C<N> co_polar(
    const J<N>& x, const J<N>& w, int model, const float* __restrict__ layers, int64_t begin,
    int64_t end, double k0) {
  const Reflection r = stack_reflection(x.v, model, layers, begin, end, k0);
  const C<N> te = chain(x, r.te_re, r.te_im, r.te_slope_re, r.te_slope_im);
  const C<N> tm = chain(x, r.tm_re, r.tm_im, r.tm_slope_re, r.tm_slope_im);
  return te * w + tm * (jconst<N>(1.0) - w);
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
  double px, py, pz;
  uint64_t seed;
};

// One facet's contribution to one row, as a complex jet over whatever lanes
// the caller seeded in its inputs.
template <int N>
__device__ __forceinline__ C<N> facet_term(
    const V3<N>& dir_in, const V3<N>& dir_out, const V3<N>& v0, const V3<N>& v1, const V3<N>& v2,
    const V3<N>& site, int64_t facet, const Law& law) {
  const V3<N> u_in = dir_in * -1.0;
  const V3<N>& u_out = dir_out;
  const V3<N> e1 = v1 - v0;
  const V3<N> e2 = v2 - v0;
  const V3<N> normal_area = cross(e1, e2);
  const J<N> twice_area = jsqrt(dot(normal_area, normal_area));
  if (!(twice_area.v > 0.0)) return cconst<N>(0.0, 0.0);
  const J<N> inv = jconst<N>(1.0) / twice_area;
  const V3<N> n = scale(normal_area, inv);
  const J<N> area = twice_area * 0.5;
  const J<N> ci = dot(n, u_in);
  const J<N> co = dot(n, u_out);
  if (!(ci.v > 0.0) || !(co.v > 0.0)) return cconst<N>(0.0, 0.0);

  const double k = law.k;
  const V3<N> q = u_in + u_out;
  const J<N> sum_cos = ci + co;
  const J<N> obliquity = sum_cos * 0.5;

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

  // Polarization weight of the TE coefficient.
  const V3<N> plane = cross(n, u_in);
  const J<N> plane2 = dot(plane, plane);
  J<N> w = jconst<N>(0.5);
  const double p2 = law.px * law.px + law.py * law.py + law.pz * law.pz;
  const J<N> p_along = u_in.x * law.px + u_in.y * law.py + u_in.z * law.pz;
  const J<N> p_perp2 = jconst<N>(p2) - p_along * p_along;
  if (plane2.v > 1.0e-12 && p_perp2.v > 1.0e-12 * p2) {
    const J<N> p_plane = plane.x * law.px + plane.y * law.py + plane.z * law.pz;
    w = p_plane * p_plane / (plane2 * p_perp2);
  }

  const int64_t material = law.facet_material[facet];
  const int model = static_cast<int>(law.material_params[material * 2]);
  const double rms = static_cast<double>(law.material_params[material * 2 + 1]);
  const int64_t begin = law.layer_offsets[material];
  const int64_t end = law.layer_offsets[material + 1];
  const C<N> gamma = co_polar(ci, w, model, law.layers, begin, end, k);

  J<N> magnitude = area * obliquity * (k * k / kPi);
  if (rms > 0.0) {
    const J<N> kh = sum_cos * (k * rms);
    magnitude = magnitude * jexp(kh * kh * -0.5);
  }
  C<N> term = times_j(gamma * form * magnitude);

  if (rms > 0.0) {
    const J<N> reflectance = gamma.re * gamma.re + gamma.im * gamma.im;
    const J<N> kh = ci * (2.0 * k * rms);
    const J<N> fraction = (jconst<N>(1.0) - jexp(kh * kh * -1.0)) * reflectance;
    const J<N> sigma_d = fraction * ci * co * area * 4.0;
    if (sigma_d.v > 1.0e-300) {
      const J<N> amplitude = jsqrt(sigma_d * (1.0 / kPi)) * k;
      const double psi = facet_phase(law.seed, facet);
      term = term + cconst<N>(cos(psi), -sin(psi)) * amplitude;
    }
  }
  return term * offset;
}

struct Rows {
  const float* dir_in;
  const float* dir_out;
  const int64_t* idx_in;
  const int64_t* idx_out;
  const int64_t* idx_site;
  const float* corners;
  const float* sites;
  const int64_t* site_offsets;
  const int64_t* site_facets;
  const int32_t* row_valid;
};

__global__ void facet_response_forward_kernel(Rows rows, Law law, float* s_re, float* s_im, int num_rows) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_rows) return;
  if (rows.row_valid[k] == 0) {
    s_re[k] = 0.0f;
    s_im[k] = 0.0f;
    return;
  }
  const int64_t s = rows.idx_site[k];
  const V3<0> din = jload<0>(rows.dir_in, rows.idx_in[k]);
  const V3<0> dout = jload<0>(rows.dir_out, rows.idx_out[k]);
  const V3<0> site = jload<0>(rows.sites, s);
  double acc_re = 0.0;
  double acc_im = 0.0;
  for (int64_t entry = rows.site_offsets[s]; entry < rows.site_offsets[s + 1]; ++entry) {
    const int64_t f = rows.site_facets[entry];
    const C<0> term = facet_term<0>(
        din, dout, jload<0>(rows.corners, 3 * f), jload<0>(rows.corners, 3 * f + 1),
        jload<0>(rows.corners, 3 * f + 2), site, f, law);
    acc_re += term.re.v;
    acc_im += term.im.v;
  }
  s_re[k] = static_cast<float>(acc_re);
  s_im[k] = static_cast<float>(acc_im);
}

__global__ void facet_response_jvp_kernel(
    Rows rows, Law law, const float* tan_dir_in, const float* tan_dir_out, const float* tan_corners,
    const float* tan_sites, float* tan_re, float* tan_im, int num_rows) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_rows) return;
  if (rows.row_valid[k] == 0) {
    tan_re[k] = 0.0f;
    tan_im[k] = 0.0f;
    return;
  }
  const int64_t s = rows.idx_site[k];
  V3<1> din = jload<1>(rows.dir_in, rows.idx_in[k]);
  V3<1> dout = jload<1>(rows.dir_out, rows.idx_out[k]);
  V3<1> site = jload<1>(rows.sites, s);
  jtangent(din, tan_dir_in, rows.idx_in[k]);
  jtangent(dout, tan_dir_out, rows.idx_out[k]);
  jtangent(site, tan_sites, s);
  double acc_re = 0.0;
  double acc_im = 0.0;
  for (int64_t entry = rows.site_offsets[s]; entry < rows.site_offsets[s + 1]; ++entry) {
    const int64_t f = rows.site_facets[entry];
    V3<1> v[3];
    for (int corner = 0; corner < 3; ++corner) {
      v[corner] = jload<1>(rows.corners, 3 * f + corner);
      jtangent(v[corner], tan_corners, 3 * f + corner);
    }
    const C<1> term = facet_term<1>(din, dout, v[0], v[1], v[2], site, f, law);
    acc_re += term.re.d[0];
    acc_im += term.im.d[0];
  }
  tan_re[k] = static_cast<float>(acc_re);
  tan_im[k] = static_cast<float>(acc_im);
}

struct Owners {
  const int64_t* by_in_offsets;
  const int64_t* by_in_rows;
  const int64_t* by_out_offsets;
  const int64_t* by_out_rows;
  const int64_t* by_site_offsets;
  const int64_t* by_site_rows;
  const int64_t* facet_site;
};

// One owner slot per thread. Families, end to end: inbound leg rows, outbound
// leg rows, facets, sites. Each sums g_re dRe(S)/dx + g_im dIm(S)/dx over the
// composed rows it owns, in the frozen CSR order.
__global__ void facet_response_backward_kernel(
    Rows rows, Law law, Owners owners, const float* grad_re, const float* grad_im,
    float* grad_dir_in, float* grad_dir_out, float* grad_corners, float* grad_sites, int num_in,
    int num_out, int num_facets, int num_sites) {
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread >= num_in + num_out + num_facets + num_sites) return;

  if (thread < num_in + num_out || thread >= num_in + num_out + num_facets) {
    // A three-lane family: a leg direction or a site position.
    int family;
    int slot;
    const int64_t* offsets;
    const int64_t* members;
    if (thread < num_in) {
      family = 0;
      slot = thread;
      offsets = owners.by_in_offsets;
      members = owners.by_in_rows;
    } else if (thread < num_in + num_out) {
      family = 1;
      slot = thread - num_in;
      offsets = owners.by_out_offsets;
      members = owners.by_out_rows;
    } else {
      family = 3;
      slot = thread - num_in - num_out - num_facets;
      offsets = owners.by_site_offsets;
      members = owners.by_site_rows;
    }
    double g[3] = {0.0, 0.0, 0.0};
    for (int64_t entry = offsets[slot]; entry < offsets[slot + 1]; ++entry) {
      const int64_t k = members[entry];
      if (rows.row_valid[k] == 0) continue;
      const int64_t s = rows.idx_site[k];
      V3<3> din = jload<3>(rows.dir_in, rows.idx_in[k]);
      V3<3> dout = jload<3>(rows.dir_out, rows.idx_out[k]);
      V3<3> site = jload<3>(rows.sites, s);
      if (family == 0) jseed(din, 0);
      if (family == 1) jseed(dout, 0);
      if (family == 3) jseed(site, 0);
      const double gr = grad_re[k];
      const double gi = grad_im[k];
      for (int64_t member = rows.site_offsets[s]; member < rows.site_offsets[s + 1]; ++member) {
        const int64_t f = rows.site_facets[member];
        const C<3> term = facet_term<3>(
            din, dout, jload<3>(rows.corners, 3 * f), jload<3>(rows.corners, 3 * f + 1),
            jload<3>(rows.corners, 3 * f + 2), site, f, law);
        for (int lane = 0; lane < 3; ++lane) g[lane] += gr * term.re.d[lane] + gi * term.im.d[lane];
      }
    }
    float* out = family == 0 ? grad_dir_in : (family == 1 ? grad_dir_out : grad_sites);
    for (int lane = 0; lane < 3; ++lane) out[static_cast<int64_t>(slot) * 3 + lane] = static_cast<float>(g[lane]);
    return;
  }

  // A facet: nine lanes over its own corners, summed over its site's rows.
  const int64_t f = thread - num_in - num_out;
  const int64_t s = owners.facet_site[f];
  double g[9];
  for (int lane = 0; lane < 9; ++lane) g[lane] = 0.0;
  for (int64_t entry = owners.by_site_offsets[s]; entry < owners.by_site_offsets[s + 1]; ++entry) {
    const int64_t k = owners.by_site_rows[entry];
    if (rows.row_valid[k] == 0) continue;
    V3<9> v[3];
    for (int corner = 0; corner < 3; ++corner) {
      v[corner] = jload<9>(rows.corners, 3 * f + corner);
      jseed(v[corner], 3 * corner);
    }
    const C<9> term = facet_term<9>(
        jload<9>(rows.dir_in, rows.idx_in[k]), jload<9>(rows.dir_out, rows.idx_out[k]), v[0], v[1], v[2],
        jload<9>(rows.sites, s), f, law);
    const double gr = grad_re[k];
    const double gi = grad_im[k];
    for (int lane = 0; lane < 9; ++lane) g[lane] += gr * term.re.d[lane] + gi * term.im.d[lane];
  }
  for (int lane = 0; lane < 9; ++lane) grad_corners[f * 9 + lane] = static_cast<float>(g[lane]);
}

// ---------------------------------------------------------------------------
// Facet geometry.
// ---------------------------------------------------------------------------

// Threads [0, F): corners of facet t. Threads [F, F + S): site s, the mean of
// its member facets' centroids. Linear in the vertices, so this kernel run on
// a vertex tangent is the forward-mode companion.
__global__ void facet_geometry_kernel(
    const float* __restrict__ vertices, const int64_t* __restrict__ faces,
    const int64_t* __restrict__ site_offsets, const int64_t* __restrict__ site_facets,
    float* __restrict__ corners, float* __restrict__ sites, int num_facets, int num_sites) {
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
  const int64_t begin = site_offsets[s];
  const int64_t end = site_offsets[s + 1];
  double sum[3] = {0.0, 0.0, 0.0};
  for (int64_t entry = begin; entry < end; ++entry) {
    const int64_t f = site_facets[entry];
    for (int corner = 0; corner < 3; ++corner) {
      const int64_t vertex = faces[f * 3 + corner];
      for (int axis = 0; axis < 3; ++axis) sum[axis] += vertices[vertex * 3 + axis];
    }
  }
  const double weight = end > begin ? 1.0 / (3.0 * static_cast<double>(end - begin)) : 0.0;
  for (int axis = 0; axis < 3; ++axis) sites[static_cast<int64_t>(s) * 3 + axis] = static_cast<float>(sum[axis] * weight);
}

// One thread per vertex: the transpose of facet_geometry_kernel, reduced over
// the vertex's (facet, corner) incidences in frozen CSR order.
__global__ void facet_geometry_backward_kernel(
    const int64_t* __restrict__ vertex_offsets, const int64_t* __restrict__ vertex_entries,
    const int64_t* __restrict__ facet_site, const int64_t* __restrict__ site_offsets,
    const float* __restrict__ grad_corners, const float* __restrict__ grad_sites,
    float* __restrict__ grad_vertices, int num_vertices) {
  const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex >= num_vertices) return;
  double g[3] = {0.0, 0.0, 0.0};
  for (int64_t entry = vertex_offsets[vertex]; entry < vertex_offsets[vertex + 1]; ++entry) {
    const int64_t incidence = vertex_entries[entry];
    const int64_t f = incidence / 3;
    const int64_t corner = incidence % 3;
    const int64_t s = facet_site[f];
    const double weight = 1.0 / (3.0 * static_cast<double>(site_offsets[s + 1] - site_offsets[s]));
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

void check_long_count(const torch::stable::Tensor& tensor, int64_t expected, const char* name) {
  check_cuda_long(tensor, name);
  STD_TORCH_CHECK(tensor.numel() == expected, name, " has the wrong number of entries.");
}

struct Shapes {
  int rows;
  int64_t facets;
  int64_t sites;
};

Shapes check_rows(
    const torch::stable::Tensor& dir_in, const torch::stable::Tensor& dir_out,
    const torch::stable::Tensor& idx_in, const torch::stable::Tensor& idx_out,
    const torch::stable::Tensor& idx_site, const torch::stable::Tensor& corners,
    const torch::stable::Tensor& sites, const torch::stable::Tensor& site_offsets,
    const torch::stable::Tensor& site_facets, const torch::stable::Tensor& facet_material,
    const torch::stable::Tensor& material_params, const torch::stable::Tensor& layer_offsets,
    const torch::stable::Tensor& layers, const torch::stable::Tensor& row_valid, int64_t num_rows) {
  const int rows = checked_int(num_rows, "num_rows");
  check_cuda_float(dir_in, "dir_in");
  check_cuda_float(dir_out, "dir_out");
  STD_TORCH_CHECK(
      dir_in.numel() % 3 == 0 && dir_out.numel() % 3 == 0,
      "dir_in and dir_out must hold three values per leg row.");
  check_index(idx_in, rows, "idx_in");
  check_index(idx_out, rows, "idx_out");
  check_index(idx_site, rows, "idx_site");
  check_cuda_float(corners, "corners");
  STD_TORCH_CHECK(corners.numel() % 9 == 0, "corners must hold nine values per facet.");
  const int64_t facets = corners.numel() / 9;
  check_cuda_float(sites, "sites");
  STD_TORCH_CHECK(sites.numel() % 3 == 0, "sites must hold three values per site.");
  const int64_t num_sites = sites.numel() / 3;
  check_long_count(site_offsets, num_sites + 1, "site_offsets");
  check_long_count(site_facets, facets, "site_facets");
  check_long_count(facet_material, facets, "facet_material");
  check_cuda_float(material_params, "material_params");
  STD_TORCH_CHECK(material_params.numel() % 2 == 0, "material_params must hold two values per material.");
  check_long_count(layer_offsets, material_params.numel() / 2 + 1, "layer_offsets");
  check_cuda_float(layers, "layers");
  STD_TORCH_CHECK(layers.numel() % 4 == 0, "layers must hold four values per layer.");
  check_cuda_int(row_valid, "row_valid");
  STD_TORCH_CHECK(row_valid.numel() == rows, "row_valid must hold one flag per composed row.");
  return {rows, facets, num_sites};
}

Rows make_rows(
    const torch::stable::Tensor& dir_in, const torch::stable::Tensor& dir_out,
    const torch::stable::Tensor& idx_in, const torch::stable::Tensor& idx_out,
    const torch::stable::Tensor& idx_site, const torch::stable::Tensor& corners,
    const torch::stable::Tensor& sites, const torch::stable::Tensor& site_offsets,
    const torch::stable::Tensor& site_facets, const torch::stable::Tensor& row_valid) {
  return {
      dir_in.const_data_ptr<float>(),       dir_out.const_data_ptr<float>(),
      idx_in.const_data_ptr<int64_t>(),     idx_out.const_data_ptr<int64_t>(),
      idx_site.const_data_ptr<int64_t>(),   corners.const_data_ptr<float>(),
      sites.const_data_ptr<float>(),        site_offsets.const_data_ptr<int64_t>(),
      site_facets.const_data_ptr<int64_t>(), row_valid.const_data_ptr<int32_t>()};
}

Law make_law(
    const torch::stable::Tensor& facet_material, const torch::stable::Tensor& material_params,
    const torch::stable::Tensor& layer_offsets, const torch::stable::Tensor& layers, double wavenumber,
    double pol_x, double pol_y, double pol_z, int64_t seed) {
  STD_TORCH_CHECK(wavenumber > 0.0, "wavenumber must be positive.");
  STD_TORCH_CHECK(pol_x != 0.0 || pol_y != 0.0 || pol_z != 0.0, "the polarization must be non-zero.");
  return {
      material_params.const_data_ptr<float>(),
      layers.const_data_ptr<float>(),
      layer_offsets.const_data_ptr<int64_t>(),
      facet_material.const_data_ptr<int64_t>(),
      wavenumber,
      pol_x,
      pol_y,
      pol_z,
      static_cast<uint64_t>(seed)};
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
  check_count(sites, static_cast<int64_t>(num_sites) * 3, "sites");
  const int total = facets + num_sites;
  if (total == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(corners.get_device_index());
  facet_geometry_kernel<<<linear_grid(total), dim3(kBlock, 1, 1), 0, current_cuda_stream(corners)>>>(
      vertices.const_data_ptr<float>(), faces.const_data_ptr<int64_t>(), site_offsets.const_data_ptr<int64_t>(),
      site_facets.const_data_ptr<int64_t>(), corners.mutable_data_ptr<float>(), sites.mutable_data_ptr<float>(),
      facets, num_sites);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void facet_geometry_backward_cuda(
    const torch::stable::Tensor& vertex_offsets, const torch::stable::Tensor& vertex_entries,
    const torch::stable::Tensor& facet_site, const torch::stable::Tensor& site_offsets,
    const torch::stable::Tensor& grad_corners, const torch::stable::Tensor& grad_sites,
    torch::stable::Tensor grad_vertices) {
  check_cuda_long(vertex_offsets, "vertex_offsets");
  const int vertices = checked_int(vertex_offsets.numel() - 1, "vertices");
  check_cuda_long(facet_site, "facet_site");
  const int64_t facets = facet_site.numel();
  check_long_count(vertex_entries, facets * 3, "vertex_entries");
  check_cuda_long(site_offsets, "site_offsets");
  check_count(grad_corners, facets * 9, "grad_corners");
  check_count(grad_sites, (site_offsets.numel() - 1) * 3, "grad_sites");
  check_count(grad_vertices, static_cast<int64_t>(vertices) * 3, "grad_vertices");
  if (vertices == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(grad_vertices.get_device_index());
  facet_geometry_backward_kernel<<<
      linear_grid(vertices), dim3(kBlock, 1, 1), 0, current_cuda_stream(grad_vertices)>>>(
      vertex_offsets.const_data_ptr<int64_t>(), vertex_entries.const_data_ptr<int64_t>(),
      facet_site.const_data_ptr<int64_t>(), site_offsets.const_data_ptr<int64_t>(),
      grad_corners.const_data_ptr<float>(), grad_sites.const_data_ptr<float>(),
      grad_vertices.mutable_data_ptr<float>(), vertices);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void facet_response_forward_cuda(
    const torch::stable::Tensor& dir_in, const torch::stable::Tensor& dir_out,
    const torch::stable::Tensor& idx_in, const torch::stable::Tensor& idx_out,
    const torch::stable::Tensor& idx_site, const torch::stable::Tensor& corners,
    const torch::stable::Tensor& sites, const torch::stable::Tensor& site_offsets,
    const torch::stable::Tensor& site_facets, const torch::stable::Tensor& facet_material,
    const torch::stable::Tensor& material_params, const torch::stable::Tensor& layer_offsets,
    const torch::stable::Tensor& layers, const torch::stable::Tensor& row_valid, torch::stable::Tensor s_re,
    torch::stable::Tensor s_im, double wavenumber, double pol_x, double pol_y, double pol_z, int64_t seed,
    int64_t num_rows) {
  const Shapes shapes = check_rows(
      dir_in, dir_out, idx_in, idx_out, idx_site, corners, sites, site_offsets, site_facets, facet_material,
      material_params, layer_offsets, layers, row_valid, num_rows);
  check_count(s_re, shapes.rows, "s_re");
  check_count(s_im, shapes.rows, "s_im");
  const Law law = make_law(facet_material, material_params, layer_offsets, layers, wavenumber, pol_x, pol_y, pol_z, seed);
  if (shapes.rows == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(s_re.get_device_index());
  facet_response_forward_kernel<<<linear_grid(shapes.rows), dim3(kBlock, 1, 1), 0, current_cuda_stream(s_re)>>>(
      make_rows(dir_in, dir_out, idx_in, idx_out, idx_site, corners, sites, site_offsets, site_facets, row_valid),
      law, s_re.mutable_data_ptr<float>(), s_im.mutable_data_ptr<float>(), shapes.rows);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void facet_response_jvp_cuda(
    const torch::stable::Tensor& dir_in, const torch::stable::Tensor& dir_out,
    const torch::stable::Tensor& idx_in, const torch::stable::Tensor& idx_out,
    const torch::stable::Tensor& idx_site, const torch::stable::Tensor& corners,
    const torch::stable::Tensor& sites, const torch::stable::Tensor& site_offsets,
    const torch::stable::Tensor& site_facets, const torch::stable::Tensor& facet_material,
    const torch::stable::Tensor& material_params, const torch::stable::Tensor& layer_offsets,
    const torch::stable::Tensor& layers, const torch::stable::Tensor& row_valid,
    const torch::stable::Tensor& tan_dir_in, const torch::stable::Tensor& tan_dir_out,
    const torch::stable::Tensor& tan_corners, const torch::stable::Tensor& tan_sites,
    torch::stable::Tensor tan_re, torch::stable::Tensor tan_im, double wavenumber, double pol_x, double pol_y,
    double pol_z, int64_t seed, int64_t num_rows) {
  const Shapes shapes = check_rows(
      dir_in, dir_out, idx_in, idx_out, idx_site, corners, sites, site_offsets, site_facets, facet_material,
      material_params, layer_offsets, layers, row_valid, num_rows);
  check_count(tan_dir_in, dir_in.numel(), "tan_dir_in");
  check_count(tan_dir_out, dir_out.numel(), "tan_dir_out");
  check_count(tan_corners, corners.numel(), "tan_corners");
  check_count(tan_sites, sites.numel(), "tan_sites");
  check_count(tan_re, shapes.rows, "tan_re");
  check_count(tan_im, shapes.rows, "tan_im");
  const Law law = make_law(facet_material, material_params, layer_offsets, layers, wavenumber, pol_x, pol_y, pol_z, seed);
  if (shapes.rows == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(tan_re.get_device_index());
  facet_response_jvp_kernel<<<linear_grid(shapes.rows), dim3(kBlock, 1, 1), 0, current_cuda_stream(tan_re)>>>(
      make_rows(dir_in, dir_out, idx_in, idx_out, idx_site, corners, sites, site_offsets, site_facets, row_valid),
      law, tan_dir_in.const_data_ptr<float>(), tan_dir_out.const_data_ptr<float>(),
      tan_corners.const_data_ptr<float>(), tan_sites.const_data_ptr<float>(), tan_re.mutable_data_ptr<float>(),
      tan_im.mutable_data_ptr<float>(), shapes.rows);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void facet_response_backward_cuda(
    const torch::stable::Tensor& dir_in, const torch::stable::Tensor& dir_out,
    const torch::stable::Tensor& idx_in, const torch::stable::Tensor& idx_out,
    const torch::stable::Tensor& idx_site, const torch::stable::Tensor& corners,
    const torch::stable::Tensor& sites, const torch::stable::Tensor& site_offsets,
    const torch::stable::Tensor& site_facets, const torch::stable::Tensor& facet_material,
    const torch::stable::Tensor& material_params, const torch::stable::Tensor& layer_offsets,
    const torch::stable::Tensor& layers, const torch::stable::Tensor& row_valid,
    const torch::stable::Tensor& by_in_offsets, const torch::stable::Tensor& by_in_rows,
    const torch::stable::Tensor& by_out_offsets, const torch::stable::Tensor& by_out_rows,
    const torch::stable::Tensor& by_site_offsets, const torch::stable::Tensor& by_site_rows,
    const torch::stable::Tensor& facet_site, const torch::stable::Tensor& grad_re,
    const torch::stable::Tensor& grad_im, torch::stable::Tensor grad_dir_in, torch::stable::Tensor grad_dir_out,
    torch::stable::Tensor grad_corners, torch::stable::Tensor grad_sites, double wavenumber, double pol_x,
    double pol_y, double pol_z, int64_t seed, int64_t num_rows) {
  const Shapes shapes = check_rows(
      dir_in, dir_out, idx_in, idx_out, idx_site, corners, sites, site_offsets, site_facets, facet_material,
      material_params, layer_offsets, layers, row_valid, num_rows);
  const int legs_in = checked_int(dir_in.numel() / 3, "num_in");
  const int legs_out = checked_int(dir_out.numel() / 3, "num_out");
  const int facets = checked_int(shapes.facets, "num_facets");
  const int num_sites = checked_int(shapes.sites, "num_sites");
  check_long_count(by_in_offsets, static_cast<int64_t>(legs_in) + 1, "by_in_offsets");
  check_long_count(by_out_offsets, static_cast<int64_t>(legs_out) + 1, "by_out_offsets");
  check_long_count(by_site_offsets, static_cast<int64_t>(num_sites) + 1, "by_site_offsets");
  check_long_count(by_in_rows, shapes.rows, "by_in_rows");
  check_long_count(by_out_rows, shapes.rows, "by_out_rows");
  check_long_count(by_site_rows, shapes.rows, "by_site_rows");
  check_long_count(facet_site, facets, "facet_site");
  check_count(grad_re, shapes.rows, "grad_re");
  check_count(grad_im, shapes.rows, "grad_im");
  check_count(grad_dir_in, dir_in.numel(), "grad_dir_in");
  check_count(grad_dir_out, dir_out.numel(), "grad_dir_out");
  check_count(grad_corners, corners.numel(), "grad_corners");
  check_count(grad_sites, sites.numel(), "grad_sites");
  const Law law = make_law(facet_material, material_params, layer_offsets, layers, wavenumber, pol_x, pol_y, pol_z, seed);
  const int total = legs_in + legs_out + facets + num_sites;
  if (total == 0) return;
  const Owners owners = {
      by_in_offsets.const_data_ptr<int64_t>(),  by_in_rows.const_data_ptr<int64_t>(),
      by_out_offsets.const_data_ptr<int64_t>(), by_out_rows.const_data_ptr<int64_t>(),
      by_site_offsets.const_data_ptr<int64_t>(), by_site_rows.const_data_ptr<int64_t>(),
      facet_site.const_data_ptr<int64_t>()};
  const torch::stable::accelerator::DeviceGuard guard(grad_corners.get_device_index());
  facet_response_backward_kernel<<<linear_grid(total), dim3(kBlock, 1, 1), 0, current_cuda_stream(grad_corners)>>>(
      make_rows(dir_in, dir_out, idx_in, idx_out, idx_site, corners, sites, site_offsets, site_facets, row_valid),
      law, owners, grad_re.const_data_ptr<float>(), grad_im.const_data_ptr<float>(),
      grad_dir_in.mutable_data_ptr<float>(), grad_dir_out.mutable_data_ptr<float>(),
      grad_corners.mutable_data_ptr<float>(), grad_sites.mutable_data_ptr<float>(), legs_in, legs_out, facets,
      num_sites);
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
