#pragma once

// Forward-mode jets for the scatter-response kernels: a value and N
// directional derivatives, carried through the same expression the forward
// evaluates. A response kernel's JVP is its expression on a one-lane jet
// seeded with the tangent; its VJP evaluates the expression on a jet seeded on
// one owner's inputs and contracts the lanes with the cotangent. Arithmetic is
// double throughout. N = 0 is the plain value.

#include <cuda_runtime.h>

#include <cstdint>

namespace radar_jets {

constexpr double kJetPi = 3.14159265358979323846264338327950288;

template <int N>
struct J {
  double v;
  double d[N > 0 ? N : 1];
};

template <int N>
__device__ __forceinline__ J<N> jconst(double value) {
  J<N> r;
  r.v = value;
#pragma unroll
  for (int i = 0; i < N; ++i) r.d[i] = 0.0;
  return r;
}

template <int N>
__device__ __forceinline__ J<N> operator+(const J<N>& a, const J<N>& b) {
  J<N> r;
  r.v = a.v + b.v;
#pragma unroll
  for (int i = 0; i < N; ++i) r.d[i] = a.d[i] + b.d[i];
  return r;
}

template <int N>
__device__ __forceinline__ J<N> operator-(const J<N>& a, const J<N>& b) {
  J<N> r;
  r.v = a.v - b.v;
#pragma unroll
  for (int i = 0; i < N; ++i) r.d[i] = a.d[i] - b.d[i];
  return r;
}

template <int N>
__device__ __forceinline__ J<N> operator-(const J<N>& a) {
  J<N> r;
  r.v = -a.v;
#pragma unroll
  for (int i = 0; i < N; ++i) r.d[i] = -a.d[i];
  return r;
}

template <int N>
__device__ __forceinline__ J<N> operator*(const J<N>& a, const J<N>& b) {
  J<N> r;
  r.v = a.v * b.v;
#pragma unroll
  for (int i = 0; i < N; ++i) r.d[i] = a.d[i] * b.v + a.v * b.d[i];
  return r;
}

template <int N>
__device__ __forceinline__ J<N> operator*(const J<N>& a, double s) {
  J<N> r;
  r.v = a.v * s;
#pragma unroll
  for (int i = 0; i < N; ++i) r.d[i] = a.d[i] * s;
  return r;
}

template <int N>
__device__ __forceinline__ J<N> operator/(const J<N>& a, const J<N>& b) {
  J<N> r;
  r.v = a.v / b.v;
#pragma unroll
  for (int i = 0; i < N; ++i) r.d[i] = (a.d[i] - r.v * b.d[i]) / b.v;
  return r;
}

// f(x) with its scalar derivative, lifted onto a jet.
template <int N>
__device__ __forceinline__ J<N> lift(const J<N>& x, double value, double slope) {
  J<N> r;
  r.v = value;
#pragma unroll
  for (int i = 0; i < N; ++i) r.d[i] = slope * x.d[i];
  return r;
}

template <int N>
__device__ __forceinline__ J<N> jsqrt(const J<N>& x) {
  const double root = sqrt(x.v);
  return lift(x, root, 0.5 / root);
}

template <int N>
__device__ __forceinline__ J<N> jexp(const J<N>& x) {
  const double e = exp(x.v);
  return lift(x, e, e);
}

template <int N>
__device__ __forceinline__ J<N> jsin(const J<N>& x) {
  return lift(x, sin(x.v), cos(x.v));
}

template <int N>
__device__ __forceinline__ J<N> jcos(const J<N>& x) {
  return lift(x, cos(x.v), -sin(x.v));
}

// sin(x) / x, by its series near zero so that value and slope stay smooth.
template <int N>
__device__ __forceinline__ J<N> jsinc(const J<N>& x) {
  const double t = x.v;
  if (fabs(t) < 1.0e-3) {
    const double t2 = t * t;
    return lift(x, 1.0 - t2 / 6.0 + t2 * t2 / 120.0, -t / 3.0 + t * t2 / 30.0);
  }
  const double s = sin(t);
  const double c = cos(t);
  return lift(x, s / t, (c * t - s) / (t * t));
}

template <int N>
struct C {
  J<N> re;
  J<N> im;
};

template <int N>
__device__ __forceinline__ C<N> cconst(double re, double im) {
  return {jconst<N>(re), jconst<N>(im)};
}

template <int N>
__device__ __forceinline__ C<N> operator+(const C<N>& a, const C<N>& b) {
  return {a.re + b.re, a.im + b.im};
}

template <int N>
__device__ __forceinline__ C<N> operator-(const C<N>& a, const C<N>& b) {
  return {a.re - b.re, a.im - b.im};
}

template <int N>
__device__ __forceinline__ C<N> operator*(const C<N>& a, const C<N>& b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

template <int N>
__device__ __forceinline__ C<N> operator*(const C<N>& a, const J<N>& s) {
  return {a.re * s, a.im * s};
}

template <int N>
__device__ __forceinline__ C<N> operator*(const C<N>& a, double s) {
  return {a.re * s, a.im * s};
}

template <int N>
__device__ __forceinline__ C<N> operator/(const C<N>& a, const C<N>& b) {
  const J<N> den = b.re * b.re + b.im * b.im;
  return {(a.re * b.re + a.im * b.im) / den, (a.im * b.re - a.re * b.im) / den};
}

// exp(j theta) for a real jet theta.
template <int N>
__device__ __forceinline__ C<N> cis(const J<N>& theta) {
  return {jcos(theta), jsin(theta)};
}

// Multiplication by j.
template <int N>
__device__ __forceinline__ C<N> times_j(const C<N>& a) {
  return {-a.im, a.re};
}

// exp(z) for a complex jet.
template <int N>
__device__ __forceinline__ C<N> cexp(const C<N>& z) {
  return cis(z.im) * jexp(z.re);
}

// The passive square root: the branch with a non-positive imaginary part, the
// decaying wave under exp(+j omega t). d sqrt(z) = dz / (2 sqrt(z)).
template <int N>
__device__ __forceinline__ C<N> csqrt_passive(const C<N>& z) {
  const double modulus = hypot(z.re.v, z.im.v);
  double wr = sqrt(0.5 * (modulus + z.re.v));
  double wi = sqrt(fmax(0.0, 0.5 * (modulus - z.re.v)));
  if (z.im.v < 0.0) wi = -wi;
  if (wi > 0.0) {
    wr = -wr;
    wi = -wi;
  }
  C<N> w;
  w.re = jconst<N>(wr);
  w.im = jconst<N>(wi);
  // dw = dz / (2 w)
  const double den = 2.0 * (wr * wr + wi * wi);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const double zr = z.re.d[i];
    const double zi = z.im.d[i];
    w.re.d[i] = (zr * wr + zi * wi) / den;
    w.im.d[i] = (zi * wr - zr * wi) / den;
  }
  return w;
}

// Outer jet of a complex function g(x) whose value and complex slope dg/dx
// were evaluated on an inner jet: chain rule into the outer lanes.
template <int N>
__device__ __forceinline__ C<N> chain(
    const J<N>& x, double gr, double gi, double sr, double si) {
  return {lift(x, gr, sr), lift(x, gi, si)};
}

template <int N>
struct V3 {
  J<N> x;
  J<N> y;
  J<N> z;
};

template <int N>
__device__ __forceinline__ V3<N> operator+(const V3<N>& a, const V3<N>& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <int N>
__device__ __forceinline__ V3<N> operator-(const V3<N>& a, const V3<N>& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template <int N>
__device__ __forceinline__ V3<N> operator*(const V3<N>& a, double s) {
  return {a.x * s, a.y * s, a.z * s};
}

template <int N>
__device__ __forceinline__ V3<N> scale(const V3<N>& a, const J<N>& s) {
  return {a.x * s, a.y * s, a.z * s};
}

template <int N>
__device__ __forceinline__ J<N> dot(const V3<N>& a, const V3<N>& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <int N>
__device__ __forceinline__ V3<N> cross(const V3<N>& a, const V3<N>& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// A row of a (rows, 3) float table as a jet vector with no derivative lanes set.
template <int N>
__device__ __forceinline__ V3<N> jload(const float* __restrict__ data, int64_t index) {
  const int64_t base = index * 3;
  return {jconst<N>(data[base]), jconst<N>(data[base + 1]), jconst<N>(data[base + 2])};
}

// Seed lanes lane, lane + 1, lane + 2 with the three components.
template <int N>
__device__ __forceinline__ void jseed(V3<N>& vector, int lane) {
  vector.x.d[lane] = 1.0;
  vector.y.d[lane + 1] = 1.0;
  vector.z.d[lane + 2] = 1.0;
}

// Load a tangent row into lane 0.
template <int N>
__device__ __forceinline__ void jtangent(V3<N>& vector, const float* __restrict__ data, int64_t index) {
  vector.x.d[0] = data[index * 3];
  vector.y.d[0] = data[index * 3 + 1];
  vector.z.d[0] = data[index * 3 + 2];
}

}  // namespace radar_jets
