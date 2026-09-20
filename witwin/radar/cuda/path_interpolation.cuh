#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// The one carrier-transport interpolation equation, shared by standalone
// probe interpolation and fused observation synthesis. tau is round-trip
// delay [s], fc is Hz, C uses Channel's exp(-j*2*pi*fc*tau) convention.
// tau=sum(w_j*tau_j); C=sum(w_j*C_j*exp(-j*2*pi*fc*(tau-tau_j))).
// The dimensionless Lagrange weights sum to one and are fixed AD metadata.
// Valid only within a topology-identical, error-tested interval; this adds
// neither scattering nor spreading. Oracles: test_path_interpolation.py and
// test_adaptive_fmcw_fusion.py, including native VJP/JVP.
struct InterpolationNode { double tau, re, im, weight; };
struct InterpolatedPath { double tau, re, im; };

template<class Nodes>
__device__ __forceinline__ InterpolatedPath interpolate_path(const Nodes& read, int64_t nodes, double fc) {
  const double k=6.2831853071795864769*fc;
  double tau=0;
  for(int64_t j=0;j<nodes;++j) {
    const auto a=read(j);
    tau+=a.weight*a.tau;
  }
  double re=0,im=0;
  for(int64_t j=0;j<nodes;++j) {
    const auto a=read(j);
    double s,c; sincos(-k*(tau-a.tau),&s,&c);
    re+=a.weight*(a.re*c-a.im*s);
    im+=a.weight*(a.re*s+a.im*c);
  }
  return {tau,re,im};
}

__device__ __forceinline__ void interpolation_jacobian(
    InterpolationNode a, InterpolatedPath value, double fc, double jac[3][3]) {
  // d(theta_j)/d(tau_m)=-2*pi*fc*(w_m-delta_jm). Delay derivatives use
  // the full transported sum, not just the coefficient at this node.
  const double k=6.2831853071795864769*fc;
  double s,c; sincos(-k*(value.tau-a.tau),&s,&c);
  const double rj=a.re*c-a.im*s, ij=a.re*s+a.im*c, w=a.weight;
  jac[0][0]=w; jac[0][1]=0; jac[0][2]=0;
  jac[1][0]=k*w*(value.im-ij); jac[1][1]=w*c; jac[1][2]=-w*s;
  jac[2][0]=-k*w*(value.re-rj); jac[2][1]=w*s; jac[2][2]=w*c;
}
