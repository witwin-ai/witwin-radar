#pragma once

// FMCW dechirp convention: positive beat phase. SI units; tau0 is round-trip
// delay [s] at frame start, rate=d(tau)/dt [s/s], u is chirp-local ADC time [s].
// Linear-delay model includes fast time: tau=tau0+rate*(slot+u).
// Weight owns fc*tau0 when carrier_rate=fc; otherwise carrier=fc owns it.
// Exactly one carrier home is nonzero. Independent oracle: test_fmcw_continuous_motion.py.
struct FmcwPhaseTerms { double cycles, d_tau, d_rate; };
__device__ __forceinline__ FmcwPhaseTerms fmcw_phase_terms(
    double tau0, double rate, double slot, double u, double slope,
    double carrier, double carrier_rate) {
  constexpr double two_pi = 6.283185307179586476925286766559;
  const double time = slot + u;
  const double drift = rate * time;
  const double tau = tau0 + drift;
  const double derivative = carrier + slope * (u-tau);
  return {carrier*tau + carrier_rate*drift + slope*tau*(u-0.5*tau),
          two_pi*derivative, two_pi*time*(derivative+carrier_rate)};
}
