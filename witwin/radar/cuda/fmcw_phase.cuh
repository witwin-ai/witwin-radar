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

// The two per-chirp extensions of the same phase (R-ADR-047), added to it rather
// than folded in so the uniform ramp keeps its exact arithmetic:
//
//   start offset f0 [Hz]: a chirp whose ramp starts at f_ref + f0. The weight
//     carries only f_ref, so f0 multiplies the FULL delay: + f0 * tau.
//   transmit phase error E(t) [cycles], t = time since the ramp start [s]:
//     E(t) = sum_i poly[i] * t^(i + 2). The beat is tx(u) * conj(tx(u - tau)),
//     so the error enters as E(u) - E(u - tau). Orders 0 and 1 are a constant
//     phase and a frequency offset, not a nonlinearity, and are not in poly.
//
// Derivatives with respect to tau, at fixed slot and ADC time: f0 + E'(u - tau),
// and the delay moves with tau0 by 1 and with the rate by (slot + u).
__device__ __forceinline__ FmcwPhaseTerms fmcw_profile_terms(
    double tau0, double rate, double slot, double u, double slope,
    double carrier, double carrier_rate, double offset, const double* poly, int poly_len) {
  constexpr double two_pi = 6.283185307179586476925286766559;
  FmcwPhaseTerms terms = fmcw_phase_terms(tau0, rate, slot, u, slope, carrier, carrier_rate);
  if (offset == 0.0 && poly_len == 0) {
    return terms;
  }
  const double time = slot + u;
  const double tau = tau0 + rate * time;
  const double lagged = u - tau;
  double error_now = 0.0, error_lagged = 0.0, slope_lagged = 0.0;
  for (int i = poly_len - 1; i >= 0; --i) {
    error_now = (error_now + poly[i]) * u;
    error_lagged = (error_lagged + poly[i]) * lagged;
    slope_lagged = slope_lagged * lagged + (i + 2) * poly[i];
  }
  error_now *= u;
  error_lagged *= lagged;
  slope_lagged *= lagged;
  const double derivative = offset + slope_lagged;
  terms.cycles += offset * tau + error_now - error_lagged;
  terms.d_tau += two_pi * derivative;
  terms.d_rate += two_pi * time * derivative;
  return terms;
}

// Cross-radar beat (R-ADR-051): receiver B's ramp mixed against a transmitter
// that is not B's own, heard through one composed row of round-trip delay tau
// [s]. Every oscillator runs at f_ref + its start offset delta [Hz] from world
// time 0; the transmitter adds its own modulation phase tx_cycles at the
// emission instant (an FMCW ramp's 0.5 * S_A * w^2 over its ramp time w, or an
// emitter's sample-table phase), whose instantaneous frequency is tx_freq
// [Hz]. The receiver mixes its LO (0.5 * S_B * v^2 over its own ramp time v)
// against the conjugate of the received field, and the Channel weight conj(C)
// carries +f_ref*tau, so the f_ref terms cancel and the beat is
// conj(C) * conj(envelope magnitude) * exp(+j 2 pi cycles):
//
//   cycles = phase0 + delta_B*t - delta_A*(t - tau) + 0.5*S_B*v^2 - tx_cycles
//
// t is the receive instant in B's frame [s]; phase0 = frac((delta_B - delta_A) * F_B)
// carries B's frame start F_B in world time [cycles], formed on the host in
// float64. With A = B on one slot (tx_cycles = 0.5*S*(v - tau)^2) this is the
// monostatic phase delta*tau + S*tau*(v - tau/2) exactly. The emission instant
// falls by tau, so d(cycles)/d(tau) = delta_A + tx_freq; beat_hz = d(cycles)/dt
// is the instantaneous beat frequency the receiver's IF band gates.
// Independent oracle: tests/test_fmcw_cross.py (long-double signals).
struct FmcwCrossTerms { double cycles, d_tau, beat_hz; };
__device__ __forceinline__ FmcwCrossTerms fmcw_cross_terms(
    double tau, double t, double v, double tx_cycles, double tx_freq, double rx_slope, double rx_offset,
    double tx_offset, double phase0) {
  constexpr double two_pi = 6.283185307179586476925286766559;
  return {phase0 + rx_offset * t - tx_offset * (t - tau) + 0.5 * rx_slope * v * v - tx_cycles,
          two_pi * (tx_offset + tx_freq),
          rx_offset - tx_offset + rx_slope * v - tx_freq};
}
