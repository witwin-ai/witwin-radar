#pragma once

// A composed row's round-trip delay as the synthesis kernels read it.
//
// The join publishes the float64 delay as a float32 pair (R-ADR-040):
// tau = tau_hi + tau_lo, with tau_hi = float32(tau) the differentiable delay
// and tau_lo = float32(tau - tau_hi) its non-differentiable residual. Every
// phase and envelope term is formed in float64 from the pair, so a delay is
// resolved to ~2^-46 tau rather than the float32 ulp. An empty residual
// (nullptr) reads tau_hi alone, bit for bit as before. The derivative with
// respect to tau_hi is the derivative with respect to tau, evaluated there.

#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdint>

__device__ __forceinline__ double path_delay(const float* hi, const float* lo, const int64_t k) {
  return lo == nullptr ? static_cast<double>(hi[k]) : static_cast<double>(hi[k]) + static_cast<double>(lo[k]);
}

// The residual's device pointer, checked against the delay it refines; nullptr when empty.
inline const float* delay_residual(const torch::stable::Tensor& tau_lo, const torch::stable::Tensor& tau_rt) {
  if (tau_lo.numel() == 0) return nullptr;
  STD_TORCH_CHECK(
      tau_lo.is_cuda() && tau_lo.is_contiguous() && tau_lo.scalar_type() == torch::headeronly::ScalarType::Float &&
          tau_lo.numel() == tau_rt.numel() && tau_lo.get_device_index() == tau_rt.get_device_index(),
      "tau_lo must be empty or a contiguous float32 CUDA tensor with one value per path, on the paths' device.");
  return tau_lo.const_data_ptr<float>();
}
