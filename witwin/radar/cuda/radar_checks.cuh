#pragma once

// Host argument checks and device index helpers shared by every radar
// translation unit. Each has exactly one definition: a dtype message or a
// clamp rule that drifted in one kernel and not another would be a contract
// change no single test file is positioned to see.

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

constexpr double kTwoPiD = 6.283185307179586476925286766559;

// Block size of the one-dimensional launches over composed rows or owner slots.
constexpr int kBlock = 256;

inline dim3 linear_grid(int count) {
  return dim3(static_cast<unsigned>((count + kBlock - 1) / kBlock), 1, 1);
}

// (fast-time, segment, slow-time) launch grid of the synthesis families.
inline dim3 sample_grid(int num_samples, int num_segments, int num_slow, int block) {
  return dim3((num_samples + block - 1) / block, num_segments, num_slow);
}

inline void check_cuda_float(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Float,
      name,
      " must have dtype torch.float32.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

inline void check_cuda_int(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Int,
      name,
      " must have dtype torch.int32.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

inline void check_cuda_long(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Long,
      name,
      " must have dtype torch.int64.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

inline int checked_int(int64_t value, const char* name) {
  STD_TORCH_CHECK(
      value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<int>::max()),
      name,
      " is out of int32 range.");
  return static_cast<int>(value);
}

inline cudaStream_t current_cuda_stream(const torch::stable::Tensor& tensor) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

inline void check_index(
    const torch::stable::Tensor& index, int rows, const char* name) {
  check_cuda_long(index, name);
  STD_TORCH_CHECK(
      index.numel() == static_cast<int64_t>(rows),
      name,
      " must hold one index per composed row.");
}

// One path batch: delay, delay rate and the complex weight, one value per path.
inline void check_path_inputs(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    int num_paths) {
  check_cuda_float(tau_rt, "tau_rt");
  check_cuda_float(tau_rate, "tau_rate");
  check_cuda_float(weight_re, "weight_re");
  check_cuda_float(weight_im, "weight_im");
  STD_TORCH_CHECK(
      tau_rt.numel() == num_paths && tau_rate.numel() == num_paths &&
          weight_re.numel() == num_paths && weight_im.numel() == num_paths,
      "tau_rt, tau_rate, weight_re, and weight_im must each hold num_paths values.");
}

struct SegmentBounds {
  int64_t start;
  int64_t end;
};

// The path rows of one CSR segment, clamped into [0, num_paths]. A
// memory-safety backstop, not a validation policy: the host wrapper checks the
// offsets table's shape but never reads its VALUES, because doing so per frame
// would be the D2H the fixed-topology capability exists to avoid. Clamping
// keeps a malformed table from walking off the path arrays; it does NOT make
// the result meaningful.
__device__ __forceinline__ SegmentBounds segment_bounds(
    const int64_t* __restrict__ path_offsets, const int segment, const int num_paths) {
  int64_t start = path_offsets[segment];
  int64_t end = path_offsets[segment + 1];
  start = start < 0 ? 0 : start;
  end = end > num_paths ? num_paths : end;
  return {start, end};
}
