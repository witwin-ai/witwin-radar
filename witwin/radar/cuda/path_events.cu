// Path birth and death detection between consecutive world observations.
//
// One family, non-differentiable by contract: an event is a discrete change of
// a row's existence, so there is no backward or JVP companion to register, and
// the Python facade passes detached payloads (witwin/radar/events.py).
//
//  path_events  one thread per CANDIDATE: a pair (before, after) naming the
//               same round-trip identity at two consecutive observations. The
//               host authors the candidate table once per pair of frozen joins
//               (a rediscovery that keeps the topology reuses its join, so the
//               table is a function of the two joins and the slot layout
//               alone). Each index is encoded:
//
//                 before == -1   the identity did not exist before;
//                 before >= 0    row `before` of the PREVIOUS pool;
//                 before <= -2   row `-2 - before` of the CURRENT pool (the
//                                previous slot of the same slot-major batch);
//                 after  == -1   the identity does not exist now;
//                 after  >= 0    row `after` of the CURRENT pool.
//
//               A row exists when it is present and its row_valid flag is set.
//               A birth is "did not exist before, exists now"; a death is the
//               reverse. Each event appends one record through an atomic
//               counter: `code = 2 * (base + candidate) + birth` and the
//               amplitude |C| of the live side (the new row for a birth, the
//               last live row for a death), in the composed transfer's units.
//               A record past `capacity` is counted but not written, so the
//               host can report how many were dropped rather than lose them
//               silently. Records are unordered; the host sorts by code.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cuda_runtime.h>
#include "radar_checks.cuh"

#include <cstdint>

namespace {

struct Side {
  bool live;
  float amplitude;
};

__device__ Side read_side(
    int64_t index, bool current_only, const bool* __restrict__ prev_valid, const float* __restrict__ prev_transfer,
    const bool* __restrict__ cur_valid, const float* __restrict__ cur_transfer) {
  Side side{false, 0.0f};
  if (index == -1) return side;
  const bool* valid = cur_valid;
  const float* transfer = cur_transfer;
  int64_t row = index;
  if (!current_only) {
    if (index >= 0) {
      valid = prev_valid;
      transfer = prev_transfer;
    } else {
      row = -2 - index;
    }
  }
  side.live = valid[row];
  side.amplitude = hypotf(transfer[2 * row], transfer[2 * row + 1]);
  return side;
}

__global__ void path_events_kernel(
    const bool* __restrict__ prev_valid, const float* __restrict__ prev_transfer, const bool* __restrict__ cur_valid,
    const float* __restrict__ cur_transfer, const int64_t* __restrict__ before, const int64_t* __restrict__ after,
    int64_t* __restrict__ event_code, float* __restrict__ event_amplitude, int32_t* counter, int candidates,
    int capacity, int64_t base) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= candidates) return;
  const Side was = read_side(before[c], false, prev_valid, prev_transfer, cur_valid, cur_transfer);
  const Side now = read_side(after[c], true, prev_valid, prev_transfer, cur_valid, cur_transfer);
  if (was.live == now.live) return;
  const int slot = atomicAdd(counter, 1);
  if (slot >= capacity) return;
  const int64_t birth = now.live ? 1 : 0;
  event_code[slot] = 2 * (base + c) + birth;
  event_amplitude[slot] = now.live ? now.amplitude : was.amplitude;
}

void check_cuda_bool(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Bool, name, " must have dtype torch.bool.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void check_pool(const torch::stable::Tensor& valid, const torch::stable::Tensor& transfer, const char* name) {
  check_cuda_bool(valid, name);
  check_cuda_float(transfer, "transfer");
  STD_TORCH_CHECK(
      transfer.dim() == 2 && transfer.size(1) == 2 && transfer.size(0) == valid.numel(),
      name, " transfer must be [rows, 2] float32, one (real, imaginary) pair per validity flag.");
}

void path_events_forward_cuda(
    const torch::stable::Tensor& prev_valid,
    const torch::stable::Tensor& prev_transfer,
    const torch::stable::Tensor& cur_valid,
    const torch::stable::Tensor& cur_transfer,
    const torch::stable::Tensor& before,
    const torch::stable::Tensor& after,
    torch::stable::Tensor event_code,
    torch::stable::Tensor event_amplitude,
    torch::stable::Tensor counter,
    int64_t base) {
  check_pool(prev_valid, prev_transfer, "prev_valid");
  check_pool(cur_valid, cur_transfer, "cur_valid");
  check_cuda_long(before, "before");
  check_cuda_long(after, "after");
  check_cuda_long(event_code, "event_code");
  check_cuda_float(event_amplitude, "event_amplitude");
  check_cuda_int(counter, "counter");
  STD_TORCH_CHECK(before.numel() == after.numel(), "before and after must hold one entry per candidate.");
  STD_TORCH_CHECK(event_code.numel() == event_amplitude.numel(), "the event buffers must share one capacity.");
  STD_TORCH_CHECK(counter.numel() == 1, "counter must hold one int32.");
  STD_TORCH_CHECK(base >= 0, "base must be non-negative.");
  const int device = cur_valid.get_device_index();
  const torch::stable::Tensor* tensors[] = {
      &prev_valid, &prev_transfer, &cur_transfer, &before, &after, &event_code, &event_amplitude, &counter};
  for (const torch::stable::Tensor* tensor : tensors) {
    STD_TORCH_CHECK(tensor->get_device_index() == device, "path_events tensors must share a CUDA device.");
  }
  const int candidates = checked_int(before.numel(), "candidates");
  const int capacity = checked_int(event_code.numel(), "capacity");
  if (candidates == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(device);
  const cudaStream_t stream = current_cuda_stream(counter);
  path_events_kernel<<<linear_grid(candidates), dim3(kBlock, 1, 1), 0, stream>>>(
      prev_valid.const_data_ptr<bool>(), prev_transfer.const_data_ptr<float>(), cur_valid.const_data_ptr<bool>(),
      cur_transfer.const_data_ptr<float>(), before.const_data_ptr<int64_t>(), after.const_data_ptr<int64_t>(),
      event_code.mutable_data_ptr<int64_t>(), event_amplitude.mutable_data_ptr<float>(),
      counter.mutable_data_ptr<int32_t>(), candidates, capacity, base);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("path_events_forward", TORCH_BOX(&path_events_forward_cuda));
}
