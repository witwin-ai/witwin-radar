#pragma once

// TDM-MIMO slow time and the (chirp, segment, fast-time) grid contract shared
// by the FMCW beat and spectrum families. Slots run in a single sequence
// across the frame: slot `chirp * num_tx + tx` starts `slot * chirp_period_s`
// into it, so two sensor pairs driven by different transmitters within the
// same chirp index are a whole chirp period apart in slow time. `num_tx = 1`
// with a zero table reduces the slot to the chirp index exactly.

#include "radar_checks.cuh"

__device__ __forceinline__ double slot_time(
    const int chirp,
    const int tx_index,
    const int num_tx,
    const double chirp_period_s) {
  const int64_t slot = static_cast<int64_t>(chirp) * num_tx + tx_index;
  return static_cast<double>(slot) * chirp_period_s;
}

// The same memory-safety backstop as segment_bounds, on the per-segment
// transmitter table: the host wrapper checks the table's shape but never reads
// its VALUES. TwoWayComposer.freeze validates the partition on the host at
// freeze time, where the table is still a Python list and checking it is free.
__device__ __forceinline__ int clamped_tx_index(
    const int32_t* __restrict__ segment_tx_index,
    const int segment,
    const int num_tx) {
  int tx = static_cast<int>(segment_tx_index[segment]);
  tx = tx < 0 ? 0 : tx;
  return tx >= num_tx ? num_tx - 1 : tx;
}

inline void check_tdm(
    const torch::stable::Tensor& segment_tx_index,
    int num_segments,
    int num_tx) {
  STD_TORCH_CHECK(num_tx > 0, "num_tx must be positive.");
  check_cuda_int(segment_tx_index, "segment_tx_index");
  STD_TORCH_CHECK(
      segment_tx_index.numel() == static_cast<int64_t>(num_segments),
      "segment_tx_index must hold one transmitter index per sensor-pair segment.");
}

// `fast_axis` names the third axis in the message: "samples" for the beat
// cube, "bins" for the spectrum cube.
inline void check_output(
    const torch::stable::Tensor& out_re,
    const torch::stable::Tensor& out_im,
    int num_chirps,
    int num_segments,
    int num_fast,
    const char* fast_axis,
    const char* name_re,
    const char* name_im) {
  check_cuda_float(out_re, name_re);
  check_cuda_float(out_im, name_im);
  STD_TORCH_CHECK(
      out_re.sizes().equals(out_im.sizes()),
      name_re, " and ", name_im, " must have the same shape.");
  STD_TORCH_CHECK(
      out_re.dim() == 3,
      name_re, " must have shape (chirps, segments, ", fast_axis, ").");
  STD_TORCH_CHECK(
      out_re.size(0) == num_chirps && out_re.size(1) == num_segments &&
          out_re.size(2) == num_fast,
      name_re, " shape disagrees with the declared grid.");
}
