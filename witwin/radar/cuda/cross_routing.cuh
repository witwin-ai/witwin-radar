#pragma once

// Routing and the three launches of every cross-radar family (R-ADR-051).
//
// A cross family synthesizes what one receiving radar hears from transmitters
// that are not its own. Its rows are one replay group's composed rows, read in
// place, slot major (row k is join row k % rows_per_slot). `row_link[k %
// rows_per_slot]` names the row's link (-1: not heard this way, skipped) and
// `row_tx` its transmitter's index within its own radar. An output cell is one
// receive cell (the receiver's slot or symbol or pulse, one receive element)
// and reads the half-open row range `segment_rows[s]`; cells may share rows.
// `segment_schedule[s]` (float64, kCrossScheduleWidth) is the cell's slot start
// in the receiver's frame [s], its first sample's offset from that start [s]
// and the receiver's own local-oscillator offset for that cell [Hz]; sample n
// adds n * sample_period_s.
//
// What a family owns is its Phase functor: for one row at one sample it says
// whether the row is heard there, and returns the unit-weight contribution u
// and du/dtau [1/s]. The cell value is sum_k W(C_k) u_k, with W the identity
// (Channel convention: sampled receivers) or the conjugate (the FMCW beat).
// The three launches below are therefore one owner of the routing, the sums
// and the chain rule:
//
//   forward: y   = sum W(C) u
//   jvp:     dy  = sum W(dC) u + W(C) du/dtau dtau
//   vjp:     dL/dRe C = Re(conj(g) u), dL/dIm C = Re(conj(g) s u)  (s = +j identity, -j conjugate)
//            dL/dtau  = Re(conj(g) W(C) du/dtau)
//
// The backward is one thread per row over the cells that read it (the inverse
// CSR `row_segment_offsets` / `row_segments`), so no atomics and a fixed
// summation order. The gates a functor applies are discrete; the derivative is
// the almost-everywhere one.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_runtime.h>
#include "path_delay.cuh"
#include "radar_checks.cuh"

#include <cstdint>

constexpr int kCrossScheduleWidth = 3;

struct CrossRows {
  const float* transfer;  // [rows, 2] Channel convention C
  const float* tau_hi;
  const float* tau_lo;
  const bool* valid;
  const int32_t* row_link;
  const int32_t* row_tx;
  const double* link_table;
  int link_width;
  int rows;
  int rows_per_slot;
};

// A row's contribution per unit weight at one sample, and its tau derivative.
struct CrossValue {
  float u_re, u_im;
  float du_re, du_im;  // d u / d tau, 1/s
};

// Where and when one sample is: the time since the receiver's frame start [s],
// the offset from the cell's slot start [s] and the receiver's LO offset [Hz].
struct CrossSample {
  double t;
  double v;
  double lo;
};

__device__ __forceinline__ const double* cross_link(const CrossRows& rows, const int64_t k, int& tx) {
  const int64_t local = k % rows.rows_per_slot;
  const int link = rows.row_link[local];
  if (link < 0 || (rows.valid != nullptr && !rows.valid[k])) return nullptr;
  tx = rows.row_tx[local];
  return rows.link_table + static_cast<int64_t>(rows.link_width) * link;
}

__device__ __forceinline__ CrossSample cross_sample(
    const double* __restrict__ schedule, const int64_t segment, const int sample, const double sample_period_s) {
  const double* cell = schedule + kCrossScheduleWidth * segment;
  const double v = cell[1] + static_cast<double>(sample) * sample_period_s;
  return {cell[0] + v, v, cell[2]};
}

// W(C) for the identity (Conjugate == false) or the conjugate weight.
template <bool Conjugate>
__device__ __forceinline__ void cross_weight(const float* pair, float& re, float& im) {
  re = pair[0];
  im = Conjugate ? -pair[1] : pair[1];
}

template <bool Conjugate, bool Tangent, class Phase>
__global__ void cross_cell_kernel(
    const CrossRows rows, const Phase phase, const int64_t* __restrict__ segment_rows,
    const double* __restrict__ segment_schedule, const float* __restrict__ tan_transfer,
    const float* __restrict__ tan_tau, float* __restrict__ out, const int num_segments, const int num_samples,
    const double sample_period_s) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  if (sample >= num_samples || segment >= num_segments) return;
  int64_t begin = segment_rows[2 * segment];
  int64_t end = segment_rows[2 * segment + 1];
  begin = begin < 0 ? 0 : begin;
  end = end > rows.rows ? rows.rows : end;
  const CrossSample when = cross_sample(segment_schedule, segment, sample, sample_period_s);
  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = begin; k < end; ++k) {
    int tx = 0;
    const double* link = cross_link(rows, k, tx);
    if (link == nullptr) continue;
    CrossValue value;
    if (!phase(link, tx, path_delay(rows.tau_hi, rows.tau_lo, k), when, value)) continue;
    float c_re, c_im;
    if (Tangent) {
      cross_weight<Conjugate>(tan_transfer + 2 * k, c_re, c_im);
      float w_re, w_im;
      cross_weight<Conjugate>(rows.transfer + 2 * k, w_re, w_im);
      const float dtau = tan_tau[k];
      const float d_re = value.du_re * dtau;
      const float d_im = value.du_im * dtau;
      acc_re += c_re * value.u_re - c_im * value.u_im + w_re * d_re - w_im * d_im;
      acc_im += c_re * value.u_im + c_im * value.u_re + w_re * d_im + w_im * d_re;
    } else {
      cross_weight<Conjugate>(rows.transfer + 2 * k, c_re, c_im);
      acc_re += c_re * value.u_re - c_im * value.u_im;
      acc_im += c_re * value.u_im + c_im * value.u_re;
    }
  }
  const int64_t index = 2 * (static_cast<int64_t>(segment) * num_samples + sample);
  out[index] = acc_re;
  out[index + 1] = acc_im;
}

template <bool Conjugate, class Phase>
__global__ void cross_backward_kernel(
    const CrossRows rows, const Phase phase, const int64_t* __restrict__ row_segment_offsets,
    const int64_t* __restrict__ row_segments, const double* __restrict__ segment_schedule,
    const float* __restrict__ grad_out, float* __restrict__ grad_transfer, float* __restrict__ grad_tau,
    const int num_segments, const int num_samples, const double sample_period_s) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= rows.rows) return;
  double d_re = 0.0, d_im = 0.0, d_tau = 0.0;
  int tx = 0;
  const double* link = cross_link(rows, k, tx);
  if (link != nullptr) {
    const double tau = path_delay(rows.tau_hi, rows.tau_lo, k);
    float w_re, w_im;
    cross_weight<Conjugate>(rows.transfer + 2 * k, w_re, w_im);
    // d W(C) / d Im C = s, the unit the imaginary part enters with.
    const double s = Conjugate ? -1.0 : 1.0;
    for (int64_t entry = row_segment_offsets[k]; entry < row_segment_offsets[k + 1]; ++entry) {
      int64_t segment = row_segments[entry];
      segment = segment < 0 ? 0 : (segment >= num_segments ? num_segments - 1 : segment);
      const float* g = grad_out + 2 * segment * static_cast<int64_t>(num_samples);
      for (int sample = 0; sample < num_samples; ++sample) {
        CrossValue value;
        if (!phase(link, tx, tau, cross_sample(segment_schedule, segment, sample, sample_period_s), value)) continue;
        const double g_re = g[2 * sample];
        const double g_im = g[2 * sample + 1];
        // Re(conj(g) u) and Re(conj(g) j u) = -Im(conj(g) u).
        const double gu_re = g_re * value.u_re + g_im * value.u_im;
        const double gu_im = g_re * value.u_im - g_im * value.u_re;
        d_re += gu_re;
        d_im += -s * gu_im;
        const double t_re = static_cast<double>(w_re) * value.du_re - static_cast<double>(w_im) * value.du_im;
        const double t_im = static_cast<double>(w_re) * value.du_im + static_cast<double>(w_im) * value.du_re;
        d_tau += g_re * t_re + g_im * t_im;
      }
    }
  }
  grad_transfer[2 * k] = static_cast<float>(d_re);
  grad_transfer[2 * k + 1] = static_cast<float>(d_im);
  grad_tau[k] = static_cast<float>(d_tau);
}

// -- host side -----------------------------------------------------------------

inline void check_cross_double(const torch::stable::Tensor& table, int64_t width, const char* name) {
  STD_TORCH_CHECK(table.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(table.scalar_type() == torch::headeronly::ScalarType::Double, name, " must have dtype torch.float64.");
  STD_TORCH_CHECK(table.is_contiguous(), name, " must be contiguous.");
  STD_TORCH_CHECK(table.dim() == 2 && table.size(1) == width, name, " has the wrong row width.");
}

inline CrossRows cross_rows(
    const torch::stable::Tensor& transfer, const torch::stable::Tensor& tau_rt, const torch::stable::Tensor& tau_lo,
    const torch::stable::Tensor& valid, const torch::stable::Tensor& row_link, const torch::stable::Tensor& row_tx,
    const torch::stable::Tensor& link_table, int64_t link_width, int64_t rows_per_slot) {
  check_cuda_float(transfer, "transfer");
  check_cuda_float(tau_rt, "tau_rt");
  const int rows = checked_int(tau_rt.numel(), "rows");
  STD_TORCH_CHECK(transfer.numel() == 2 * static_cast<int64_t>(rows), "transfer must be (rows, 2).");
  if (valid.numel() != 0) {
    STD_TORCH_CHECK(
        valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::headeronly::ScalarType::Bool &&
            valid.numel() == rows,
        "valid must be empty or one contiguous torch.bool CUDA value per row.");
  }
  check_cuda_int(row_link, "row_link");
  check_cuda_int(row_tx, "row_tx");
  const int per_slot = checked_int(rows_per_slot, "rows_per_slot");
  STD_TORCH_CHECK(per_slot > 0 || rows == 0, "rows_per_slot must be positive.");
  STD_TORCH_CHECK(
      row_link.numel() == per_slot && row_tx.numel() == per_slot, "row_link and row_tx hold one value per row of a slot.");
  STD_TORCH_CHECK(per_slot == 0 || rows % per_slot == 0, "rows must be whole slots of rows_per_slot.");
  check_cross_double(link_table, link_width, "link_table");
  return {transfer.const_data_ptr<float>(),
          tau_rt.const_data_ptr<float>(),
          delay_residual(tau_lo, tau_rt),
          valid.numel() == 0 ? nullptr : valid.const_data_ptr<bool>(),
          row_link.const_data_ptr<int32_t>(),
          row_tx.const_data_ptr<int32_t>(),
          link_table.const_data_ptr<double>(),
          static_cast<int>(link_width),
          rows,
          per_slot > 0 ? per_slot : 1};
}

inline void check_cross_cells(
    const torch::stable::Tensor& segment_rows, const torch::stable::Tensor& segment_schedule,
    const torch::stable::Tensor& out, int segments, int samples, const char* name) {
  check_cuda_long(segment_rows, "segment_rows");
  STD_TORCH_CHECK(segment_rows.numel() == 2 * static_cast<int64_t>(segments), "segment_rows must be (segments, 2).");
  check_cross_double(segment_schedule, kCrossScheduleWidth, "segment_schedule");
  STD_TORCH_CHECK(segment_schedule.size(0) == segments, "segment_schedule must hold one row per segment.");
  check_cuda_float(out, name);
  STD_TORCH_CHECK(out.numel() == 2 * static_cast<int64_t>(segments) * samples, name, " must be (segments, samples, 2).");
}

// The three launches, each a few lines for a family to call with its functor.
template <bool Conjugate, class Phase>
void cross_forward(
    const CrossRows& rows, const Phase& phase, const torch::stable::Tensor& segment_rows,
    const torch::stable::Tensor& segment_schedule, torch::stable::Tensor& out, int64_t num_samples,
    double sample_period_s) {
  const int segments = checked_int(segment_schedule.size(0), "segments");
  const int samples = checked_int(num_samples, "num_samples");
  check_cross_cells(segment_rows, segment_schedule, out, segments, samples, "out");
  if (segments == 0 || samples == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(out.get_device_index());
  cross_cell_kernel<Conjugate, false, Phase><<<sample_grid(samples, segments, 1, kBlock), dim3(kBlock, 1, 1), 0,
                                              current_cuda_stream(out)>>>(
      rows, phase, segment_rows.const_data_ptr<int64_t>(), segment_schedule.const_data_ptr<double>(), nullptr,
      nullptr, out.mutable_data_ptr<float>(), segments, samples, sample_period_s);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool Conjugate, class Phase>
void cross_jvp(
    const CrossRows& rows, const Phase& phase, const torch::stable::Tensor& segment_rows,
    const torch::stable::Tensor& segment_schedule, const torch::stable::Tensor& tan_transfer,
    const torch::stable::Tensor& tan_tau, torch::stable::Tensor& tan_out, int64_t num_samples,
    double sample_period_s) {
  const int segments = checked_int(segment_schedule.size(0), "segments");
  const int samples = checked_int(num_samples, "num_samples");
  check_cross_cells(segment_rows, segment_schedule, tan_out, segments, samples, "tan_out");
  check_cuda_float(tan_transfer, "tan_transfer");
  check_cuda_float(tan_tau, "tan_tau");
  STD_TORCH_CHECK(
      tan_transfer.numel() == 2 * static_cast<int64_t>(rows.rows) && tan_tau.numel() == rows.rows,
      "the tangents must match the rows they perturb.");
  if (segments == 0 || samples == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(tan_out.get_device_index());
  cross_cell_kernel<Conjugate, true, Phase><<<sample_grid(samples, segments, 1, kBlock), dim3(kBlock, 1, 1), 0,
                                             current_cuda_stream(tan_out)>>>(
      rows, phase, segment_rows.const_data_ptr<int64_t>(), segment_schedule.const_data_ptr<double>(),
      tan_transfer.const_data_ptr<float>(), tan_tau.const_data_ptr<float>(), tan_out.mutable_data_ptr<float>(),
      segments, samples, sample_period_s);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool Conjugate, class Phase>
void cross_backward(
    const CrossRows& rows, const Phase& phase, const torch::stable::Tensor& row_segment_offsets,
    const torch::stable::Tensor& row_segments, const torch::stable::Tensor& segment_schedule,
    const torch::stable::Tensor& grad_out, torch::stable::Tensor& grad_transfer, torch::stable::Tensor& grad_tau,
    int64_t num_samples, double sample_period_s) {
  const int segments = checked_int(segment_schedule.size(0), "segments");
  const int samples = checked_int(num_samples, "num_samples");
  check_cross_double(segment_schedule, kCrossScheduleWidth, "segment_schedule");
  check_cuda_long(row_segment_offsets, "row_segment_offsets");
  check_cuda_long(row_segments, "row_segments");
  STD_TORCH_CHECK(
      row_segment_offsets.numel() == static_cast<int64_t>(rows.rows) + 1,
      "row_segment_offsets must hold rows + 1 values.");
  check_cuda_float(grad_out, "grad_out");
  STD_TORCH_CHECK(
      grad_out.numel() == 2 * static_cast<int64_t>(segments) * samples, "grad_out must be (segments, samples, 2).");
  check_cuda_float(grad_transfer, "grad_transfer");
  check_cuda_float(grad_tau, "grad_tau");
  STD_TORCH_CHECK(
      grad_transfer.numel() == 2 * static_cast<int64_t>(rows.rows) && grad_tau.numel() == rows.rows,
      "the gradients must match the rows they belong to.");
  if (rows.rows == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(grad_tau.get_device_index());
  cross_backward_kernel<Conjugate, Phase><<<linear_grid(rows.rows), dim3(kBlock, 1, 1), 0,
                                            current_cuda_stream(grad_tau)>>>(
      rows, phase, row_segment_offsets.const_data_ptr<int64_t>(), row_segments.const_data_ptr<int64_t>(),
      segment_schedule.const_data_ptr<double>(), grad_out.const_data_ptr<float>(),
      grad_transfer.mutable_data_ptr<float>(), grad_tau.mutable_data_ptr<float>(), segments, samples,
      sample_period_s);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}
