// Time-domain OFDM: the received samples of a modulated symbol stream with a
// cyclic prefix, so intra-symbol motion produces inter-carrier interference and
// a long echo inter-symbol interference (R-ADR-046).
//
// The transmitted symbol j, of duration T_sym = T_cp + T_u with T_u = 1 / df,
// carries the useful waveform
//
//   x_j(t) = (1 / sqrt(N)) * sum_{n=0}^{N-1} X[j][n] * exp(+j 2 pi n df t),
//
// periodic in T_u, so its first T_cp seconds are the cyclic prefix x_j(w - T_cp)
// for w in [0, T_cp). With E|X|^2 = 1 the envelope has unit mean power, so a
// sample is C[k] times a dimensionless envelope in sqrt(W) (R-ADR-029), and the
// unitary N-point transform of the N post-prefix samples is exactly C X_m
// exp(-j 2 pi m df tau) for a still echo inside the prefix: the CFR kernel's
// value times the symbol.
//
// Received sample i < num_samples of output symbol l (after the prefix), in
// the Channel convention:
//
//   t_loc  = T_cp + i T_s             (or the per-segment table's start + i T_s)
//   t_abs  = l T_sym + t_loc          (time since the frame start)
//   drift  = tau_rate[k] * t_abs,     tau = tau_rt[k] + drift
//   u      = t_loc - tau              (time into transmit symbol l)
//   j      = l + floor(u / T_sym),    w = u - (j - l) T_sym in [0, T_sym)
//   cycles = -(carrier_rate_hz * drift + carrier_hz * tau)
//   y[l][p][i] = sum_k C[k] * x_j(w - T_cp) * exp(+j 2 pi cycles)
//
// j < l is the previous symbol: an echo longer than the prefix bleeds into the
// window (ISI). j < 0 is the silence before the frame and contributes zero.
// The delay WALKS across the symbol, which is what the CFR kernel cannot
// express: its tone changes inside the FFT window, spreads into the other bins,
// and the classic inter-carrier interference follows.
//
// `segment_table` (float64 [segments, 2]: the window start t_loc of sample 0
// [s] and the symbol index l) replaces the uniform grid per segment, so one
// launch covers many fast-time instants of a refreshed frame. `symbols` is the
// frame's whole table X, float32 [num_table_symbols, N, 2] (re, im); it is
// waveform data and carries no derivative.
//
// Derivatives (x' = dx/dt = (1/sqrt(N)) sum X_n (j 2 pi n df) exp(...)):
//
//   dy/dtau_rt   = C e^{j phi} (-x' - j 2 pi carrier_hz x)
//   dy/dtau_rate = t_abs * C e^{j phi} (-x' - j 2 pi (carrier_hz + carrier_rate_hz) x)
//
// The symbol switch at an ISI boundary is a genuine discontinuity; the
// almost-everywhere derivative is returned there.
//
// Numerics: the subcarrier sum runs a float rotation r^n, re-anchored every
// kAnchor subcarriers from a double-precision phase wrapped to [0, 1), so its
// error stays at a few float ulps whatever N is. The carrier cycles are summed
// in double and wrapped before a single sincosf.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cuda_runtime.h>
#include "radar_checks.cuh"
#include "path_delay.cuh"

#include <cmath>
#include <cstdint>

namespace {

constexpr int kAnchor = 32;

struct Grid {
  int num_table;
  int num_subcarriers;
  int num_samples;
  double df;
  double cp;
  double t_sym;
  double t_s;
  double inv_sqrt_n;
  const float* symbols;
  const double* segment_table;
};

struct Envelope {
  float x_re, x_im;    // x_j(w - T_cp)
  float dx_re, dx_im;  // dx/dt, 1/s
  bool live;
};

__device__ __forceinline__ Envelope envelope_at(const Grid& grid, const int l, const double u) {
  Envelope env{0.0f, 0.0f, 0.0f, 0.0f, false};
  const double shift = floor(u / grid.t_sym);
  const int j = l + static_cast<int>(shift);
  if (j < 0 || j >= grid.num_table) {
    return env;
  }
  const double t = u - shift * grid.t_sym - grid.cp;
  const float* row = grid.symbols + 2 * static_cast<int64_t>(j) * grid.num_subcarriers;
  // r = exp(j 2 pi df t); the n-th subcarrier phasor is r^n, re-anchored.
  const double base = grid.df * t;
  float r_s, r_c;
  {
    const double frac = base - floor(base);
    sincosf(static_cast<float>(kTwoPiD * frac), &r_s, &r_c);
  }
  float z_re = 1.0f, z_im = 0.0f;
  float acc_re = 0.0f, acc_im = 0.0f, dacc_re = 0.0f, dacc_im = 0.0f;
  for (int n = 0; n < grid.num_subcarriers; ++n) {
    if (n % kAnchor == 0 && n > 0) {
      const double cycles = static_cast<double>(n) * base;
      const double frac = cycles - floor(cycles);
      sincosf(static_cast<float>(kTwoPiD * frac), &z_im, &z_re);
    }
    const float a = row[2 * n];
    const float b = row[2 * n + 1];
    const float v_re = a * z_re - b * z_im;
    const float v_im = a * z_im + b * z_re;
    acc_re += v_re;
    acc_im += v_im;
    // (j 2 pi n df) * v
    const float w = static_cast<float>(kTwoPiD * grid.df) * static_cast<float>(n);
    dacc_re -= w * v_im;
    dacc_im += w * v_re;
    const float next_re = z_re * r_c - z_im * r_s;
    z_im = z_re * r_s + z_im * r_c;
    z_re = next_re;
  }
  const float scale = static_cast<float>(grid.inv_sqrt_n);
  env.x_re = acc_re * scale;
  env.x_im = acc_im * scale;
  env.dx_re = dacc_re * scale;
  env.dx_im = dacc_im * scale;
  env.live = true;
  return env;
}

struct SampleTime {
  int symbol;
  double t_loc;
  double t_abs;
};

__device__ __forceinline__ SampleTime sample_time(const Grid& grid, const int symbol, const int segment, const int i) {
  double start = grid.cp;
  int l = symbol;
  if (grid.segment_table != nullptr) {
    start = grid.segment_table[2 * static_cast<int64_t>(segment)];
    l = static_cast<int>(grid.segment_table[2 * static_cast<int64_t>(segment) + 1]);
  }
  const double t_loc = start + static_cast<double>(i) * grid.t_s;
  return {l, t_loc, static_cast<double>(l) * grid.t_sym + t_loc};
}

// The phasor of one row at one instant, and the two delay derivatives of the
// complex sample before the weight multiplies it.
struct RowTerm {
  float e_re, e_im;        // x * e^{j phi}
  float d_rt_re, d_rt_im;  // d(x e^{j phi}) / d tau_rt
  float d_rr_re, d_rr_im;  // d(x e^{j phi}) / d tau_rate
  bool live;
};

__device__ __forceinline__ RowTerm row_term(
    const Grid& grid, const SampleTime& when, const double tau_rt, const double rate,
    const double carrier_hz, const double carrier_rate_hz) {
  RowTerm term{0, 0, 0, 0, 0, 0, false};
  const double drift = rate * when.t_abs;
  const double tau = tau_rt + drift;
  const Envelope env = envelope_at(grid, when.symbol, when.t_loc - tau);
  if (!env.live) {
    return term;
  }
  const double cycles = -(carrier_rate_hz * drift + carrier_hz * tau);
  const double frac = cycles - floor(cycles);
  float s, c;
  sincosf(static_cast<float>(kTwoPiD * frac), &s, &c);
  term.e_re = env.x_re * c - env.x_im * s;
  term.e_im = env.x_re * s + env.x_im * c;
  const float dx_re = env.dx_re * c - env.dx_im * s;
  const float dx_im = env.dx_re * s + env.dx_im * c;
  const float k_rt = static_cast<float>(kTwoPiD * carrier_hz);
  const float k_rr = static_cast<float>(kTwoPiD * (carrier_hz + carrier_rate_hz));
  // -x' e^{j phi} - j k x e^{j phi}
  term.d_rt_re = -dx_re + k_rt * term.e_im;
  term.d_rt_im = -dx_im - k_rt * term.e_re;
  const float t_abs = static_cast<float>(when.t_abs);
  term.d_rr_re = t_abs * (-dx_re + k_rr * term.e_im);
  term.d_rr_im = t_abs * (-dx_im - k_rr * term.e_re);
  term.live = true;
  return term;
}

__global__ void ofdm_time_forward_kernel(
    const float* __restrict__ tau_rt, const float* __restrict__ tau_lo, const float* __restrict__ tau_rate, const float* __restrict__ weight_re,
    const float* __restrict__ weight_im, const int64_t* __restrict__ path_offsets, float* __restrict__ out_re,
    float* __restrict__ out_im, const int num_paths, const int num_segments, const Grid grid, const double carrier_hz,
    const double carrier_rate_hz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int symbol = blockIdx.z;
  if (i >= grid.num_samples || segment >= num_segments) {
    return;
  }
  const SegmentBounds bounds = segment_bounds(path_offsets, segment, num_paths);
  const SampleTime when = sample_time(grid, symbol, segment, i);
  float acc_re = 0.0f, acc_im = 0.0f;
  for (int64_t k = bounds.start; k < bounds.end; ++k) {
    const RowTerm term = row_term(grid, when, path_delay(tau_rt, tau_lo, k), tau_rate[k], carrier_hz, carrier_rate_hz);
    const float w_re = weight_re[k];
    const float w_im = weight_im[k];
    acc_re += w_re * term.e_re - w_im * term.e_im;
    acc_im += w_re * term.e_im + w_im * term.e_re;
  }
  const int64_t out = (static_cast<int64_t>(symbol) * num_segments + segment) * grid.num_samples + i;
  out_re[out] = acc_re;
  out_im[out] = acc_im;
}

__global__ void ofdm_time_jvp_kernel(
    const float* __restrict__ tau_rt, const float* __restrict__ tau_lo, const float* __restrict__ tau_rate, const float* __restrict__ weight_re,
    const float* __restrict__ weight_im, const int64_t* __restrict__ path_offsets,
    const float* __restrict__ tan_tau_rt, const float* __restrict__ tan_tau_rate,
    const float* __restrict__ tan_weight_re, const float* __restrict__ tan_weight_im, float* __restrict__ out_re,
    float* __restrict__ out_im, const int num_paths, const int num_segments, const Grid grid, const double carrier_hz,
    const double carrier_rate_hz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int symbol = blockIdx.z;
  if (i >= grid.num_samples || segment >= num_segments) {
    return;
  }
  const SegmentBounds bounds = segment_bounds(path_offsets, segment, num_paths);
  const SampleTime when = sample_time(grid, symbol, segment, i);
  float acc_re = 0.0f, acc_im = 0.0f;
  for (int64_t k = bounds.start; k < bounds.end; ++k) {
    const RowTerm term = row_term(grid, when, path_delay(tau_rt, tau_lo, k), tau_rate[k], carrier_hz, carrier_rate_hz);
    const float w_re = weight_re[k];
    const float w_im = weight_im[k];
    const float a = tan_tau_rt[k];
    const float b = tan_tau_rate[k];
    const float g_re = a * term.d_rt_re + b * term.d_rr_re;
    const float g_im = a * term.d_rt_im + b * term.d_rr_im;
    const float tw_re = tan_weight_re[k];
    const float tw_im = tan_weight_im[k];
    acc_re += tw_re * term.e_re - tw_im * term.e_im + w_re * g_re - w_im * g_im;
    acc_im += tw_re * term.e_im + tw_im * term.e_re + w_re * g_im + w_im * g_re;
  }
  const int64_t out = (static_cast<int64_t>(symbol) * num_segments + segment) * grid.num_samples + i;
  out_re[out] = acc_re;
  out_im[out] = acc_im;
}

// One block per (path, output symbol). Its threads stride the symbol's samples
// and reduce in shared memory in a fixed order, so the partial sums are
// deterministic; the caller sums the symbol axis.
__global__ void ofdm_time_backward_kernel(
    const float* __restrict__ tau_rt, const float* __restrict__ tau_lo, const float* __restrict__ tau_rate, const float* __restrict__ weight_re,
    const float* __restrict__ weight_im, const int64_t* __restrict__ path_segment,
    const float* __restrict__ grad_re, const float* __restrict__ grad_im, double* __restrict__ partial,
    const int num_paths, const int num_segments, const int num_symbols, const Grid grid, const double carrier_hz,
    const double carrier_rate_hz) {
  __shared__ double shared[4][kBlock];
  const int k = blockIdx.x;
  const int symbol = blockIdx.y;
  int64_t segment = path_segment[k];
  segment = segment < 0 ? 0 : (segment >= num_segments ? num_segments - 1 : segment);
  const float w_re = weight_re[k];
  const float w_im = weight_im[k];
  double d_rt = 0.0, d_rr = 0.0, d_wr = 0.0, d_wi = 0.0;
  for (int i = threadIdx.x; i < grid.num_samples; i += blockDim.x) {
    const SampleTime when = sample_time(grid, symbol, static_cast<int>(segment), i);
    const RowTerm term = row_term(grid, when, path_delay(tau_rt, tau_lo, k), tau_rate[k], carrier_hz, carrier_rate_hz);
    const int64_t index = (static_cast<int64_t>(symbol) * num_segments + segment) * grid.num_samples + i;
    const double g_re = grad_re[index];
    const double g_im = grad_im[index];
    // Real cotangent pairing <g, dy> = g_re dy_re + g_im dy_im.
    d_wr += g_re * term.e_re + g_im * term.e_im;
    d_wi += -g_re * term.e_im + g_im * term.e_re;
    const double rt_re = w_re * term.d_rt_re - w_im * term.d_rt_im;
    const double rt_im = w_re * term.d_rt_im + w_im * term.d_rt_re;
    const double rr_re = w_re * term.d_rr_re - w_im * term.d_rr_im;
    const double rr_im = w_re * term.d_rr_im + w_im * term.d_rr_re;
    d_rt += g_re * rt_re + g_im * rt_im;
    d_rr += g_re * rr_re + g_im * rr_im;
  }
  shared[0][threadIdx.x] = d_rt;
  shared[1][threadIdx.x] = d_rr;
  shared[2][threadIdx.x] = d_wr;
  shared[3][threadIdx.x] = d_wi;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      for (int q = 0; q < 4; ++q) {
        shared[q][threadIdx.x] += shared[q][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < 4) {
    partial[(static_cast<int64_t>(k) * num_symbols + symbol) * 4 + threadIdx.x] = shared[threadIdx.x][0];
  }
}

void check_double(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Double, name, " must have dtype torch.float64.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

Grid make_grid(
    const torch::stable::Tensor& symbols, const torch::stable::Tensor& segment_table, int segments, int64_t subcarriers,
    int64_t samples, double df, double cp) {
  const int n = checked_int(subcarriers, "num_subcarriers");
  STD_TORCH_CHECK(n > 0, "num_subcarriers must be positive.");
  const int m = checked_int(samples, "num_samples");
  STD_TORCH_CHECK(m > 0, "num_samples must be positive.");
  STD_TORCH_CHECK(df > 0.0 && cp >= 0.0, "subcarrier spacing must be positive and the prefix non-negative.");
  check_cuda_float(symbols, "symbols");
  STD_TORCH_CHECK(
      symbols.dim() == 3 && symbols.size(1) == n && symbols.size(2) == 2 && symbols.size(0) > 0,
      "symbols must have shape (num_table_symbols, num_subcarriers, 2).");
  check_double(segment_table, "segment_table");
  Grid grid{};
  grid.num_table = checked_int(symbols.size(0), "num_table_symbols");
  grid.num_subcarriers = n;
  grid.num_samples = m;
  grid.df = df;
  grid.cp = cp;
  grid.t_sym = 1.0 / df + cp;
  grid.t_s = 1.0 / (df * n);
  grid.inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n));
  grid.symbols = symbols.const_data_ptr<float>();
  grid.segment_table = nullptr;
  if (segment_table.numel() != 0) {
    STD_TORCH_CHECK(
        segment_table.dim() == 2 && segment_table.size(0) == segments && segment_table.size(1) == 2,
        "segment_table must have shape (num_segments, 2).");
    grid.segment_table = segment_table.const_data_ptr<double>();
  }
  return grid;
}

void check_out(const torch::stable::Tensor& re, const torch::stable::Tensor& im, int symbols, int segments, int n) {
  check_cuda_float(re, "out_re");
  check_cuda_float(im, "out_im");
  STD_TORCH_CHECK(
      re.dim() == 3 && re.size(0) == symbols && re.size(1) == segments && re.size(2) == n &&
          re.sizes().equals(im.sizes()),
      "the OFDM time-domain output must have shape (symbols, segments, num_samples).");
}

}  // namespace

void ofdm_time_forward_cuda(
    const torch::stable::Tensor& tau_rt, const torch::stable::Tensor& tau_lo, const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re, const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_offsets, const torch::stable::Tensor& symbols,
    const torch::stable::Tensor& segment_table, torch::stable::Tensor& out_re, torch::stable::Tensor& out_im,
    int64_t num_paths, int64_t num_segments, int64_t num_symbols, int64_t num_subcarriers,
    int64_t num_samples, double subcarrier_spacing_hz, double cyclic_prefix_s, double carrier_hz, double carrier_rate_hz) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int out_symbols = checked_int(num_symbols, "num_symbols");
  STD_TORCH_CHECK(segments > 0 && out_symbols > 0, "num_segments and num_symbols must be positive.");
  const Grid grid = make_grid(
      symbols, segment_table, segments, num_subcarriers, num_samples, subcarrier_spacing_hz, cyclic_prefix_s);
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths);
  check_cuda_long(path_offsets, "path_offsets");
  STD_TORCH_CHECK(path_offsets.numel() == static_cast<int64_t>(segments) + 1, "path_offsets must hold num_segments + 1 values.");
  check_out(out_re, out_im, out_symbols, segments, grid.num_samples);
  const torch::stable::accelerator::DeviceGuard device_guard(out_re.get_device_index());
  ofdm_time_forward_kernel<<<
      sample_grid(grid.num_samples, segments, out_symbols, kBlock), dim3(kBlock, 1, 1), 0,
      current_cuda_stream(out_re)>>>(
      tau_rt.const_data_ptr<float>(), delay_residual(tau_lo, tau_rt), tau_rate.const_data_ptr<float>(), weight_re.const_data_ptr<float>(),
      weight_im.const_data_ptr<float>(), path_offsets.const_data_ptr<int64_t>(), out_re.mutable_data_ptr<float>(),
      out_im.mutable_data_ptr<float>(), paths, segments, grid, carrier_hz, carrier_rate_hz);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void ofdm_time_jvp_cuda(
    const torch::stable::Tensor& tau_rt, const torch::stable::Tensor& tau_lo, const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re, const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_offsets, const torch::stable::Tensor& symbols,
    const torch::stable::Tensor& segment_table, const torch::stable::Tensor& tan_tau_rt,
    const torch::stable::Tensor& tan_tau_rate, const torch::stable::Tensor& tan_weight_re,
    const torch::stable::Tensor& tan_weight_im, torch::stable::Tensor& out_re, torch::stable::Tensor& out_im,
    int64_t num_paths, int64_t num_segments, int64_t num_symbols, int64_t num_subcarriers,
    int64_t num_samples, double subcarrier_spacing_hz, double cyclic_prefix_s, double carrier_hz, double carrier_rate_hz) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int out_symbols = checked_int(num_symbols, "num_symbols");
  STD_TORCH_CHECK(segments > 0 && out_symbols > 0, "num_segments and num_symbols must be positive.");
  const Grid grid = make_grid(
      symbols, segment_table, segments, num_subcarriers, num_samples, subcarrier_spacing_hz, cyclic_prefix_s);
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths);
  check_path_inputs(tan_tau_rt, tan_tau_rate, tan_weight_re, tan_weight_im, paths);
  check_cuda_long(path_offsets, "path_offsets");
  STD_TORCH_CHECK(path_offsets.numel() == static_cast<int64_t>(segments) + 1, "path_offsets must hold num_segments + 1 values.");
  check_out(out_re, out_im, out_symbols, segments, grid.num_samples);
  const torch::stable::accelerator::DeviceGuard device_guard(out_re.get_device_index());
  ofdm_time_jvp_kernel<<<
      sample_grid(grid.num_samples, segments, out_symbols, kBlock), dim3(kBlock, 1, 1), 0,
      current_cuda_stream(out_re)>>>(
      tau_rt.const_data_ptr<float>(), delay_residual(tau_lo, tau_rt), tau_rate.const_data_ptr<float>(), weight_re.const_data_ptr<float>(),
      weight_im.const_data_ptr<float>(), path_offsets.const_data_ptr<int64_t>(), tan_tau_rt.const_data_ptr<float>(),
      tan_tau_rate.const_data_ptr<float>(), tan_weight_re.const_data_ptr<float>(),
      tan_weight_im.const_data_ptr<float>(), out_re.mutable_data_ptr<float>(), out_im.mutable_data_ptr<float>(), paths,
      segments, grid, carrier_hz, carrier_rate_hz);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void ofdm_time_backward_cuda(
    const torch::stable::Tensor& tau_rt, const torch::stable::Tensor& tau_lo, const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re, const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_segment, const torch::stable::Tensor& symbols,
    const torch::stable::Tensor& segment_table, const torch::stable::Tensor& grad_re,
    const torch::stable::Tensor& grad_im, torch::stable::Tensor& partial, int64_t num_paths, int64_t num_segments,
    int64_t num_symbols, int64_t num_subcarriers, int64_t num_samples, double subcarrier_spacing_hz, double cyclic_prefix_s,
    double carrier_hz, double carrier_rate_hz) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int out_symbols = checked_int(num_symbols, "num_symbols");
  STD_TORCH_CHECK(segments > 0 && out_symbols > 0, "num_segments and num_symbols must be positive.");
  const Grid grid = make_grid(
      symbols, segment_table, segments, num_subcarriers, num_samples, subcarrier_spacing_hz, cyclic_prefix_s);
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths);
  check_cuda_long(path_segment, "path_segment");
  STD_TORCH_CHECK(path_segment.numel() == paths, "path_segment must hold one segment index per path.");
  check_out(grad_re, grad_im, out_symbols, segments, grid.num_samples);
  check_double(partial, "partial");
  STD_TORCH_CHECK(
      partial.numel() == static_cast<int64_t>(paths) * out_symbols * 4, "partial must hold paths * symbols * 4 values.");
  if (paths == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(partial.get_device_index());
  ofdm_time_backward_kernel<<<dim3(paths, out_symbols, 1), dim3(kBlock, 1, 1), 0, current_cuda_stream(partial)>>>(
      tau_rt.const_data_ptr<float>(), delay_residual(tau_lo, tau_rt), tau_rate.const_data_ptr<float>(), weight_re.const_data_ptr<float>(),
      weight_im.const_data_ptr<float>(), path_segment.const_data_ptr<int64_t>(), grad_re.const_data_ptr<float>(),
      grad_im.const_data_ptr<float>(), partial.mutable_data_ptr<double>(), paths, segments, out_symbols, grid,
      carrier_hz, carrier_rate_hz);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("ofdm_time_forward", TORCH_BOX(&ofdm_time_forward_cuda));
  m.impl("ofdm_time_backward", TORCH_BOX(&ofdm_time_backward_cuda));
  m.impl("ofdm_time_jvp", TORCH_BOX(&ofdm_time_jvp_cuda));
}
