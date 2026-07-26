// OFDM channel frequency response over the (symbol, subcarrier) grid.
//
// This is the Phase-6 OFDM synthesis primitive. Like the beat family it is
// registered in the existing `witwin_radar_dirichlet_cuda` library because the
// packaging chain assumes a single native artifact stem; the physical rename is
// Phase-10 work and is recorded in R-ADR-004.
//
// Closed form, stated verbatim:
//
//   t_l           = l * symbol_period_s
//   drift_k(l)    = tau_rate[k] * t_l
//   tau_k(l)      = tau_rt[k] + drift_k(l)
//   f_sub(n)      = n * subcarrier_spacing_hz
//
//   cycles_k(l,n) = - ( f_sub(n) * tau_k(l)
//                     + carrier_rate_hz * drift_k(l)
//                     + carrier_hz * tau_k(l) )
//
//   H[l][p][n]    = sum_{k in segment p} C[k] * exp(+j * 2 * pi * cycles_k(l,n))
//
// THE ASYMMETRY THAT IS THE MOST LIKELY IMPLEMENTATION ERROR IN THIS FILE:
// the subcarrier term multiplies `tau_k(l)`, the FULL delay, while the
// carrier-rate term multiplies `drift_k(l)`, the delay CHANGE only. They are
// not the same quantity and swapping them produces a plausible cube. The reason
// is provenance: a Channel coefficient carries `exp(-j 2 pi f_ref tau_rt)` and
// nothing else, so
//
//   * the `f_ref` component of the absolute phase is ALREADY in the weight,
//     frozen at the per-frame `tau_rt`, and only its slow-time CHANGE is
//     missing - that is what `carrier_rate_hz` supplies; whereas
//   * the `n * df` component is NOT in the weight at any delay, because the
//     weight was evaluated at one frequency. The whole subcarrier phase,
//     including the part at `tau_rt`, has to be applied here.
//
// The sign is NEGATIVE inside `cycles`, so the published cube is in the CHANNEL
// phasor convention `exp(-j k d)` under `exp(+j 2 pi f t)` time dependence. It
// is NOT conjugated, unlike the FMCW beat cube: OFDM demodulation is
// per-subcarrier equalisation `H = Y / X`, which removes the transmitted symbol
// but not the carrier convention. Publishing in Channel's convention makes
// `H[0][p][0] == C_rt` an exact identity when `n = 0` is pinned to `f_ref`, and
// the Python owner carries the phasor string so no consumer has to guess. There
// is therefore no conjugation anywhere in this family.
//
// The carrier has the same two homes as in the beat family and exactly one of
// them may be nonzero:
//
//   carrier_hz = f_ref, carrier_rate_hz = 0   the kernel owns the whole
//     absolute carrier phase. A kernel-owned carrier multiplies the FULL
//     tau_k(l) and therefore already walks across symbols.
//   carrier_hz = 0, carrier_rate_hz = f_ref   the production path for a
//     Channel-sourced weight, which already carries exp(-j 2 pi f_ref tau_rt)
//     at the frozen per-frame delay. carrier_rate_hz supplies the inter-symbol
//     Doppler term the frozen weight cannot express.
//
// Dropping that rate term leaves only the `n * df` slow-time phase, which
// understates Doppler by `f_ref / (n * df)`. At 77 GHz with a 64 x 120 kHz band
// that is a factor of 1e4 at the top subcarrier and INFINITE at n = 0 - one to
// two orders of magnitude worse than the same bug in FMCW, and just as
// invisible in a magnitude-only range-Doppler map. The Python contract refuses
// a spec that names the carrier in both homes.
//
// WIDEBAND, and the term that moves when the weight becomes per-subcarrier.
//
// `weight_columns` is 1 for a narrowband weight `C[k]` and `num_subcarriers`
// for a wideband one `C[k][n] = H_k(f_ref + n*df)`, laid out row major so the
// element is `weight[k * weight_columns + n]`. Indexing is NOT the whole change
// and getting only the indexing right is the single most dangerous error
// available here, because it produces a plausible cube:
//
//   narrowband: the weight holds exp(-j 2 pi f_ref tau_rt) and NOTHING at
//     n * df, so the kernel owns the whole subcarrier phase and the subcarrier
//     term multiplies the FULL delay tau_k(l).
//   wideband:   column n holds exp(-j 2 pi (f_ref + n*df) tau_rt) - the whole
//     subcarrier phase, already, at the FROZEN delay. What is missing is only
//     its slow-time CHANGE, so the subcarrier term multiplies the DRIFT
//     drift_k(l) instead.
//
// Applying `f_sub * tau` to a wideband weight counts the n * df tau_rt phase
// twice, which puts every tap at twice its delay in the range profile. That is
// why `sub_delay` below selects between `tau` and `tau_drift` rather than the
// weight index alone changing.
//
// The derivative follows: d(phi)/d(tau_rt) loses its `f_sub` term in wideband
// mode, because the whole tau_rt dependence of the subcarrier phase now lives
// inside the weight, whose gradient this family already produces.
// d(phi)/d(tau_rate) is unchanged in both modes.
//
// What the wideband route does NOT remove is dispersion: a Core DispersionSpec
// is evaluated once at compile, so Channel refuses a dispersive scene with a
// band rather than approximating it. See the Python contract for the quantified
// narrowband error law.
//
// Cyclic prefix: the single-tap-per-subcarrier form above is exact only while
// the whole echo lands inside the CP window. That is a host-side check on
// CONFIGURED values in the Python spec - a measured maximum delay would be a
// per-frame device-to-host transfer - and it fails loud. There is no clamp and
// no reduced-accuracy mode, so this kernel needs no branch for it.
//
// Numerics, identical to `fmcw_beat.cu` and not re-derived: the cycle count is
// accumulated in double and wrapped to [0, 1) before a single `sincosf`, fast
// math stays off in this translation unit, and no window is applied. Any
// spectral window is a downstream DSP choice and is not in the synthesis
// contract.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr double kTwoPiD = 6.283185307179586476925286766559;

struct CfrPhase {
  float sin_phi;
  float cos_phi;
  // d(phi) / d(tau_rt) in radians per second, at fixed (t_l, f_sub).
  double dphi_dtau_rt;
  // d(phi) / d(tau_rate) in radians, at fixed (t_l, f_sub). This is NOT
  // `dphi_dtau_rt * t_l`: `carrier_rate_hz` multiplies the drift, which depends
  // on `tau_rate` but not on `tau_rt`, so the rate derivative carries an extra
  // `carrier_rate_hz * t_l` the base-delay derivative does not.
  double dphi_dtau_rate;
};

// Phase of one path at one (symbol, subcarrier) grid point. The cycle count is
// a large number when the kernel owns the carrier - thousands of cycles for a
// metre-scale target at 77 GHz - so it is formed and wrapped in double before
// it is handed to the single-precision trigonometric unit.
__device__ __forceinline__ CfrPhase cfr_phase(
    const double tau,
    const double tau_drift,
    const double t_l,
    const double f_sub,
    const double carrier_hz,
    const double carrier_rate_hz,
    const bool wideband) {
  // The whole wideband difference, in two lines. A wideband weight already
  // carries the subcarrier phase at the frozen delay, so only its drift is
  // missing; a narrowband weight carries none of it, so the full delay is.
  const double sub_delay = wideband ? tau_drift : tau;
  const double sub_dtau_rt = wideband ? 0.0 : f_sub;
  const double cycles =
      -(f_sub * sub_delay + carrier_rate_hz * tau_drift + carrier_hz * tau);
  const double frac = cycles - floor(cycles);
  float sin_phi;
  float cos_phi;
  sincosf(static_cast<float>(kTwoPiD * frac), &sin_phi, &cos_phi);
  const double dphi_dtau_rt = -kTwoPiD * (sub_dtau_rt + carrier_hz);
  const double dphi_dtau_rate =
      -kTwoPiD * t_l * (f_sub + carrier_hz + carrier_rate_hz);
  return {sin_phi, cos_phi, dphi_dtau_rt, dphi_dtau_rate};
}

// Where subcarrier `n` reads path `k`'s weight. `weight_columns == 1` collapses
// to the narrowband `weight[k]` exactly, which is what keeps a narrowband cube
// bit-identical to the pre-band one.
__device__ __forceinline__ int64_t weight_index(
    const int64_t k, const int subcarrier, const int weight_columns) {
  return weight_columns > 1
      ? k * static_cast<int64_t>(weight_columns) + subcarrier
      : k;
}

__global__ void ofdm_cfr_forward_kernel(
    const float* __restrict__ tau_rt,
    const float* __restrict__ tau_rate,
    const float* __restrict__ weight_re,
    const float* __restrict__ weight_im,
    const int64_t* __restrict__ path_offsets,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    const int num_paths,
    const int num_segments,
    const int num_subcarriers,
    const int weight_columns,
    const double subcarrier_spacing_hz,
    const double symbol_period_s,
    const double carrier_hz,
    const double carrier_rate_hz) {
  const int subcarrier = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int symbol = blockIdx.z;
  if (subcarrier >= num_subcarriers || segment >= num_segments) {
    return;
  }

  // A memory-safety backstop, not a validation policy: the host wrapper checks
  // the table's SHAPE but never reads its values, because doing so per frame
  // would be the D2H the fixed-topology capability exists to avoid.
  int64_t start = path_offsets[segment];
  int64_t end = path_offsets[segment + 1];
  start = start < 0 ? 0 : start;
  end = end > num_paths ? num_paths : end;

  const bool wideband = weight_columns > 1;
  const double t_l = static_cast<double>(symbol) * symbol_period_s;
  const double f_sub = static_cast<double>(subcarrier) * subcarrier_spacing_hz;

  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = start; k < end; ++k) {
    const double drift = static_cast<double>(tau_rate[k]) * t_l;
    const double tau = static_cast<double>(tau_rt[k]) + drift;
    const CfrPhase phase = cfr_phase(
        tau, drift, t_l, f_sub, carrier_hz, carrier_rate_hz, wideband);
    const int64_t w = weight_index(k, subcarrier, weight_columns);
    const float w_re = weight_re[w];
    const float w_im = weight_im[w];
    acc_re += w_re * phase.cos_phi - w_im * phase.sin_phi;
    acc_im += w_re * phase.sin_phi + w_im * phase.cos_phi;
  }

  const int64_t out_idx =
      (static_cast<int64_t>(symbol) * num_segments + segment) * num_subcarriers +
      subcarrier;
  out_re[out_idx] = acc_re;
  out_im[out_idx] = acc_im;
}

__global__ void ofdm_cfr_jvp_kernel(
    const float* __restrict__ tau_rt,
    const float* __restrict__ tau_rate,
    const float* __restrict__ weight_re,
    const float* __restrict__ weight_im,
    const int64_t* __restrict__ path_offsets,
    const float* __restrict__ tan_tau_rt,
    const float* __restrict__ tan_tau_rate,
    const float* __restrict__ tan_weight_re,
    const float* __restrict__ tan_weight_im,
    float* __restrict__ tan_out_re,
    float* __restrict__ tan_out_im,
    const int num_paths,
    const int num_segments,
    const int num_subcarriers,
    const int weight_columns,
    const double subcarrier_spacing_hz,
    const double symbol_period_s,
    const double carrier_hz,
    const double carrier_rate_hz) {
  const int subcarrier = blockIdx.x * blockDim.x + threadIdx.x;
  const int segment = blockIdx.y;
  const int symbol = blockIdx.z;
  if (subcarrier >= num_subcarriers || segment >= num_segments) {
    return;
  }

  int64_t start = path_offsets[segment];
  int64_t end = path_offsets[segment + 1];
  start = start < 0 ? 0 : start;
  end = end > num_paths ? num_paths : end;

  const bool wideband = weight_columns > 1;
  const double t_l = static_cast<double>(symbol) * symbol_period_s;
  const double f_sub = static_cast<double>(subcarrier) * subcarrier_spacing_hz;

  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int64_t k = start; k < end; ++k) {
    const double drift = static_cast<double>(tau_rate[k]) * t_l;
    const double tau = static_cast<double>(tau_rt[k]) + drift;
    const CfrPhase phase = cfr_phase(
        tau, drift, t_l, f_sub, carrier_hz, carrier_rate_hz, wideband);
    const int64_t w = weight_index(k, subcarrier, weight_columns);
    const float w_re = weight_re[w];
    const float w_im = weight_im[w];
    const float re = w_re * phase.cos_phi - w_im * phase.sin_phi;
    const float im = w_re * phase.sin_phi + w_im * phase.cos_phi;

    const double dphi_d =
        phase.dphi_dtau_rt * static_cast<double>(tan_tau_rt[k]) +
        phase.dphi_dtau_rate * static_cast<double>(tan_tau_rate[k]);
    const float dphi = static_cast<float>(dphi_d);
    const float tw_re = tan_weight_re[w];
    const float tw_im = tan_weight_im[w];
    acc_re += tw_re * phase.cos_phi - tw_im * phase.sin_phi - dphi * im;
    acc_im += tw_re * phase.sin_phi + tw_im * phase.cos_phi + dphi * re;
  }

  const int64_t out_idx =
      (static_cast<int64_t>(symbol) * num_segments + segment) * num_subcarriers +
      subcarrier;
  tan_out_re[out_idx] = acc_re;
  tan_out_im[out_idx] = acc_im;
}

// One thread per path, looping the whole (symbol, subcarrier) grid. Each path
// owns exactly one output slot per weight COLUMN in each gradient array, so the
// reduction needs no atomics and the summation order is fixed by the loop nest.
//
// Two loop nests, not one. A narrowband weight has a single gradient slot per
// path and accumulates over the whole grid in symbol-major order; a wideband
// weight has one slot per subcarrier and must accumulate over symbols only, for
// a fixed subcarrier. Reusing the symbol-major nest for both would mean either
// an unbounded per-thread accumulator array or a changed narrowband summation
// order. The narrowband nest below is therefore preserved VERBATIM, which is
// what makes a narrowband gradient bit-identical to the pre-band one, and the
// wideband nest is subcarrier major. `d_tau_rt` and `d_tau_rate` reduce over the
// whole grid in both, in whatever order their own nest visits it.
__global__ void ofdm_cfr_backward_kernel(
    const float* __restrict__ tau_rt,
    const float* __restrict__ tau_rate,
    const float* __restrict__ weight_re,
    const float* __restrict__ weight_im,
    const int64_t* __restrict__ path_segment,
    const float* __restrict__ grad_out_re,
    const float* __restrict__ grad_out_im,
    float* __restrict__ grad_tau_rt,
    float* __restrict__ grad_tau_rate,
    float* __restrict__ grad_weight_re,
    float* __restrict__ grad_weight_im,
    const int num_paths,
    const int num_segments,
    const int num_symbols,
    const int num_subcarriers,
    const int weight_columns,
    const double subcarrier_spacing_hz,
    const double symbol_period_s,
    const double carrier_hz,
    const double carrier_rate_hz) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_paths) {
    return;
  }

  // Same backstop. On the production route path_segment is DERIVED from the
  // same offsets table, so it cannot disagree with it.
  int64_t segment = path_segment[k];
  segment = segment < 0 ? 0 : segment;
  segment = segment >= num_segments ? num_segments - 1 : segment;

  const bool wideband = weight_columns > 1;
  const double base_tau = static_cast<double>(tau_rt[k]);
  const double rate = static_cast<double>(tau_rate[k]);

  double d_tau_rt = 0.0;
  double d_tau_rate = 0.0;

  if (!wideband) {
    const float w_re = weight_re[k];
    const float w_im = weight_im[k];
    double d_w_re = 0.0;
    double d_w_im = 0.0;

    for (int symbol = 0; symbol < num_symbols; ++symbol) {
      const double t_l = static_cast<double>(symbol) * symbol_period_s;
      const double drift = rate * t_l;
      const double tau = base_tau + drift;
      const int64_t row_base =
          (static_cast<int64_t>(symbol) * num_segments + segment) *
          num_subcarriers;
      for (int subcarrier = 0; subcarrier < num_subcarriers; ++subcarrier) {
        const double f_sub =
            static_cast<double>(subcarrier) * subcarrier_spacing_hz;
        const CfrPhase phase = cfr_phase(
            tau, drift, t_l, f_sub, carrier_hz, carrier_rate_hz, false);
        const float g_re = grad_out_re[row_base + subcarrier];
        const float g_im = grad_out_im[row_base + subcarrier];
        const float re = w_re * phase.cos_phi - w_im * phase.sin_phi;
        const float im = w_re * phase.sin_phi + w_im * phase.cos_phi;

        d_w_re += static_cast<double>(g_re) * phase.cos_phi +
            static_cast<double>(g_im) * phase.sin_phi;
        d_w_im += -static_cast<double>(g_re) * phase.sin_phi +
            static_cast<double>(g_im) * phase.cos_phi;

        const double d_phi =
            -static_cast<double>(g_re) * im + static_cast<double>(g_im) * re;
        d_tau_rt += d_phi * phase.dphi_dtau_rt;
        d_tau_rate += d_phi * phase.dphi_dtau_rate;
      }
    }
    grad_weight_re[k] = static_cast<float>(d_w_re);
    grad_weight_im[k] = static_cast<float>(d_w_im);
  } else {
    for (int subcarrier = 0; subcarrier < num_subcarriers; ++subcarrier) {
      const double f_sub =
          static_cast<double>(subcarrier) * subcarrier_spacing_hz;
      const int64_t w =
          static_cast<int64_t>(k) * weight_columns + subcarrier;
      const float w_re = weight_re[w];
      const float w_im = weight_im[w];
      double d_w_re = 0.0;
      double d_w_im = 0.0;

      for (int symbol = 0; symbol < num_symbols; ++symbol) {
        const double t_l = static_cast<double>(symbol) * symbol_period_s;
        const double drift = rate * t_l;
        const double tau = base_tau + drift;
        const int64_t out_idx =
            (static_cast<int64_t>(symbol) * num_segments + segment) *
                num_subcarriers +
            subcarrier;
        const CfrPhase phase = cfr_phase(
            tau, drift, t_l, f_sub, carrier_hz, carrier_rate_hz, true);
        const float g_re = grad_out_re[out_idx];
        const float g_im = grad_out_im[out_idx];
        const float re = w_re * phase.cos_phi - w_im * phase.sin_phi;
        const float im = w_re * phase.sin_phi + w_im * phase.cos_phi;

        d_w_re += static_cast<double>(g_re) * phase.cos_phi +
            static_cast<double>(g_im) * phase.sin_phi;
        d_w_im += -static_cast<double>(g_re) * phase.sin_phi +
            static_cast<double>(g_im) * phase.cos_phi;

        const double d_phi =
            -static_cast<double>(g_re) * im + static_cast<double>(g_im) * re;
        d_tau_rt += d_phi * phase.dphi_dtau_rt;
        d_tau_rate += d_phi * phase.dphi_dtau_rate;
      }
      grad_weight_re[w] = static_cast<float>(d_w_re);
      grad_weight_im[w] = static_cast<float>(d_w_im);
    }
  }

  grad_tau_rt[k] = static_cast<float>(d_tau_rt);
  grad_tau_rate[k] = static_cast<float>(d_tau_rate);
}

void check_cuda_float(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Float,
      name,
      " must have dtype torch.float32.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void check_cuda_long(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Long,
      name,
      " must have dtype torch.int64.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

int checked_int(int64_t value, const char* name) {
  STD_TORCH_CHECK(
      value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<int>::max()),
      name,
      " is out of int32 range.");
  return static_cast<int>(value);
}

cudaStream_t current_cuda_stream(const torch::stable::Tensor& tensor) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

void check_path_inputs(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    int num_paths,
    int weight_columns) {
  check_cuda_float(tau_rt, "tau_rt");
  check_cuda_float(tau_rate, "tau_rate");
  check_cuda_float(weight_re, "weight_re");
  check_cuda_float(weight_im, "weight_im");
  STD_TORCH_CHECK(
      tau_rt.numel() == num_paths && tau_rate.numel() == num_paths,
      "tau_rt and tau_rate must each hold num_paths values.");
  const int64_t weights =
      static_cast<int64_t>(num_paths) * static_cast<int64_t>(weight_columns);
  STD_TORCH_CHECK(
      weight_re.numel() == weights && weight_im.numel() == weights,
      "weight_re and weight_im must each hold num_paths * weight_columns values.");
}

// `weight_columns` is 1 for a narrowband weight and num_subcarriers for a
// wideband one. Nothing between the two is meaningful: the kernel pairs column
// n with subcarrier n, so a partial band would either read past the weight or
// leave subcarriers unpaired, and there is no interpolation here to fill a
// coarser grid in.
int checked_weight_columns(int64_t value, int num_subcarriers) {
  const int columns = checked_int(value, "weight_columns");
  STD_TORCH_CHECK(
      columns == 1 || columns == num_subcarriers,
      "weight_columns must be 1 (narrowband) or num_subcarriers (wideband).");
  return columns;
}

void check_output(
    const torch::stable::Tensor& out_re,
    const torch::stable::Tensor& out_im,
    int num_symbols,
    int num_segments,
    int num_subcarriers,
    const char* name_re,
    const char* name_im) {
  check_cuda_float(out_re, name_re);
  check_cuda_float(out_im, name_im);
  STD_TORCH_CHECK(
      out_re.sizes().equals(out_im.sizes()),
      "CFR output components must have the same shape.");
  STD_TORCH_CHECK(
      out_re.dim() == 3,
      "CFR output must have shape (symbols, segments, subcarriers).");
  STD_TORCH_CHECK(
      out_re.size(0) == num_symbols && out_re.size(1) == num_segments &&
          out_re.size(2) == num_subcarriers,
      "CFR output shape disagrees with the declared grid.");
}

dim3 subcarrier_grid(
    int num_subcarriers,
    int num_segments,
    int num_symbols,
    int block) {
  return dim3(
      (num_subcarriers + block - 1) / block, num_segments, num_symbols);
}

}  // namespace

void ofdm_cfr_forward_cuda(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_offsets,
    torch::stable::Tensor& out_re,
    torch::stable::Tensor& out_im,
    int64_t num_paths,
    int64_t num_segments,
    int64_t num_symbols,
    int64_t num_subcarriers,
    int64_t weight_columns,
    double subcarrier_spacing_hz,
    double symbol_period_s,
    double carrier_hz,
    double carrier_rate_hz) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int symbols = checked_int(num_symbols, "num_symbols");
  const int subcarriers = checked_int(num_subcarriers, "num_subcarriers");
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(symbols > 0, "num_symbols must be positive.");
  STD_TORCH_CHECK(subcarriers > 0, "num_subcarriers must be positive.");
  const int columns = checked_weight_columns(weight_columns, subcarriers);
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths, columns);
  check_cuda_long(path_offsets, "path_offsets");
  STD_TORCH_CHECK(
      path_offsets.numel() == static_cast<int64_t>(segments) + 1,
      "path_offsets must hold num_segments + 1 values.");
  check_output(
      out_re, out_im, symbols, segments, subcarriers, "out_re", "out_im");

  const torch::stable::accelerator::DeviceGuard device_guard(
      out_re.get_device_index());
  constexpr int block_size = 256;
  ofdm_cfr_forward_kernel<<<
      subcarrier_grid(subcarriers, segments, symbols, block_size),
      dim3(block_size, 1, 1),
      0,
      current_cuda_stream(out_re)>>>(
      tau_rt.const_data_ptr<float>(),
      tau_rate.const_data_ptr<float>(),
      weight_re.const_data_ptr<float>(),
      weight_im.const_data_ptr<float>(),
      path_offsets.const_data_ptr<int64_t>(),
      out_re.mutable_data_ptr<float>(),
      out_im.mutable_data_ptr<float>(),
      paths,
      segments,
      subcarriers,
      columns,
      subcarrier_spacing_hz,
      symbol_period_s,
      carrier_hz,
      carrier_rate_hz);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void ofdm_cfr_jvp_cuda(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_offsets,
    const torch::stable::Tensor& tan_tau_rt,
    const torch::stable::Tensor& tan_tau_rate,
    const torch::stable::Tensor& tan_weight_re,
    const torch::stable::Tensor& tan_weight_im,
    torch::stable::Tensor& tan_out_re,
    torch::stable::Tensor& tan_out_im,
    int64_t num_paths,
    int64_t num_segments,
    int64_t num_symbols,
    int64_t num_subcarriers,
    int64_t weight_columns,
    double subcarrier_spacing_hz,
    double symbol_period_s,
    double carrier_hz,
    double carrier_rate_hz) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int symbols = checked_int(num_symbols, "num_symbols");
  const int subcarriers = checked_int(num_subcarriers, "num_subcarriers");
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(symbols > 0, "num_symbols must be positive.");
  STD_TORCH_CHECK(subcarriers > 0, "num_subcarriers must be positive.");
  const int columns = checked_weight_columns(weight_columns, subcarriers);
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths, columns);
  check_path_inputs(
      tan_tau_rt, tan_tau_rate, tan_weight_re, tan_weight_im, paths, columns);
  check_cuda_long(path_offsets, "path_offsets");
  STD_TORCH_CHECK(
      path_offsets.numel() == static_cast<int64_t>(segments) + 1,
      "path_offsets must hold num_segments + 1 values.");
  check_output(
      tan_out_re,
      tan_out_im,
      symbols,
      segments,
      subcarriers,
      "tan_out_re",
      "tan_out_im");

  const torch::stable::accelerator::DeviceGuard device_guard(
      tan_out_re.get_device_index());
  constexpr int block_size = 256;
  ofdm_cfr_jvp_kernel<<<
      subcarrier_grid(subcarriers, segments, symbols, block_size),
      dim3(block_size, 1, 1),
      0,
      current_cuda_stream(tan_out_re)>>>(
      tau_rt.const_data_ptr<float>(),
      tau_rate.const_data_ptr<float>(),
      weight_re.const_data_ptr<float>(),
      weight_im.const_data_ptr<float>(),
      path_offsets.const_data_ptr<int64_t>(),
      tan_tau_rt.const_data_ptr<float>(),
      tan_tau_rate.const_data_ptr<float>(),
      tan_weight_re.const_data_ptr<float>(),
      tan_weight_im.const_data_ptr<float>(),
      tan_out_re.mutable_data_ptr<float>(),
      tan_out_im.mutable_data_ptr<float>(),
      paths,
      segments,
      subcarriers,
      columns,
      subcarrier_spacing_hz,
      symbol_period_s,
      carrier_hz,
      carrier_rate_hz);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void ofdm_cfr_backward_cuda(
    const torch::stable::Tensor& tau_rt,
    const torch::stable::Tensor& tau_rate,
    const torch::stable::Tensor& weight_re,
    const torch::stable::Tensor& weight_im,
    const torch::stable::Tensor& path_segment,
    const torch::stable::Tensor& grad_out_re,
    const torch::stable::Tensor& grad_out_im,
    torch::stable::Tensor& grad_tau_rt,
    torch::stable::Tensor& grad_tau_rate,
    torch::stable::Tensor& grad_weight_re,
    torch::stable::Tensor& grad_weight_im,
    int64_t num_paths,
    int64_t num_segments,
    int64_t num_symbols,
    int64_t num_subcarriers,
    int64_t weight_columns,
    double subcarrier_spacing_hz,
    double symbol_period_s,
    double carrier_hz,
    double carrier_rate_hz) {
  const int paths = checked_int(num_paths, "num_paths");
  const int segments = checked_int(num_segments, "num_segments");
  const int symbols = checked_int(num_symbols, "num_symbols");
  const int subcarriers = checked_int(num_subcarriers, "num_subcarriers");
  STD_TORCH_CHECK(segments > 0, "num_segments must be positive.");
  STD_TORCH_CHECK(symbols > 0, "num_symbols must be positive.");
  STD_TORCH_CHECK(subcarriers > 0, "num_subcarriers must be positive.");
  const int columns = checked_weight_columns(weight_columns, subcarriers);
  check_path_inputs(tau_rt, tau_rate, weight_re, weight_im, paths, columns);
  check_cuda_long(path_segment, "path_segment");
  STD_TORCH_CHECK(
      path_segment.numel() == static_cast<int64_t>(paths),
      "path_segment must hold one segment index per path.");
  check_output(
      grad_out_re,
      grad_out_im,
      symbols,
      segments,
      subcarriers,
      "grad_out_re",
      "grad_out_im");
  check_path_inputs(
      grad_tau_rt, grad_tau_rate, grad_weight_re, grad_weight_im, paths, columns);

  if (paths == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      grad_tau_rt.get_device_index());
  constexpr int block_size = 256;
  ofdm_cfr_backward_kernel<<<
      dim3((paths + block_size - 1) / block_size, 1, 1),
      dim3(block_size, 1, 1),
      0,
      current_cuda_stream(grad_tau_rt)>>>(
      tau_rt.const_data_ptr<float>(),
      tau_rate.const_data_ptr<float>(),
      weight_re.const_data_ptr<float>(),
      weight_im.const_data_ptr<float>(),
      path_segment.const_data_ptr<int64_t>(),
      grad_out_re.const_data_ptr<float>(),
      grad_out_im.const_data_ptr<float>(),
      grad_tau_rt.mutable_data_ptr<float>(),
      grad_tau_rate.mutable_data_ptr<float>(),
      grad_weight_re.mutable_data_ptr<float>(),
      grad_weight_im.mutable_data_ptr<float>(),
      paths,
      segments,
      symbols,
      subcarriers,
      columns,
      subcarrier_spacing_hz,
      symbol_period_s,
      carrier_hz,
      carrier_rate_hz);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(witwin_radar_dirichlet_cuda, CUDA, m) {
  m.impl("ofdm_cfr_forward", TORCH_BOX(&ofdm_cfr_forward_cuda));
  m.impl("ofdm_cfr_backward", TORCH_BOX(&ofdm_cfr_backward_cuda));
  m.impl("ofdm_cfr_jvp", TORCH_BOX(&ofdm_cfr_jvp_cuda));
}
