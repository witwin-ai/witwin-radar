// The two-way join: inbound leg x outbound leg -> radar round trip.
//
// Per composed row k, with i = idx_in[k], o = idx_out[k], s = idx_s[k]:
//
//   tau_rt[k]  = tau_in[i]  + tau_out[o]
//   rate_rt[k] = rate_in[i] + rate_out[o]
//   C_rt[k]    = (C_out[o] * S[s]) * C_in[i]
//
// and every payload is exactly zero when the row is dead.
//
// Why a kernel at all, from measurement rather than assumption: the Torch
// composition it replaces issued roughly 17-19 device-side aten ops (8
// index_select, 4 mul, 2 add, 1 exp, 1 bitwise_and, 2 _to_copy, 1 fill_) and
// measured a flat 0.2-0.6 ms from K = 4 to K = 24000. It was launch bound, not
// bandwidth bound, so one fused launch is the entire win.
//
// Five rules this file encodes, each pinned by a test:
//
//  1. The association is (C_out * S) * C_in, copied verbatim from the Torch
//     composer it replaces. Re-associating a complex product is a numerical
//     change and would need its own ADR, not a rewrite.
//  2. A dead row's payload is exactly zero. The row is a complete answer that
//     this round trip does not exist at these endpoint positions. Publishing
//     tau_in + 0 for it, as the Torch composer used to, is a plausible number
//     no consumer should read.
//  3. rate_rt carries a ZERO tangent and the rate inputs take no gradient.
//     delay_rate arrives already unpacked from a forward-only dual and is
//     published as a primal, which deliberately severs d(delay_rate)/dx. The
//     Python facade REFUSES a rate input that carries requires_grad or a
//     tangent, so "returns None" can never be confused with "dropped it".
//  4. The VJP uses the frozen CSR tables: one thread owns one gradient slot
//     and loops its own segment, so there are no atomics and the summation
//     order is a property of the frozen join. That is what makes a
//     bit-identical gradient comparison across a permuted leg order a
//     legitimate assertion rather than a lucky one.
//  5. Sums accumulate in double and store float32, matching fmcw_beat.cu. For
//     the delay this is free insurance; the two mixed combined paths in the
//     Phase-5 fixture are 20 ps apart, about 1e4 float32 ULPs, so float32
//     alone would also do, and nothing here depends on the difference.
//
// Validity itself is computed in Torch, not here, and enters as an int32 mask.
// It is the conjunction of two gathered per-leg masks: row selection and
// metadata, which the architecture explicitly leaves to the Python boundary.
// Doing it in the kernel would need either a bool ABI that Torch's stable
// data_ptr does not instantiate or two extra dtype conversions, and would add
// a non-differentiable output whose JVP contract is pure noise.

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

struct Complex {
  double re;
  double im;
};

__device__ __forceinline__ Complex mul(const Complex a, const Complex b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

__device__ __forceinline__ Complex load(
    const float* __restrict__ re,
    const float* __restrict__ im,
    const int64_t index) {
  return {static_cast<double>(re[index]), static_cast<double>(im[index])};
}

__global__ void two_way_join_forward_kernel(
    const float* __restrict__ tau_in,
    const float* __restrict__ tau_out,
    const float* __restrict__ rate_in,
    const float* __restrict__ rate_out,
    const float* __restrict__ c_in_re,
    const float* __restrict__ c_in_im,
    const float* __restrict__ c_out_re,
    const float* __restrict__ c_out_im,
    const float* __restrict__ s_re,
    const float* __restrict__ s_im,
    const int32_t* __restrict__ row_valid,
    const int64_t* __restrict__ idx_in,
    const int64_t* __restrict__ idx_out,
    const int64_t* __restrict__ idx_s,
    float* __restrict__ tau_rt,
    float* __restrict__ rate_rt,
    float* __restrict__ c_rt_re,
    float* __restrict__ c_rt_im,
    const int num_rows) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_rows) {
    return;
  }
  if (row_valid[k] == 0) {
    tau_rt[k] = 0.0f;
    rate_rt[k] = 0.0f;
    c_rt_re[k] = 0.0f;
    c_rt_im[k] = 0.0f;
    return;
  }
  const int64_t i = idx_in[k];
  const int64_t o = idx_out[k];
  const int64_t s = idx_s[k];
  tau_rt[k] = static_cast<float>(
      static_cast<double>(tau_in[i]) + static_cast<double>(tau_out[o]));
  rate_rt[k] = static_cast<float>(
      static_cast<double>(rate_in[i]) + static_cast<double>(rate_out[o]));
  const Complex product = mul(
      mul(load(c_out_re, c_out_im, o), load(s_re, s_im, s)),
      load(c_in_re, c_in_im, i));
  c_rt_re[k] = static_cast<float>(product.re);
  c_rt_im[k] = static_cast<float>(product.im);
}

__global__ void two_way_join_jvp_kernel(
    const float* __restrict__ c_in_re,
    const float* __restrict__ c_in_im,
    const float* __restrict__ c_out_re,
    const float* __restrict__ c_out_im,
    const float* __restrict__ s_re,
    const float* __restrict__ s_im,
    const int32_t* __restrict__ row_valid,
    const int64_t* __restrict__ idx_in,
    const int64_t* __restrict__ idx_out,
    const int64_t* __restrict__ idx_s,
    const float* __restrict__ tan_tau_in,
    const float* __restrict__ tan_tau_out,
    const float* __restrict__ tan_c_in_re,
    const float* __restrict__ tan_c_in_im,
    const float* __restrict__ tan_c_out_re,
    const float* __restrict__ tan_c_out_im,
    const float* __restrict__ tan_s_re,
    const float* __restrict__ tan_s_im,
    float* __restrict__ tan_tau_rt,
    float* __restrict__ tan_rate_rt,
    float* __restrict__ tan_c_rt_re,
    float* __restrict__ tan_c_rt_im,
    const int num_rows) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_rows) {
    return;
  }
  // rate_rt = rate_in + rate_out and both are primal by contract, so its
  // tangent is structurally zero rather than merely unpopulated.
  tan_rate_rt[k] = 0.0f;
  if (row_valid[k] == 0) {
    tan_tau_rt[k] = 0.0f;
    tan_c_rt_re[k] = 0.0f;
    tan_c_rt_im[k] = 0.0f;
    return;
  }
  const int64_t i = idx_in[k];
  const int64_t o = idx_out[k];
  const int64_t s = idx_s[k];
  tan_tau_rt[k] = static_cast<float>(
      static_cast<double>(tan_tau_in[i]) + static_cast<double>(tan_tau_out[o]));

  const Complex c_in = load(c_in_re, c_in_im, i);
  const Complex c_out = load(c_out_re, c_out_im, o);
  const Complex response = load(s_re, s_im, s);
  const Complex d_in = load(tan_c_in_re, tan_c_in_im, i);
  const Complex d_out = load(tan_c_out_re, tan_c_out_im, o);
  const Complex d_response = load(tan_s_re, tan_s_im, s);

  // Product rule, each term keeping the primal association.
  const Complex first = mul(mul(d_out, response), c_in);
  const Complex second = mul(mul(c_out, d_response), c_in);
  const Complex third = mul(mul(c_out, response), d_in);
  tan_c_rt_re[k] = static_cast<float>(first.re + second.re + third.re);
  tan_c_rt_im[k] = static_cast<float>(first.im + second.im + third.im);
}

// One thread owns one gradient slot. The three owner families - inbound rows,
// outbound rows, sites - are laid end to end on a single grid so the whole VJP
// is one launch, and each thread walks only its own CSR segment.
__global__ void two_way_join_backward_kernel(
    const float* __restrict__ c_in_re,
    const float* __restrict__ c_in_im,
    const float* __restrict__ c_out_re,
    const float* __restrict__ c_out_im,
    const float* __restrict__ s_re,
    const float* __restrict__ s_im,
    const int32_t* __restrict__ row_valid,
    const int64_t* __restrict__ idx_in,
    const int64_t* __restrict__ idx_out,
    const int64_t* __restrict__ idx_s,
    const int64_t* __restrict__ by_in_offsets,
    const int64_t* __restrict__ by_in_rows,
    const int64_t* __restrict__ by_out_offsets,
    const int64_t* __restrict__ by_out_rows,
    const int64_t* __restrict__ by_s_offsets,
    const int64_t* __restrict__ by_s_rows,
    const float* __restrict__ grad_tau_rt,
    const float* __restrict__ grad_c_rt_re,
    const float* __restrict__ grad_c_rt_im,
    float* __restrict__ grad_tau_in,
    float* __restrict__ grad_tau_out,
    float* __restrict__ grad_c_in_re,
    float* __restrict__ grad_c_in_im,
    float* __restrict__ grad_c_out_re,
    float* __restrict__ grad_c_out_im,
    float* __restrict__ grad_s_re,
    float* __restrict__ grad_s_im,
    const int num_in,
    const int num_out,
    const int num_sites) {
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_in + num_out + num_sites;
  if (thread >= total) {
    return;
  }

  int family;
  int slot;
  const int64_t* offsets;
  const int64_t* segment_rows;
  if (thread < num_in) {
    family = 0;
    slot = thread;
    offsets = by_in_offsets;
    segment_rows = by_in_rows;
  } else if (thread < num_in + num_out) {
    family = 1;
    slot = thread - num_in;
    offsets = by_out_offsets;
    segment_rows = by_out_rows;
  } else {
    family = 2;
    slot = thread - num_in - num_out;
    offsets = by_s_offsets;
    segment_rows = by_s_rows;
  }

  double d_tau = 0.0;
  double d_re = 0.0;
  double d_im = 0.0;
  const int64_t begin = offsets[slot];
  const int64_t end = offsets[slot + 1];
  for (int64_t entry = begin; entry < end; ++entry) {
    const int64_t k = segment_rows[entry];
    if (row_valid[k] == 0) {
      continue;
    }
    const double g_re = static_cast<double>(grad_c_rt_re[k]);
    const double g_im = static_cast<double>(grad_c_rt_im[k]);
    // The cofactor: the product of everything this slot did NOT contribute.
    Complex cofactor;
    if (family == 0) {
      cofactor = mul(
          load(c_out_re, c_out_im, idx_out[k]), load(s_re, s_im, idx_s[k]));
      d_tau += static_cast<double>(grad_tau_rt[k]);
    } else if (family == 1) {
      cofactor = mul(
          load(s_re, s_im, idx_s[k]), load(c_in_re, c_in_im, idx_in[k]));
      d_tau += static_cast<double>(grad_tau_rt[k]);
    } else {
      cofactor = mul(
          load(c_out_re, c_out_im, idx_out[k]),
          load(c_in_re, c_in_im, idx_in[k]));
    }
    // Real-pair backpropagation through z = w * cofactor, the same split the
    // beat kernel uses: no complex tensor ever crosses the autograd boundary,
    // so the conjugate-Wirtinger convention cannot be got wrong here.
    d_re += g_re * cofactor.re + g_im * cofactor.im;
    d_im += -g_re * cofactor.im + g_im * cofactor.re;
  }

  if (family == 0) {
    grad_tau_in[slot] = static_cast<float>(d_tau);
    grad_c_in_re[slot] = static_cast<float>(d_re);
    grad_c_in_im[slot] = static_cast<float>(d_im);
  } else if (family == 1) {
    grad_tau_out[slot] = static_cast<float>(d_tau);
    grad_c_out_re[slot] = static_cast<float>(d_re);
    grad_c_out_im[slot] = static_cast<float>(d_im);
  } else {
    grad_s_re[slot] = static_cast<float>(d_re);
    grad_s_im[slot] = static_cast<float>(d_im);
  }
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

void check_cuda_int(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::Int,
      name,
      " must have dtype torch.int32.");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

int checked_int(int64_t value, const char* name) {
  STD_TORCH_CHECK(
      value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<int>::max()),
      name,
      " is out of int32 range.");
  return static_cast<int>(value);
}

cudaStream_t join_stream(const torch::stable::Tensor& tensor) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

void check_float_len(
    const torch::stable::Tensor& tensor, int64_t expected, const char* name) {
  check_cuda_float(tensor, name);
  STD_TORCH_CHECK(
      tensor.numel() == expected,
      name,
      " must hold one value per owning row.");
}

void check_pair(
    const torch::stable::Tensor& re,
    const torch::stable::Tensor& im,
    int64_t expected,
    const char* name) {
  check_float_len(re, expected, name);
  check_float_len(im, expected, name);
}

void check_index(
    const torch::stable::Tensor& index, int rows, const char* name) {
  check_cuda_long(index, name);
  STD_TORCH_CHECK(
      index.numel() == static_cast<int64_t>(rows),
      name,
      " must hold one index per composed row.");
}

constexpr int kBlock = 256;

dim3 linear_grid(int count) {
  return dim3(static_cast<unsigned>((count + kBlock - 1) / kBlock), 1, 1);
}

}  // namespace

void two_way_join_forward_cuda(
    const torch::stable::Tensor& tau_in,
    const torch::stable::Tensor& tau_out,
    const torch::stable::Tensor& rate_in,
    const torch::stable::Tensor& rate_out,
    const torch::stable::Tensor& c_in_re,
    const torch::stable::Tensor& c_in_im,
    const torch::stable::Tensor& c_out_re,
    const torch::stable::Tensor& c_out_im,
    const torch::stable::Tensor& s_re,
    const torch::stable::Tensor& s_im,
    const torch::stable::Tensor& row_valid,
    const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out,
    const torch::stable::Tensor& idx_s,
    torch::stable::Tensor& tau_rt,
    torch::stable::Tensor& rate_rt,
    torch::stable::Tensor& c_rt_re,
    torch::stable::Tensor& c_rt_im,
    int64_t num_rows) {
  const int rows = checked_int(num_rows, "num_rows");
  check_cuda_float(tau_in, "tau_in");
  check_cuda_float(tau_out, "tau_out");
  check_cuda_float(rate_in, "rate_in");
  check_cuda_float(rate_out, "rate_out");
  STD_TORCH_CHECK(
      rate_in.numel() == tau_in.numel(),
      "rate_in must hold one value per inbound row.");
  STD_TORCH_CHECK(
      rate_out.numel() == tau_out.numel(),
      "rate_out must hold one value per outbound row.");
  check_pair(c_in_re, c_in_im, tau_in.numel(), "c_in");
  check_pair(c_out_re, c_out_im, tau_out.numel(), "c_out");
  check_pair(s_re, s_im, s_re.numel(), "s");
  check_cuda_int(row_valid, "row_valid");
  STD_TORCH_CHECK(
      row_valid.numel() == static_cast<int64_t>(rows),
      "row_valid must hold one flag per composed row.");
  check_index(idx_in, rows, "idx_in");
  check_index(idx_out, rows, "idx_out");
  check_index(idx_s, rows, "idx_s");
  check_pair(tau_rt, rate_rt, rows, "tau_rt/rate_rt");
  check_pair(c_rt_re, c_rt_im, rows, "c_rt");

  if (rows == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      tau_rt.get_device_index());
  two_way_join_forward_kernel<<<
      linear_grid(rows), dim3(kBlock, 1, 1), 0, join_stream(tau_rt)>>>(
      tau_in.const_data_ptr<float>(),
      tau_out.const_data_ptr<float>(),
      rate_in.const_data_ptr<float>(),
      rate_out.const_data_ptr<float>(),
      c_in_re.const_data_ptr<float>(),
      c_in_im.const_data_ptr<float>(),
      c_out_re.const_data_ptr<float>(),
      c_out_im.const_data_ptr<float>(),
      s_re.const_data_ptr<float>(),
      s_im.const_data_ptr<float>(),
      row_valid.const_data_ptr<int32_t>(),
      idx_in.const_data_ptr<int64_t>(),
      idx_out.const_data_ptr<int64_t>(),
      idx_s.const_data_ptr<int64_t>(),
      tau_rt.mutable_data_ptr<float>(),
      rate_rt.mutable_data_ptr<float>(),
      c_rt_re.mutable_data_ptr<float>(),
      c_rt_im.mutable_data_ptr<float>(),
      rows);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void two_way_join_jvp_cuda(
    const torch::stable::Tensor& c_in_re,
    const torch::stable::Tensor& c_in_im,
    const torch::stable::Tensor& c_out_re,
    const torch::stable::Tensor& c_out_im,
    const torch::stable::Tensor& s_re,
    const torch::stable::Tensor& s_im,
    const torch::stable::Tensor& row_valid,
    const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out,
    const torch::stable::Tensor& idx_s,
    const torch::stable::Tensor& tan_tau_in,
    const torch::stable::Tensor& tan_tau_out,
    const torch::stable::Tensor& tan_c_in_re,
    const torch::stable::Tensor& tan_c_in_im,
    const torch::stable::Tensor& tan_c_out_re,
    const torch::stable::Tensor& tan_c_out_im,
    const torch::stable::Tensor& tan_s_re,
    const torch::stable::Tensor& tan_s_im,
    torch::stable::Tensor& tan_tau_rt,
    torch::stable::Tensor& tan_rate_rt,
    torch::stable::Tensor& tan_c_rt_re,
    torch::stable::Tensor& tan_c_rt_im,
    int64_t num_rows) {
  const int rows = checked_int(num_rows, "num_rows");
  check_pair(c_in_re, c_in_im, c_in_re.numel(), "c_in");
  check_pair(c_out_re, c_out_im, c_out_re.numel(), "c_out");
  check_pair(s_re, s_im, s_re.numel(), "s");
  check_pair(tan_c_in_re, tan_c_in_im, c_in_re.numel(), "tan_c_in");
  check_pair(tan_c_out_re, tan_c_out_im, c_out_re.numel(), "tan_c_out");
  check_pair(tan_s_re, tan_s_im, s_re.numel(), "tan_s");
  check_cuda_float(tan_tau_in, "tan_tau_in");
  check_cuda_float(tan_tau_out, "tan_tau_out");
  STD_TORCH_CHECK(
      tan_tau_in.numel() == c_in_re.numel(),
      "tan_tau_in must hold one value per inbound row.");
  STD_TORCH_CHECK(
      tan_tau_out.numel() == c_out_re.numel(),
      "tan_tau_out must hold one value per outbound row.");
  check_cuda_int(row_valid, "row_valid");
  STD_TORCH_CHECK(
      row_valid.numel() == static_cast<int64_t>(rows),
      "row_valid must hold one flag per composed row.");
  check_index(idx_in, rows, "idx_in");
  check_index(idx_out, rows, "idx_out");
  check_index(idx_s, rows, "idx_s");
  check_pair(tan_tau_rt, tan_rate_rt, rows, "tan_tau_rt/tan_rate_rt");
  check_pair(tan_c_rt_re, tan_c_rt_im, rows, "tan_c_rt");

  if (rows == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      tan_tau_rt.get_device_index());
  two_way_join_jvp_kernel<<<
      linear_grid(rows), dim3(kBlock, 1, 1), 0, join_stream(tan_tau_rt)>>>(
      c_in_re.const_data_ptr<float>(),
      c_in_im.const_data_ptr<float>(),
      c_out_re.const_data_ptr<float>(),
      c_out_im.const_data_ptr<float>(),
      s_re.const_data_ptr<float>(),
      s_im.const_data_ptr<float>(),
      row_valid.const_data_ptr<int32_t>(),
      idx_in.const_data_ptr<int64_t>(),
      idx_out.const_data_ptr<int64_t>(),
      idx_s.const_data_ptr<int64_t>(),
      tan_tau_in.const_data_ptr<float>(),
      tan_tau_out.const_data_ptr<float>(),
      tan_c_in_re.const_data_ptr<float>(),
      tan_c_in_im.const_data_ptr<float>(),
      tan_c_out_re.const_data_ptr<float>(),
      tan_c_out_im.const_data_ptr<float>(),
      tan_s_re.const_data_ptr<float>(),
      tan_s_im.const_data_ptr<float>(),
      tan_tau_rt.mutable_data_ptr<float>(),
      tan_rate_rt.mutable_data_ptr<float>(),
      tan_c_rt_re.mutable_data_ptr<float>(),
      tan_c_rt_im.mutable_data_ptr<float>(),
      rows);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void two_way_join_backward_cuda(
    const torch::stable::Tensor& c_in_re,
    const torch::stable::Tensor& c_in_im,
    const torch::stable::Tensor& c_out_re,
    const torch::stable::Tensor& c_out_im,
    const torch::stable::Tensor& s_re,
    const torch::stable::Tensor& s_im,
    const torch::stable::Tensor& row_valid,
    const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out,
    const torch::stable::Tensor& idx_s,
    const torch::stable::Tensor& by_in_offsets,
    const torch::stable::Tensor& by_in_rows,
    const torch::stable::Tensor& by_out_offsets,
    const torch::stable::Tensor& by_out_rows,
    const torch::stable::Tensor& by_s_offsets,
    const torch::stable::Tensor& by_s_rows,
    const torch::stable::Tensor& grad_tau_rt,
    const torch::stable::Tensor& grad_c_rt_re,
    const torch::stable::Tensor& grad_c_rt_im,
    torch::stable::Tensor& grad_tau_in,
    torch::stable::Tensor& grad_tau_out,
    torch::stable::Tensor& grad_c_in_re,
    torch::stable::Tensor& grad_c_in_im,
    torch::stable::Tensor& grad_c_out_re,
    torch::stable::Tensor& grad_c_out_im,
    torch::stable::Tensor& grad_s_re,
    torch::stable::Tensor& grad_s_im,
    int64_t num_rows,
    int64_t num_in,
    int64_t num_out,
    int64_t num_sites) {
  const int rows = checked_int(num_rows, "num_rows");
  const int legs_in = checked_int(num_in, "num_in");
  const int legs_out = checked_int(num_out, "num_out");
  const int sites = checked_int(num_sites, "num_sites");
  check_pair(c_in_re, c_in_im, legs_in, "c_in");
  check_pair(c_out_re, c_out_im, legs_out, "c_out");
  check_pair(s_re, s_im, sites, "s");
  check_cuda_int(row_valid, "row_valid");
  STD_TORCH_CHECK(
      row_valid.numel() == static_cast<int64_t>(rows),
      "row_valid must hold one flag per composed row.");
  check_index(idx_in, rows, "idx_in");
  check_index(idx_out, rows, "idx_out");
  check_index(idx_s, rows, "idx_s");
  check_cuda_long(by_in_offsets, "by_in_offsets");
  check_cuda_long(by_in_rows, "by_in_rows");
  check_cuda_long(by_out_offsets, "by_out_offsets");
  check_cuda_long(by_out_rows, "by_out_rows");
  check_cuda_long(by_s_offsets, "by_s_offsets");
  check_cuda_long(by_s_rows, "by_s_rows");
  STD_TORCH_CHECK(
      by_in_offsets.numel() == static_cast<int64_t>(legs_in) + 1 &&
          by_out_offsets.numel() == static_cast<int64_t>(legs_out) + 1 &&
          by_s_offsets.numel() == static_cast<int64_t>(sites) + 1,
      "each CSR offsets table must hold one entry per owner plus one.");
  STD_TORCH_CHECK(
      by_in_rows.numel() == static_cast<int64_t>(rows) &&
          by_out_rows.numel() == static_cast<int64_t>(rows) &&
          by_s_rows.numel() == static_cast<int64_t>(rows),
      "each CSR row table must permute every composed row exactly once.");
  check_float_len(grad_tau_rt, rows, "grad_tau_rt");
  check_pair(grad_c_rt_re, grad_c_rt_im, rows, "grad_c_rt");
  check_float_len(grad_tau_in, legs_in, "grad_tau_in");
  check_pair(grad_c_in_re, grad_c_in_im, legs_in, "grad_c_in");
  check_float_len(grad_tau_out, legs_out, "grad_tau_out");
  check_pair(grad_c_out_re, grad_c_out_im, legs_out, "grad_c_out");
  check_pair(grad_s_re, grad_s_im, sites, "grad_s");

  const int total = legs_in + legs_out + sites;
  if (total == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      grad_s_re.get_device_index());
  two_way_join_backward_kernel<<<
      linear_grid(total), dim3(kBlock, 1, 1), 0, join_stream(grad_s_re)>>>(
      c_in_re.const_data_ptr<float>(),
      c_in_im.const_data_ptr<float>(),
      c_out_re.const_data_ptr<float>(),
      c_out_im.const_data_ptr<float>(),
      s_re.const_data_ptr<float>(),
      s_im.const_data_ptr<float>(),
      row_valid.const_data_ptr<int32_t>(),
      idx_in.const_data_ptr<int64_t>(),
      idx_out.const_data_ptr<int64_t>(),
      idx_s.const_data_ptr<int64_t>(),
      by_in_offsets.const_data_ptr<int64_t>(),
      by_in_rows.const_data_ptr<int64_t>(),
      by_out_offsets.const_data_ptr<int64_t>(),
      by_out_rows.const_data_ptr<int64_t>(),
      by_s_offsets.const_data_ptr<int64_t>(),
      by_s_rows.const_data_ptr<int64_t>(),
      grad_tau_rt.const_data_ptr<float>(),
      grad_c_rt_re.const_data_ptr<float>(),
      grad_c_rt_im.const_data_ptr<float>(),
      grad_tau_in.mutable_data_ptr<float>(),
      grad_tau_out.mutable_data_ptr<float>(),
      grad_c_in_re.mutable_data_ptr<float>(),
      grad_c_in_im.mutable_data_ptr<float>(),
      grad_c_out_re.mutable_data_ptr<float>(),
      grad_c_out_im.mutable_data_ptr<float>(),
      grad_s_re.mutable_data_ptr<float>(),
      grad_s_im.mutable_data_ptr<float>(),
      legs_in,
      legs_out,
      sites);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("two_way_join_forward", TORCH_BOX(&two_way_join_forward_cuda));
  m.impl("two_way_join_backward", TORCH_BOX(&two_way_join_backward_cuda));
  m.impl("two_way_join_jvp", TORCH_BOX(&two_way_join_jvp_cuda));
}
