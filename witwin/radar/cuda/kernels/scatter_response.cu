// Aspect-dependent scatter response: one complex factor per COMPOSED row.
//
// This is the kernel that item 6b/6c of the Phase-7 plan needs and that
// ``TwoWayComposer.compose`` refuses to evaluate in Torch. A response that
// depends on the inbound and outbound directions varies per path, so it is
// hot-path physics; the standing refusal in ``paths/two_way.py`` exists to
// stop it becoming a Torch expression, and this file is the route through it.
//
// Per composed row k, with i = idx_in[k], o = idx_out[k], s = idx_site[k]:
//
//   u  = axis[s]                     (unit, validated on the host)
//   ci = -dot(dir_in[i], u)          incidence cosine at the site
//   co =  dot(dir_out[o], u)         scattering cosine at the site
//   gi = ci > 0 ? ci^n : 0
//   go = co > 0 ? co^n : 0
//   m  = amplitude[s] * gi * go
//   S  = m * exp(-i * phase[s])
//
// The sign of ``ci`` is the whole content of the geometry: ``dir_in`` is the
// PROPAGATION direction of the inbound field, so it points INTO the site, and
// the cosine against an outward aspect axis is its negative. ``dir_out``
// points away from the site and enters directly. Getting that one sign wrong
// produces a lobe that is exactly backwards and still looks like a lobe.
//
// The clamp is physical rather than numerical: a negative cosine is a
// direction on the far side of the aspect plane, which this separable law does
// not illuminate. Its derivative is the right-hand limit, which is zero for
// n >= 1; the clamp boundary is a measure-zero set that no fixture sits on.
//
// exp(-i phase) is the CHANNEL phasor, matching the transports the response
// multiplies. The conversion to the beat convention happens once, downstream.
//
// Four rules, each pinned by a test:
//
//  1. Everything accumulates in double and stores float32, matching
//     two_way_join.cu and fmcw_beat.cu. ``pow`` is evaluated in double.
//  2. A dead row publishes exactly zero. The join zeroes its own payload for
//     the same row, so this is belt and braces rather than the authority - but
//     a dead row can carry a stale or uninitialised direction, and feeding a
//     NaN into the join's complex product would poison a LIVE row's tangent
//     through nothing but the response tensor's shared storage.
//  3. The VJP owns one gradient slot per thread and uses the SAME frozen CSR
//     tables as the join backward, so there are no atomics and the summation
//     order is a property of the frozen composition. Three owner families:
//     inbound leg rows (dir_in), outbound leg rows (dir_out), and sites
//     (axis, amplitude, phase).
//  4. The exponent is a host scalar and is NOT differentiable. It selects the
//     law, and a fractional derivative of a clamped power is not something
//     this contract claims.
//
// Row validity is computed in Torch and enters as an int32 mask, exactly as in
// two_way_join.cu; the reasons are recorded there and are unchanged.

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

struct Vec3 {
  double x;
  double y;
  double z;
};

__device__ __forceinline__ Vec3 load3(
    const float* __restrict__ data, const int64_t index) {
  const int64_t base = index * 3;
  return {
      static_cast<double>(data[base]),
      static_cast<double>(data[base + 1]),
      static_cast<double>(data[base + 2])};
}

__device__ __forceinline__ double dot3(const Vec3 a, const Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// The clamped lobe and its derivative, evaluated together because every caller
// needs both and recomputing pow() is the expensive half.
__device__ __forceinline__ void lobe(
    const double cosine, const double exponent, double* value, double* slope) {
  if (!(cosine > 0.0)) {
    *value = 0.0;
    *slope = 0.0;
    return;
  }
  *value = pow(cosine, exponent);
  *slope = exponent * pow(cosine, exponent - 1.0);
}

struct RowTerms {
  double gi;
  double go;
  double slope_i;
  double slope_o;
  double amplitude;
  double phase;
  double magnitude;
};

__device__ __forceinline__ RowTerms row_terms(
    const Vec3 d_in,
    const Vec3 d_out,
    const Vec3 axis,
    const double amplitude,
    const double phase,
    const double exponent) {
  RowTerms terms;
  const double ci = -dot3(d_in, axis);
  const double co = dot3(d_out, axis);
  lobe(ci, exponent, &terms.gi, &terms.slope_i);
  lobe(co, exponent, &terms.go, &terms.slope_o);
  terms.amplitude = amplitude;
  terms.phase = phase;
  terms.magnitude = amplitude * terms.gi * terms.go;
  return terms;
}

__global__ void scatter_response_aspect_forward_kernel(
    const float* __restrict__ dir_in,
    const float* __restrict__ dir_out,
    const int64_t* __restrict__ idx_in,
    const int64_t* __restrict__ idx_out,
    const int64_t* __restrict__ idx_site,
    const float* __restrict__ axis,
    const float* __restrict__ amplitude,
    const float* __restrict__ phase_rad,
    const int32_t* __restrict__ row_valid,
    float* __restrict__ s_re,
    float* __restrict__ s_im,
    const double exponent,
    const int num_rows) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_rows) {
    return;
  }
  if (row_valid[k] == 0) {
    s_re[k] = 0.0f;
    s_im[k] = 0.0f;
    return;
  }
  const int64_t s = idx_site[k];
  const RowTerms terms = row_terms(
      load3(dir_in, idx_in[k]),
      load3(dir_out, idx_out[k]),
      load3(axis, s),
      static_cast<double>(amplitude[s]),
      static_cast<double>(phase_rad[s]),
      exponent);
  s_re[k] = static_cast<float>(terms.magnitude * cos(terms.phase));
  s_im[k] = static_cast<float>(-terms.magnitude * sin(terms.phase));
}

__global__ void scatter_response_aspect_jvp_kernel(
    const float* __restrict__ dir_in,
    const float* __restrict__ dir_out,
    const int64_t* __restrict__ idx_in,
    const int64_t* __restrict__ idx_out,
    const int64_t* __restrict__ idx_site,
    const float* __restrict__ axis,
    const float* __restrict__ amplitude,
    const float* __restrict__ phase_rad,
    const int32_t* __restrict__ row_valid,
    const float* __restrict__ tan_dir_in,
    const float* __restrict__ tan_dir_out,
    const float* __restrict__ tan_axis,
    const float* __restrict__ tan_amplitude,
    const float* __restrict__ tan_phase_rad,
    float* __restrict__ tan_s_re,
    float* __restrict__ tan_s_im,
    const double exponent,
    const int num_rows) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_rows) {
    return;
  }
  if (row_valid[k] == 0) {
    tan_s_re[k] = 0.0f;
    tan_s_im[k] = 0.0f;
    return;
  }
  const int64_t i = idx_in[k];
  const int64_t o = idx_out[k];
  const int64_t s = idx_site[k];
  const Vec3 d_in = load3(dir_in, i);
  const Vec3 d_out = load3(dir_out, o);
  const Vec3 u = load3(axis, s);
  const RowTerms terms = row_terms(
      d_in,
      d_out,
      u,
      static_cast<double>(amplitude[s]),
      static_cast<double>(phase_rad[s]),
      exponent);

  const Vec3 t_in = load3(tan_dir_in, i);
  const Vec3 t_out = load3(tan_dir_out, o);
  const Vec3 t_u = load3(tan_axis, s);
  // ci = -dot(d_in, u) and co = dot(d_out, u), both bilinear.
  const double t_ci = -(dot3(t_in, u) + dot3(d_in, t_u));
  const double t_co = dot3(t_out, u) + dot3(d_out, t_u);
  const double t_magnitude =
      terms.gi * terms.go * static_cast<double>(tan_amplitude[s]) +
      terms.amplitude * terms.go * terms.slope_i * t_ci +
      terms.amplitude * terms.gi * terms.slope_o * t_co;
  const double t_phase = static_cast<double>(tan_phase_rad[s]);
  const double cos_phase = cos(terms.phase);
  const double sin_phase = sin(terms.phase);
  tan_s_re[k] = static_cast<float>(
      cos_phase * t_magnitude - terms.magnitude * sin_phase * t_phase);
  tan_s_im[k] = static_cast<float>(
      -sin_phase * t_magnitude - terms.magnitude * cos_phase * t_phase);
}

// One thread owns one gradient slot. The three owner families - inbound leg
// rows, outbound leg rows, sites - are laid end to end on a single grid, so a
// tangent-free reverse pass is one launch and no thread touches another's
// output. The tables are the join's own frozen CSR, so this VJP sums in the
// same order the join's does.
__global__ void scatter_response_aspect_backward_kernel(
    const float* __restrict__ dir_in,
    const float* __restrict__ dir_out,
    const int64_t* __restrict__ idx_in,
    const int64_t* __restrict__ idx_out,
    const int64_t* __restrict__ idx_site,
    const float* __restrict__ axis,
    const float* __restrict__ amplitude,
    const float* __restrict__ phase_rad,
    const int32_t* __restrict__ row_valid,
    const int64_t* __restrict__ by_in_offsets,
    const int64_t* __restrict__ by_in_rows,
    const int64_t* __restrict__ by_out_offsets,
    const int64_t* __restrict__ by_out_rows,
    const int64_t* __restrict__ by_site_offsets,
    const int64_t* __restrict__ by_site_rows,
    const float* __restrict__ grad_s_re,
    const float* __restrict__ grad_s_im,
    float* __restrict__ grad_dir_in,
    float* __restrict__ grad_dir_out,
    float* __restrict__ grad_axis,
    float* __restrict__ grad_amplitude,
    float* __restrict__ grad_phase_rad,
    const double exponent,
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
    offsets = by_site_offsets;
    segment_rows = by_site_rows;
  }

  Vec3 vector = {0.0, 0.0, 0.0};
  double scalar_a = 0.0;
  double scalar_b = 0.0;
  const int64_t begin = offsets[slot];
  const int64_t end = offsets[slot + 1];
  for (int64_t entry = begin; entry < end; ++entry) {
    const int64_t k = segment_rows[entry];
    if (row_valid[k] == 0) {
      continue;
    }
    const int64_t i = idx_in[k];
    const int64_t o = idx_out[k];
    const int64_t s = idx_site[k];
    const Vec3 d_in = load3(dir_in, i);
    const Vec3 d_out = load3(dir_out, o);
    const Vec3 u = load3(axis, s);
    const RowTerms terms = row_terms(
        d_in,
        d_out,
        u,
        static_cast<double>(amplitude[s]),
        static_cast<double>(phase_rad[s]),
        exponent);
    const double cos_phase = cos(terms.phase);
    const double sin_phase = sin(terms.phase);
    const double g_re = static_cast<double>(grad_s_re[k]);
    const double g_im = static_cast<double>(grad_s_im[k]);
    // S = m cos(phi) - i m sin(phi), carried as a real pair, so the pullback
    // to (m, phi) is the plain real Jacobian transpose. No complex tensor ever
    // crosses the autograd boundary, so the Wirtinger convention cannot enter.
    const double g_magnitude = g_re * cos_phase - g_im * sin_phase;
    const double g_phase =
        -terms.magnitude * (g_re * sin_phase + g_im * cos_phase);
    const double g_ci = terms.amplitude * terms.go * terms.slope_i * g_magnitude;
    const double g_co = terms.amplitude * terms.gi * terms.slope_o * g_magnitude;

    if (family == 0) {
      // ci = -dot(d_in, u)
      vector.x += -u.x * g_ci;
      vector.y += -u.y * g_ci;
      vector.z += -u.z * g_ci;
    } else if (family == 1) {
      // co = dot(d_out, u)
      vector.x += u.x * g_co;
      vector.y += u.y * g_co;
      vector.z += u.z * g_co;
    } else {
      vector.x += -d_in.x * g_ci + d_out.x * g_co;
      vector.y += -d_in.y * g_ci + d_out.y * g_co;
      vector.z += -d_in.z * g_ci + d_out.z * g_co;
      scalar_a += terms.gi * terms.go * g_magnitude;
      scalar_b += g_phase;
    }
  }

  if (family == 0) {
    const int64_t base = static_cast<int64_t>(slot) * 3;
    grad_dir_in[base] = static_cast<float>(vector.x);
    grad_dir_in[base + 1] = static_cast<float>(vector.y);
    grad_dir_in[base + 2] = static_cast<float>(vector.z);
  } else if (family == 1) {
    const int64_t base = static_cast<int64_t>(slot) * 3;
    grad_dir_out[base] = static_cast<float>(vector.x);
    grad_dir_out[base + 1] = static_cast<float>(vector.y);
    grad_dir_out[base + 2] = static_cast<float>(vector.z);
  } else {
    const int64_t base = static_cast<int64_t>(slot) * 3;
    grad_axis[base] = static_cast<float>(vector.x);
    grad_axis[base + 1] = static_cast<float>(vector.y);
    grad_axis[base + 2] = static_cast<float>(vector.z);
    grad_amplitude[slot] = static_cast<float>(scalar_a);
    grad_phase_rad[slot] = static_cast<float>(scalar_b);
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

void check_len(
    const torch::stable::Tensor& tensor, int64_t expected, const char* name) {
  check_cuda_float(tensor, name);
  STD_TORCH_CHECK(
      tensor.numel() == expected, name, " must hold one value per owning slot.");
}

void check_vec3(
    const torch::stable::Tensor& tensor, int64_t rows, const char* name) {
  check_cuda_float(tensor, name);
  STD_TORCH_CHECK(
      tensor.numel() == rows * 3, name, " must hold three values per row.");
}

void check_index(
    const torch::stable::Tensor& index, int rows, const char* name) {
  check_cuda_long(index, name);
  STD_TORCH_CHECK(
      index.numel() == static_cast<int64_t>(rows),
      name,
      " must hold one index per composed row.");
}

cudaStream_t response_stream(const torch::stable::Tensor& tensor) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

constexpr int kBlock = 256;

dim3 linear_grid(int count) {
  return dim3(static_cast<unsigned>((count + kBlock - 1) / kBlock), 1, 1);
}

void check_shared_inputs(
    const torch::stable::Tensor& dir_in,
    const torch::stable::Tensor& dir_out,
    const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out,
    const torch::stable::Tensor& idx_site,
    const torch::stable::Tensor& axis,
    const torch::stable::Tensor& amplitude,
    const torch::stable::Tensor& phase_rad,
    const torch::stable::Tensor& row_valid,
    int rows) {
  check_cuda_float(dir_in, "dir_in");
  check_cuda_float(dir_out, "dir_out");
  STD_TORCH_CHECK(
      dir_in.numel() % 3 == 0 && dir_out.numel() % 3 == 0,
      "dir_in and dir_out must hold three values per leg row.");
  check_index(idx_in, rows, "idx_in");
  check_index(idx_out, rows, "idx_out");
  check_index(idx_site, rows, "idx_site");
  check_cuda_float(axis, "axis");
  STD_TORCH_CHECK(
      axis.numel() % 3 == 0, "axis must hold three values per site.");
  const int64_t sites = axis.numel() / 3;
  check_len(amplitude, sites, "amplitude");
  check_len(phase_rad, sites, "phase_rad");
  check_cuda_int(row_valid, "row_valid");
  STD_TORCH_CHECK(
      row_valid.numel() == static_cast<int64_t>(rows),
      "row_valid must hold one flag per composed row.");
}

}  // namespace

void scatter_response_aspect_forward_cuda(
    const torch::stable::Tensor& dir_in,
    const torch::stable::Tensor& dir_out,
    const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out,
    const torch::stable::Tensor& idx_site,
    const torch::stable::Tensor& axis,
    const torch::stable::Tensor& amplitude,
    const torch::stable::Tensor& phase_rad,
    const torch::stable::Tensor& row_valid,
    torch::stable::Tensor& s_re,
    torch::stable::Tensor& s_im,
    double exponent,
    int64_t num_rows) {
  const int rows = checked_int(num_rows, "num_rows");
  check_shared_inputs(
      dir_in,
      dir_out,
      idx_in,
      idx_out,
      idx_site,
      axis,
      amplitude,
      phase_rad,
      row_valid,
      rows);
  check_len(s_re, rows, "s_re");
  check_len(s_im, rows, "s_im");
  STD_TORCH_CHECK(exponent >= 1.0, "exponent must be at least 1.");

  if (rows == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      s_re.get_device_index());
  scatter_response_aspect_forward_kernel<<<
      linear_grid(rows), dim3(kBlock, 1, 1), 0, response_stream(s_re)>>>(
      dir_in.const_data_ptr<float>(),
      dir_out.const_data_ptr<float>(),
      idx_in.const_data_ptr<int64_t>(),
      idx_out.const_data_ptr<int64_t>(),
      idx_site.const_data_ptr<int64_t>(),
      axis.const_data_ptr<float>(),
      amplitude.const_data_ptr<float>(),
      phase_rad.const_data_ptr<float>(),
      row_valid.const_data_ptr<int32_t>(),
      s_re.mutable_data_ptr<float>(),
      s_im.mutable_data_ptr<float>(),
      exponent,
      rows);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_response_aspect_jvp_cuda(
    const torch::stable::Tensor& dir_in,
    const torch::stable::Tensor& dir_out,
    const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out,
    const torch::stable::Tensor& idx_site,
    const torch::stable::Tensor& axis,
    const torch::stable::Tensor& amplitude,
    const torch::stable::Tensor& phase_rad,
    const torch::stable::Tensor& row_valid,
    const torch::stable::Tensor& tan_dir_in,
    const torch::stable::Tensor& tan_dir_out,
    const torch::stable::Tensor& tan_axis,
    const torch::stable::Tensor& tan_amplitude,
    const torch::stable::Tensor& tan_phase_rad,
    torch::stable::Tensor& tan_s_re,
    torch::stable::Tensor& tan_s_im,
    double exponent,
    int64_t num_rows) {
  const int rows = checked_int(num_rows, "num_rows");
  check_shared_inputs(
      dir_in,
      dir_out,
      idx_in,
      idx_out,
      idx_site,
      axis,
      amplitude,
      phase_rad,
      row_valid,
      rows);
  check_vec3(tan_dir_in, dir_in.numel() / 3, "tan_dir_in");
  check_vec3(tan_dir_out, dir_out.numel() / 3, "tan_dir_out");
  check_vec3(tan_axis, axis.numel() / 3, "tan_axis");
  check_len(tan_amplitude, amplitude.numel(), "tan_amplitude");
  check_len(tan_phase_rad, phase_rad.numel(), "tan_phase_rad");
  check_len(tan_s_re, rows, "tan_s_re");
  check_len(tan_s_im, rows, "tan_s_im");
  STD_TORCH_CHECK(exponent >= 1.0, "exponent must be at least 1.");

  if (rows == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      tan_s_re.get_device_index());
  scatter_response_aspect_jvp_kernel<<<
      linear_grid(rows), dim3(kBlock, 1, 1), 0, response_stream(tan_s_re)>>>(
      dir_in.const_data_ptr<float>(),
      dir_out.const_data_ptr<float>(),
      idx_in.const_data_ptr<int64_t>(),
      idx_out.const_data_ptr<int64_t>(),
      idx_site.const_data_ptr<int64_t>(),
      axis.const_data_ptr<float>(),
      amplitude.const_data_ptr<float>(),
      phase_rad.const_data_ptr<float>(),
      row_valid.const_data_ptr<int32_t>(),
      tan_dir_in.const_data_ptr<float>(),
      tan_dir_out.const_data_ptr<float>(),
      tan_axis.const_data_ptr<float>(),
      tan_amplitude.const_data_ptr<float>(),
      tan_phase_rad.const_data_ptr<float>(),
      tan_s_re.mutable_data_ptr<float>(),
      tan_s_im.mutable_data_ptr<float>(),
      exponent,
      rows);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_response_aspect_backward_cuda(
    const torch::stable::Tensor& dir_in,
    const torch::stable::Tensor& dir_out,
    const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out,
    const torch::stable::Tensor& idx_site,
    const torch::stable::Tensor& axis,
    const torch::stable::Tensor& amplitude,
    const torch::stable::Tensor& phase_rad,
    const torch::stable::Tensor& row_valid,
    const torch::stable::Tensor& by_in_offsets,
    const torch::stable::Tensor& by_in_rows,
    const torch::stable::Tensor& by_out_offsets,
    const torch::stable::Tensor& by_out_rows,
    const torch::stable::Tensor& by_site_offsets,
    const torch::stable::Tensor& by_site_rows,
    const torch::stable::Tensor& grad_s_re,
    const torch::stable::Tensor& grad_s_im,
    torch::stable::Tensor& grad_dir_in,
    torch::stable::Tensor& grad_dir_out,
    torch::stable::Tensor& grad_axis,
    torch::stable::Tensor& grad_amplitude,
    torch::stable::Tensor& grad_phase_rad,
    double exponent,
    int64_t num_rows,
    int64_t num_in,
    int64_t num_out,
    int64_t num_sites) {
  const int rows = checked_int(num_rows, "num_rows");
  const int legs_in = checked_int(num_in, "num_in");
  const int legs_out = checked_int(num_out, "num_out");
  const int sites = checked_int(num_sites, "num_sites");
  check_shared_inputs(
      dir_in,
      dir_out,
      idx_in,
      idx_out,
      idx_site,
      axis,
      amplitude,
      phase_rad,
      row_valid,
      rows);
  STD_TORCH_CHECK(
      dir_in.numel() == static_cast<int64_t>(legs_in) * 3 &&
          dir_out.numel() == static_cast<int64_t>(legs_out) * 3 &&
          axis.numel() == static_cast<int64_t>(sites) * 3,
      "the declared owner counts must match the direction and axis tables.");
  check_cuda_long(by_in_offsets, "by_in_offsets");
  check_cuda_long(by_in_rows, "by_in_rows");
  check_cuda_long(by_out_offsets, "by_out_offsets");
  check_cuda_long(by_out_rows, "by_out_rows");
  check_cuda_long(by_site_offsets, "by_site_offsets");
  check_cuda_long(by_site_rows, "by_site_rows");
  STD_TORCH_CHECK(
      by_in_offsets.numel() == static_cast<int64_t>(legs_in) + 1 &&
          by_out_offsets.numel() == static_cast<int64_t>(legs_out) + 1 &&
          by_site_offsets.numel() == static_cast<int64_t>(sites) + 1,
      "each CSR offsets table must hold one entry per owner plus one.");
  STD_TORCH_CHECK(
      by_in_rows.numel() == static_cast<int64_t>(rows) &&
          by_out_rows.numel() == static_cast<int64_t>(rows) &&
          by_site_rows.numel() == static_cast<int64_t>(rows),
      "each CSR row table must permute every composed row exactly once.");
  check_len(grad_s_re, rows, "grad_s_re");
  check_len(grad_s_im, rows, "grad_s_im");
  check_vec3(grad_dir_in, legs_in, "grad_dir_in");
  check_vec3(grad_dir_out, legs_out, "grad_dir_out");
  check_vec3(grad_axis, sites, "grad_axis");
  check_len(grad_amplitude, sites, "grad_amplitude");
  check_len(grad_phase_rad, sites, "grad_phase_rad");
  STD_TORCH_CHECK(exponent >= 1.0, "exponent must be at least 1.");

  const int total = legs_in + legs_out + sites;
  if (total == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      grad_axis.get_device_index());
  scatter_response_aspect_backward_kernel<<<
      linear_grid(total), dim3(kBlock, 1, 1), 0, response_stream(grad_axis)>>>(
      dir_in.const_data_ptr<float>(),
      dir_out.const_data_ptr<float>(),
      idx_in.const_data_ptr<int64_t>(),
      idx_out.const_data_ptr<int64_t>(),
      idx_site.const_data_ptr<int64_t>(),
      axis.const_data_ptr<float>(),
      amplitude.const_data_ptr<float>(),
      phase_rad.const_data_ptr<float>(),
      row_valid.const_data_ptr<int32_t>(),
      by_in_offsets.const_data_ptr<int64_t>(),
      by_in_rows.const_data_ptr<int64_t>(),
      by_out_offsets.const_data_ptr<int64_t>(),
      by_out_rows.const_data_ptr<int64_t>(),
      by_site_offsets.const_data_ptr<int64_t>(),
      by_site_rows.const_data_ptr<int64_t>(),
      grad_s_re.const_data_ptr<float>(),
      grad_s_im.const_data_ptr<float>(),
      grad_dir_in.mutable_data_ptr<float>(),
      grad_dir_out.mutable_data_ptr<float>(),
      grad_axis.mutable_data_ptr<float>(),
      grad_amplitude.mutable_data_ptr<float>(),
      grad_phase_rad.mutable_data_ptr<float>(),
      exponent,
      legs_in,
      legs_out,
      sites);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(witwin_radar_dirichlet_cuda, CUDA, m) {
  m.impl(
      "scatter_response_aspect_forward",
      TORCH_BOX(&scatter_response_aspect_forward_cuda));
  m.impl(
      "scatter_response_aspect_jvp",
      TORCH_BOX(&scatter_response_aspect_jvp_cuda));
  m.impl(
      "scatter_response_aspect_backward",
      TORCH_BOX(&scatter_response_aspect_backward_cuda));
}
