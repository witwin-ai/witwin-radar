// Aspect-dependent scatter response: one complex factor per COMPOSED row.
//
// This is the kernel that ``TwoWayComposer.compose`` refuses to evaluate in
// Torch. A response that depends on the inbound and outbound directions varies
// per path, so it is hot-path physics; the standing refusal in ``paths.py``
// exists to stop it becoming a Torch expression, and this file is the route
// through it.
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
#include "radar_checks.cuh"
#include "scatter_jets.cuh"

#include <cstdint>

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

// The slot stride of a slot-major batch: idx_in, idx_out and idx_site
// describe ONE slot of rows_per_slot composed rows; row k is join row
// k % rows_per_slot of slot g = k / rows_per_slot, whose leg rows are offset by
// g times the legs' per-slot row counts. Sites are shared by every slot
// (sites_per_slot == 0) or slot local (sites_per_slot > 0: slot g's site s is
// owner g * sites_per_slot + s). A reverse-pass owner walks its per-slot CSR
// once per slot it appears in, in slot order: the summation order of the
// per-slot table copies, so the strided route is bit-identical to them.
struct Strided {
  int64_t i, o, s;
};

struct OwnerSlots {
  int64_t local, first, last;
};

struct Stride {
  int64_t rows_per_slot;
  int64_t in_per_slot;
  int64_t out_per_slot;
  int64_t sites_per_slot;
  int64_t slots;

  __device__ __forceinline__ Strided row(
      const int64_t k, const int64_t* __restrict__ idx_in, const int64_t* __restrict__ idx_out,
      const int64_t* __restrict__ idx_site) const {
    const int64_t g = k / rows_per_slot;
    const int64_t r = k - g * rows_per_slot;
    return {idx_in[r] + g * in_per_slot, idx_out[r] + g * out_per_slot, idx_site[r] + g * sites_per_slot};
  }

  // An owner that belongs to one slot: global index -> (local, slot, slot + 1).
  __device__ __forceinline__ OwnerSlots owner(const int64_t global, const int64_t per_slot) const {
    const int64_t g = global / per_slot;
    return {global - g * per_slot, g, g + 1};
  }

  // An owner shared by every slot.
  __device__ __forceinline__ OwnerSlots shared(const int64_t local) const { return {local, 0, slots}; }
};

inline Stride make_stride(
    const torch::stable::Tensor& idx_in, const torch::stable::Tensor& dir_in, const torch::stable::Tensor& dir_out,
    const int64_t sites, const int64_t slots, const int64_t slot_local_sites) {
  STD_TORCH_CHECK(slots >= 1, "slots must be at least 1.");
  STD_TORCH_CHECK(
      (dir_in.numel() / 3) % slots == 0 && (dir_out.numel() / 3) % slots == 0,
      "every slot carries the same leg rows.");
  STD_TORCH_CHECK(slot_local_sites == 0 || sites % slots == 0, "slot-local sites hold the same count in every slot.");
  return {
      idx_in.numel(), dir_in.numel() / 3 / slots, dir_out.numel() / 3 / slots,
      slot_local_sites == 0 ? 0 : sites / slots, slots};
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
    const Stride stride,
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
  const Strided at = stride.row(k, idx_in, idx_out, idx_site);
  const int64_t s = at.s;
  const RowTerms terms = row_terms(
      load3(dir_in, at.i),
      load3(dir_out, at.o),
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
    const Stride stride,
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
  const Strided at = stride.row(k, idx_in, idx_out, idx_site);
  const int64_t i = at.i;
  const int64_t o = at.o;
  const int64_t s = at.s;
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
    const Stride stride,
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
  OwnerSlots owned;
  if (thread < num_in) {
    family = 0;
    slot = thread;
    offsets = by_in_offsets;
    segment_rows = by_in_rows;
    owned = stride.owner(thread, stride.in_per_slot);
  } else if (thread < num_in + num_out) {
    family = 1;
    slot = thread - num_in;
    offsets = by_out_offsets;
    segment_rows = by_out_rows;
    owned = stride.owner(slot, stride.out_per_slot);
  } else {
    family = 2;
    slot = thread - num_in - num_out;
    offsets = by_site_offsets;
    segment_rows = by_site_rows;
    owned = stride.sites_per_slot > 0 ? stride.owner(slot, stride.sites_per_slot) : stride.shared(slot);
  }

  Vec3 vector = {0.0, 0.0, 0.0};
  double scalar_a = 0.0;
  double scalar_b = 0.0;
  const int64_t begin = offsets[owned.local];
  const int64_t end = offsets[owned.local + 1];
  for (int64_t g = owned.first; g < owned.last; ++g) {
  for (int64_t entry = begin; entry < end; ++entry) {
    const int64_t k = g * stride.rows_per_slot + segment_rows[entry];
    if (row_valid[k] == 0) {
      continue;
    }
    const Strided at = stride.row(k, idx_in, idx_out, idx_site);
    const int64_t i = at.i;
    const int64_t o = at.o;
    const int64_t s = at.s;
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
  const int per_slot = checked_int(idx_in.numel(), "rows_per_slot");
  check_index(idx_in, per_slot, "idx_in");
  check_index(idx_out, per_slot, "idx_out");
  check_index(idx_site, per_slot, "idx_site");
  STD_TORCH_CHECK(per_slot == 0 ? rows == 0 : rows % per_slot == 0, "the rows must be whole slots of the index tables.");
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
    int64_t num_rows,
    int64_t slots,
    int64_t slot_local_sites) {
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
      linear_grid(rows), dim3(kBlock, 1, 1), 0, current_cuda_stream(s_re)>>>(
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
      make_stride(idx_in, dir_in, dir_out, axis.numel() / 3, slots, slot_local_sites),
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
    int64_t num_rows,
    int64_t slots,
    int64_t slot_local_sites) {
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
      linear_grid(rows), dim3(kBlock, 1, 1), 0, current_cuda_stream(tan_s_re)>>>(
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
      make_stride(idx_in, dir_in, dir_out, axis.numel() / 3, slots, slot_local_sites),
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
    int64_t num_sites,
    int64_t slots,
    int64_t slot_local_sites) {
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
  const Stride stride = make_stride(idx_in, dir_in, dir_out, sites, slots, slot_local_sites);
  const int64_t site_owners = stride.sites_per_slot > 0 ? stride.sites_per_slot : sites;
  STD_TORCH_CHECK(
      by_in_offsets.numel() == stride.in_per_slot + 1 &&
          by_out_offsets.numel() == stride.out_per_slot + 1 &&
          by_site_offsets.numel() == site_owners + 1,
      "each CSR offsets table must hold one entry per owner of one slot plus one.");
  STD_TORCH_CHECK(
      by_in_rows.numel() == stride.rows_per_slot &&
          by_out_rows.numel() == stride.rows_per_slot &&
          by_site_rows.numel() == stride.rows_per_slot,
      "each CSR row table must permute every composed row of one slot exactly once.");
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
      linear_grid(total), dim3(kBlock, 1, 1), 0, current_cuda_stream(grad_axis)>>>(
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
      stride,
      legs_in,
      legs_out,
      sites);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

// Unit departure bearing u=(b-a)/|b-a|. Positions are metres. Both JVP and
// VJP apply the symmetric Jacobian (I-u*u^T)/|b-a|. Degenerate/dead segments
// return zero. tests/test_multipath_directions.py supplies independent oracles.
template <bool Derivative>
__global__ void scatter_direction_kernel(
    const float* a, const float* b, const float* tangent, float* out, int n) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const Vec3 av = load3(a, row), bv = load3(b, row);
  const Vec3 d{bv.x-av.x, bv.y-av.y, bv.z-av.z};
  const double length = sqrt(dot3(d, d));
  Vec3 result{0., 0., 0.};
  if (length > 1e-20) {
    const Vec3 u{d.x/length, d.y/length, d.z/length};
    if constexpr (Derivative) {
      const Vec3 v = load3(tangent, row);
      const double parallel = dot3(u, v);
      result = {(v.x-u.x*parallel)/length, (v.y-u.y*parallel)/length, (v.z-u.z*parallel)/length};
    } else result = u;
  }
  out[3*row] = result.x; out[3*row+1] = result.y; out[3*row+2] = result.z;
}

template <bool Derivative>
void scatter_direction_launch(const torch::stable::Tensor& a, const torch::stable::Tensor& b,
    const torch::stable::Tensor& v, torch::stable::Tensor out) {
  const int n = checked_int(a.numel()/3, "rows");
  check_vec3(a, n, "origin"); check_vec3(b, n, "target"); check_vec3(out, n, "output");
  if constexpr (Derivative) check_vec3(v, n, "direction");
  STD_TORCH_CHECK(a.get_device_index() == b.get_device_index() &&
      a.get_device_index() == out.get_device_index() && a.get_device_index() == v.get_device_index(),
      "scatter direction tensors must share a CUDA device");
  if (!n) return;
  const torch::stable::accelerator::DeviceGuard guard(a.get_device_index());
  scatter_direction_kernel<Derivative><<<linear_grid(n), dim3(kBlock,1,1), 0, current_cuda_stream(out)>>>(
      a.const_data_ptr<float>(), b.const_data_ptr<float>(), v.const_data_ptr<float>(), out.mutable_data_ptr<float>(), n);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_direction_forward_cuda(const torch::stable::Tensor& a, const torch::stable::Tensor& b,
    torch::stable::Tensor out) { scatter_direction_launch<false>(a, b, a, out); }
void scatter_direction_derivative_cuda(const torch::stable::Tensor& a, const torch::stable::Tensor& b,
    const torch::stable::Tensor& v, torch::stable::Tensor out) { scatter_direction_launch<true>(a, b, v, out); }

// ---------------------------------------------------------------------------
// Tabulated response: S(theta) interpolated per composed row.
// ---------------------------------------------------------------------------
//
// theta is the bistatic aspect angle between the site's unit axis and the
// bisector b = u_in + u_out of the two leg directions at the site (u_in = -dir_in
// toward the wave's source, u_out = dir_out toward the receiver side); it is
// the monostatic aspect when the two coincide. The table holds the complex
// response S = sqrt(4 pi sigma)/lambda exp(-j phi) at theta_i = i pi / (A - 1),
// already reduced to the carrier on the host, and is interpolated linearly in
// theta. A row with b = 0 (exact forward scatter) reads the table at theta = 0
// with zero derivative; so does a row on the axis (theta = 0 or pi), where
// d theta / d cos theta is singular and the derivative is the one-sided limit
// of a measure-zero set. The table takes no derivative.

namespace table_response {

using namespace radar_jets;

template <int N>
__device__ __forceinline__ C<N> row(
    const V3<N>& dir_in, const V3<N>& dir_out, const V3<N>& axis, const float* __restrict__ table_re,
    const float* __restrict__ table_im, int entries) {
  const V3<N> bisector = dir_out - dir_in;
  const J<N> length2 = dot(bisector, bisector);
  if (!(length2.v > 1.0e-24)) {
    return cconst<N>(table_re[0], table_im[0]);
  }
  J<N> cosine = dot(axis, bisector) / jsqrt(length2);
  double theta;
  double slope;
  if (1.0 - cosine.v * cosine.v < 1.0e-12) {
    theta = cosine.v > 0.0 ? 0.0 : kJetPi;
    slope = 0.0;
  } else {
    theta = acos(cosine.v);
    slope = -1.0 / sqrt(1.0 - cosine.v * cosine.v);
  }
  const J<N> angle = lift(cosine, theta, slope);
  const double step = kJetPi / static_cast<double>(entries - 1);
  int index = static_cast<int>(floor(theta / step));
  index = index < 0 ? 0 : (index > entries - 2 ? entries - 2 : index);
  const J<N> t = (angle - jconst<N>(index * step)) * (1.0 / step);
  const C<N> low = cconst<N>(table_re[index], table_im[index]);
  const C<N> high = cconst<N>(table_re[index + 1], table_im[index + 1]);
  return low + (high - low) * t;
}

}  // namespace table_response

__global__ void scatter_response_table_forward_kernel(
    const float* dir_in, const float* dir_out, const int64_t* idx_in, const int64_t* idx_out,
    const int64_t* idx_site, const float* axis, const float* table_re, const float* table_im,
    const int32_t* row_valid, float* s_re, float* s_im, int entries, Stride stride, int num_rows) {
  using namespace radar_jets;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_rows) return;
  if (row_valid[k] == 0) {
    s_re[k] = 0.0f;
    s_im[k] = 0.0f;
    return;
  }
  const Strided at = stride.row(k, idx_in, idx_out, idx_site);
  const C<0> value = table_response::row<0>(
      jload<0>(dir_in, at.i), jload<0>(dir_out, at.o), jload<0>(axis, at.s), table_re, table_im, entries);
  s_re[k] = static_cast<float>(value.re.v);
  s_im[k] = static_cast<float>(value.im.v);
}

__global__ void scatter_response_table_jvp_kernel(
    const float* dir_in, const float* dir_out, const int64_t* idx_in, const int64_t* idx_out,
    const int64_t* idx_site, const float* axis, const float* table_re, const float* table_im,
    const int32_t* row_valid, const float* tan_dir_in, const float* tan_dir_out, const float* tan_axis,
    float* tan_re, float* tan_im, int entries, Stride stride, int num_rows) {
  using namespace radar_jets;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_rows) return;
  if (row_valid[k] == 0) {
    tan_re[k] = 0.0f;
    tan_im[k] = 0.0f;
    return;
  }
  const Strided at = stride.row(k, idx_in, idx_out, idx_site);
  V3<1> din = jload<1>(dir_in, at.i);
  V3<1> dout = jload<1>(dir_out, at.o);
  V3<1> u = jload<1>(axis, at.s);
  jtangent(din, tan_dir_in, at.i);
  jtangent(dout, tan_dir_out, at.o);
  jtangent(u, tan_axis, at.s);
  const C<1> value = table_response::row<1>(din, dout, u, table_re, table_im, entries);
  tan_re[k] = static_cast<float>(value.re.d[0]);
  tan_im[k] = static_cast<float>(value.im.d[0]);
}

// One owner slot per thread over inbound leg rows, outbound leg rows and
// sites, each reduced in the join's frozen CSR order.
__global__ void scatter_response_table_backward_kernel(
    const float* dir_in, const float* dir_out, const int64_t* idx_in, const int64_t* idx_out,
    const int64_t* idx_site, const float* axis, const float* table_re, const float* table_im,
    const int32_t* row_valid, const int64_t* by_in_offsets, const int64_t* by_in_rows,
    const int64_t* by_out_offsets, const int64_t* by_out_rows, const int64_t* by_site_offsets,
    const int64_t* by_site_rows, const float* grad_re, const float* grad_im, float* grad_dir_in,
    float* grad_dir_out, float* grad_axis, int entries, Stride stride, int num_in, int num_out, int num_sites) {
  using namespace radar_jets;
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread >= num_in + num_out + num_sites) return;
  int family;
  int slot;
  const int64_t* offsets;
  const int64_t* members;
  OwnerSlots owned;
  if (thread < num_in) {
    family = 0, slot = thread, offsets = by_in_offsets, members = by_in_rows;
    owned = stride.owner(slot, stride.in_per_slot);
  } else if (thread < num_in + num_out) {
    family = 1, slot = thread - num_in, offsets = by_out_offsets, members = by_out_rows;
    owned = stride.owner(slot, stride.out_per_slot);
  } else {
    family = 2, slot = thread - num_in - num_out, offsets = by_site_offsets, members = by_site_rows;
    owned = stride.sites_per_slot > 0 ? stride.owner(slot, stride.sites_per_slot) : stride.shared(slot);
  }
  double g[3] = {0.0, 0.0, 0.0};
  for (int64_t gslot = owned.first; gslot < owned.last; ++gslot)
  for (int64_t entry = offsets[owned.local]; entry < offsets[owned.local + 1]; ++entry) {
    const int64_t k = gslot * stride.rows_per_slot + members[entry];
    if (row_valid[k] == 0) continue;
    const Strided at = stride.row(k, idx_in, idx_out, idx_site);
    V3<3> din = jload<3>(dir_in, at.i);
    V3<3> dout = jload<3>(dir_out, at.o);
    V3<3> u = jload<3>(axis, at.s);
    if (family == 0) jseed(din, 0);
    if (family == 1) jseed(dout, 0);
    if (family == 2) jseed(u, 0);
    const C<3> value = table_response::row<3>(din, dout, u, table_re, table_im, entries);
    for (int lane = 0; lane < 3; ++lane) {
      g[lane] += static_cast<double>(grad_re[k]) * value.re.d[lane] + static_cast<double>(grad_im[k]) * value.im.d[lane];
    }
  }
  float* out = family == 0 ? grad_dir_in : (family == 1 ? grad_dir_out : grad_axis);
  for (int lane = 0; lane < 3; ++lane) out[static_cast<int64_t>(slot) * 3 + lane] = static_cast<float>(g[lane]);
}

void check_table_inputs(
    const torch::stable::Tensor& dir_in, const torch::stable::Tensor& dir_out, const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out, const torch::stable::Tensor& idx_site, const torch::stable::Tensor& axis,
    const torch::stable::Tensor& table_re, const torch::stable::Tensor& table_im,
    const torch::stable::Tensor& row_valid, int rows) {
  check_cuda_float(dir_in, "dir_in");
  check_cuda_float(dir_out, "dir_out");
  STD_TORCH_CHECK(
      dir_in.numel() % 3 == 0 && dir_out.numel() % 3 == 0, "dir_in and dir_out must hold three values per leg row.");
  const int per_slot = checked_int(idx_in.numel(), "rows_per_slot");
  check_index(idx_in, per_slot, "idx_in");
  check_index(idx_out, per_slot, "idx_out");
  check_index(idx_site, per_slot, "idx_site");
  STD_TORCH_CHECK(per_slot == 0 ? rows == 0 : rows % per_slot == 0, "the rows must be whole slots of the index tables.");
  check_cuda_float(axis, "axis");
  STD_TORCH_CHECK(axis.numel() % 3 == 0, "axis must hold three values per site.");
  check_cuda_float(table_re, "table_re");
  check_cuda_float(table_im, "table_im");
  STD_TORCH_CHECK(
      table_re.numel() >= 2 && table_re.numel() == table_im.numel(),
      "the response table must hold at least two complex entries.");
  check_cuda_int(row_valid, "row_valid");
  STD_TORCH_CHECK(row_valid.numel() == rows, "row_valid must hold one flag per composed row.");
}

void scatter_response_table_forward_cuda(
    const torch::stable::Tensor& dir_in, const torch::stable::Tensor& dir_out, const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out, const torch::stable::Tensor& idx_site, const torch::stable::Tensor& axis,
    const torch::stable::Tensor& table_re, const torch::stable::Tensor& table_im,
    const torch::stable::Tensor& row_valid, torch::stable::Tensor s_re, torch::stable::Tensor s_im, int64_t num_rows, int64_t slots,
    int64_t slot_local_sites) {
  const int rows = checked_int(num_rows, "num_rows");
  check_table_inputs(dir_in, dir_out, idx_in, idx_out, idx_site, axis, table_re, table_im, row_valid, rows);
  check_len(s_re, rows, "s_re");
  check_len(s_im, rows, "s_im");
  if (rows == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(s_re.get_device_index());
  scatter_response_table_forward_kernel<<<linear_grid(rows), dim3(kBlock, 1, 1), 0, current_cuda_stream(s_re)>>>(
      dir_in.const_data_ptr<float>(), dir_out.const_data_ptr<float>(), idx_in.const_data_ptr<int64_t>(),
      idx_out.const_data_ptr<int64_t>(), idx_site.const_data_ptr<int64_t>(), axis.const_data_ptr<float>(),
      table_re.const_data_ptr<float>(), table_im.const_data_ptr<float>(), row_valid.const_data_ptr<int32_t>(),
      s_re.mutable_data_ptr<float>(), s_im.mutable_data_ptr<float>(), static_cast<int>(table_re.numel()),
      make_stride(idx_in, dir_in, dir_out, axis.numel() / 3, slots, slot_local_sites), rows);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_response_table_jvp_cuda(
    const torch::stable::Tensor& dir_in, const torch::stable::Tensor& dir_out, const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out, const torch::stable::Tensor& idx_site, const torch::stable::Tensor& axis,
    const torch::stable::Tensor& table_re, const torch::stable::Tensor& table_im,
    const torch::stable::Tensor& row_valid, const torch::stable::Tensor& tan_dir_in,
    const torch::stable::Tensor& tan_dir_out, const torch::stable::Tensor& tan_axis, torch::stable::Tensor tan_re,
    torch::stable::Tensor tan_im, int64_t num_rows, int64_t slots, int64_t slot_local_sites) {
  const int rows = checked_int(num_rows, "num_rows");
  check_table_inputs(dir_in, dir_out, idx_in, idx_out, idx_site, axis, table_re, table_im, row_valid, rows);
  check_vec3(tan_dir_in, dir_in.numel() / 3, "tan_dir_in");
  check_vec3(tan_dir_out, dir_out.numel() / 3, "tan_dir_out");
  check_vec3(tan_axis, axis.numel() / 3, "tan_axis");
  check_len(tan_re, rows, "tan_re");
  check_len(tan_im, rows, "tan_im");
  if (rows == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(tan_re.get_device_index());
  scatter_response_table_jvp_kernel<<<linear_grid(rows), dim3(kBlock, 1, 1), 0, current_cuda_stream(tan_re)>>>(
      dir_in.const_data_ptr<float>(), dir_out.const_data_ptr<float>(), idx_in.const_data_ptr<int64_t>(),
      idx_out.const_data_ptr<int64_t>(), idx_site.const_data_ptr<int64_t>(), axis.const_data_ptr<float>(),
      table_re.const_data_ptr<float>(), table_im.const_data_ptr<float>(), row_valid.const_data_ptr<int32_t>(),
      tan_dir_in.const_data_ptr<float>(), tan_dir_out.const_data_ptr<float>(), tan_axis.const_data_ptr<float>(),
      tan_re.mutable_data_ptr<float>(), tan_im.mutable_data_ptr<float>(), static_cast<int>(table_re.numel()),
      make_stride(idx_in, dir_in, dir_out, axis.numel() / 3, slots, slot_local_sites), rows);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_response_table_backward_cuda(
    const torch::stable::Tensor& dir_in, const torch::stable::Tensor& dir_out, const torch::stable::Tensor& idx_in,
    const torch::stable::Tensor& idx_out, const torch::stable::Tensor& idx_site, const torch::stable::Tensor& axis,
    const torch::stable::Tensor& table_re, const torch::stable::Tensor& table_im,
    const torch::stable::Tensor& row_valid, const torch::stable::Tensor& by_in_offsets,
    const torch::stable::Tensor& by_in_rows, const torch::stable::Tensor& by_out_offsets,
    const torch::stable::Tensor& by_out_rows, const torch::stable::Tensor& by_site_offsets,
    const torch::stable::Tensor& by_site_rows, const torch::stable::Tensor& grad_re,
    const torch::stable::Tensor& grad_im, torch::stable::Tensor grad_dir_in, torch::stable::Tensor grad_dir_out,
    torch::stable::Tensor grad_axis, int64_t num_rows, int64_t slots, int64_t slot_local_sites) {
  const int rows = checked_int(num_rows, "num_rows");
  check_table_inputs(dir_in, dir_out, idx_in, idx_out, idx_site, axis, table_re, table_im, row_valid, rows);
  const int legs_in = checked_int(dir_in.numel() / 3, "num_in");
  const int legs_out = checked_int(dir_out.numel() / 3, "num_out");
  const int sites = checked_int(axis.numel() / 3, "num_sites");
  check_cuda_long(by_in_offsets, "by_in_offsets");
  check_cuda_long(by_in_rows, "by_in_rows");
  check_cuda_long(by_out_offsets, "by_out_offsets");
  check_cuda_long(by_out_rows, "by_out_rows");
  check_cuda_long(by_site_offsets, "by_site_offsets");
  check_cuda_long(by_site_rows, "by_site_rows");
  const Stride stride = make_stride(idx_in, dir_in, dir_out, sites, slots, slot_local_sites);
  const int64_t site_owners = stride.sites_per_slot > 0 ? stride.sites_per_slot : sites;
  STD_TORCH_CHECK(
      by_in_offsets.numel() == stride.in_per_slot + 1 && by_out_offsets.numel() == stride.out_per_slot + 1 &&
          by_site_offsets.numel() == site_owners + 1,
      "each CSR offsets table must hold one entry per owner of one slot plus one.");
  STD_TORCH_CHECK(
      by_in_rows.numel() == stride.rows_per_slot && by_out_rows.numel() == stride.rows_per_slot &&
          by_site_rows.numel() == stride.rows_per_slot,
      "each CSR row table must permute every composed row of one slot exactly once.");
  check_len(grad_re, rows, "grad_re");
  check_len(grad_im, rows, "grad_im");
  check_vec3(grad_dir_in, legs_in, "grad_dir_in");
  check_vec3(grad_dir_out, legs_out, "grad_dir_out");
  check_vec3(grad_axis, sites, "grad_axis");
  const int total = legs_in + legs_out + sites;
  if (total == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(grad_axis.get_device_index());
  scatter_response_table_backward_kernel<<<linear_grid(total), dim3(kBlock, 1, 1), 0, current_cuda_stream(grad_axis)>>>(
      dir_in.const_data_ptr<float>(), dir_out.const_data_ptr<float>(), idx_in.const_data_ptr<int64_t>(),
      idx_out.const_data_ptr<int64_t>(), idx_site.const_data_ptr<int64_t>(), axis.const_data_ptr<float>(),
      table_re.const_data_ptr<float>(), table_im.const_data_ptr<float>(), row_valid.const_data_ptr<int32_t>(),
      by_in_offsets.const_data_ptr<int64_t>(), by_in_rows.const_data_ptr<int64_t>(),
      by_out_offsets.const_data_ptr<int64_t>(), by_out_rows.const_data_ptr<int64_t>(),
      by_site_offsets.const_data_ptr<int64_t>(), by_site_rows.const_data_ptr<int64_t>(), grad_re.const_data_ptr<float>(),
      grad_im.const_data_ptr<float>(), grad_dir_in.mutable_data_ptr<float>(), grad_dir_out.mutable_data_ptr<float>(),
      grad_axis.mutable_data_ptr<float>(), static_cast<int>(table_re.numel()), stride, legs_in, legs_out, sites);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Swerling fluctuation: one amplitude factor per site.
// ---------------------------------------------------------------------------
//
// out[s] = sqrt(X_s) with E[X] = 1: X ~ Exp(1) (chi-square, 2 degrees of
// freedom; Swerling I and II) when degrees == 2, X ~ Gamma(2, 1/2) (chi-square,
// 4 degrees of freedom; Swerling III and IV) when degrees == 4. The uniforms
// are splitmix64 of (key, site, draw), so the factor is a pure function of the
// key the host derives from the seed and the instant (or the coherent
// interval) and of the site. The factor has no differentiable input.

__device__ __forceinline__ double fluctuation_uniform(uint64_t key, int64_t site, int draw) {
  uint64_t z = key + 0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(site) * 2ULL + static_cast<uint64_t>(draw) + 1ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  // (0, 1]: never zero, so the logarithm is finite.
  return (static_cast<double>(z >> 11) + 1.0) * (1.0 / 9007199254740992.0);
}

__global__ void fluctuation_factor_kernel(float* out, uint64_t key, int degrees, int sites) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= sites) return;
  double power = -log(fluctuation_uniform(key, s, 0));
  if (degrees == 4) power = 0.5 * (power - log(fluctuation_uniform(key, s, 1)));
  out[s] = static_cast<float>(sqrt(power));
}

void fluctuation_factor_forward_cuda(torch::stable::Tensor out, int64_t key, int64_t degrees) {
  check_cuda_float(out, "out");
  STD_TORCH_CHECK(degrees == 2 || degrees == 4, "degrees must be 2 (Swerling I, II) or 4 (Swerling III, IV).");
  STD_TORCH_CHECK(key >= 0, "key must be non-negative.");
  const int sites = checked_int(out.numel(), "sites");
  if (sites == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(out.get_device_index());
  fluctuation_factor_kernel<<<linear_grid(sites), dim3(kBlock, 1, 1), 0, current_cuda_stream(out)>>>(
      out.mutable_data_ptr<float>(), static_cast<uint64_t>(key), static_cast<int>(degrees), sites);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("scatter_direction_forward", TORCH_BOX(&scatter_direction_forward_cuda));
  m.impl("scatter_direction_backward", TORCH_BOX(&scatter_direction_derivative_cuda));
  m.impl("scatter_direction_jvp", TORCH_BOX(&scatter_direction_derivative_cuda));
  m.impl(
      "scatter_response_aspect_forward",
      TORCH_BOX(&scatter_response_aspect_forward_cuda));
  m.impl(
      "scatter_response_aspect_jvp",
      TORCH_BOX(&scatter_response_aspect_jvp_cuda));
  m.impl(
      "scatter_response_aspect_backward",
      TORCH_BOX(&scatter_response_aspect_backward_cuda));
  m.impl("scatter_response_table_forward", TORCH_BOX(&scatter_response_table_forward_cuda));
  m.impl("scatter_response_table_jvp", TORCH_BOX(&scatter_response_table_jvp_cuda));
  m.impl("scatter_response_table_backward", TORCH_BOX(&scatter_response_table_backward_cuda));
  m.impl("fluctuation_factor_forward", TORCH_BOX(&fluctuation_factor_forward_cuda));
}
