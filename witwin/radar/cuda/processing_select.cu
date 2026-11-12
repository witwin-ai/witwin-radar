// Processing selections that are not a few library calls on a CUDA map.
//
// Two families, both post-synthesis detection stages and both non-
// differentiable by contract (the Python facades refuse a derivative before
// reaching here, so there is no backward or JVP companion to register):
//
//  cfar_order_statistic  the ordered-statistic CFAR noise level. One thread
//                        per map cell gathers its N training cells from the
//                        padded map (Doppler wrapped, range padded with +inf
//                        by the caller) into a thread-local buffer and selects
//                        the rank[r]-th smallest in place (Hoare quickselect).
//                        No [cells, N] patch tensor is ever formed, which is
//                        the whole point: the Torch route materialises and
//                        sorts one. Same statistic of the same cells.
//
//  dbscan_labels         DBSCAN over N points with F float64 features. Four
//                        launches: neighbour counts and core flags; a
//                        lock-free union-find over core-core neighbour pairs
//                        that always hooks the larger root under the smaller,
//                        so a component's root is its smallest point index;
//                        full path compression; and the border pass, where a
//                        non-core point takes the smallest root among its core
//                        neighbours or -1. The labels are therefore exactly
//                        the Torch route's (min-label propagation), which is
//                        the oracle the GPU tests compare against.
//
// Distances are squared Euclidean in double, compared against eps^2 with <=.

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

// The largest training ring the in-thread selection holds. A ring this large
// is a 33 x 33 window; the facade refuses a larger one by name.
constexpr int kMaxTraining = 1024;

template <typename T>
__device__ T select_kth(T* values, int count, int k) {
  int low = 0;
  int high = count - 1;
  while (low < high) {
    const T pivot = values[low + ((high - low) >> 1)];
    int i = low;
    int j = high;
    while (i <= j) {
      while (values[i] < pivot) ++i;
      while (values[j] > pivot) --j;
      if (i <= j) {
        const T swap = values[i];
        values[i] = values[j];
        values[j] = swap;
        ++i;
        --j;
      }
    }
    if (k <= j) {
      high = j;
    } else if (k >= i) {
      low = i;
    } else {
      return values[k];
    }
  }
  return values[k];
}

template <typename T>
__global__ void cfar_order_statistic_kernel(
    const T* __restrict__ padded,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ rank,
    T* __restrict__ level,
    int batch,
    int doppler,
    int ranges,
    int padded_doppler,
    int padded_ranges,
    int window_width,
    int count) {
  const int64_t cell = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(batch) * doppler * ranges;
  if (cell >= total) return;
  const int r = static_cast<int>(cell % ranges);
  const int d = static_cast<int>((cell / ranges) % doppler);
  const int b = static_cast<int>(cell / (static_cast<int64_t>(ranges) * doppler));
  // The window's top-left corner in the padded map is (d, r).
  const T* corner = padded + (static_cast<int64_t>(b) * padded_doppler + d) * padded_ranges + r;
  T buffer[kMaxTraining];
  for (int n = 0; n < count; ++n) {
    const int offset = offsets[n];
    buffer[n] = corner[static_cast<int64_t>(offset / window_width) * padded_ranges + offset % window_width];
  }
  level[cell] = select_kth(buffer, count, rank[r]);
}

void cfar_order_statistic_forward_cuda(
    const torch::stable::Tensor& padded,
    const torch::stable::Tensor& offsets,
    const torch::stable::Tensor& rank,
    torch::stable::Tensor level,
    int64_t window_height,
    int64_t window_width) {
  STD_TORCH_CHECK(padded.is_cuda() && padded.is_contiguous(), "padded must be a contiguous CUDA tensor.");
  STD_TORCH_CHECK(padded.dim() == 3, "padded must be [batch, padded_doppler, padded_range].");
  const auto dtype = padded.scalar_type();
  STD_TORCH_CHECK(
      dtype == torch::headeronly::ScalarType::Float || dtype == torch::headeronly::ScalarType::Double,
      "padded must be float32 or float64.");
  STD_TORCH_CHECK(level.scalar_type() == dtype, "level must share the padded map's dtype.");
  STD_TORCH_CHECK(level.is_cuda() && level.is_contiguous() && level.dim() == 3, "level must be [batch, doppler, range].");
  check_cuda_int(offsets, "offsets");
  check_cuda_int(rank, "rank");
  const int count = checked_int(offsets.numel(), "training cells");
  STD_TORCH_CHECK(count >= 1 && count <= kMaxTraining, "the training ring must hold 1 to 1024 cells.");
  const int batch = checked_int(padded.size(0), "batch");
  const int padded_doppler = checked_int(padded.size(1), "padded doppler");
  const int padded_ranges = checked_int(padded.size(2), "padded range");
  const int height = checked_int(window_height, "window height");
  const int width = checked_int(window_width, "window width");
  STD_TORCH_CHECK(height >= 1 && width >= 1 && height <= padded_doppler && width <= padded_ranges,
      "the window must fit in the padded map.");
  const int doppler = padded_doppler - height + 1;
  const int ranges = padded_ranges - width + 1;
  STD_TORCH_CHECK(level.size(0) == batch && level.size(1) == doppler && level.size(2) == ranges,
      "level must be [batch, padded_doppler - window_height + 1, padded_range - window_width + 1].");
  STD_TORCH_CHECK(rank.numel() == ranges, "rank must hold one index per range column.");
  STD_TORCH_CHECK(padded.get_device_index() == level.get_device_index() &&
      padded.get_device_index() == offsets.get_device_index() && padded.get_device_index() == rank.get_device_index(),
      "order-statistic tensors must share a CUDA device.");
  const int64_t total = static_cast<int64_t>(batch) * doppler * ranges;
  if (total == 0) return;
  STD_TORCH_CHECK(total <= static_cast<int64_t>(kBlock) * 2147483647LL, "the map is too large for one launch.");
  const torch::stable::accelerator::DeviceGuard guard(padded.get_device_index());
  const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock), 1, 1);
  if (dtype == torch::headeronly::ScalarType::Float) {
    cfar_order_statistic_kernel<float><<<grid, dim3(kBlock, 1, 1), 0, current_cuda_stream(level)>>>(
        padded.const_data_ptr<float>(), offsets.const_data_ptr<int32_t>(), rank.const_data_ptr<int32_t>(),
        level.mutable_data_ptr<float>(), batch, doppler, ranges, padded_doppler, padded_ranges, width, count);
  } else {
    cfar_order_statistic_kernel<double><<<grid, dim3(kBlock, 1, 1), 0, current_cuda_stream(level)>>>(
        padded.const_data_ptr<double>(), offsets.const_data_ptr<int32_t>(), rank.const_data_ptr<int32_t>(),
        level.mutable_data_ptr<double>(), batch, doppler, ranges, padded_doppler, padded_ranges, width, count);
  }
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ bool within(const double* features, int i, int j, int dims, double eps_sq) {
  double total = 0.0;
  for (int f = 0; f < dims; ++f) {
    const double delta = features[static_cast<int64_t>(i) * dims + f] - features[static_cast<int64_t>(j) * dims + f];
    total += delta * delta;
  }
  return total <= eps_sq;
}

__global__ void dbscan_core_kernel(
    const double* __restrict__ features, int32_t* parent, int32_t* core, int count, int dims, double eps_sq,
    int min_points) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int neighbours = 0;
  for (int j = 0; j < count; ++j) neighbours += within(features, i, j, dims, eps_sq) ? 1 : 0;
  core[i] = neighbours >= min_points ? 1 : 0;
  parent[i] = i;
}

// Path halving. Only a non-root's pointer is rewritten, and only to one of its
// ancestors; roots change solely through the CAS in unite().
__device__ int find_root(volatile int32_t* parent, int x) {
  int next = parent[x];
  while (next != x) {
    const int grand = parent[next];
    if (grand != next) parent[x] = grand;
    x = next;
    next = parent[x];
  }
  return x;
}

__device__ void unite(int32_t* parent, int a, int b) {
  volatile int32_t* view = parent;
  while (true) {
    a = find_root(view, a);
    b = find_root(view, b);
    if (a == b) return;
    if (a > b) {
      const int swap = a;
      a = b;
      b = swap;
    }
    // Hook the larger root under the smaller one, so a root is always the
    // smallest index of its component.
    const int previous = atomicCAS(parent + b, b, a);
    if (previous == b) return;
  }
}

__global__ void dbscan_union_kernel(
    const double* __restrict__ features, int32_t* parent, const int32_t* __restrict__ core, int count, int dims,
    double eps_sq) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || !core[i]) return;
  for (int j = i + 1; j < count; ++j) {
    if (core[j] && within(features, i, j, dims, eps_sq)) unite(parent, i, j);
  }
}

__global__ void dbscan_compress_kernel(int32_t* parent, const int32_t* __restrict__ core, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || !core[i]) return;
  parent[i] = find_root(parent, i);
}

__global__ void dbscan_border_kernel(
    const double* __restrict__ features, int32_t* parent, const int32_t* __restrict__ core, int count, int dims,
    double eps_sq) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || core[i]) return;
  int label = -1;
  for (int j = 0; j < count; ++j) {
    if (core[j] && within(features, i, j, dims, eps_sq)) {
      const int root = parent[j];
      if (label < 0 || root < label) label = root;
    }
  }
  parent[i] = label;
}

void dbscan_labels_forward_cuda(
    const torch::stable::Tensor& features,
    torch::stable::Tensor parent,
    torch::stable::Tensor core,
    double eps,
    int64_t min_points) {
  STD_TORCH_CHECK(features.is_cuda() && features.is_contiguous() && features.dim() == 2,
      "features must be a contiguous [N, F] CUDA tensor.");
  STD_TORCH_CHECK(features.scalar_type() == torch::headeronly::ScalarType::Double, "features must be float64.");
  check_cuda_int(parent, "parent");
  check_cuda_int(core, "core");
  const int count = checked_int(features.size(0), "points");
  const int dims = checked_int(features.size(1), "features");
  STD_TORCH_CHECK(parent.numel() == count && core.numel() == count, "parent and core must hold one entry per point.");
  STD_TORCH_CHECK(features.get_device_index() == parent.get_device_index() &&
      features.get_device_index() == core.get_device_index(), "dbscan tensors must share a CUDA device.");
  STD_TORCH_CHECK(eps > 0.0, "eps must be positive.");
  STD_TORCH_CHECK(min_points >= 1, "min_points must be positive.");
  if (count == 0) return;
  const double eps_sq = eps * eps;
  const int minimum = checked_int(min_points, "min_points");
  const torch::stable::accelerator::DeviceGuard guard(features.get_device_index());
  const cudaStream_t stream = current_cuda_stream(parent);
  const double* data = features.const_data_ptr<double>();
  int32_t* links = parent.mutable_data_ptr<int32_t>();
  int32_t* flags = core.mutable_data_ptr<int32_t>();
  dbscan_core_kernel<<<linear_grid(count), dim3(kBlock, 1, 1), 0, stream>>>(data, links, flags, count, dims, eps_sq, minimum);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  dbscan_union_kernel<<<linear_grid(count), dim3(kBlock, 1, 1), 0, stream>>>(data, links, flags, count, dims, eps_sq);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  dbscan_compress_kernel<<<linear_grid(count), dim3(kBlock, 1, 1), 0, stream>>>(links, flags, count);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
  dbscan_border_kernel<<<linear_grid(count), dim3(kBlock, 1, 1), 0, stream>>>(data, links, flags, count, dims, eps_sq);
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

STABLE_TORCH_LIBRARY_IMPL(_radar_native, CUDA, m) {
  m.impl("cfar_order_statistic_forward", TORCH_BOX(&cfar_order_statistic_forward_cuda));
  m.impl("dbscan_labels_forward", TORCH_BOX(&dbscan_labels_forward_cuda));
}
