"""The native processing selections against the host route and the independent oracles.

``cfar_order_statistic_forward`` and ``dbscan_labels_forward`` are the CUDA
routes of :func:`os_cfar` and :func:`cluster_points`. Each is checked three
ways: against the Torch route on the same input moved to the host (the same
algorithm, so the comparison is exact for the ordered statistic, which is a
selection, and for the labels, which are integers); against the independent
oracle of the CPU tests; and for its launch and synchronization budget.
"""

from __future__ import annotations

import pytest
import torch
from support.reference_cfar import reference_cfar

from witwin.radar.processing import PointCloud, cluster_points, os_cfar

pytestmark = pytest.mark.gpu


def _noise_power(shape, seed, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float64)
    return (real.square() + imaginary.square()).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("guard", "training"), [((1, 2), (2, 3)), ((2, 4), (4, 8)), ((0, 1), (1, 2))])
def test_the_native_order_statistic_equals_the_host_route(dtype, guard, training):
    power = _noise_power((3, 64, 96), seed=11, dtype=dtype)
    power[1, 0, 2] = 500.0
    host = os_cfar(power, guard_cells=guard, training_cells=training, pfa=1e-3)
    device = os_cfar(power.cuda(), guard_cells=guard, training_cells=training, pfa=1e-3)
    assert torch.equal(device.threshold.cpu(), host.threshold)
    assert torch.equal(device.mask.cpu(), host.mask)


def test_the_native_order_statistic_matches_the_oracle():
    power = _noise_power((2, 24, 40), seed=77, dtype=torch.float64)
    device = os_cfar(power.cuda(), guard_cells=(1, 2), training_cells=(2, 3), pfa=1e-3)
    expected = reference_cfar("os", power, guard_cells=(1, 2), training_cells=(2, 3), pfa=1e-3)
    torch.testing.assert_close(device.threshold.cpu(), expected, rtol=1e-9, atol=1e-12)


def test_the_native_order_statistic_holds_its_false_alarm_rate():
    power = _noise_power((32, 256, 256), seed=20261112).cuda()
    for pfa in (1e-2, 1e-3, 1e-4):
        count = int(os_cfar(power, guard_cells=(2, 4), training_cells=(4, 8), pfa=pfa).mask.sum())
        assert abs(count / power.numel() / pfa - 1.0) <= 0.2, (pfa, count / power.numel() / pfa)


def test_a_ring_beyond_the_native_buffer_is_refused_by_name():
    power = torch.ones((1, 64, 64), device="cuda")
    with pytest.raises(ValueError, match="thread-local buffer"):
        os_cfar(power, guard_cells=(1, 1), training_cells=(15, 15))


def _profile(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        fn()
        torch.cuda.synchronize()
    names = [event.name for event in trace.events()]
    return names


def test_the_ordered_statistic_route_is_one_selection_launch_and_no_host_round_trip():
    power = _noise_power((8, 128, 256), seed=3).cuda()
    os_cfar(power)  # warm the per-configuration constant cache
    names = _profile(lambda: os_cfar(power))
    assert sum("cfar_order_statistic_kernel" in name for name in names if "Launch" not in name) == 1
    assert not any("DtoH" in name or "Memcpy DtoH" in name for name in names), [n for n in names if "Memcpy" in n]
    # Every device synchronization in the trace is the profiling harness's own.
    harness = _profile(lambda: None)
    assert names.count("cudaDeviceSynchronize") == harness.count("cudaDeviceSynchronize")
    assert names.count("cudaStreamSynchronize") == 0
    # No [cells, N] patch tensor: the unfold route is not taken on the device.
    assert not any("im2col" in name or "Im2Col" in name for name in names)


def _cloud(xyz: torch.Tensor) -> PointCloud:
    count = int(xyz.shape[0])
    zero = torch.zeros(count, dtype=torch.float64, device=xyz.device)
    return PointCloud(xyz=xyz, velocity_mps=zero, energy=zero, range_m=xyz.norm(dim=1))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_native_dbscan_labels_equal_the_host_route(seed):
    generator = torch.Generator().manual_seed(seed)
    centres = torch.rand((6, 3), generator=generator, dtype=torch.float64) * 20.0
    xyz = centres.repeat_interleave(60, dim=0) + torch.randn((360, 3), generator=generator, dtype=torch.float64)
    xyz = torch.cat((xyz, torch.rand((40, 3), generator=generator, dtype=torch.float64) * 20.0))
    host = cluster_points(_cloud(xyz), eps_m=0.9, min_points=5)
    device = cluster_points(_cloud(xyz.cuda()), eps_m=0.9, min_points=5)
    assert torch.equal(device.labels.cpu(), host.labels)
    torch.testing.assert_close(device.centroid.cpu(), host.centroid)
    assert torch.equal(device.size.cpu(), host.size)


def test_the_native_dbscan_is_four_launches():
    xyz = torch.rand((500, 3), dtype=torch.float64, device="cuda") * 10.0
    cluster_points(_cloud(xyz), eps_m=0.5, min_points=4)
    names = _profile(lambda: cluster_points(_cloud(xyz), eps_m=0.5, min_points=4))
    kernels = [name for name in names if "dbscan_" in name]
    launched = set(kernels)
    for stage in ("dbscan_core_kernel", "dbscan_union_kernel", "dbscan_compress_kernel", "dbscan_border_kernel"):
        assert sum(stage in name for name in kernels) == 1, (stage, launched)
