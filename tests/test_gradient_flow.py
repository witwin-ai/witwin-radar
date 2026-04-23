"""Gradient coverage for radar scene -> renderer -> solver."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch


MODEL_ROOT = str(pathlib.Path(__file__).resolve().parents[1] / "models" / "smpl_models")

GRADIENT_CONFIG = {
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 32,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "chirp_per_frame": 1,
    "num_tx": 1,
    "num_rx": 1,
    "idle_time": 7,
    "ramp_end_time": 58,
    "frame_per_second": 10,
    "num_doppler_bins": 1,
    "num_range_bins": 32,
    "num_angle_bins": 1,
    "power": 1,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
}

BACKENDS = ("pytorch", "dirichlet", "slang")
SAMPLINGS = ("pixel", "triangle")
MESH_COMPONENT = (0, 2)


def _smpl_available():
    try:
        from witwin.core.geometry.smpl import SMPLBody  # noqa: F401
        from smplpytorch.pytorch.smpl_layer import SMPL_Layer  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.gpu
needs_smpl = pytest.mark.skipif(not _smpl_available(), reason="smplpytorch not installed")


def _make_sensor(wr):
    return wr.Sensor(origin=(0, 0, 0), target=(0, 0, -5), up=(0, 1, 0), fov=60)


def _make_smpl_scene(pose, shape, *, position=(0.0, -0.1, -3.0)):
    import witwin.radar as wr

    return (
        wr.Scene(sensor=_make_sensor(wr))
        .add_smpl(
            name="human",
            pose=pose,
            shape=shape,
            position=position,
            model_root=MODEL_ROOT,
        )
        .update_structure("human", material=wr.Material(eps_r=3.0))
    )


def _make_mesh_scene(vertices):
    import witwin.radar as wr

    _, faces = wr.Box(position=(0.0, -0.1, -3.0), size=(0.8, 1.6, 0.4)).to_mesh()
    return (
        wr.Scene(sensor=_make_sensor(wr))
        .add_mesh(
            name="target",
            vertices=vertices,
            faces=faces,
            dynamic=True,
            material=wr.Material(eps_r=3.0),
        )
    )


def _make_geometry_scene(size):
    import witwin.radar as wr

    geometry = wr.Box(position=(0.0, -0.1, -3.0), size=size)
    return (
        wr.Scene(sensor=_make_sensor(wr))
        .add_mesh(
            name="target",
            geometry=geometry,
            dynamic=True,
            material=wr.Material(eps_r=3.0),
        )
    )


def _base_mesh_vertices():
    import witwin.radar as wr

    vertices, _ = wr.Box(position=(0.0, -0.1, -3.0), size=(0.8, 1.6, 0.4)).to_mesh()
    return torch.as_tensor(vertices, dtype=torch.float32, device="cuda")


def _base_box_size():
    return torch.tensor([0.8, 1.6, 0.4], dtype=torch.float32, device="cuda")


def _trace_case_params(sampling: str):
    pose = torch.zeros(72, dtype=torch.float32, device="cuda")
    shape = torch.zeros(10, dtype=torch.float32, device="cuda")
    if sampling == "pixel":
        pose_index, shape_index = 17, 5
    elif sampling == "triangle":
        pose_index, shape_index = 17, 8
    else:
        raise ValueError(f"Unsupported sampling '{sampling}'.")
    pose[pose_index] = 0.2
    shape[shape_index] = 0.2
    return pose, shape, pose_index, shape_index


def _default_pose():
    pose = torch.zeros(72, dtype=torch.float32, device="cuda")
    pose[17] = 0.2
    return pose


def _default_shape():
    shape = torch.zeros(10, dtype=torch.float32, device="cuda")
    shape[5] = 0.2
    return shape


def _run_simulation(scene, *, backend: str, sampling: str):
    import witwin.radar as wr

    try:
        return wr.Simulation.mimo(
            scene,
            config=GRADIENT_CONFIG,
            backend=backend,
            resolution=24,
            sampling=sampling,
        )
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        pytest.skip(f"{backend} backend unavailable: {exc}")


def _run_trace(scene, *, sampling: str):
    import witwin.radar as wr

    try:
        return wr.Renderer(scene, resolution=24, sampling=sampling).trace()
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        pytest.skip(f"renderer unavailable: {exc}")


def _signal_loss(signal: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(
        0.2,
        1.1,
        signal.numel(),
        device=signal.device,
        dtype=torch.float64,
    ).reshape(signal.shape)
    real_term = (signal.real.to(torch.float64) * weights).sum()
    imag_term = (signal.imag.to(torch.float64) * weights.flip(-1)).sum()
    return real_term + 0.125 * imag_term


def _trace_loss(trace) -> torch.Tensor:
    return trace.points.to(torch.float64).square().sum() + trace.intensities.to(torch.float64).square().sum()


def _centered_fd_scalar(evaluate, eps: float) -> float:
    return float((evaluate(+eps) - evaluate(-eps)) / (2.0 * eps))


def _assert_grad_close(ad_grad: float, fd_grad: float, *, label: str, rel_tol: float, abs_tol: float):
    scale = max(abs(ad_grad), abs(fd_grad), 1.0)
    error = abs(ad_grad - fd_grad)
    assert error <= abs_tol + rel_tol * scale, (
        f"{label}: AD={ad_grad:.6e}, FD={fd_grad:.6e}, "
        f"abs_error={error:.6e}, rel_error={error / scale:.6e}"
    )


@needs_smpl
def test_renderer_trace_triangle_is_differentiable_by_default():
    import witwin.radar as wr

    pose = _default_pose().clone().requires_grad_(True)
    shape = _default_shape().clone().requires_grad_(True)
    scene = _make_smpl_scene(pose, shape)
    renderer = wr.Renderer(scene, resolution=24, sampling="triangle")

    trace = renderer.trace()
    assert trace.points.grad_fn is not None
    assert trace.intensities.grad_fn is not None
    assert trace._tri_indices is not None

    _trace_loss(trace).backward()
    assert pose.grad is not None and pose.grad.abs().sum() > 0
    assert shape.grad is not None and shape.grad.abs().sum() > 0


@needs_smpl
def test_renderer_trace_pixel_is_differentiable_by_default():
    import witwin.radar as wr

    pose = _default_pose().clone().requires_grad_(True)
    shape = _default_shape().clone().requires_grad_(True)
    scene = _make_smpl_scene(pose, shape)
    renderer = wr.Renderer(scene, resolution=24, sampling="pixel")

    trace = renderer.trace()
    assert trace.points.grad_fn is not None
    assert trace.intensities.grad_fn is not None

    _trace_loss(trace).backward()
    assert pose.grad is not None and pose.grad.abs().sum() > 0
    assert shape.grad is not None and shape.grad.abs().sum() > 0


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_trace_mesh_gradients_match_fd_for_pixel_and_triangle(sampling):
    import witwin.radar as wr

    vertices = _base_mesh_vertices().clone().requires_grad_(True)
    trace = wr.Renderer(_make_mesh_scene(vertices), resolution=24, sampling=sampling).trace()
    _trace_loss(trace).backward()

    ad_grad = float(vertices.grad[MESH_COMPONENT].item())
    base = _base_mesh_vertices()

    def evaluate(delta: float) -> float:
        perturbed = base.clone()
        perturbed[MESH_COMPONENT] += delta
        trace_local = wr.Renderer(_make_mesh_scene(perturbed), resolution=24, sampling=sampling).trace()
        return float(_trace_loss(trace_local).item())

    fd_grad = _centered_fd_scalar(evaluate, eps=1e-3)
    _assert_grad_close(
        ad_grad,
        fd_grad,
        label=f"trace mesh gradient ({sampling})",
        rel_tol=0.05,
        abs_tol=5e-3,
    )


def test_trace_parametric_box_size_gradients_exist_for_triangle():
    size = _base_box_size().clone().requires_grad_(True)
    trace = _run_trace(_make_geometry_scene(size), sampling="triangle")
    _trace_loss(trace).backward()
    assert size.grad is not None and size.grad.abs().sum() > 0


def test_trace_parametric_box_size_gradient_matches_fd_for_triangle():
    size = _base_box_size().clone().requires_grad_(True)
    trace = _run_trace(_make_geometry_scene(size), sampling="triangle")
    _trace_loss(trace).backward()
    ad_grad = float(size.grad[0].item())

    base = _base_box_size()

    def evaluate(delta: float) -> float:
        perturbed = base.clone()
        perturbed[0] += delta
        trace_local = _run_trace(_make_geometry_scene(perturbed), sampling="triangle")
        return float(_trace_loss(trace_local).item())

    fd_grad = _centered_fd_scalar(evaluate, eps=1e-3)
    _assert_grad_close(
        ad_grad,
        fd_grad,
        label="parametric box size trace gradient (triangle)",
        rel_tol=0.08,
        abs_tol=8e-3,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_signal_mesh_vertex_gradients_exist_for_all_render_solver_pairs(backend, sampling):
    vertices = _base_mesh_vertices().clone().requires_grad_(True)
    result = _run_simulation(_make_mesh_scene(vertices), backend=backend, sampling=sampling)
    _signal_loss(result.signal()).backward()
    assert vertices.grad is not None and vertices.grad.abs().sum() > 0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_signal_parametric_box_size_gradients_exist_for_all_render_solver_pairs(backend, sampling):
    size = _base_box_size().clone().requires_grad_(True)
    result = _run_simulation(_make_geometry_scene(size), backend=backend, sampling=sampling)
    _signal_loss(result.signal()).backward()
    assert size.grad is not None and size.grad.abs().sum() > 0


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_signal_parametric_box_size_gradient_matches_fd_for_dirichlet(sampling):
    size = _base_box_size().clone().requires_grad_(True)
    result = _run_simulation(_make_geometry_scene(size), backend="dirichlet", sampling=sampling)
    _signal_loss(result.signal()).backward()
    ad_grad = float(size.grad[0].item())

    base = _base_box_size()

    def evaluate(delta: float) -> float:
        perturbed = base.clone()
        perturbed[0] += delta
        result_local = _run_simulation(
            _make_geometry_scene(perturbed),
            backend="dirichlet",
            sampling=sampling,
        )
        return float(_signal_loss(result_local.signal()).item())

    fd_grad = _centered_fd_scalar(evaluate, eps=2e-3)
    _assert_grad_close(
        ad_grad,
        fd_grad,
        label=f"parametric box size signal gradient ({sampling}, dirichlet)",
        rel_tol=0.3,
        abs_tol=2e-2,
    )


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_signal_mesh_vertex_gradient_matches_fd_for_dirichlet(sampling):
    vertices = _base_mesh_vertices().clone().requires_grad_(True)
    result = _run_simulation(_make_mesh_scene(vertices), backend="dirichlet", sampling=sampling)
    _signal_loss(result.signal()).backward()
    ad_grad = float(vertices.grad[MESH_COMPONENT].item())

    base = _base_mesh_vertices()

    def evaluate(delta: float) -> float:
        perturbed = base.clone()
        perturbed[MESH_COMPONENT] += delta
        result_local = _run_simulation(_make_mesh_scene(perturbed), backend="dirichlet", sampling=sampling)
        return float(_signal_loss(result_local.signal()).item())

    fd_grad = _centered_fd_scalar(evaluate, eps=2e-3)
    _assert_grad_close(
        ad_grad,
        fd_grad,
        label=f"mesh vertex signal gradient ({sampling}, dirichlet)",
        rel_tol=0.25,
        abs_tol=2e-2,
    )


@needs_smpl
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_signal_smpl_gradients_exist_for_all_render_solver_pairs(backend, sampling):
    base_pose, base_shape, _, _ = _trace_case_params(sampling)
    pose = base_pose.clone().requires_grad_(True)
    shape = base_shape.clone().requires_grad_(True)
    result = _run_simulation(_make_smpl_scene(pose, shape), backend=backend, sampling=sampling)
    _signal_loss(result.signal()).backward()
    assert pose.grad is not None and pose.grad.abs().sum() > 0
    assert shape.grad is not None and shape.grad.abs().sum() > 0


@needs_smpl
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_signal_smpl_gradients_match_fd_for_dirichlet(sampling):
    base_pose, base_shape, pose_index, shape_index = _trace_case_params(sampling)
    pose = base_pose.clone().requires_grad_(True)
    shape = base_shape.clone().requires_grad_(True)
    result = _run_simulation(_make_smpl_scene(pose, shape), backend="dirichlet", sampling=sampling)
    _signal_loss(result.signal()).backward()

    ad_pose = float(pose.grad[pose_index].item())
    ad_shape = float(shape.grad[shape_index].item())

    def pose_eval(delta: float) -> float:
        perturbed_pose = base_pose.clone()
        perturbed_pose[pose_index] += delta
        result_local = _run_simulation(
            _make_smpl_scene(perturbed_pose, base_shape.clone()),
            backend="dirichlet",
            sampling=sampling,
        )
        return float(_signal_loss(result_local.signal()).item())

    def shape_eval(delta: float) -> float:
        perturbed_shape = base_shape.clone()
        perturbed_shape[shape_index] += delta
        result_local = _run_simulation(
            _make_smpl_scene(base_pose.clone(), perturbed_shape),
            backend="dirichlet",
            sampling=sampling,
        )
        return float(_signal_loss(result_local.signal()).item())

    fd_pose = _centered_fd_scalar(pose_eval, eps=1e-3)
    fd_shape = _centered_fd_scalar(shape_eval, eps=1e-3)

    _assert_grad_close(
        ad_pose,
        fd_pose,
        label=f"SMPL pose signal gradient ({sampling}, dirichlet)",
        rel_tol=0.3,
        abs_tol=2e-2,
    )
    _assert_grad_close(
        ad_shape,
        fd_shape,
        label=f"SMPL shape signal gradient ({sampling}, dirichlet)",
        rel_tol=0.25,
        abs_tol=2e-2,
    )


@needs_smpl
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_simulation_accepts_scene_module_and_backpropagates(sampling):
    import witwin.radar as wr

    class TrainableHumanScene(wr.SceneModule):
        def __init__(self):
            super().__init__()
            self.pose = torch.nn.Parameter(_default_pose().clone())
            self.shape = torch.nn.Parameter(_default_shape().clone())

        def to_scene(self):
            return _make_smpl_scene(self.pose, self.shape)

    scene_module = TrainableHumanScene()
    result = _run_simulation(scene_module, backend="pytorch", sampling=sampling)

    _signal_loss(result.signal()).backward()
    assert scene_module.pose.grad is not None and scene_module.pose.grad.abs().sum() > 0
    assert scene_module.shape.grad is not None and scene_module.shape.grad.abs().sum() > 0
