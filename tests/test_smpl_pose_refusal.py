"""The SMPL deformation bridge: pose and shape are reverse-mode leaves, tangents are refused.

``SmplPoseDeformation`` publishes a rest ``Mesh`` that a ``Structure`` carries
and a per-frame ``DeformationState``. Both hold the posed vertices as a
differentiable expression of the pose, the shape and the body transform, so a
pose or shape gradient reaches every facet and every leg (R-ADR-052). The
end-to-end finite-difference evidence runs on a synthetic rig through the same
code path in ``test_smpl_pose_leaves.py``; this file needs the real model
pickles and skips without them.

What is refused: a forward TANGENT on the pose, the shape or the body transform
(``velocity_at`` opens its own forward level over the pose, and PyTorch's
forward AD cannot nest a caller's level around it), and ``pose_rate`` as a leaf, which
under ADR-038 is a tangent direction rather than an input.
"""

from __future__ import annotations

import pathlib

import pytest
import torch
import torch.autograd.forward_ad as forward_ad
from support.smpl_models import smpl_model_root


@pytest.fixture(scope="module")
def smpl():
    pytest.importorskip("smplpytorch")
    root = smpl_model_root()
    if root is None:
        pytest.skip("no SMPL model files available in this checkout")
    from witwin.radar.smpl import SMPLBody, SmplPoseDeformation

    return SMPLBody, SmplPoseDeformation, root


def _pose(*, requires_grad: bool = False) -> torch.Tensor:
    return torch.zeros(72).requires_grad_(requires_grad)


def _shape(*, requires_grad: bool = False) -> torch.Tensor:
    return torch.zeros(10).requires_grad_(requires_grad)


def _rate() -> torch.Tensor:
    rate = torch.zeros(72)
    rate[3 * 18 + 2] = 3.0
    return rate


# --------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("marked", ["pose", "shape"])
def test_a_pose_or_shape_gradient_reaches_every_published_state(smpl, marked):
    """The bridge carries the leaf into the rest mesh and every deformation state."""

    SMPLBody, SmplPoseDeformation, root = smpl
    pose, shape = _pose(requires_grad=marked == "pose"), _shape(requires_grad=marked == "shape")
    body = SMPLBody(pose=pose, shape=shape, model_root=root, device="cpu")
    deformation = SmplPoseDeformation(body, pose_rate=_rate())
    assert deformation.rest_mesh().vertices.requires_grad
    state = deformation.at(0.01)
    state.vertices.square().sum().backward()
    leaf = pose if marked == "pose" else shape
    assert leaf.grad is not None and float(leaf.grad.abs().sum()) > 0.0


def test_a_pose_carrying_a_forward_dual_is_refused_too(smpl):
    """A caller-supplied dual is not the internal one ``velocity_at`` makes.

    ``velocity_at`` opens its own level and duals the pose there. A dual that
    arrived from OUTSIDE would nest inside it, which is a second-order forward
    request nothing in either package ships.
    """

    SMPLBody, SmplPoseDeformation, root = smpl
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(_pose(), torch.ones(72))
        body = SMPLBody(pose=dual, shape=_shape(), model_root=root, device="cpu")
        with pytest.raises(RuntimeError) as raised:
            SmplPoseDeformation(body, pose_rate=_rate())
    assert "forward tangent" in str(raised.value)
    assert "ad_mode='vjp'" in str(raised.value)


def test_a_pose_rate_that_requires_grad_is_refused_by_adr_038(smpl):
    """A rate is a tangent direction, exactly like a kinematics velocity."""

    SMPLBody, SmplPoseDeformation, root = smpl
    body = SMPLBody(pose=_pose(), shape=_shape(), model_root=root, device="cpu")
    with pytest.raises(RuntimeError) as raised:
        SmplPoseDeformation(body, pose_rate=_rate().requires_grad_(True))
    message = str(raised.value)
    assert "pose_rate" in message
    assert "ADR-038" in message
    assert "tangent" in message


def test_a_transform_tangent_is_refused_by_rest_mesh(smpl):
    """The one the constructor cannot see: a forward dual on the body's ``position``.

    ``_transform_mesh_verts`` applies the body's rotation and position to the
    posed vertices, so a dual position reaches the rest mesh without the pose
    or the shape ever carrying one. A gradient there is carried; a tangent is
    refused for the same nesting reason as the pose.
    """

    SMPLBody, SmplPoseDeformation, root = smpl
    with forward_ad.dual_level():
        position = forward_ad.make_dual(torch.zeros(3), torch.ones(3))
        body = SMPLBody(pose=_pose(), shape=_shape(), position=position, model_root=root, device="cpu")
        deformation = SmplPoseDeformation(body, pose_rate=_rate())
        with pytest.raises(RuntimeError) as raised:
            deformation.rest_mesh()
    assert "position or rotation" in str(raised.value)


# --------------------------------------------------------------------------
# What still works
# --------------------------------------------------------------------------


def test_the_rest_mesh_of_an_undifferentiated_body_is_unchanged(smpl):
    """On a body that carries no derivative the mesh is byte for byte the body's, with no graph."""

    SMPLBody, SmplPoseDeformation, root = smpl
    body = SMPLBody(pose=_pose(), shape=_shape(), model_root=root, device="cpu")
    deformation = SmplPoseDeformation(body, pose_rate=_rate())
    mesh = deformation.rest_mesh()
    vertices = mesh.vertices
    assert not vertices.requires_grad
    assert vertices.grad_fn is None
    expected, _ = body.to_mesh(device="cpu")
    assert torch.equal(vertices, expected)


def test_the_analytic_vertex_velocity_still_comes_from_the_pose_rate(smpl):
    """``pose_rate`` cannot be a leaf and IS the tangent direction of the exact vertex velocity."""

    SMPLBody, SmplPoseDeformation, root = smpl
    body = SMPLBody(pose=_pose(), shape=_shape(), model_root=root, device="cpu")
    deformation = SmplPoseDeformation(body, pose_rate=_rate())
    velocity = deformation.velocity_at(0.0)
    assert velocity.shape[1] == 3
    assert torch.isfinite(velocity).all()
    fastest = float(velocity.norm(dim=1).max())
    assert fastest > 0.2, fastest

    # Independent oracle: a central difference of the posing function itself,
    # scaled by the fastest vertex rather than compared per component. At
    # h = 1e-4 s the float32 cancellation floor on metre-scale coordinates is
    # about 6e-4 m/s, which is the size of the SLOWEST vertices' whole velocity;
    # a per-component relative test there would be measuring the floor. This is
    # the same normalisation ``test_moving_structures`` uses.
    step = 1.0e-4
    difference = (deformation.vertices_at(step) - deformation.vertices_at(-step)) / (2.0 * step)
    error = float((difference - velocity).norm(dim=1).max()) / fastest
    assert error < 5.0e-3, error


def test_the_default_model_root_is_the_checkout_models_directory():
    """``models/smpl_models`` beside ``witwin/``.

    The SMPL pickles are licensed separately and are not in the repository, so
    the location is asserted everywhere and its presence only where a
    developer installed them.
    """

    import witwin.radar.smpl as smpl_module

    root = pathlib.Path(smpl_module._default_smpl_model_root())
    assert root == pathlib.Path(smpl_module.__file__).resolve().parents[2] / "models" / "smpl_models"
    if not root.is_dir():
        pytest.skip(f"no SMPL model directory in this checkout: {root}")


def test_the_fast_lbs_matches_smplpytorch(smpl):
    """The batched skinning is the same function ``SMPL_Layer.forward`` runs."""

    from smplpytorch.pytorch.smpl_layer import SMPL_Layer

    from witwin.radar.smpl import _fast_smpl_forward, _get_smpl_layer

    _, _, root = smpl
    generator = torch.Generator().manual_seed(3)
    pose = 0.4 * torch.randn(72, generator=generator)
    shape = torch.randn(10, generator=generator)
    layer = _get_smpl_layer(gender="male", model_root=root, device="cpu")
    assert isinstance(layer, SMPL_Layer)
    reference, _ = layer(pose.unsqueeze(0), shape.unsqueeze(0))
    torch.testing.assert_close(_fast_smpl_forward(layer, pose, shape), reference[0], rtol=1e-4, atol=1e-5)


def test_the_smpl_body_itself_still_publishes_a_pose_gradient(smpl):
    """``SMPLBody``'s own vertices carry the pose gradient the bridge forwards."""

    SMPLBody, _, root = smpl
    pose = _pose(requires_grad=True)
    body = SMPLBody(pose=pose, shape=_shape(), model_root=root, device="cpu")
    vertices, _ = body.to_mesh(device="cpu")
    assert vertices.requires_grad
    vertices.square().sum().backward()
    assert pose.grad is not None
    assert float(pose.grad.abs().sum()) > 0.0


def test_mesh_faces_are_owned_by_the_result_and_shape_updates_stay_live(smpl):
    SMPLBody, _, root = smpl
    shape = _shape()
    body = SMPLBody(pose=_pose(), shape=shape, model_root=root, device="cpu")
    vertices, faces = body.to_mesh()
    expected_faces = faces.clone()
    faces.zero_()
    # The shared layer owns model constants, never a caller's mutable result.
    shape[0] = 1.0
    updated, topology = body.to_mesh()
    assert topology.dtype == torch.int64
    assert torch.equal(topology, expected_faces)
    assert not torch.equal(vertices, updated)
