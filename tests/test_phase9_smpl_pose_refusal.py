"""The SMPL deformation bridge refuses a pose derivative instead of severing it.

``SmplPoseDeformation`` publishes two things into the Core world model: a rest
``Mesh`` that a ``Structure`` carries, and a per-frame ``DeformationState``.
Both cross the Core/Channel COMPILE boundary. ``rest_mesh`` used to build its
mesh from ``vertices.detach()``, so a caller could mark the pose, run a whole
epoch loop, and read ``pose.grad is None`` back with nothing having failed. The
posing function itself is differentiable Torch and ``velocity_at`` depends on
that, so the severance was invisible from inside the module too.

Phase 9 replaces the detach with a refusal and names the deferral: plumbing a
pose derivative into a compiled scene is a Core/Channel design to be accepted,
not a detach to delete, and a half-working pose gradient is exactly the defect
class this phase removes.

**What is NOT refused, and why.** ``SMPLBody`` still publishes differentiable
vertices, and its pose gradient reaches the LEGACY radar
``Scene.compile_renderables()`` mesh - measured at ``|d(sum v^2)/d(pose)|_1 =
4695.12`` through both ``SMPLBody._evaluate`` and the legacy compile. That is a
working capability in a different owner and refusing it would be a removal, so
this module's refusal is scoped to the deformation bridge. Deliverable 5's
mesh-vertex tests are the supported differentiable-geometry route.

``pose_rate`` gets the same ADR-038 treatment as a velocity: it is a tangent
DIRECTION, ``velocity_at`` feeds it to ``make_dual``, and ``d(loss)/d(pose_rate)``
does not exist.
"""

from __future__ import annotations

import pathlib

import pytest
import torch
import torch.autograd.forward_ad as forward_ad


def _smpl_model_root() -> str | None:
    """The checkout's SMPL model directory, or ``None``.

    Same two candidates as ``test_phase7_moving_structures``: a git worktree
    sits one level deeper than the checkout the default path is written
    against, so the models live beside the main checkout.
    """

    import witwin.radar.smpl as smpl_module

    candidates = (
        pathlib.Path(smpl_module._default_smpl_model_root()),
        pathlib.Path(__file__).resolve().parents[3]
        / "radar"
        / "models"
        / "smpl_models",
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.pkl")):
            return str(candidate)
    return None


@pytest.fixture(scope="module")
def smpl():
    pytest.importorskip("smplpytorch")
    root = _smpl_model_root()
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
def test_a_pose_or_shape_derivative_is_refused_at_the_deformation_bridge(
    smpl, marked
):
    """Refused at construction, before a rest mesh or a state can exist."""

    SMPLBody, SmplPoseDeformation, root = smpl
    body = SMPLBody(
        pose=_pose(requires_grad=marked == "pose"),
        shape=_shape(requires_grad=marked == "shape"),
        model_root=root,
        device="cpu",
    )
    with pytest.raises(RuntimeError) as raised:
        SmplPoseDeformation(body, pose_rate=_rate())
    message = str(raised.value)
    assert f"body.{marked}" in message
    assert "requires_grad" in message
    # The deferral is named, and so is the route that DOES work today.
    assert "compile boundary" in message
    assert "witwin.core.Mesh" in message


def test_a_pose_carrying_a_forward_dual_is_refused_too(smpl):
    """A caller-supplied dual is not the internal one ``velocity_at`` makes.

    ``velocity_at`` opens its own level and duals the pose there. A dual that
    arrived from OUTSIDE would nest inside it, which is a second-order forward
    request nothing in either package ships.
    """

    SMPLBody, SmplPoseDeformation, root = smpl
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(_pose(), torch.ones(72))
        body = SMPLBody(
            pose=dual, shape=_shape(), model_root=root, device="cpu"
        )
        with pytest.raises(RuntimeError) as raised:
            SmplPoseDeformation(body, pose_rate=_rate())
    assert "forward tangent" in str(raised.value)


def test_a_pose_rate_that_requires_grad_is_refused_by_adr_038(smpl):
    """A rate is a tangent direction, exactly like a kinematics velocity."""

    SMPLBody, SmplPoseDeformation, root = smpl
    body = SMPLBody(
        pose=_pose(), shape=_shape(), model_root=root, device="cpu"
    )
    with pytest.raises(RuntimeError) as raised:
        SmplPoseDeformation(body, pose_rate=_rate().requires_grad_(True))
    message = str(raised.value)
    assert "pose_rate" in message
    assert "ADR-038" in message
    assert "tangent" in message


def test_a_transform_derivative_is_refused_by_rest_mesh(smpl):
    """The one the constructor cannot see: a live ``position`` on the body.

    ``_transform_mesh_verts`` applies the body's rotation and position to the
    posed vertices, so a marked position reaches the rest mesh without the pose
    or the shape ever being marked. This is the case the detach used to swallow
    and the reason the check is repeated at the mesh boundary.
    """

    SMPLBody, SmplPoseDeformation, root = smpl
    body = SMPLBody(
        pose=_pose(),
        shape=_shape(),
        position=torch.zeros(3).requires_grad_(True),
        model_root=root,
        device="cpu",
    )
    deformation = SmplPoseDeformation(body, pose_rate=_rate())
    with pytest.raises(RuntimeError) as raised:
        deformation.rest_mesh()
    assert "position or rotation" in str(raised.value)


# --------------------------------------------------------------------------
# What still works
# --------------------------------------------------------------------------


def test_the_rest_mesh_of_an_undifferentiated_body_is_unchanged(smpl):
    """Removing the detach changed no value on the supported route.

    The detach is gone, replaced by the refusal above. On a body that carries no
    derivative the mesh is byte for byte what it was, and the vertices carry no
    graph - which is now a fact about the input rather than something the mesh
    imposed.
    """

    SMPLBody, SmplPoseDeformation, root = smpl
    body = SMPLBody(
        pose=_pose(), shape=_shape(), model_root=root, device="cpu"
    )
    deformation = SmplPoseDeformation(body, pose_rate=_rate())
    mesh = deformation.rest_mesh()
    vertices = mesh.vertices
    assert not vertices.requires_grad
    assert vertices.grad_fn is None
    expected, _ = body.to_mesh(device="cpu")
    assert torch.equal(vertices, expected)


def test_the_analytic_vertex_velocity_still_comes_from_the_pose_rate(smpl):
    """The supported route the refusal message points at, executed.

    ``pose_rate`` cannot be a leaf and IS the tangent direction that produces an
    exact vertex velocity through the same linear blend skinning the primal
    uses. Refusing the leaf did not touch it.
    """

    SMPLBody, SmplPoseDeformation, root = smpl
    body = SMPLBody(
        pose=_pose(), shape=_shape(), model_root=root, device="cpu"
    )
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
    # the same normalisation ``test_phase7_moving_structures`` uses.
    step = 1.0e-4
    difference = (
        deformation.vertices_at(step) - deformation.vertices_at(-step)
    ) / (2.0 * step)
    error = float((difference - velocity).norm(dim=1).max()) / fastest
    assert error < 5.0e-3, error


def test_the_smpl_body_itself_still_publishes_a_pose_gradient(smpl):
    """The capability this refusal deliberately does NOT remove.

    ``SMPLBody`` is legacy-``Scene`` geometry as well as the deformation's body,
    and on that route a pose gradient reaches the compiled mesh and is real.
    Refusing it inside ``SMPLBody`` would have been a removal rather than a
    narrowing, so the scope of the Phase-9 refusal is the bridge, and this pins
    the boundary of that scope.
    """

    SMPLBody, _, root = smpl
    pose = _pose(requires_grad=True)
    body = SMPLBody(
        pose=pose, shape=_shape(), model_root=root, device="cpu"
    )
    vertices, _ = body.to_mesh(device="cpu")
    assert vertices.requires_grad
    vertices.square().sum().backward()
    assert pose.grad is not None
    assert float(pose.grad.abs().sum()) > 0.0
