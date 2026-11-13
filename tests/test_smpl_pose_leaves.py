"""SMPL pose, shape and body transform as leaves, end to end (R-ADR-052).

The derivative path is the production one: ``SMPLBody`` poses the template
with ``_fast_smpl_forward`` (linear blend skinning), ``SmplPoseDeformation``
publishes the posed vertices as each ``DeformationState``, ``mesh_target``
makes the body's faces the scatter sites, the native facet geometry and facet
response differentiate with respect to the vertices, and Channel's
fixed-topology replay carries the site positions. No SMPL model pickles exist
on this machine, so the rig is the synthetic 24-joint linear-blend-skinning
layer of ``support/synthetic_smpl.py``, installed in the layer cache and run
through exactly that path; the real-model checks in
``test_smpl_pose_refusal.py`` skip without the pickles.

Oracle: central differences of ``Radar.trace`` along a random direction. The
facet response is float32 and a facet's lit/visible state is a discrete gate,
so the steps are small and the tolerance is the vertex-gradient test's 3 %.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as fwad
from support import oracle_scenes as sc
from support import synthetic_smpl

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def rig():
    pytest.importorskip("witwin.channel")
    return synthetic_smpl.install("cuda")


def _body(rig, pose, shape, position=(3.0, 0.0, -0.1)):
    from witwin.radar.smpl import SMPLBody

    gender, root = rig
    return SMPLBody(pose=pose, shape=shape, position=position, gender=gender, model_root=root, device="cuda")


def _loss(rig, pose, shape, position=(3.0, 0.0, -0.1), ad_mode="vjp") -> torch.Tensor:
    from witwin.core import PhysicalMaterial, Scene

    from witwin.radar.smpl import SmplPoseDeformation

    deformation = SmplPoseDeformation(_body(rig, pose, shape, position), pose_rate=torch.zeros(72))
    scene, targets = deformation.mesh_target(
        Scene(structures=(), endpoints=[]), structure_id=7, material=PhysicalMaterial.perfect_conductor()
    )
    rows = sc.radar().trace(scene, targets, times=(0.0,), reflections=0, ad_mode=ad_mode).rows()
    assert rows.path_count > 0
    total = rows.complex_transfer_ref.sum()
    return (total.real * 0.6 + total.imag * 1.1) * 1e6


def _parameters():
    generator = torch.Generator().manual_seed(4)
    pose = 0.05 * torch.randn(72, generator=generator)
    shape = 0.5 * torch.randn(10, generator=generator)
    return pose, shape, torch.randn(72, generator=generator), torch.randn(10, generator=generator)


def test_the_pose_gradient_matches_central_differences(rig):
    pose, shape, direction, _ = _parameters()
    leaf = pose.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(_loss(rig, leaf, shape), leaf)
    step = 5e-4
    with torch.no_grad():
        central = float(_loss(rig, pose + step * direction, shape)) - float(_loss(rig, pose - step * direction, shape))
    assert float((gradient * direction).sum()) == pytest.approx(central / (2.0 * step), rel=3e-2)


def test_the_shape_gradient_matches_central_differences(rig):
    pose, shape, _, direction = _parameters()
    leaf = shape.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(_loss(rig, pose, leaf), leaf)
    step = 3e-3
    with torch.no_grad():
        central = float(_loss(rig, pose, shape + step * direction)) - float(_loss(rig, pose, shape - step * direction))
    assert float((gradient * direction).sum()) == pytest.approx(central / (2.0 * step), rel=3e-2)


def test_the_body_position_is_a_leaf_too(rig):
    """The body transform is applied to the posed vertices, so its gradient is carried the same way."""

    pose, shape, _, _ = _parameters()
    base = torch.tensor([3.0, 0.0, -0.1], device="cuda")
    direction = torch.tensor([0.2, 0.5, -0.3], device="cuda")
    leaf = base.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(_loss(rig, pose, shape, position=leaf), leaf)
    step = 3e-5
    with torch.no_grad():
        plus = float(_loss(rig, pose, shape, position=base + step * direction))
        minus = float(_loss(rig, pose, shape, position=base - step * direction))
    assert float((gradient * direction).sum()) == pytest.approx((plus - minus) / (2.0 * step), rel=3e-2)


def test_a_pose_tangent_is_refused_before_any_compile(rig):
    """A caller's forward level cannot nest around velocity_at's own; refused at the bridge."""

    from witwin.radar.smpl import SmplPoseDeformation

    pose, shape, direction, _ = _parameters()
    with fwad.dual_level():
        body = _body(rig, fwad.make_dual(pose, direction), shape)
        with pytest.raises(RuntimeError, match="forward tangent"):
            SmplPoseDeformation(body, pose_rate=torch.zeros(72))


def test_a_pose_rate_leaf_is_still_refused(rig):
    from witwin.radar.smpl import SmplPoseDeformation

    pose, shape, _, _ = _parameters()
    with pytest.raises(RuntimeError, match="ADR-038"):
        SmplPoseDeformation(_body(rig, pose, shape), pose_rate=torch.zeros(72, requires_grad=True))
