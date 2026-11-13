"""Mesh-target material parameters as leaves (R-ADR-052).

The facet response reads each material's front-roughness rms height and every
layer's thickness, complex permittivity and permeability. They are
differentiable inputs of the native facet operators: the forward-mode companion
carries their tangents on its lane beside the geometry's, and the reverse pass
gives each facet a second sweep over its rows, one layer per pass, reduced per
material in facet order.

Checked at the operator (the synthetic two-site fixture of
``test_facet_scatter_kernel.py``, single and two-layer stacks, smooth and
rough) against central differences and the adjoint identity, and through
``Radar.trace`` from a Core ``PhysicalMaterial`` whose ``eps_r``, ``sigma_e``,
``thickness_m`` or roughness ``rms_height_m`` is a leaf, in both modes. The
material reverse pass is launched only when a material derivative is asked for.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwad
from support import oracle_scenes as sc
from test_facet_scatter_kernel import DEVICE, Fixture

pytestmark = pytest.mark.gpu


def _fixture(stack: str, rms: float) -> Fixture:
    if stack == "pec":
        return Fixture(material="pec", rms=rms)
    fixture = Fixture(material=(5.24, -0.18), rms=rms)
    if stack == "two_layers":
        fixture.layers = np.asarray([[0.004, 3.1, -0.05, 1.0], [0.011, 6.5, -0.4, 1.2]], dtype=np.float64)
        fixture.layer_offsets = np.asarray([0, 2])
    return fixture


@pytest.mark.parametrize("rms", [0.0, 0.003], ids=["smooth", "rough"])
@pytest.mark.parametrize("stack", ["pec", "one_layer", "two_layers"])
def test_the_material_tables_match_central_differences_in_both_modes(stack, rms):
    fixture = _fixture(stack, rms)
    params, layers = fixture.material_tables()
    generator = torch.Generator().manual_seed(5)
    # The model column selects the law and takes no derivative; it is not moved.
    params_direction = torch.tensor([0.0, 1e-3 if rms > 0 else 0.0]).repeat(params.numel() // 2).to(DEVICE)
    layers_direction = (torch.randn(layers.shape, generator=generator) * layers.abs().cpu() * 0.05).to(DEVICE)
    weights = torch.randn((4, 4, 2), generator=generator).to(DEVICE)
    geometry = [value.detach() for value in fixture.inputs()]

    def loss(material_params, material_layers):
        matrix, _ = fixture.evaluate(*geometry, materials=(material_params, material_layers))
        return (weights * matrix).sum()

    params_leaf = params.clone().requires_grad_(True)
    layers_leaf = layers.clone().requires_grad_(True)
    grad_params, grad_layers = torch.autograd.grad(loss(params_leaf, layers_leaf), (params_leaf, layers_leaf))
    reverse = float((grad_params.double() * params_direction.double()).sum())
    reverse += float((grad_layers.double() * layers_direction.double()).sum())
    assert float(grad_params[0::2].abs().sum()) == 0.0
    with fwad.dual_level():
        forward = loss(fwad.make_dual(params, params_direction), fwad.make_dual(layers, layers_direction))
        forward = float(fwad.unpack_dual(forward).tangent)

    def shifted(t):
        return float(loss(params + t * params_direction, layers + t * layers_direction))

    central = (shifted(1.0) - shifted(-1.0)) / 2.0
    assert forward == pytest.approx(reverse, rel=1e-4, abs=1e-9)
    if stack == "pec" and rms == 0.0:
        # A smooth conductor has no material parameter at all: exactly zero, in both modes.
        assert forward == 0.0 and reverse == 0.0
    else:
        assert forward == pytest.approx(central, rel=2e-2)


def test_the_material_reverse_pass_runs_only_when_asked():
    """A geometry-only backward launches no material sweep and no material reduction."""

    from torch.profiler import ProfilerActivity, profile

    fixture = _fixture("one_layer", 0.003)
    params, layers = fixture.material_tables()

    def kernels(*, material: bool) -> list[str]:
        inputs = [value.requires_grad_(True) for value in fixture.inputs()]
        tables = (params.clone().requires_grad_(material), layers.clone().requires_grad_(material))
        matrix, _ = fixture.evaluate(*inputs, materials=tables)
        with profile(activities=[ProfilerActivity.CUDA]) as trace:
            matrix.sum().backward()
            torch.cuda.synchronize()
        return [event.name for event in trace.events() if "facet" in event.name and event.device_type.name == "CUDA"]

    without = kernels(material=False)
    with_material = kernels(material=True)
    assert sum("facet_material_reduce" in name for name in without) == 0, without
    assert sum("facet_material_reduce" in name for name in with_material) == 1, with_material
    assert sum("facet_response_backward" in name for name in with_material) == 1, with_material


# ---------------------------------------------------------------------------
# Through the public verb, from a Core material
# ---------------------------------------------------------------------------


BASE = {"eps_r": 4.0, "sigma_e": 0.3, "thickness_m": 0.004, "rms_height_m": 2e-4}
STEPS = {"eps_r": 1e-2, "sigma_e": 1e-3, "thickness_m": 1e-5, "rms_height_m": 1e-6}


def _trace_loss(values: dict, grad: str) -> torch.Tensor:
    pytest.importorskip("witwin.channel")
    from witwin.core import PhysicalMaterial, SurfaceRoughness

    from witwin.radar import MeshTargets

    vertices, faces = sc.plate_mesh(4, side_m=0.05, range_m=3.0)
    vertices = vertices + np.outer(vertices[:, 1], [0.15, 0.0, 0.0])
    roughness = SurfaceRoughness(
        rms_height_m=values["rms_height_m"], correlation_length_x_m=0.01, correlation_length_y_m=0.01
    )
    material = PhysicalMaterial(
        eps_r=values["eps_r"],
        sigma_e=values["sigma_e"],
        thickness_m=values["thickness_m"],
        name="slab",
        roughness_front=roughness,
    )
    scene = sc.mesh_scene(vertices, faces, material=material)
    rows = sc.radar().trace(scene, MeshTargets(), times=(0.0,), reflections=0, ad_mode=grad).rows()
    total = rows.complex_transfer_ref.sum()
    return (total.real * 0.7 + total.imag * 1.3) * 1e6


@pytest.mark.parametrize("name", sorted(BASE))
def test_a_core_material_leaf_reaches_the_traced_rows_in_both_modes(name):
    base = {key: torch.tensor(value) for key, value in BASE.items()}
    values = dict(base)
    values[name] = base[name].clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(_trace_loss(values, "vjp"), values[name])
    with fwad.dual_level():
        values = dict(base)
        values[name] = fwad.make_dual(base[name], torch.tensor(1.0))
        jvp = float(fwad.unpack_dual(_trace_loss(values, "jvp")).tangent)
    step = STEPS[name]
    plus, minus = dict(base), dict(base)
    plus[name] = base[name] + step
    minus[name] = base[name] - step
    central = (float(_trace_loss(plus, "none")) - float(_trace_loss(minus, "none"))) / (2.0 * step)
    assert float(gradient) == pytest.approx(jvp, rel=1e-4)
    assert jvp == pytest.approx(central, rel=5e-3)


def test_a_roughness_leaf_at_exactly_zero_is_refused():
    """A smooth surface is the edge of the roughness model's domain: the diffuse amplitude grows like |h|."""

    values = {key: torch.tensor(value) for key, value in BASE.items()}
    values["rms_height_m"] = torch.tensor(0.0, requires_grad=True)
    with pytest.raises(RuntimeError, match="one-sided derivative"):
        _trace_loss(values, "vjp")
