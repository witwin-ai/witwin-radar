"""Per-row responses on a slot-major batch: the strided route against per-slot table copies.

A slot-major replay of ``G`` instants composes ``G`` copies of one join.
``TwoWayComposer.for_slots`` builds those copies as block-diagonal tables; the
strided route hands the per-row response families ONE slot's tables and a slot
count instead. Each family walks its per-slot CSR once per slot in slot order,
which is the summation order of the copies, so the two routes must agree bit
for bit: forward values, every reverse-mode gradient and every forward
tangent, compared with ``torch.equal``. The aspect response is checked with
sites shared by every slot and with one site set per slot (a time-resolved
axis); the tabulated and facet responses share their sites.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwad
from support import join_fixture as fx

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
SOURCES = [10, 11]
SINKS = [30, 31]
SITES = [20, 21, 22]
SLOTS = 3


def _composer():
    from witwin.radar.paths import TwoWayComposer

    inbound = fx.frozen_leg(fx.leg_rows(SOURCES, SITES, (0, 1)), device=DEVICE)
    outbound = fx.frozen_leg(fx.outbound_leg_rows(SITES, SINKS, (0,)), device=DEVICE)
    return TwoWayComposer.freeze(
        inbound,
        outbound,
        torch.tensor(SITES, dtype=torch.int64, device=DEVICE),
        radar_source_ids=SOURCES,
        radar_sink_ids=SINKS,
        reference_frequency_hz=77e9,
    )


def _units(rows: int, mean, seed: int) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    values = np.asarray(mean, dtype=np.float64)[None, :] + 0.3 * generator.standard_normal((rows, 3))
    return torch.tensor(values / np.linalg.norm(values, axis=1, keepdims=True), dtype=torch.float32, device=DEVICE)


class Legs:
    """Slot-major inbound and outbound leg payloads whose geometry is the AD leaf."""

    def __init__(self, composer):
        rows_in = composer.inbound_row_count * SLOTS
        rows_out = composer.outbound_row_count * SLOTS
        # Inbound waves propagate toward +x into the sites; outbound legs arrive from -x.
        self.k_in = _units(rows_in, (1.0, 0.1, -0.1), seed=1)
        self.k_out = _units(rows_out, (1.0, -0.15, 0.05), seed=2)
        self.departure = _units(rows_out, (-1.0, 0.15, 0.1), seed=3)
        self.rows_in, self.rows_out = rows_in, rows_out

    def leaves(self):
        return [self.k_in, self.k_out, self.departure]

    def batches(self, k_in, k_out, departure):
        delay_in = torch.zeros(self.rows_in, device=DEVICE)
        delay_out = torch.zeros(self.rows_out, device=DEVICE)
        ones_in = torch.ones(self.rows_in, dtype=torch.complex64, device=DEVICE)
        ones_out = torch.ones(self.rows_out, dtype=torch.complex64, device=DEVICE)
        # The inbound leg's last point before the site is far up-range: never the site itself.
        inbound = fx.leg_batch(delay_in, ones_in, direction=k_in, site_departure=-100.0 * k_in)
        outbound = fx.leg_batch(delay_out, ones_out, direction=k_out, site_departure=departure)
        return inbound, outbound


def _aspect(sites: int):
    from witwin.radar.scattering import AspectScatterResponse

    axis = _units(sites, (-1.0, 0.2, 0.1), seed=5).requires_grad_(True)
    amplitude = torch.linspace(0.5, 1.5, sites, device=DEVICE).requires_grad_(True)
    phase = torch.linspace(-0.4, 0.7, sites, device=DEVICE).requires_grad_(True)
    response = AspectScatterResponse(
        axis=axis, amplitude=amplitude, phase_rad=phase, exponent=2.0, coherent_interval_s=1e-3
    )
    return response, [axis, amplitude, phase]


def _table(sites: int):
    from witwin.radar.scattering import TableScatterResponse

    samples = np.linspace(0.0, math.pi, 19)
    axis = _units(sites, (-1.0, 0.2, 0.1), seed=6).requires_grad_(True)
    response = TableScatterResponse(
        axis=axis,
        table_re=torch.tensor(np.cos(2 * samples) + 2.0, dtype=torch.float32, device=DEVICE),
        table_im=torch.tensor(np.sin(3 * samples), dtype=torch.float32, device=DEVICE),
    )
    return response, [axis]


def _facets(sites: int):
    """One facet per site, facing -x, a few centimetres across, near the origin."""

    from witwin.radar.facets import MODEL_PEC, FacetFrame, FacetModel
    from witwin.radar.scattering import FacetScatterResponse

    rng = np.random.default_rng(9)
    base = np.array([[0.0, -0.02, -0.015], [0.004, 0.0, 0.03], [0.0, 0.025, -0.02]])
    corners = np.stack(
        [(base + 0.004 * rng.standard_normal((3, 3)) + [0.0, 0.03 * s, 0.0]).reshape(-1) for s in range(sites)]
    )
    centroids = corners.reshape(sites, 3, 3).mean(axis=1)

    def long(values):
        return torch.as_tensor(np.asarray(values), dtype=torch.int64, device=DEVICE)

    model = FacetModel(
        structure_ids=(1,),
        vertex_counts=(3 * sites,),
        faces=long(np.arange(3 * sites).reshape(sites, 3)),
        site_offsets=long(np.arange(sites + 1)),
        site_facets=long(np.arange(sites)),
        facet_site=long(np.arange(sites)),
        vertex_offsets=long(np.arange(3 * sites + 1)),
        vertex_entries=long(np.arange(3 * sites)),
        facet_material=long(np.zeros(sites)),
        material_params=torch.tensor([[float(MODEL_PEC), 0.0]], device=DEVICE),
        layer_offsets=long([0, 0]),
        layers=torch.tensor([[0.0, 1.0, 0.0, 1.0]], device=DEVICE),
        key=(),
    )
    corner_leaf = torch.tensor(corners, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    site_leaf = torch.tensor(centroids, dtype=torch.float64, device=DEVICE).requires_grad_(True)
    frame = FacetFrame(
        model=model, corners=corner_leaf, sites=site_leaf, material_params=model.material_params, layers=model.layers
    )
    return FacetScatterResponse(wavenumber_rad_per_m=60.0, frame=frame), [corner_leaf, site_leaf]


def _outputs(result):
    return result if isinstance(result, tuple) else (result,)


def _run(response, composer, legs: Legs, flags, *, slots: int):
    inputs = [value.clone().requires_grad_(True) for value in legs.leaves()]
    inbound, outbound = legs.batches(*inputs)
    result = (
        response.evaluate_rows(composer, inbound, outbound, flags, slots=slots)
        if slots > 1
        else (response.evaluate_rows(composer, inbound, outbound, flags))
    )
    return [value for value in _outputs(result) if value.dtype == torch.float32 and value.requires_grad], inputs


@pytest.mark.parametrize("kind", ["aspect-shared", "aspect-per-slot", "table", "facet"])
def test_the_strided_route_is_bit_identical_to_per_slot_table_copies(kind):
    composer = _composer()
    local = kind == "aspect-per-slot"
    copies = composer.for_slots(SLOTS, per_slot_response=local)
    sites = composer.site_count * (SLOTS if local else 1)
    build = {"aspect-shared": _aspect, "aspect-per-slot": _aspect, "table": _table, "facet": _facets}[kind]
    response, parameters = build(sites)
    legs = Legs(composer)
    rows = composer.path_count * SLOTS
    valid = torch.ones(rows, dtype=torch.int32, device=DEVICE)
    valid[1] = 0

    strided, strided_inputs = _run(response, composer, legs, valid, slots=SLOTS)
    copied, copied_inputs = _run(response, copies, legs, valid, slots=1)
    for a, b in zip(strided, copied, strict=True):
        assert torch.equal(a, b)

    weights = [torch.randn(value.shape, generator=torch.Generator().manual_seed(4)).to(DEVICE) for value in strided]
    leaves_s = strided_inputs + parameters
    grads_s = torch.autograd.grad(
        sum((w * v).sum() for w, v in zip(weights, strided, strict=True)), leaves_s, allow_unused=True
    )
    grads_c = torch.autograd.grad(
        sum((w * v).sum() for w, v in zip(weights, copied, strict=True)), copied_inputs + parameters, allow_unused=True
    )
    for a, b in zip(grads_s, grads_c, strict=True):
        assert (a is None and b is None) or torch.equal(a, b)

    tangents = [
        torch.randn(value.shape, generator=torch.Generator().manual_seed(7)).to(DEVICE) for value in legs.leaves()
    ]
    tangents = [t.to(v.dtype) for t, v in zip(tangents, legs.leaves(), strict=True)]

    def forward_tangent(target, slots):
        with fwad.dual_level():
            duals = [fwad.make_dual(v.detach(), t) for v, t in zip(legs.leaves(), tangents, strict=True)]
            inbound, outbound = legs.batches(*duals)
            result = _outputs(
                response.evaluate_rows(target, inbound, outbound, valid, slots=slots)
                if slots > 1
                else response.evaluate_rows(target, inbound, outbound, valid)
            )
            return [fwad.unpack_dual(value).tangent for value in result if value.dtype == torch.float32]

    for a, b in zip(forward_tangent(composer, SLOTS), forward_tangent(copies, 1), strict=True):
        assert (a is None and b is None) or torch.equal(a, b)
