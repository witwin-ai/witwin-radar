"""The join's float64 path length, element factor and scattering matrix: forward, VJP and JVP.

R-ADR-040 and R-ADR-041 put three new terms into the native join: the
round-trip length recomputed in float64 from the legs' geometry (with
Channel's float32 phase removed), the element factor ``1/|P_perp(d) p|`` and
the site's 2x2 scattering matrix in its BSA basis. These tests drive the join
with fabricated legs that carry that geometry and hold it to an independent
float64 Torch statement of the same law (``tests/reference/two_way_torch.py``):
the primal, the reverse-mode gradient of every differentiable input, and the
forward-mode tangent through the adjoint identity.
"""

from __future__ import annotations

import math

import pytest
import torch
from reference.two_way_torch import element_factor_f64, geometric_join_reference, leg_length_f64
from support import join_fixture as fx

from witwin.radar.paths import SPEED_OF_LIGHT_M_PER_S, TwoWayComposer, channel_wave_number

pytestmark = pytest.mark.gpu

SOURCES = [10, 11]
SITES = [20, 21]
SINKS = [30, 31]
COMPONENTS = (0, 1)  # the fabricated rows alias depth to the component: depth 0 and 1
FREQUENCY_HZ = 77.0e9
POLARIZATION = ((0.0, 1.0, 0.2), (0.1, 1.0, 0.0))


def _composer() -> TwoWayComposer:
    return TwoWayComposer.freeze(
        fx.frozen_leg(fx.leg_rows(SOURCES, SITES, COMPONENTS), device="cuda"),
        fx.frozen_leg(fx.outbound_leg_rows(SITES, SINKS, COMPONENTS), device="cuda"),
        torch.tensor(SITES, dtype=torch.int64, device="cuda"),
        radar_source_ids=SOURCES,
        radar_sink_ids=SINKS,
        reference_frequency_hz=FREQUENCY_HZ,
    )


def _leg_state(rows: int, *, seed: int, polarization, site_dtype=torch.float32) -> dict[str, torch.Tensor]:
    """Element, one interaction point, site, depth, Channel length, field, direction, launch polarization.

    The launch polarization differs per row (a pose turns it per instant), so
    the join must read each row's own.
    """

    generator = torch.Generator().manual_seed(seed)

    def uniform(*shape, low=0.0, high=1.0):
        return low + (high - low) * torch.rand(*shape, generator=generator, dtype=torch.float64)

    source = uniform(rows, 3, low=-0.05, high=0.05)
    site = torch.stack((uniform(rows, low=6.0, high=9.0), uniform(rows, low=-1.0, high=1.0), uniform(rows)), dim=1)
    points = (0.5 * (source + site) + uniform(rows, 3, low=-0.5, high=0.5))[:, None, :]
    depth = torch.tensor([row % 2 for row in range(rows)], dtype=torch.int32)
    exact = leg_length_f64(source, points, depth, site)
    # Channel's float32 length differs from the float64 one at the float32 ulp.
    channel = (exact * (1.0 + 3.0e-7 * (uniform(rows) - 0.5))).float()
    field = torch.complex(uniform(rows, 3, low=-1.0, high=1.0), uniform(rows, 3, low=-1.0, high=1.0))
    direction = site - points[:, 0]
    direction = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    return {
        "source": source.float(),
        "site": site.to(site_dtype),
        "points": points.float(),
        "depth": depth,
        "length": channel,
        "field": field.to(torch.complex64),
        "direction": direction.float(),
        "polarization": (torch.tensor(polarization, dtype=torch.float64) + 0.1 * uniform(rows, 3)).float(),
    }


CONSTANT = ("depth", "polarization")


def _cuda_leaf(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to("cuda").contiguous().requires_grad_(value.is_floating_point() or value.is_complex())


def _batch(state, leaves):
    return fx.leg_batch(
        torch.zeros(state["field"].shape[0], dtype=torch.float32, device="cuda"),
        torch.zeros(state["field"].shape[0], dtype=torch.complex64, device="cuda"),
        field=leaves["field"],
        direction=leaves["direction"],
        geometry={
            "source": leaves["source"],
            "site": leaves["site"],
            "points": leaves["points"],
            "depth": state["depth"].to("cuda"),
            "length": leaves["length"],
            "polarization": None if state["polarization"] is None else state["polarization"].to("cuda"),
        },
    )


class _Matrix:
    """A per-site scattering matrix response on the composer's protocol."""

    matrix_width = 4

    def __init__(self, matrix: torch.Tensor, up: torch.Tensor) -> None:
        self.matrix, self.up = matrix, up

    @property
    def is_geometry_dependent(self) -> bool:
        return False

    def evaluate(self, row_count, device):
        return self.matrix

    def site_up(self, row_count, device):
        return self.up


def _case(*, matrix: bool, site_dtype=torch.float32, seed: int = 5):
    composer = _composer()
    states = (
        _leg_state(composer.inbound_row_count, seed=seed, polarization=POLARIZATION[0], site_dtype=site_dtype),
        _leg_state(composer.outbound_row_count, seed=seed + 1, polarization=POLARIZATION[1], site_dtype=site_dtype),
    )
    leaves = [{name: _cuda_leaf(value) for name, value in state.items() if name not in CONSTANT} for state in states]
    generator = torch.Generator().manual_seed(seed + 2)
    width = (composer.site_count, 4) if matrix else (composer.site_count,)
    s = torch.complex(torch.rand(width, generator=generator), torch.rand(width, generator=generator)) - (0.5 + 0.5j)
    s_leaf = _cuda_leaf(s.to(torch.complex64))
    up = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.1, 1.0]], dtype=torch.float32, device="cuda")
    up = up / torch.linalg.vector_norm(up, dim=1, keepdim=True)
    response = _Matrix(s_leaf, up) if matrix else fx_response(s_leaf)
    return composer, states, leaves, s_leaf, up, response


def fx_response(value: torch.Tensor):
    from reference.two_way_torch import PerSiteResponse

    return PerSiteResponse(value)


def _reference(composer, states, leaves, s_leaf, up, *, matrix: bool):
    """The float64 CPU reference, on float64 copies of every leaf."""

    copies = [
        {
            name: value.detach().cpu().to(torch.complex128 if value.is_complex() else torch.float64)
            for name, value in leg.items()
        }
        for leg in leaves
    ]
    for leg in copies:
        for value in leg.values():
            value.requires_grad_(True)
    s64 = s_leaf.detach().cpu().to(torch.complex128).requires_grad_(True)
    depth = [torch.tensor([row % 2 for row in range(int(leg["source"].shape[0]))]) for leg in copies]
    lengths = [
        leg_length_f64(leg["source"], leg["points"], d, leg["site"]) for leg, d in zip(copies, depth, strict=True)
    ]
    factors = [
        element_factor_f64(leg["source"], leg["points"], d, leg["site"], state["polarization"].double())
        for leg, d, state in zip(copies, depth, states, strict=True)
    ]
    tau, value = geometric_join_reference(
        field_in=copies[0]["field"],
        field_out=copies[1]["field"],
        length_in=lengths[0],
        length_out=lengths[1],
        channel_length_in=copies[0]["length"],
        channel_length_out=copies[1]["length"],
        factor_in=factors[0],
        factor_out=factors[1],
        matrix=s64,
        k_in=copies[0]["direction"] if matrix else None,
        k_out=copies[1]["direction"] if matrix else None,
        up=up.cpu().double() if matrix else None,
        idx_in=composer.inbound_row.cpu(),
        idx_out=composer.outbound_row.cpu(),
        idx_s=composer.response_slot.cpu(),
        k64=2.0 * math.pi * FREQUENCY_HZ / SPEED_OF_LIGHT_M_PER_S,
        k32=channel_wave_number(FREQUENCY_HZ),
        speed_of_light=SPEED_OF_LIGHT_M_PER_S,
    )
    return tau, value, copies, s64


def _weights(rows: int):
    generator = torch.Generator().manual_seed(77)
    return torch.rand(3, rows, generator=generator, dtype=torch.float64) - 0.5


@pytest.mark.parametrize("matrix", [False, True], ids=["scalar", "matrix"])
@pytest.mark.parametrize("site_dtype", [torch.float32, torch.float64], ids=["site32", "site64"])
def test_forward_matches_the_float64_reference(matrix, site_dtype):
    composer, states, leaves, s_leaf, up, response = _case(matrix=matrix, site_dtype=site_dtype)
    composed = composer.compose(_batch(states[0], leaves[0]), _batch(states[1], leaves[1]), response)
    tau, value, _, _ = _reference(composer, states, leaves, s_leaf, up, matrix=matrix)
    assert composed.total_delay_residual_s is not None
    produced_tau = composed.total_delay_s.double().cpu() + composed.total_delay_residual_s.double().cpu()
    # The float32 pair carries the float64 delay to ~1e-15 relative.
    assert float(((produced_tau - tau.detach()) / tau.detach()).abs().max()) < 1e-13
    produced = composed.complex_transfer_ref.cpu().to(torch.complex128)
    scale = float(value.detach().abs().max())
    assert float((produced - value.detach()).abs().max()) < 2e-6 * scale


@pytest.mark.parametrize("matrix", [False, True], ids=["scalar", "matrix"])
@pytest.mark.parametrize("site_dtype", [torch.float32, torch.float64], ids=["site32", "site64"])
def test_reverse_mode_matches_float64_autograd(matrix, site_dtype):
    composer, states, leaves, s_leaf, up, response = _case(matrix=matrix, site_dtype=site_dtype)
    composed = composer.compose(_batch(states[0], leaves[0]), _batch(states[1], leaves[1]), response)
    w = _weights(composer.path_count)
    loss = (
        (w[0].cuda().float() * composed.complex_transfer_ref.real).sum()
        + (w[1].cuda().float() * composed.complex_transfer_ref.imag).sum()
        + (w[2].cuda().float() * composed.total_delay_s).sum() * 1e9
    )
    loss.backward()
    tau, value, copies, s64 = _reference(composer, states, leaves, s_leaf, up, matrix=matrix)
    reference = (w[0] * value.real).sum() + (w[1] * value.imag).sum() + (w[2] * tau).sum() * 1e9
    reference.backward()
    names = ["field", "source", "site", "points", "length"] + (["direction"] if matrix else [])
    for leg, copy in zip(leaves, copies, strict=True):
        for name in names:
            produced = leg[name].grad.cpu().to(copy[name].grad.dtype)
            expected = copy[name].grad
            scale = float(expected.abs().max())
            assert scale > 0.0, name
            assert float((produced - expected).abs().max()) < 2e-4 * scale, name
    produced = s_leaf.grad.cpu().to(torch.complex128)
    assert float((produced - s64.grad).abs().max()) < 2e-4 * float(s64.grad.abs().max())


@pytest.mark.parametrize("matrix", [False, True], ids=["scalar", "matrix"])
def test_forward_mode_satisfies_the_adjoint_identity(matrix):
    import torch.autograd.forward_ad as forward_ad

    composer, states, leaves, s_leaf, up, response = _case(matrix=matrix, site_dtype=torch.float64)
    generator = torch.Generator().manual_seed(91)
    tangents = [
        {
            name: (torch.rand(value.shape, generator=generator, dtype=torch.float64) - 0.5).to(value.device)
            if not value.is_complex()
            else torch.complex(
                torch.rand(value.shape, generator=generator) - 0.5, torch.rand(value.shape, generator=generator) - 0.5
            ).to(value.device)
            for name, value in leg.items()
        }
        for leg in leaves
    ]
    s_tangent = torch.complex(
        torch.rand(s_leaf.shape, generator=generator) - 0.5, torch.rand(s_leaf.shape, generator=generator) - 0.5
    ).to("cuda")
    w = _weights(composer.path_count).cuda()

    with forward_ad.dual_level():
        duals = [
            {
                name: forward_ad.make_dual(value.detach(), tangent.to(value.dtype))
                for (name, value), tangent in zip(leg.items(), tangent_set.values(), strict=True)
            }  # fmt: skip
            for leg, tangent_set in zip(leaves, tangents, strict=True)
        ]
        s_dual = forward_ad.make_dual(s_leaf.detach(), s_tangent.to(torch.complex64))
        dual_response = _Matrix(s_dual, up) if matrix else fx_response(s_dual)
        composed = composer.compose(_batch(states[0], duals[0]), _batch(states[1], duals[1]), dual_response)
        out = forward_ad.unpack_dual(composed.complex_transfer_ref).tangent.to(torch.complex128)
        out_tau = forward_ad.unpack_dual(composed.total_delay_s).tangent.double()
    forward_value = float((w[0] * out.real).sum() + (w[1] * out.imag).sum() + (w[2] * out_tau).sum() * 1e9)

    composed = composer.compose(_batch(states[0], leaves[0]), _batch(states[1], leaves[1]), response)
    loss = (
        (w[0].float() * composed.complex_transfer_ref.real).sum()
        + (w[1].float() * composed.complex_transfer_ref.imag).sum()
        + (w[2].float() * composed.total_delay_s).sum() * 1e9
    )
    loss.backward()
    reverse_value = 0.0
    for leg, tangent_set in zip(leaves, tangents, strict=True):
        for name, value in leg.items():
            grad = value.grad
            tangent = tangent_set[name]
            if grad is None:
                # Not an input of this response kind (the direction of a scalar response).
                continue
            if value.is_complex():
                reverse_value += float((grad.real.double() * tangent.real + grad.imag.double() * tangent.imag).sum())
            else:
                reverse_value += float((grad.double() * tangent.double()).sum())
    reverse_value += float(
        (s_leaf.grad.real.double() * s_tangent.real + s_leaf.grad.imag.double() * s_tangent.imag).sum()
    )
    assert abs(forward_value - reverse_value) <= 1e-4 * max(abs(forward_value), abs(reverse_value)), (
        forward_value,
        reverse_value,
    )


def test_without_launch_polarization_the_factor_is_one():
    """Legs that publish no launch polarization compose Channel's launched field as it stands."""

    composer = _composer()
    states = (
        _leg_state(composer.inbound_row_count, seed=3, polarization=POLARIZATION[0]),
        _leg_state(composer.outbound_row_count, seed=4, polarization=POLARIZATION[1]),
    )
    leaves = [{name: _cuda_leaf(value) for name, value in state.items() if name not in CONSTANT} for state in states]
    s = torch.ones(composer.site_count, dtype=torch.complex64, device="cuda")
    bare = [{**state, "polarization": None} for state in states]
    composed = composer.compose(_batch(bare[0], leaves[0]), _batch(bare[1], leaves[1]), fx_response(s))
    with_factor = composer.compose(_batch(states[0], leaves[0]), _batch(states[1], leaves[1]), fx_response(s))
    ratio = (with_factor.complex_transfer_ref / composed.complex_transfer_ref).abs().cpu()
    expected = []
    for row in range(composer.path_count):
        i, o = int(composer.inbound_row[row]), int(composer.outbound_row[row])
        factors = [
            element_factor_f64(
                state["source"].double(),
                state["points"].double(),
                state["depth"],
                state["site"].double(),
                state["polarization"].double(),
            )[index]
            for state, index in zip(states, (i, o), strict=True)
        ]
        expected.append(float(factors[0] * factors[1]))
    assert torch.allclose(ratio.double(), torch.tensor(expected, dtype=torch.float64), rtol=1e-5, atol=0.0)
