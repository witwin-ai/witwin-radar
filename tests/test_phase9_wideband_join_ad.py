"""An endpoint gradient through ``_compose_band``, column by column.

Phase 8 shipped the wideband band and pinned two things about the join's
per-column loop (``witwin/radar/paths.py``): the primal product
``H_in(f_j) * S * H_out(f_j)`` and the launch count, one per column. It pinned
NOTHING about the loop's derivative. Every Phase-8 AD test builds synthetic
``[K, F]`` band tensors and differentiates the SYNTHESIS kernel with respect to
them; no test drives an endpoint leaf through the loop at all. That leaves a
64-column band whose gradient path is 64 separate ``_TwoWayJoin.apply`` calls,
none of which is exercised from a scene leaf in either mode.

This module closes that. The loss is taken on the published ``[K, F]`` band, the
leaf is a scene endpoint, and the oracle is a difference of the production chain
at perturbed positions.

**Why the fourth-order stencil.** The band's phase is the propagation phase at
77 GHz, so the loss oscillates with a wavenumber of ``2 * 2 * pi * f / c`` =
3226 rad/m in a round-trip position. A second-order difference at a step large
enough to clear the float32 noise floor carries 0.4-4.5% truncation from that
oscillation alone, which is the same size as the disagreement it would be trying
to find. Measured second-order relative error at steps of 1e-5, 5e-5, 1e-4 and
2e-4 m: 1.8%, 0.37%, 1.3%, 4.5%. Fourth order at the same steps: 0.80%, 0.50%,
0.16%, 0.26%.

**Why each column must differ.** A loop that reused one column's join for every
column would reproduce the primal only if the leg responses were flat across the
band, which they are not; but a loop that reused one column's DERIVATIVE would
still reproduce every primal test in the tree. So the per-column derivatives are
compared to each other as well as to a difference.

Finite differences are the test oracle and never a production route.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

pytest.importorskip("witwin.channel")

from support import fd  # noqa: E402
from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402

from witwin.radar.channel import ChannelPropagationAdapter  # noqa: E402

pytestmark = pytest.mark.gpu


F_REF = geo.REFERENCE_FREQUENCY_HZ
SUBCARRIER_SPACING_HZ = 25.0e6

#: Four columns rather than sixteen. The loop is the same loop at any width and
#: every test below differences it once per leaf component per column, so the
#: width buys nothing but wall time. The aliasing test sweeps wider.
BAND_COLUMNS = 4

#: Metres, fourth-order stencil. See the module docstring for the sweep.
STEP_M = 1.0e-4
FD_RTOL = 1.0e-2

#: Below this a gradient component is structurally zero.
ZERO_FLOOR = 1.0e-6


def _offsets(count: int) -> tuple[float, ...]:
    return tuple(float(n * SUBCARRIER_SPACING_HZ) for n in range(count))


@pytest.fixture(scope="module")
def narrow():
    return drv.MultiEndpointSpike()


def _banded(narrow, columns: int):
    """A banded spike over the SAME compiled scene as the narrowband one.

    Sharing the compiled scene removes the question of whether a difference
    between the two is a band effect or a compile effect.
    """

    adapter = ChannelPropagationAdapter(
        narrow.compiled,
        reference_frequency_hz=F_REF,
        components=drv.MULTIPATH_COMPONENTS,
        max_depth=1,
        frequency_offsets_hz=_offsets(columns),
    )
    return drv.MultiEndpointSpike(compiled=narrow.compiled, adapter=adapter)


@pytest.fixture(scope="module")
def banded(narrow):
    return _banded(narrow, BAND_COLUMNS)


@pytest.fixture(scope="module")
def weights(banded):
    """A deterministic sign-mixed cotangent over ``[K, F]``, never all ones."""

    generator = torch.Generator().manual_seed(4177)
    raw = torch.rand(2, banded.composer.path_count, BAND_COLUMNS, generator=generator, dtype=torch.float32)
    return torch.complex(2.0 * raw[0] - 1.0, 2.0 * raw[1] - 1.0).cuda()


def _band(spike, sites=None, *, ad_mode: str = "none"):
    composed, _, _ = spike.frame(sites, drv.make_response(), ad_mode=ad_mode, include_delay_rate=False)
    band = composed.frequency_response
    assert band is not None, "this spike declared a band"
    return band


def _band_loss(spike, weights, sites=None, *, ad_mode: str = "none"):
    band = _band(spike, sites, ad_mode=ad_mode)
    return (band.real * weights.real + band.imag * weights.imag).sum()


# --------------------------------------------------------------------------
# 1. The gradient exists and is right
# --------------------------------------------------------------------------


def test_the_published_band_carries_a_graph_from_the_endpoint_leaf(banded):
    """The premise. Before anything is differenced, the tape must be there."""

    sites = banded.site_tensor(requires_grad=True)
    band = _band(banded, sites, ad_mode="vjp")
    assert band.shape == (banded.composer.path_count, BAND_COLUMNS)
    assert band.requires_grad
    assert band.grad_fn is not None


def test_a_reverse_endpoint_gradient_reaches_every_wideband_column(banded, weights):
    """``d(loss over [K, F])/d(site)`` against a fourth-order difference."""

    base = banded.site_tensor()
    sites = base.clone().requires_grad_(True)
    _band_loss(banded, weights, sites, ad_mode="vjp").backward()
    gradient = sites.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()

    for site in range(base.shape[0]):
        for axis in range(2):
            samples = {}
            realized = {}
            for offset in (-2, -1, 1, 2):
                moved = base.clone()
                moved[site, axis] += offset * STEP_M
                samples[offset] = float(_band_loss(banded, weights, moved))
                realized[offset] = float(moved[site, axis])
            measured = fd.fourth_order_difference(samples, (realized[1] - realized[-1]) / 2.0)
            expected = float(gradient[site, axis])
            # Non-vacuity: this fixture's smallest live component is ~0.1, so a
            # zero here would mean the leaf stopped reaching the band.
            assert abs(expected) > ZERO_FLOOR, (site, axis, expected)
            assert fd.relative_error(measured, expected, floor=ZERO_FLOOR) < FD_RTOL, (site, axis, measured, expected)


def test_a_forward_tangent_on_an_endpoint_reaches_the_band(banded, weights):
    """JVP against VJP projected against a difference, on the same loss.

    An ADR-038 forward-only dual. Each column's ``_TwoWayJoin.jvp`` runs on its
    own saved context, so a loop that forwarded the reference column's tangent
    to every column would disagree with the projection here.
    """

    direction = torch.tensor([[0.3, -0.9, 0.0], [0.7, 0.4, 0.0]], dtype=torch.float32, device="cuda")
    base = banded.site_tensor()

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(base.clone(), direction)
        assert not dual.requires_grad
        tangent = forward_ad.unpack_dual(_band_loss(banded, weights, dual, ad_mode="jvp")).tangent
        assert tangent is not None
        forward = float(tangent)

    sites = base.clone().requires_grad_(True)
    _band_loss(banded, weights, sites, ad_mode="vjp").backward()
    projected = float((sites.grad * direction).sum())

    samples = {}
    for offset in (-2, -1, 1, 2):
        samples[offset] = float(_band_loss(banded, weights, base + offset * STEP_M * direction))
    differenced = fd.fourth_order_difference(samples, STEP_M)

    assert abs(forward) > ZERO_FLOOR
    assert abs(forward - projected) <= 1.0e-4 * abs(projected)
    assert fd.relative_error(differenced, forward, floor=ZERO_FLOOR) < FD_RTOL


def test_a_combined_endpoint_perturbation_equals_the_sum_of_its_parts(banded, weights):
    """Sites AND transmitters live at once, against two single-leaf differences.

    The join is bilinear in the two legs' coefficients and the transmitter only
    enters the inbound one, so a mistake that reduced one leg's cotangent into
    the other's slot would still pass every single-leaf test above. Perturbing
    both at once and comparing against the SUM of the single-leaf differences is
    what separates them.
    """

    site_direction = torch.tensor([[0.4, -0.8, 0.0], [-0.6, 0.3, 0.0]], dtype=torch.float32, device="cuda")
    tx_direction = torch.tensor([[0.7, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
    site_base = banded.site_tensor()
    tx_base = banded.transmitter_tensor()

    sites = site_base.clone().requires_grad_(True)
    transmitters = tx_base.clone().requires_grad_(True)
    composed, _, _ = banded.frame(
        sites, drv.make_response(), transmitters=transmitters, ad_mode="vjp", include_delay_rate=False
    )
    band = composed.frequency_response
    (band.real * weights.real + band.imag * weights.imag).sum().backward()
    combined = float((sites.grad * site_direction).sum() + (transmitters.grad * tx_direction).sum())

    def loss(site_step: float, tx_step: float) -> float:
        composed, _, _ = banded.frame(
            site_base + site_step * site_direction,
            drv.make_response(),
            transmitters=tx_base + tx_step * tx_direction,
            include_delay_rate=False,
        )
        value = composed.frequency_response
        return float((value.real * weights.real + value.imag * weights.imag).sum())

    def difference(site_scale: float, tx_scale: float) -> float:
        samples = {offset: loss(offset * STEP_M * site_scale, offset * STEP_M * tx_scale) for offset in (-2, -1, 1, 2)}
        return fd.fourth_order_difference(samples, STEP_M)

    site_only = difference(1.0, 0.0)
    tx_only = difference(0.0, 1.0)
    both = difference(1.0, 1.0)

    # Each leaf really moves the loss, so the sum is not one term plus noise.
    assert abs(site_only) > ZERO_FLOOR
    assert abs(tx_only) > ZERO_FLOOR
    assert fd.relative_error(site_only + tx_only, both, floor=ZERO_FLOOR) < FD_RTOL
    assert fd.relative_error(combined, both, floor=ZERO_FLOOR) < FD_RTOL


# --------------------------------------------------------------------------
# 2. The columns are genuinely independent
# --------------------------------------------------------------------------


def test_each_wideband_column_carries_a_different_endpoint_derivative(banded):
    """A per-column gradient, one column at a time, all four different.

    The measured spread against column 0, as a fraction of column 0's largest
    component: 0.82, 2.14, 0.95. A loop that reused one column's derivative
    would put all three at exactly zero.
    """

    base = banded.site_tensor()
    gradients = []
    for column in range(BAND_COLUMNS):
        sites = base.clone().requires_grad_(True)
        band = _band(banded, sites, ad_mode="vjp")
        (band[:, column].real.sum() + band[:, column].imag.sum()).backward()
        assert sites.grad is not None
        gradients.append(sites.grad.clone())

    scale = float(gradients[0].abs().max())
    assert scale > ZERO_FLOOR
    for column in range(1, BAND_COLUMNS):
        spread = float((gradients[column] - gradients[0]).abs().max()) / scale
        assert spread > 0.1, (column, spread)


def test_the_reference_column_gradient_equals_the_narrowband_join_gradient(narrow, banded):
    """Column 0 is the narrowband join, in the derivative as well as the value.

    Phase 8 pinned the PRIMAL equality bitwise. If the band loop recomputed the
    reference column through a different path its gradient could diverge while
    that primal test stayed green, so the same claim is made about the
    derivative here. The comparison is a tolerance rather than bitwise because
    the two backward passes reduce over different numbers of contributions.
    """

    base = narrow.site_tensor()

    def gradient(spike, *, column: int | None) -> torch.Tensor:
        sites = base.clone().requires_grad_(True)
        composed, _, _ = spike.frame(sites, drv.make_response(), ad_mode="vjp", include_delay_rate=False)
        value = composed.complex_transfer_ref if column is None else composed.frequency_response[:, column]
        (value.real.sum() + value.imag.sum()).backward()
        return sites.grad

    reference = gradient(narrow, column=None)
    column_zero = gradient(banded, column=0)
    scale = float(reference.abs().max())
    assert scale > ZERO_FLOOR
    assert float((reference - column_zero).abs().max()) <= 1.0e-5 * scale


# --------------------------------------------------------------------------
# 3. What the loop actually retains, structurally
# --------------------------------------------------------------------------


@pytest.mark.parametrize("columns", [1, 2, 4, 8])
def test_the_band_loop_keeps_one_join_context_per_column_and_aliases_its_tables(narrow, columns):
    """``F + 1`` contexts, and seven of each context's ten saved tensors alias.

    This is a STRUCTURAL statement, not a budget: the loop retains one autograd
    context per column plus the reference one, and the only per-column storage
    in that context is the two legs' coefficient slices. The index tables, the
    validity mask and the scatter response are the same objects in every
    context, so the retained bytes grow by ``4 * 2 * (R_in + R_out)`` per column
    and not by the whole context. The survey's "64 copies of the join tape"
    reading is therefore an overestimate of the memory and an exact count of the
    contexts, and a refactor that broke the aliasing would be invisible to every
    primal test.
    """

    import witwin.radar.paths as two_way

    spike = _banded(narrow, columns)
    saved = []
    original = two_way._TwoWayJoin.setup_context

    def recording(ctx, inputs, output):
        original(ctx, inputs, output)
        # Read the identities NOW. ``ctx.saved_tensors`` is only legal inside
        # the backward pass, and the storage is released the moment the graph
        # is, so a pointer collected afterwards would be a use-after-free
        # dressed up as an aliasing measurement.
        saved.append(tuple((tensor.data_ptr(), tensor.numel() * tensor.element_size()) for tensor in ctx.to_save))

    two_way._TwoWayJoin.setup_context = staticmethod(recording)
    try:
        sites = spike.site_tensor(requires_grad=True)
        band = _band(spike, sites, ad_mode="vjp")
        assert band.shape[1] == columns
    finally:
        two_way._TwoWayJoin.setup_context = staticmethod(original)

    assert len(saved) == columns + 1, len(saved)
    # Saved order: c_in_re, c_in_im, c_out_re, c_out_im, s_re, s_im, row_valid,
    # idx_in, idx_out, idx_s. The first four are the per-column slices; the last
    # six are frame or topology constants.
    for tensors in saved:
        assert len(tensors) == 10
    for slot in range(4, 10):
        pointers = {tensors[slot][0] for tensors in saved}
        assert len(pointers) == 1, (slot, pointers)
    for slot in range(4):
        pointers = {tensors[slot][0] for tensors in saved}
        assert len(pointers) == columns + 1, (slot, len(pointers))

    # And the marginal cost of a column is exactly those four slices.
    per_column = sum(tensors[slot][1] for slot in range(4))
    whole_context = sum(size for _, size in saved[0])
    assert 0 < per_column < whole_context
