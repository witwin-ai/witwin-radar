"""JVP and VJP of the aspect response, against central differences.

Finite differences are the TEST oracle and never a production route. They are
taken on the float64 host law in ``support/aspect_oracle.py`` rather than on the
kernel's own float32 output, because differencing a float32 primal at
``h = 1e-3`` throws away most of the digits the comparison needs; the reference
is then exact to well past the 1e-3 tolerance and a disagreement is a Jacobian
bug rather than a cancellation artefact.

Both companions are exercised THROUGH the production facade as well as
directly, because ``_AspectResponse.jvp`` and ``.backward`` are the manifest's
named end-to-end callers for two of the three symbols and a direct-only test
would leave that claim unchecked.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

from witwin.radar.paths import TwoWayComposer
from witwin.radar.scattering import AspectScatterResponse

from support import aspect_oracle as oracle  # noqa: E402
from support import join_fixture as fx  # noqa: E402


pytestmark = pytest.mark.gpu

SOURCES = [10, 11]
SINKS = [30]
SITES = [20, 21]
REFERENCE_FREQUENCY_HZ = 77.0e9
EXPONENT = 2.0
COHERENT_INTERVAL_S = 1.0e-3

#: The central-difference step, on unit direction vectors. Large enough that
#: the float64 primal difference is far above rounding, small enough that the
#: quadratic truncation error of a cos^2 lobe is ~1e-6 relative.
FD_STEP = 1.0e-3

#: Agreement between the analytic companion and the differenced reference.
FD_RTOL = 1.0e-3


def _ops():
    from witwin.radar.cuda import build

    return build.build_extension()


def _composer(device: str = "cuda") -> TwoWayComposer:
    inbound = fx.frozen_leg(fx.leg_rows(SOURCES, SITES, (0,)), device=device)
    outbound = fx.frozen_leg(fx.leg_rows(SITES, SINKS, (0,)), device=device)
    return TwoWayComposer.freeze(
        inbound,
        outbound,
        torch.tensor(SITES, dtype=torch.int64, device=device),
        radar_source_ids=SOURCES,
        radar_sink_ids=SINKS,
        reference_frequency_hz=REFERENCE_FREQUENCY_HZ,
    )


def _unit(rows: int, *, seed: int, sign: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.rand(rows, 3, generator=generator, dtype=torch.float64)
    vectors = torch.stack(
        [0.7 + 0.3 * raw[:, 0], 0.5 * (raw[:, 1] - 0.5), 0.5 * (raw[:, 2] - 0.5)],
        dim=1,
    )
    vectors = sign * vectors / torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    return vectors


def _state(device: str = "cuda"):
    """Directions, parameters and a cotangent, all deterministic.

    The directions sit well inside the illuminated cone: a finite difference
    that straddles the clamp measures the clamp, not the lobe.
    """

    composer = _composer(device)
    dir_in = _unit(composer.inbound_row_count, seed=71, sign=-1.0)
    dir_out = _unit(composer.outbound_row_count, seed=72, sign=1.0)
    axis = oracle.unit_rows(
        [(1.0, 0.2 * index, -0.15 * index) for index in range(composer.site_count)]
    )
    amplitude = torch.tensor(
        [1.3 + 0.4 * index for index in range(composer.site_count)],
        dtype=torch.float64,
    )
    phase = torch.tensor(
        [0.35 + 0.25 * index for index in range(composer.site_count)],
        dtype=torch.float64,
    )
    return composer, dir_in, dir_out, axis, amplitude, phase


def _reference(composer, dir_in, dir_out, axis, amplitude, phase) -> torch.Tensor:
    return oracle.aspect_response(
        dir_in,
        dir_out,
        composer.inbound_row.tolist(),
        composer.outbound_row.tolist(),
        composer.response_slot.tolist(),
        axis,
        amplitude,
        phase,
        EXPONENT,
    )


def _cuda(value: torch.Tensor) -> torch.Tensor:
    return value.to(device="cuda", dtype=torch.float32).contiguous()


def _response(axis, amplitude, phase, *, requires_grad: bool = False):
    return AspectScatterResponse(
        axis=_cuda(axis).requires_grad_(requires_grad),
        amplitude=_cuda(amplitude).requires_grad_(requires_grad),
        phase_rad=_cuda(phase).requires_grad_(requires_grad),
        exponent=EXPONENT,
        coherent_interval_s=COHERENT_INTERVAL_S,
    )


def test_the_aspect_kernel_jvp_matches_finite_differences():
    """Directional derivative of every row, on all five differentiable inputs.

    One tangent per input, all non-zero at once: a companion that dropped a
    single term would still pass a one-input-at-a-time sweep whenever the other
    contributions happened to dominate.
    """

    composer, dir_in, dir_out, axis, amplitude, phase = _state()
    generator = torch.Generator().manual_seed(907)

    def tangent(shape):
        return torch.rand(shape, generator=generator, dtype=torch.float64) - 0.5

    t_dir_in = tangent(dir_in.shape)
    t_dir_out = tangent(dir_out.shape)
    t_axis = tangent(axis.shape)
    t_amplitude = tangent(amplitude.shape)
    t_phase = tangent(phase.shape)

    rows = composer.path_count
    tan_s_re = torch.empty(rows, dtype=torch.float32, device="cuda")
    tan_s_im = torch.empty_like(tan_s_re)
    _ops().scatter_response_aspect_jvp(
        _cuda(dir_in),
        _cuda(dir_out),
        composer.inbound_row,
        composer.outbound_row,
        composer.response_slot,
        _cuda(axis),
        _cuda(amplitude),
        _cuda(phase),
        torch.ones(rows, dtype=torch.int32, device="cuda"),
        _cuda(t_dir_in),
        _cuda(t_dir_out),
        _cuda(t_axis),
        _cuda(t_amplitude),
        _cuda(t_phase),
        tan_s_re,
        tan_s_im,
        EXPONENT,
        rows,
    )
    measured = torch.complex(tan_s_re.double(), tan_s_im.double()).cpu()

    def shifted(step: float) -> torch.Tensor:
        return _reference(
            composer,
            dir_in + step * t_dir_in,
            dir_out + step * t_dir_out,
            axis + step * t_axis,
            amplitude + step * t_amplitude,
            phase + step * t_phase,
        )

    expected = (shifted(FD_STEP) - shifted(-FD_STEP)) / (2.0 * FD_STEP)
    assert float(expected.abs().max()) > 1.0e-3, "the reference must not be vacuous"
    torch.testing.assert_close(measured, expected, rtol=FD_RTOL, atol=1.0e-5)


def test_the_aspect_kernel_vjp_matches_finite_differences():
    """Every gradient family, against differences of one scalar loss.

    ``L = sum_k (w_re[k] s_re[k] + w_im[k] s_im[k])`` with a fixed random
    cotangent. Differencing a per-row output directly would only check the
    Jacobian's diagonal; a scalar loss checks the CSR reduction as well, which
    is the half a per-row check cannot see.
    """

    composer, dir_in, dir_out, axis, amplitude, phase = _state()
    generator = torch.Generator().manual_seed(311)
    rows = composer.path_count
    w_re = torch.rand(rows, generator=generator, dtype=torch.float64) - 0.5
    w_im = torch.rand(rows, generator=generator, dtype=torch.float64) - 0.5

    grads = {
        "dir_in": torch.empty(dir_in.shape, dtype=torch.float32, device="cuda"),
        "dir_out": torch.empty(dir_out.shape, dtype=torch.float32, device="cuda"),
        "axis": torch.empty(axis.shape, dtype=torch.float32, device="cuda"),
        "amplitude": torch.empty(
            amplitude.shape, dtype=torch.float32, device="cuda"
        ),
        "phase": torch.empty(phase.shape, dtype=torch.float32, device="cuda"),
    }
    _ops().scatter_response_aspect_backward(
        _cuda(dir_in),
        _cuda(dir_out),
        composer.inbound_row,
        composer.outbound_row,
        composer.response_slot,
        _cuda(axis),
        _cuda(amplitude),
        _cuda(phase),
        torch.ones(rows, dtype=torch.int32, device="cuda"),
        composer.by_inbound_offsets,
        composer.by_inbound_rows,
        composer.by_outbound_offsets,
        composer.by_outbound_rows,
        composer.by_response_offsets,
        composer.by_response_rows,
        _cuda(w_re),
        _cuda(w_im),
        grads["dir_in"],
        grads["dir_out"],
        grads["axis"],
        grads["amplitude"],
        grads["phase"],
        EXPONENT,
        rows,
        composer.inbound_row_count,
        composer.outbound_row_count,
        composer.site_count,
    )

    inputs = {
        "dir_in": dir_in,
        "dir_out": dir_out,
        "axis": axis,
        "amplitude": amplitude,
        "phase": phase,
    }

    def loss(overrides) -> float:
        values = dict(inputs)
        values.update(overrides)
        response = _reference(
            composer,
            values["dir_in"],
            values["dir_out"],
            values["axis"],
            values["amplitude"],
            values["phase"],
        )
        return float((w_re * response.real + w_im * response.imag).sum())

    for name, value in inputs.items():
        analytic = grads[name].double().cpu()
        differenced = torch.zeros_like(value)
        flat = differenced.reshape(-1)
        for index in range(value.numel()):
            plus = value.clone().reshape(-1)
            minus = value.clone().reshape(-1)
            plus[index] += FD_STEP
            minus[index] -= FD_STEP
            flat[index] = (
                loss({name: plus.reshape(value.shape)})
                - loss({name: minus.reshape(value.shape)})
            ) / (2.0 * FD_STEP)
        assert float(differenced.abs().max()) > 1.0e-4, name
        torch.testing.assert_close(
            analytic, differenced, rtol=FD_RTOL, atol=1.0e-5, msg=name
        )


def _frame(composer, dir_in, dir_out):
    tau_in, _, c_in = fx.payload(composer.inbound_row_count, seed=21)
    tau_out, _, c_out = fx.payload(composer.outbound_row_count, seed=22)
    inbound = fx.leg_batch(
        tau_in.float(), c_in.to(torch.complex64), direction=_cuda(dir_in)
    )
    outbound = fx.leg_batch(
        tau_out.float(), c_out.to(torch.complex64), direction=_cuda(dir_out)
    )
    return inbound, outbound


def test_a_reverse_gradient_reaches_the_response_parameters_through_compose():
    """``_AspectResponse.backward``, reached the way production reaches it.

    The manifest names it as the end-to-end caller of
    ``scatter_response_aspect_backward``; this is that claim, executed. Each
    site's parameters own several composed rows, so a gradient that arrived
    here is also a gradient the CSR reduced.
    """

    composer, dir_in, dir_out, axis, amplitude, phase = _state()
    response = _response(axis, amplitude, phase, requires_grad=True)
    inbound, outbound = _frame(composer, dir_in, dir_out)

    composed = composer.compose(inbound, outbound, response)
    composed.complex_transfer_ref.abs().sum().backward()

    for name in ("axis", "amplitude", "phase_rad"):
        gradient = getattr(response, name).grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert float(gradient.abs().max()) > 0.0, name


def test_a_forward_tangent_on_the_aspect_axis_reaches_the_composed_transfer():
    """``_AspectResponse.jvp``, reached the way production reaches it.

    An ADR-038 forward-only dual: no ``requires_grad`` anywhere, so a facade
    that short-circuited autograd on ``requires_grad`` would silently publish a
    tangent-free result and this would catch it.
    """

    composer, dir_in, dir_out, axis, amplitude, phase = _state()
    inbound, outbound = _frame(composer, dir_in, dir_out)
    primal = _cuda(axis)
    rate = torch.zeros_like(primal)
    rate[:, 1] = 1.0

    with forward_ad.dual_level():
        dual_axis = forward_ad.make_dual(primal, rate)
        assert not dual_axis.requires_grad
        response = AspectScatterResponse(
            axis=dual_axis,
            amplitude=_cuda(amplitude),
            phase_rad=_cuda(phase),
            exponent=EXPONENT,
            coherent_interval_s=COHERENT_INTERVAL_S,
        )
        composed = composer.compose(inbound, outbound, response)
        tangent = forward_ad.unpack_dual(composed.complex_transfer_ref).tangent
        assert tangent is not None
        measured = tangent.detach().clone()

    assert torch.isfinite(measured).all()
    assert float(measured.abs().max()) > 0.0

    # And it is the right tangent: differencing the axis by the same rate
    # through the float64 reference reproduces it, up to the join's constant
    # per-row leg product.
    def response_rows(step: float) -> torch.Tensor:
        return _reference(
            composer,
            dir_in,
            dir_out,
            axis + step * rate.double().cpu(),
            amplitude,
            phase,
        )

    differenced = (response_rows(FD_STEP) - response_rows(-FD_STEP)) / (2.0 * FD_STEP)
    _, _, c_in = fx.payload(composer.inbound_row_count, seed=21)
    _, _, c_out = fx.payload(composer.outbound_row_count, seed=22)
    idx_in = composer.inbound_row.tolist()
    idx_out = composer.outbound_row.tolist()
    expected = torch.stack(
        [
            (c_out.cpu().to(torch.complex128)[idx_out[row]] * differenced[row])
            * c_in.cpu().to(torch.complex128)[idx_in[row]]
            for row in range(composer.path_count)
        ]
    )
    torch.testing.assert_close(
        measured.cpu().to(torch.complex128), expected, rtol=FD_RTOL, atol=1.0e-6
    )
