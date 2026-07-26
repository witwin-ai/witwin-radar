"""AD companions of the fused TDM-frame MIMO launch (plan work item 8, A9).

``forward_mimo_linear_chunked`` had no registered backward and no registered
jvp, so ``DirichletSolver`` dispatched a reverse-mode call to a Torch
expression that re-derived ``dist = d0 + rate * t`` and the ``d0 / dist``
range-loss update in a different dtype, a different reduction order, and a
different memory shape. A gradient that comes from different code than the
value is not the gradient of the value, and criterion A9 forbids it. This file
pins the two companions that let that replay be deleted.

The oracle is a float64 closed form of the kernel's own documented expression,
written here in Torch and used three ways - primal, VJP by ``torch.autograd``,
and JVP by central finite differences - so one reference statement covers both
companions. It is independent of ``witwin.radar.synthesis.dirichlet_spectrum``:
it imports nothing from the module it validates.

``fc`` is 0 in every case below, which is the production carrier home for a
weight that already carries the reference phase. It is also what makes a
float32 kernel comparable to a float64 oracle at all: at ``fc = 77 GHz`` the
absolute phase ``2 pi fc tau`` is of order ``1e4`` radians and float32 resolves
it to about a milliradian, which swamps every derivative term this file is
trying to measure. The carrier home is exercised separately, in
``test_phase6_dirichlet_complex.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.radar.synthesis.dirichlet_spectrum import (
    DirichletSpectrumSpec,
    MimoLinearFramePlan,
    mimo_linear_spectra,
)

pytestmark = pytest.mark.gpu


NUM_BINS = 24
N_FFT = 48
NUM_TX = 2
NUM_RX = 2
NUM_PAIRS = NUM_TX * NUM_RX
TARGETS_PER_PAIR = 3
CHIRPS = 4
CHIRP_PERIOD_S = 6.0e-5
SLOPE_HZ_PER_S = 5.0e13
SAMPLE_RATE_HZ = 5.0e6
T_START_S = 6.0e-6
K0_PER_METER = (SLOPE_HZ_PER_S * 2.0 / 299792458.0) * N_FFT / SAMPLE_RATE_HZ
N_HALF = (NUM_BINS - 1) / 2


def _spec() -> DirichletSpectrumSpec:
    return DirichletSpectrumSpec(
        n=N_HALF,
        k0_per_meter=K0_PER_METER,
        num_bins=NUM_BINS,
        n_fft=N_FFT,
        fc=0.0,
        slope_hz_per_s=SLOPE_HZ_PER_S,
        t_start_s=T_START_S,
        tau_is_seconds=0,
    )


def _plan(*, range_loss_update: bool) -> MimoLinearFramePlan:
    return MimoLinearFramePlan(
        targets_per_pair=TARGETS_PER_PAIR,
        num_pairs=NUM_PAIRS,
        chirp_per_frame=CHIRPS,
        chirp_period_s=CHIRP_PERIOD_S,
        num_tx=NUM_TX,
        range_loss_update=range_loss_update,
    )


def reference_mimo_linear(
    d0: torch.Tensor,
    d_rate: torch.Tensor,
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    *,
    range_loss_update: bool,
) -> torch.Tensor:
    """float64 closed form of ``forward_mimo_linear_chunked``, in Torch.

    Written from the kernel's header comment, not from its code, and holding no
    import of the module under test. Shapes follow the kernel exactly:
    ``(chirps, pairs, bins)`` with ``slot = chirp * num_tx + tx(pair)``.
    """

    rows = d0.reshape(NUM_PAIRS, TARGETS_PER_PAIR)
    rates = d_rate.reshape(NUM_PAIRS, TARGETS_PER_PAIR)
    weight = torch.complex(
        a_re.reshape(NUM_PAIRS, TARGETS_PER_PAIR),
        a_im.reshape(NUM_PAIRS, TARGETS_PER_PAIR),
    )
    tx_of_pair = torch.arange(NUM_PAIRS, dtype=torch.float64) // NUM_RX
    chirp_ids = torch.arange(CHIRPS, dtype=torch.float64).view(-1, 1)
    slot = chirp_ids * NUM_TX + tx_of_pair.view(1, -1)
    chirp_time = (slot * CHIRP_PERIOD_S).unsqueeze(-1)  # (chirps, pairs, 1)

    dist = rows.unsqueeze(0) + rates.unsqueeze(0) * chirp_time
    amp = weight.unsqueeze(0).expand_as(dist).to(torch.complex128)
    if range_loss_update:
        amp = amp * (rows.unsqueeze(0) / torch.clamp(dist, min=1e-6)).to(torch.complex128)

    tau = 2.0 * dist / 299792458.0
    phi0 = 2.0 * math.pi * (SLOPE_HZ_PER_S * tau * (T_START_S - 0.5 * tau))
    k0 = dist * K0_PER_METER
    bins = torch.arange(NUM_BINS, dtype=torch.float64)
    x = 2.0 * math.pi * (bins.view(1, 1, 1, -1) - k0.unsqueeze(-1)) / N_FFT

    half = torch.sin(0.5 * x)
    scale = torch.where(
        half.abs() < 1e-9,
        torch.full_like(half, 2.0 * N_HALF + 1.0),
        torch.sin((N_HALF + 0.5) * x) / torch.where(half.abs() < 1e-9, torch.ones_like(half), half),
    )
    response = scale.to(torch.complex128) * torch.exp(-1j * N_HALF * x.to(torch.complex128))
    response = response * torch.exp(1j * phi0.unsqueeze(-1).to(torch.complex128))
    return (amp.unsqueeze(-1) * response).sum(dim=2)


def _inputs(device: str = "cuda", *, seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    rows = NUM_PAIRS * TARGETS_PER_PAIR
    d0 = (2.0 + 4.0 * torch.rand(rows, generator=generator, dtype=torch.float64))
    d_rate = (torch.rand(rows, generator=generator, dtype=torch.float64) - 0.5) * 20.0
    a_re = torch.randn(rows, generator=generator, dtype=torch.float64)
    a_im = torch.randn(rows, generator=generator, dtype=torch.float64)
    return d0, d_rate, a_re, a_im


def _native(d0, d_rate, a_re, a_im, *, range_loss_update: bool) -> torch.Tensor:
    return mimo_linear_spectra(
        d0,
        d_rate,
        a_re,
        a_im,
        spec=_spec(),
        plan=_plan(range_loss_update=range_loss_update),
    )


def _to_cuda(*tensors):
    return tuple(t.to(dtype=torch.float32, device="cuda").contiguous() for t in tensors)


@pytest.mark.parametrize("range_loss_update", [False, True])
def test_the_fused_frame_matches_its_float64_closed_form(range_loss_update):
    """The primal first: a companion of the wrong forward proves nothing."""

    d0, d_rate, a_re, a_im = _inputs()
    native = _native(*_to_cuda(d0, d_rate, a_re, a_im), range_loss_update=range_loss_update)
    reference = reference_mimo_linear(
        d0, d_rate, a_re, a_im, range_loss_update=range_loss_update
    )
    peak = reference.abs().max().item()
    deviation = (native.cpu().to(torch.complex128) - reference).abs().max().item()
    assert deviation < 1e-4 * peak, (deviation, peak)


@pytest.mark.parametrize("range_loss_update", [False, True])
def test_the_backward_matches_the_float64_reference_gradients(range_loss_update):
    """VJP against ``torch.autograd`` on the independent float64 oracle.

    All four leaves are checked in one pass with one cotangent, because a
    per-leaf check with a per-leaf cotangent can pass while the leaves are
    swapped.
    """

    d0, d_rate, a_re, a_im = _inputs()
    cot = torch.randn(CHIRPS, NUM_PAIRS, NUM_BINS, dtype=torch.float64)

    leaves = [t.clone().requires_grad_(True) for t in (d0, d_rate, a_re, a_im)]
    out = reference_mimo_linear(*leaves, range_loss_update=range_loss_update)
    (out.real * cot).sum().backward()
    expected = [leaf.grad.clone() for leaf in leaves]

    native_leaves = [
        t.to(dtype=torch.float32, device="cuda").contiguous().requires_grad_(True)
        for t in (d0, d_rate, a_re, a_im)
    ]
    native = _native(*native_leaves, range_loss_update=range_loss_update)
    (native.real * cot.to(dtype=torch.float32, device="cuda")).sum().backward()

    for name, leaf, want in zip(("d0", "d_rate", "a_re", "a_im"), native_leaves, expected):
        got = leaf.grad.cpu().to(torch.float64)
        scale = want.abs().max().item()
        deviation = (got - want).abs().max().item()
        assert deviation < 2e-3 * scale, (name, deviation, scale)


def _directional_tangent(primals, index, direction, *, range_loss_update: bool):
    """The native jvp's tangent along ONE input direction."""

    from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

    with dual_level():
        duals = list(primals)
        duals[index] = make_dual(primals[index], direction)
        cube = _native(*duals, range_loss_update=range_loss_update)
        tangent = unpack_dual(cube).tangent
    assert tangent is not None
    return tangent


# d0 and d_rate move on very different scales - a metre-sized distance against
# a metres-per-second rate multiplied by a 60 microsecond slot time - so the
# step is chosen per variable and recorded here rather than shared.
# Each step is chosen so that the perturbation moves the ramp phase
# 2 pi slope tau (t_start - tau/2) by about 0.025 rad: large enough that the
# float32 difference is not noise, small enough that the central difference's
# own O(h^2) error stays under 1e-4 relative. A slot time of order 4e-4 s is
# what turns a rate step of 5 m/s into the same 2e-3 m as the distance step.
_FD_STEPS = {"d0": 2e-3, "d_rate": 5.0, "a_re": 1e-2, "a_im": 1e-2}


@pytest.mark.parametrize("range_loss_update", [False, True])
@pytest.mark.parametrize("index,name", list(enumerate(("d0", "d_rate", "a_re", "a_im"))))
def test_the_jvp_matches_central_finite_differences(index, name, range_loss_update):
    """Forward mode against central FD of the SAME kernel, one leaf at a time.

    FD is a test oracle only. It is taken of the native forward rather than of
    the float64 reference so that the comparison isolates the tangent rule from
    the float32 evaluation both sides share. Per leaf rather than all four at
    once, because a combined direction lets a wrong term hide behind a right
    one.
    """

    primals = _to_cuda(*_inputs())
    generator = torch.Generator(device="cuda").manual_seed(11)
    direction = torch.randn(
        primals[0].shape, generator=generator, device="cuda", dtype=torch.float32
    )
    analytic = _directional_tangent(
        primals, index, direction, range_loss_update=range_loss_update
    )

    step = _FD_STEPS[name]
    shifted = list(primals)
    shifted[index] = primals[index] + direction * step
    plus = _native(*shifted, range_loss_update=range_loss_update)
    shifted[index] = primals[index] - direction * step
    minus = _native(*shifted, range_loss_update=range_loss_update)
    fd = (plus - minus) / (2.0 * step)

    scale = fd.abs().max().item()
    deviation = (analytic - fd).abs().max().item()
    assert deviation < 2e-3 * scale, (name, deviation, scale)


@pytest.mark.parametrize("range_loss_update", [False, True])
def test_the_backward_is_the_exact_adjoint_of_the_forward_mode(range_loss_update):
    """``<cot, JVP(tan)> == <VJP(cot), tan>``, in the kernels' own arithmetic.

    This is the statement finite differences cannot make: it is exact rather
    than approximate, and it fails for every sign flip, transposed index, or
    dropped term, in either companion.
    """

    from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

    d0, d_rate, a_re, a_im = _to_cuda(*_inputs())
    generator = torch.Generator(device="cuda").manual_seed(3)
    tangents = [
        torch.randn(d0.shape, generator=generator, device="cuda", dtype=torch.float32)
        for _ in range(4)
    ]
    cot_re = torch.randn(
        CHIRPS, NUM_PAIRS, NUM_BINS, generator=generator, device="cuda", dtype=torch.float32
    )
    cot_im = torch.randn(
        CHIRPS, NUM_PAIRS, NUM_BINS, generator=generator, device="cuda", dtype=torch.float32
    )

    with dual_level():
        duals = [
            make_dual(primal, tangent)
            for primal, tangent in zip((d0, d_rate, a_re, a_im), tangents)
        ]
        cube = _native(*duals, range_loss_update=range_loss_update)
        tangent_out = unpack_dual(cube).tangent
    forward_side = (cot_re * tangent_out.real + cot_im * tangent_out.imag).sum()

    leaves = [primal.clone().requires_grad_(True) for primal in (d0, d_rate, a_re, a_im)]
    native = _native(*leaves, range_loss_update=range_loss_update)
    (native.real * cot_re + native.imag * cot_im).sum().backward()
    reverse_side = sum(
        (leaf.grad * tangent).sum() for leaf, tangent in zip(leaves, tangents)
    )

    scale = max(abs(forward_side.item()), abs(reverse_side.item()))
    deviation = abs(forward_side.item() - reverse_side.item())
    assert deviation < 1e-5 * scale, (deviation, scale)


def test_the_frame_path_has_no_requires_grad_branch():
    """A9, statically: the route a gradient takes is the route the value took.

    The deleted replay was reachable only through ``if x.requires_grad``. This
    asserts the solver has no such branch left, so there is no second frame
    implementation for a gradient to fall into.
    """

    import ast
    import inspect

    from witwin.radar.solvers import solver_dirichlet

    tree = ast.parse(inspect.getsource(solver_dirichlet))
    attributes = [
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    ]
    assert "requires_grad" not in attributes
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "_mimo_from_path_tensors_linear_autograd" not in names
    assert "_total_path_length_rates" not in names
