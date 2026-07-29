"""No second-order AD anywhere in Radar, and every request fails loudly.

Phase-9 item 5. Before this file there was no explicit higher-order rejection
in the radar package at all, and the three ways of asking produced three
different silences:

* ``grad(..., create_graph=True)`` SUCCEEDED. The returned gradient carried
  ``requires_grad=True`` and an autograd ``Error`` node, so the failure arrived
  one step later as ``RuntimeError: One of the differentiated Tensors appears
  to not have been used in the graph`` - a message naming Torch, not the owner
  that cannot answer - and with ``allow_unused=True`` it came back as a silent
  ``None``.
* a ``grad_output`` carrying a forward tangent was accepted and the mixed
  second derivative was published as an exact zero with no error. This is the
  worse of the two: a wrong answer that looks like a correct one.
* a ``grad_output`` that itself required grad was accepted as well.

The rule matches the one Channel landed in ADR-043 (S0), deliberately, because
two conventions for one question is how a caller ends up believing a mixed
request means different things on either side of the boundary:

* **reverse over reverse** is detected by ``torch.is_grad_enabled()`` inside the
  backward, which ``create_graph=True`` is precisely what leaves on;
* **forward over reverse** is detected on the cotangents;
* **``jvp`` plus ``requires_grad`` is NOT refused.** S0 verified that a forward
  dual built on a ``requires_grad`` primal is a legitimate FIRST-order request
  under ADR-038 - the two modes agree bit for bit and one ``Function`` serves
  both - and dropped that half of the rule rather than break the Channel test
  that uses exactly that shape. Radar follows.
* **nested forward levels stay Torch-owned.** Torch raises its own error and it
  is pinned here rather than wrapped.

The nesting order of the two decorators is load bearing and was measured, not
reasoned about: ``once_differentiable`` runs the backward body inside
``torch.no_grad()``, so a grad-mode check written INSIDE the body sees grad mode
already off even under ``create_graph=True``. The guard therefore wraps
``once_differentiable`` from outside, and the last test in this file pins that
ordering by measuring it on a toy ``Function``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch
import torch.autograd.forward_ad as forward_ad
from support import ad_boundaries as ab

#: Only the per-boundary tests need a device. The structural scan, the
#: Torch-owned nested-forward pin and the decorator-ordering measurement are
#: contract questions and run in the default suite, where a regression in the
#: guard shows up without a GPU.
gpu = pytest.mark.gpu


#: Every registered autograd ``Function`` in the package and the file that owns
#: it. Written out rather than discovered so that a new ``Function`` added
#: without a first-order decision fails the structural test below.
REGISTERED_BACKWARDS = {
    "witwin/radar/paths.py": 1,
    "witwin/radar/scattering.py": 1,
    "witwin/radar/sensors.py": 1,
    "witwin/radar/synthesis/fmcw.py": 1,
    "witwin/radar/synthesis/ofdm.py": 1,
    "witwin/radar/synthesis/pulsed.py": 1,
    "witwin/radar/frontend.py": 2,
}


def _radar_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "witwin" / "radar"


# ---------------------------------------------------------------------------
# 1. Structural: one owner, applied at every site
# ---------------------------------------------------------------------------


def test_every_registered_backward_is_decorated_by_the_one_owner():
    """Eight backwards, one decorator, no bare ``once_differentiable`` left.

    There were ten until Phase 11 deleted the two ``dirichlet_spectrum``
    contexts with their route. The per-boundary tests below drive six of the
    eight through a real call. This is what covers the other two and, more
    importantly, what fails when a ninth ``Function`` is added: a new backward
    with no decorator is a new
    silent second-order hole, and it would otherwise be invisible until someone
    asked for a grad of a grad.
    """

    total_functions = 0
    for relative, expected in REGISTERED_BACKWARDS.items():
        source = (_radar_root().parent.parent / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(isinstance(base, ast.Attribute) and base.attr == "Function" for base in node.bases)
        ]
        assert len(functions) == expected, (relative, len(functions))
        total_functions += len(functions)
        for function in functions:
            backwards = [
                node for node in function.body if isinstance(node, ast.FunctionDef) and node.name == "backward"
            ]
            assert len(backwards) == 1, (relative, function.name)
            names = {decorator.id for decorator in backwards[0].decorator_list if isinstance(decorator, ast.Name)}
            assert "first_order_only" in names, (relative, function.name)
            # The raw decorator cannot be left beside it: applying both would
            # put the grad-mode check inside the no_grad body and disarm it.
            assert "once_differentiable" not in names, (relative, function.name)
        assert "once_differentiable" not in source.replace("``once_differentiable``", ""), relative
    assert total_functions == 8, total_functions


def test_the_package_names_no_second_higher_order_rule():
    """One implementation, in one file, imported by everyone else."""

    from witwin.radar import policy

    assert policy.first_order_only.__module__ == "witwin.radar.policy"
    owners = [
        path for path in _radar_root().rglob("*.py") if "def first_order_only" in path.read_text(encoding="utf-8")
    ]
    assert [path.name for path in owners] == ["policy.py"], owners


# ---------------------------------------------------------------------------
# 2. Per boundary: reverse over reverse
# ---------------------------------------------------------------------------


@gpu
@pytest.mark.parametrize("name", ab.BOUNDARY_NAMES)
def test_a_grad_of_grad_request_fails_loudly_and_names_the_owner(name):
    """``create_graph=True`` raises ``NotImplementedError`` at the owner.

    Before Phase 9 this call SUCCEEDED at every one of these boundaries and
    handed back a detached gradient, so the request that could not be answered
    was answered and the failure surfaced one step later somewhere else.
    """

    boundary = ab.boundary(name)
    leaf = boundary.leaf.detach().clone().requires_grad_(True)
    loss = boundary.loss(leaf)

    with pytest.raises(NotImplementedError, match="first-order only") as raised:
        torch.autograd.grad(loss, leaf, create_graph=True)
    message = str(raised.value)
    assert boundary.owner in message, message
    assert "witwin.radar." in message


@gpu
@pytest.mark.parametrize("name", ab.BOUNDARY_NAMES)
def test_the_first_order_request_over_the_same_graph_still_works(name):
    """The refusal is of the SECOND order request, not of the first.

    Asserted per boundary and immediately after the refusal above, because a
    guard placed one level too high would refuse everything and every test in
    this file would still pass.
    """

    boundary = ab.boundary(name)
    leaf = boundary.leaf.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(boundary.loss(leaf), leaf)
    assert grad is not None
    assert torch.isfinite(grad.view(torch.float32) if grad.is_complex() else grad).all()
    assert float(grad.abs().sum()) > 0.0, name


# ---------------------------------------------------------------------------
# 3. Per boundary: forward over reverse, and a live cotangent
# ---------------------------------------------------------------------------


@gpu
@pytest.mark.parametrize("name", ab.BOUNDARY_NAMES)
def test_a_cotangent_carrying_a_forward_tangent_is_refused(name):
    """The worst silence of the three: a mixed second derivative read as zero.

    Handing a dual-carrying ``grad_output`` into a backward is a mixed second
    derivative request. Every family accepted it, computed the correct FIRST
    derivative, and published a tangent of ``None`` - which a caller reads as an
    exact zero mixed partial, with no warning anywhere.
    """

    boundary = ab.boundary(name)
    leaf = boundary.leaf.detach().clone().requires_grad_(True)
    loss = boundary.loss(leaf)
    with forward_ad.dual_level():
        cotangent = forward_ad.make_dual(torch.ones_like(loss.detach()), torch.ones_like(loss.detach()))
        with pytest.raises(NotImplementedError, match="first-order only") as raised:
            torch.autograd.grad(loss, leaf, grad_outputs=cotangent)
    assert "a forward tangent" in str(raised.value)
    assert boundary.owner in str(raised.value)


def test_a_cotangent_that_itself_requires_grad_is_refused():
    """The same request spelled differently, and equally unanswerable.

    Asserted on a toy ``Function`` rather than per boundary, and the reason is a
    measurement rather than convenience: with ``create_graph=False`` Torch runs
    every intermediate backward under ``no_grad``, so a ``requires_grad``
    cotangent handed to a COMPOSED loss arrives at the innermost backward
    already stripped. The second half of this test measures exactly that. The
    shape therefore only reaches a production backward when the boundary IS the
    output - which is what is checked here - or under ``create_graph=True``,
    which the grad-mode branch already refuses at every boundary above.
    """

    from witwin.radar.policy import first_order_only

    arrived: list[bool] = []

    class Doubler(torch.autograd.Function):
        @staticmethod
        def forward(x):
            return x * 2

        @staticmethod
        def setup_context(ctx, inputs, output):
            pass

        @staticmethod
        @first_order_only
        def backward(ctx, grad_out):
            arrived.append(grad_out.requires_grad)
            return grad_out * 2

    leaf = torch.tensor([1.0], requires_grad=True)
    direct = Doubler.apply(leaf)
    cotangent = torch.ones_like(direct.detach()).requires_grad_(True)
    with pytest.raises(NotImplementedError, match="first-order only") as raised:
        torch.autograd.grad(direct, leaf, grad_outputs=cotangent, retain_graph=True)
    assert "a gradient" in str(raised.value)
    assert arrived == []

    # The measurement: one intermediate Torch operation is enough for Torch to
    # strip the cotangent's own graph before it reaches the boundary.
    composed = direct.square().sum()
    torch.autograd.grad(composed, leaf, grad_outputs=torch.ones_like(composed.detach()).requires_grad_(True))
    assert arrived == [False], arrived


@gpu
@pytest.mark.parametrize("name", ab.BOUNDARY_NAMES)
def test_no_gradient_survives_the_refusal(name):
    """Fail before a partial second-order result, not merely fail.

    ``.backward()`` accumulates into ``leaf.grad`` as a side effect, so the
    refusal has to leave that slot untouched: a caller who catches the exception
    must not find a half-finished second-order gradient waiting in it.
    """

    boundary = ab.boundary(name)
    leaf = boundary.leaf.detach().clone().requires_grad_(True)
    loss = boundary.loss(leaf)
    with pytest.raises(NotImplementedError):
        loss.backward(create_graph=True)
    assert leaf.grad is None


# ---------------------------------------------------------------------------
# 4. Torch-owned: nested forward levels
# ---------------------------------------------------------------------------


def test_nested_forward_levels_stay_torch_owned():
    """Pinned, not wrapped, and the ownership is stated.

    Torch refuses a second ``dual_level`` itself. Radar adds nothing here, and
    this test exists so that the absence of a Radar-owned message is a recorded
    decision rather than a gap in the wall.
    """

    with forward_ad.dual_level():
        with pytest.raises(RuntimeError, match="Nested forward mode AD"):
            with forward_ad.dual_level():
                pass


# ---------------------------------------------------------------------------
# 5. The decorator ordering, measured
# ---------------------------------------------------------------------------


def test_once_differentiable_cannot_replace_the_grad_mode_check():
    """Why the guard wraps ``once_differentiable`` and not the other way round.

    ``once_differentiable`` runs the backward body inside ``torch.no_grad()``.
    A grad-mode check written INSIDE the body therefore sees ``False`` even
    under ``create_graph=True`` - measured here on a toy ``Function`` - so the
    check has to sit outside it. Getting this backwards would produce a guard
    that never fires and a test suite that never notices.
    """

    observed: list[bool] = []

    class Inner(torch.autograd.Function):
        @staticmethod
        def forward(x):
            return x * 2

        @staticmethod
        def setup_context(ctx, inputs, output):
            pass

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_out):
            observed.append(torch.is_grad_enabled())
            return grad_out * 2

    leaf = torch.tensor([1.0], requires_grad=True)
    out = Inner.apply(leaf)
    torch.autograd.grad(out, leaf, grad_outputs=torch.ones_like(out), retain_graph=True)
    torch.autograd.grad(out, leaf, grad_outputs=torch.ones_like(out), create_graph=True, retain_graph=True)
    # Both calls, including the create_graph one, saw grad mode OFF.
    assert observed == [False, False], observed

    # And the owner, wrapping from outside, sees it ON for the second.
    from witwin.radar.policy import first_order_only

    seen: list[bool] = []

    class Outer(torch.autograd.Function):
        @staticmethod
        def forward(x):
            return x * 2

        @staticmethod
        def setup_context(ctx, inputs, output):
            pass

        @staticmethod
        @first_order_only
        def backward(ctx, grad_out):
            seen.append(torch.is_grad_enabled())
            return grad_out * 2

    leaf = torch.tensor([1.0], requires_grad=True)
    out = Outer.apply(leaf)
    torch.autograd.grad(out, leaf, grad_outputs=torch.ones_like(out), retain_graph=True)
    with pytest.raises(NotImplementedError, match="first-order only"):
        torch.autograd.grad(out, leaf, grad_outputs=torch.ones_like(out), create_graph=True, retain_graph=True)
    # The body ran once, for the first-order call, and never for the second.
    assert seen == [False], seen
