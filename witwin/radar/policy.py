"""What Radar refuses in AD: the wall, and the first-order-only rule.

Two rules live here, and nothing else.

**One: the non-differentiability wall.** The simulation chain and the linear
signal processing above it are differentiable. Below the first DISCRETE
DECISION nothing is, and :func:`refuse_derivative` is the single guard that
says so. The wall sits at the first discrete decision rather than at
"post-processing", which is a much later and much less defensible line: a
matched filter, a range transform, a Doppler transform, a beam cube and a MUSIC
pseudo-spectrum are all smooth functions of their input and stay live. A
threshold comparison, a ``topk`` peak pick, an ``argmax`` bin index, an
``argwhere`` detection list and a ``round`` are not, and every stage built on
one refuses at its entry.

The refusal happens BEFORE any compute, so no result object exists when it
fires. That is the point rather than a nicety: a stage that computes a full
detection list and then complains has already spent the frame, and a caller who
catches the exception is holding a half-built answer.

**Two: first order only.** :func:`first_order_only` decorates every registered
``backward`` in the package. Radar publishes first derivatives and nothing
higher. ``create_graph=True`` is precisely what leaves grad mode enabled while a
backward runs, so the check is an exact detector that fires before any launch
and names the owner that cannot answer. It also refuses a ``grad_output`` that
itself carries a derivative, which is the forward-over-reverse composition and
the worse failure of the two: without the check the mixed second derivative
comes back as an exact zero with no error at all.

This module owns both AD admission and host-parameter admission because both
rules are cross-domain. ``processing`` and ``frontend`` both need the wall;
``paths``, ``scattering``, ``sensors``, ``synthesis`` and ``frontend`` all need
the first-order rule. Putting either one inside a domain package would mean a
domain importing a sibling domain purely to reach a validator, and would end
with two copies and two wordings - which is exactly the state the guard exists
to remove.

Both refusals name the owner, and both say WHY the derivative does not exist,
because "not differentiable" alone sends the reader looking for a bug rather
than for the modelling decision.
``docs/dev/radar-ad-capability-matrix.md`` carries the same reasons as ``REF``
rows.
"""

from __future__ import annotations

import functools

import torch


def _carries_derivative(value: object) -> str | None:
    """``"a gradient"`` / ``"a forward tangent"`` / ``None`` for one value.

    Both AD modes are checked, always. Checking only ``requires_grad`` is the
    defect this module was written to close: a forward dual carries a live
    tangent through a stage that has no business publishing one, and
    ``requires_grad`` is ``False`` the whole time.
    """

    if not isinstance(value, torch.Tensor):
        return None
    if value.requires_grad:
        return "a gradient"
    if torch.autograd.forward_ad.unpack_dual(value).tangent is not None:
        return "a forward tangent"
    return None


def refuse_derivative(stage: str, reason: str, **tensors: object) -> None:
    """Refuse a differentiable input to a stage below the wall, at its entry.

    ``stage`` is the dotted owner - module and function - so the message names
    the code that cannot answer rather than the layer that noticed. ``reason``
    is the modelling statement: which discrete decision the stage is built on.
    ``tensors`` are the stage's inputs by name; non-tensors are ignored, so a
    caller can hand the whole argument set without filtering it.

    Checked in the order given, so the message always names the first offender
    in signature order rather than in dict-iteration luck.
    """

    for name, value in tensors.items():
        carrier = _carries_derivative(value)
        if carrier is None:
            continue
        raise RuntimeError(
            f"{stage} is not differentiable and {name} carries {carrier}: "
            f"{reason} This stage ships no backward and no jvp, so the "
            "derivative it would publish is the derivative of a value at a "
            "frozen discrete choice - a plausible number describing the wrong "
            "function. A straight-through or soft surrogate is an explicit "
            "modelling decision with its own design, not something this stage "
            "may choose. Detach at the call site, deliberately, so the "
            "severing is visible where it is made."
        )


def first_order_only(backward):
    """The one backward decorator: first-order guard over ``once_differentiable``.

    Radar publishes first derivatives only, in both modes, and every
    second-order request fails loudly before any partial second-order result.
    Three compositions, and what happens to each:

    * **reverse over reverse** (``create_graph=True``): grad mode is left
      enabled while the backward runs, which is what the check below detects.
      Without it the first gradient comes back silently detached and the
      failure surfaces one step later as a generic Torch message that names
      Torch rather than the owner - or, with ``allow_unused=True``, as a silent
      ``None``.
    * **forward over reverse**: the ``grad_output`` carries a forward tangent.
      This is the worse of the two, because today it does not fail at all: the
      gradient value is right, its tangent is ``None``, and the mixed second
      derivative reads as an exact zero.
    * a ``grad_output`` that itself ``requires_grad``, which is the same request
      spelled differently and is equally unanswerable.

    ``torch.autograd.function.once_differentiable`` is applied UNDERNEATH, as
    defence in depth, from here rather than from every call site. It cannot
    replace the check and the nesting order is not cosmetic: ``once`` runs the
    backward body inside ``torch.no_grad()``, so a check written inside the body
    sees grad mode already off even under ``create_graph=True``. Measured, not
    assumed.
    """

    once = torch.autograd.function.once_differentiable(backward)
    owner = f"{backward.__module__}.{backward.__qualname__}"

    @functools.wraps(backward)
    def guarded(ctx, *grad_outputs):
        if torch.is_grad_enabled():
            raise NotImplementedError(
                f"{owner} is first-order only: Radar does not support "
                "higher-order AD. This is a create_graph=True (grad-of-grad) "
                "request, and it fails here rather than returning a detached "
                "first gradient whose second derivative would come back None "
                "with nothing to say so."
            )
        for index, grad_output in enumerate(grad_outputs):
            carrier = _carries_derivative(grad_output)
            if carrier is None:
                continue
            raise NotImplementedError(
                f"{owner} is first-order only: Radar does not support "
                f"higher-order AD. Cotangent {index} carries {carrier}, which "
                "is a second-order request - a forward tangent through a "
                "backward is a mixed second derivative, and this family would "
                "otherwise publish it as an exact zero with no error at all."
            )
        return once(ctx, *grad_outputs)

    return guarded


def require_host_float(name: str, value: object, *, owner: str, reason: str) -> None:
    """Refuse a tensor where a host float is the contract.

    ``name`` is the field, ``owner`` the class that declares it, and ``reason``
    the modelling statement that says why no derivative flows through it. All
    three appear in the message.
    """

    if not isinstance(value, torch.Tensor):
        return
    carrier = "a requires_grad tensor" if value.requires_grad else f"a {tuple(value.shape)} torch.Tensor"
    raise TypeError(
        f"{owner}.{name} must be a host float and got {carrier}. {reason} There "
        "is no tangent or gradient slot for it in either AD mode, so a tensor "
        "handed here would be flattened by float() and the derivative the "
        "caller asked for would come back None with nothing to say so. Pass "
        "the number; if this value has to be optimised, that is a new "
        "capability with its own decision rather than something a spec may "
        "accept quietly."
    )


def require_host_floats(owner: str, reason: str, **fields: object) -> None:
    """:func:`require_host_float` over one owner's whole field set.

    One call per ``__post_init__`` with one reason for the whole spec, because
    the reason is a property of the OWNER - a waveform declaration, a device
    description, a unit convention - and repeating it per field would invite
    the reasons to drift apart.

    Fields are checked in the order given, so the message always names the
    first offender in declaration order rather than in dict-iteration luck.
    """

    for name, value in fields.items():
        require_host_float(name, value, owner=owner, reason=reason)


__all__ = ["first_order_only", "refuse_derivative", "require_host_float", "require_host_floats"]
