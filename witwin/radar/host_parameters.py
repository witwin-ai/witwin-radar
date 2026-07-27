"""One rule for every configuration scalar: it is a host float, not a tensor.

A waveform spec, a frontend stage description and a Dirichlet plan are all
DECLARATIONS. They select a waveform, describe a device, or name a unit
convention; none of them is scene state. Every one of their scalar fields is
read on the host - ``float(self.gain_db)``, ``10.0 ** (x / 20.0)``,
``math.sqrt(...)`` - and handed to a kernel by value.

Until Phase 9 nothing said so. A caller could pass a 0-dim ``requires_grad``
tensor into any of them, ``float(...)`` would silently strip it, the whole
chain would run, and the caller would get ``grad = None`` back for the
parameter they were optimising. That is the exact defect shape this phase
exists to remove: not a wrong number, a MISSING derivative that looks like a
successful run.

**This refuses ANY tensor, not only a grad-carrying one, and that is
deliberate.** A tensor spec field that happens not to require grad today is
precisely the input that starts requiring grad tomorrow, at which point the
failure is silent again. ``float()`` on a device tensor is also a host
synchronisation, on a per-frame object, which no spec may hide. Refusing the
type rather than the flag is the only version of this rule that stays true.

The refusal names the field, its owner, and WHY that owner has no derivative
slot, because "must be a float" alone sends the reader looking for a type bug
rather than for the modelling decision. ``docs/dev/radar-ad-capability-matrix.md``
carries the same reasons as ``REF`` rows.

The one configuration scalar that is NOT here is a radar cross section:
:meth:`witwin.radar.scattering.ScalarRcsResponse.from_rcs` accepts a tensor
``sigma_m2`` on purpose, because a cross section is scene state and the
canonical inverse-design leaf. That asymmetry is the point of stating the rule
explicitly instead of leaving it to ``float()``.
"""

from __future__ import annotations

import torch


def require_host_float(name: str, value: object, *, owner: str, reason: str) -> None:
    """Refuse a tensor where a host float is the contract.

    ``name`` is the field, ``owner`` the class that declares it, and ``reason``
    the modelling statement that says why no derivative flows through it. All
    three appear in the message.
    """

    if not isinstance(value, torch.Tensor):
        return
    carrier = (
        "a requires_grad tensor"
        if value.requires_grad
        else f"a {tuple(value.shape)} torch.Tensor"
    )
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


__all__ = ["require_host_float", "require_host_floats"]
