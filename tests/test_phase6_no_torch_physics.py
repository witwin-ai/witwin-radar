"""Criterion A9, statically: no Torch physics, no Dr.Jit, no silent fallback.

Three claims, each asserted over the SOURCE rather than over a run, because a
run only visits the branch it happened to take:

* T5.12 no Torch physics remains under ``solvers/``, and no ``requires_grad``
  gates a route anywhere in the Phase-6 owners;
* T5.13 waveform dispatch is a lookup on a stored discriminator, with no
  ``try``/``except``, no capability probe, and no default;
* the packaged graph names ``drjit`` nowhere.

The existing import-boundary file scans for host observation and Dr.Jit in the
spike modules. This one scans for the specific expressions plan work item 8
moved, in the specific package it moved them out of.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Torch calls that evaluate GEOMETRY or a PHASE. ``torch.fft`` is deliberately
#: absent: it is the allowlisted DSP exception and the Dirichlet solver's
#: ``ifft`` is a real production caller.
FORBIDDEN_TORCH_CALLS = (
    "cdist",
    "exp",
    "sin",
    "cos",
    "polar",
    "atan2",
    "linalg.norm",
)

#: The four Phase-6 owner packages. ``sigproc`` is NOT here: the plan's
#: Torch/DSP exception is what it exists under.
OWNER_PACKAGES = ("solvers", "synthesis", "sensors", "frontend")


def _modules(package: str) -> list[pathlib.Path]:
    root = REPO_ROOT / "witwin" / "radar" / package
    return sorted(path for path in root.rglob("*.py"))


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_the_solver_package_evaluates_no_geometry_and_no_phase_in_torch():
    """T5.12: the five migrated expressions cannot come back under a new name.

    ``torch.cdist`` was two distance fields, ``torch.linalg.norm`` was the unit
    directions the delay rate is built from, and ``torch.exp`` / ``sin`` /
    ``cos`` are a phase. All five now live in one CUDA kernel. ``torch.fft`` is
    allowlisted and is asserted to still be CALLED, so this is a scan for the
    right thing rather than a scan that passes because the package is empty.
    """

    offenders = []
    fft_callers = []
    for path in _modules("solvers"):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call):
                continue
            name = _dotted(node.func)
            if name.startswith("torch.fft."):
                fft_callers.append(path.name)
                continue
            if not name.startswith("torch."):
                continue
            if name[len("torch."):] in FORBIDDEN_TORCH_CALLS:
                offenders.append((path.name, name, node.lineno))
    assert offenders == [], offenders
    assert fft_callers, "torch.fft is the allowlisted DSP exception and is still used"


def test_no_owner_gates_a_route_on_requires_grad():
    """A route chosen by ``requires_grad`` swallows an ADR-038 forward dual.

    A forward-only dual has ``requires_grad == False``, so a branch of the form
    ``if x.requires_grad: <one route> else: <another>`` sends a tangent down the
    route that does not carry one. ``samples_require_grad`` survives as a
    predicate and may be READ; what is forbidden is an ``if`` whose test
    mentions ``requires_grad`` and whose body SELECTS something. A branch whose
    only statement is ``raise`` is the opposite of a fallback - it is the
    frontend quantiser refusing a differentiable input rather than detaching it
    - and is exempt.
    """

    offenders = []
    for package in OWNER_PACKAGES:
        for path in _modules(package):
            for node in ast.walk(_tree(path)):
                if not isinstance(node, ast.If):
                    continue
                mentioned = any(
                    isinstance(inner, ast.Attribute) and inner.attr == "requires_grad"
                    for inner in ast.walk(node.test)
                )
                refuses = all(
                    isinstance(statement, ast.Raise) for statement in node.body
                ) and not node.orelse
                if mentioned and not refuses:
                    offenders.append((package, path.name, node.lineno))
    assert offenders == [], offenders


def test_no_owner_names_drjit():
    """Zero ``drjit`` names anywhere in the production graph."""

    offenders = []
    packages = (*OWNER_PACKAGES, "propagation", "paths", "scattering", "sigproc")
    for package in packages:
        for path in _modules(package):
            if "drjit" in path.read_text(encoding="utf-8"):
                offenders.append(path.name)
    assert offenders == [], offenders


def _synthesize_source() -> ast.FunctionDef:
    tree = _tree(REPO_ROOT / "witwin" / "radar" / "radar.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "synthesize":
            return node
    raise AssertionError("Radar.synthesize must exist")


def test_waveform_dispatch_has_no_fallback_and_no_capability_probe():
    """T5.13, statically. A dict lookup on a STORED discriminator.

    Not a ``try``/``except``, because an exception handler turns a missing owner
    into a different waveform's answer. Not a ``hasattr`` probe, because a probe
    turns an unbuilt owner into silence. Not a ``dict.get``, because a default
    turns an unknown waveform into a plausible cube.
    """

    function = _synthesize_source()
    for node in ast.walk(function):
        assert not isinstance(node, ast.Try), node.lineno
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"hasattr", "getattr"}, node.lineno
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "get", node.lineno
    raises = [node for node in ast.walk(function) if isinstance(node, ast.Raise)]
    assert raises, "an unknown waveform kind must raise"


class _UnknownSystemConfig:
    """The smallest thing that can name a waveform kind with no owner."""

    kind = "chirp_sequence_from_the_future"

    def waveform_spec(self):  # pragma: no cover - the dispatch never gets here
        raise AssertionError("an unowned kind must be refused before this")


def test_an_unknown_waveform_kind_raises_at_runtime():
    """The same statement, executed. A static scan alone can be out of date."""

    from witwin.radar.radar import Radar

    radar = Radar.__new__(Radar)
    radar.system_config = _UnknownSystemConfig()
    with pytest.raises(ValueError, match="no synthesis owner"):
        radar.synthesize(object(), slow_time_mode=None)
