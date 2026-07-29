"""Criterion A9, statically: no Torch physics, no Dr.Jit, no silent fallback.

Four claims, each asserted over the SOURCE rather than over a run, because a
run only visits the branch it happened to take:

* T5.12 no Torch physics remains in the Phase-6 owner packages, and no
  ``requires_grad`` gates a route in any of them;
* T5.13 waveform dispatch is a lookup on a stored discriminator, with no
  ``try``/``except``, no capability probe, and no default;
* the packaged graph names ``drjit`` nowhere;
* every surviving Torch physics expression in the facade ``radar.py`` is
  RECORDED, one by one, and the record cannot grow.

The existing import-boundary file scans for host observation and Dr.Jit in the
spike modules. This one scans for the specific expressions plan work item 8
moved, in the specific package it moved them out of.

Work item 8 named two sources, ``solvers/common.py`` and ``radar.py``, and the
scan originally covered only the packages. That was the wrong half: the module
the item named by hand was where the Torch chirp expression lived
(``Radar.waveform``), so the guard did not look where the survivor was. The
facade scan below closes that hole. Phase 11 deleted both of work item 8's
named sources - ``solvers/`` with the Dirichlet route and ``Radar.waveform``
with it - so the facade record below is now the whole of what is left.

**Phase 9 extends the same discipline to what Phase 9 itself added.** The phase
put roughly a thousand lines of guard and orchestration into the production
graph - a first-order-only decorator, a wall of refusals, a host-float
validator, a velocity-leaf refusal, an SMPL deformation refusal - and every one
of them is new Torch in a package this file is supposed to police. The last
three sections record what was added, in exactly two categories:

* **refusal predicates**, which ASK Torch a question and never construct a
  value: ``is_grad_enabled``, ``unpack_dual``, ``once_differentiable``. A guard
  module that started computing something would show up as a call outside that
  set.
* **result construction**, which is one expression: ``rcs_amplitude``'s
  ``torch.sqrt``. It runs once per response rather than once per path, and
  every per-path product downstream of it is still a native kernel, which is why
  the capability matrix records its mechanism as ``torch-orchestration`` and not
  as physics.

There is no third category, and the tests below fail if one appears.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Torch calls that evaluate GEOMETRY or a PHASE. ``torch.fft`` is deliberately
#: absent: it is the allowlisted DSP exception, and the range and Doppler
#: transforms in ``processing/`` are real production callers.
FORBIDDEN_TORCH_CALLS = ("cdist", "exp", "sin", "cos", "polar", "atan2", "linalg.norm")

#: The four Phase-6 owner packages. ``sigproc`` is NOT here: the plan's
#: Torch/DSP exception is what it exists under.
OWNER_PACKAGES = ("synthesis", "sensors", "frontend")


def _modules(package: str) -> list[pathlib.Path]:
    root = REPO_ROOT / "witwin" / "radar" / package
    if root.is_dir():
        return sorted(root.rglob("*.py"))
    module = root.with_suffix(".py")
    return [module] if module.is_file() else []


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


def test_the_migrated_expressions_did_not_come_back_under_another_owner():
    """T5.12: the five migrated expressions cannot come back under a new name.

    ``torch.cdist`` was two distance fields, ``torch.linalg.norm`` was the unit
    directions the delay rate is built from, and ``torch.exp`` / ``sin`` /
    ``cos`` are a phase. All five now live in one CUDA kernel, the
    ``sensor_weight`` family.

    This used to scan ``solvers/`` for forbidden Torch calls and assert that
    ``torch.fft`` was still CALLED there, so that it could not pass by the
    package being empty. Phase 11 deleted the package, which makes that scan
    vacuous in the strongest possible way, and the emptiness guard becomes the
    opposite claim: the directory must not exist, and none of the five
    functions may reappear under any owner.

    The tree-wide forbidden-call scan is NOT restated here. It has an owner -
    ``ci/check_torch_physics_allowlist.py`` walks all of ``witwin/`` with an
    empty exclusion list and a frozen digest - and duplicating it with a
    different allowlist is how two gates end up disagreeing.
    """

    assert not (REPO_ROOT / "witwin" / "radar" / "solvers").exists()

    migrated = ("compute_total_path_lengths", "compute_antenna_pattern_gains", "compute_slot_path_tensors")
    offenders = []
    scanned = 0
    for package in OWNER_PACKAGES:
        for path in _modules(package):
            scanned += 1
            for node in ast.walk(_tree(path)):
                if isinstance(node, ast.FunctionDef) and node.name in migrated:
                    offenders.append((path.name, node.name, node.lineno))
    assert offenders == [], offenders
    assert scanned > 0, "the scan must walk real modules"


#: Every Torch call in ``radar.py`` that this file's forbidden list matches,
#: named by the function it sits in, with the reason it is still there.
#:
#: Two kinds, and the difference matters:
#:
#: * FREEZE-TIME SETUP, permanent. ``_set_pose_fields`` and
#:   ``_world_from_local_matrix`` orthonormalise a pose once per radar or once
#:   per ``set_pose``. They are not a per-path hot path and work item 8 does
#:   not name them.
#: * WORK-ITEM-8 SURVIVORS, debt. There are NONE left. There were two, and
#:   naming them here is what a closed debt looks like.
#:   ``Radar.waveform`` was the Torch chirp ``exp(j 2 pi (fc t + S t^2 / 2))``,
#:   held alive only because ``tests/reference/dsp_oracles.py`` needed it to
#:   build the independent time-domain reference; both died with the Dirichlet
#:   route they belonged to. ``_apply_phase_noise`` belonged to the legacy
#:   ``NoiseModelRuntime`` that ``FrontendChain`` replaced, and it died with
#:   that runtime. What is left below is freeze-time pose setup only.
#:
#: ``_normalize_rows`` left the FREEZE-TIME group at the same time: its only
#: caller was ``PolarizationRuntime.from_config``.
#:
#: Equality, not containment. A new Torch physics expression in the facade is a
#: failure, and so is a stale entry for one that was finally deleted.
RADAR_FACADE_TORCH_PHYSICS = {
    ("_set_pose_fields", "torch.linalg.norm"),
    ("_world_from_local_matrix", "torch.linalg.norm"),
}


def _enclosing_functions(tree: ast.Module) -> dict[int, str]:
    names: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
            names.setdefault(line, node.name)
    return names


def test_the_radar_facade_carries_no_unrecorded_torch_physics():
    """The half of work item 8 the package scan cannot see.

    ``radar.py`` is a facade, not an owner package, so it is not under any of
    the scanned directories - and it is the module work item 8 named alongside
    ``solvers/common.py``. Scanning it with an explicit, reasoned
    allowlist records the survivors where a reader of the guard will see them,
    and turns "the migration is not finished" from a report sentence into a
    test that fails if the list grows.
    """

    path = REPO_ROOT / "witwin" / "radar" / "radar.py"
    tree = _tree(path)
    functions = _enclosing_functions(tree)
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if not name.startswith("torch.") or name[len("torch.") :] not in FORBIDDEN_TORCH_CALLS:
            continue
        found.add((functions.get(node.lineno, "<module>"), name))
    assert found == RADAR_FACADE_TORCH_PHYSICS, sorted(found ^ RADAR_FACADE_TORCH_PHYSICS)


def test_no_owner_gates_a_route_on_requires_grad():
    """A route chosen by ``requires_grad`` swallows an ADR-038 forward dual.

    A forward-only dual has ``requires_grad == False``, so a branch of the form
    ``if x.requires_grad: <one route> else: <another>`` sends a tangent down the
    route that does not carry one. A ``requires_grad`` predicate may be READ;
    what is forbidden is an ``if`` whose test
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
                    isinstance(inner, ast.Attribute) and inner.attr == "requires_grad" for inner in ast.walk(node.test)
                )
                refuses = all(isinstance(statement, ast.Raise) for statement in node.body) and not node.orelse
                if mentioned and not refuses:
                    offenders.append((package, path.name, node.lineno))
    assert offenders == [], offenders


def test_no_owner_names_drjit():
    """Zero ``drjit`` names anywhere in the production graph."""

    offenders = []
    packages = (*OWNER_PACKAGES, "propagation", "paths", "scattering")
    for package in packages:
        for path in _modules(package):
            if "drjit" in path.read_text(encoding="utf-8"):
                offenders.append(path.name)
    assert offenders == [], offenders


def _synthesize_source() -> ast.FunctionDef:
    tree = _tree(REPO_ROOT / "witwin" / "radar" / "radar.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_synthesize":
            return node
    raise AssertionError("Radar._synthesize must exist")


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
        radar._synthesize(object(), slow_time_mode=None)


# ---------------------------------------------------------------------------
# Phase 9: the guards and the orchestration this phase added
# ---------------------------------------------------------------------------


#: The package-root policy owner decides whether a call may proceed and never
#: produces a number.
PHASE9_GUARD_OWNERS = ("policy.py",)

#: Everything the policy module is allowed to ask Torch. Every entry is a
#: PREDICATE or a decorator; none of them constructs, allocates or computes.
PHASE9_GUARD_TORCH_CALLS = frozenset(
    {"torch.is_grad_enabled", "torch.autograd.forward_ad.unpack_dual", "torch.autograd.function.once_differentiable"}
)

#: The ONE arithmetic Torch expression Phase 9 added to the production graph,
#: with the constructors that place its result. ``rcs_amplitude`` is the
#: ``sqrt(4 pi sigma)/lambda`` law and it runs once per response, off the
#: per-path loop; ``torch.tensor`` places the phase beside a live amplitude
#: because ``device=`` cannot move a graph-bearing tensor. ``evaluate``'s
#: ``torch.exp`` predates Phase 9 and is the response's own complex assembly.
#:
#: Equality, not containment, and for the usual reason: a second Torch physics
#: expression added to this module must fail here, and so must a stale entry.
SCATTERING_TORCH_CALLS = {
    ("__post_init__", "torch.all"),
    ("__post_init__", "torch.linalg.vector_norm"),
    ("backward", "torch.empty_like"),
    ("evaluate", "torch.exp"),
    ("forward", "torch.empty"),
    ("forward", "torch.empty_like"),
    ("from_rcs", "torch.tensor"),
    ("from_values", "torch.tensor"),
    ("jvp", "torch.empty"),
    ("jvp", "torch.empty_like"),
    ("jvp", "torch.zeros_like"),
    ("rcs_amplitude", "torch.sqrt"),
}
#: Packages that gained a Phase-9 guard. Wider than ``OWNER_PACKAGES``: the wall
#: is in ``processing``, the velocity refusal is in ``propagation`` and the
#: deformation refusal is in top-level ``smpl.py``.
PHASE9_GUARDED_PACKAGES = (
    "processing",
    "propagation",
    "paths",
    "scattering",
    "smpl",
    "sensors",
    "frontend",
    "synthesis",
)

#: The one place in the package where an ``if`` on ``requires_grad`` genuinely
#: SELECTS behaviour rather than refusing or classifying. ``SMPLBody._evaluate``
#: nudges a grad-carrying shape by 1e-8 to keep the SMPL layer's backward
#: defined, which is a legacy numerical workaround inside a legacy path that IS
#: driven to a loss. Recorded rather than removed: deleting it would change a
#: working legacy capability, and that is a numerical decision with its own
#: evidence rather than an architecture cleanup.
PHASE9_KNOWN_REQUIRES_GRAD_ROUTES = set()


def _torch_calls(path: pathlib.Path) -> set:
    """``(enclosing function, dotted call)`` for every ``torch.*`` call."""

    tree = _tree(path)
    functions = _enclosing_functions(tree)
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if not name.startswith("torch."):
            continue
        found.add((functions.get(node.lineno, "<module>"), name))
    return found


def _selects_on_requires_grad(node: ast.If) -> bool:
    """True when an ``if`` on ``requires_grad`` chooses rather than refuses.

    Three shapes are legitimate and are not selection: a body that only
    ``raise``s (a refusal), a body that only ``return``s a constant or a bare
    name (a classifier, or an early return inside a refusal helper), and a body
    that does neither but has no ``else`` and only reassigns - which is the one
    recorded route.
    """

    if not any(isinstance(inner, ast.Attribute) and inner.attr == "requires_grad" for inner in ast.walk(node.test)):
        return False
    if node.orelse:
        return True
    if all(isinstance(item, ast.Raise) for item in node.body):
        return False
    if all(
        isinstance(item, ast.Return) and (item.value is None or isinstance(item.value, (ast.Constant, ast.Name)))
        for item in node.body
    ):
        return False
    return True


def test_the_phase9_guard_owners_only_ask_torch_questions():
    """A refusal owner that started computing would appear here.

    The whole value of putting the wall and the first-order rule in two
    dedicated modules is that their contents are checkable at a glance. This is
    that glance, automated.
    """

    offenders = []
    for name in PHASE9_GUARD_OWNERS:
        path = REPO_ROOT / "witwin" / "radar" / name
        assert path.exists(), path
        for function, call in _torch_calls(path):
            if call not in PHASE9_GUARD_TORCH_CALLS:
                offenders.append((name, function, call))
    assert offenders == [], offenders


def test_the_phase9_guard_scan_is_not_vacuous():
    """Calibration: the guards really do call the predicates they claim to.

    Without this, deleting ``first_order_only``'s body would leave the
    assertion above passing on an empty set.
    """

    calls = set()
    for name in PHASE9_GUARD_OWNERS:
        calls |= {call for _, call in _torch_calls(REPO_ROOT / "witwin" / "radar" / name)}
    assert "torch.is_grad_enabled" in calls
    assert "torch.autograd.forward_ad.unpack_dual" in calls
    assert "torch.autograd.function.once_differentiable" in calls


def test_scattering_torch_calls_are_an_exact_audited_set():
    """The consolidated scattering axis must not grow hidden Torch physics.

    The exact set includes validation, native output-buffer construction, and
    the scalar-RCS amplitude law. Any additional Torch call requires an explicit
    architecture review instead of silently moving physics out of native code.
    """

    path = REPO_ROOT / "witwin" / "radar" / "scattering.py"
    found = _torch_calls(path)
    assert found == SCATTERING_TORCH_CALLS, sorted(found ^ SCATTERING_TORCH_CALLS)


def test_no_phase9_guarded_package_gates_a_route_on_requires_grad():
    """The Phase-6 rule, over every package Phase 9 touched.

    A forward-only dual has ``requires_grad == False`` the whole time, so a
    branch that selects on it sends a tangent down the side that does not carry
    one. Only the recorded route is allowed.
    """

    offenders = []
    for package in PHASE9_GUARDED_PACKAGES:
        for path in _modules(package):
            tree = _tree(path)
            functions = _enclosing_functions(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue
                if not _selects_on_requires_grad(node):
                    continue
                key = (package, path.name, functions.get(node.lineno, "<module>"))
                if key in PHASE9_KNOWN_REQUIRES_GRAD_ROUTES:
                    continue
                offenders.append((package, path.name, node.lineno))
    assert offenders == [], offenders


def test_no_phase9_guard_branches_on_requires_grad():
    """Derivative capability is declared at typed boundaries, never by nudging data."""

    found = set()
    for package in PHASE9_GUARDED_PACKAGES:
        for path in _modules(package):
            tree = _tree(path)
            functions = _enclosing_functions(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.If) and _selects_on_requires_grad(node):
                    found.add((package, path.name, functions.get(node.lineno, "<module>")))
    assert found == set(), sorted(found)


def test_no_phase9_guard_answers_with_a_detach_or_a_zero():
    """The refusal owners must not sever a graph instead of refusing.

    ``detach`` and ``zeros_like`` are how a stage answers a question it cannot
    answer while looking like it did. Neither appears in a guard module, and a
    guard that grew one would be publishing exactly the silent zero this phase
    exists to remove.
    """

    offenders = []
    for name in PHASE9_GUARD_OWNERS:
        path = REPO_ROOT / "witwin" / "radar" / name
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call):
                continue
            call = _dotted(node.func)
            if call.endswith(".detach") or call.endswith("zeros_like"):
                offenders.append((name, call, node.lineno))
    assert offenders == [], offenders
