"""The import boundary, measured in a fresh process.

Radar consumes the stable Channel propagation consumer and nothing else. It is
not a second exception to ADR-008 and it does not reach a Channel solver, the
enumerated engine, the internal propagation contracts, or the native extension.

Importing ``witwin.channel.propagation.consumer`` alone already initializes a
large part of the Channel package, including ``witwin.channel.runtime.*``.
That is Channel's own package initialization; Radar names none of it. The
assertion is therefore that Radar adds NOTHING to the facade's own closure,
together with an absolute assertion that no solver and no internal propagation
module is ever loaded, and that no ``drjit`` or ``rayd`` module is loaded by any
route. Every probe runs in a SUBPROCESS: by the time a test runs in process, the
pytest session has already imported half the world.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Modules that must NEVER appear in the process when Radar imports the spike:
# every Channel solver, the enumerated engine, and the internal propagation
# contracts. These are the ADR-008 properties and they hold absolutely.
SOLVER_AND_INTERNALS = (
    "witwin.channel.path",
    "witwin.channel.deterministic",
    "witwin.channel.montecarlo",
    "witwin.channel.propagation.enumerated",
    "witwin.channel.propagation.models",
    "witwin.channel.propagation.topology",
    "witwin.channel.propagation.geometry",
    "witwin.channel.propagation.fields",
    "witwin.channel._channel",
)

SPIKE_IMPORTS = textwrap.dedent(
    """
    import witwin.radar.channel
    import witwin.radar.paths
    import witwin.radar.scattering
    import witwin.radar.synthesis.fmcw
    import witwin.radar.synthesis.ofdm
    import witwin.radar.synthesis.pulsed
    import witwin.radar.sensors
    import witwin.radar.frontend
    import witwin.radar.processing.range_doppler
    """
).strip()


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


def _subprocess_modules(body: str) -> set[str]:
    script = textwrap.dedent(body).strip() + "\nimport sys\nprint('\\n'.join(sorted(sys.modules)))\n"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_child_env(),
        timeout=600,
    )
    if completed.returncode != 0:
        raise AssertionError(f"probe subprocess failed:\n{completed.stdout}\n{completed.stderr}")
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


# --------------------------------------------------------------------------
# Layer 1: subprocess sys.modules probe
# --------------------------------------------------------------------------


def test_no_channel_solver_or_internal_module_is_ever_loaded():
    """Solver neutrality and the ADR-008 boundary, asserted absolutely."""

    pytest.importorskip("witwin.channel")
    modules = _subprocess_modules(SPIKE_IMPORTS)
    offenders = sorted(name for name in modules if _matches(name, SOLVER_AND_INTERNALS))
    assert offenders == [], offenders
    assert "witwin.channel.propagation.consumer" in modules


def test_radar_adds_nothing_to_the_consumer_facade_closure():
    """Radar reaches the facade and stops there.

    Everything else under ``witwin.channel`` that ends up loaded was loaded by
    importing the facade itself, which is Channel's own package initialization
    and not something Radar can or should avoid.
    """

    pytest.importorskip("witwin.channel")
    facade = {
        name
        for name in _subprocess_modules("import witwin.channel.propagation.consumer")
        if name.startswith("witwin.channel")
    }
    spike = {name for name in _subprocess_modules(SPIKE_IMPORTS) if name.startswith("witwin.channel")}
    assert spike - facade == set(), sorted(spike - facade)


def test_the_propagation_package_alone_does_not_require_channel():
    modules = _subprocess_modules("import witwin.radar.propagation")
    offenders = sorted(name for name in modules if name.startswith("witwin.channel"))
    assert offenders == [], offenders


def test_synthesis_scattering_and_paths_do_not_require_channel():
    modules = _subprocess_modules(
        "import witwin.radar.sensors\n"
        "import witwin.radar.frontend\n"
        "import witwin.radar.synthesis.fmcw\n"
        "import witwin.radar.synthesis.ofdm\n"
        "import witwin.radar.synthesis.pulsed\n"
        "import witwin.radar.processing.range_doppler\n"
        "import witwin.radar.scattering\n"
        "import witwin.radar.paths"
    )
    offenders = sorted(name for name in modules if name.startswith("witwin.channel"))
    assert offenders == [], offenders


# --------------------------------------------------------------------------
# Layer 2: the process-global Dr.Jit assertion
# --------------------------------------------------------------------------


def test_no_drjit_or_rayd_in_the_process_after_importing_witwin_radar():
    """The strict form: not in the closure, not by any route, not at all.

    This is a statement about the PROCESS, not about a delta. Importing the
    package root is the widest thing a caller can do, and after it neither
    ``drjit`` nor ``rayd`` is in ``sys.modules``. No lazy import is left that a
    later call could trigger either: the three modules that imported them are
    deleted, and the entry points that reached them raise.
    """

    modules = _subprocess_modules("import witwin.radar")
    offenders = sorted(name for name in modules if name.split(".")[0] in ("drjit", "rayd"))
    assert offenders == [], offenders


def test_the_spike_adds_no_drjit_rayd_or_channel_internals():
    pytest.importorskip("witwin.channel")
    baseline = _subprocess_modules("import witwin.radar")
    with_spike = _subprocess_modules("import witwin.radar\n" + SPIKE_IMPORTS)
    added = with_spike - baseline

    offenders = sorted(name for name in with_spike if name.split(".")[0] in ("drjit", "rayd"))
    assert offenders == [], offenders
    assert not any(_matches(name, SOLVER_AND_INTERNALS) for name in added)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_test_support_resolves_inside_this_worktree():
    """``support`` is a common top-level name; make sure it is OURS."""

    import support

    import witwin.radar

    assert pathlib.Path(support.__file__).resolve().parent == REPO_ROOT / "tests" / "support"
    assert pathlib.Path(witwin.radar.__file__).resolve().is_relative_to(REPO_ROOT)


def test_the_consumer_contract_is_the_version_this_adapter_was_built_against():
    # Version 6 (Channel ADR-043): ``field_direction`` carries a derivative on
    # the fixed-topology route, ``PropagationCapabilities`` publishes the
    # per-route AD record the adapter queries, and ``PropagationDiagnostics``
    # carries AD launch and tape accounting.
    pytest.importorskip("witwin.channel")
    from witwin.channel.propagation import consumer

    assert consumer.CONTRACT_VERSION == 6
    capabilities = consumer.capabilities()
    assert capabilities.fixed_topology_components == frozenset({"los", "reflection"})
    assert capabilities.supports_fixed_topology
    assert "scalar_transport" in capabilities.fixed_topology_responses
    # ADR-043: the components whose ``field_direction`` is graph-bearing under
    # a frozen topology. Radar's aspect chain (Deliverable 1) depends on this
    # covering everything Radar ever freezes.
    assert capabilities.direction_differentiable_components == frozenset({"los", "reflection"})
    assert capabilities.fixed_topology_components.issubset(capabilities.direction_differentiable_components)
