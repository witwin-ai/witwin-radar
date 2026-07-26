"""The Phase-4 import boundary, in three layers plus an AST prohibition scan.

Radar consumes the stable Channel propagation consumer and nothing else. It is
not a second exception to ADR-008 and it does not reach a Channel solver, the
enumerated engine, the internal propagation contracts, or the native extension.

One measured fact shapes what these tests can honestly assert, and it is
stated rather than worked around: importing
``witwin.channel.propagation.consumer`` ALONE already initializes a large part
of the Channel package, including ``witwin.channel.runtime.*``. That is
Channel's own package initialization; Radar names none of it. The assertion is
therefore that Radar adds NOTHING to the facade's own closure, which is the
real boundary property, together with an absolute assertion that no solver and
no internal propagation module is ever loaded.

The Dr.Jit assertion is now the strict, process-global one. It used to be
unachievable: importing any ``witwin.radar.*`` submodule initialized
``witwin/radar/__init__.py``, which imported ``trace.py``, which imported
``drjit``. That edge is deleted, so the weaker baseline-delta form it forced,
and the by-name exclusion of the package root from the static closure, are gone
with it. This file predicted the conversion and failed loudly at exactly the
right moment when the edge disappeared.

The three layers do different jobs and all three are required:

1. A SUBPROCESS ``sys.modules`` probe. In-process would prove nothing: by the
   time this test runs, the pytest session has already imported half the world.
2. A PROCESS-GLOBAL assertion that importing the package pulls in neither
   ``drjit`` nor ``rayd``, at all, by any route.
3. A STATIC AST CLOSURE over the new modules, which proves that no NEW module
   names anything forbidden even if some future edge reintroduces it elsewhere.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RADAR_ROOT = REPO_ROOT / "witwin" / "radar"

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

# Modules Radar must never NAME. Some of them are loaded anyway by Channel's or
# Radar's own package initialization; the static closure is what proves Radar
# does not reach for them.
NEVER_NAMED = SOLVER_AND_INTERNALS + (
    "witwin.channel.runtime.extension",
    "drjit",
    "rayd.drjit",
)

# The exact Channel imports the spike is allowed to name.
ALLOWED_CHANNEL_IMPORTS = frozenset(
    {"witwin.channel.propagation", "witwin.channel.propagation.consumer"}
)

SPIKE_MODULES = (
    "witwin/radar/propagation/__init__.py",
    "witwin/radar/propagation/contracts.py",
    "witwin/radar/propagation/channel_consumer.py",
    # The Core-kinematics seam. It names no witwin package at module scope - it
    # is duck typed over Core's snapshot shape - and it is scanned here because
    # it runs once per frame and is the natural place for a host read of a
    # velocity to appear.
    "witwin/radar/propagation/kinematics.py",
    # The scene-driven epoch loop. It decides when a moving world pays for a
    # compile and a rediscovery, so it runs per frame and is scanned for the
    # same reasons. It takes its scene compiler as an ARGUMENT precisely so
    # that it does not become a second module naming ``witwin.channel``.
    "witwin/radar/propagation/epochs.py",
    "witwin/radar/paths/__init__.py",
    "witwin/radar/paths/contracts.py",
    "witwin/radar/paths/_identity.py",
    "witwin/radar/paths/direct.py",
    "witwin/radar/paths/two_way.py",
    "witwin/radar/scattering/__init__.py",
    "witwin/radar/scattering/base.py",
    "witwin/radar/scattering/rcs.py",
    "witwin/radar/synthesis/__init__.py",
    "witwin/radar/synthesis/assembly.py",
    "witwin/radar/synthesis/contracts.py",
    "witwin/radar/synthesis/dirichlet_spectrum.py",
    "witwin/radar/synthesis/fmcw_beat.py",
    "witwin/radar/synthesis/ofdm_cfr.py",
    "witwin/radar/synthesis/pulsed_echo.py",
    "witwin/radar/sensors/__init__.py",
    "witwin/radar/sensors/contracts.py",
    "witwin/radar/sensors/pattern.py",
    "witwin/radar/sensors/legacy_paths.py",
    "witwin/radar/sensors/weights.py",
    "witwin/radar/frontend/__init__.py",
    "witwin/radar/frontend/contracts.py",
    "witwin/radar/frontend/chain.py",
    # DSP glue, and in this list on purpose. It is subjected to the same
    # host-observation and Dr.Jit scans as the native owners because being on
    # the Torch side of the processing exception is permission to use an FFT,
    # not permission to read a device tensor to the host per frame.
    "witwin/radar/sigproc/matched_filter.py",
)

SPIKE_IMPORTS = textwrap.dedent(
    """
    import witwin.radar.propagation.channel_consumer
    import witwin.radar.paths.two_way
    import witwin.radar.scattering.rcs
    import witwin.radar.synthesis.fmcw_beat
    import witwin.radar.synthesis.ofdm_cfr
    import witwin.radar.synthesis.pulsed_echo
    import witwin.radar.sensors.legacy_paths
    import witwin.radar.sensors.weights
    import witwin.radar.frontend.chain
    import witwin.radar.sigproc.matched_filter
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
    script = (
        textwrap.dedent(body).strip()
        + "\nimport sys\nprint('\\n'.join(sorted(sys.modules)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_child_env(),
        timeout=600,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"probe subprocess failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


# --------------------------------------------------------------------------
# Layer 1: subprocess sys.modules probe
# --------------------------------------------------------------------------


def test_no_channel_solver_or_internal_module_is_ever_loaded():
    """Solver neutrality and the ADR-008 boundary, asserted absolutely."""

    pytest.importorskip("witwin.channel")
    modules = _subprocess_modules(SPIKE_IMPORTS)
    offenders = sorted(
        name for name in modules if _matches(name, SOLVER_AND_INTERNALS)
    )
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
    spike = {
        name for name in _subprocess_modules(SPIKE_IMPORTS)
        if name.startswith("witwin.channel")
    }
    assert spike - facade == set(), sorted(spike - facade)


def test_the_propagation_package_alone_does_not_require_channel():
    modules = _subprocess_modules("import witwin.radar.propagation")
    offenders = sorted(name for name in modules if name.startswith("witwin.channel"))
    assert offenders == [], offenders


def test_synthesis_scattering_and_paths_do_not_require_channel():
    modules = _subprocess_modules(
        "import witwin.radar.sensors.weights\n"
        "import witwin.radar.frontend.chain\n"
        "import witwin.radar.synthesis.fmcw_beat\n"
        "import witwin.radar.synthesis.ofdm_cfr\n"
        "import witwin.radar.synthesis.pulsed_echo\n"
        "import witwin.radar.sigproc.matched_filter\n"
        "import witwin.radar.scattering.rcs\n"
        "import witwin.radar.paths.two_way"
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
    offenders = sorted(
        name for name in modules if name.split(".")[0] in ("drjit", "rayd")
    )
    assert offenders == [], offenders


def test_the_spike_adds_no_drjit_rayd_or_channel_internals():
    pytest.importorskip("witwin.channel")
    baseline = _subprocess_modules("import witwin.radar")
    with_spike = _subprocess_modules("import witwin.radar\n" + SPIKE_IMPORTS)
    added = with_spike - baseline

    offenders = sorted(
        name for name in with_spike if name.split(".")[0] in ("drjit", "rayd")
    )
    assert offenders == [], offenders
    assert not any(_matches(name, SOLVER_AND_INTERNALS) for name in added)


# --------------------------------------------------------------------------
# Layer 3: static AST closure over the new modules
# --------------------------------------------------------------------------


def _module_name(relative: str) -> str:
    stem = (
        relative.removeprefix("witwin/radar/")
        .removesuffix("/__init__.py")
        .removesuffix(".py")
        .replace("/", ".")
    )
    return f"witwin.radar.{stem}" if stem else "witwin.radar"


def _module_path(name: str) -> pathlib.Path | None:
    relative = name.removeprefix("witwin.radar.").replace(".", "/")
    for candidate in (
        RADAR_ROOT / f"{relative}.py",
        RADAR_ROOT / relative / "__init__.py",
    ):
        if candidate.exists():
            return candidate
    return None


def _imports_of(path: pathlib.Path, module_name: str) -> set[str]:
    """Every module name a file imports, with relative imports resolved."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module_name if path.name == "__init__.py" else module_name.rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                keep = len(parts) - (node.level - 1)
                base = ".".join(parts[:keep]) if keep > 0 else parts[0]
                root = f"{base}.{node.module}" if node.module else base
            else:
                root = node.module or ""
            found.add(root)
            found.update(f"{root}.{alias.name}" for alias in node.names)
    return found


def _static_closure() -> tuple[set[str], set[str]]:
    """Modules reachable from the spike files, and every import they name.

    Descends only into ``witwin.radar.*``. It used to exclude the package root
    by name, because that file owned the legacy Dr.Jit edge and following it
    would have made this layer measure the root instead of the new code. The
    edge is gone, so the exclusion is gone.
    """

    pending = [_module_name(relative) for relative in SPIKE_MODULES]
    seen: set[str] = set()
    every_import: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_path(name)
        if path is None:
            continue
        for imported in _imports_of(path, name):
            every_import.add(imported)
            if imported.startswith("witwin.radar.") and _module_path(imported):
                pending.append(imported)
    return seen, every_import


def test_static_closure_of_the_new_modules_names_nothing_forbidden():
    reachable, imports = _static_closure()
    assert "witwin.radar.propagation.channel_consumer" in reachable

    offenders = sorted(name for name in imports if _matches(name, NEVER_NAMED))
    assert offenders == [], offenders

    channel = {name for name in imports if name.startswith("witwin.channel")}
    assert channel == ALLOWED_CHANNEL_IMPORTS, sorted(channel)

    # The deleted modules are gone from the source tree, so no edge can lead
    # back to them even by accident.
    for removed in ("trace.py", "material.py", "_rayd_bridge.py"):
        assert not (RADAR_ROOT / removed).exists(), removed


def test_only_the_adapter_crosses_the_channel_boundary():
    crossing = [
        relative
        for relative in SPIKE_MODULES
        if any(
            name.startswith("witwin.channel")
            for name in _imports_of(REPO_ROOT / relative, _module_name(relative))
        )
    ]
    assert crossing == ["witwin/radar/propagation/channel_consumer.py"], crossing


# --------------------------------------------------------------------------
# AST prohibition scan
#
# Scanned with the AST, not with text: every one of these tokens also appears
# in prose in these files' docstrings, where it is documentation of the rule
# rather than a violation of it.
# --------------------------------------------------------------------------


HOST_OBSERVATION_METHODS = frozenset({"cpu", "numpy", "tolist", "item"})

# paths/_identity.py reads frozen leg row identity to the host once, at freeze
# time, where the consumer has already synchronized. That is the sanctioned
# host observation, and naming ONE owner is what keeps it that way: neither
# composer may grow its own read.
HOST_OBSERVATION_OWNERS = {
    "witwin/radar/paths/_identity.py": frozenset({"tolist"}),
    # synthesis/assembly.py::validate_pair_ordering is the freeze-time check
    # that a frozen pair partition really is this array's TX x RX grid. It
    # reads the pair ranks on the host ONCE, where the topology is decided and
    # the read is free, and it is a separate function from assemble_frame_cube
    # precisely so the per-frame path cannot inherit the read.
    # test_the_frame_assembly_path_reads_no_tensor_value pins that split.
    "witwin/radar/synthesis/assembly.py": frozenset({"tolist"}),
}


def _host_observation_calls(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in HOST_OBSERVATION_METHODS
    ]


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_no_hot_path_host_observation_in_the_new_modules():
    offenders: list[tuple[str, str]] = []
    for relative in SPIKE_MODULES:
        allowed = HOST_OBSERVATION_OWNERS.get(relative, frozenset())
        for method in _host_observation_calls(REPO_ROOT / relative):
            if method not in allowed:
                offenders.append((relative, method))
    assert offenders == [], offenders


def test_no_drjit_reference_of_any_kind_in_the_new_modules():
    offenders: list[tuple[str, str]] = []
    for relative in SPIKE_MODULES:
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in ("drjit", "dr"):
                offenders.append((relative, node.id))
            elif isinstance(node, ast.Attribute):
                dotted = _dotted(node)
                if dotted.startswith(("drjit.", "dr.")):
                    offenders.append((relative, dotted))
    assert offenders == [], offenders


def test_the_synthesis_hot_loop_is_native_not_torch():
    facade = REPO_ROOT / "witwin/radar/synthesis/fmcw_beat.py"
    tree = ast.parse(facade.read_text(encoding="utf-8"))

    # No Python iteration over paths or samples anywhere in the facade.
    loops = [
        type(node).__name__
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.For | ast.While | ast.ListComp | ast.GeneratorExp | ast.DictComp,
        )
    ]
    assert loops == [], loops

    # And no Torch evaluation of the phasor itself.
    called = {
        _dotted(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for forbidden in ("torch.exp", "torch.sin", "torch.cos", "torch.cdist"):
        assert forbidden not in called, forbidden

    source = facade.read_text(encoding="utf-8")
    assert "_FmcwBeatSynthesis.apply(" in source
    for operator in ("fmcw_beat_forward", "fmcw_beat_backward", "fmcw_beat_jvp"):
        assert operator in source, operator

    # TDM slot time is a KERNEL ARGUMENT, not a Torch expression. The facade
    # passes the per-segment transmitter index across the boundary to all three
    # operators and never forms a slot time itself: computing
    # (chirp * num_tx + tx) * chirp_period in Torch would be a second, silently
    # divergent copy of the slow-time axis, and broadcasting it over the sample
    # axis would materialise the per-path-per-sample intermediate the kernel
    # exists to avoid.
    assert "segment_tx_index" in source
    for operator in ("fmcw_beat_forward", "fmcw_beat_backward", "fmcw_beat_jvp"):
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _dotted(node.func).endswith(operator)
        )
        passed = [
            argument.id
            for argument in call.args
            if isinstance(argument, ast.Name)
        ]
        assert "tx_index" in passed, (operator, passed)

    kernel = (REPO_ROOT / "witwin/radar/cuda/kernels/fmcw_beat.cu").read_text(
        encoding="utf-8"
    )
    for symbol in (
        "__global__ void fmcw_beat_forward_kernel",
        "__global__ void fmcw_beat_backward_kernel",
        "__global__ void fmcw_beat_jvp_kernel",
        "sincosf",
        "segment_tx_index",
        "slot_time",
    ):
        assert symbol in kernel, symbol


def test_the_ofdm_hot_loop_is_native_not_torch():
    """The CFR sum is a kernel, and the OFDM family conjugates nothing.

    The second half of that sentence is the part a shape test cannot see. OFDM
    publishes the CHANNEL convention and FMCW publishes its conjugate, so a
    single stray ``conj`` here would flip every Doppler sign and every target
    phase while leaving all magnitudes, all shapes, and the delay slope
    untouched.
    """

    facade = REPO_ROOT / "witwin/radar/synthesis/ofdm_cfr.py"
    tree = ast.parse(facade.read_text(encoding="utf-8"))

    loops = [
        type(node).__name__
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.For | ast.While | ast.ListComp | ast.GeneratorExp | ast.DictComp,
        )
    ]
    assert loops == [], loops

    called = {
        _dotted(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for forbidden in (
        "torch.exp",
        "torch.sin",
        "torch.cos",
        "torch.cdist",
        "torch.conj",
    ):
        assert forbidden not in called, forbidden

    source = facade.read_text(encoding="utf-8")
    assert "_OfdmCfrSynthesis.apply(" in source
    for operator in ("ofdm_cfr_forward", "ofdm_cfr_backward", "ofdm_cfr_jvp"):
        assert operator in source, operator

    # The conjugation site belongs to the FMCW owner and is never reached from
    # here. Scanned with the AST, not with text: the docstring names both the
    # function and the other waveform on purpose, to say what this module does
    # NOT do, and a text scan would forbid the explanation along with the act.
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        _dotted(node.func).rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "channel_phasor_to_beat_weight" not in called_names
    assert "conj" not in called_names
    imported = _imports_of(facade, "witwin.radar.synthesis.ofdm_cfr")
    assert not any("fmcw" in name for name in imported), sorted(imported)

    kernel = (REPO_ROOT / "witwin/radar/cuda/kernels/ofdm_cfr.cu").read_text(
        encoding="utf-8"
    )
    for symbol in (
        "__global__ void ofdm_cfr_forward_kernel",
        "__global__ void ofdm_cfr_backward_kernel",
        "__global__ void ofdm_cfr_jvp_kernel",
        "sincosf",
    ):
        assert symbol in kernel, symbol

    # The kernel's own sign: `cycles` is negated, which is what keeps the cube
    # in Channel's exp(-j k d) convention rather than the beat convention.
    assert "-(f_sub * tau + carrier_rate_hz * tau_drift + carrier_hz * tau)" in kernel

    # No TDM slot table: OFDM slow time is per symbol, shared by every pair.
    assert "segment_tx_index" not in kernel


def test_the_pulsed_hot_loop_is_native_not_torch():
    """The pulse train is a kernel, and the pulse is analytic in this family.

    The second half is what a shape test cannot see. The whole design decision
    of this waveform is that ``p(t - tau)`` is evaluated at a CONTINUOUS
    fractional argument, so the facade must contain no indexing, no gather, no
    interpolation, and no sampled pulse at all - the only sampled replica in the
    package lives in the DSP matched filter, and that file must not be reachable
    from here.
    """

    facade = REPO_ROOT / "witwin/radar/synthesis/pulsed_echo.py"
    tree = ast.parse(facade.read_text(encoding="utf-8"))

    loops = [
        type(node).__name__
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.For | ast.While | ast.ListComp | ast.GeneratorExp | ast.DictComp,
        )
    ]
    assert loops == [], loops

    called = {
        _dotted(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for forbidden in (
        "torch.exp",
        "torch.sin",
        "torch.cos",
        "torch.cdist",
        "torch.conj",
        "torch.gather",
        "torch.searchsorted",
        "torch.round",
    ):
        assert forbidden not in called, forbidden

    source = facade.read_text(encoding="utf-8")
    assert "_PulsedEchoSynthesis.apply(" in source
    for operator in (
        "pulsed_echo_forward",
        "pulsed_echo_backward",
        "pulsed_echo_jvp",
    ):
        assert operator in source, operator

    # No conjugation site and no reach into either neighbouring waveform.
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        _dotted(node.func).rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "channel_phasor_to_beat_weight" not in called_names
    assert "conj" not in called_names
    imported = _imports_of(facade, "witwin.radar.synthesis.pulsed_echo")
    assert not any("fmcw" in name for name in imported), sorted(imported)
    assert not any("ofdm" in name for name in imported), sorted(imported)
    # The matched filter is the CONSUMER of this module's output, never a
    # dependency of it. An edge in this direction would put a sampled replica
    # inside the synthesis owner.
    assert not any("sigproc" in name for name in imported), sorted(imported)

    kernel = (REPO_ROOT / "witwin/radar/cuda/kernels/pulsed_echo.cu").read_text(
        encoding="utf-8"
    )
    for symbol in (
        "__global__ void pulsed_echo_forward_kernel",
        "__global__ void pulsed_echo_backward_kernel",
        "__global__ void pulsed_echo_jvp_kernel",
        "sincosf",
    ):
        assert symbol in kernel, symbol

    # Scanned over the CODE, with the line comments stripped. The header
    # explains at length what the kernel must not do, naming the very tokens
    # this scan forbids, and a whole-file text scan would reject the
    # explanation along with the act.
    code = "\n".join(
        line for line in kernel.splitlines() if not line.lstrip().startswith("//")
    )

    # The kernel's own sign: `cycles` is negated, which keeps the train in
    # Channel's exp(-j k d) convention rather than the beat convention.
    assert "-(carrier_rate_hz * tau_drift + carrier_hz * tau)" in code

    # The envelope argument is formed from the CONTINUOUS delay and handed
    # straight to the analytic pulse. `t_fast - tau` with a float tau is the
    # fractional evaluation; a rounded or truncated sample index would be the
    # snapping this family exists to avoid.
    assert "t_fast - tau" in code
    for snapped in ("lrint", "__float2int", "nearbyint", "round(", "tex1D"):
        assert snapped not in code, snapped

    # The support test is HALF-OPEN. A closed one puts an extra sample inside
    # the pulse at exactly the delays that land on the grid, so the sampled
    # pulse changes length with the delay and the matched filter sees a
    # mismatched tap.
    assert "u >= 0.0 && u < pulse_width_s" in code

    # No TDM slot table: pulsed slow time is per pulse, shared by every pair.
    assert "segment_tx_index" not in kernel


def test_the_matched_filter_re_derives_no_geometry():
    """The DSP side owns the filter and nothing else.

    An FFT correlation is allowlisted; a delay, a carrier, or a Doppler term
    computed here would be a second, silently divergent copy of physics that the
    synthesis kernel already owns. Scanned by name because every one of these
    would appear as an attribute read on the spec.
    """

    module = REPO_ROOT / "witwin/radar/sigproc/matched_filter.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))

    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    for forbidden in (
        "carrier_hz",
        "carrier_rate_hz",
        "reference_frequency_hz",
        "total_delay_s",
        "delay_rate",
        "complex_transfer_ref",
        "doppler_frequency_hz",
        "slow_time_phase_step_rad",
        "pri_s",
        "num_pulses",
    ):
        assert forbidden not in attributes, forbidden

    imported = _imports_of(module, "witwin.radar.sigproc.matched_filter")
    assert not any(name.startswith("witwin") for name in imported), sorted(imported)

    source = module.read_text(encoding="utf-8")
    assert "torch.fft" in source
    assert "torch.conj" in source


def test_the_frame_assembly_path_reads_no_tensor_value():
    """The per-frame half of ``assembly.py`` is pure structural packing.

    ``validate_pair_ordering`` is allowed one host read because it runs at
    freeze time. The point of splitting it into its own function is that the
    ones the frame loop actually calls cannot inherit that read, so the scan is
    scoped to those functions rather than to the module - a module-level
    allowance that covered everything would stop meaning anything.
    """

    module = REPO_ROOT / "witwin/radar/synthesis/assembly.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    per_frame = {"assemble_frame_cube", "pair_tx_index", "pair_rx_index"}
    seen: set[str] = set()
    offenders: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in per_frame:
            continue
        seen.add(node.name)
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr in HOST_OBSERVATION_METHODS
            ):
                offenders.append((node.name, inner.func.attr))
    assert seen == per_frame, seen
    assert offenders == [], offenders


def test_the_two_way_join_hot_loop_is_native_not_torch():
    """The per-frame join is a kernel; only ``freeze`` is Python.

    The AST scan is scoped to ``compose`` rather than to the module, because
    ``freeze`` legitimately iterates: it runs ONCE per frozen topology, on the
    host, after the consumer has already synchronized. Scanning the whole file
    would either forbid that or force a blanket exemption that stops meaning
    anything.
    """

    facade = REPO_ROOT / "witwin/radar/paths/two_way.py"
    tree = ast.parse(facade.read_text(encoding="utf-8"))
    compose = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "compose"
    )

    loops = [
        type(node).__name__
        for node in ast.walk(compose)
        if isinstance(
            node,
            ast.For | ast.While | ast.ListComp | ast.GeneratorExp | ast.DictComp,
        )
    ]
    assert loops == [], loops

    called = {
        _dotted(node.func)
        for node in ast.walk(compose)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for forbidden in ("torch.exp", "torch.sin", "torch.cos", "torch.where"):
        assert forbidden not in called, forbidden

    source = facade.read_text(encoding="utf-8")
    assert "_TwoWayJoin.apply(" in source
    for operator in (
        "two_way_join_forward",
        "two_way_join_backward",
        "two_way_join_jvp",
    ):
        assert operator in source, operator

    kernel = (REPO_ROOT / "witwin/radar/cuda/kernels/two_way_join.cu").read_text(
        encoding="utf-8"
    )
    for symbol in (
        "__global__ void two_way_join_forward_kernel",
        "__global__ void two_way_join_backward_kernel",
        "__global__ void two_way_join_jvp_kernel",
    ):
        assert symbol in kernel, symbol


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_test_support_resolves_inside_this_worktree():
    """``support`` is a common top-level name; make sure it is OURS."""

    import support

    import witwin.radar

    assert (
        pathlib.Path(support.__file__).resolve().parent
        == REPO_ROOT / "tests" / "support"
    )
    assert pathlib.Path(witwin.radar.__file__).resolve().is_relative_to(REPO_ROOT)


def test_the_consumer_contract_is_the_version_this_spike_was_built_against():
    # Version 5 = ADR-042: a fixed-topology request may declare
    # frequency_offsets_hz and receive the same frozen rows evaluated at each
    # absolute frequency. Additive behind a None default, so every call this
    # adapter makes today is unchanged bit for bit. Version 4 = ADR-040: a
    # discovered topology carries the world it was discovered against and a
    # frozen replay against a moved world is refused by name. Additive for this
    # adapter, which holds one compiled scene and freezes against it. Version 3
    # = ADR-039: the transport carries the declared source amplitude
    # sqrt(powers_w), which changes no shape, dtype or vocabulary and is pinned
    # by the cross-waveform radar-equation tests.
    pytest.importorskip("witwin.channel")
    from witwin.channel.propagation import consumer

    assert consumer.CONTRACT_VERSION == 5
    capabilities = consumer.capabilities()
    assert capabilities.fixed_topology_components == frozenset({"los", "reflection"})
    assert capabilities.supports_fixed_topology
    assert "scalar_transport" in capabilities.fixed_topology_responses
