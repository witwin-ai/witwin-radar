"""The tape stays inside its owner: it never reaches a public result.

The plan requires it and nothing tested it. Two statements, and they need
different instruments:

**One - no public result carries a tape.** Asserted on REAL objects produced by
the production chain, not on type annotations. An annotation says what a field
was meant to hold; only an instance says what it holds. Every field of every
public result is walked transitively, and the only things allowed at a leaf are
a tensor, a primitive, a string, ``None``, a plain container of those, or
another such record. An autograd ``Function``, a context object, or a
``saved_tensors`` tuple anywhere in that walk is the defect.

A ``grad_fn`` is NOT a leak and is not what this file is about. A differentiable
result is supposed to carry its graph; ``cube.grad_fn`` is how a backward finds
its way home. What must never happen is a result FIELD holding the tape - the
saved tensors themselves, or the context that owns them - because that turns a
data record into a handle on somebody else's memory and makes the tape's
lifetime the result's lifetime.

**Two - no module outside a tape's own owner reads it.** Asserted by parsing
every production module and finding every ``ctx.saved_tensors`` read, then
checking that each one sits inside a ``backward`` or ``jvp`` of an autograd
``Function`` in one of the nine owner files. Its limit, stated honestly: a read
spelled through an alias (``c = ctx; c.saved_tensors``) or reached by
``getattr`` would not be found. That is a real gap and it is accepted, because
the failure this test exists to catch is a consumer reaching into another
owner's context, which nobody writes obfuscated.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "witwin" / "radar"

#: The nine files that own a ``torch.autograd.Function``. ``frontend/chain.py``
#: owns two, which is why there are ten tape owners and nine files.
TAPE_OWNER_FILES = frozenset(
    {
        "frontend/chain.py",
        "paths/two_way.py",
        "scattering/aspect.py",
        "sensors/weights.py",
        "synthesis/dirichlet_spectrum.py",
        "synthesis/fmcw_beat.py",
        "synthesis/ofdm_cfr.py",
        "synthesis/pulsed_echo.py",
    }
)

#: The two methods that are allowed to read a context. ``backward`` is the VJP
#: and ``jvp`` is the forward companion; ``setup_context`` WRITES the tape and
#: never reads it back.
TAPE_READER_METHODS = frozenset({"backward", "jvp"})


def _production_modules():
    return sorted(
        path
        for path in PACKAGE.rglob("*.py")
        if "__pycache__" not in path.parts
    )


# ---------------------------------------------------------------------------
# 1. Nobody outside an owner reads a context
# ---------------------------------------------------------------------------


def _saved_tensor_reads(path: pathlib.Path):
    """``(function name, line)`` for every ``<name>.saved_tensors`` read."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    enclosing: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                enclosing.setdefault(id(child), node.name)
    reads = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {
            "saved_tensors",
            "saved_for_forward",
            "to_save",
        }:
            reads.append((enclosing.get(id(node), "<module>"), node.lineno, node.attr))
    return reads


def test_every_context_read_sits_inside_a_tape_owner():
    """A consumer that unpacked another owner's tape would appear here."""

    offenders = []
    for path in _production_modules():
        relative = path.relative_to(PACKAGE).as_posix()
        reads = _saved_tensor_reads(path)
        if not reads:
            continue
        if relative not in TAPE_OWNER_FILES:
            offenders.append((relative, reads))
            continue
        for function, line, attr in reads:
            if function not in TAPE_READER_METHODS:
                offenders.append((relative, function, line, attr))
    assert not offenders, offenders


def test_the_context_scan_is_not_vacuous():
    """Calibration: the scan finds the reads that are supposed to be there.

    Ten owners, each reading its tape in exactly two places - the backward and
    the jvp - is twenty reads. A scanner that silently matched nothing would
    make the assertion above pass forever.
    """

    found = {
        path.relative_to(PACKAGE).as_posix(): _saved_tensor_reads(path)
        for path in _production_modules()
    }
    live = {name: reads for name, reads in found.items() if reads}
    assert set(live) == TAPE_OWNER_FILES, sorted(live)
    total = sum(len(reads) for reads in live.values())
    assert total == 20, {name: len(reads) for name, reads in live.items()}


def test_no_production_module_stores_a_context_on_an_object():
    """``self.ctx = ...`` or a module-level context would outlive the backward.

    A tape owner legitimately writes ``ctx.spec``, ``ctx.plan`` and friends -
    that is the context carrying its own configuration. Writing a CONTEXT onto
    something else is the shape that leaks it, and it is what this looks for.
    """

    offenders = []
    for path in _production_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not isinstance(value, ast.Name) or value.id != "ctx":
                continue
            offenders.append((path.relative_to(PACKAGE).as_posix(), node.lineno))
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# 2. No public result carries a tape
# ---------------------------------------------------------------------------


def _forbidden(value: object) -> str | None:
    """The name of the tape-shaped thing ``value`` is, or ``None``."""

    if isinstance(value, torch.autograd.Function):
        return "an autograd Function instance"
    if isinstance(value, type) and issubclass(value, torch.autograd.Function):
        return "an autograd Function class"
    if isinstance(value, torch.autograd.function.FunctionCtx):
        return "an autograd context"
    if type(value).__name__ in {"BackwardCFunction", "FunctionCtx", "FunctionMeta"}:
        return f"an autograd internal ({type(value).__name__})"
    if hasattr(value, "saved_tensors") and not isinstance(value, torch.Tensor):
        return "an object exposing saved_tensors"
    return None


def _walk(value: object, path: str, seen: set[int], found: list) -> None:
    if id(value) in seen:
        return
    seen.add(id(value))
    problem = _forbidden(value)
    if problem is not None:
        found.append((path, problem))
        return
    if value is None or isinstance(value, (torch.Tensor, str, bytes, int, float, bool)):
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            _walk(item, f"{path}[{index}]", seen, found)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _walk(item, f"{path}[{key!r}]", seen, found)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _walk(getattr(value, field.name), f"{path}.{field.name}", seen, found)
        return
    if hasattr(value, "__slots__") and not isinstance(value, type):
        for name in value.__slots__:
            _walk(getattr(value, name, None), f"{path}.{name}", seen, found)
        return
    # Anything else is a plan, a spec, a device or a callable. Its own fields
    # are walked when it is a record; otherwise there is nothing to reach.


def _assert_clean(label: str, value: object) -> None:
    found: list = []
    _walk(value, label, set(), found)
    assert not found, found


@pytest.fixture(scope="module")
def frame():
    pytest.importorskip("witwin.channel")
    from support import multi_endpoint_driver as drv

    spike = drv.MultiEndpointSpike()
    sites = spike.site_tensor(requires_grad=True)
    composed, inbound, outbound = spike.frame(
        sites, drv.make_response(), ad_mode="vjp", include_delay_rate=False
    )
    return spike, composed, inbound, outbound


@pytest.mark.gpu
def test_the_leg_batch_and_the_composed_batch_carry_no_tape(frame):
    """``RadarLegBatch`` and ``RadarPathBatch``, with a LIVE graph on them.

    Taken under ``ad_mode='vjp'`` on purpose: a result produced with no graph at
    all could not leak a tape it never had, and would be a test of nothing.
    """

    _, composed, inbound, outbound = frame
    assert composed.complex_transfer_ref.requires_grad, "the fixture must be live"
    _assert_clean("inbound", inbound)
    _assert_clean("outbound", outbound)
    _assert_clean("composed", composed)


@pytest.mark.gpu
def test_the_channel_evaluation_radar_receives_carries_no_tape(frame):
    """The upstream half: what the consumer hands the adapter.

    Radar cannot fix a Channel leak, but it can notice one, and a Channel result
    holding a Channel context would arrive here as a Radar field holding it.
    """

    spike, _, _, _ = frame
    # ``FrozenTopology.prepared`` is the Channel ``PreparedFixedTopology`` the
    # adapter replays every frame, and it is the longest-lived Channel object
    # Radar holds. If a Channel context were going to arrive anywhere, here.
    _assert_clean("inbound.prepared", spike.inbound.prepared)
    _assert_clean("outbound.prepared", spike.outbound.prepared)


@pytest.mark.gpu
def test_the_synthesis_and_sensor_results_carry_no_tape(frame):
    """A synthesis result, a sensor weight result and a frontend output."""

    from support import ad_boundaries as ab
    from support import multi_endpoint_driver as drv
    from witwin.radar.synthesis import synthesize_fmcw_beat
    from witwin.radar.synthesis.contracts import SynthesisResult

    _, composed, _, _ = frame
    spec = drv.make_spec(num_chirps=2)
    cube = synthesize_fmcw_beat(drv.to_synthesis(composed), spec)
    assert cube.requires_grad, "the fixture must be live"
    _assert_clean("synthesis", SynthesisResult.from_fmcw_beat(cube, spec))

    # The sensor-weight owner, through its production entry point with a live
    # leaf, so the walked ``SensorWeightResult`` is one that HAS a tape behind
    # it rather than one that never built a graph.
    from witwin.radar.sensors.weights import SensorWeightResult

    boundary = ab.boundary("sensor_weight")
    captured: list = []
    original = SensorWeightResult.from_components

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        captured.append(value)
        return value

    # The RESULT is what this test needs and the boundary returns a scalar, so
    # the record is taken at the one place that builds it. Patching the
    # constructor rather than the entry point is deliberate: the owner imports
    # its entry point by value, so a module-attribute patch would never be seen.
    SensorWeightResult.from_components = capture
    try:
        leaf = boundary.leaf.detach().clone().requires_grad_(True)
        boundary.loss(leaf)
    finally:
        SensorWeightResult.from_components = original
    assert captured, "the sensor weight owner did not build a result"
    assert captured[0].weight.requires_grad, "the fixture must be live"
    _assert_clean("sensor_weight", captured[0])

    frontend = ab.boundary("frontend")
    from witwin.radar.frontend import (
        FrontendChain,
        FrontendSpec,
        LnaSpec,
        PortSpec,
        SeedSpec,
    )

    chain = FrontendChain(
        FrontendSpec(port=PortSpec(50.0), lna=LnaSpec(gain_db=10.0), seed=SeedSpec(5))
    )
    output = chain.apply(frontend.leaf.detach().clone().requires_grad_(True))
    assert output.signal.requires_grad
    _assert_clean("frontend", output)


@pytest.mark.gpu
def test_the_processing_results_carry_no_tape():
    """``PointCloud`` and ``DetectionFrame``, at the far side of the wall.

    These carry no graph by construction - the wall refuses a differentiable
    input - so the statement here is narrower than the one above and is worth
    making anyway: a stage that answered by handing back its own context would
    be doing it exactly where nobody is looking.
    """

    pytest.importorskip("witwin.channel")
    from support.pipeline_chain import pipeline_inputs, run_pipeline
    from witwin.radar.processing.tracking import DetectionFrame

    cloud = run_pipeline(*pipeline_inputs(num_chirps=4))
    _assert_clean("point_cloud", cloud)
    _assert_clean(
        "detection_frame",
        DetectionFrame.from_point_cloud(cloud, time_s=0.0, frame_index=0),
    )


def _simulate_once(radar=None, *, ad_mode: str = "vjp"):
    """One ``Radar.simulate`` frame, on a fresh radar unless one is given."""

    pytest.importorskip("witwin.channel")
    from support import multi_endpoint_driver as drv
    from support import multi_endpoint_geometry as geo
    from support import multi_endpoint_world as world
    from witwin.radar import Radar, ScatterSitePolicy
    from witwin.radar.scattering import ScalarRcsResponse

    if radar is None:
        radar = Radar(
            dict(geo.FIXTURE_RADAR_CONFIG),
            position=(0.0, 0.0, 0.0),
            target=(1.0, 0.0, 0.0),
        )
    scene, mesh = world.make_scene()
    world.assert_world_coordinates_survived(mesh)
    sites = torch.tensor(
        (geo.SITE_P_POSITION_M, geo.SITE_Q_POSITION_M),
        dtype=torch.float32,
        device=radar.device,
    ).requires_grad_(ad_mode != "none")
    result = radar.simulate(
        scene,
        times=(0.0,),
        response=ScalarRcsResponse.from_values(
            drv.FIXTURE_AMPLITUDE, drv.FIXTURE_PHASE_RAD, device=radar.device
        ),
        sites=ScatterSitePolicy.explicit(sites),
        ad_mode=ad_mode,
    )
    return radar, result


@pytest.fixture(scope="module")
def simulated():
    """One ``Radar.simulate`` run with a LIVE graph on its site positions.

    The scene-driven entry is a NEW retention site: four ``last_*`` properties
    on a long-lived ``Radar``, each holding a typed record from the last frame.
    A result that never built a graph could not leak a tape it never had, so
    this is taken under ``ad_mode='vjp'`` for the same reason the leg fixture
    above is.
    """

    return _simulate_once()


#: The retention rule for the scene-driven entry's diagnostics, stated once.
#:
#: The four ``last_*`` members are NOT detached. Detaching them would cost a
#: copy of every payload tensor and would make the one diagnostic a caller
#: reaches for while debugging a gradient the one that cannot carry it. What is
#: forbidden is the same thing forbidden everywhere else in this file: a FIELD
#: holding the tape - a context, a Function, a ``saved_tensors`` tuple - which
#: turns a data record into a handle on somebody else's memory. Holding a
#: ``grad_fn`` is not that, and a caller who wants the graph released simply
#: drops the result or runs another ``simulate``, which clears all four first.
DIAGNOSTIC_RETENTION_RULE = "aliased_and_live, never a tape field"


@pytest.mark.gpu
def test_the_simulate_diagnostics_carry_no_tape(simulated):
    """All four ``last_*`` attributes, walked, with a live graph behind them."""

    radar, result = simulated
    assert result.cube.requires_grad, "the fixture must be live"
    for name in (
        "last_snapshot",
        "last_compiled_scene",
        "last_propagation",
        "last_radar_paths",
    ):
        value = getattr(radar, name)
        assert value is not None, name
        _assert_clean(f"radar.{name}", value)
    _assert_clean("simulation_result", result)


@pytest.mark.gpu
def test_the_diagnostics_alias_the_frame_rather_than_a_detached_copy(simulated):
    """The retention rule above, pinned rather than merely written down.

    If a later change decides to detach these, this test is where the decision
    has to be re-made, and the rule constant next to it is what has to change
    with it.
    """

    radar, result = simulated
    assert radar.last_radar_paths is result.last_radar_paths
    assert radar.last_propagation is result.last_propagation
    assert result.last_radar_paths.complex_transfer_ref.requires_grad
    assert result.last_propagation.inbound.coefficient.requires_grad


@pytest.mark.gpu
def test_a_second_simulate_replaces_the_retained_frame():
    """The bound on the retention: there is only ever one live frame here.

    Its own radar, deliberately: this is the one test in the file that MUTATES
    the diagnostic state, and sharing the module fixture with it would make the
    other two depend on running first.
    """

    radar, live = _simulate_once()
    assert live.cube.requires_grad
    _, replacement = _simulate_once(radar, ad_mode="none")
    assert radar.last_result is replacement
    assert radar.last_radar_paths is not live.last_radar_paths
    assert not replacement.cube.requires_grad
    _assert_clean("replacement", replacement)


def test_the_walk_would_find_a_planted_tape():
    """Calibration. Without it the walker could be inspecting nothing.

    A record carrying a context is built on purpose and the walker has to
    object to it; then the same record without the context has to pass.
    """

    seen: list = []

    class Doubler(torch.autograd.Function):
        @staticmethod
        def forward(x):
            return x * 2

        @staticmethod
        def setup_context(ctx, inputs, output):
            seen.append(ctx)

    @dataclasses.dataclass
    class Leaky:
        value: torch.Tensor
        context: object

    leaf = torch.tensor([1.0], requires_grad=True)
    Doubler.apply(leaf)
    assert seen, "the planted Function must have run"

    found: list = []
    _walk(Leaky(value=leaf, context=seen[0]), "leaky", set(), found)
    assert found, "the walker missed a planted context"
    assert "context" in found[0][0]

    clean: list = []
    _walk(Leaky(value=leaf, context=None), "clean", set(), clean)
    assert not clean, clean
