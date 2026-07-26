"""Waveform, ADC, and receiver vocabulary never reaches a propagation request.

Work item 6's assertion, in two forms that catch different mistakes.

The RUNTIME test captures the keyword sets a real ``freeze`` and a real
``reevaluate`` build and compares them by EQUALITY. A containment check would
pass the moment a field is added, which is the only way this ever goes wrong:
nobody writes ``slope=...`` into a propagation request on purpose, they widen a
config object that a request happens to splat.

The STATIC test scans every module under ``witwin/radar/propagation/`` for the
waveform and frontend vocabulary and requires zero hits. It catches the mistake
the runtime test cannot: a field that is read but not forwarded, which changes
what the propagation layer KNOWS even when it does not change what it sends.

Both run on CPU. The consumer's request constructors are captured and its three
entry points are stubbed, so this is a test of what Radar builds rather than of
what Channel does with it - which is the boundary actually under discussion.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROPAGATION_ROOT = REPO_ROOT / "witwin" / "radar" / "propagation"

#: The exact keyword sets a propagation request is allowed to carry. Frozen by
#: equality: ``reference_frequency_hz`` is the ONE legitimate crossing and
#: everything else here is geometry, topology, or an AD mode.
DISCOVERY_KEYWORDS = frozenset(
    {
        "sources",
        "sinks",
        "reference_frequency_hz",
        "components",
        "max_depth",
        "response",
        "topology_mode",
        "ad_mode",
    }
)
#: ``slot_count`` is how many stacked time slots the endpoint batches carry. It
#: is a batch SHAPE, in the same family as the endpoint counts, and it says
#: nothing about chirps, symbols or pulses: a caller that maps TDM slots onto it
#: does that mapping in ``witwin.radar.synthesis``, and propagation never learns
#: which waveform asked. That is why it may cross where ``chirp_period_s`` may
#: not.
REEVALUATION_KEYWORDS = frozenset(
    {
        "sources",
        "sinks",
        "reference_frequency_hz",
        "topology",
        "response",
        "ad_mode",
        "slot_count",
        # What the caller declares happened to the WORLD between discovery and
        # this replay - ``"frozen_world"`` or ``"fixed_winner_replay"``. It
        # describes scene geometry, which is precisely what a propagation
        # request is about, and it carries no waveform, ADC or receive-chain
        # vocabulary. It is what lets a moving structure be REPLAYED instead of
        # silently answered out of geometry that moved on.
        "world_motion",
    }
)

#: Vocabulary that describes a waveform, an ADC, or a receive chain. None of it
#: has any business inside a propagation module: propagation answers "which
#: paths exist and what does each one do to a narrowband phasor", and none of
#: those words is part of that question.
FORBIDDEN_VOCABULARY = (
    "waveform",
    "chirp",
    "slope",
    "sample_rate",
    "adc",
    "agc",
    "lna",
    "subcarrier",
    "symbol_period",
    "pulse",
    "pri",
    "noise",
    "quantiz",
    "receiver_chain",
    "full_scale",
)


def _populated_config():
    """A configuration with every one of the five blocks non-trivial.

    A boundary test on a minimal configuration proves nothing: the fields that
    could leak are precisely the ones a minimal configuration leaves out.
    """

    from conftest import STANDARD_CONFIG
    from witwin.radar.config import RadarSystemConfig
    from witwin.radar.validation import validate_frontend_config, validate_radar_config

    flat = validate_radar_config(
        {
            **STANDARD_CONFIG,
            "polarization": {"tx": "vertical", "rx": "horizontal"},
        }
    )
    frontend = validate_frontend_config(
        {
            "port": {"reference_impedance_ohm": 50.0},
            "noise": {
                "noise_figure_db": 6.0,
                "bandwidth_hz": 4.4e6,
                "phase_noise_dbc_per_hz": -90.0,
                "phase_offset_hz": 1e5,
                "phase_sample_rate_hz": 4.4e6,
            },
            "lna": {"gain_db": 20.0},
            "agc": {"target_rms": 0.5, "mode": "per_rx"},
            "adc": {"bits": 12, "full_scale": 1.0},
            "seed": 4,
        }
    )
    return RadarSystemConfig.from_radar_config(flat, frontend=frontend)


def _endpoints(count: int, *, role: str):
    from witwin.radar.propagation.contracts import RadarEndpointSpec

    positions = torch.zeros(count, 3, dtype=torch.float32)
    positions[:, 2] = -torch.arange(1, count + 1, dtype=torch.float32)
    polarizations = torch.zeros(count, 3, dtype=torch.float32)
    polarizations[:, 1] = 1.0
    return RadarEndpointSpec(
        stable_ids=torch.arange(count, dtype=torch.int64),
        positions_m=positions,
        polarizations=polarizations,
        powers_w=(
            torch.ones(count, dtype=torch.float32) if role == "source" else None
        ),
    )


class _Recorder:
    """Captures the keyword set of every request the adapter builds."""

    def __init__(self) -> None:
        self.discovery: list[frozenset[str]] = []
        self.reevaluation: list[frozenset[str]] = []
        self.reference_frequencies: list[float] = []

    def propagation_request(self, **kwargs):
        self.discovery.append(frozenset(kwargs))
        self.reference_frequencies.append(kwargs["reference_frequency_hz"])
        return kwargs

    def fixed_topology_request(self, **kwargs):
        self.reevaluation.append(frozenset(kwargs))
        self.reference_frequencies.append(kwargs["reference_frequency_hz"])
        return kwargs


def _install_stubs(monkeypatch, recorder, *, rows: int, pairs: int):
    """Stub the three consumer entry points around the captured constructors.

    Capturing the request constructors necessarily breaks the real call, so the
    entry points have to be stubbed too. That is not a weakening: what this file
    is asserting is what RADAR builds, and Channel's behaviour on a real request
    is pinned by the Phase-4 and Phase-5 adapter tests.
    """

    from witwin.channel.propagation import consumer

    class _Topology:
        source_id = torch.arange(rows, dtype=torch.int64)
        sink_id = torch.arange(rows, dtype=torch.int64)
        # int32 for the index-like columns and int64 for the stable IDs,
        # matching what RadarLegBatch validates. A stub that got these wrong
        # would be tested against nothing.
        component_id = torch.zeros(rows, dtype=torch.int32)
        depth = torch.zeros(rows, dtype=torch.int32)
        primitive_sequence = torch.zeros(rows, 1, dtype=torch.int32)
        material_sequence = torch.zeros(rows, 1, dtype=torch.int32)
        interaction_type = torch.zeros(rows, 1, dtype=torch.int32)
        source_index = torch.zeros(rows, dtype=torch.int32)
        sink_index = torch.zeros(rows, dtype=torch.int32)

    class _Bucket:
        component = "los"

    class _Prepared:
        buckets = (_Bucket(),)
        prepare_d2h_copies = 0
        prepare_d2h_bytes = 0
        prepare_synchronizations = 0

    class _Paths:
        topology = _Topology()
        path_count = rows
        pair_count = pairs
        pair_index = torch.zeros(rows, dtype=torch.int64)
        # A CSR partition needs `pairs + 1` boundaries. Every row is put in
        # the last pair, which is legal and exercises the empty-segment case
        # the multi-endpoint fixture also produces.
        pair_offsets = torch.tensor([0] * pairs + [rows], dtype=torch.int64)
        geometry = type(
            "_Geometry",
            (),
            {
                "delay_s": torch.zeros(rows, dtype=torch.float32),
                # The consumer's geometry always publishes a propagation
                # direction and the adapter always passes it through; a stub
                # that omitted it would make this file the only place the
                # adapter is exercised against a geometry that does not exist.
                "field_direction": torch.zeros(rows, 3, dtype=torch.float32),
            },
        )()
        transport = type(
            "_Transport",
            (),
            {"coefficient": torch.zeros(rows, dtype=torch.complex64)},
        )()

    class _Evaluation:
        paths = _Paths()

    class _Result:
        paths = _Paths()
        row_valid = torch.ones(rows, dtype=torch.bool)
        diagnostics = None

    monkeypatch.setattr(consumer, "PropagationRequest", recorder.propagation_request)
    monkeypatch.setattr(
        consumer, "FixedTopologyRequest", recorder.fixed_topology_request
    )
    monkeypatch.setattr(consumer, "EndpointBatch", lambda **kwargs: kwargs)
    monkeypatch.setattr(consumer, "evaluate", lambda scene, request: _Evaluation())
    monkeypatch.setattr(
        consumer, "prepare_fixed_topology", lambda topology: _Prepared()
    )
    monkeypatch.setattr(consumer, "reevaluate", lambda scene, request: _Result())


def _adapter(system_config):
    from witwin.radar.propagation.channel_consumer import ChannelPropagationAdapter

    block = system_config.propagation
    return ChannelPropagationAdapter(
        compiled_scene=object(),
        reference_frequency_hz=block.reference_frequency_hz,
        components=block.components,
        max_depth=block.max_depth,
    )


# ---------------------------------------------------------------------------
# T4.14 - runtime, positive
# ---------------------------------------------------------------------------


def test_a_fully_populated_config_sends_exactly_the_allowed_keywords(monkeypatch):
    """Equality, not containment, on both request shapes.

    The captured sets are printed on failure so the offending field names it
    itself rather than leaving the reader to diff two frozensets by eye.
    """

    pytest.importorskip("witwin.channel")

    system_config = _populated_config()
    recorder = _Recorder()
    _install_stubs(monkeypatch, recorder, rows=4, pairs=2)

    adapter = _adapter(system_config)
    sources = _endpoints(2, role="source")
    sinks = _endpoints(2, role="sink")
    frozen = adapter.freeze(sources, sinks)
    adapter.reevaluate(frozen, sources, sinks, ad_mode="none")

    assert recorder.discovery == [DISCOVERY_KEYWORDS], sorted(
        recorder.discovery[0] ^ DISCOVERY_KEYWORDS
    )
    assert recorder.reevaluation == [REEVALUATION_KEYWORDS], sorted(
        recorder.reevaluation[0] ^ REEVALUATION_KEYWORDS
    )


def test_the_adapter_constructor_accepts_only_the_propagation_block():
    """Folding a waveform field into the adapter is not writable.

    The constructor takes three scalars from ``PropagationConfig`` and nothing
    that could carry a waveform, which is what makes the leak structurally
    impossible rather than merely absent today.
    """

    import inspect

    from witwin.radar.propagation.channel_consumer import ChannelPropagationAdapter

    parameters = set(
        inspect.signature(ChannelPropagationAdapter.__init__).parameters
    ) - {"self"}
    assert parameters == {
        "compiled_scene",
        "reference_frequency_hz",
        "components",
        "max_depth",
    }, sorted(parameters)


# ---------------------------------------------------------------------------
# T4.16 - OFDM does not multiply the reference frequency
# ---------------------------------------------------------------------------


def test_an_ofdm_band_still_produces_exactly_one_reference_frequency(monkeypatch):
    """64 subcarriers, one ``reference_frequency_hz``, unchanged by the count.

    A per-subcarrier set of reference frequencies is what an OFDM band looks like
    to somebody who has not read the narrowband law. It is not that: the band is
    a per-subcarrier phase the Radar synthesis kernel applies from ONE
    coefficient, and a wideband material response is Phase-8 work that the
    synthesis contract declares and refuses rather than silently ignores.
    """

    pytest.importorskip("witwin.channel")

    from dataclasses import replace

    from witwin.radar.config import OfdmWaveformConfig

    base = _populated_config()
    reference = base.propagation.reference_frequency_hz
    for count in (1, 64, 512):
        system_config = replace(
            base,
            waveform=OfdmWaveformConfig(
                subcarrier_spacing_hz=120e3,
                num_subcarriers=count,
                cyclic_prefix_s=2e-6,
                num_symbols=16,
                max_expected_delay_s=1e-6,
            ),
        )
        assert system_config.kind == "ofdm"
        spec = system_config.waveform_spec()
        assert spec.num_subcarriers == count
        assert spec.reference_frequency_hz == reference

        recorder = _Recorder()
        _install_stubs(monkeypatch, recorder, rows=4, pairs=2)
        adapter = _adapter(system_config)
        sources = _endpoints(2, role="source")
        sinks = _endpoints(2, role="sink")
        frozen = adapter.freeze(sources, sinks)
        adapter.reevaluate(frozen, sources, sinks, ad_mode="none")
        assert recorder.reference_frequencies == [reference, reference], (
            count,
            recorder.reference_frequencies,
        )


# ---------------------------------------------------------------------------
# T4.15 - static, negative
# ---------------------------------------------------------------------------


#: Vocabulary matched as WHOLE tokens rather than as substrings. The
#: distinction is not pedantry: ``primitive_sequence`` contains ``pri`` and
#: ``slope`` is a substring of nothing useful, so a substring scan reports the
#: propagation layer for naming a primitive sequence and the test stops meaning
#: anything the day someone silences it.
_FORBIDDEN_TOKENS = frozenset(
    {
        "waveform",
        "chirp",
        "slope",
        "adc",
        "agc",
        "lna",
        "subcarrier",
        "pulse",
        "pri",
        "noise",
    }
)
_FORBIDDEN_PREFIXES = ("quantiz",)
_FORBIDDEN_PHRASES = (
    ("sample", "rate"),
    ("symbol", "period"),
    ("receiver", "chain"),
    ("full", "scale"),
)


def _names_forbidden_vocabulary(text: str) -> bool:
    tokens = [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]
    if any(token in _FORBIDDEN_TOKENS for token in tokens):
        return True
    if any(token.startswith(_FORBIDDEN_PREFIXES) for token in tokens):
        return True
    pairs = set(zip(tokens, tokens[1:]))
    return bool(pairs & set(_FORBIDDEN_PHRASES))


def _identifier_text(path: pathlib.Path) -> list[tuple[str, str]]:
    """Every identifier, attribute, and string literal a module names.

    Scanned with the AST rather than as text, because these words appear in
    these files' PROSE - documenting the rule is not breaking it - and a text
    scan would forbid the explanation along with the act.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.append(("name", node.id))
        elif isinstance(node, ast.Attribute):
            found.append(("attribute", node.attr))
        elif isinstance(node, ast.arg):
            found.append(("argument", node.arg))
        elif isinstance(node, ast.keyword) and node.arg is not None:
            found.append(("keyword", node.arg))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Docstrings are the prose this scan must not police. Every other
            # string literal is a real name: a response label, a mode, a key.
            found.append(("string", node.value))
    return found


def test_no_propagation_module_names_waveform_or_frontend_vocabulary():
    modules = sorted(PROPAGATION_ROOT.glob("*.py"))
    assert modules, "the propagation package moved"

    docstrings = set()
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(
                node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                text = ast.get_docstring(node, clean=False)
                if text is not None:
                    docstrings.add(text)

    offenders: list[tuple[str, str, str]] = []
    for path in modules:
        for kind, text in _identifier_text(path):
            if kind == "string" and text in docstrings:
                continue
            if _names_forbidden_vocabulary(text):
                offenders.append((path.name, kind, text))
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# The facade: dispatch on a STORED discriminator, with no fallback
# ---------------------------------------------------------------------------


def _synthesis_batch(radar, *, rows: int = 3):
    """A Channel-provenance batch, built directly rather than through a solve.

    What is under test here is DISPATCH, not physics, so the batch is the
    smallest thing that satisfies the contract: three rows in one sensor pair
    with Channel's provenance, which is what makes the frozen-weight carrier
    rule the applicable one.
    """

    from witwin.radar.paths.contracts import RadarPathTopology
    from witwin.radar.synthesis import SlowTimeMode, SynthesisPathBatch

    device = radar.device
    ids = torch.zeros(rows, dtype=torch.int64, device=device)
    return SynthesisPathBatch(
        sensor_pair_count=1,
        path_count=rows,
        sensor_pair_index=ids.clone(),
        pair_offsets=torch.tensor([0, rows], dtype=torch.int64, device=device),
        total_delay_s=torch.full((rows,), 2e-8, dtype=torch.float32, device=device),
        delay_rate=torch.zeros(rows, dtype=torch.float32, device=device),
        complex_transfer_ref=torch.ones(rows, dtype=torch.complex64, device=device),
        reference_frequency_hz=float(radar.config.fc),
        frequency_response=None,
        frequency_offsets_hz=None,
        topology=RadarPathTopology(
            radar_source_id=ids.clone(),
            site_id=ids.clone(),
            radar_sink_id=ids.clone(),
            inbound_row=ids.clone(),
            outbound_row=ids.clone(),
        ),
        row_valid=None,
        join_mode="multipath",
        weight_includes_reference_phase=True,
        weight_includes_spreading=True,
        weight_includes_tx_power=True,
        slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
    )


@pytest.mark.gpu
def test_synthesize_dispatches_on_the_stored_waveform_kind():
    """One entry point, three waveforms, each with its own axes and phasor.

    The FMCW cube is CONJUGATED relative to the other two and the result says
    so, which is the point of carrying a convention as data: a consumer that
    compared two of these products without reading it would find a sign error
    and conclude that the physics disagreed.
    """

    from dataclasses import replace

    from conftest import MINIMAL_CONFIG, make_radar_or_skip
    from witwin.radar.config import OfdmWaveformConfig, PulsedWaveformConfig
    from witwin.radar.synthesis import BEAT_PHASOR, CHANNEL_PHASOR, SlowTimeMode

    radar = make_radar_or_skip(MINIMAL_CONFIG)
    batch = _synthesis_batch(radar)
    mode = SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE

    beat = radar.synthesize(batch, slow_time_mode=mode)
    assert beat.kind == "fmcw"
    assert beat.axes == ("chirp", "sensor_pair", "sample")
    assert beat.phasor == BEAT_PHASOR
    assert beat.cube.shape == (
        radar.config.chirp_per_frame,
        1,
        radar.config.adc_samples,
    )

    radar.system_config = replace(
        radar.system_config,
        waveform=OfdmWaveformConfig(
            subcarrier_spacing_hz=120e3,
            num_subcarriers=16,
            cyclic_prefix_s=2e-6,
            num_symbols=4,
            max_expected_delay_s=1e-6,
        ),
    )
    cfr = radar.synthesize(batch, slow_time_mode=mode)
    assert cfr.kind == "ofdm"
    assert cfr.axes == ("symbol", "sensor_pair", "subcarrier")
    assert cfr.phasor == CHANNEL_PHASOR
    assert cfr.cube.shape == (4, 1, 16)

    radar.system_config = replace(
        radar.system_config,
        waveform=PulsedWaveformConfig(
            pulse_kind="lfm",
            pulse_width_s=1e-6,
            bandwidth_hz=2e7,
            pri_s=1e-4,
            num_pulses=4,
            sample_rate_hz=5e7,
            num_samples=256,
            range_gate_start_s=0.0,
            max_expected_delay_rate=0.0,
        ),
    )
    train = radar.synthesize(batch, slow_time_mode=mode)
    assert train.kind == "pulsed"
    assert train.axes == ("pulse", "sensor_pair", "sample")
    assert train.phasor == CHANNEL_PHASOR
    assert train.cube.shape == (4, 1, 256)


@pytest.mark.gpu
def test_an_unknown_waveform_kind_is_a_hard_error_and_never_a_fallback():
    """A kind with no owner has no physics; returning a plausible cube is worse.

    Dispatch is a dict lookup on a stored discriminator, so this is the only
    shape the failure can take. A ``try``/``except`` around three synthesis calls
    would have picked whichever one did not raise.
    """

    from dataclasses import dataclass, replace

    from conftest import MINIMAL_CONFIG, make_radar_or_skip
    from witwin.radar.synthesis import SlowTimeMode

    @dataclass(frozen=True, slots=True)
    class _Unowned:
        kind = "stepped_frequency"

    radar = make_radar_or_skip(MINIMAL_CONFIG)
    batch = _synthesis_batch(radar)
    radar.system_config = object.__new__(type(radar.system_config))
    object.__setattr__(radar.system_config, "waveform", _Unowned())
    with pytest.raises(ValueError, match="no synthesis owner"):
        radar.synthesize(
            batch, slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
        )


def test_a_frontend_block_and_a_legacy_runtime_cannot_both_be_configured():
    """Two chains would be two answers about where the LNA sits.

    The legacy pair could not say this about itself - that is the whole reason
    the frontend block exists - so the refusal lives where the two meet.
    """

    from witwin.radar import RadarConfig
    from witwin.radar.radar import Radar
    from witwin.radar.validation import validate_frontend_config, validate_radar_config

    from conftest import MINIMAL_CONFIG

    flat = validate_radar_config(
        {**MINIMAL_CONFIG, "noise_model": {"thermal": {"std": 1e-6}}}
    )
    frontend = validate_frontend_config(
        {"noise": {"noise_figure_db": 3.0, "bandwidth_hz": 1e6}}
    )
    import dataclasses

    with pytest.raises(ValueError, match="frontend replaces"):
        Radar._validate_runtime_config(
            dataclasses.replace(flat, frontend=frontend)
        )
    assert isinstance(flat, RadarConfig)
