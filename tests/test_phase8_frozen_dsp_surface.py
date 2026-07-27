"""Criterion 9 and criterion 7: the frozen DSP surface, and the Channel boundary.

Three scans, all static over the SOURCE, following the shape
``test_phase6_no_torch_physics.py`` already established, plus one runtime
assertion against the live Channel capability record.

1. **The frozen vendor DSP primitive list.** Every vendor DSP primitive the
   processing facade calls is enumerated here, by EQUALITY, so the list cannot
   grow silently and cannot quietly shrink either.

2. **The fence, and the allowance list.** ``tests/processing/test_cutover.py``
   already asserts that no DSP expression survives outside
   ``witwin/radar/processing/``. What is asserted here is the OTHER half of the
   criterion, which nothing else covers: the DSP exception list has not expanded
   into Radar physics or synthesis. The two lists it could expand into -
   ``test_cutover.FENCE_ALLOWANCES`` and
   ``test_phase6_no_torch_physics.RADAR_FACADE_TORCH_PHYSICS`` - are asserted by
   equality against the values recorded when the fence was built.

3. **No Radar processing field in the Channel capability record**, by keyword-set
   equality rather than by containment: a containment check passes the moment a
   field is added, which is the only way this ever goes wrong.

**What "vendor DSP primitive" means here**, stated because S4 asked and because
an unstated scope is a scope that drifts. In scope: TRANSFORMS, POOLING and
PATCH EXTRACTION, DECOMPOSITIONS and SOLVES, and ORDER STATISTICS / SELECTION.
Out of scope: elementwise arithmetic (``exp``, ``angle``, ``polar``, ``log10``,
``sqrt``), shape manipulation and construction (``arange``, ``stack``,
``zeros``, ``reshape``), contraction (``einsum``, ``tensordot`` - a contraction
is a sum of products, and the beamformer's is not a signal-processing
algorithm), and random sampling. Those are ordinary tensor operations; freezing
them would freeze arithmetic. The six adjacent calls a reader might nonetheless
mistake for DSP are recorded in :data:`RECORDED_ADJACENT_CALLS`, also by
equality.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSING = REPO_ROOT / "witwin" / "radar" / "processing"


# ---------------------------------------------------------------------------
# 1. The frozen vendor DSP primitive list
# ---------------------------------------------------------------------------

#: Transforms. Exactly the six the design named.
FROZEN_TORCH_FFT = frozenset(
    {"fft", "ifft", "fft2", "fftshift", "ifftshift", "fftfreq"}
)

#: Pooling, padding and patch extraction, from ``torch.nn.functional``.
FROZEN_FUNCTIONAL = frozenset({"avg_pool2d", "unfold", "pad"})

#: Decompositions and solves. ``solve`` is there because ``mvdr_weights`` must
#: never form an explicit inverse.
FROZEN_TORCH_LINALG = frozenset({"eigh", "solve"})

#: Order statistics and selection, whether written as ``torch.f(x)`` or as
#: ``x.f()``. The design's list was ``{topk, sort, cumsum, argwhere, where}``;
#: four entries were added when the facade was actually built, each with a
#: reason:
#:
#: * ``argsort`` - the tracker orders its association candidates;
#: * ``gather`` - the angle estimators read a peak out of a padded spectrum;
#: * ``index_select`` - the Doppler sign reconciliation is a frequency-index
#:   gather, chosen BECAUSE it has no arithmetic and is therefore exact;
#: * ``unfold`` (the TENSOR method, not ``F.unfold``) - the MUSIC sub-aperture
#:   view and the micro-Doppler framing. It replaced an ``(L + 1) ** 2``-way
#:   ``torch.stack`` over a list comprehension.
FROZEN_SELECTION = frozenset(
    {
        "argsort",
        "argwhere",
        "cumsum",
        "gather",
        "index_select",
        "sort",
        "topk",
        "unfold",
        "where",
    }
)

#: The vocabulary the selection scan looks for. A name in here that appears in
#: the facade must be in :data:`FROZEN_SELECTION`; a name in here that does NOT
#: appear must not be in it. That two-sided rule is what makes the frozen list a
#: statement about today rather than a wish list.
SELECTION_VOCABULARY = frozenset(
    {
        "argmax_pool",  # deliberately absent: guards the scan against typos
        "argsort",
        "argwhere",
        "bucketize",
        "cumsum",
        "cumprod",
        "cummax",
        "cummin",
        "gather",
        "histc",
        "histogram",
        "index_select",
        "kthvalue",
        "median",
        "mode",
        "msort",
        "nonzero",
        "quantile",
        "searchsorted",
        "sort",
        "topk",
        "unfold",
        "where",
    }
)

#: Vendor calls that are NOT DSP primitives but that a reader could mistake for
#: them. Recorded by equality so a change is deliberate, not so it is forbidden.
RECORDED_ADJACENT_CALLS = frozenset(
    {
        "torch.angle",
        "torch.einsum",
        "torch.polar",
        "torch.randint",
        "torch.randperm",
        "torch.tensordot",
    }
)


def _modules() -> list[pathlib.Path]:
    return sorted(PROCESSING.rglob("*.py"))


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _calls() -> list[tuple[pathlib.Path, ast.Call, str]]:
    found = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                found.append((path, node, _dotted(node.func)))
    return found


def _namespace(prefix: str) -> set[str]:
    return {
        name[len(prefix) :]
        for _, _, name in _calls()
        if name.startswith(prefix) and "." not in name[len(prefix) :]
    }


def test_the_transform_surface_is_exactly_the_frozen_list():
    """``torch.fft`` entries, by equality."""

    used = _namespace("torch.fft.")
    assert used == FROZEN_TORCH_FFT, sorted(used ^ FROZEN_TORCH_FFT)


def test_the_pooling_surface_is_exactly_the_frozen_list():
    """``torch.nn.functional`` entries, imported as ``F``, by equality."""

    used = _namespace("F.") | _namespace("torch.nn.functional.")
    assert used == FROZEN_FUNCTIONAL, sorted(used ^ FROZEN_FUNCTIONAL)


def test_the_decomposition_surface_is_exactly_the_frozen_list():
    """``torch.linalg`` entries, by equality. No explicit inverse anywhere."""

    used = _namespace("torch.linalg.")
    assert used == FROZEN_TORCH_LINALG, sorted(used ^ FROZEN_TORCH_LINALG)
    assert "inv" not in used
    assert "pinv" not in used


def test_the_selection_surface_is_exactly_the_frozen_list():
    """Order statistics and gathers, function form and method form together.

    A method call is included on purpose: ``x.cumsum(...)`` and
    ``torch.cumsum(x, ...)`` are the same primitive, and a scan that saw only
    the function form would report an empty CFAR integral image.
    """

    used = set()
    for _, node, name in _calls():
        if not isinstance(node.func, ast.Attribute):
            continue
        leaf = node.func.attr
        if leaf not in SELECTION_VOCABULARY:
            continue
        if name.startswith(("torch.fft.", "torch.linalg.", "F.", "torch.nn.functional.")):
            continue
        used.add(leaf)
    assert used == FROZEN_SELECTION, sorted(used ^ FROZEN_SELECTION)


def test_no_window_constructor_is_called_from_the_vendor_library():
    """The windows are built from ``arange`` and ``cos``, and that is the point.

    ``torch.hamming_window(N, periodic=False)`` and
    ``torch.hamming_window(N, periodic=True)`` are DIFFERENT sequences, and the
    difference is invisible at a call site. The facade owns one window family
    with an explicit periodic/symmetric distinction, so the vendor constructors
    are not on the frozen list and must not appear.
    """

    offenders = [
        (path.name, name)
        for path, _, name in _calls()
        if name.startswith("torch.") and name.split(".")[-1].endswith("_window")
    ]
    assert offenders == [], offenders
    # And the facade's own window owner is still there, so this is a scan for
    # the right thing rather than one that passes because nothing windows.
    from witwin.radar.processing.primitives import WINDOWS

    assert "hamming" in WINDOWS and "hamming_symmetric" in WINDOWS


def test_the_adjacent_vendor_calls_are_recorded_and_have_not_grown():
    """Not DSP, but named here so nobody has to decide again."""

    ordinary = {
        "torch.angle",
        "torch.einsum",
        "torch.polar",
        "torch.randint",
        "torch.randperm",
        "torch.tensordot",
        "torch.stft",
        "torch.istft",
        "torch.conv1d",
        "torch.conv2d",
        "torch.corrcoef",
        "torch.cov",
    }
    used = {name for _, _, name in _calls() if name in ordinary}
    assert used == RECORDED_ADJACENT_CALLS, sorted(used ^ RECORDED_ADJACENT_CALLS)


# ---------------------------------------------------------------------------
# 2. The fence, and the allowance lists
# ---------------------------------------------------------------------------

#: What ``tests/processing/test_cutover.py`` allowed when the fence was built:
#: one simulation owner, inverting a SYNTHESIZED spectrum into time samples.
EXPECTED_FENCE_ALLOWANCES = {"witwin/radar/solvers/solver_dirichlet.py"}

#: What ``tests/test_phase6_no_torch_physics.py`` recorded as surviving Torch
#: physics in the facade module. Phase 8 added nothing to it and must not.
EXPECTED_FACADE_TORCH_PHYSICS = {
    ("_normalize_rows", "torch.linalg.norm"),
    ("_set_pose_fields", "torch.linalg.norm"),
    ("_world_from_local_matrix", "torch.linalg.norm"),
    ("waveform", "torch.exp"),
    ("_apply_phase_noise", "torch.polar"),
}


def test_the_dsp_exception_list_has_not_expanded_into_physics_or_synthesis():
    """Criterion 9's second clause, which no other test states.

    The first clause - nothing scattered outside the facade - is
    ``test_cutover.py``'s. This one asserts that the two lists a DSP exception
    could be ADDED to have not been added to.
    """

    from processing import test_cutover
    from test_phase6_no_torch_physics import RADAR_FACADE_TORCH_PHYSICS

    assert set(test_cutover.FENCE_ALLOWANCES) == EXPECTED_FENCE_ALLOWANCES
    assert RADAR_FACADE_TORCH_PHYSICS == EXPECTED_FACADE_TORCH_PHYSICS


def test_no_synthesis_or_physics_module_calls_a_frozen_dsp_primitive():
    """The transforms, stated over the SIMULATION packages by name.

    ``test_cutover`` scans the whole tree outside the facade for a fixed list of
    calls. This one scans the four owner packages for the FROZEN list, so a
    primitive added to the frozen list is automatically forbidden in physics
    too rather than needing a second edit.
    """

    packages = ("solvers", "synthesis", "sensors", "frontend", "paths", "propagation")
    allowed = EXPECTED_FENCE_ALLOWANCES
    scanned = 0
    offenders = []
    for package in packages:
        root = REPO_ROOT / "witwin" / "radar" / package
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative in allowed:
                continue
            scanned += 1
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _dotted(node.func)
                if name.startswith("torch.fft."):
                    offenders.append((relative, name))
                if name.startswith("torch.linalg.") and name.split(".")[-1] in (
                    FROZEN_TORCH_LINALG
                ):
                    offenders.append((relative, name))
                if name.startswith(("F.", "torch.nn.functional.")) and name.split(".")[
                    -1
                ] in FROZEN_FUNCTIONAL:
                    offenders.append((relative, name))
    assert offenders == [], offenders
    # Non-vacuity: the six packages exist and were actually walked.
    assert scanned >= 25, scanned


# ---------------------------------------------------------------------------
# 3. No Radar processing field in the Channel capability record
# ---------------------------------------------------------------------------

#: Every field of the live ``PropagationCapabilities`` record, by EQUALITY.
#:
#: This is the plan's seventh criterion made mechanical. A processing field -
#: a range-bin count, a Doppler-bin count, a CFAR parameter, an angle grid, a
#: subcarrier count, an FFT size, a bandwidth - would have to be ADDED here to
#: pass, which is the point: a new field becomes visible rather than tolerated.
#:
#: ``max_frequency_offset_count``, ``wideband_*`` and
#: ``native_frequency_resolution_law`` are ADR-042's, and they are propagation
#: frequencies in Hz. A frequency at which a field is evaluated is propagation
#: vocabulary; a subcarrier COUNT is not, and the ADR refuses one.
#:
#: ``ad_accounting``, ``component_material_leaves``,
#: ``differentiable_geometry_outputs``, ``direction_differentiable_components``,
#: ``primal_only_ad_inputs`` and ``supports_higher_order_ad`` are ADR-043's.
#: They are AD vocabulary over propagation leaves and propagation outputs; not
#: one of them names a processing stage, a bin count, or a detector parameter.
EXPECTED_CAPABILITY_FIELDS = frozenset(
    {
        "ad_accounting",
        "ad_modes",
        "component_ad_modes",
        "component_material_leaves",
        "components",
        "contract_version",
        "differentiable_geometry_outputs",
        "direction_differentiable_components",
        "fixed_topology_components",
        "fixed_topology_responses",
        "fixed_topology_row_validity_components",
        "max_frequency_offset_count",
        "max_slot_count",
        "native_frequency_resolution_law",
        "polarimetric_frozen_ad_inputs",
        "primal_only_ad_inputs",
        "response_ad_modes",
        "response_components",
        "responses",
        "supports_fixed_topology",
        "supports_higher_order_ad",
        "supports_los_jones",
        "supports_slot_batching",
        "supports_wideband_offsets",
        "topology_modes",
        "wideband_components",
        "wideband_dispersive_materials",
        "wideband_responses",
        "wideband_rough_materials",
        "world_motions",
        "world_version_domains",
    }
)

#: Processing vocabulary. Not one of these substrings may appear in a Channel
#: capability field name, at any nesting level of the published record.
PROCESSING_VOCABULARY = (
    "range_bin",
    "doppler_bin",
    "num_range",
    "num_doppler",
    "num_angle",
    "angle_bin",
    "cfar",
    "guard_cell",
    "training_cell",
    "pfa",
    "false_alarm",
    "beamform",
    "steering",
    "point_cloud",
    "detection",
    "track",
    "subcarrier",
    "fft_size",
    "num_fft",
    "window",
    "bandwidth",
    "sample_rate",
    "chirp",
    "symbol",
    "pulse",
    "adc",
    "velocity_bin",
)


def test_the_channel_capability_record_publishes_exactly_these_fields():
    """Keyword-set equality against the LIVE record."""

    pytest.importorskip("witwin.channel")
    from witwin.channel.propagation import consumer

    fields = frozenset(
        field.name for field in dataclasses.fields(consumer.capabilities())
    )
    assert fields == EXPECTED_CAPABILITY_FIELDS, sorted(
        fields ^ EXPECTED_CAPABILITY_FIELDS
    )


def test_no_processing_vocabulary_appears_anywhere_in_the_channel_capabilities():
    """The whole published record, walked, including the nested solver blocks.

    The consumer record is one dataclass; the package-level ``capabilities()``
    is a nested dict that EMBEDS it. Walking both is what makes the criterion a
    statement about Channel's public capability surface rather than about one
    record that happens to be clean.
    """

    pytest.importorskip("witwin.channel")
    import witwin.channel as channel
    from witwin.channel.propagation import consumer

    def walk(value, trail=""):
        offenders = []
        if isinstance(value, dict):
            for key, item in value.items():
                name = str(key).lower()
                for token in PROCESSING_VOCABULARY:
                    if token in name:
                        offenders.append((f"{trail}.{key}", token))
                offenders += walk(item, f"{trail}.{key}")
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                name = field.name.lower()
                for token in PROCESSING_VOCABULARY:
                    if token in name:
                        offenders.append((f"{trail}.{field.name}", token))
                offenders += walk(getattr(value, field.name), f"{trail}.{field.name}")
        return offenders

    assert walk(consumer.capabilities(), "consumer") == []
    assert walk(channel.capabilities(), "channel") == []


def test_the_native_binding_manifest_carries_no_processing_symbol():
    """Criterion 7's ABI half, on the RADAR manifest.

    Phase 8 added no native symbol at all - the manifest is asserted UNCHANGED
    by ``test_phase4_binding_manifest.py`` - so what is asserted here is the
    narrower and more durable claim: no manifest entry names a processing stage.
    """

    import json

    manifest = json.loads(
        (REPO_ROOT / "ci" / "native-binding-manifest.json").read_text(encoding="utf-8")
    )
    blob = json.dumps(manifest).lower()
    for token in (
        "cfar",
        "beamform",
        "point_cloud",
        "range_profile",
        "range_doppler",
        "music",
        "aoa",
        "steering",
        "track",
    ):
        assert token not in blob, token
