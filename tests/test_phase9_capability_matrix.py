"""The matrix document is a fixture, and this is what makes it authoritative.

``docs/dev/radar-ad-capability-matrix.md`` is the Phase-9 deliverable that says
which AD cells are supported, which are structurally zero, which are refused and
which are declared non-differentiable outputs. A document nobody executes rots
in exactly one direction: a test gets renamed, a row keeps citing it, and the
matrix quietly becomes a description of a tree that no longer exists.

So the document is parsed here, not read. Every row's state, mechanism, mode and
validation has to be inside a closed vocabulary; every row's ``test`` cell has to
be non-empty and has to resolve to a function that exists; every row's ``owner``
path has to exist; every mirrored Channel row has to agree with the live
``capabilities()`` record; and every section's row count is frozen, so adding a
leaf without adding its row fails here rather than passing silently.

**There is no ``TODO`` state and no ``SILENT`` state.** That is the whole point
of the phase: a cell nobody decided is a defect, not a status. A cell we are
deliberately not doing is ``REF`` or ``DECL`` with a named deferral in the
document's own "Deferred" section.

**How the test resolution works, and its limit.** Node ids resolve by parsing
the cited module with :mod:`ast` and looking for the function name. That
deliberately does not import the test module - importing every test module at
collection time would run their module-level fixtures and CUDA imports - and it
deliberately does not shell out to ``pytest --collect-only``, which costs a
process per run. The limit is that a parametrized id is matched by its function
name rather than by its parameter set, which is what a node id without brackets
means anyway.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "dev" / "radar-ad-capability-matrix.md"

HEADER = (
    "| route | leaf-or-output | mode | state | mechanism | owner | test | validation |"
)

#: The four target states of ``01-design.md`` section 1. ``SILENT`` is not one
#: of them and never becomes one.
STATES = frozenset({"SUP", "ZERO", "REF", "DECL"})

MECHANISMS = frozenset(
    {
        "native-companion",
        "native-declared",
        "torch-orchestration",
        "host-declaration",
    }
)

VALIDATIONS = frozenset(
    {"fd", "oracle-f64", "analytic", "adjoint", "declaration", "refusal"}
)

MODES = frozenset({"jvp", "vjp", "both"})

#: The validations that measure a derivative rather than assert a structure.
NUMERICAL_VALIDATIONS = frozenset({"fd", "oracle-f64", "analytic", "adjoint"})

#: The ``SUP`` rows whose claim is structural or a liveness statement - a
#: compact row identity, an autograd context and its saved-tensor count, a
#: reduction order, a tape non-leak, "this legacy route still publishes a
#: nonzero gradient" - and which are therefore proved by a structural test
#: rather than by an oracle. Frozen by ``(route, leaf-or-output, mode)`` so that
#: flipping some OTHER row to ``SUP`` and pointing it at a declaration-style
#: test fails here instead of quietly advertising an unmeasured derivative.
STRUCTURAL_SUP_ROWS = frozenset(
    {
        ("reevaluate/prepared", "out:field_direction", "both"),
        ("join/_compose_band", "out:autograd context aliasing", "vjp"),
        (
            "kinematics/two_way_duals",
            "position leaf beside a velocity tangent",
            "both",
        ),
        ("legacy-scene/SMPLBody", "pose", "vjp"),
        ("sensor_weight/evaluate", "antenna position reduction order", "vjp"),
        ("chain/any", "one dual level over all three endpoint sets", "jvp"),
        (
            "reevaluate/prepared",
            "out:compact row identity across none, jvp and vjp",
            "both",
        ),
        (
            "reevaluate/prepared",
            "out:topology identity, scene-leaf compile against the shared one",
            "both",
        ),
        ("frontend/noise", "out:the Philox realisation under AD", "vjp"),
        (
            "tape/two_way",
            "out:join context, 10 saved tensors, one launch each way",
            "vjp",
        ),
        (
            "tape/aspect",
            "out:aspect context, 9 saved tensors, one launch each way",
            "vjp",
        ),
        (
            "tape/fmcw_beat",
            "out:beat context, backward saves segment where forward saves offsets",
            "both",
        ),
        ("tape/ofdm_cfr", "out:cfr context, same forward/backward asymmetry", "both"),
        (
            "tape/pulsed_echo",
            "out:echo context, same forward/backward asymmetry",
            "both",
        ),
        (
            "tape/sensor_weight",
            "out:weight context, 9 saved tensors, one launch each way",
            "vjp",
        ),
        (
            "tape/frontend",
            "out:noise and AGC contexts, two owners in one call",
            "vjp",
        ),
        (
            "tape/compose_band",
            "out:tape bytes as a linear law in the band column count",
            "vjp",
        ),
        ("tape/any", "out:no tape reaches a public result record", "both"),
        ("tape/any", "out:no module outside an owner reads a context", "both"),
        ("reevaluate/prepared", "out:ad_companion_launches, ad_tape_bytes", "vjp"),
    }
)

#: Rows per document section, frozen. A new leaf without a row changes a number
#: here; so does a deleted row. Both are deliberate acts and both should show up
#: in a diff rather than in nobody's attention.
SECTION_ROWS = {
    "Mirrored Channel rows": 11,
    "Aspect scatter response (`witwin/radar/scattering.py`)": 6,
    "Two-way join and the wideband band (`witwin/radar/paths.py`)": 5,
    "Kinematics (`witwin/radar/propagation.py`)": 8,
    "SMPL authoring (`witwin/radar/smpl.py`)": 6,
    # 12 until Phase 11 deleted DirichletSpectrumSpec with its route.
    "The host-float rule (`witwin/radar/policy.py`)": 11,
    # 10 leaves of the family itself, plus the three Phase-11 rows for the
    # PRODUCTION route that reaches it, `sensors.py`. The section
    # keeps the family's name because there is still one numerical owner.
    "Sensor weight (`witwin/radar/sensors.py`)": 13,
    "Scatter response (`witwin/radar/scattering.py`)": 6,
    "FMCW beat synthesis (`witwin/radar/synthesis/fmcw.py`)": 5,
    "End-to-end waveform chains (`tests/support/waveform_chains.py`)": 9,
    "Frontend chain (`witwin/radar/frontend.py`)": 1,
    "Above the wall (`SUP`)": 7,
    "Below the wall (`REF`)": 16,
    "Higher order: first derivatives only, everywhere": 13,
    "The combined-input matrix: one scenario, one frozen topology": 11,
    "Row validity: a row that stops existing": 8,
    "Refused tangents, driven through the whole chain": 10,
    "The four chains that had no AD coverage": 9,
    # 12 until Phase 11 deleted the dirichlet_spectrum tape row.
    "Tape ownership and the budget pins": 11,
}

TOTAL_ROWS = sum(SECTION_ROWS.values())


class Row:
    """One matrix row, with the section it was written under."""

    __slots__ = (
        "line",
        "section",
        "route",
        "leaf",
        "mode",
        "state",
        "mechanism",
        "owner",
        "test",
        "validation",
    )

    def __init__(self, line: int, section: str, cells: list[str]) -> None:
        self.line = line
        self.section = section
        (
            self.route,
            self.leaf,
            self.mode,
            self.state,
            self.mechanism,
            self.owner,
            self.test,
            self.validation,
        ) = cells

    def __repr__(self) -> str:  # pragma: no cover - failure messages only
        return f"<row line {self.line} {self.section!r} {self.route}/{self.leaf}>"


def _parse():
    """Every table row under the matrix header, with its section heading."""

    rows = []
    malformed = []
    section = "<no section>"
    inside = False
    for number, raw in enumerate(DOC.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            inside = False
            continue
        if line == HEADER:
            inside = True
            continue
        if not inside:
            continue
        if line.startswith("|---"):
            continue
        if not line.startswith("|"):
            inside = False
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 8:
            malformed.append((number, len(cells)))
            continue
        rows.append(Row(number, section, cells))
    return rows, malformed


@pytest.fixture(scope="module")
def parsed():
    rows, malformed = _parse()
    assert not malformed, malformed
    return rows


def _functions(path: pathlib.Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return frozenset(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


# ---------------------------------------------------------------------------
# 1. The document exists and parses
# ---------------------------------------------------------------------------


def test_the_matrix_document_parses_into_the_frozen_number_of_rows(parsed):
    """The premise, asserted before anything is read out of a row.

    A parser that silently matched nothing would make every assertion below
    vacuously true, which is the classic way a document test passes forever.
    """

    assert DOC.exists(), DOC
    assert len(parsed) == TOTAL_ROWS, len(parsed)
    assert len(parsed) > 100, "the matrix is the phase's deliverable, not a stub"


def test_every_section_carries_its_frozen_row_count(parsed):
    """Adding a leaf without adding its row fails here.

    Per section rather than in total: a total-only pin is satisfied by moving a
    row from one family to another, which is exactly the edit that loses a cell.
    """

    counted: dict[str, int] = {}
    for row in parsed:
        counted[row.section] = counted.get(row.section, 0) + 1
    assert counted == SECTION_ROWS


# ---------------------------------------------------------------------------
# 2. The closed vocabularies
# ---------------------------------------------------------------------------


def test_every_row_uses_the_four_target_states(parsed):
    """There is no ``TODO`` state and no ``SILENT`` state."""

    bad = [(row.line, row.state) for row in parsed if row.state not in STATES]
    assert not bad, bad


def test_every_row_names_one_of_the_four_mechanisms(parsed):
    bad = [
        (row.line, row.mechanism) for row in parsed if row.mechanism not in MECHANISMS
    ]
    assert not bad, bad


def test_every_row_declares_a_mode_and_a_validation(parsed):
    bad = [
        (row.line, row.mode, row.validation)
        for row in parsed
        if row.mode not in MODES or row.validation not in VALIDATIONS
    ]
    assert not bad, bad


def test_no_row_has_an_empty_test_cell(parsed):
    """The single rule that makes the rest of the document mean anything.

    A row without a test is a claim with no evidence, and a matrix full of them
    is a wish list.
    """

    bad = [(row.line, row.route, row.leaf) for row in parsed if not row.test]
    assert not bad, bad


def test_a_supported_row_is_never_justified_by_a_refusal(parsed):
    """``SUP`` cannot be evidenced by a raise.

    ``declaration`` remains legal for a ``SUP`` row whose claim is structural -
    a compact row identity, an autograd context aliasing, a reduction order -
    because a structural fact is not proved by a finite difference. Which rows
    those are is frozen in :data:`STRUCTURAL_SUP_ROWS` rather than left open;
    see the two tests below. What is never legal, in any row, is a supported
    derivative whose only evidence is that something else was refused.
    """

    bad = [
        (row.line, row.leaf)
        for row in parsed
        if row.state == "SUP" and row.validation == "refusal"
    ]
    assert not bad, bad


def test_a_supported_numerical_row_carries_a_real_oracle(parsed):
    """The hole the narrowed rule left, closed by an allowlist.

    Allowing ``declaration`` on any ``SUP`` row made one edit invisible: flip a
    ``DECL`` row that already cites a declaration-style test to ``SUP`` and the
    document advertises a supported derivative with no oracle behind it while
    every gate still passes. A mutation run confirmed exactly that on
    ``sensor_weight/evaluate out:pattern_gain``.

    So the structural rows are enumerated. Every other ``SUP`` row has to carry
    ``fd``, ``oracle-f64``, ``analytic`` or ``adjoint`` - the evidence bar the
    document's own state table states.
    """

    bad = [
        (row.line, row.route, row.leaf, row.validation)
        for row in parsed
        if row.state == "SUP"
        and row.validation not in NUMERICAL_VALIDATIONS
        and (row.route, row.leaf, row.mode) not in STRUCTURAL_SUP_ROWS
    ]
    assert not bad, bad


def test_the_structural_allowlist_has_no_stale_entry(parsed):
    """An allowlist nobody prunes becomes a second place for a claim to hide.

    Every enumerated key must still be a ``SUP`` row validated by
    ``declaration``. Deleting or relabelling such a row therefore has to delete
    its key in the same diff, and a new structural row has to be added here
    deliberately rather than by inheriting a permissive rule.
    """

    present = {
        (row.route, row.leaf, row.mode)
        for row in parsed
        if row.state == "SUP" and row.validation == "declaration"
    }
    assert present == STRUCTURAL_SUP_ROWS


def test_a_refusal_validation_appears_only_on_a_refused_row(parsed):
    """The direction that actually holds, and it is the load-bearing one.

    The converse - every ``REF`` row is validated by ``refusal`` - is FALSE in
    this tree and correctly so: one row refuses a stale discovery-time answer
    and proves it by comparing against the analytically known correct answer at
    the moved geometry, which is a stronger statement than the raise alone.
    """

    bad = [
        (row.line, row.state)
        for row in parsed
        if row.validation == "refusal" and row.state != "REF"
    ]
    assert not bad, bad


def test_a_declared_row_is_evidenced_by_its_declaration(parsed):
    """``DECL`` is the narrow escape hatch, and the declaration IS the evidence.

    A ``DECL`` row validated by a finite difference would be a supported cell
    mislabelled, and one validated by a refusal would be a refused cell
    mislabelled. Both are ways of losing a decision.
    """

    bad = [
        (row.line, row.validation)
        for row in parsed
        if row.state == "DECL" and row.validation != "declaration"
    ]
    assert not bad, bad
    assert sum(1 for row in parsed if row.state == "DECL") >= 1


# ---------------------------------------------------------------------------
# 3. Resolution: owners and tests exist
# ---------------------------------------------------------------------------


def test_every_owner_path_exists(parsed):
    """``path.py:line``, and the file half must be real."""

    bad = []
    for row in parsed:
        path = ROOT / row.owner.split(":")[0]
        if not path.exists():
            bad.append((row.line, row.owner))
    assert not bad, bad


def test_every_cited_test_node_id_resolves(parsed):
    """The anti-rot rule: a renamed test must break its row.

    Resolution is by ``ast`` over the cited module rather than by importing it
    or by shelling out to ``pytest --collect-only``; see the module docstring
    for why, and for the one limit that costs.
    """

    cache: dict[pathlib.Path, frozenset[str]] = {}
    bad = []
    for row in parsed:
        for node_id in (part.strip() for part in row.test.split(",")):
            if "::" not in node_id:
                bad.append((row.line, f"not a node id: {node_id!r}"))
                continue
            module, function = node_id.split("::", 1)
            path = ROOT / module
            if not path.exists():
                bad.append((row.line, f"missing module {module}"))
                continue
            if path not in cache:
                cache[path] = _functions(path)
            if function.split("[")[0] not in cache[path]:
                bad.append((row.line, f"missing function {node_id}"))
    assert not bad, bad


def test_the_resolver_would_notice_a_broken_node_id():
    """Calibration. Without this the resolver could be matching nothing.

    A test that only ever resolves real ids cannot distinguish a working
    resolver from one whose lookup set is accidentally everything.
    """

    path = ROOT / "tests" / "test_phase9_capability_matrix.py"
    names = _functions(path)
    assert "test_every_cited_test_node_id_resolves" in names
    assert "test_a_function_that_was_never_written" not in names


# ---------------------------------------------------------------------------
# 4. The mirrored Channel rows agree with the live record
# ---------------------------------------------------------------------------


def test_the_mirrored_channel_rows_agree_with_the_live_capability_record():
    """The anti-rot pin BETWEEN the two repositories.

    Radar's acceptance cannot read Channel's document at test time, so the rows
    Radar consumes are mirrored into Radar's own matrix. A mirror that nothing
    checks is a copy that drifts. This asserts the five fields the mirrored
    section makes claims about, against the record the adapter actually queries.
    """

    pytest.importorskip("witwin.channel")
    from witwin.channel.propagation import consumer

    assert consumer.CONTRACT_VERSION == 6
    record = consumer.capabilities()

    assert record.direction_differentiable_components == frozenset(
        {"los", "reflection"}
    )
    modes = dict(record.component_ad_modes)
    # Narrowed by ADR-043: the diffraction AD column was advertised and could
    # not produce a row, so the pre-compute refusal is now the honest answer.
    assert modes["diffraction"] == frozenset({"none"})
    for component in ("los", "reflection"):
        assert modes[component] == frozenset({"none", "jvp", "vjp"})

    leaves = dict(record.component_material_leaves)
    assert leaves["los"] == ()
    assert leaves["reflection"] == ("eps_r", "sigma_e", "thickness_m", "gain")

    geometry = dict(record.differentiable_geometry_outputs)
    assert geometry["discovery"] == frozenset({"path_length_m", "delay_s"})
    assert geometry["fixed_topology"] == frozenset(
        {"path_length_m", "delay_s", "interaction_positions_m", "field_direction"}
    )

    assert record.supports_higher_order_ad is False
    assert record.ad_accounting is True
    assert "sources.powers_w" in record.primal_only_ad_inputs


def test_the_mirrored_section_only_claims_cells_radar_can_reach():
    """Radar freezes ``{los, reflection}`` and asks for ``scalar_transport``.

    A mirrored row for a component Radar never freezes would be documentation of
    somebody else's capability, and would pass a matrix test while describing a
    route no Radar test can drive.
    """

    pytest.importorskip("witwin.channel")
    from witwin.channel.propagation import consumer

    record = consumer.capabilities()
    assert record.fixed_topology_components == frozenset({"los", "reflection"})
    assert record.fixed_topology_components.issubset(
        record.direction_differentiable_components
    )
    assert "scalar_transport" in record.fixed_topology_responses


# ---------------------------------------------------------------------------
# 5. The deferral section is real
# ---------------------------------------------------------------------------


#: The sentinel every deferral bullet carries. A prose reason is worth more
#: than a table cell and the earlier stages wrote good ones, so the machine
#: check is on the one part that is a fact rather than an argument: who picks
#: this up. A deferral without a named owner is a hole with a nicer name.
DEFERRAL_OWNER_MARK = "Follow-up owner:"


def test_every_deferral_names_a_follow_up_owner():
    """Each ``## Deferred`` bullet carries a reason and an owner.

    The reason is checked structurally rather than semantically - a bullet has
    to be prose of some length - because a machine cannot tell a good reason
    from a bad one, and pretending otherwise would only produce a test that
    passes on the word "later".
    """

    text = DOC.read_text(encoding="utf-8")
    assert "## Deferred" in text
    deferred = text.split("## Deferred", 1)[1].split("\n## ", 1)[0]
    bullets = [
        block
        for block in deferred.split("\n- ")[1:]
        if block.strip()
    ]
    assert len(bullets) >= 8, len(bullets)
    for block in bullets:
        head = block.strip().splitlines()[0]
        assert block.strip().startswith("**"), head
        assert len(block) > 120, head
        assert DEFERRAL_OWNER_MARK in block, head
        owner = block.split(DEFERRAL_OWNER_MARK, 1)[1].strip()
        assert len(owner.split(".")[0].strip()) >= 4, head


def test_the_acceptance_record_maps_every_plan_criterion():
    """The seven acceptance criteria each name the tests that prove them."""

    text = DOC.read_text(encoding="utf-8")
    assert "## Acceptance record" in text
    record = text.split("## Acceptance record", 1)[1]
    rows = [
        line
        for line in record.splitlines()
        if line.strip().startswith("|")
        and not line.strip().startswith("|---")
        and "criterion" not in line.lower()
    ]
    assert len(rows) == 7, len(rows)
    for line in rows:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        assert len(cells) == 3, line
        assert all(cells), line
        assert cells[2] in {"proved", "partially proved", "not proved"}, cells
