"""The sensor-weight frozen constants refuse a derivative instead of returning None.

``SensorWeightGeometry``'s docstring has said since Phase 6 that every field on
it except the four position tensors is a constant with respect to the
derivative. Nothing enforced it. Thirteen float tensors - three velocity sets,
the fixed leg length, the facet normals, the two polarization sets, the local
frame, and the five resident pattern tables - reach the native operator through slots
whose ``backward`` returns ``None`` by construction, and five of them are not
inputs of the autograd ``Function`` at all. A caller who marked one got a full
frame, a full result object, and ``grad = None``, with nothing anywhere saying
the slot had no derivative.

That is the defect class this phase exists to remove, and the fix is a refusal at
CONSTRUCTION - before ``validate``, before a plan, before any launch. Almost
everything here is deliberately CPU-only: the refusal is a property of the
contract and asking for a GPU to check it would have made it a test people skip.
The last section is the exception, and it earns its ``--gpu`` mark - it drives
the refusal from a real production entry point, ``Radar.mimo_from_trace``, which
is where the one displaced caller in the tree was found.

Three boundaries are pinned alongside the refusal, because over-refusing is the
opposite mistake and just as easy to make:

* the six differentiable inputs still work and are covered by
  ``test_phase6_sensor_weight.py``, whose gradient tests would fail if this
  refusal had been placed one level too high;
* ``pattern_gain`` stays as it is - a published OUTPUT with
  ``mark_non_differentiable``, which is a DECLARATION rather than a silence;
* the index tensors are not checked at all, because ``int64`` and ``int32``
  cannot carry a derivative for autograd to lose.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

from witwin.radar.sensors import ROW_KIND_VIA  # noqa: E402
from witwin.radar.sensors.weights import (  # noqa: E402
    SensorWeightGeometry,
    SensorWeightModes,
    SensorWeightPlan,
)


NUM_TX = 2
NUM_RX = 2
ROWS = 3

#: The eight frozen float fields on the geometry, with the shape each one has.
#: Written out rather than derived from the dataclass so that a field added
#: without a decision about its derivative shows up as a mismatch below rather
#: than being silently swept into the loop.
GEOMETRY_SHAPES = {
    "tx_velocity": (NUM_TX, 3),
    "rx_velocity": (NUM_RX, 3),
    "site_velocity": (ROWS, 3),
    "fixed_length_m": (ROWS,),
    "normals": (ROWS, 3),
    "pol_tx": (NUM_TX, 3),
    "pol_rx": (NUM_RX, 3),
    "local_axes": (3, 3),
}


def _geometry_fields() -> dict:
    fields = {
        "num_tx": NUM_TX,
        "num_rx": NUM_RX,
        "tx_index": torch.zeros(ROWS, dtype=torch.int64),
        "rx_index": torch.zeros(ROWS, dtype=torch.int64),
        "row_kind": torch.full((ROWS,), ROW_KIND_VIA, dtype=torch.int32),
    }
    for name, shape in GEOMETRY_SHAPES.items():
        fields[name] = torch.zeros(shape, dtype=torch.float32)
    fields["fixed_length_m"] = torch.ones(ROWS, dtype=torch.float32)
    return fields


def _plan(tables=None) -> SensorWeightPlan:
    return SensorWeightPlan(
        pattern_kind=0,
        tables=tuple(
            torch.zeros(4, dtype=torch.float32) for _ in range(5)
        )
        if tables is None
        else tables,
        c0=299792458.0,
        wavelength_m=3.894e-3,
        tx_amplitude=1.0,
        modes=SensorWeightModes(
            spreading=False, tx_power=False, legacy_real_polarization=False
        ),
    )


# --------------------------------------------------------------------------
# 1. The premise
# --------------------------------------------------------------------------


def test_an_unmarked_geometry_and_plan_still_construct_and_validate():
    """The refusal did not become a refusal of the normal case.

    ``validate`` is called too, because the refusal was inserted BEFORE it and
    a mistake there would have made every production frame raise. The whole
    ``--gpu`` sensor-weight suite is the real proof of that; this is the cheap
    guard that fails first and locally.
    """

    geometry = SensorWeightGeometry(**_geometry_fields())
    geometry.validate()
    assert geometry.path_count == ROWS
    assert _plan().kernel_tail(geometry)[0] == NUM_TX


def test_the_declared_frozen_field_list_matches_the_dataclass():
    """A new float field cannot be added without a decision about it.

    ``FROZEN_FIELDS`` plus the four index/count fields plus the differentiable
    positions is the whole surface. If someone adds, say, a per-row temperature
    to the geometry, this fails and the new field gets a decision instead of a
    silent ``None``.
    """

    import dataclasses

    declared = {field.name for field in dataclasses.fields(SensorWeightGeometry)}
    accounted = set(SensorWeightGeometry.FROZEN_FIELDS) | {
        "num_tx",
        "num_rx",
        "tx_index",
        "rx_index",
        "row_kind",
    }
    assert declared == accounted, sorted(declared ^ accounted)
    assert set(SensorWeightGeometry.FROZEN_FIELDS) == set(GEOMETRY_SHAPES)


# --------------------------------------------------------------------------
# 2. Every frozen field refuses, in both AD modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", sorted(GEOMETRY_SHAPES))
def test_a_marked_frozen_geometry_field_is_refused(field):
    fields = _geometry_fields()
    fields[field] = fields[field].clone().requires_grad_(True)
    with pytest.raises(RuntimeError) as excinfo:
        SensorWeightGeometry(**fields)
    message = str(excinfo.value)
    assert f"SensorWeightGeometry.{field}" in message
    assert "requires_grad" in message
    # The message must say what IS differentiable, or the caller has nowhere to
    # go from here.
    assert "tx_pos" in message and "intensity" in message


@pytest.mark.parametrize("field", sorted(GEOMETRY_SHAPES))
def test_a_dual_carrying_frozen_geometry_field_is_refused(field):
    """Forward mode as well, and it is the mode that matters more here.

    An ADR-038 forward-only dual marks nothing, so a refusal written against
    ``requires_grad`` alone would have let a dual through, computed a whole
    frame, and published a tangent of exactly zero - which is what a genuinely
    static scene publishes, so it is indistinguishable from a correct answer by
    inspection.
    """

    fields = _geometry_fields()
    with forward_ad.dual_level():
        primal = fields[field]
        fields[field] = forward_ad.make_dual(primal, torch.ones_like(primal))
        assert not fields[field].requires_grad
        with pytest.raises(RuntimeError) as excinfo:
            SensorWeightGeometry(**fields)
    assert f"SensorWeightGeometry.{field}" in str(excinfo.value)
    assert "forward tangent" in str(excinfo.value)


def test_a_velocity_refusal_names_the_reason_a_velocity_is_never_a_leaf():
    """The three velocity slots get the extra sentence and it is not decoration.

    A normal or a polarization has no gradient slot in THIS operator and might
    plausibly gain one. A velocity is different in kind: under ADR-038 it is the
    tangent direction of a position dual, so ``d(loss)/d(velocity)`` does not
    exist in either mode and never will under this contract. The message has to
    distinguish "not implemented here" from "not a thing", because they call for
    different next steps.
    """

    for field in ("tx_velocity", "rx_velocity", "site_velocity"):
        fields = _geometry_fields()
        fields[field] = fields[field].clone().requires_grad_(True)
        with pytest.raises(RuntimeError) as excinfo:
            SensorWeightGeometry(**fields)
        message = str(excinfo.value)
        assert "ADR-038" in message
        assert "tangent direction" in message


@pytest.mark.parametrize("index", range(5))
def test_a_marked_pattern_table_is_refused(index):
    """The tables are a resident lookup, not a differentiable input.

    The pattern's real contribution to the derivative is carried by the WEIGHT,
    through the positions that decide which angle is interpolated - which is
    covered, in both modes, by ``test_phase6_sensor_weight.py``. Marking the
    table itself asks for the derivative of the tabulated VALUES, and the kernel
    has no slot for it.
    """

    tables = [torch.zeros(4, dtype=torch.float32) for _ in range(5)]
    tables[index] = tables[index].clone().requires_grad_(True)
    with pytest.raises(RuntimeError) as excinfo:
        _plan(tuple(tables))
    assert f"SensorWeightPlan.tables[{index}]" in str(excinfo.value)


# --------------------------------------------------------------------------
# 3. It is a refusal: nothing is produced
# --------------------------------------------------------------------------


def test_no_geometry_object_survives_a_refusal():
    """Fails before a partial result, which is the phase's acceptance wording.

    ``__post_init__`` raising means ``__init__`` raises, so there is no
    half-built geometry for an exception handler to pick up and no plan, no
    launch, and no result behind it.
    """

    fields = _geometry_fields()
    fields["normals"] = fields["normals"].clone().requires_grad_(True)
    captured = None
    try:
        captured = SensorWeightGeometry(**fields)
    except RuntimeError:
        pass
    assert captured is None


def test_the_refusal_precedes_the_shape_validation():
    """A marked tensor of the WRONG shape still reports the derivative.

    Ordering matters for the message a caller reads. ``validate`` would say
    "normals must have shape (3, 3)", the caller would fix the shape, and the
    silent ``None`` would come back on the next run. The refusal fires at
    construction and ``validate`` never gets the chance.
    """

    fields = _geometry_fields()
    fields["normals"] = torch.zeros(ROWS + 4, 3, requires_grad=True)
    with pytest.raises(RuntimeError) as excinfo:
        SensorWeightGeometry(**fields)
    assert "frozen geometric constant" in str(excinfo.value)


# --------------------------------------------------------------------------
# 4. The boundaries: what is deliberately NOT refused
# --------------------------------------------------------------------------


def test_the_index_tensors_are_not_checked_because_they_cannot_carry_a_gradient():
    """An integer tensor has nothing to lose, so a check there would be dead code.

    Asserted rather than assumed: if a future dtype change made one of these
    floating point, the guard list would have to grow with it, and this test is
    where that shows up.
    """

    fields = _geometry_fields()
    for name in ("tx_index", "rx_index", "row_kind"):
        tensor = fields[name]
        assert not tensor.is_floating_point()
        with pytest.raises(RuntimeError):
            tensor.requires_grad_(True)


def test_a_plain_tensor_without_a_derivative_is_accepted_unlike_a_spec_scalar():
    """The sensor rule is about the DERIVATIVE; the spec rule is about the TYPE.

    They are deliberately different and this pins the difference. A geometry
    field is genuinely a tensor and has to stay one - it is per row, on the
    device, and consumed by a kernel - so refusing the type is not available
    here; only a marked tensor is refused. A frontend or waveform scalar is a
    host float, where refusing the type is both available and stronger.
    """

    from witwin.radar.frontend.contracts import LnaSpec

    geometry = SensorWeightGeometry(**_geometry_fields())
    assert isinstance(geometry.normals, torch.Tensor)
    assert not geometry.normals.requires_grad

    with pytest.raises(TypeError):
        LnaSpec(gain_db=torch.tensor(20.0))


# --------------------------------------------------------------------------
# 5. The one displaced caller, from its real production entry point
# --------------------------------------------------------------------------


@pytest.mark.gpu
def test_a_marked_velocity_into_mimo_from_trace_is_refused():
    """The only caller in the tree that the new rule displaced, pinned.

    ``Radar.mimo_from_trace(trace, velocities=...)`` routes the velocity into
    ``SensorWeightGeometry.site_velocity`` through
    ``witwin/radar/sensors/legacy_paths.py``. A marked velocity there was
    MEASURED, before this change, to run the whole frame and return
    ``velocities.grad is None`` while the position gradient came back correctly:
    the velocity is not an input of the operator's autograd ``Function`` at all.

    So no capability was removed and the caller was fixed rather than the rule
    softened - ``tests/solvers/test_mimo_cross.py`` now detaches at the call
    site, with the reason written there. This test is what stops the mark from
    drifting back in, and it drives the refusal through the real entry point
    rather than through a hand-built geometry.
    """

    from conftest import make_radar_or_skip
    from solvers.test_mimo_cross import _mimo_config

    from witwin.radar import TraceResult

    config = _mimo_config(
        num_tx=1,
        num_rx=1,
        tx_loc=[[0, 0, 0]],
        rx_loc=[[0, 0, 0]],
        chirp_per_frame=4,
        num_doppler_bins=4,
        adc_start_time=6,
    )
    radar = make_radar_or_skip(config)
    points = torch.tensor(
        [[0.0, 0.0, -3.0]], dtype=torch.float32, device="cuda", requires_grad=True
    )
    intensities = torch.tensor([0.9], dtype=torch.float32, device="cuda")
    velocities = torch.tensor(
        [[0.0, 0.0, -0.5]], dtype=torch.float32, device="cuda", requires_grad=True
    )
    trace = TraceResult(points, intensities)

    with pytest.raises(RuntimeError) as excinfo:
        radar.mimo_from_trace(trace, velocities=velocities, t0=0.4)
    assert "SensorWeightGeometry.site_velocity" in str(excinfo.value)

    # And the same call with the velocity detached still works and still carries
    # the POSITION gradient, which is what that route is actually for.
    frame = radar.mimo_from_trace(trace, velocities=velocities.detach(), t0=0.4)
    frame.abs().square().sum().backward()
    assert points.grad is not None
    assert float(points.grad.abs().sum()) > 0.0
