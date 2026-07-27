"""Two-way composition: the identity join and the composed payload.

Freezing the join is host bookkeeping and runs on the CPU. Composing a frame is
a native CUDA kernel, so every test that calls ``compose`` is marked ``gpu`` and
builds its fabricated legs on the device. Fabricated legs are what make the
permutation and multi-site cases reachable at all: the single real
line-of-sight leg has one row and cannot distinguish a correct join from a
positional one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.radar.paths import TwoWayComposer
from witwin.radar.propagation import RadarLegBatch
from witwin.radar.scattering import ScalarRcsResponse


def _frozen(
    source_ids,
    sink_ids,
    *,
    components=None,
    depths=None,
    primitives=None,
    device="cpu",
):
    """A duck-typed frozen leg topology with full row identity.

    ``components``/``depths``/``primitives`` default to distinct per-row values
    so that a fabricated leg with several rows per endpoint pair has a unique
    identity key, exactly as a real multipath leg does. Passing them explicitly
    is how a test builds two rows that DO collide.
    """

    rows = len(source_ids)
    components = list(range(rows)) if components is None else list(components)
    depths = list(components) if depths is None else list(depths)
    primitives = (
        [[value] for value in components] if primitives is None else list(primitives)
    )
    def make(values, dtype):
        return torch.tensor(values, dtype=dtype, device=device)

    return SimpleNamespace(
        source_id=make(source_ids, torch.int64),
        sink_id=make(sink_ids, torch.int64),
        component_id=make(components, torch.int32),
        depth=make(depths, torch.int32),
        primitive_sequence=make(primitives, torch.int32).reshape(rows, -1),
        material_sequence=make(primitives, torch.int32).reshape(rows, -1),
    )


def _legs(delays, coefficients, *, rates=None, valid=None, device="cpu"):
    rows = len(delays)

    def zeros(dtype, shape=None):
        return torch.zeros(shape or (rows,), dtype=dtype, device=device)

    def make(values, dtype):
        return torch.tensor(values, dtype=dtype, device=device)

    return RadarLegBatch(
        leg_count=rows,
        pair_count=1,
        pair_index=zeros(torch.int64),
        pair_offsets=make([0, rows], torch.int64),
        source_index=zeros(torch.int32),
        sink_index=zeros(torch.int32),
        depth=zeros(torch.int32),
        component_id=zeros(torch.int32),
        source_id=zeros(torch.int64),
        sink_id=zeros(torch.int64),
        primitive_sequence=zeros(torch.int32, (rows, 1)),
        material_sequence=zeros(torch.int32, (rows, 1)),
        interaction_type=zeros(torch.int32, (rows, 1)),
        delay_s=make(delays, torch.float32),
        coefficient=make(coefficients, torch.complex64),
        delay_rate=None if rates is None else make(rates, torch.float32),
        row_valid=None if valid is None else make(valid, torch.bool),
        diagnostics=None,
    )


def _response(amplitude=2.0, phase=0.3):
    return ScalarRcsResponse.from_values(amplitude, phase)


def _one_site_composer(device="cpu"):
    return TwoWayComposer.freeze(
        _frozen([10], [20], device=device),
        _frozen([20], [30], device=device),
        torch.tensor([20], dtype=torch.int64, device=device),
        radar_source_ids=[10],
        radar_sink_ids=[30],
        reference_frequency_hz=77.0e9,
    )


@pytest.mark.gpu
def test_delay_is_additive_and_transfer_factorizes():
    composer = _one_site_composer("cuda")
    inbound = _legs([1.0e-8], [0.5 + 0.25j], rates=[3.0e-9], device="cuda")
    outbound = _legs([2.0e-8], [-0.125 + 0.75j], rates=[-1.0e-9], device="cuda")
    composed = composer.compose(inbound, outbound, _response())

    assert composed.path_count == 1
    assert composed.sensor_pair_count == 1
    # Explicit relative tolerance with atol=0. torch.testing's float32 default
    # atol is 1e-5, which dwarfs a nanosecond-scale delay: with defaults these
    # two assertions pass for ANY delay value, including one that dropped the
    # outbound leg entirely. They were vacuous, and a mutation that removed
    # tau_out survived this whole file.
    torch.testing.assert_close(
        composed.total_delay_s.cpu(),
        torch.tensor([3.0e-8], dtype=torch.float32),
        rtol=1e-6,
        atol=0.0,
    )
    torch.testing.assert_close(
        composed.delay_rate.cpu(),
        torch.tensor([2.0e-9], dtype=torch.float32),
        rtol=1e-6,
        atol=0.0,
    )
    expected = (
        outbound.coefficient
        * _response().evaluate(1, torch.device("cuda"))
        * inbound.coefficient
    )
    torch.testing.assert_close(
        composed.complex_transfer_ref, expected, rtol=1e-6, atol=0.0
    )
    assert composed.topology.radar_source_id.tolist() == [10]
    assert composed.topology.site_id.tolist() == [20]
    assert composed.topology.radar_sink_id.tolist() == [30]


def _count_tolist(monkeypatch) -> dict[str, int]:
    calls = {"n": 0}
    original = torch.Tensor.tolist

    def counting_tolist(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)
    return calls


def test_freeze_host_reads_are_counted(monkeypatch):
    """The composer's freeze-time host traffic, quantified (R-ADR-006).

    ``TwoWayComposer.freeze`` reads leg identity to the host to build the join.
    That is sanctioned and one-time, but Channel never sees those reads, so no
    Channel diagnostic counts them and the one-time total was implied rather
    than measured. Fourteen reads: six identity columns per leg (source_id,
    sink_id, component_id, depth, primitive_sequence, material_sequence), plus
    site_ids, plus the composed pair index that
    ``synthesis.assembly.validate_pair_ordering`` checks. The front-end
    endpoint IDs are passed as Python lists here, so they add none.

    The fourteenth read is Phase 7's. Wiring the layout gate is the plan's own
    Phase-6 gap 5, and the gate reads its input on the host BY DESIGN - that is
    why it is a freeze-time function and not part of ``assemble_frame_cube``.
    It costs one one-time read and it is what makes the sink-major assertion
    non-empty in production; the per-FRAME budget is untouched, which
    ``test_compose_performs_no_host_observation_at_all`` still pins at zero.
    """

    calls = _count_tolist(monkeypatch)
    _one_site_composer()
    assert calls["n"] == 14, calls["n"]


@pytest.mark.gpu
def test_compose_performs_no_host_observation_at_all(monkeypatch):
    """The assertion that actually matters: per frame, nothing crosses back.

    ``compose`` runs once per frame inside the loop the fixed-topology
    capability exists to make cheap. A single host read here would undo it.
    """

    composer = _one_site_composer("cuda")
    inbound = _legs([1.0e-8], [0.5 + 0.25j], rates=[3.0e-9], device="cuda")
    outbound = _legs([2.0e-8], [-0.125 + 0.75j], rates=[-1.0e-9], device="cuda")
    response = _response()
    composer.compose(inbound, outbound, response)  # warm the operator table

    calls = _count_tolist(monkeypatch)
    observed: list[str] = []
    for name in ("cpu", "item", "numpy"):
        original = getattr(torch.Tensor, name)

        def observing(self, *args, _name=name, _original=original, **kwargs):
            observed.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, observing)

    composer.compose(inbound, outbound, response)
    assert calls["n"] == 0
    assert observed == []


@pytest.mark.gpu
def test_join_is_by_identity_not_by_array_position():
    """Permuting the outbound frozen rows must not change the result.

    A positional join would silently pair the wrong legs here and produce a
    plausible-looking, wrong answer.
    """

    sites = torch.tensor([20, 21], dtype=torch.int64, device="cuda")
    inbound_frozen = _frozen([10, 10], [20, 21], components=[0, 0], device="cuda")
    inbound = _legs([1.0e-8, 4.0e-8], [0.5 + 0.0j, 0.1 + 0.2j], device="cuda")

    straight = TwoWayComposer.freeze(
        inbound_frozen,
        _frozen([20, 21], [30, 30], components=[0, 0], device="cuda"),
        sites,
        radar_source_ids=[10],
        radar_sink_ids=[30],
        reference_frequency_hz=77.0e9,
    ).compose(
        inbound,
        _legs([2.0e-8, 8.0e-8], [1.0 + 0.0j, 0.0 + 1.0j], device="cuda"),
        _response(),
    )
    permuted = TwoWayComposer.freeze(
        inbound_frozen,
        _frozen([21, 20], [30, 30], components=[0, 0], device="cuda"),
        sites,
        radar_source_ids=[10],
        radar_sink_ids=[30],
        reference_frequency_hz=77.0e9,
    ).compose(
        inbound,
        _legs([8.0e-8, 2.0e-8], [0.0 + 1.0j, 1.0 + 0.0j], device="cuda"),
        _response(),
    )

    # Bit-identical, not merely close: the same rows in the same order do the
    # same arithmetic. Default float32 tolerances would hide a wrong pairing at
    # these delay magnitudes entirely.
    torch.testing.assert_close(
        straight.total_delay_s, permuted.total_delay_s, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        straight.complex_transfer_ref,
        permuted.complex_transfer_ref,
        rtol=0.0,
        atol=0.0,
    )
    assert straight.topology.site_id.tolist() == permuted.topology.site_id.tolist()
    # The join really did have to reorder: the outbound row indices differ.
    assert (
        straight.topology.outbound_row.tolist()
        != permuted.topology.outbound_row.tolist()
    )


def _two_by_two_composer(device="cpu"):
    return TwoWayComposer.freeze(
        _frozen([10, 11], [20, 20], components=[0, 0], device=device),
        _frozen([20, 20], [30, 31], components=[0, 0], device=device),
        torch.tensor([20], dtype=torch.int64),
        radar_source_ids=[10, 11],
        radar_sink_ids=[30, 31],
        reference_frequency_hz=77.0e9,
    )


def test_rows_are_sorted_into_a_valid_pair_partition():
    composer = _two_by_two_composer()
    # Two radar sources x two radar sinks through one site: four sensor pairs.
    assert composer.sensor_pair_count == 4
    assert composer.path_count == 4
    offsets = composer.pair_offsets.tolist()
    assert offsets == [0, 1, 2, 3, 4]
    assert composer.sensor_pair_index.tolist() == sorted(
        composer.sensor_pair_index.tolist()
    )
    # SINK-MAJOR, mirroring the Channel consumer's own pair index
    # (``sink_row_index * source_count + source_row_index``). One convention
    # crosses the boundary rather than two; a source-major order here would be
    # a second, silently different, virtual-array numbering.
    pairs = list(
        zip(
            composer.topology.radar_source_id.tolist(),
            composer.topology.radar_sink_id.tolist(),
            strict=True,
        )
    )
    assert pairs == [(10, 30), (11, 30), (10, 31), (11, 31)]


def test_the_pair_partition_spans_the_front_end_not_the_surviving_rows():
    """A pair that discovered nothing still owns an (empty) segment.

    ``synthesize_fmcw_beat`` shapes its output ``[chirps, sensor_pair_count,
    samples]``. Deriving the pair set from surviving composed rows would
    renumber and reshape the IQ cube whenever a site failed discovery for one
    TX/RX pair, which is a silent wrong answer rather than a missing one.
    """

    composer = TwoWayComposer.freeze(
        _frozen([10], [20]),
        _frozen([20], [30]),
        torch.tensor([20], dtype=torch.int64),
        radar_source_ids=[10, 11],
        radar_sink_ids=[30, 31],
        reference_frequency_hz=77.0e9,
    )
    # Only the (10, 30) pair has legs; the other three are empty but present.
    assert composer.sensor_pair_count == 4
    assert composer.path_count == 1
    assert composer.pair_offsets.tolist() == [0, 1, 1, 1, 1]
    assert composer.sensor_pair_index.tolist() == [0]


def test_the_frozen_offsets_partition_is_validated_where_it_is_free():
    """The kernel clamps a malformed offsets table; the composer refuses one.

    Clamping is a memory-safety backstop, not a validation policy: it turns a
    malformed table into a plausible wrong answer rather than an error, and the
    kernel cannot do better because reading the values per frame is exactly the
    D2H the fixed-topology capability exists to avoid. Freeze time is where the
    check is free, so that is where it lives, and it is what makes the
    production route unable to reach the clamp at all.
    """

    composer = _two_by_two_composer()
    offsets = composer.pair_offsets.tolist()
    assert offsets[0] == 0
    assert offsets[-1] == composer.path_count
    assert offsets == sorted(offsets)
    assert len(offsets) == composer.sensor_pair_count + 1

    # And the check is a real gate, exercised rather than grepped for: a table
    # that failed either condition raises instead of reaching the kernel.
    from witwin.radar.paths import _identity

    with pytest.raises(ValueError, match="would not partition all composed rows"):
        _identity.pair_offsets([0, 0, 1], pair_count=1)
    with pytest.raises(ValueError, match="pair_count must be positive"):
        _identity.pair_offsets([], pair_count=0)
    # Empty segments are legal; only an out-of-range pair rank is not.
    assert _identity.pair_offsets([0, 0, 3], pair_count=4) == [0, 2, 2, 2, 3]


def test_a_site_without_a_leg_is_refused_rather_than_dropped():
    with pytest.raises(ValueError, match="no outbound leg row"):
        TwoWayComposer.freeze(
            _frozen([10], [20]),
            _frozen([21], [30]),
            torch.tensor([20], dtype=torch.int64),
            radar_source_ids=[10],
            radar_sink_ids=[30],
            reference_frequency_hz=77.0e9,
        )
    with pytest.raises(ValueError, match="no inbound leg row"):
        TwoWayComposer.freeze(
            _frozen([10], [21]),
            _frozen([20], [30]),
            torch.tensor([20], dtype=torch.int64),
            radar_source_ids=[10],
            radar_sink_ids=[30],
            reference_frequency_hz=77.0e9,
        )


def test_a_leg_endpoint_outside_the_declared_front_end_is_refused():
    """A stray radar endpoint is a silent drop, not an empty segment.

    The empty-segment rule covers a pair that discovered nothing. A leg row
    whose radar endpoint is not in the declared front end is the opposite
    problem: it exists and would simply never be visited.
    """

    with pytest.raises(ValueError, match="not in radar_source_ids"):
        TwoWayComposer.freeze(
            _frozen([11], [20]),
            _frozen([20], [30]),
            torch.tensor([20], dtype=torch.int64),
            radar_source_ids=[10],
            radar_sink_ids=[30],
            reference_frequency_hz=77.0e9,
        )
    with pytest.raises(ValueError, match="not in radar_sink_ids"):
        TwoWayComposer.freeze(
            _frozen([10], [20]),
            _frozen([20], [31]),
            torch.tensor([20], dtype=torch.int64),
            radar_source_ids=[10],
            radar_sink_ids=[30],
            reference_frequency_hz=77.0e9,
        )


def test_two_leg_rows_that_share_an_identity_key_are_refused():
    """An ambiguous canonical order is refused, not tie-broken on position.

    Two rows of one leg with the same component, depth, and interaction
    sequence cannot be ordered by identity. Falling back on row position for
    the tie would reintroduce exactly the positional dependence the identity
    join exists to remove, and it would make the permutation test vacuous.
    """

    with pytest.raises(ValueError, match="share the identity key"):
        TwoWayComposer.freeze(
            _frozen([10, 10], [20, 20], components=[1, 1]),
            _frozen([20], [30]),
            torch.tensor([20], dtype=torch.int64),
            radar_source_ids=[10],
            radar_sink_ids=[30],
            reference_frequency_hz=77.0e9,
        )


@pytest.mark.gpu
def test_row_validity_is_the_conjunction_of_both_legs():
    composer = TwoWayComposer.freeze(
        _frozen([10, 10], [20, 21], components=[0, 0], device="cuda"),
        _frozen([20, 21], [30, 30], components=[0, 0], device="cuda"),
        torch.tensor([20, 21], dtype=torch.int64, device="cuda"),
        radar_source_ids=[10],
        radar_sink_ids=[30],
        reference_frequency_hz=77.0e9,
    )
    composed = composer.compose(
        _legs(
            [1.0e-8, 2.0e-8],
            [1.0 + 0j, 1.0 + 0j],
            valid=[True, False],
            device="cuda",
        ),
        _legs(
            [1.0e-8, 2.0e-8],
            [1.0 + 0j, 1.0 + 0j],
            valid=[True, True],
            device="cuda",
        ),
        _response(),
    )
    assert composed.row_valid.tolist() == [True, False]
    # A dead row's payload is exactly zero, not a partial composition. It used
    # to publish ``0 + tau_out``, a plausible number that no consumer should
    # ever read; validity is the authority, and the payload agrees with it.
    assert float(composed.total_delay_s[0]) == pytest.approx(2.0e-8, rel=1e-6)
    assert float(composed.total_delay_s[1]) == 0.0
    assert complex(composed.complex_transfer_ref[1]) == 0j


@pytest.mark.gpu
def test_delay_rate_is_only_composed_when_both_legs_have_one():
    composer = _one_site_composer("cuda")
    with_rate = _legs([1.0e-8], [1.0 + 0j], rates=[1.0e-9], device="cuda")
    without = _legs([1.0e-8], [1.0 + 0j], device="cuda")
    assert composer.compose(with_rate, without, _response()).delay_rate is None
    assert composer.compose(without, with_rate, _response()).delay_rate is None
    assert composer.compose(with_rate, with_rate, _response()).delay_rate is not None
    # A position-perturbation dual is not a velocity, so the caller can say so.
    assert (
        composer.compose(
            with_rate, with_rate, _response(), include_delay_rate=False
        ).delay_rate
        is None
    )


def test_a_geometry_dependent_response_is_refused():
    class PerPathResponse:
        is_geometry_dependent = True

        def evaluate(self, row_count, device):
            raise AssertionError("must not be reached")

    composer = _one_site_composer()
    with pytest.raises(NotImplementedError, match="native kernel"):
        composer.compose(
            _legs([1.0e-8], [1.0 + 0j]),
            _legs([1.0e-8], [1.0 + 0j]),
            PerPathResponse(),
        )


def test_scalar_response_is_a_broadcast_parameter_scale():
    response = ScalarRcsResponse.from_values(3.0, 0.25)
    values = response.evaluate(4, torch.device("cpu"))
    assert values.shape == (4,)
    assert values.dtype == torch.complex64
    assert not response.is_geometry_dependent
    expected = 3.0 * complex(torch.cos(torch.tensor(-0.25)), torch.sin(torch.tensor(-0.25)))
    assert abs(complex(values[0]) - expected) < 1e-6
    assert all(complex(v) == complex(values[0]) for v in values)


def test_scalar_response_moves_to_the_device_it_is_asked_for():
    """``device`` was accepted and ignored.

    ``TwoWayComposer.compose`` passes the device its composed rows live on, so a
    CPU-authored response used to be accepted here and then fail with a device
    mismatch several frames away from the parameter that caused it. The move is
    autograd-aware, so a differentiable response keeps its tape across it.
    """

    response = ScalarRcsResponse.from_values(2.0, 0.3, requires_grad=True)
    values = response.evaluate(2, torch.device("cpu"))
    assert values.device.type == "cpu"
    assert values.requires_grad and values.grad_fn is not None

    if not torch.cuda.is_available():
        pytest.skip("device move across accelerators needs CUDA")
    moved = response.evaluate(2, torch.device("cuda"))
    assert moved.device.type == "cuda"
    # The gradient survives the move and reaches the CPU-authored parameters.
    moved.real.sum().backward()
    assert response.amplitude.grad is not None
    assert float(response.amplitude.grad) != 0.0


def test_scalar_response_rejects_malformed_parameters():
    with pytest.raises(TypeError, match="amplitude must be a torch.Tensor"):
        ScalarRcsResponse(amplitude=1.0, phase_rad=torch.tensor(0.0))
    with pytest.raises(ValueError, match="must be a 0-dim tensor"):
        ScalarRcsResponse(
            amplitude=torch.ones(2), phase_rad=torch.tensor(0.0)
        )
    with pytest.raises(TypeError, match="must use torch.float32"):
        ScalarRcsResponse(
            amplitude=torch.tensor(1.0, dtype=torch.float64),
            phase_rad=torch.tensor(0.0),
        )
    with pytest.raises(ValueError, match="row_count must be non-negative"):
        ScalarRcsResponse.from_values(1.0, 0.0).evaluate(-1, torch.device("cpu"))
