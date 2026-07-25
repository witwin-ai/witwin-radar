"""Two-way composition: the identity join and the composed payload.

Most of this runs on the CPU with fabricated legs. The composer is bookkeeping
plus device arithmetic, and testing it against hand-built legs is what makes
the permutation and multi-site cases reachable at all; the single real
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


def _frozen(source_ids, sink_ids):
    return SimpleNamespace(
        source_id=torch.tensor(source_ids, dtype=torch.int64),
        sink_id=torch.tensor(sink_ids, dtype=torch.int64),
    )


def _legs(delays, coefficients, *, rates=None, valid=None):
    rows = len(delays)
    return RadarLegBatch(
        leg_count=rows,
        pair_count=1,
        pair_index=torch.zeros(rows, dtype=torch.int64),
        pair_offsets=torch.tensor([0, rows], dtype=torch.int64),
        source_index=torch.zeros(rows, dtype=torch.int32),
        sink_index=torch.zeros(rows, dtype=torch.int32),
        depth=torch.zeros(rows, dtype=torch.int32),
        component_id=torch.zeros(rows, dtype=torch.int32),
        delay_s=torch.tensor(delays, dtype=torch.float32),
        coefficient=torch.tensor(coefficients, dtype=torch.complex64),
        delay_rate=None if rates is None else torch.tensor(rates, dtype=torch.float32),
        row_valid=None if valid is None else torch.tensor(valid, dtype=torch.bool),
        diagnostics=None,
    )


def _response(amplitude=2.0, phase=0.3):
    return ScalarRcsResponse.from_values(amplitude, phase)


def test_delay_is_additive_and_transfer_factorizes():
    composer = TwoWayComposer.freeze(
        _frozen([10], [20]),
        _frozen([20], [30]),
        torch.tensor([20], dtype=torch.int64),
        reference_frequency_hz=77.0e9,
    )
    inbound = _legs([1.0e-8], [0.5 + 0.25j], rates=[3.0e-9])
    outbound = _legs([2.0e-8], [-0.125 + 0.75j], rates=[-1.0e-9])
    composed = composer.compose(inbound, outbound, _response())

    assert composed.path_count == 1
    assert composed.sensor_pair_count == 1
    # Explicit relative tolerance with atol=0. torch.testing's float32 default
    # atol is 1e-5, which dwarfs a nanosecond-scale delay: with defaults these
    # two assertions pass for ANY delay value, including one that dropped the
    # outbound leg entirely. They were vacuous, and a mutation that removed
    # tau_out survived this whole file.
    torch.testing.assert_close(
        composed.total_delay_s,
        torch.tensor([3.0e-8], dtype=torch.float32),
        rtol=1e-6,
        atol=0.0,
    )
    torch.testing.assert_close(
        composed.delay_rate,
        torch.tensor([2.0e-9], dtype=torch.float32),
        rtol=1e-6,
        atol=0.0,
    )
    expected = (
        outbound.coefficient
        * _response().evaluate(1, torch.device("cpu"))
        * inbound.coefficient
    )
    torch.testing.assert_close(composed.complex_transfer_ref, expected)
    assert composed.topology.radar_source_id.tolist() == [10]
    assert composed.topology.site_id.tolist() == [20]
    assert composed.topology.radar_sink_id.tolist() == [30]


def test_join_is_by_identity_not_by_array_position():
    """Permuting the outbound frozen rows must not change the result.

    A positional join would silently pair the wrong legs here and produce a
    plausible-looking, wrong answer.
    """

    sites = torch.tensor([20, 21], dtype=torch.int64)
    inbound_frozen = _frozen([10, 10], [20, 21])
    inbound = _legs([1.0e-8, 4.0e-8], [0.5 + 0.0j, 0.1 + 0.2j])

    straight = TwoWayComposer.freeze(
        inbound_frozen,
        _frozen([20, 21], [30, 30]),
        sites,
        reference_frequency_hz=77.0e9,
    ).compose(
        inbound,
        _legs([2.0e-8, 8.0e-8], [1.0 + 0.0j, 0.0 + 1.0j]),
        _response(),
    )
    permuted = TwoWayComposer.freeze(
        inbound_frozen,
        _frozen([21, 20], [30, 30]),
        sites,
        reference_frequency_hz=77.0e9,
    ).compose(
        inbound,
        _legs([8.0e-8, 2.0e-8], [0.0 + 1.0j, 1.0 + 0.0j]),
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


def test_rows_are_sorted_into_a_valid_pair_partition():
    composer = TwoWayComposer.freeze(
        _frozen([10, 11], [20, 20]),
        _frozen([20, 20], [30, 31]),
        torch.tensor([20], dtype=torch.int64),
        reference_frequency_hz=77.0e9,
    )
    # Two radar sources x two radar sinks through one site: four sensor pairs.
    assert composer.sensor_pair_count == 4
    assert composer.path_count == 4
    offsets = composer.pair_offsets.tolist()
    assert offsets == [0, 1, 2, 3, 4]
    assert composer.sensor_pair_index.tolist() == sorted(
        composer.sensor_pair_index.tolist()
    )
    pairs = list(
        zip(
            composer.topology.radar_source_id.tolist(),
            composer.topology.radar_sink_id.tolist(),
            strict=True,
        )
    )
    assert pairs == sorted(pairs)


def test_a_site_without_a_leg_is_refused_rather_than_dropped():
    with pytest.raises(ValueError, match="no outbound leg row"):
        TwoWayComposer.freeze(
            _frozen([10], [20]),
            _frozen([21], [30]),
            torch.tensor([20], dtype=torch.int64),
            reference_frequency_hz=77.0e9,
        )
    with pytest.raises(ValueError, match="no inbound leg row"):
        TwoWayComposer.freeze(
            _frozen([10], [21]),
            _frozen([20], [30]),
            torch.tensor([20], dtype=torch.int64),
            reference_frequency_hz=77.0e9,
        )


def test_row_validity_is_the_conjunction_of_both_legs():
    composer = TwoWayComposer.freeze(
        _frozen([10, 10], [20, 21]),
        _frozen([20, 21], [30, 30]),
        torch.tensor([20, 21], dtype=torch.int64),
        reference_frequency_hz=77.0e9,
    )
    composed = composer.compose(
        _legs([1.0e-8, 2.0e-8], [1.0 + 0j, 1.0 + 0j], valid=[True, False]),
        _legs([1.0e-8, 2.0e-8], [1.0 + 0j, 1.0 + 0j], valid=[True, True]),
        _response(),
    )
    assert composed.row_valid.tolist() == [True, False]


def test_delay_rate_is_only_composed_when_both_legs_have_one():
    composer = TwoWayComposer.freeze(
        _frozen([10], [20]),
        _frozen([20], [30]),
        torch.tensor([20], dtype=torch.int64),
        reference_frequency_hz=77.0e9,
    )
    with_rate = _legs([1.0e-8], [1.0 + 0j], rates=[1.0e-9])
    without = _legs([1.0e-8], [1.0 + 0j])
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

    composer = TwoWayComposer.freeze(
        _frozen([10], [20]),
        _frozen([20], [30]),
        torch.tensor([20], dtype=torch.int64),
        reference_frequency_hz=77.0e9,
    )
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
