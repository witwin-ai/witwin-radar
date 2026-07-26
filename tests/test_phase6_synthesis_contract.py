"""The one contract all three waveforms consume, and the eight rules that make
the double-count hazards unrepresentable.

Every assertion here runs on the CPU. That is deliberate: the provenance rules
are the part of Phase 6 that decides whether a number is off by 58 dB or by a
factor of 215, and a check that needs a GPU is a check that does not run in the
default suite.

Each rule test asserts on the MESSAGE as well as the exception type. The
failure each rule prevents is a plausible-looking cube rather than a crash, so
an error that only says "invalid configuration" sends the reader looking for a
bug in the physics instead of at the two lines of configuration that caused it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from witwin.radar.paths.contracts import RadarPathBatch, RadarPathTopology
from witwin.radar.synthesis import (
    SlowTimeMode,
    SynthesisPathBatch,
    WaveformSpecProtocol,
    require_compatible,
)


F_REF = 77.0e9
C0 = 299792458.0


@dataclass(frozen=True)
class _Spec:
    """The smallest thing that satisfies WaveformSpecProtocol.

    A stand-in rather than ``FmcwBeatSpec``, because the rules are statements
    about the protocol and not about FMCW: an OFDM or pulsed spec has to answer
    the same four questions, and testing through one waveform's dataclass would
    quietly make the rules FMCW-shaped.
    """

    carrier_hz: float = 0.0
    carrier_rate_hz: float = F_REF
    reference_frequency_hz: float = F_REF
    applies_spreading: bool = False


@dataclass(frozen=True)
class _SensorAwareSpec(_Spec):
    """A spec that also declares the sensor-weight owner's TX power mode."""

    tx_power_mode: str = "already_in_weight"


def _topology(rows: int) -> RadarPathTopology:
    zeros = lambda: torch.zeros(rows, dtype=torch.int64)  # noqa: E731
    return RadarPathTopology(zeros(), zeros(), zeros(), zeros(), zeros())


def _radar_paths(rows: int = 3, *, with_rate: bool = True) -> RadarPathBatch:
    delay = torch.linspace(1.0e-8, 3.0e-8, rows, dtype=torch.float32).contiguous()
    weight = torch.complex(
        torch.linspace(0.25, 1.0, rows, dtype=torch.float32),
        torch.linspace(-0.5, 0.5, rows, dtype=torch.float32),
    ).contiguous()
    return RadarPathBatch(
        sensor_pair_count=2,
        path_count=rows,
        sensor_pair_index=torch.tensor([0] + [1] * (rows - 1), dtype=torch.int64),
        pair_offsets=torch.tensor([0, 1, rows], dtype=torch.int64),
        total_delay_s=delay,
        delay_rate=(
            torch.full((rows,), 1.0e-8, dtype=torch.float32) if with_rate else None
        ),
        complex_transfer_ref=weight,
        reference_frequency_hz=F_REF,
        row_valid=torch.ones(rows, dtype=torch.bool),
        topology=_topology(rows),
        join_mode="multipath",
    )


def _channel_batch(**overrides) -> SynthesisPathBatch:
    batch = SynthesisPathBatch.from_radar_paths(
        _radar_paths(), slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    )
    if not overrides:
        return batch
    from dataclasses import replace

    return replace(batch, **overrides)


def _real_batch(amplitudes: torch.Tensor | None = None) -> SynthesisPathBatch:
    if amplitudes is None:
        amplitudes = torch.tensor([0.5, -0.25, 1.0], dtype=torch.float32)
    rows = int(amplitudes.shape[0])
    return SynthesisPathBatch.from_real_amplitudes(
        torch.linspace(1.0, 3.0, rows, dtype=torch.float32),
        amplitudes,
        pair_offsets=torch.tensor([0, 1, rows], dtype=torch.int64),
        topology=_topology(rows),
        c0=C0,
        reference_frequency_hz=F_REF,
    )


# ---------------------------------------------------------------------------
# T0.8 - the RadarPathBatch mapping is aliasing, not conversion
# ---------------------------------------------------------------------------


def test_from_radar_paths_is_zero_copy():
    """Object identity, not value equality.

    ``assert_close`` would pass on a ``.clone()``, and a clone on the frame path
    is both a wasted allocation and a severed gradient. Storage identity is what
    the repository's standing rule about preserving row identity, order,
    aliasing, stride, dtype, device, and gradient state actually means.
    """

    paths = _radar_paths()
    batch = SynthesisPathBatch.from_radar_paths(
        paths, slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    )

    for name in (
        "total_delay_s",
        "complex_transfer_ref",
        "pair_offsets",
        "sensor_pair_index",
        "delay_rate",
        "row_valid",
    ):
        original = getattr(paths, name)
        mapped = getattr(batch, name)
        assert mapped is original, name
        assert mapped.data_ptr() == original.data_ptr(), name
    assert batch.topology is paths.topology
    assert batch.join_mode == paths.join_mode
    assert batch.path_count == paths.path_count
    assert batch.sensor_pair_count == paths.sensor_pair_count


def test_from_radar_paths_writes_channels_provenance_not_the_callers():
    batch = _channel_batch()
    assert batch.weight_includes_reference_phase is True
    assert batch.weight_includes_spreading is True
    assert batch.weight_includes_tx_power is True
    assert batch.frequency_response is None


def test_slow_time_mode_has_no_default():
    with pytest.raises(TypeError):
        SynthesisPathBatch.from_radar_paths(_radar_paths())


def test_from_radar_paths_refuses_a_foreign_type():
    with pytest.raises(TypeError, match="RadarPathBatch"):
        SynthesisPathBatch.from_radar_paths(
            object(), slow_time_mode=SlowTimeMode.REFRESHED_WEIGHT_NO_RATE
        )


# ---------------------------------------------------------------------------
# T0.4 (contract half) - the real-amplitude embedding
# ---------------------------------------------------------------------------


def test_real_amplitudes_keep_their_sign():
    """A negative legacy amplitude is a reflection flip, not a magnitude.

    ``complex(abs(amp), 0)`` fails this. It is the only phase a real amplitude
    can carry, and dropping it is a silent 180-degree error.
    """

    batch = _real_batch(torch.tensor([-0.5, 0.5], dtype=torch.float32))
    assert torch.equal(
        batch.complex_transfer_ref,
        torch.tensor([-0.5 + 0j, 0.5 + 0j], dtype=torch.complex64),
    )
    assert batch.complex_transfer_ref.imag.abs().max().item() == 0.0


def test_real_amplitudes_become_round_trip_delay_once():
    distances = torch.tensor([1.0, 3.7], dtype=torch.float32)
    batch = SynthesisPathBatch.from_real_amplitudes(
        distances,
        torch.ones(2, dtype=torch.float32),
        pair_offsets=torch.tensor([0, 2], dtype=torch.int64),
        topology=_topology(2),
        c0=C0,
        reference_frequency_hz=F_REF,
    )
    expected = distances.to(torch.float64) * 2.0 / C0
    assert torch.allclose(
        batch.total_delay_s.to(torch.float64), expected, rtol=1e-6, atol=0.0
    )


def test_real_amplitudes_declare_no_reference_phase():
    batch = _real_batch()
    assert batch.weight_includes_reference_phase is False
    assert batch.slow_time_mode is SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE


def test_the_pair_partition_is_derived_half_open():
    batch = _real_batch(torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32))
    assert torch.equal(
        batch.sensor_pair_index, torch.tensor([0, 1, 1, 1], dtype=torch.int64)
    )


# ---------------------------------------------------------------------------
# Structural validation - host-only, no tensor value is read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("total_delay_s", torch.zeros(3, dtype=torch.float64), "float32"),
        ("complex_transfer_ref", torch.zeros(3, dtype=torch.complex128), "complex64"),
        ("sensor_pair_index", torch.zeros(3, dtype=torch.int32), "int64"),
        ("row_valid", torch.zeros(3, dtype=torch.uint8), "bool"),
    ],
)
def test_a_wrong_dtype_is_refused_at_construction(field, value, message):
    with pytest.raises(TypeError, match=message):
        _channel_batch(**{field: value.contiguous()})


def test_a_noncontiguous_tensor_is_refused_rather_than_silently_copied():
    strided = torch.zeros(6, dtype=torch.float32)[::2]
    assert not strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        _channel_batch(total_delay_s=strided)


def test_reference_frequency_must_be_positive():
    with pytest.raises(ValueError, match="reference_frequency_hz"):
        _channel_batch(reference_frequency_hz=0.0)


def test_topology_row_count_must_match_path_count():
    with pytest.raises(ValueError, match="path_count rows"):
        _channel_batch(topology=_topology(2))


def test_a_frequency_response_without_its_grid_is_refused():
    with pytest.raises(ValueError, match="together"):
        _channel_batch(frequency_response=torch.zeros(3, 4, dtype=torch.complex64))


def test_an_empty_batch_is_legal():
    batch = SynthesisPathBatch.from_real_amplitudes(
        torch.zeros(0, dtype=torch.float32),
        torch.zeros(0, dtype=torch.float32),
        pair_offsets=torch.tensor([0, 0, 0], dtype=torch.int64),
        topology=_topology(0),
        c0=C0,
        reference_frequency_hz=F_REF,
    )
    assert batch.path_count == 0
    assert batch.sensor_pair_count == 2


def test_construction_reads_no_tensor_value():
    """Validation must cost no device-to-host transfer.

    Asserted structurally: a tensor subclass that raises on every host-observing
    access survives ``__post_init__``. An implementation that checked
    ``pair_offsets[-1] == path_count`` on device would fail here, which is the
    point - that check is a producer obligation, not a hot-path read.
    """

    class NoHostReads(torch.Tensor):
        def item(self):  # pragma: no cover - must never be called
            raise AssertionError("host observation during contract validation")

        def tolist(self):  # pragma: no cover
            raise AssertionError("host observation during contract validation")

        def cpu(self):  # pragma: no cover
            raise AssertionError("host observation during contract validation")

        def numpy(self):  # pragma: no cover
            raise AssertionError("host observation during contract validation")

    guarded = torch.zeros(3, dtype=torch.float32).as_subclass(NoHostReads)
    batch = _channel_batch(total_delay_s=guarded)
    assert batch.path_count == 3


# ---------------------------------------------------------------------------
# T0.7 - the eight provenance rules
# ---------------------------------------------------------------------------


def test_a_compatible_channel_pair_is_accepted():
    require_compatible(_channel_batch(), _Spec())


def test_a_compatible_real_pair_is_accepted():
    require_compatible(_real_batch(), _Spec(carrier_hz=F_REF, carrier_rate_hz=0.0))


def test_r1_a_channel_weight_with_a_kernel_carrier_is_refused():
    """The H5 guard: the moment the family gains complex Channel weights it
    inherits the carrier double-count, so this rule lands with them."""

    with pytest.raises(ValueError, match="double-counted carrier phase"):
        require_compatible(_channel_batch(), _Spec(carrier_hz=F_REF, carrier_rate_hz=0.0))


def test_r2_a_phaseless_weight_with_no_carrier_owner_is_refused():
    with pytest.raises(ValueError, match="missing carrier phase"):
        require_compatible(_real_batch(), _Spec(carrier_hz=0.0, carrier_rate_hz=0.0))


def test_r3_a_frozen_channel_weight_needs_the_carrier_rate():
    batch = _channel_batch()
    with pytest.raises(ValueError, match="understated Doppler"):
        require_compatible(batch, _Spec(carrier_rate_hz=0.5 * F_REF))


def test_r3_a_frozen_real_weight_needs_the_kernel_carrier_at_f_ref():
    with pytest.raises(ValueError, match="understated Doppler"):
        require_compatible(
            _real_batch(), _Spec(carrier_hz=0.5 * F_REF, carrier_rate_hz=0.0)
        )


def test_r4_a_refreshed_weight_refuses_a_carrier_rate():
    batch = _channel_batch(slow_time_mode=SlowTimeMode.REFRESHED_WEIGHT_NO_RATE)
    with pytest.raises(ValueError, match="double-counted Doppler"):
        require_compatible(batch, _Spec(carrier_rate_hz=F_REF))


def test_r4_a_refreshed_weight_refuses_a_published_delay_rate():
    batch = _channel_batch(slow_time_mode=SlowTimeMode.REFRESHED_WEIGHT_NO_RATE)
    with pytest.raises(ValueError, match="double-counted Doppler"):
        require_compatible(batch, _Spec(carrier_rate_hz=0.0))


def test_r4_a_refreshed_weight_without_a_rate_is_accepted():
    batch = _channel_batch(
        slow_time_mode=SlowTimeMode.REFRESHED_WEIGHT_NO_RATE, delay_rate=None
    )
    require_compatible(batch, _Spec(carrier_rate_hz=0.0))


def test_r5_spreading_applied_twice_is_refused():
    with pytest.raises(ValueError, match="double-counted free-space spreading"):
        require_compatible(_channel_batch(), _Spec(applies_spreading=True))


def test_r6_transmit_power_applied_twice_is_refused():
    with pytest.raises(ValueError, match="double-counted transmit power"):
        require_compatible(
            _channel_batch(), _SensorAwareSpec(tx_power_mode="config_power_dbm")
        )


def test_r6_accepts_the_sensor_owner_that_defers_to_the_weight():
    require_compatible(_channel_batch(), _SensorAwareSpec())


def test_r7_a_reference_frequency_mismatch_is_refused():
    with pytest.raises(ValueError, match="reference frequency mismatch"):
        require_compatible(_channel_batch(), _Spec(reference_frequency_hz=24.0e9))


def test_r8_a_wideband_response_is_refused_and_names_phase_8():
    batch = _channel_batch(
        frequency_response=torch.zeros(3, 4, dtype=torch.complex64),
        frequency_offsets_hz=torch.zeros(4, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="Phase 8"):
        require_compatible(batch, _Spec())


def test_a_spec_that_does_not_declare_the_protocol_is_refused():
    class Partial:
        carrier_hz = 0.0
        carrier_rate_hz = F_REF

    with pytest.raises(TypeError, match="WaveformSpecProtocol"):
        require_compatible(_channel_batch(), Partial())


def test_the_protocol_is_runtime_checkable_for_the_specs_used_here():
    assert isinstance(_Spec(), WaveformSpecProtocol)
