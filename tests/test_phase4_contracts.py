"""CPU-testable validation of the Phase-4 Radar contracts.

These run without CUDA and without ``witwin-channel`` on purpose: the endpoint
structural checks are device-agnostic so that a caller gets the shape or dtype
complaint it actually made, and ``witwin.radar.propagation`` never imports the
Channel adapter, so the package is importable wherever Radar is.
"""

from __future__ import annotations

import pytest
import torch

from witwin.radar.propagation import (
    RadarEndpointSpec,
    RadarLegBatch,
    require_endpoint_role,
)


def _spec(rows: int = 2, *, with_power: bool = False) -> RadarEndpointSpec:
    return RadarEndpointSpec(
        stable_ids=torch.arange(rows, dtype=torch.int64),
        positions_m=torch.zeros((rows, 3), dtype=torch.float32),
        polarizations=torch.zeros((rows, 3), dtype=torch.float32),
        powers_w=torch.ones(rows, dtype=torch.float32) if with_power else None,
    )


def test_propagation_package_does_not_import_the_channel_adapter():
    """The package root must not pull the adapter in, so importing
    ``witwin.radar.propagation`` never requires ``witwin-channel``.

    This is asserted on the source text, not on ``hasattr``: importing the
    adapter anywhere in the process binds it as an attribute of its package,
    so a runtime probe would only measure whether some other test ran first.
    The isolated-process proof lives in ``test_phase4_import_boundary.py``.
    """

    import ast
    import pathlib

    import witwin.radar.propagation as package

    tree = ast.parse(pathlib.Path(package.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(alias.name for alias in node.names)
    assert "channel_consumer" not in imported
    assert not any(name.startswith("witwin.channel") for name in imported)
    assert "ChannelPropagationAdapter" not in package.__all__


def test_valid_endpoint_spec_reports_count_and_device():
    spec = _spec(3, with_power=True)
    assert spec.count == 3
    assert spec.device == torch.device("cpu")


def test_endpoint_spec_rejects_wrong_dtype():
    with pytest.raises(TypeError, match="positions_m must use"):
        RadarEndpointSpec(
            stable_ids=torch.zeros(1, dtype=torch.int64),
            positions_m=torch.zeros((1, 3), dtype=torch.float64),
            polarizations=torch.zeros((1, 3), dtype=torch.float32),
        )
    with pytest.raises(TypeError, match="stable_ids must use"):
        RadarEndpointSpec(
            stable_ids=torch.zeros(1, dtype=torch.int32),
            positions_m=torch.zeros((1, 3), dtype=torch.float32),
            polarizations=torch.zeros((1, 3), dtype=torch.float32),
        )


def test_endpoint_spec_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"positions_m must have shape \(N, 3\)"):
        RadarEndpointSpec(
            stable_ids=torch.zeros(1, dtype=torch.int64),
            positions_m=torch.zeros((1, 2), dtype=torch.float32),
            polarizations=torch.zeros((1, 3), dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="polarizations must have shape"):
        RadarEndpointSpec(
            stable_ids=torch.zeros(2, dtype=torch.int64),
            positions_m=torch.zeros((2, 3), dtype=torch.float32),
            polarizations=torch.zeros((1, 3), dtype=torch.float32),
        )


def test_endpoint_spec_rejects_non_tensor_and_non_contiguous():
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        RadarEndpointSpec(
            stable_ids=torch.zeros(1, dtype=torch.int64),
            positions_m=[[0.0, 0.0, 0.0]],
            polarizations=torch.zeros((1, 3), dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="must be contiguous"):
        RadarEndpointSpec(
            stable_ids=torch.zeros(2, dtype=torch.int64),
            positions_m=torch.zeros((3, 2), dtype=torch.float32).transpose(0, 1),
            polarizations=torch.zeros((2, 3), dtype=torch.float32),
        )


def test_endpoint_spec_requires_one_device():
    if not torch.cuda.is_available():
        pytest.skip("needs a second device to disagree with")
    with pytest.raises(ValueError, match="must share the positions_m device"):
        RadarEndpointSpec(
            stable_ids=torch.zeros(1, dtype=torch.int64, device="cuda"),
            positions_m=torch.zeros((1, 3), dtype=torch.float32),
            polarizations=torch.zeros((1, 3), dtype=torch.float32),
        )


def test_endpoint_role_contract():
    require_endpoint_role(_spec(with_power=True), "source")
    require_endpoint_role(_spec(), "sink")
    with pytest.raises(ValueError, match="source endpoint requires powers_w"):
        require_endpoint_role(_spec(), "source")
    with pytest.raises(ValueError, match="sink endpoint must not carry powers_w"):
        require_endpoint_role(_spec(with_power=True), "sink")
    with pytest.raises(ValueError, match="role must be"):
        require_endpoint_role(_spec(), "transmitter")


def _leg(**overrides) -> RadarLegBatch:
    rows = 2
    fields = {
        "leg_count": rows,
        "pair_count": 1,
        "pair_index": torch.zeros(rows, dtype=torch.int64),
        "pair_offsets": torch.tensor([0, rows], dtype=torch.int64),
        "source_index": torch.zeros(rows, dtype=torch.int32),
        "sink_index": torch.zeros(rows, dtype=torch.int32),
        "depth": torch.zeros(rows, dtype=torch.int32),
        "component_id": torch.zeros(rows, dtype=torch.int32),
        "delay_s": torch.zeros(rows, dtype=torch.float32),
        "coefficient": torch.zeros(rows, dtype=torch.complex64),
        "delay_rate": None,
        "row_valid": None,
        "diagnostics": None,
    }
    fields.update(overrides)
    return RadarLegBatch(**fields)


def test_leg_batch_accepts_a_well_formed_batch():
    leg = _leg()
    assert leg.device == torch.device("cpu")
    assert _leg(row_valid=torch.ones(2, dtype=torch.bool)).row_valid is not None
    assert _leg(delay_rate=torch.zeros(2, dtype=torch.float32)).delay_rate is not None


def test_leg_batch_rejects_inconsistent_rows():
    with pytest.raises(ValueError, match="pair_offsets must have shape"):
        _leg(pair_offsets=torch.tensor([0, 1, 2], dtype=torch.int64))
    with pytest.raises(ValueError, match="delay_s must have shape"):
        _leg(delay_s=torch.zeros(3, dtype=torch.float32))
    with pytest.raises(TypeError, match="coefficient must use"):
        _leg(coefficient=torch.zeros(2, dtype=torch.complex128))
    with pytest.raises(ValueError, match="leg_count must be a non-negative int"):
        _leg(leg_count=-1)
    with pytest.raises(ValueError, match="pair_count must be a non-negative int"):
        _leg(pair_count=-1)
