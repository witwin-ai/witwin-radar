"""Structural packing between the synthesis pair axis and the sigproc array.

A waveform kernel produces one cube per frame, ``[chirp, sensor_pair, sample]``,
because a sensor pair is exactly the partition the path rows are grouped by.
Every ``sigproc`` consumer instead wants ``[TX, RX, chirp, sample]``, because a
virtual-array index is what an AoA estimator steers. Converting between them is
pure structural packing - a permute, a view, and a contiguous copy - and is
therefore Python's job under the plan's orchestration allowlist. It evaluates no
physics and reads no tensor value.

The load-bearing detail is the pair NUMBERING, and the two sides do not agree:

* The composed pair rank is SINK MAJOR, ``pair = rx_rank * num_tx + tx_rank``.
  It comes from :func:`witwin.radar.paths._identity.sink_major_rank`, which
  mirrors the Channel consumer's own ``sink_row_index * source_count +
  source_row_index`` so that Radar does not put a second, silently different,
  virtual-array numbering on the same data.
* The ``sigproc`` virtual antenna index is TX MAJOR, ``va = tx * num_rx + rx``.
  ``sigproc/pointcloud.py::_compensate_tdm_phase`` slices ``va_start =
  tx * num_rx``, and ``frame2pointcloud`` flattens axes 0 and 1 of the rank-4
  frame in that order.

The two are transposes of each other, so assembling the cube with a bare
``view`` would swap TX and RX channels whenever ``num_tx != num_rx`` and would
silently mis-steer every angle whenever ``num_tx == num_rx``. The transpose is
performed here, once, in :func:`assemble_frame_cube`, and the same statement of
the numbering drives :func:`pair_tx_index`, which tells the beat kernel which
TDM slot each pair sits in. One convention, two consumers, no second copy.
"""

from __future__ import annotations

import torch


#: The composed pair numbering, stated once. Quoted by the tests that pin it
#: against ``witwin.radar.paths._identity.sink_major_rank``.
PAIR_RANK_LAYOUT = "sink_major: pair = rx_rank * num_tx + tx_rank"

#: The layout the rank-4 frame cube is published in.
FRAME_CUBE_AXES = ("tx", "rx", "chirp", "sample")


def _require_array(num_tx: int, num_rx: int) -> int:
    if type(num_tx) is not int or num_tx < 1:
        raise ValueError(f"num_tx must be a positive int, got {num_tx!r}")
    if type(num_rx) is not int or num_rx < 1:
        raise ValueError(f"num_rx must be a positive int, got {num_rx!r}")
    return num_tx * num_rx


def pair_tx_index(
    *,
    num_tx: int,
    num_rx: int,
    sensor_pair_count: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Which transmitter drives each sensor pair, as ``int32[pair_count]``.

    Under :data:`PAIR_RANK_LAYOUT` the transmitter rank is ``pair % num_tx``,
    not ``pair // num_rx``. Getting that backwards assigns the wrong TDM slot to
    every pair, which shifts each channel's Doppler phase by a whole chirp
    period and still produces a cube that looks entirely reasonable.

    The pair count is checked against the declared array rather than trusted:
    a batch frozen over a different front end than the waveform spec describes
    is a configuration error, and reinterpreting its pairs as this array's is
    exactly the silent wrong answer the TDM slot table exists to prevent.
    """

    expected = _require_array(num_tx, num_rx)
    if sensor_pair_count != expected:
        raise ValueError(
            f"this batch spans {sensor_pair_count} sensor pairs but the waveform "
            f"spec declares a {num_tx} x {num_rx} array, which is {expected} "
            "pairs; the pair partition and the array must be the same front end"
        )
    ranks = torch.arange(expected, device=device, dtype=torch.int32)
    return torch.remainder(ranks, num_tx)


def pair_rx_index(
    *,
    num_tx: int,
    num_rx: int,
    sensor_pair_count: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Which receiver owns each sensor pair, as ``int32[pair_count]``.

    The companion of :func:`pair_tx_index` under the same numbering. Nothing in
    Phase 6 needs it on the hot path; it exists so the numbering is written down
    in both directions in one place instead of being rederived by a reader.
    """

    expected = _require_array(num_tx, num_rx)
    if sensor_pair_count != expected:
        raise ValueError(
            f"this batch spans {sensor_pair_count} sensor pairs but the waveform "
            f"spec declares a {num_tx} x {num_rx} array, which is {expected} "
            "pairs; the pair partition and the array must be the same front end"
        )
    ranks = torch.arange(expected, device=device, dtype=torch.int32)
    return torch.div(ranks, num_tx, rounding_mode="floor")


def assemble_frame_cube(
    cube: torch.Tensor, *, num_tx: int, num_rx: int
) -> torch.Tensor:
    """``[chirp, pair, sample]`` -> ``[TX, RX, chirp, sample]``.

    Pure structural packing: one permute to bring the pair axis to the front,
    one view that splits it into ``(rx, tx)`` because the rank is sink major,
    one permute that puts TX first because ``sigproc`` is tx major, and one
    contiguous copy. No tensor value is read, no arithmetic is performed, and
    the gradient passes straight through, so this runs inside the per-frame host
    observation budget.

    The pair-count check is the reshape's precondition, not a physics check:
    a cube whose pair axis is not exactly ``num_tx * num_rx`` long cannot be
    split into the array at all.
    """

    expected = _require_array(num_tx, num_rx)
    if cube.dim() != 3:
        raise ValueError(
            "a synthesis cube must have shape (chirps, sensor_pairs, samples), "
            f"got {tuple(cube.shape)}"
        )
    if cube.shape[1] != expected:
        raise ValueError(
            f"this cube spans {cube.shape[1]} sensor pairs but the array is "
            f"{num_tx} x {num_rx} = {expected} pairs"
        )
    num_chirps = int(cube.shape[0])
    num_samples = int(cube.shape[2])
    return (
        cube.permute(1, 0, 2)
        .reshape(num_rx, num_tx, num_chirps, num_samples)
        .permute(1, 0, 2, 3)
        .contiguous()
    )


def validate_pair_ordering(
    sensor_pair_index: torch.Tensor,
    *,
    num_tx: int,
    num_rx: int,
    sensor_pair_count: int,
) -> None:
    """Check ONCE, at freeze time, that the pair partition is this array's.

    This reads ``sensor_pair_index`` on the host. That is deliberate and it is
    why it is a separate function from :func:`assemble_frame_cube`: the check
    belongs at freeze time, where the topology is decided and a host read costs
    nothing, and it must never run inside the frame loop, where the same read
    would be a per-frame device-to-host transfer.

    What is verified:

    * the declared pair count is the array's ``num_tx * num_rx``;
    * every row's pair rank is inside that range;
    * ranks are NON-DECREASING, which is what makes ``pair_offsets`` a half-open
      partition and the cube's pair axis an ordered TX/RX grid rather than a
      scatter.

    Empty segments stay legal - a transmitter/receiver pair that discovered
    nothing still owns a segment, and the cube keeps its declared shape. That is
    also why the pair count is a parameter rather than ``max(index) + 1``: with
    a trailing empty segment, and the multi-endpoint fixture produces exactly
    that, deriving it from the values would silently shrink the array.
    """

    expected = _require_array(num_tx, num_rx)
    if sensor_pair_count != expected:
        raise ValueError(
            f"the frozen topology spans {sensor_pair_count} sensor pairs but the "
            f"array is {num_tx} x {num_rx} = {expected} pairs"
        )
    if sensor_pair_index.dtype != torch.int64:
        raise TypeError(
            f"sensor_pair_index must be int64, got {sensor_pair_index.dtype}"
        )
    if sensor_pair_index.dim() != 1:
        raise ValueError(
            f"sensor_pair_index must be 1-D, got shape {tuple(sensor_pair_index.shape)}"
        )
    ranks = sensor_pair_index.tolist()
    previous = -1
    for row, rank in enumerate(ranks):
        if rank < 0 or rank >= expected:
            raise ValueError(
                f"row {row} names sensor pair {rank}, which is outside the "
                f"{num_tx} x {num_rx} array's range [0, {expected})"
            )
        if rank < previous:
            raise ValueError(
                f"sensor pair ranks must be non-decreasing so that pair_offsets "
                f"is a half-open partition; row {row} drops from {previous} to "
                f"{rank}"
            )
        previous = rank


__all__ = [
    "FRAME_CUBE_AXES",
    "PAIR_RANK_LAYOUT",
    "assemble_frame_cube",
    "pair_rx_index",
    "pair_tx_index",
    "validate_pair_ordering",
]
