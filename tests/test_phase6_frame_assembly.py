"""Structural packing from the synthesis pair axis to the sigproc frame cube.

CPU only and value-free on purpose: nothing here is physics, and everything
here is the kind of index bookkeeping that is silently wrong for a long time.
The failure this file exists to prevent has no numerical signature at all - a
TX/RX transpose produces a frame of exactly the right shape, full of exactly the
right numbers, in exactly the wrong channels.
"""

from __future__ import annotations

import pytest
import torch

import witwin.radar.paths as paths  # noqa: E402
from witwin.radar.paths import validate_pair_ordering  # noqa: E402
from witwin.radar.synthesis.assembly import (  # noqa: E402
    FRAME_CUBE_AXES,
    PAIR_RANK_LAYOUT,
    assemble_frame_cube,
    pair_rx_index,
    pair_tx_index,
)

NUM_TX = 3
NUM_RX = 4
NUM_CHIRPS = 2
NUM_SAMPLES = 5


def _pair_rank(tx: int, rx: int) -> int:
    """The composed pair rank, written out from :data:`PAIR_RANK_LAYOUT`."""

    return rx * NUM_TX + tx


def _labelled_cube() -> torch.Tensor:
    """``[chirp, pair, sample]`` where every entry names its own coordinates."""

    cube = torch.empty((NUM_CHIRPS, NUM_TX * NUM_RX, NUM_SAMPLES), dtype=torch.complex64)
    for chirp in range(NUM_CHIRPS):
        for pair in range(NUM_TX * NUM_RX):
            for sample in range(NUM_SAMPLES):
                cube[chirp, pair, sample] = complex(pair * 1000 + chirp * 10 + sample, 0.0)
    return cube


# --------------------------------------------------------------------------
# The numbering, stated once and checked against its source
# --------------------------------------------------------------------------


def test_the_declared_pair_numbering_is_the_composers_own():
    """``PAIR_RANK_LAYOUT`` must be what ``paths.sink_major_rank`` does.

    The assembly module writes the numbering down in prose and then relies on
    it arithmetically. If the composer ever changed convention, every statement
    in this file would still be internally consistent and every frame would have
    its TX and RX axes swapped, so the prose is checked against the code that
    produces the ranks rather than against itself.
    """

    assert PAIR_RANK_LAYOUT == "sink_major: pair = rx_rank * num_tx + tx_rank"
    assert FRAME_CUBE_AXES == ("tx", "rx", "chirp", "sample")

    sources = list(range(100, 100 + NUM_TX))
    sinks = list(range(200, 200 + NUM_RX))
    rank = paths.sink_major_rank(sources, sinks)
    for tx, source in enumerate(sources):
        for rx, sink in enumerate(sinks):
            assert rank(source, sink) == _pair_rank(tx, rx)


def test_the_transmitter_and_receiver_tables_invert_the_pair_rank():
    tx_index = pair_tx_index(num_tx=NUM_TX, num_rx=NUM_RX, sensor_pair_count=NUM_TX * NUM_RX, device="cpu")
    rx_index = pair_rx_index(num_tx=NUM_TX, num_rx=NUM_RX, sensor_pair_count=NUM_TX * NUM_RX, device="cpu")
    assert tx_index.dtype is torch.int32
    assert rx_index.dtype is torch.int32
    for tx in range(NUM_TX):
        for rx in range(NUM_RX):
            pair = _pair_rank(tx, rx)
            assert int(tx_index[pair]) == tx
            assert int(rx_index[pair]) == rx

    # The transposed derivation, which is the natural wrong answer, does not
    # agree - so the test above is discriminating rather than tautological.
    transposed = torch.div(torch.arange(NUM_TX * NUM_RX), NUM_RX, rounding_mode="floor").to(torch.int32)
    assert not torch.equal(tx_index, transposed)


# --------------------------------------------------------------------------
# T1.9  the frame cube
# --------------------------------------------------------------------------


def test_the_frame_cube_puts_every_pair_in_its_own_tx_rx_slot():
    """The round trip, with entries that name their own coordinates.

    Each source entry is ``pair * 1000 + chirp * 10 + sample``, so an assembled
    entry that lands anywhere other than ``(tx, rx, chirp, sample)`` reads back
    a number that says exactly where it came from.
    """

    frame = assemble_frame_cube(_labelled_cube(), num_tx=NUM_TX, num_rx=NUM_RX)
    assert tuple(frame.shape) == (NUM_TX, NUM_RX, NUM_CHIRPS, NUM_SAMPLES)
    assert frame.is_contiguous()

    for tx in range(NUM_TX):
        for rx in range(NUM_RX):
            for chirp in range(NUM_CHIRPS):
                for sample in range(NUM_SAMPLES):
                    assert complex(frame[tx, rx, chirp, sample]) == complex(
                        _pair_rank(tx, rx) * 1000 + chirp * 10 + sample, 0.0
                    )


def test_assembly_is_a_transpose_and_not_a_bare_reshape():
    """The failure mode, made explicit.

    ``permute(1, 0, 2).view(num_tx, num_rx, ...)`` is the obvious
    implementation and it is wrong: the composed rank is sink major, so a bare
    view splits the pair axis into ``(rx, tx)`` and labels it ``(tx, rx)``. At
    a square array the shapes even agree, which is how the bug would ship.
    """

    cube = _labelled_cube()
    correct = assemble_frame_cube(cube, num_tx=NUM_TX, num_rx=NUM_RX)
    bare_view = cube.permute(1, 0, 2).reshape(NUM_RX, NUM_TX, NUM_CHIRPS, NUM_SAMPLES).contiguous()
    assert not torch.equal(correct, bare_view.permute(0, 1, 2, 3)[:NUM_TX])
    assert torch.equal(correct, bare_view.permute(1, 0, 2, 3).contiguous())


def test_a_square_array_still_transposes():
    """The square case, where a wrong assembly has the right shape.

    With ``num_tx == num_rx`` a transposed assembly raises nothing, produces the
    declared shape, and mis-steers every angle. This is the only test in the
    file that would fail against the obvious wrong implementation while every
    shape assertion elsewhere still passed.
    """

    cube = torch.empty((1, 4, 1), dtype=torch.complex64)
    for pair in range(4):
        cube[0, pair, 0] = complex(pair, 0.0)
    frame = assemble_frame_cube(cube, num_tx=2, num_rx=2)
    # pair = rx * 2 + tx, so (tx=1, rx=0) is pair 1 and (tx=0, rx=1) is pair 2.
    assert complex(frame[1, 0, 0, 0]) == complex(1.0, 0.0)
    assert complex(frame[0, 1, 0, 0]) == complex(2.0, 0.0)


def test_assembly_keeps_the_gradient_and_touches_no_value():
    """Structural packing, so the tape passes straight through it."""

    cube = torch.zeros((NUM_CHIRPS, NUM_TX * NUM_RX, NUM_SAMPLES), dtype=torch.float32).requires_grad_(True)
    frame = assemble_frame_cube(cube, num_tx=NUM_TX, num_rx=NUM_RX)
    assert frame.requires_grad
    frame.sum().backward()
    assert torch.equal(cube.grad, torch.ones_like(cube))


def test_a_cube_whose_pair_axis_is_not_the_array_is_refused():
    cube = _labelled_cube()
    with pytest.raises(ValueError, match="sensor pairs but the array is"):
        assemble_frame_cube(cube, num_tx=NUM_TX, num_rx=NUM_RX + 1)
    with pytest.raises(ValueError, match="must have shape"):
        assemble_frame_cube(cube[0], num_tx=NUM_TX, num_rx=NUM_RX)
    with pytest.raises(ValueError, match="num_tx must be a positive int"):
        assemble_frame_cube(cube, num_tx=0, num_rx=NUM_RX)


def test_a_pair_count_that_is_not_the_array_is_refused_by_the_tables():
    with pytest.raises(ValueError, match="the same front end"):
        pair_tx_index(num_tx=NUM_TX, num_rx=NUM_RX, sensor_pair_count=7, device="cpu")
    with pytest.raises(ValueError, match="the same front end"):
        pair_rx_index(num_tx=NUM_TX, num_rx=NUM_RX, sensor_pair_count=7, device="cpu")


# --------------------------------------------------------------------------
# The freeze-time check
# --------------------------------------------------------------------------


def test_pair_ordering_accepts_a_partition_with_empty_segments():
    """Empty segments are normal, including a trailing one.

    A transmitter that discovered nothing still owns a segment, and the cube
    keeps its declared shape. That is also why the pair count is a parameter
    rather than ``max(rank) + 1``: with the last segment empty, deriving it from
    the values would silently shrink the array by a channel.
    """

    ranks = torch.tensor([0, 0, 1, 3, 3], dtype=torch.int64)
    validate_pair_ordering(ranks, num_tx=2, num_rx=3, sensor_pair_count=6)
    assert int(ranks.max()) + 1 != 6

    empty = torch.zeros(0, dtype=torch.int64)
    validate_pair_ordering(empty, num_tx=2, num_rx=3, sensor_pair_count=6)


def test_pair_ordering_refuses_a_partition_that_is_not_this_array():
    ranks = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    with pytest.raises(ValueError, match="but the array is"):
        validate_pair_ordering(ranks, num_tx=2, num_rx=3, sensor_pair_count=4)


def test_pair_ordering_refuses_a_shuffled_or_out_of_range_partition():
    shuffled = torch.tensor([0, 2, 1, 3], dtype=torch.int64)
    with pytest.raises(ValueError, match="non-decreasing"):
        validate_pair_ordering(shuffled, num_tx=2, num_rx=2, sensor_pair_count=4)

    out_of_range = torch.tensor([0, 1, 9], dtype=torch.int64)
    with pytest.raises(ValueError, match="outside the"):
        validate_pair_ordering(out_of_range, num_tx=2, num_rx=2, sensor_pair_count=4)

    wrong_dtype = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(TypeError, match="must be int64"):
        validate_pair_ordering(wrong_dtype, num_tx=2, num_rx=2, sensor_pair_count=4)
