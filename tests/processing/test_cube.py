"""The attach point, and the assumption it is allowed to make.

``ProcessingCube.from_synthesis`` is a transpose and a metadata pairing. What
matters is what it is allowed NOT to do: it carries no ``row_valid`` awareness,
because a dead row was masked on the WEIGHT before the waveform kernel launched
and its contribution to the cube is a literal zero. That is asserted here rather
than assumed, and it is asserted the strongest available way - bitwise against a
cube synthesized from a batch that never contained the row at all.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from support import exact_bin_grid as grid
from support import multi_endpoint_driver as drv
from witwin.radar.processing import ProcessingAxes, ProcessingCube
from witwin.radar.synthesis import synthesize_fmcw_beat
from witwin.radar.synthesis.assembly import assemble_frame_cube
from witwin.radar.synthesis.contracts import SynthesisResult

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def frame():
    pytest.importorskip("witwin.channel")
    spike = grid.make_spike()
    composed, _, _ = spike.frame()
    return spike, composed, drv.to_synthesis(composed)


def _cube(batch, spec):
    result = SynthesisResult.from_fmcw_beat(synthesize_fmcw_beat(batch, spec), spec)
    axes = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())
    return ProcessingCube.from_synthesis(result, axes), result, axes


def test_the_cube_is_the_frame_assembly_transpose_and_not_a_second_packer(frame):
    """Bitwise against ``assemble_frame_cube``: one layout, one owner."""

    _, _, batch = frame
    spec = grid.fmcw_spec(2)
    cube, result, axes = _cube(batch, spec)
    expected = assemble_frame_cube(
        result.cube, num_tx=axes.num_tx, num_rx=axes.num_rx
    )
    assert torch.equal(cube.data, expected)
    assert tuple(cube.data.shape) == (
        axes.num_tx,
        axes.num_rx,
        spec.num_chirps,
        spec.num_samples,
    )
    assert cube.axes is axes


def test_a_dead_row_reaches_the_cube_as_an_exact_zero(frame):
    """Processing needs no row mask, and this is why.

    ``torch.equal``, not ``allclose``: the waveform kernel writes a literal
    ``0.0`` into a masked row's accumulation slot, so a cube built with the row
    masked and a cube built from a batch that never held it are the same
    numbers. Anything looser would let a dead row leak a denormal into a
    component export.
    """

    _, _, batch = frame
    spec = grid.fmcw_spec(2)
    keep = torch.ones(batch.path_count, dtype=torch.bool, device=batch.device)
    keep[1] = False
    keep[batch.path_count - 1] = False

    masked, _, _ = _cube(replace(batch, row_valid=keep.contiguous()), spec)
    reduced, _, _ = _cube(_select(batch, keep), spec)
    assert torch.equal(masked.data, reduced.data)


def _select(batch, keep):
    """A batch physically containing only ``keep``, with rebuilt CSR offsets."""

    from witwin.radar.paths.contracts import RadarPathTopology

    index = torch.nonzero(keep, as_tuple=False).flatten()
    pair = batch.sensor_pair_index[index].contiguous()
    counts = torch.bincount(pair, minlength=batch.sensor_pair_count)
    offsets = torch.zeros(
        batch.sensor_pair_count + 1, dtype=torch.int64, device=pair.device
    )
    offsets[1:] = torch.cumsum(counts, dim=0)
    topology = batch.topology
    return replace(
        batch,
        path_count=int(index.numel()),
        sensor_pair_index=pair,
        pair_offsets=offsets.contiguous(),
        total_delay_s=batch.total_delay_s[index].contiguous(),
        delay_rate=(
            None if batch.delay_rate is None else batch.delay_rate[index].contiguous()
        ),
        complex_transfer_ref=batch.complex_transfer_ref[index].contiguous(),
        topology=RadarPathTopology(
            radar_source_id=topology.radar_source_id[index].contiguous(),
            site_id=topology.site_id[index].contiguous(),
            radar_sink_id=topology.radar_sink_id[index].contiguous(),
            inbound_row=topology.inbound_row[index].contiguous(),
            outbound_row=topology.outbound_row[index].contiguous(),
        ),
        row_valid=None,
    )


def test_the_cube_refuses_a_result_its_metadata_record_does_not_describe(frame):
    """Two crossings, each of which would be a silent wrong answer."""

    _, _, batch = frame
    spec = grid.fmcw_spec(2)
    result = SynthesisResult.from_fmcw_beat(synthesize_fmcw_beat(batch, spec), spec)
    axes = ProcessingAxes.from_synthesis(result, spec, grid.array_spec())

    with pytest.raises(ValueError, match="different products"):
        ProcessingCube.from_synthesis(
            replace(result, axes=("symbol", "sensor_pair", "subcarrier")), axes
        )
    with pytest.raises(ValueError, match="reconciled backwards"):
        ProcessingCube.from_synthesis(
            replace(result, phasor="exp(-j*k*d)"), axes
        )


def test_the_cube_refuses_a_real_tensor(frame):
    _, _, batch = frame
    spec = grid.fmcw_spec(2)
    _, result, axes = _cube(batch, spec)
    with pytest.raises(TypeError, match="complex IQ"):
        ProcessingCube(data=result.cube.real, axes=axes)
