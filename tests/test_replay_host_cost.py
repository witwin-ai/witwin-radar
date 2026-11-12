"""What a frame's replay costs on the host as the scene gains structures.

Channel's consumer refuses a compiled scene whose source world was edited in
place, and it checks that on every ``evaluate`` and ``reevaluate``: it reads the
structure, material and assignment state of the source world on the host and
compares it with the state recorded at compile time. That walk is ``O(structures)``
host attribute reads, with no device work, and it sits inside Radar's frame
loop, which replays each frozen leg once per frame.

This file pins it as counts on a scene of one wall plus ``N`` far-away tiles
that contribute no row: one walk per replayed leg whatever ``N`` is, a walk
that visits every structure, and a device-to-host budget that does not grow
with ``N``. Wall time is printed (contended, informational) and never asserted.
"""

from __future__ import annotations

import time

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv
from support import multi_endpoint_geometry as geo
from support import multi_endpoint_world as world
from support.dsp_ledger import DspLedger

pytestmark = pytest.mark.gpu

#: Tile counts compared. 63 tiles plus the wall is a 64-structure scene.
TILE_COUNTS = (0, 63)

#: The tiles lie in the plane ``y = 50`` m, 1 m squares from ``x = -100`` m
#: down. Every specular point of the fixture endpoints on that plane is near
#: ``x = 1``, so no tile adds a reflection row and the replayed legs are the
#: same at every tile count.
TILE_PLANE_Y_M = 50.0


def _scene_with_tiles(tile_count: int):
    from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

    scene, _ = world.make_scene()
    structures = list(scene.structures)
    faces = torch.tensor(((0, 1, 2), (0, 2, 3)), dtype=torch.int64)
    for index in range(tile_count):
        x0 = -100.0 - 2.0 * index
        vertices = torch.tensor(
            (
                (x0, TILE_PLANE_Y_M, 0.0),
                (x0 + 1.0, TILE_PLANE_Y_M, 0.0),
                (x0 + 1.0, TILE_PLANE_Y_M, 1.0),
                (x0, TILE_PLANE_Y_M, 1.0),
            ),
            dtype=torch.float32,
        )
        mesh = Mesh(vertices=vertices, faces=faces, recenter=False, fill_mode="surface", topology_diagnostics=False)
        ids = 1000 + index
        structures.append(
            Structure(
                geometry=mesh,
                material=PhysicalMaterial(name=f"tile{index}", eps_r=geo.WALL_EPS_R, sigma_e=geo.WALL_SIGMA_E),
                structure_id=ids,
                material_id=ids,
                assignment_id=ids,
                surface_id=ids,
            )
        )
    return Scene(structures=tuple(structures), endpoints=list(scene.endpoints))


def _compiled(tile_count: int):
    from witwin.channel.scene import compile as compile_scene

    return compile_scene(_scene_with_tiles(tile_count), reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ)


def test_c9_each_replayed_leg_walks_the_source_world_once_and_reads_nothing_more_from_the_device(monkeypatch):
    import witwin.channel.scene.compiler as channel_compiler

    measured = {}
    for tile_count in TILE_COUNTS:
        compiled = _compiled(tile_count)
        assert len(compiled.structures) == 1 + tile_count
        spike = drv.MultiEndpointSpike(compiled=compiled)
        rows = [int(leg.row_valid.numel()) for leg in spike.legs()]

        walks = []
        original = channel_compiler.world_state

        def counting(scene_or_snapshot, _original=original, _walks=walks):
            state = _original(scene_or_snapshot)
            _walks.append(len(state.geometry_rows))
            return state

        monkeypatch.setattr(channel_compiler, "world_state", counting)
        ledger = DspLedger(monkeypatch, ())
        legs = spike.legs()
        monkeypatch.undo()

        assert [int(leg.row_valid.numel()) for leg in legs] == rows
        measured[tile_count] = {"rows": rows, "walks": list(walks), "host": dict(ledger.host)}

        seconds = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            spike.legs()
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - start)
        print(f"C-9 {1 + tile_count} structures: one frame's two replays {min(seconds) * 1e3:.3f} ms (contended)")

    small, large = (measured[count] for count in TILE_COUNTS)
    assert small["rows"] == large["rows"]
    # One walk per replayed leg (inbound and outbound), visiting every structure.
    assert small["walks"] == [1, 1]
    assert large["walks"] == [1 + TILE_COUNTS[1]] * 2
    # The host-observation budget is the two validation copies, at any size.
    assert small["host"] == large["host"]
    assert small["host"]["item"] + small["host"]["cpu"] + small["host"]["tolist"] <= 2, small["host"]
