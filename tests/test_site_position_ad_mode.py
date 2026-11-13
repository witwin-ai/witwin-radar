"""A scatter-site position that carries a derivative needs the AD mode that carries it.

A site position is an endpoint of both legs. Channel's fixed-topology replay
publishes the endpoint-position derivative only in the mode the verb asks for:
``ad_mode="vjp"`` for a gradient, ``ad_mode="jvp"`` for a forward tangent. Under
``"none"`` the legs are discovery's detached rows, so a marked position used to
come back with no derivative at all, or - for a trajectory or a mesh target,
whose join and facet response still saw the tape - with the Radar-side part
alone and the legs' part missing. Either is a silently wrong derivative, so a
marked site position in a mode that does not carry it is refused before any
discovery, any native launch and any result.

The routes are the three ways a site position is formed: ``PointTargets``
positions (refused when the session's site policy is built, before even the
scene compile), a ``PointTargets`` trajectory's output and a ``MeshTargets``
structure's world vertices (both refused when the frame resolves them, after
Channel's compile and before anything reads them), plus a ``StructureTargets``
anchor through the same policy. The instrument is calibrated against the same
call in the mode that carries the derivative, so a count of zero is not a
vacuous one.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwad

from witwin.radar.binding import ScatterSitePolicy

#: (what the position carries, the verb's ad_mode, the mode the refusal asks for).
MISMATCHES = [
    ("gradient", "none", "vjp"),
    ("gradient", "jvp", "vjp"),
    ("tangent", "none", "jvp"),
    ("tangent", "vjp", "jvp"),
]
POSITIONS = ((6.0, 0.5, 1.0), (5.0, -0.4, -0.8))


def _marked(base: torch.Tensor, marked: str) -> torch.Tensor:
    """``base`` as a gradient leaf, or as a forward dual (inside an open dual level)."""

    if marked == "gradient":
        return base.clone().requires_grad_(True)
    return fwad.make_dual(base.clone(), torch.ones_like(base))


class _Work:
    """Counts the work a session does: compiles, discoveries, joins and facet launches."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from witwin.radar import channel
        from witwin.radar.cuda import runtime

        self.calls = {"compile": 0, "discover": 0, "join": 0, "facet_geometry": 0}
        table = runtime.build_extension()
        compile_scene, discover = channel.compile_scene, channel.ChannelPropagationAdapter.discover
        join, facets = table.two_way_join_forward, table.facet_geometry_forward

        def counted(name, original):
            def call(*args, **kwargs):
                self.calls[name] += 1
                return original(*args, **kwargs)

            return call

        monkeypatch.setattr(channel, "compile_scene", counted("compile", compile_scene))
        monkeypatch.setattr(channel.ChannelPropagationAdapter, "discover", counted("discover", discover))
        monkeypatch.setattr(table, "two_way_join_forward", counted("join", join))
        monkeypatch.setattr(table, "facet_geometry_forward", counted("facet_geometry", facets))


def _point_cube(targets, ad_mode: str) -> torch.Tensor:
    from test_pattern_leaves import _radar
    from witwin.core import Scene

    return _radar().simulate(Scene(), targets, times=(0.0,), reflections=0, ad_mode=ad_mode).cube


def _static(position: torch.Tensor):
    from witwin.radar import PointTargets

    return PointTargets(positions=position, rcs=1.0)


def _walking(position: torch.Tensor):
    from witwin.radar import PointTargets

    step = torch.tensor([[0.0, 0.0, 1.0]], device=position.device)
    return PointTargets(positions=position.detach(), rcs=1.0, trajectory=lambda t: position + step * t)


def _mesh_scene(vertices: torch.Tensor):
    from support import oracle_scenes as sc
    from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

    _, faces = sc.plate_mesh(2, side_m=0.05, range_m=3.0)
    mesh = Mesh(
        vertices=vertices, faces=torch.tensor(faces), recenter=False, fill_mode="surface", topology_diagnostics=False
    )
    plate = Structure(
        geometry=mesh,
        material=PhysicalMaterial.perfect_conductor(),
        structure_id=1,
        material_id=1,
        assignment_id=1,
        surface_id=1,
    )
    return Scene(structures=(plate,), endpoints=[])


def _plate_vertices() -> torch.Tensor:
    from support import oracle_scenes as sc

    vertices, _ = sc.plate_mesh(2, side_m=0.05, range_m=3.0)
    # A slight tilt, as tests/test_mesh_targets.py uses, so no derivative is degenerate.
    return torch.tensor(vertices + np.outer(vertices[:, 1], [0.08, 0.0, 0.0]), dtype=torch.float32)


def _mesh_rows(vertices: torch.Tensor, ad_mode: str) -> torch.Tensor:
    from support import oracle_scenes as sc

    from witwin.radar import MeshTargets

    scene = _mesh_scene(vertices)
    return (
        sc.radar().trace(scene, MeshTargets(), times=(0.0,), reflections=0, ad_mode=ad_mode).rows().complex_transfer_ref
    )


def _carried(value: torch.Tensor, marked: str) -> bool:
    return value.requires_grad if marked == "gradient" else fwad.unpack_dual(value).tangent is not None


# ---------------------------------------------------------------------------
# PointTargets: a declared tensor is refused before the session does anything
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(("marked", "ad_mode", "wanted"), MISMATCHES)
def test_a_marked_point_target_position_is_refused_before_any_work(monkeypatch, marked, ad_mode, wanted):
    pytest.importorskip("witwin.channel")
    work = _Work(monkeypatch)
    base = torch.tensor(POSITIONS, device="cuda")
    with fwad.dual_level():
        targets = _static(_marked(base, marked))
        with pytest.raises(
            RuntimeError, match=r"PointTargets\.positions carries a (gradient|forward tangent)"
        ) as raised:
            _point_cube(targets, ad_mode)
    assert f"pass ad_mode={wanted!r} (got ad_mode={ad_mode!r})" in str(raised.value)
    assert "detach" in str(raised.value)
    # Refused while the session's site policy is built: not even the scene compiles.
    assert work.calls == {"compile": 0, "discover": 0, "join": 0, "facet_geometry": 0}


@pytest.mark.gpu
@pytest.mark.parametrize(("marked", "ad_mode", "wanted"), MISMATCHES)
def test_a_marked_point_target_trajectory_is_refused_before_any_discovery(monkeypatch, marked, ad_mode, wanted):
    pytest.importorskip("witwin.channel")
    work = _Work(monkeypatch)
    base = torch.tensor(POSITIONS, device="cuda")
    with fwad.dual_level():
        targets = _walking(_marked(base, marked))
        with pytest.raises(RuntimeError, match=r"PointTargets\.trajectory carries a") as raised:
            _point_cube(targets, ad_mode)
    assert f"pass ad_mode={wanted!r} (got ad_mode={ad_mode!r})" in str(raised.value)
    # The trajectory is evaluated per frame, so the frame's scene compile has
    # run; nothing that reads a site position has.
    assert work.calls["discover"] == 0 and work.calls["join"] == 0


@pytest.mark.gpu
@pytest.mark.parametrize("marked", ["gradient", "tangent"])
@pytest.mark.parametrize("route", ["static", "trajectory"])
def test_the_matching_mode_still_carries_a_point_target_derivative(monkeypatch, route, marked):
    """The calibration: the same call in the mode that carries it does the work and publishes the derivative."""

    pytest.importorskip("witwin.channel")
    work = _Work(monkeypatch)
    build = _static if route == "static" else _walking
    with fwad.dual_level():
        cube = _point_cube(build(_marked(torch.tensor(POSITIONS, device="cuda"), marked)), _mode(marked))
        assert _carried(cube, marked)
    assert work.calls["compile"] >= 1 and work.calls["discover"] >= 2 and work.calls["join"] >= 1


@pytest.mark.gpu
def test_a_plain_position_under_none_keeps_its_radar_side_leaves():
    """Only a marked site position is refused: a cross-section leaf still differentiates under ``none``."""

    pytest.importorskip("witwin.channel")
    from witwin.radar import PointTargets

    rcs = torch.tensor(2.0, device="cuda", requires_grad=True)
    cube = _point_cube(PointTargets(positions=torch.tensor(POSITIONS, device="cuda"), rcs=rcs), "none")
    (gradient,) = torch.autograd.grad(cube.abs().square().sum(), rcs)
    assert float(gradient) > 0.0


def _mode(marked: str) -> str:
    return "vjp" if marked == "gradient" else "jvp"


# ---------------------------------------------------------------------------
# MeshTargets: the world vertices place every facet site
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(("marked", "ad_mode", "wanted"), MISMATCHES)
def test_marked_mesh_target_vertices_are_refused_before_the_facet_operator(monkeypatch, marked, ad_mode, wanted):
    pytest.importorskip("witwin.channel")
    work = _Work(monkeypatch)
    with fwad.dual_level():
        vertices = _marked(_plate_vertices(), marked)
        with pytest.raises(RuntimeError, match=r"MeshTargets world-vertex tensor .* carries a") as raised:
            _mesh_rows(vertices, ad_mode)
    assert f"pass ad_mode={wanted!r} (got ad_mode={ad_mode!r})" in str(raised.value)
    assert work.calls["facet_geometry"] == 0 and work.calls["discover"] == 0 and work.calls["join"] == 0


@pytest.mark.gpu
@pytest.mark.parametrize("marked", ["gradient", "tangent"])
def test_the_matching_mode_still_carries_a_mesh_vertex_derivative(monkeypatch, marked):
    pytest.importorskip("witwin.channel")
    work = _Work(monkeypatch)
    with fwad.dual_level():
        rows = _mesh_rows(_marked(_plate_vertices(), marked), _mode(marked))
        assert _carried(rows, marked)
    assert work.calls["facet_geometry"] >= 1 and work.calls["discover"] >= 2 and work.calls["join"] >= 1


# ---------------------------------------------------------------------------
# The site policy itself, on the host: every source admits the same way
# ---------------------------------------------------------------------------


def _anchored(translation: torch.Tensor) -> SimpleNamespace:
    """A snapshot with one structure whose rigid motion publishes ``translation``."""

    state = SimpleNamespace(structure_id=4, rigid_motion=SimpleNamespace(translation=translation))
    return SimpleNamespace(structures=(state,), time_s=0.0)


@pytest.mark.parametrize(("marked", "ad_mode", "wanted"), MISMATCHES)
def test_every_site_source_refuses_a_derivative_its_mode_does_not_carry(marked, ad_mode, wanted):
    cpu = torch.device("cpu")
    base = torch.tensor(POSITIONS)
    with fwad.dual_level():
        with pytest.raises(RuntimeError, match=rf"PointTargets\.positions .*pass ad_mode={wanted!r}"):
            ScatterSitePolicy.explicit(_marked(base, marked), ad_mode=ad_mode)
        moving = _marked(base, marked)
        trajectory = SimpleNamespace(at=lambda t: _kinematics(moving))
        policy = ScatterSitePolicy.explicit(base, trajectory=trajectory, ad_mode=ad_mode)
        with pytest.raises(RuntimeError, match=rf"PointTargets\.trajectory .*pass ad_mode={wanted!r}"):
            policy.resolve(SimpleNamespace(time_s=0.0), device=cpu)
        anchor = ScatterSitePolicy.structure_anchor(ad_mode=ad_mode)
        with pytest.raises(RuntimeError, match=rf"StructureTargets anchor .*pass ad_mode={wanted!r}"):
            anchor.resolve(_anchored(_marked(torch.tensor([1.0, 2.0, 3.0]), marked)), device=cpu)


@pytest.mark.parametrize("marked", ["gradient", "tangent"])
def test_every_site_source_admits_the_mode_that_carries_it_and_a_reader_with_no_mode(marked):
    cpu = torch.device("cpu")
    with fwad.dual_level():
        for ad_mode in (_mode(marked), None):
            position = _marked(torch.tensor(POSITIONS), marked)
            assert ScatterSitePolicy.explicit(position, ad_mode=ad_mode).resolve(None, device=cpu) is position
            anchors = ScatterSitePolicy.structure_anchor(ad_mode=ad_mode).resolve(
                _anchored(_marked(torch.tensor([1.0, 2.0, 3.0]), marked)), device=cpu
            )
            assert _carried(anchors, marked)


def _kinematics(positions: torch.Tensor):
    from witwin.radar.propagation import Kinematics

    return Kinematics(positions_m=positions)
