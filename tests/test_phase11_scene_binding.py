"""The production scene -> endpoint/site binding (Phase 11 work item 1).

Until this phase the glue between a Core world and the propagation pipeline
existed only under ``tests/support``: ``multi_endpoint_world.compile_snapshot``
compiled, ``endpoint_batch`` packed, and the stable IDs were hard-coded fixture
constants. ``witwin.radar.simulation`` is the production owner of the same
three answers, and ``compile_scene`` is the production crossing that replaces
the fixture's private call to ``witwin.channel.scene.compile``.

What is asserted here is the part that a downstream numerical test cannot see,
because a wrong answer in any of it still produces a plausible cube:

* an ID is a function of ``(role, index)`` and of nothing else, so two bindings
  of one world agree - a frozen leg topology names its rows by identity;
* the two site specs SHARE one position tensor, so a gradient accumulates over
  both legs instead of over one;
* a source carries ``powers_w`` and a sink does not;
* an unsupported site request is refused by name rather than approximated;
* a reference-frequency mismatch is refused before any propagation request is
  built, and never silently recompiled;
* ``witwin.core.Mesh`` still rewrites authored world coordinates unless
  ``recenter=False`` is passed, which is the footgun R-ADR-009 records.
"""

from __future__ import annotations

import pytest
import torch
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import multi_endpoint_world as world  # noqa: E402

from witwin.radar.simulation import (
    DEFAULT_RECEIVER_ID_BASE,
    DEFAULT_SITE_ID_BASE,
    DEFAULT_TRANSMITTER_ID_BASE,
    SITE_EXCITATION_POWER_W,
    RadarWorldBinding,
    ScatterSitePolicy,
    StableIdAllocator,
    bind_radar_world,
)

# ---------------------------------------------------------------------------
# The allocator - host only, no CUDA, no Channel
# ---------------------------------------------------------------------------


def test_stable_ids_are_a_function_of_role_and_index_only():
    allocator = StableIdAllocator()
    first = allocator.allocate(transmitter_count=2, receiver_count=2, site_count=3)
    second = StableIdAllocator().allocate(transmitter_count=2, receiver_count=2, site_count=3)
    assert first == second
    transmitters, receivers, sites = first
    assert transmitters == (DEFAULT_TRANSMITTER_ID_BASE, DEFAULT_TRANSMITTER_ID_BASE + 1)
    assert receivers == (DEFAULT_RECEIVER_ID_BASE, DEFAULT_RECEIVER_ID_BASE + 1)
    assert sites == tuple(DEFAULT_SITE_ID_BASE + k for k in range(3))


def test_overlapping_id_blocks_are_refused_rather_than_renumbered():
    """Two endpoints sharing an ID join the wrong rows and still answer."""

    allocator = StableIdAllocator(transmitter_base=0, receiver_base=4, site_base=100)
    assert allocator.allocate(transmitter_count=4, receiver_count=2, site_count=1)
    with pytest.raises(ValueError, match="overlaps the receiver block"):
        allocator.allocate(transmitter_count=5, receiver_count=2, site_count=1)


def test_an_empty_endpoint_block_is_refused():
    with pytest.raises(ValueError, match="site_count must be a positive int"):
        StableIdAllocator().allocate(transmitter_count=1, receiver_count=1, site_count=0)


# ---------------------------------------------------------------------------
# The site policy - the refusal is the design, not a gap
# ---------------------------------------------------------------------------


def test_a_structure_without_a_world_anchor_names_the_mesh_deferral():
    """A static structure publishes no Core-owned site position.

    The message has to name the deferral, because the alternative a caller will
    otherwise reach for is a centroid or a surface sample, and that is new Torch
    geometry on a production path.
    """

    scene, mesh = world.make_scene()
    world.assert_world_coordinates_survived(mesh)
    policy = ScatterSitePolicy.structure_anchor()
    with pytest.raises(NotImplementedError, match="named Phase-11 deferral"):
        policy.resolve(scene.snapshot(0.0), device=torch.device("cpu"))


def test_a_moving_structure_publishes_its_core_owned_world_anchor():
    dynamic = world.make_dynamic_scene(wall_origin=(1.0, 2.0, 3.0), wall_velocity=(4.0, 0.0, 0.0))
    policy = ScatterSitePolicy.structure_anchor()
    anchors = policy.resolve(dynamic.at(0.5), device=torch.device("cpu"))
    assert anchors.shape == (1, 3)
    assert anchors.dtype == torch.float32
    torch.testing.assert_close(anchors[0], torch.tensor((3.0, 2.0, 3.0), dtype=torch.float32))


def test_a_structure_anchor_policy_refuses_an_unknown_structure():
    dynamic = world.make_dynamic_scene(wall_velocity=(1.0, 0.0, 0.0))
    policy = ScatterSitePolicy.structure_anchor(structure_ids=(7,))
    with pytest.raises(ValueError, match=r"structure_ids \[7\] are not in"):
        policy.resolve(dynamic.at(0.0), device=torch.device("cpu"))


def test_an_explicit_policy_passes_a_live_tensor_through_untouched():
    """The tape is the point: a rebuilt tensor is a different AD leaf."""

    positions = torch.tensor([geo.SITE_P_POSITION_M], dtype=torch.float32).requires_grad_(True)
    policy = ScatterSitePolicy.explicit(positions)
    resolved = policy.resolve(None, device=torch.device("cpu"))
    assert resolved is positions


def test_an_explicit_policy_refuses_a_silent_cast():
    positions = torch.tensor([geo.SITE_P_POSITION_M], dtype=torch.float64)
    policy = ScatterSitePolicy.explicit(positions)
    with pytest.raises(TypeError, match="must use torch.float32"):
        policy.resolve(None, device=torch.device("cpu"))


def test_the_two_site_sources_cannot_be_declared_together():
    with pytest.raises(ValueError, match="belongs to the structure_anchor"):
        ScatterSitePolicy(source="explicit", positions_m=[geo.SITE_P_POSITION_M], structure_ids=(1,))


# ---------------------------------------------------------------------------
# The Core.Mesh recentre footgun (R-ADR-009), still live
# ---------------------------------------------------------------------------


def test_core_mesh_still_rewrites_authored_world_coordinates_by_default():
    """``recenter=False`` is mandatory, and this is where it is measurable.

    Not a test of Core: it is the pin that keeps the production binding honest
    about a default it does not control. If Core ever fixes the default, this
    test fails and the contract note in ``simulation`` gets deleted with it,
    which is the outcome R-ADR-009 asks for.

    Measured on ``world_vertices``, which is what a compiler consumes, and NOT
    on ``mesh.vertices``. ``Mesh.vertices`` returns the authored tensor
    unchanged whatever ``recenter`` says - the subtraction happens inside
    ``_local_vertices_tensor`` - so a check written against ``mesh.vertices``
    cannot see the rewrite at all. That distinction is the whole reason this
    test exists at the production boundary rather than being left to the
    fixture guard.
    """

    from witwin.core import Mesh

    vertices = torch.tensor(geo.WALL_VERTICES_M, dtype=torch.float32)
    faces = torch.tensor(geo.WALL_FACES, dtype=torch.int64)

    def _mesh(**overrides):
        return Mesh(vertices=vertices, faces=faces, fill_mode="surface", topology_diagnostics=False, **overrides)

    recentred = _mesh().world_vertices
    assert float(recentred[:, 0].min()) != pytest.approx(geo.WALL_PLANE_X_M)
    assert float(recentred[:, 0].min()) == pytest.approx(0.0)

    authored = _mesh(recenter=False).world_vertices
    assert float(authored[:, 0].min()) == pytest.approx(geo.WALL_PLANE_X_M)
    assert float(authored[:, 1].max()) == pytest.approx(geo.WALL_HALF_Y_M)


# ---------------------------------------------------------------------------
# The configuration surface
# ---------------------------------------------------------------------------


def test_components_and_max_depth_reach_the_propagation_block():
    from witwin.radar import RadarConfig
    from witwin.radar.radar import RadarSystemConfig

    flat = RadarConfig.from_dict(dict(geo.FIXTURE_RADAR_CONFIG))
    default = RadarSystemConfig.from_radar_config(flat)
    assert default.propagation.components == frozenset({"los", "reflection"})
    assert default.propagation.max_depth == 1

    narrowed = RadarSystemConfig.from_radar_config(flat, components=frozenset({"los"}), max_depth=0)
    assert narrowed.propagation.components == frozenset({"los"})
    assert narrowed.propagation.max_depth == 0
    assert narrowed.propagation.reference_frequency_hz == default.propagation.reference_frequency_hz


def test_with_propagation_overrides_one_solve_without_mutating_the_radar():
    from witwin.radar import RadarConfig
    from witwin.radar.radar import RadarSystemConfig

    flat = RadarConfig.from_dict(dict(geo.FIXTURE_RADAR_CONFIG))
    stored = RadarSystemConfig.from_radar_config(flat)
    overridden = stored.with_propagation(max_depth=2)
    assert overridden.propagation.max_depth == 2
    assert stored.propagation.max_depth == 1
    assert stored.with_propagation() is stored


def test_the_waveform_block_is_selectable_rather_than_inferred():
    from witwin.radar import RadarConfig
    from witwin.radar.radar import WAVEFORM_FMCW, WAVEFORM_OFDM, OfdmWaveformConfig, RadarSystemConfig

    flat = RadarConfig.from_dict(dict(geo.FIXTURE_RADAR_CONFIG))
    assert RadarSystemConfig.from_radar_config(flat).kind == WAVEFORM_FMCW

    ofdm = RadarSystemConfig.from_radar_config(
        flat,
        waveform=OfdmWaveformConfig(
            subcarrier_spacing_hz=120e3,
            num_subcarriers=64,
            cyclic_prefix_s=1.0e-6,
            num_symbols=8,
            max_expected_delay_s=5.0e-7,
        ),
    )
    assert ofdm.kind == WAVEFORM_OFDM
    spec = ofdm.waveform_spec()
    assert spec.num_subcarriers == 64
    assert spec.reference_frequency_hz == ofdm.propagation.reference_frequency_hz


# ---------------------------------------------------------------------------
# The compile facade and the full binding - CUDA and Channel
# ---------------------------------------------------------------------------


pytest.importorskip("witwin.channel")


@pytest.fixture
def radar():
    from witwin.radar import Radar

    return Radar(dict(geo.FIXTURE_RADAR_CONFIG))


@pytest.mark.gpu
def test_the_compile_facade_produces_the_scene_the_adapter_consumes():
    from witwin.radar.channel import ChannelPropagationAdapter, compile_scene

    scene, mesh = world.make_scene()
    world.assert_world_coordinates_survived(mesh)
    compiled = compile_scene(scene, reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ)
    adapter = ChannelPropagationAdapter(
        compiled, reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ, components=frozenset({"los"}), max_depth=0
    )
    assert adapter.compiled_scene is compiled


@pytest.mark.gpu
def test_a_reference_frequency_mismatch_is_refused_not_recompiled():
    """The refusal is Channel's exact-match contract, surfaced at the binding.

    Asserted by message rather than by exception type alone: a test that
    accepted any exception would pass for an import error or a typo in the call
    and would say nothing about the frequency at all.
    """

    from witwin.radar.channel import compile_scene, require_reference_frequency

    compiled = compile_scene(world.make_scene()[0], reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ)
    require_reference_frequency(compiled, geo.REFERENCE_FREQUENCY_HZ)
    with pytest.raises(ValueError, match="reference_frequency_hz does not exactly match"):
        require_reference_frequency(compiled, 24.0e9)


def test_require_reference_frequency_refuses_a_non_compiled_scene():
    from witwin.radar.channel import require_reference_frequency

    with pytest.raises(TypeError, match="require_reference_frequency"):
        require_reference_frequency(object(), geo.REFERENCE_FREQUENCY_HZ)


@pytest.mark.gpu
def test_the_binding_publishes_the_source_and_sink_power_contract(radar):
    sites = ScatterSitePolicy.explicit([geo.SITE_P_POSITION_M, geo.SITE_Q_POSITION_M])
    binding = bind_radar_world(radar, radar_snapshot(), sites=sites)

    assert isinstance(binding, RadarWorldBinding)
    assert binding.transmitters.powers_w is not None
    assert binding.site_sources.powers_w is not None
    assert binding.receivers.powers_w is None
    assert binding.site_sinks.powers_w is None

    expected_w = radar.system_config.sensors.tx_power.transmit_power_watts
    torch.testing.assert_close(
        binding.transmitters.powers_w,
        torch.full((binding.transmitters.count,), float(expected_w), dtype=torch.float32, device=binding.device),
    )
    torch.testing.assert_close(
        binding.site_sources.powers_w,
        torch.full((2,), SITE_EXCITATION_POWER_W, dtype=torch.float32, device=binding.device),
    )


@pytest.mark.gpu
def test_two_bindings_of_one_snapshot_agree_on_every_stable_id(radar):
    snapshot = radar_snapshot()
    sites = ScatterSitePolicy.explicit([geo.SITE_P_POSITION_M, geo.SITE_Q_POSITION_M])
    first = bind_radar_world(radar, snapshot, sites=sites)
    second = bind_radar_world(radar, snapshot, sites=sites)

    assert first.transmitter_ids == second.transmitter_ids
    assert first.receiver_ids == second.receiver_ids
    assert first.site_ids == second.site_ids
    assert torch.equal(first.transmitters.stable_ids, second.transmitters.stable_ids)
    assert torch.equal(first.site_sinks.stable_ids, second.site_sinks.stable_ids)

    every_id = set(first.transmitter_ids) | set(first.receiver_ids) | set(first.site_ids)
    assert len(every_id) == (len(first.transmitter_ids) + len(first.receiver_ids) + len(first.site_ids))


@pytest.mark.gpu
def test_the_two_site_roles_share_one_position_tensor(radar):
    positions = torch.tensor([geo.SITE_P_POSITION_M], dtype=torch.float32, device=radar.device).requires_grad_(True)
    binding = bind_radar_world(radar, radar_snapshot(), sites=ScatterSitePolicy.explicit(positions))
    assert binding.site_sources.positions_m is positions
    assert binding.site_sinks.positions_m is positions
    assert binding.site_positions_m is positions
    assert binding.transmitters.positions_m is radar.tx_pos
    assert binding.receivers.positions_m is radar.rx_pos


@pytest.mark.gpu
def test_the_binding_refuses_a_site_id_that_collides_with_the_array(radar):
    sites = ScatterSitePolicy.explicit([geo.SITE_P_POSITION_M], stable_ids=(DEFAULT_TRANSMITTER_ID_BASE,))
    with pytest.raises(ValueError, match="collide with the transmitter"):
        bind_radar_world(radar, radar_snapshot(), sites=sites)


@pytest.mark.gpu
def test_the_binding_refuses_a_site_tensor_from_another_device(radar):
    positions = torch.tensor([geo.SITE_P_POSITION_M], dtype=torch.float32)
    with pytest.raises(ValueError, match="but this binding is on"):
        bind_radar_world(radar, radar_snapshot(), sites=ScatterSitePolicy.explicit(positions))


def radar_snapshot():
    scene, mesh = world.make_scene()
    world.assert_world_coordinates_survived(mesh)
    return scene.snapshot(0.0)
