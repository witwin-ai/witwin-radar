"""The antenna pattern on the scene-driven route (Phase 11 work item 4).

Before this stage existed the ``sensor_weight`` family had exactly one importer
and that importer belonged to the legacy Dirichlet route. Deleting the route
without this would have left three registered ABI symbols with no production
caller, which acceptance criterion 6 forbids, and it would have removed the
capability rather than migrating it.

What this file pins:

* **The no-op is BITWISE.** With :data:`ISOTROPIC_PATTERN` the kernel's whole
  surviving factor is ``sqrt(1) * sqrt(1 * 1) == 1.0f``, so the published weight
  is the composed weight unchanged - not close to it. ``torch.equal`` is the
  assertion, because a tolerance here would hide exactly the drift the stage was
  introduced to avoid.
* **A directional pattern is the native family's answer, not a Torch one.** The
  ratio of the directional to the isotropic-baseline weight is compared against the
  independent Torch pattern oracle evaluated at the row's own local-frame
  directions, row by row, which also proves that the row-to-element and
  row-to-site tables are the right way round.
* **The derivative is carried.** Reverse gradients and forward tangents through
  the site positions and through the transmit element positions are checked
  against a central finite difference. The FD lives here and nowhere else: a
  production finite-difference derivative is forbidden.
* **The provenance is honest.** ``weight_includes_antenna_pattern`` is published
  by the stage, carried into ``SynthesisPathBatch``, and a second application is
  refused rather than counted.

The finite-difference tests deliberately choose a pattern whose knots are far
from the queried angles. A knot is a genuine non-differentiability where the
kernel returns the almost-everywhere derivative; an FD that straddled one would
disagree with the kernel AND with its Torch oracle, and that disagreement would
be correct behaviour rather than a defect.
"""

from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import multi_endpoint_world as world  # noqa: E402

from witwin.radar import Radar  # noqa: E402
from witwin.radar.simulation import ScatterSitePolicy  # noqa: E402
from witwin.radar.paths import RadarPathBatch, RadarPathTopology  # noqa: E402
from witwin.radar.scattering import ScalarRcsResponse  # noqa: E402
from witwin.radar.sensors import AntennaPatternSpec  # noqa: E402
from witwin.radar.sensors import (  # noqa: E402
    ISOTROPIC_PATTERN,
    RoundTripPatternStage,
)
from witwin.radar.synthesis import SlowTimeMode, SynthesisPathBatch  # noqa: E402


pytestmark = pytest.mark.gpu

LOOK_AT_M = (1.0, 0.0, 0.0)

SITE_POSITIONS_M = (geo.SITE_P_POSITION_M, geo.SITE_Q_POSITION_M)

#: A pattern that varies in BOTH angles, with its knots at -90, 0 and +90 while
#: every angle these fixtures query sits strictly inside one segment. That is
#: what makes a central finite difference a legitimate oracle here.
DIRECTIONAL_PATTERN = AntennaPatternSpec(
    kind="separable",
    x_angles_deg=(-90.0, 0.0, 90.0),
    x_values=(0.2, 1.0, 0.3),
    y_angles_deg=(-90.0, 0.0, 90.0),
    y_values=(0.4, 1.0, 0.5),
)

#: Off the ``z = 0`` plane on purpose: the multi-endpoint sites are coplanar
#: with the array, which puts the elevation query exactly on the pattern's
#: middle knot and hides every elevation derivative behind a zero.
UNIT_SITE_POSITIONS_M = ((2.0, 0.6, 0.35), (1.7, -0.9, -0.5))


def _pattern_config(pattern: AntennaPatternSpec) -> dict:
    config = {
        "kind": pattern.kind,
        "x_angles_deg": list(pattern.x_angles_deg),
        "y_angles_deg": list(pattern.y_angles_deg),
    }
    if pattern.kind == "separable":
        config["x_values"] = list(pattern.x_values)
        config["y_values"] = list(pattern.y_values)
    else:
        config["values"] = [list(row) for row in pattern.values]
    return config


def _radar(pattern: AntennaPatternSpec = ISOTROPIC_PATTERN) -> Radar:
    config = dict(geo.FIXTURE_RADAR_CONFIG)
    config["antenna_pattern"] = _pattern_config(pattern)
    return Radar(
        config,
        position=(0.0, 0.0, 0.0),
        target=LOOK_AT_M,
    )


def _response(radar: Radar) -> ScalarRcsResponse:
    return ScalarRcsResponse.from_values(
        drv.FIXTURE_AMPLITUDE, drv.FIXTURE_PHASE_RAD, device=radar.device
    )


def _sites(radar: Radar, *, requires_grad: bool = False):
    positions = torch.tensor(
        SITE_POSITIONS_M, dtype=torch.float32, device=radar.device
    ).requires_grad_(requires_grad)
    return ScatterSitePolicy.explicit(positions), positions


def _static_scene():
    scene, mesh = world.make_scene()
    world.assert_world_coordinates_survived(mesh)
    return scene


def _simulate(radar: Radar, *, times=(0.0,), **options):
    policy, _ = _sites(radar)
    return radar.simulate(
        _static_scene(),
        times=times,
        response=_response(radar),
        sites=policy,
        **options,
    )


# ---------------------------------------------------------------------------
# 1. The no-op, asserted bitwise
# ---------------------------------------------------------------------------


def test_the_stored_isotropic_pattern_is_applied_to_every_simulation():
    result = _simulate(_radar(ISOTROPIC_PATTERN))
    assert result.last_radar_paths.weight_includes_antenna_pattern is True


def test_the_stored_default_dipole_is_applied_to_every_simulation():
    result = _simulate(_radar(AntennaPatternSpec.half_wave_dipole()))
    assert result.last_radar_paths.weight_includes_antenna_pattern is True

# ---------------------------------------------------------------------------
# 2. A directional pattern, against the independent Torch oracle
# ---------------------------------------------------------------------------


def _pattern_gain_from_vectors(pattern: AntennaPatternSpec, vectors: torch.Tensor) -> torch.Tensor:
    forward = -vectors[..., 2]
    x_angles_deg = torch.rad2deg(torch.atan2(vectors[..., 0], forward))
    y_angles_deg = torch.rad2deg(torch.atan2(vectors[..., 1], forward))
    return pattern.evaluate_xy(x_angles_deg, y_angles_deg)

def _oracle_amplitude(
    radar: Radar, paths: RadarPathBatch, sites: torch.Tensor, pattern
) -> torch.Tensor:
    """``sqrt(G_t * G_r)`` per composed row, from the Torch pattern evaluator.

    Independent of the kernel in the way that matters: it resolves the row's
    transmitter, receiver and site from the PUBLISHED topology and pair rank
    rather than from the stage's own tables, so a stage that paired the wrong
    element with the wrong site would disagree here even though its own
    bookkeeping was self-consistent.
    """

    num_tx = radar.system_config.sensors.array.num_tx
    pair = paths.sensor_pair_index
    tx_index = torch.remainder(pair, num_tx)
    rx_index = torch.div(pair, num_tx, rounding_mode="floor")
    site_ids = sorted(int(value) for value in set(paths.topology.site_id.tolist()))
    lookup = {value: rank for rank, value in enumerate(site_ids)}
    site_row = torch.tensor(
        [lookup[int(value)] for value in paths.topology.site_id.tolist()],
        dtype=torch.int64,
        device=paths.topology.site_id.device,
    )
    site = sites.index_select(0, site_row)
    tx = radar.tx_pos.index_select(0, tx_index)
    rx = radar.rx_pos.index_select(0, rx_index)
    gain_tx = _pattern_gain_from_vectors(pattern, radar._local_from_world_vectors(site - tx))
    gain_rx = _pattern_gain_from_vectors(pattern, radar._local_from_world_vectors(site - rx))
    return (gain_tx * gain_rx).clamp_min(0.0).sqrt()


def test_a_directional_pattern_scales_each_row_by_its_own_gain():
    baseline_radar = _radar(ISOTROPIC_PATTERN)
    plain_rows = _simulate(baseline_radar).last_radar_paths

    radar = _radar(DIRECTIONAL_PATTERN)
    rows = _simulate(radar).last_radar_paths

    sites = torch.tensor(
        SITE_POSITIONS_M, dtype=torch.float32, device=radar.device
    )
    expected = _oracle_amplitude(radar, rows, sites, DIRECTIONAL_PATTERN)
    assert float(expected.min()) > 0.0
    assert float(expected.max()) < 1.0

    torch.testing.assert_close(
        rows.complex_transfer_ref,
        plain_rows.complex_transfer_ref * expected.to(torch.complex64),
        rtol=2e-6,
        atol=0.0,
    )


def test_the_two_sites_are_attenuated_differently():
    """The stage would pass every check above with one gain for all rows.

    The two fixture sites sit at very different bearings, so a per-row lookup
    must produce two distinct gains. A single shared gain is exactly what a
    stage that fumbled its site table would produce.
    """

    plain = _simulate(_radar(ISOTROPIC_PATTERN)).last_radar_paths
    rows = _simulate(_radar(DIRECTIONAL_PATTERN)).last_radar_paths

    ratio = rows.complex_transfer_ref.abs() / plain.complex_transfer_ref.abs()
    by_site: dict[int, set[float]] = {}
    for value, site in zip(
        ratio.tolist(), rows.topology.site_id.tolist(), strict=True
    ):
        by_site.setdefault(int(site), set()).add(round(float(value), 6))
    assert len(by_site) == 2, sorted(by_site)
    first, second = (sorted(values)[0] for values in by_site.values())
    assert abs(first - second) > 1e-3, (first, second)


def test_the_cube_changes_when_a_pattern_is_applied():
    """The E2E statement: the gain reaches the published frame cube."""

    plain = _simulate(_radar(ISOTROPIC_PATTERN))
    patterned = _simulate(_radar(DIRECTIONAL_PATTERN))

    assert plain.cube.shape == patterned.cube.shape
    assert not torch.equal(plain.cube, patterned.cube)
    assert float(patterned.cube.abs().sum()) < float(plain.cube.abs().sum())


# ---------------------------------------------------------------------------
# 3. Provenance
# ---------------------------------------------------------------------------


def test_the_stage_publishes_its_provenance_and_synthesis_carries_it():
    radar = _radar(DIRECTIONAL_PATTERN)
    rows = _simulate(radar).last_radar_paths

    assert rows.weight_includes_antenna_pattern is True
    batch = SynthesisPathBatch.from_radar_paths(
        rows, slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    )
    assert batch.weight_includes_antenna_pattern is True
    # The other three stay Channel's published contract.
    assert batch.weight_includes_reference_phase is True
    assert batch.weight_includes_spreading is True
    assert batch.weight_includes_tx_power is True


def test_the_stored_pattern_provenance_reaches_synthesis():
    radar = _radar(ISOTROPIC_PATTERN)
    rows = _simulate(radar).last_radar_paths
    batch = SynthesisPathBatch.from_radar_paths(
        rows, slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
    )
    assert batch.weight_includes_antenna_pattern is True


# ---------------------------------------------------------------------------
# 4. The direct contract: refusals and the AD checks
# ---------------------------------------------------------------------------


def _unit_stage(radar: Radar, pattern) -> tuple[RoundTripPatternStage, RadarPathBatch]:
    """A two-site, four-pair frozen stage plus a batch of unit weights.

    The composer is duck-typed here rather than frozen from two real legs: the
    stage reads four things off a join - its pair index, its pair count, its
    site count and its response slot - and building a Channel round trip to
    supply four small integer tensors would make this an end-to-end test with
    an end-to-end test's failure modes. The end-to-end statement is above.
    """

    device = radar.device
    array = radar.system_config.sensors.array
    pairs = array.num_tx * array.num_rx
    sites = len(UNIT_SITE_POSITIONS_M)
    rows = pairs * sites
    pair_index = torch.arange(pairs, device=device, dtype=torch.int64).repeat_interleave(
        sites
    )
    response_slot = torch.arange(sites, device=device, dtype=torch.int64).repeat(pairs)
    join = types.SimpleNamespace(
        sensor_pair_index=pair_index,
        sensor_pair_count=pairs,
        site_count=sites,
        path_count=rows,
        response_slot=response_slot,
    )
    stage = RoundTripPatternStage.freeze(
        radar,
        join,
        site_ids=tuple(3_000_000 + index for index in range(sites)),
        pattern=pattern,
    )
    weight = torch.complex(
        torch.full((rows,), 0.75, dtype=torch.float32, device=device),
        torch.full((rows,), -0.25, dtype=torch.float32, device=device),
    )
    zeros = torch.zeros(rows, dtype=torch.int64, device=device)
    batch = RadarPathBatch(
        sensor_pair_count=pairs,
        path_count=rows,
        sensor_pair_index=pair_index,
        pair_offsets=torch.arange(
            0, rows + 1, sites, device=device, dtype=torch.int64
        ),
        total_delay_s=torch.zeros(rows, dtype=torch.float32, device=device),
        delay_rate=None,
        complex_transfer_ref=weight,
        reference_frequency_hz=float(geo.REFERENCE_FREQUENCY_HZ),
        row_valid=None,
        topology=RadarPathTopology(
            radar_source_id=zeros,
            site_id=response_slot + 3_000_000,
            radar_sink_id=zeros,
            inbound_row=zeros,
            outbound_row=zeros,
        ),
        join_mode="multipath",
    )
    return stage, batch


def _unit_sites(radar: Radar, *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(
        UNIT_SITE_POSITIONS_M, dtype=torch.float32, device=radar.device
    ).requires_grad_(requires_grad)


def _unit_loss(stage, batch, sites, tx_pos, rx_pos) -> torch.Tensor:
    weight = stage.apply(
        batch, tx_pos=tx_pos, rx_pos=rx_pos, site_positions_m=sites
    ).complex_transfer_ref
    # A squared magnitude, so the loss is real and smooth in the gain and has
    # no phase in it at all: the stage multiplies by a REAL scale.
    return weight.real.square().sum() + weight.imag.square().sum()


def test_applying_the_stage_twice_is_refused():
    radar = _radar()
    stage, batch = _unit_stage(radar, DIRECTIONAL_PATTERN)
    once = stage.apply(
        batch,
        tx_pos=radar.tx_pos,
        rx_pos=radar.rx_pos,
        site_positions_m=_unit_sites(radar),
    )
    with pytest.raises(ValueError, match="weight_includes_antenna_pattern"):
        stage.apply(
            once,
            tx_pos=radar.tx_pos,
            rx_pos=radar.rx_pos,
            site_positions_m=_unit_sites(radar),
        )


def test_a_direct_join_is_refused_by_name():
    radar = _radar()
    stage, batch = _unit_stage(radar, DIRECTIONAL_PATTERN)
    import dataclasses

    direct = dataclasses.replace(batch, join_mode="direct")
    with pytest.raises(NotImplementedError, match="join_mode"):
        stage.apply(
            direct,
            tx_pos=radar.tx_pos,
            rx_pos=radar.rx_pos,
            site_positions_m=_unit_sites(radar),
        )


def test_a_batch_from_another_topology_is_refused():
    radar = _radar()
    stage, batch = _unit_stage(radar, DIRECTIONAL_PATTERN)
    import dataclasses

    shrunk = dataclasses.replace(
        batch,
        path_count=batch.path_count - 1,
        sensor_pair_index=batch.sensor_pair_index[:-1],
        total_delay_s=batch.total_delay_s[:-1],
        complex_transfer_ref=batch.complex_transfer_ref[:-1],
        pair_offsets=batch.pair_offsets.clone().index_put_(
            (torch.tensor([-1], device=batch.pair_offsets.device),),
            batch.pair_offsets[-1:] - 1,
        ),
        topology=RadarPathTopology(
            radar_source_id=batch.topology.radar_source_id[:-1],
            site_id=batch.topology.site_id[:-1],
            radar_sink_id=batch.topology.radar_sink_id[:-1],
            inbound_row=batch.topology.inbound_row[:-1],
            outbound_row=batch.topology.outbound_row[:-1],
        ),
    )
    with pytest.raises(ValueError, match="frozen topology"):
        stage.apply(
            shrunk,
            tx_pos=radar.tx_pos,
            rx_pos=radar.rx_pos,
            site_positions_m=_unit_sites(radar),
        )


def test_a_pattern_that_is_not_a_spec_is_refused():
    radar = _radar()
    stage, _ = _unit_stage(radar, DIRECTIONAL_PATTERN)
    join = types.SimpleNamespace(
        sensor_pair_index=stage.tx_index,
        sensor_pair_count=stage.num_tx * stage.num_rx,
        site_count=stage.site_count,
        path_count=stage.row_count,
        response_slot=stage.site_slot,
    )
    with pytest.raises(TypeError, match="AntennaPatternSpec"):
        RoundTripPatternStage.freeze(
            radar,
            join,
            site_ids=(3_000_000, 3_000_001),
            pattern={"kind": "separable"},
        )


def test_the_isotropic_stage_is_the_identity_on_a_unit_batch():
    radar = _radar()
    stage, batch = _unit_stage(radar, ISOTROPIC_PATTERN)
    published = stage.apply(
        batch,
        tx_pos=radar.tx_pos,
        rx_pos=radar.rx_pos,
        site_positions_m=_unit_sites(radar),
    )
    assert torch.equal(
        published.complex_transfer_ref, batch.complex_transfer_ref
    )


def test_the_reverse_gradient_of_a_site_matches_a_central_difference():
    radar = _radar()
    stage, batch = _unit_stage(radar, DIRECTIONAL_PATTERN)
    sites = _unit_sites(radar, requires_grad=True)
    _unit_loss(stage, batch, sites, radar.tx_pos, radar.rx_pos).backward()
    measured = sites.grad.detach().clone()

    step = 2.0e-3
    expected = torch.zeros_like(measured)
    for row in range(measured.shape[0]):
        for axis in range(3):
            values = []
            for sign in (+1.0, -1.0):
                shifted = _unit_sites(radar)
                shifted[row, axis] += sign * step
                values.append(
                    float(_unit_loss(stage, batch, shifted, radar.tx_pos, radar.rx_pos))
                )
            expected[row, axis] = (values[0] - values[1]) / (2.0 * step)

    assert float(measured.abs().max()) > 1e-3
    torch.testing.assert_close(measured, expected, rtol=2e-3, atol=1e-4)


def test_the_reverse_gradient_of_a_transmit_element_matches_a_central_difference():
    """The antenna-position gradient is the family's ordered reduction.

    Many rows share one transmitter, so this cell is a real sum rather than an
    elementwise map, and it is the one the kernel does in a second pass with an
    explicit scratch buffer instead of with atomics.
    """

    radar = _radar()
    stage, batch = _unit_stage(radar, DIRECTIONAL_PATTERN)
    sites = _unit_sites(radar)
    tx_pos = radar.tx_pos.detach().clone().requires_grad_(True)
    _unit_loss(stage, batch, sites, tx_pos, radar.rx_pos).backward()
    measured = tx_pos.grad.detach().clone()

    step = 2.0e-3
    expected = torch.zeros_like(measured)
    for row in range(measured.shape[0]):
        for axis in range(3):
            values = []
            for sign in (+1.0, -1.0):
                shifted = radar.tx_pos.detach().clone()
                shifted[row, axis] += sign * step
                values.append(
                    float(_unit_loss(stage, batch, sites, shifted, radar.rx_pos))
                )
            expected[row, axis] = (values[0] - values[1]) / (2.0 * step)

    assert float(measured.abs().max()) > 1e-3
    torch.testing.assert_close(measured, expected, rtol=2e-3, atol=1e-4)


def test_the_forward_tangent_of_a_site_matches_a_central_difference():
    import torch.autograd.forward_ad as forward_ad

    radar = _radar()
    stage, batch = _unit_stage(radar, DIRECTIONAL_PATTERN)
    direction = torch.tensor(
        [[0.3, -0.7, 0.5], [-0.4, 0.2, 0.9]],
        dtype=torch.float32,
        device=radar.device,
    )
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(_unit_sites(radar), direction)
        loss = _unit_loss(stage, batch, dual, radar.tx_pos, radar.rx_pos)
        tangent = forward_ad.unpack_dual(loss).tangent
        assert tangent is not None, "the stage dropped the forward tangent"
        measured = float(tangent)

    step = 2.0e-3
    values = []
    for sign in (+1.0, -1.0):
        shifted = _unit_sites(radar) + sign * step * direction
        values.append(
            float(_unit_loss(stage, batch, shifted, radar.tx_pos, radar.rx_pos))
        )
    expected = (values[0] - values[1]) / (2.0 * step)

    assert abs(measured) > 1e-3
    assert measured == pytest.approx(expected, rel=2e-3, abs=1e-4)


def test_a_reverse_gradient_reaches_the_frame_cube_through_the_pattern():
    """The end-to-end liveness statement, deliberately not an FD.

    A finite difference of the whole pipeline against a site position measures
    the round-trip DELAY's derivative, which at 77 GHz turns over within a
    micrometre and drowns the pattern's contribution in phase wrap. What is
    checkable end to end is that the pattern changes the gradient at all, which
    a stage that detached its weight would fail.
    """

    radar = _radar(DIRECTIONAL_PATTERN)
    policy, sites = _sites(radar, requires_grad=True)
    result = radar.simulate(
        _static_scene(),
        times=(0.0,),
        response=_response(radar),
        sites=policy,
        ad_mode="vjp",
    )
    result.cube.real.square().sum().backward()

    assert sites.grad is not None
    assert torch.isfinite(sites.grad).all()
    assert float(sites.grad.abs().max()) > 0.0
