"""When a frozen topology stops being an answer (plan items 2 and 7).

Before Phase 7 the adapter bound one compiled scene at construction and held it
for its whole lifetime. A moving wall therefore produced a new ``CompiledScene``
that nothing consumed, and the frozen rows kept replaying against geometry that
had moved on - silently, at full strength, with a plausible delay.

Three things close that, and all three are tested here:

* ``rediscovery_required`` - the cheap per-frame poll that names the world
  version domain that moved;
* ``refreeze`` - the supported rebind, which retires every frozen handle;
* the refusal - a handle from a retired epoch, or a topology whose world moved,
  raises by name instead of answering.

The other half of "what a freeze is allowed to assume" - the pair-rank layout
gate - lives in ``test_phase7_pair_ordering.py``.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import multi_endpoint_world as world  # noqa: E402

pytestmark = pytest.mark.gpu

WALL_SPEED_M_PER_S = 0.5


def _moving_wall_scene():
    """One scene, one wall, one constant-velocity trajectory for it.

    A ``DynamicScene`` over ONE ``Scene`` is what isolates the geometry domain:
    two independently built scenes differ in ``topology_version`` too, because
    that version folds tensor identity, and the refusal would then name a
    different domain than the one this test is about.
    """

    from witwin.core.dynamics import DynamicScene, LinearTrajectory

    scene, mesh = world.make_scene()
    world.assert_world_coordinates_survived(mesh)
    return DynamicScene(
        scene,
        structure_trajectories={
            1: LinearTrajectory(origin=torch.zeros(3), velocity=torch.tensor([WALL_SPEED_M_PER_S, 0.0, 0.0]))
        },
    )


def _compile(snapshot):
    from witwin.channel.scene import compile as compile_scene

    return compile_scene(snapshot, reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ)


def test_rediscovery_is_not_required_while_the_world_holds_still():
    spike = drv.MultiEndpointSpike()
    assert spike.adapter.rediscovery_required(spike.inbound) is None
    assert spike.adapter.rediscovery_required(spike.outbound) is None
    assert spike.adapter.epoch == 0


def test_rediscovery_required_costs_no_host_observation(monkeypatch):
    """It is polled per frame, so it has to be free.

    Four host integer comparisons and nothing else: no ``.item()``, no
    ``.cpu()``, no synchronization. A poll that cost a device-to-host copy
    would put back exactly the traffic the fixed-topology contract removed.
    """

    spike = drv.MultiEndpointSpike()
    spike.adapter.rediscovery_required(spike.inbound)  # resolve any lazy import

    counts = dict.fromkeys(("item", "cpu", "tolist", "numpy", "synchronize"), 0)
    for name in ("item", "cpu", "tolist", "numpy"):
        original = getattr(torch.Tensor, name)

        def observing(tensor, *args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, observing)
    original_sync = torch.cuda.synchronize

    def counting_sync(*args, **kwargs):
        counts["synchronize"] += 1
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", counting_sync)

    spike.adapter.rediscovery_required(spike.inbound)
    spike.adapter.rediscovery_required(spike.outbound)
    assert counts == dict.fromkeys(counts, 0), counts


def test_refreeze_is_required_after_a_structure_moves():
    """The moving-wall cadence, end to end.

    Poll, refuse, rebind, rediscover, replay. The refusal has to name
    ``geometry_version`` because that is the domain that moved and it is what a
    caller keys its policy on.
    """

    dynamic = _moving_wall_scene()
    early = _compile(dynamic.at(0.0))
    late = _compile(dynamic.at(1.0))
    assert early.topology_version == late.topology_version
    assert early.geometry_version != late.geometry_version
    assert early.time_s == 0.0 and late.time_s == 1.0

    spike = drv.MultiEndpointSpike(compiled=early)
    sites = spike.site_tensor()
    before = spike.adapter.reevaluate(
        spike.inbound,
        spike._stacked_ids(
            spike.stacked([position for _, position in spike.transmitters], 1), spike.transmitter_ids, geo.TX_POWER_W
        ),
        spike._stacked_ids(sites, spike.site_ids, None),
        ad_mode="none",
    )
    assert before.leg_count == spike.inbound.row_count

    # 1. The poll fires, and it names the domain.
    spike.adapter.refreeze(late)
    assert spike.adapter.epoch == 1
    assert spike.adapter.compiled_scene is late
    assert spike.adapter.rediscovery_required(spike.inbound) == "geometry_version"

    # 2. The stale handle is refused rather than answered, and the message
    #    carries the same domain name the poll returned.
    with pytest.raises(ValueError, match="geometry_version"):
        spike.adapter.reevaluate(
            spike.inbound,
            spike._stacked_ids(
                spike.stacked([position for _, position in spike.transmitters], 1),
                spike.transmitter_ids,
                geo.TX_POWER_W,
            ),
            spike._stacked_ids(sites, spike.site_ids, None),
            ad_mode="none",
        )

    # 3. Rediscovery through the rebound adapter produces a working replay, and
    #    the wall really did move: the reflection rows change while the
    #    line-of-sight rows do not.
    rebuilt = drv.MultiEndpointSpike(compiled=late)
    assert rebuilt.adapter.rediscovery_required(rebuilt.inbound) is None
    after = rebuilt.adapter.reevaluate(
        rebuilt.inbound,
        rebuilt._stacked_ids(
            rebuilt.stacked([position for _, position in rebuilt.transmitters], 1),
            rebuilt.transmitter_ids,
            geo.TX_POWER_W,
        ),
        rebuilt._stacked_ids(sites, rebuilt.site_ids, None),
        ad_mode="none",
    )
    assert after.leg_count == before.leg_count
    los = before.component_id == geo.LOS_COMPONENT_ID
    reflection = before.component_id == geo.REFLECTION_COMPONENT_ID
    assert bool(los.any()) and bool(reflection.any())
    assert torch.equal(before.delay_s[los], after.delay_s[los])
    assert not torch.equal(before.delay_s[reflection], after.delay_s[reflection])


def test_a_retired_handle_is_refused_even_when_no_version_moved():
    """The staleness class the world versions cannot see.

    A compiled scene and the rows discovered on it always agree with each
    other, so recompiling an UNCHANGED world produces a handle that Channel
    cannot call stale. The adapter still refuses it: the caller rebound on
    purpose, and answering out of the scene it replaced is exactly the stale
    answer this cadence exists to prevent.
    """

    dynamic = _moving_wall_scene()
    spike = drv.MultiEndpointSpike(compiled=_compile(dynamic.at(0.0)))
    # A DIFFERENT compiled scene of the SAME world instant: every version
    # domain is bit identical, so Channel's provenance check sees nothing.
    twin = _compile(dynamic.at(0.0))
    assert twin is not spike.adapter.compiled_scene
    assert twin.geometry_version == spike.adapter.compiled_scene.geometry_version
    spike.adapter.refreeze(twin)
    assert spike.adapter.rediscovery_required(spike.inbound) is None
    with pytest.raises(ValueError, match="refreeze"):
        spike.adapter.reevaluate(
            spike.inbound,
            spike._stacked_ids(
                spike.stacked([position for _, position in spike.transmitters], 1),
                spike.transmitter_ids,
                geo.TX_POWER_W,
            ),
            spike._stacked_ids(spike.site_tensor(), spike.site_ids, None),
            ad_mode="none",
        )


def test_refreeze_refuses_a_missing_scene():
    spike = drv.MultiEndpointSpike()
    with pytest.raises(ValueError, match="requires a compiled scene"):
        spike.adapter.refreeze(None)
