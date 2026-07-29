"""Static and dynamic clutter: what each mobility declaration costs and loses.

Plan Phase-8 work item 2, the half that is about TIME. A clutter component is
not only a set of rows; it is a claim about whether that set can change, and
the three claims a caller can make - ``static``, ``replay``, ``rediscover`` -
have different prices and different failure modes. This file drives all three
through the production :class:`SceneEpochLoop` and measures them.

The most important test here is the one that DEMONSTRATES A LIMITATION rather
than asserting it away. A fixed-winner replay is subtractive by construction: a
frozen row that stops existing publishes ``row_valid=False`` with an exactly
zero payload, which is a complete answer, but a row that STARTS existing is
simply absent and nothing on the device can report its absence. The
path-gaining scene shows the born clutter row missing under ``replay`` and
present after the declared rediscovery cadence, and asserts both halves. That
is honest evidence of a designed boundary, not a bug report.

``mobility`` is a caller DECLARATION and is never inferred. Whether a moving
wall can gain a path is a question about the world, and no replay can answer
it from inside itself.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import clutter_scenes as cs  # noqa: E402
from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import multi_endpoint_world as world  # noqa: E402

from witwin.radar.paths import ENVIRONMENT_CLUTTER, TARGET  # noqa: E402
from witwin.radar.propagation import FIRST_FRAME, MOTION_EVENT_CADENCE, ClutterComponentSpec, epoch_policy  # noqa: E402

pytestmark = pytest.mark.gpu

#: The consumer's freezable component set, quoted rather than copied. Every
#: mobility refusal below is measured against THIS, so a Channel that widens or
#: narrows its wideband cell moves the refusal with it.
FREEZABLE = frozenset({"los", "reflection"})


def _freezable():
    from witwin.channel.propagation import consumer

    return consumer.capabilities().fixed_topology_components


def _policy(*specs):
    return epoch_policy(specs, fixed_topology_components=_freezable())


def _keys(state):
    composed, _, _ = state.spike.frame()
    return composed, drv.composed_keys(state.spike, composed)


# --------------------------------------------------------------------------
# The declaration resolves to a loop configuration
# --------------------------------------------------------------------------


def test_the_consumer_still_freezes_exactly_los_and_reflection():
    """The constant this file measures refusals against is the live one."""

    assert _freezable() == FREEZABLE


def test_each_mobility_mix_resolves_to_one_loop_configuration():
    """Four cases, and the mixed one is the interesting one.

    There is ONE compiled scene and ONE epoch loop per session, so a
    declaration naming several components has to resolve to a single
    ``world_motion`` and a single cadence. Replay and rediscover together
    resolve to a replayed loop ON the shortest declared cadence, which
    downgrades neither: frames between the ticks replay, and the tick pays the
    discovery.
    """

    static = ClutterComponentSpec("wall", "static")
    replay = ClutterComponentSpec("vehicle", "replay")
    rediscover = ClutterComponentSpec("foliage", "rediscover", rediscovery_period_frames=4)
    faster = ClutterComponentSpec("rotor", "rediscover", rediscovery_period_frames=2)

    assert _policy(static) == _policy(static)
    for specs, expected in (
        ((static,), ("frozen_world", None)),
        ((static, replay), ("fixed_winner_replay", None)),
        ((static, rediscover), ("frozen_world", 4)),
        ((replay, rediscover, faster), ("fixed_winner_replay", 2)),
    ):
        policy = _policy(*specs)
        assert (policy.world_motion, policy.motion_event_period_frames) == expected


def test_a_non_freezable_component_may_not_declare_a_replay():
    """Diffraction and transmission clutter must rediscover, by capability.

    The refusal quotes the consumer's own ``fixed_topology_components`` rather
    than a local copy, and it fires for both non-rediscovering mobilities: a
    component that cannot be frozen cannot ride the fixed-topology inner loop
    at all, whether the caller thinks it is standing still or moving.
    """

    for mobility in ("static", "replay"):
        spec = ClutterComponentSpec("edge", mobility, components=frozenset({"diffraction"}))
        with pytest.raises(NotImplementedError) as excinfo:
            _policy(spec)
        message = str(excinfo.value)
        assert "diffraction" in message
        assert "los" in message and "reflection" in message
        assert "rediscover" in message

    # The same component IS expressible with a declared cadence.
    allowed = _policy(
        ClutterComponentSpec("edge", "rediscover", components=frozenset({"diffraction"}), rediscovery_period_frames=3)
    )
    assert allowed.motion_event_period_frames == 3


def test_a_cadence_belongs_to_a_rediscovering_component_and_to_nothing_else():
    """Dead configuration later reads as a promise, so it is refused."""

    with pytest.raises(ValueError, match="never fire"):
        ClutterComponentSpec("wall", "static", rediscovery_period_frames=4)
    with pytest.raises(ValueError, match="positive int"):
        ClutterComponentSpec("foliage", "rediscover")
    with pytest.raises(ValueError, match="mobility must be one of"):
        ClutterComponentSpec("wall", "frozen")
    with pytest.raises(ValueError, match="declared twice"):
        _policy(ClutterComponentSpec("wall", "static"), ClutterComponentSpec("wall", "replay"))


# --------------------------------------------------------------------------
# Static clutter costs nothing
# --------------------------------------------------------------------------


def test_static_clutter_never_recompiles_and_never_rediscovers():
    """Exact integers over eight frames, against the loop's own counters.

    A wall with no trajectory and no deformation is the zero-cost case by
    construction: the compiled scene is built once, discovery runs once, and
    every later frame is one batched replay per leg. The per-frame poll still
    runs, which is what makes "never rediscovers" a measurement rather than an
    absence of instrumentation.
    """

    compiles = []

    def counting_compile(snapshot, *, reference_frequency_hz):
        compiles.append(float(snapshot.time_s))
        return world.compile_snapshot(snapshot, reference_frequency_hz=reference_frequency_hz)

    policy = _policy(ClutterComponentSpec("wall", "static"), ClutterComponentSpec("sites", "static"))
    loop, state = cs.clutter_epoch_loop(cs.static_clutter_scene(), policy, compile_scene=counting_compile)
    assert loop.structures_move is False
    frames = [loop.frame(index * 1.0e-3) for index in range(8)]

    assert loop.compile_count == 1, compiles
    assert loop.discovery_count == 1
    assert state.binds == 1
    assert frames[0].reason == FIRST_FRAME
    assert all(not frame.rediscovered for frame in frames[1:])
    assert all(not frame.recompiled for frame in frames[1:])
    assert loop.poll_count == 2 * (len(frames) - 1)
    assert loop.revalidation_count == 0

    # The clutter really is there and really is static: the same partition
    # every frame, and the same rows.
    assert state.index.count(ENVIRONMENT_CLUTTER) == 7
    assert state.index.count(TARGET) == 4
    first, _, _ = state.spike.frame()
    last, _, _ = state.spike.frame()
    assert torch.equal(first.complex_transfer_ref, last.complex_transfer_ref)


# --------------------------------------------------------------------------
# Replayed clutter
# --------------------------------------------------------------------------


def test_replayed_clutter_evolves_without_a_single_rediscovery():
    """A moving wall, replayed: the clutter rows move and the target rows do not.

    This is the whole value of ``mobility="replay"``. The loop pays a compile
    per frame because the structure declared a trajectory, but it pays NO
    discovery: the frozen winner set is held fixed by declaration. The physics
    check is that only the rows that touch the wall change, which is a
    statement no row count can make.
    """

    policy = _policy(ClutterComponentSpec("wall", "replay"))
    assert policy.world_motion == "fixed_winner_replay"
    assert policy.motion_event_period_frames is None

    loop, state = cs.clutter_epoch_loop(cs.replayed_clutter_scene(), policy)
    assert loop.structures_move is True

    delays = []
    for index in range(4):
        loop.frame(index * 1.0e-3)
        composed, _, _ = state.spike.frame()
        delays.append(composed.total_delay_s.double().clone())

    assert loop.compile_count == 4
    assert loop.discovery_count == 1
    assert state.binds == 1

    clutter = state.index.mask(ENVIRONMENT_CLUTTER)
    target = state.index.mask(TARGET)
    assert int(clutter.sum()) == 7 and int(target.sum()) == 4
    # A line of sight does not touch the wall, so its delay is not merely
    # stable, it is the same float.
    assert torch.equal(delays[0][target], delays[-1][target])
    moved = (delays[-1][clutter] - delays[0][clutter]).abs()
    assert float(moved.min()) > 0.0
    print(f"\nreplayed clutter delay drift over 3 ms: {float(moved.min()):.3e} to {float(moved.max()):.3e} s")


def test_a_dying_clutter_row_is_a_complete_answer_and_keeps_its_class():
    """Validity and class are orthogonal, and the payload is exactly zero.

    A frozen clutter row whose specular point leaves the facet publishes
    ``row_valid=False`` with an exactly zero payload. That is data, not an
    error - and the component index still classifies it, because the class is a
    property of the frozen identity and not of whether the row happens to exist
    at this instant. Reading the mask as a liveness signal would be the same
    mistake as reading a zero payload as one.
    """

    policy = _policy(ClutterComponentSpec("wall", "replay"))
    loop, state = cs.clutter_epoch_loop(cs.path_gaining_clutter_scene(), policy)
    loop.frame(0.0)
    parked, _, _ = state.spike.frame()
    assert bool(parked.row_valid.all())
    assert state.index.count(ENVIRONMENT_CLUTTER) == 5
    assert state.index.count(TARGET) == 8

    loop.frame(1.0)
    replayed, _, _ = state.spike.frame()
    clutter = state.index.mask(ENVIRONMENT_CLUTTER)
    dead = ~replayed.row_valid

    # Every clutter row died when the wall arrived over the specular points.
    assert bool((clutter & dead).any())
    payload = replayed.complex_transfer_ref[clutter & dead]
    assert torch.equal(payload, torch.zeros_like(payload))
    assert torch.equal(replayed.total_delay_s[clutter & dead], torch.zeros_like(replayed.total_delay_s[clutter & dead]))
    # The class survives the death: the mask is unchanged and still names those
    # rows clutter.
    assert torch.equal(clutter, state.index.mask(ENVIRONMENT_CLUTTER))
    assert state.index.count(ENVIRONMENT_CLUTTER) == 5
    # Non-vacuity: something is still alive, so this is not a scene that died.
    assert bool(replayed.row_valid.any())


# --------------------------------------------------------------------------
# The subtractive boundary, demonstrated
# --------------------------------------------------------------------------


def test_replay_cannot_gain_a_clutter_row_and_a_cadence_recovers_it():
    """Both halves, explicitly. This is the designed limitation of replay.

    A wall parked clear of the geometry arrives over one second and CREATES the
    ``TX_A -> SITE_P`` reflection round trips. Under ``mobility="replay"`` the
    frozen topology cannot express them: the composed row count does not
    change, no signal reports the absence, and the clutter class stays the size
    it was frozen at. Declaring a rediscovery cadence alongside the replay is
    the mitigation, and after the tick the born rows are there.
    """

    replay_only = _policy(ClutterComponentSpec("wall", "replay"))
    loop, state = cs.clutter_epoch_loop(cs.path_gaining_clutter_scene(), replay_only)
    loop.frame(0.0)
    parked, parked_keys = _keys(state)
    assert parked.path_count == 13

    frame = loop.frame(1.0)
    assert not frame.rediscovered
    replayed, replayed_keys = _keys(state)
    assert replayed.path_count == 13
    assert replayed_keys == parked_keys
    alive = {key for key, valid in zip(replayed_keys, replayed.row_valid.tolist(), strict=True) if valid}

    # The same world, discovered fresh, has rows the replay does not have.
    mixed = _policy(
        ClutterComponentSpec("wall", "replay"),
        ClutterComponentSpec("arriving_wall", "rediscover", rediscovery_period_frames=1),
    )
    assert mixed.world_motion == "fixed_winner_replay"
    assert mixed.motion_event_period_frames == 1
    rediscovering, fresh_state = cs.clutter_epoch_loop(cs.path_gaining_clutter_scene(), mixed)
    rediscovering.frame(0.0)
    assert set(_keys(fresh_state)[1]) == set(parked_keys)
    recovered = rediscovering.frame(1.0)
    assert recovered.rediscovered and recovered.reason == MOTION_EVENT_CADENCE
    fresh, fresh_keys = _keys(fresh_state)

    born = set(fresh_keys) - set(replayed_keys)
    assert born, sorted(fresh_keys)
    # The born rows are the arriving wall's, and they are CLUTTER.
    for key in born:
        assert "reflection" in key[3:], key
    assert (geo.TX_A_STABLE_ID, geo.SITE_P_STABLE_ID, geo.RX_A_STABLE_ID, "los", "reflection") in born
    # Nothing in the replay reported them: the alive set is a strict subset of
    # what a fresh discovery finds, and no replayed row names a born one.
    assert alive < set(fresh_keys)
    assert born & set(replayed_keys) == set()
    assert fresh_state.index.count(ENVIRONMENT_CLUTTER) == 7
    assert state.index.count(ENVIRONMENT_CLUTTER) == 5
