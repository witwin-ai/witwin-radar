"""Per-frame orchestration budgets of the scene session, per ``Motion`` kind.

Every public verb runs one session, so this is the cost every ``Radar.simulate``
call pays outside the kernels. Each case is measured twice, over one frame and
over two, and the budget pins both the first frame (which carries the session
setup and the first discovery) and the marginal frame (the difference), so a
change that moves work out of the frame loop shows up in the second column
rather than vanishing into a total.

The numbers are exact integers for these fixtures. They are a ceiling, pinned
at the measured value: a change that lowers one lowers the pin in the same
commit, and a change that raises one has to explain itself here.
``python tests/test_session_budget.py`` prints the measured table, one fresh
interpreter per case as the test measures it.

What is counted, and its blind spots, is stated once in
:mod:`support.session_ledger`.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

#: A deliberately small FMCW front end: 2 x 2 elements, 2 chirps of 8 samples,
#: so an ADC-refreshed frame has 2 * 2 * 8 = 32 observation instants.
CONFIG = {
    "num_tx": 2,
    "num_rx": 2,
    "fc": 77e9,
    "slope": 60.0,
    "adc_samples": 8,
    "adc_start_time": 1,
    "sample_rate": 2000,
    "idle_time": 1,
    "ramp_end_time": 10,
    "chirp_per_frame": 2,
    "power": 12,
    "tx_loc": [[0, 0, 0], [2, 0, 0]],
    "rx_loc": [[-3, 0, 0], [-2, 0, 0]],
}

TARGET_M = (0.0, 0.0, -3.0)
SECOND_SITE_M = (0.5, 0.2, -4.0)
FRAME_TIMES_S = (0.0, 0.01)

#: A fresh Core antenna ID per built scene, far from every other test's block.
_ENDPOINT_IDS = iter(range(993_001, 994_000))

#: The counters a budget pins, in report order.
COUNTERS = (
    "evaluate",
    "prepare_fixed_topology",
    "reevaluate",
    "replicate_over_slots",
    "rediscovery_required",
    "native",
    "synchronizations",
    "d2h",
    "h2d",
    "kernels",
    "compiles",
    "discoveries",
    "world_advances",
    "compositions",
    "joins",
)

#: The profiled columns. Each case is measured in a fresh interpreter, where
#: they do not depend on which tests ran before, and each count is the largest
#: of three profiled passes, which undoes a dropped profiler record in one
#: pass. Under a contended device every pass can still lose a record or a few
#: dozen kernels (at most 26 of 24368 measured, 0.1 %), and a loss in the
#: one-frame run RAISES the marginal column, so these hold to their pin plus
#: :data:`PROFILED_MARGIN` (at least two). Every other counter is exact.
PROFILED = frozenset({"d2h", "h2d", "kernels"})
PROFILED_MARGIN = 0.005

#: Measured ceilings: ``case -> (first frame, marginal frame)``, each a tuple
#: in :data:`COUNTERS` order, the largest of two fresh-interpreter runs. The
#: transport pass (R-ADR-039 to 041) adds about ten kernels per discovered
#: frame and no host observation: the legs' launch polarization and float64
#: site gathers, the join's polarization table and Channel's complex3 field on
#: the discovery rows.
BUDGETS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "adaptive_point": (
        (8, 0, 0, 0, 12, 4, 96, 65, 98, 1332, 1, 4, 4, 1, 1),
        (8, 0, 0, 0, 16, 4, 60, 54, 62, 1203, 0, 4, 4, 1, 0),
    ),
    "adaptive_structure": (
        (8, 0, 0, 0, 12, 7, 168, 93, 175, 1577, 4, 4, 4, 4, 1),
        (8, 0, 0, 0, 16, 7, 159, 93, 166, 1540, 4, 4, 4, 4, 0),
    ),
    "adc_point": (
        (64, 0, 0, 0, 124, 3, 359, 331, 487, 8421, 1, 32, 32, 1, 1),
        (64, 0, 0, 0, 128, 3, 321, 320, 449, 8292, 0, 32, 32, 1, 0),
    ),
    "adc_point_per_chirp": (
        (8, 2, 2, 2, 68, 3, 122, 63, 134, 1630, 1, 4, 32, 1, 1),
        (8, 2, 2, 2, 72, 3, 83, 52, 95, 1501, 0, 4, 32, 1, 0),
    ),
    "adc_point_per_frame": (
        (2, 2, 2, 2, 62, 3, 95, 33, 95, 862, 1, 1, 32, 1, 1),
        (2, 2, 2, 2, 66, 3, 56, 22, 56, 733, 0, 1, 32, 1, 0),
    ),
    "adc_structure": (
        (64, 0, 0, 0, 124, 95, 1179, 672, 1307, 12202, 32, 32, 32, 32, 1),
        (64, 0, 0, 0, 128, 96, 1152, 672, 1280, 12166, 32, 32, 32, 32, 0),
    ),
    "auto_point": (
        (8, 0, 0, 0, 12, 4, 96, 65, 98, 1332, 1, 4, 4, 1, 1),
        (8, 0, 0, 0, 16, 4, 60, 54, 62, 1203, 0, 4, 4, 1, 0),
    ),
    "chirp_point": (
        (8, 0, 0, 0, 12, 3, 79, 51, 95, 1251, 1, 4, 4, 1, 1),
        (8, 0, 0, 0, 16, 3, 41, 40, 57, 1122, 0, 4, 4, 1, 0),
    ),
    "chirp_structure": (
        (8, 0, 0, 0, 12, 11, 157, 84, 173, 1560, 4, 4, 4, 4, 1),
        (8, 0, 0, 0, 16, 12, 144, 84, 160, 1524, 4, 4, 4, 4, 0),
    ),
    "static": ((2, 0, 0, 0, 0, 2, 45, 21, 49, 410, 1, 1, 1, 1, 1), (0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)),
}


def _ceiling(name: str, limit: int) -> int:
    return limit + max(2, int(limit * PROFILED_MARGIN)) if name in PROFILED else limit


def _radar():
    from witwin.radar import Radar

    return Radar.from_dict(CONFIG, position=(0.0, 0.0, 0.0), look_at=(0.0, 0.0, -1.0))


def _wall_scene(*, moving: bool):
    from witwin.core import AntennaState, Mesh, PhysicalMaterial, Scene, Structure
    from witwin.core.dynamics import DynamicScene, LinearTrajectory
    from witwin.core.identity import reserve_antenna_id

    mesh = Mesh(
        vertices=torch.tensor(((-2.0, -2.0, -5.0), (2.0, -2.0, -5.0), (2.0, 2.0, -5.0), (-2.0, 2.0, -5.0))),
        faces=torch.tensor(((0, 1, 2), (0, 2, 3)), dtype=torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    wall = Structure(
        geometry=mesh,
        material=PhysicalMaterial(name="concrete", eps_r=5.24, sigma_e=0.0462),
        structure_id=1,
        material_id=1,
        assignment_id=1,
        surface_id=1,
    )
    endpoint = AntennaState(reserve_antenna_id(next(_ENDPOINT_IDS)), "tx", torch.zeros(3))
    scene = Scene(structures=(wall,), endpoints=[endpoint])
    if not moving:
        return scene
    trajectory = LinearTrajectory(origin=torch.zeros(3), velocity=torch.tensor([0.0, 0.0, 3.0]))
    return DynamicScene(scene, structure_trajectories={1: trajectory})


def _moving_site(time_s: float) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, TARGET_M[2] + 5.0 * float(time_s)]], dtype=torch.float32, device="cuda")


def _targets(kind: str):
    from witwin.radar import PointTargets

    if kind == "still":
        return PointTargets(positions=[TARGET_M, SECOND_SITE_M], rcs=1.0)
    return PointTargets(positions=[TARGET_M], rcs=1.0, trajectory=_moving_site)


def _motion(kind: str):
    from witwin.radar import Motion

    return {
        "static": Motion.static(),
        "chirp": Motion.chirp(),
        "adc": Motion.adc(),
        "adaptive": Motion.adaptive(max_interval=2e-4),
        "adc_per_frame": Motion.adc(rediscover_every_frames=1),
        "adc_per_chirp": Motion.adc(rediscover_every_chirps=1),
        "auto": Motion.auto(),
    }[kind]


#: ``case -> (motion kind, target kind, whether the wall moves)``. Multipath is
#: on in every case: the wall is in every scene and ``reflections=1``.
CASES = {
    "static": ("static", "still", False),
    "chirp_point": ("chirp", "mover", False),
    "adc_point": ("adc", "mover", False),
    "adaptive_point": ("adaptive", "mover", False),
    "chirp_structure": ("chirp", "still", True),
    "adc_structure": ("adc", "still", True),
    "adaptive_structure": ("adaptive", "still", True),
    # Static geometry and a declared once-a-frame discovery: the case where
    # the observation count should cost nothing at the Channel boundary.
    "adc_point_per_frame": ("adc_per_frame", "mover", False),
    # The default for a moving FMCW scene resolves to adaptive sampling.
    "auto_point": ("auto", "mover", False),
    # Discover once per chirp, replay at every ADC instant of it.
    "adc_point_per_chirp": ("adc_per_chirp", "mover", False),
}


def _simulate(case: str, frames: int):
    motion, targets, moving = CASES[case]
    radar = _radar()
    scene, sites = _wall_scene(moving=moving), _targets(targets)
    return lambda: radar.simulate(scene, sites, times=FRAME_TIMES_S[:frames], reflections=1, motion=_motion(motion))


def _measure_once(case: str, frames: int, monkeypatch) -> tuple[int, ...]:
    from support.session_ledger import SessionLedger, profiled_copies

    copies = profiled_copies(_simulate(case, frames))
    with monkeypatch.context() as patch:
        ledger = SessionLedger(patch)
        result = ledger.run(_simulate(case, frames))
    return (
        ledger.channel["evaluate"],
        ledger.channel["prepare_fixed_topology"],
        ledger.channel["reevaluate"],
        ledger.channel["replicate_over_slots"],
        ledger.channel["rediscovery_required"],
        sum(ledger.native.values()),
        ledger.synchronizations,
        copies["d2h"],
        copies["h2d"],
        copies["kernels"],
        int(result.compile_count),
        int(result.discovery_count),
        ledger.trips["SceneEpochLoop.frame"],
        ledger.trips["TwoWayComposer.compose"] + ledger.trips["TwoWayComposer.compose_slots"],
        ledger.trips["TwoWayComposer.freeze"],
    )


def measure(case: str, monkeypatch) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The first frame's counts and the marginal frame's, for one case."""

    # Untouched runs of both lengths first, so lazy imports, the native load
    # and every first-use cache are warm before either length is counted.
    for frames in (1, 2):
        _simulate(case, frames)()
    one = _measure_once(case, 1, monkeypatch)
    two = _measure_once(case, 2, monkeypatch)
    return one, tuple(b - a for a, b in zip(one, two, strict=True))


def measure_case(case: str) -> list[list[int]]:
    """:func:`measure` with its own monkeypatch; what the fresh interpreter runs."""

    with pytest.MonkeyPatch.context() as patch:
        return [list(values) for values in measure(case, patch)]


def _report(case: str, measured, budget) -> str:
    rows = []
    for label, values, ceiling in (("first", measured[0], budget[0]), ("marginal", measured[1], budget[1])):
        cells = ", ".join(
            f"{name}={value}/{limit}" for name, value, limit in zip(COUNTERS, values, ceiling, strict=True)
        )
        rows.append(f"{case} {label}: {cells}")
    return "\n".join(rows)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_session_stays_within_its_per_frame_budget(case):
    pytest.importorskip("witwin.channel")
    if not torch.cuda.is_available():
        pytest.skip("the scene session requires CUDA")
    from support.session_ledger import measure_isolated

    first, marginal = measure_isolated("test_session_budget", "measure_case", case)
    measured = (tuple(first), tuple(marginal))
    budget = BUDGETS[case]
    over = [
        f"{label}.{name}: {value} > {limit}"
        for label, values, ceiling in (("first", measured[0], budget[0]), ("marginal", measured[1], budget[1]))
        for name, value, limit in zip(COUNTERS, values, ceiling, strict=True)
        if value > _ceiling(name, limit)
    ]
    assert not over, f"{over}\n{_report(case, measured, budget)}"


if __name__ == "__main__":
    import sys

    sys.path[:0] = ["tests", "."]
    from support.session_ledger import measure_isolated

    for name in sys.argv[1:] or sorted(CASES):
        first, marginal = measure_isolated("test_session_budget", "measure_case", name)
        print(f'    "{name}": ({tuple(first)}, {tuple(marginal)}),', flush=True)
