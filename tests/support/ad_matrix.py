"""One scenario, eight leaves, three waveforms - the Phase-9 AD matrix fixture.

Every earlier stage drove ONE leaf at a time. That is what a per-cell test is
for, and it cannot see the defect this scenario exists to find: a combined
backward that silently drops a leaf, writes one leaf's cotangent into another's
slot, or answers a different topology in one AD mode than in another. Those need
every supported leaf live in the SAME backward, over the SAME frozen topology,
in all three modes.

**The eight leaves.** ``vertices`` and the two material fields are scene state
baked into the compiled scene; ``sites``, ``transmitters`` and ``receivers`` are
endpoint positions replayed against the frozen topology; ``sigma_m2`` and
``phase_rad`` are the target response. They reach the loss through four
different parts of the chain - the specular geometry, the Fresnel coefficient,
the endpoint delays, and the response multiply - which is what makes a
cross-contaminated reduction visible.

**One frozen topology.** ``multi_endpoint_driver.MultiEndpointSpike`` freezes
both legs and the join once. A scene leaf has to be inside the compile, so a
scenario built with live scene leaves is necessarily a different Python object
from one built without; what makes it the SAME topology is that the composed
identity keys, the row order and the primal are identical, and
``test_phase9_combined_ad_matrix.py`` asserts exactly that rather than assuming
it.

**Why the loss has two terms.** ``sum |cube|^2`` alone is invariant under the
response phase: ONE ``ScalarRcsResponse`` multiplies every composed row, so
``phase_rad`` is a global phase and a magnitude loss cannot see it - measured,
its gradient there is 1e-8 of the loss, which is float32 roundoff of an exact
zero. Adding ``sum Re(cube^2)`` breaks that invariance: a global rotation by
``theta`` multiplies ``cube^2`` by ``exp(2 j theta)``. The second term is the
same degree in the cube as the first, so the two do not fight over scale, and
the magnitude term still dominates the conditioning. A test pins the invariance
of the magnitude half so this choice cannot be mistaken for decoration.

Finite differences here are the test oracle and never a production route.
"""

from __future__ import annotations

import torch

from . import multi_endpoint_driver as drv
from . import multi_endpoint_geometry as geo
from . import multi_endpoint_world as world
from . import waveform_chains as wc

WAVEFORMS = ("fmcw", "ofdm", "pulsed")

#: Every supported leaf of the scenario, in the order a combined backward marks
#: them. The order is the order the assertions report in, nothing more.
LEAF_NAMES = ("vertices", "eps_r", "sigma_e", "sites", "transmitters", "receivers", "sigma_m2", "phase_rad")

#: The scene leaves must be inside the compile; the rest are per-frame inputs.
SCENE_LEAVES = ("vertices", "eps_r", "sigma_e")

BASE_VERTICES = torch.tensor(geo.WALL_VERTICES_M, dtype=torch.float32)
BASE_EPS_R = float(geo.WALL_EPS_R)
BASE_SIGMA_E = float(geo.WALL_SIGMA_E)

#: Square metres. A physically ordinary vehicle-scale cross section; the value
#: is arbitrary and only its being strictly positive matters, because
#: ``d(amplitude)/d(sigma)`` is unbounded at zero.
BASE_SIGMA_M2 = 3.5

#: Radians. Non-zero on purpose: at exactly zero the response is real and a
#: sign error in the conjugation convention would be invisible.
BASE_PHASE_RAD = 0.7


def base_values(spike) -> dict:
    """The eight leaves at their base values, all detached.

    ``spike`` supplies the endpoint positions so that the scenario cannot drift
    from the fixture's own geometry constants.
    """

    return {
        "vertices": BASE_VERTICES.clone(),
        "eps_r": torch.tensor(BASE_EPS_R),
        "sigma_e": torch.tensor(BASE_SIGMA_E),
        "sites": spike.site_tensor(),
        "transmitters": spike.transmitter_tensor(),
        "receivers": spike.receiver_tensor(),
        "sigma_m2": torch.tensor(BASE_SIGMA_M2, dtype=torch.float32, device=spike.device),
        "phase_rad": torch.tensor(BASE_PHASE_RAD, dtype=torch.float32, device=spike.device),
    }


def marked(values: dict, names) -> dict:
    """A copy of ``values`` with ``names`` marked as reverse-mode leaves."""

    live = dict(values)
    for name in names:
        live[name] = values[name].clone().requires_grad_(True)
    return live


def response_of(values: dict):
    """The target response built from ``sigma_m2`` and ``phase_rad``.

    Through ``from_rcs`` rather than ``from_values``: the cross section is the
    leaf an inverse-design caller actually holds, and the ``sqrt(4 pi sigma)/lam``
    law is the only place in the package that knows what a square metre is
    worth. The phase is attached afterwards because ``from_rcs`` takes it as a
    host float and this scenario needs it live.
    """

    from witwin.radar.scattering import ScalarRcsResponse

    amplitude = ScalarRcsResponse.from_rcs(
        values["sigma_m2"], reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ, device=values["sigma_m2"].device
    ).amplitude
    return ScalarRcsResponse(amplitude=amplitude, phase_rad=values["phase_rad"])


def build_spike(values: dict):
    """Compile the world at ``values`` and freeze both legs and the join."""

    return drv.MultiEndpointSpike(
        compiled=world.compile_fixture_scene(
            vertices=values["vertices"], eps_r=values["eps_r"], sigma_e=values["sigma_e"]
        )
    )


def combined_loss(cube: torch.Tensor) -> torch.Tensor:
    """``sum |c|^2 + sum Re(c^2)``. See the module docstring for the second term."""

    return cube.abs().square().sum() + (cube * cube).real.sum()


def frame(values: dict, *, ad_mode: str = "none"):
    """One whole frame at ``values``: compile, freeze, replay, compose.

    Returns ``(composed, spike)``. The caller synthesizes, because the waveform
    is the one thing that varies across the matrix while the topology does not.
    """

    spike = build_spike(values)
    composed, _, _ = spike.frame(
        values["sites"],
        response_of(values),
        transmitters=values["transmitters"],
        receivers=values["receivers"],
        ad_mode=ad_mode,
        include_delay_rate=False,
    )
    return composed, spike


def replay(spike, values: dict, *, ad_mode: str = "none"):
    """One frame on an EXISTING spike: no compile, no freeze, no discovery.

    This is the entry the mode-identity and row-validity groups use, because
    the acceptance criterion is about one frozen topology and a fresh
    ``MultiEndpointSpike`` is a fresh ``PreparedFixedTopology``. It ignores the
    three scene leaves - they are inside the compile ``spike`` already carries -
    so a caller that marks one and passes it here would be marking a tensor the
    graph never reaches. ``build_spike`` is the entry for those.
    """

    composed, _, _ = spike.frame(
        values["sites"],
        response_of(values),
        transmitters=values["transmitters"],
        receivers=values["receivers"],
        ad_mode=ad_mode,
        include_delay_rate=False,
    )
    return composed


def cube_of(kind: str, values: dict, *, ad_mode: str = "none", spike=None) -> torch.Tensor:
    """The synthesized cube for one waveform at ``values``."""

    composed = frame(values, ad_mode=ad_mode)[0] if spike is None else replay(spike, values, ad_mode=ad_mode)
    return wc.synthesize(kind, composed, wc.make_spec(kind))


def loss_of(kind: str, values: dict, *, ad_mode: str = "none", spike=None) -> torch.Tensor:
    """Core leaf -> propagation -> RCS -> two-way -> cube -> scalar."""

    return combined_loss(cube_of(kind, values, ad_mode=ad_mode, spike=spike))


#: The direction magnitude per leaf, in that leaf's own units. One finite
#: difference step ``h`` moves every leaf at once, so the magnitudes carry the
#: unit conversion: ``h = 1e-4`` moves a vertex by 1e-4 m along the wall normal,
#: an endpoint by up to 8e-5 m, ``eps_r`` and ``sigma_m2`` by 5e-4, ``sigma_e``
#: by 5e-6 S/m and the phase by 5e-4 rad.
DIRECTION_SCALES = {
    "vertices": ((1.0, 0.0, 0.0),) * 4,
    "eps_r": 5.0,
    "sigma_e": 0.05,
    "sites": ((0.4, 0.8, 0.0), (0.6, 0.3, 0.0)),
    # TX_B publishes no rows at all in this fixture, so its direction is zero:
    # a nonzero one would add a perturbation the loss cannot see and would
    # only dilute the difference.
    "transmitters": ((0.5, 0.5, 0.0), (0.0, 0.0, 0.0)),
    "receivers": ((0.5, 0.5, 0.0), (0.5, 0.5, 0.0)),
    "sigma_m2": 5.0,
    "phase_rad": 5.0,
}

#: The finite-difference step shared by every leaf, with the fourth-order
#: stencil. Swept over 5e-5, 1e-4, 2e-4 and 4e-4 on all three waveforms; the
#: relative disagreement between the sum of the eight single-leaf differences
#: and the all-at-once difference runs
#:
#:     h        5e-5      1e-4      2e-4      4e-4
#:     fmcw     1.8%      0.19%     3.1%      13%
#:     ofdm     0.18%     0.60%     8.1%       -
#:     pulsed   0.004%    0.71%     10%        -
#:
#: and the analytic-against-all-at-once disagreement is 0.94%, 0.67% and 0.71%
#: at 1e-4. Below 1e-4 the float32 loss stops resolving the difference; above
#: it the 77 GHz phase turns inside the stencil. 1e-4 is the only step inside
#: the window for all three waveforms.
FD_STEP = 1.0e-4


def direction(name: str, like: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """The perturbation direction for ``name``, signed so the parts ADD.

    The sign comes from the gradient itself rather than from a hand-written
    table. With an arbitrary sign pattern the live components of a leaf's
    contribution cancel - S1 measured a factor of four amplification of the
    difference's noise on the site leaf and S2 measured a factor of twenty on
    the OFDM one - and a cancelled directional derivative is a finite-difference
    check with no resolution left. Signing by the gradient makes every component
    contribute with the same sign by construction, and the test asserts that
    property directly so it cannot rot.
    """

    scale = torch.as_tensor(DIRECTION_SCALES[name], dtype=like.dtype, device=like.device)
    return torch.sign(gradient) * scale


def perturbed(values: dict, directions: dict, active, offset: float, step: float) -> dict:
    """``values`` moved by ``offset * step`` along ``directions`` for ``active``."""

    moved = dict(values)
    for name in active:
        moved[name] = values[name] + (offset * step) * directions[name].to(values[name].device)
    return moved


__all__ = [
    "BASE_EPS_R",
    "BASE_PHASE_RAD",
    "BASE_SIGMA_E",
    "BASE_SIGMA_M2",
    "BASE_VERTICES",
    "DIRECTION_SCALES",
    "FD_STEP",
    "LEAF_NAMES",
    "SCENE_LEAVES",
    "WAVEFORMS",
    "base_values",
    "build_spike",
    "combined_loss",
    "cube_of",
    "direction",
    "frame",
    "loss_of",
    "marked",
    "perturbed",
    "replay",
    "response_of",
]
