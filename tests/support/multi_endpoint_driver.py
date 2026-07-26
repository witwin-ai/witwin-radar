"""Orchestration for the real multi-endpoint two-way fixture.

compile -> freeze inbound -> freeze outbound -> freeze the join -> per-frame
reevaluate -> compose, with 2 TX, 2 scatter sites and 2 RX through the
production ``ChannelPropagationAdapter``. The sites are the SINKS of the
inbound leg and the SOURCES of the outbound leg, and one position tensor plays
both roles, so a site gradient accumulates from both legs exactly as in the
Phase-4 spike.

Every endpoint set is passed in as an explicit ordered sequence of
``(stable_id, position)``. That is not decoration: Channel's frozen leg row
order is a function of the ENDPOINT BATCH ROW POSITIONS, while the join's
canonical composed order is a function of stable IDENTITY. Declaring the batch
order is what lets a test drive the two apart without fabricating a leg.

This lives under ``tests/`` because it is fixture orchestration, not a
production owner. Every numerical primitive it calls is a production module.
"""

from __future__ import annotations

import torch

from . import multi_endpoint_geometry as geo
from . import multi_endpoint_world as world
from .synthesis_batch import to_synthesis


MULTIPATH_COMPONENTS = frozenset({"los", "reflection"})

# A target response strong enough that the synthesized IQ is the same order of
# magnitude as the Phase-4 reference. The absolute scale is arbitrary.
FIXTURE_AMPLITUDE = 1.0e5
FIXTURE_PHASE_RAD = 0.7


def make_response(*, requires_grad: bool = False, device: str = "cuda"):
    from witwin.radar.scattering import ScalarRcsResponse

    return ScalarRcsResponse.from_values(
        FIXTURE_AMPLITUDE,
        FIXTURE_PHASE_RAD,
        device=device,
        requires_grad=requires_grad,
    )


def make_spec(*, num_chirps: int | None = None, carrier_hz: float = 0.0):
    from witwin.radar import RadarConfig
    from witwin.radar.synthesis import FmcwBeatSpec

    config = RadarConfig.from_dict(dict(geo.FIXTURE_RADAR_CONFIG))
    spec = FmcwBeatSpec.from_radar_config(config, carrier_hz=carrier_hz)
    if num_chirps is not None:
        from dataclasses import replace

        spec = replace(spec, num_chirps=num_chirps)
    return spec


class MultiEndpointSpike:
    """One compiled scene, two frozen multi-pair legs, one frozen join.

    ``transmitters``, ``sites`` and ``receivers`` are ordered sequences of
    ``(stable_id, position)`` in endpoint batch order. ``declared_*`` override
    the identity lists handed to :meth:`TwoWayComposer.freeze` and default to
    the batch's own IDs; overriding them is how a test reaches the composer's
    stray-endpoint and unreachable-site refusals with REAL legs rather than
    fabricated ones.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        compiled=None,
        transmitters=geo.TRANSMITTERS,
        sites=geo.SITES,
        receivers=geo.RECEIVERS,
        declared_source_ids=None,
        declared_sink_ids=None,
        declared_site_ids=None,
        components: frozenset[str] = MULTIPATH_COMPONENTS,
        max_depth: int = 1,
    ) -> None:
        from witwin.radar.paths import TwoWayComposer
        from witwin.radar.propagation.channel_consumer import (
            ChannelPropagationAdapter,
        )

        self.device = device
        self.transmitters = tuple(transmitters)
        self.sites = tuple(sites)
        self.receivers = tuple(receivers)
        self.compiled = (
            world.compile_fixture_scene() if compiled is None else compiled
        )
        self.adapter = ChannelPropagationAdapter(
            self.compiled,
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
            components=components,
            max_depth=max_depth,
        )

        self.transmitter_ids, transmitter_positions = world.split(self.transmitters)
        self.site_ids, site_positions = world.split(self.sites)
        self.receiver_ids, receiver_positions = world.split(self.receivers)
        self.site_positions = site_positions

        # Freeze once, outside every loop.
        self.inbound = self.adapter.freeze(
            self._transmitter_batch(transmitter_positions),
            self._site_batch(site_positions, role="sink"),
        )
        self.outbound = self.adapter.freeze(
            self._site_batch(site_positions, role="source"),
            self._receiver_batch(receiver_positions),
        )
        self.composer = TwoWayComposer.freeze(
            self.inbound,
            self.outbound,
            # Declared in ENDPOINT BATCH ORDER, deliberately unsorted. The join
            # sorts the site IDs itself to build the site rank; handing it an
            # already-sorted list would make that sort unreachable and the
            # ``[Q, P]`` variant would stop testing it.
            torch.tensor(
                list(
                    self.site_ids
                    if declared_site_ids is None
                    else declared_site_ids
                ),
                dtype=torch.int64,
                device=device,
            ),
            radar_source_ids=list(
                self.transmitter_ids
                if declared_source_ids is None
                else declared_source_ids
            ),
            radar_sink_ids=list(
                self.receiver_ids if declared_sink_ids is None else declared_sink_ids
            ),
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        )

    # -- endpoint batches ---------------------------------------------------

    def _transmitter_batch(self, positions):
        return world.endpoint_batch(
            positions,
            self.transmitter_ids,
            power_w=geo.TX_POWER_W,
            device=self.device,
        )

    def _receiver_batch(self, positions):
        return world.endpoint_batch(
            positions, self.receiver_ids, device=self.device
        )

    def _site_batch(self, positions, *, role: str):
        """The site endpoint, excited at exactly 1 W when it is the source.

        It used to be excited at ``TX_POWER_W``, the same value as the real
        transmitter, so the composed coefficient carried ``sqrt(P_tx)`` twice.
        With ``TX_POWER_W = 1.0`` that was numerically invisible. The site is a
        re-radiator: the whole target strength belongs to the join's ``S``.
        """

        return world.endpoint_batch(
            positions,
            self.site_ids,
            power_w=geo.SITE_POWER_W if role == "source" else None,
            device=self.device,
        )

    def site_tensor(self, positions=None, *, requires_grad: bool = False):
        """The site positions as one ``(N, 3)`` float32 CUDA tensor."""

        values = self.site_positions if positions is None else positions
        tensor = torch.tensor(list(values), dtype=torch.float32, device=self.device)
        return tensor.requires_grad_(requires_grad)

    # -- one frame ----------------------------------------------------------

    def legs(self, sites=None, *, ad_mode: str = "none"):
        """Reevaluate both frozen legs at ``sites``.

        ``sites`` may be a live tensor: the SAME object is handed to the inbound
        sink batch and the outbound source batch, which is what makes a reverse
        gradient accumulate over both legs and a forward velocity dual produce a
        two-way Doppler rate.
        """

        positions = self.site_positions if sites is None else sites
        inbound = self.adapter.reevaluate(
            self.inbound,
            self._transmitter_batch(
                [position for _, position in self.transmitters]
            ),
            self._site_batch(positions, role="sink"),
            ad_mode=ad_mode,
        )
        outbound = self.adapter.reevaluate(
            self.outbound,
            self._site_batch(positions, role="source"),
            self._receiver_batch([position for _, position in self.receivers]),
            ad_mode=ad_mode,
        )
        return inbound, outbound

    def frame(
        self,
        sites=None,
        response=None,
        *,
        ad_mode: str = "none",
        include_delay_rate: bool = True,
    ):
        """One frame: two reevaluations and one composition."""

        inbound, outbound = self.legs(sites, ad_mode=ad_mode)
        composed = self.composer.compose(
            inbound,
            outbound,
            make_response() if response is None else response,
            include_delay_rate=include_delay_rate,
        )
        return composed, inbound, outbound

    # -- a whole slot-major frame in one call per leg ------------------------

    def stacked(self, positions, slot_count: int) -> torch.Tensor:
        """Repeat one endpoint set once per slot, slot major.

        ``Tensor.repeat`` and not a rebuild from Python values: a forward-AD
        dual carried by ``positions`` survives a differentiable op and dies the
        moment the caller reads it back into a list. A dead tangent publishes
        ``delay_rate = 0``, which is indistinguishable from a correct
        stationary answer, so the way the stack is built is load bearing.
        """

        values = (
            positions
            if isinstance(positions, torch.Tensor)
            else torch.tensor(
                list(positions), dtype=torch.float32, device=self.device
            )
        )
        return values.repeat(int(slot_count), 1)

    def slot_legs(
        self,
        site_positions,
        *,
        slot_count: int,
        ad_mode: str = "none",
        transmitter_positions=None,
        receiver_positions=None,
    ):
        """Both legs of a whole frame, in exactly ONE consumer call each.

        ``site_positions`` is the SLOT-MAJOR stack, ``(slot_count * sites, 3)``:
        slot ``t`` owns rows ``[t * sites, (t + 1) * sites)``. The transmitters
        and receivers default to the fixture's static positions repeated per
        slot; passing a stack instead is how a moving-platform scenario is
        driven.
        """

        sites = self.stacked(site_positions, 1)
        transmitters = (
            self.stacked(
                [position for _, position in self.transmitters], slot_count
            )
            if transmitter_positions is None
            else self.stacked(transmitter_positions, 1)
        )
        receivers = (
            self.stacked([position for _, position in self.receivers], slot_count)
            if receiver_positions is None
            else self.stacked(receiver_positions, 1)
        )
        inbound = self.adapter.reevaluate_slots(
            self.inbound,
            self._stacked_ids(transmitters, self.transmitter_ids, geo.TX_POWER_W),
            self._stacked_ids(sites, self.site_ids, None),
            slot_count=slot_count,
            ad_mode=ad_mode,
        )
        outbound = self.adapter.reevaluate_slots(
            self.outbound,
            self._stacked_ids(sites, self.site_ids, geo.SITE_POWER_W),
            self._stacked_ids(receivers, self.receiver_ids, None),
            slot_count=slot_count,
            ad_mode=ad_mode,
        )
        return inbound, outbound

    def _stacked_ids(self, positions: torch.Tensor, ids, power_w):
        """An endpoint spec whose stable IDs repeat once per slot.

        The frozen rows name their endpoints by stable IDENTITY, so every slot
        has to carry the SAME IDs in the same order. A stack that renamed them
        would be describing a different topology rather than a later instant,
        and Channel's stable-ID validation refuses it.
        """

        rows = int(positions.shape[0])
        listed = list(ids)
        if rows % len(listed):
            raise ValueError(
                f"a stack of {rows} endpoints is not a whole number of "
                f"{len(listed)}-endpoint slots"
            )
        slots = rows // len(listed)
        return world.endpoint_batch(
            positions, listed * slots, power_w=power_w, device=self.device
        )

    def slot_frames(self, inbound, outbound, response=None, *, include_delay_rate=True):
        """Compose every slot of a slot-major pair of legs.

        This is the REFRESHED-weight oracle and it is deliberately a test-side
        loop. The production inner loop composes ONCE per frame from the frozen
        weight and lets the waveform kernel own the slow-time carrier; this
        instead re-composes per slot so the two can be compared. The propagation
        replay it consumes is still a single batched call, so the loop adds no
        propagation launch and no host observation of its own.
        """

        target = make_response() if response is None else response
        return [
            self.composer.compose(
                inbound.slot(slot),
                outbound.slot(slot),
                target,
                include_delay_rate=include_delay_rate,
            )
            for slot in range(inbound.slot_count)
        ]

    # -- the single-pair comparison spike -----------------------------------

    def single_pair(self, transmitter, site, receiver) -> "MultiEndpointSpike":
        """The 1 x 1 x 1 Phase-4/5-shaped spike for one endpoint triple.

        Built on the SAME compiled scene, so any difference between it and the
        batched frame is batching and nothing else.
        """

        return MultiEndpointSpike(
            device=self.device,
            compiled=self.compiled,
            transmitters=(transmitter,),
            sites=(site,),
            receivers=(receiver,),
        )

    # -- what the geometry predicts for THIS endpoint declaration -----------

    def predicted_inbound_rows(self):
        return geo.leg_rows(self.transmitters, self.sites)

    def predicted_outbound_rows(self):
        return geo.leg_rows(self.sites, self.receivers)

    def predicted_combined_rows(self):
        return geo.combined_rows(self.transmitters, self.sites, self.receivers)


def composed_keys(spike: MultiEndpointSpike, composed):
    """Name every composed row by frame-invariant identity.

    ``(source_id, site_id, sink_id, inbound component, outbound component)``,
    resolved through the composed topology's leg row indices. Naming a row by
    its position in a sorted delay list would be unusable here: the mixed
    ``(los, reflection)`` and ``(reflection, los)`` round trips through site P
    and RX_A differ by 20 ps.
    """

    names = {
        geo.LOS_COMPONENT_ID: "los",
        geo.REFLECTION_COMPONENT_ID: "reflection",
    }
    inbound_components = spike.inbound.component_id.tolist()
    outbound_components = spike.outbound.component_id.tolist()
    topology = spike.composer.topology
    inbound_row = topology.inbound_row.tolist()
    outbound_row = topology.outbound_row.tolist()
    return [
        (
            int(topology.radar_source_id[row]),
            int(topology.site_id[row]),
            int(topology.radar_sink_id[row]),
            names[int(inbound_components[inbound_row[row]])],
            names[int(outbound_components[outbound_row[row]])],
        )
        for row in range(composed.path_count)
    ]


def slot_site_stack(base: torch.Tensor, velocity, times_s) -> torch.Tensor:
    """``base + v * t`` for every slot, stacked slot major.

    Built as ONE differentiable expression over ``base`` so that a forward-AD
    dual on ``base`` reaches every slot. ``velocity`` is the site kinematics in
    metres per second and ``times_s`` are the slot times - the TDM slot table's
    own values, not a second time axis invented here.
    """

    device = base.device
    offsets = torch.as_tensor(times_s, dtype=torch.float32, device=device)
    direction = torch.as_tensor(velocity, dtype=torch.float32, device=device)
    if direction.ndim == 1:
        direction = direction.reshape(1, 3).expand(base.shape[0], 3)
    displacement = offsets.reshape(-1, 1, 1) * direction.reshape(1, -1, 3)
    return (base.reshape(1, -1, 3) + displacement).reshape(-1, 3)


__all__ = [
    "FIXTURE_AMPLITUDE",
    "FIXTURE_PHASE_RAD",
    "MULTIPATH_COMPONENTS",
    "MultiEndpointSpike",
    "composed_keys",
    "make_response",
    "make_spec",
    "slot_site_stack",
]
