"""End-to-end orchestration for the single-site AD spike.

compile -> discover -> freeze -> per-frame reevaluate -> compose -> synthesize
-> scalar loss, with one differentiable scatter site that is the sink of the
inbound leg and the source of the outbound leg. The SAME position tensor plays
both roles; nothing special is needed to make that work, and its gradient
accumulates from both legs.

This lives under ``tests/`` because it is orchestration for a spike, not a
production owner. Every numerical primitive it calls is a production module.
The response, the waveform spec and the loss are the multi-endpoint driver's
and the reference chain's, bound to this fixture's front end.
"""

from __future__ import annotations

import torch

from . import multi_endpoint_driver
from . import single_site_geometry as geo
from . import single_site_world as world
from .multi_endpoint_driver import FIXTURE_AMPLITUDE as SPIKE_AMPLITUDE
from .multi_endpoint_driver import FIXTURE_PHASE_RAD as SPIKE_PHASE_RAD
from .multi_endpoint_driver import make_response
from .reference_chain import radar_loss as _radar_loss
from .synthesis_batch import to_synthesis


def make_spec(*, num_chirps: int | None = None, carrier_hz: float = 0.0):
    return multi_endpoint_driver.make_spec(
        config=geo.FIXTURE_RADAR_CONFIG, num_chirps=num_chirps, carrier_hz=carrier_hz
    )


class Phase4Spike:
    """One compiled scene, two frozen legs, one frozen two-way join.

    ``components`` and ``max_depth`` default to the line-of-sight values so
    every single-site expectation keeps its numbers. Passing
    ``frozenset({"los", "reflection"}), max_depth=1`` turns the SAME fixture
    into the multipath case: no adapter change and no new fixture geometry are
    needed, because the consumer already accepts both components for fixed
    topology and the fixture wall was authored for exactly this.

    A multipath leg cannot take the raw-topology fast path - the consumer
    rejects a raw ``PropagationTopology`` whose interaction sequence is
    non-empty - so it necessarily goes through the prepared handle. The adapter
    already does; this is recorded rather than branched on.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        components: frozenset[str] = frozenset({"los"}),
        max_depth: int = 0,
        compiled=None,
    ) -> None:
        from witwin.radar.channel import ChannelPropagationAdapter
        from witwin.radar.paths import TwoWayComposer

        self.device = device
        self.compiled = world.compile_fixture_scene() if compiled is None else compiled
        self.adapter = ChannelPropagationAdapter(
            self.compiled, reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ, components=components, max_depth=max_depth
        )
        # Freeze once, outside every loop.
        self.inbound = self.adapter.freeze(
            self._source(geo.TX_POSITION_M, geo.TX_STABLE_ID), self._sink(geo.SITE_POSITION_M, geo.SITE_STABLE_ID)
        )
        self.outbound = self.adapter.freeze(
            self._source(geo.SITE_POSITION_M, geo.SITE_STABLE_ID), self._sink(geo.RX_POSITION_M, geo.RX_STABLE_ID)
        )
        self.composer = TwoWayComposer.freeze(
            self.inbound,
            self.outbound,
            torch.tensor([geo.SITE_STABLE_ID], dtype=torch.int64, device=device),
            radar_source_ids=torch.tensor([geo.TX_STABLE_ID], dtype=torch.int64, device=device),
            radar_sink_ids=torch.tensor([geo.RX_STABLE_ID], dtype=torch.int64, device=device),
            reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        )

    def _source(self, position, stable_id):
        return world.endpoint_spec(position, stable_id, power_w=geo.TX_POWER_W, device=self.device)

    def _sink(self, position, stable_id):
        return world.endpoint_spec(position, stable_id, device=self.device)

    def paths(self, tx: torch.Tensor, site: torch.Tensor, rx: torch.Tensor, response, *, ad_mode: str = "none"):
        """One frame: two reevaluations and one composition."""

        inbound = self.adapter.reevaluate_slots(
            self.inbound,
            self._source(tx, geo.TX_STABLE_ID),
            self._sink(site, geo.SITE_STABLE_ID),
            slot_count=1,
            ad_mode=ad_mode,
        )
        outbound = self.adapter.reevaluate_slots(
            self.outbound,
            self._source(site, geo.SITE_STABLE_ID),
            self._sink(rx, geo.RX_STABLE_ID),
            slot_count=1,
            ad_mode=ad_mode,
        )
        composed = self.composer.compose(inbound, outbound, response)
        return composed, inbound, outbound

    def synthesize(
        self, tx: torch.Tensor, site: torch.Tensor, rx: torch.Tensor, response, spec, *, ad_mode: str = "none"
    ) -> torch.Tensor:
        from witwin.radar.synthesis.fmcw import synthesize_fmcw

        composed, _, _ = self.paths(tx, site, rx, response, ad_mode=ad_mode)
        return synthesize_fmcw(to_synthesis(composed), spec)

    def loss(
        self,
        tx: torch.Tensor,
        site: torch.Tensor,
        rx: torch.Tensor,
        response,
        spec,
        reference_iq: torch.Tensor,
        *,
        ad_mode: str = "vjp",
    ) -> torch.Tensor:
        iq = self.synthesize(tx, site, rx, response, spec, ad_mode=ad_mode)
        return radar_loss(iq, reference_iq)


def radar_loss(iq: torch.Tensor, reference_iq: torch.Tensor) -> torch.Tensor:
    """``reference_chain.radar_loss`` with the CPU reference moved to ``iq``'s device."""

    return _radar_loss(iq, reference_iq.to(device=iq.device))


def make_reference_iq(spec, *, seed: int = 20260724, scale: float = 2.0e-3):
    """A fixed pseudo-random target signal, on the CPU in float64.

    It is a CONSTANT shared by the production chain and the oracle, so it
    cancels out of the comparison while keeping the loss phase sensitive.
    """

    generator = torch.Generator().manual_seed(seed)
    shape = (spec.num_chirps, 1, spec.num_samples)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imag = torch.randn(shape, generator=generator, dtype=torch.float64)
    return scale * torch.complex(real, imag)


def positions(*, requires_grad: bool = False, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def make(value):
        tensor = torch.tensor([value], dtype=torch.float32, device=device)
        return tensor.requires_grad_(requires_grad)

    return (make(geo.TX_POSITION_M), make(geo.SITE_POSITION_M), make(geo.RX_POSITION_M))


def oracle_positions() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The same three positions as float64 CPU tensors for the oracle."""

    return tuple(
        torch.tensor(value, dtype=torch.float64)
        for value in (geo.TX_POSITION_M, geo.SITE_POSITION_M, geo.RX_POSITION_M)
    )


__all__ = [
    "Phase4Spike",
    "SPIKE_AMPLITUDE",
    "SPIKE_PHASE_RAD",
    "make_reference_iq",
    "make_response",
    "make_spec",
    "oracle_positions",
    "positions",
    "radar_loss",
]
