"""One reusable owner for the OFDM and pulsed end-to-end chains.

Every Phase-4 to Phase-8 end-to-end AD test synthesizes FMCW. The OFDM and
pulsed families are covered at the OPERATOR level - synthetic row tensors, a
float64 oracle, four inputs each - and nowhere else, so the question "does a
Core leaf reach an OFDM cube" had no test at all until Phase 9. It is not a
small question: the route from an endpoint position to a cube runs through the
Channel consumer, the two-way join, the scatter response and one of three
kernels, and only the last of those four is shared with the operator tests.

This module owns the two specs and the two chain functions so that the S2
single-leaf tests and the S4 combined-input matrix build on ONE scenario rather
than three. Everything here is fixture orchestration; every numerical primitive
it calls is a production module.

**Two choices that are load bearing and would be easy to get subtly wrong.**

*The pulse must be an LFM.* A rectangular pulse's dependence on ``tau_rt`` is
entirely through its support test - a rectangle has no phase for the delay to
move, and the frozen weight already owns the carrier - so its almost-everywhere
delay derivative is EXACTLY zero. That is a real property of the model, pinned
by ``test_phase6_pulsed_ad.py``, and a pulsed end-to-end AD test built on a
rectangle would be asserting that zero equals zero while looking exactly like a
test of the chain.

*The OFDM cyclic prefix has to contain the whole echo window.* The fixture's
round-trip delays run from 1.3e-8 s to about 3e-8 s, so the configured
``max_expected_delay_s`` is 1e-7 s and the prefix 2e-7 s. Both are CONFIGURED
bounds, never measured from the device rows, for the reason
``require_ofdm_compatible`` states.
"""

from __future__ import annotations

import torch

from . import multi_endpoint_geometry as geo
from .synthesis_batch import to_synthesis


#: A narrowband OFDM frame that fits the fixture. Eight subcarriers at 1 MHz is
#: an 8 MHz band, which is deliberately modest: the point here is the CHAIN, and
#: a wide band would drag in the wideband route that
#: ``test_phase8_wideband_ofdm.py`` already owns.
def ofdm_spec(*, num_symbols: int = 2, num_subcarriers: int = 8):
    from witwin.radar.synthesis import OfdmCfrSpec

    return OfdmCfrSpec(
        num_subcarriers=num_subcarriers,
        num_symbols=num_symbols,
        subcarrier_spacing_hz=1.0e6,
        cyclic_prefix_s=2.0e-7,
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        max_expected_delay_s=1.0e-7,
        carrier_hz=0.0,
        carrier_rate_hz=geo.REFERENCE_FREQUENCY_HZ,
    )


#: A linear-FM pulse train that fits the fixture. 500 MSPS over 32 samples is a
#: 6.4e-8 s gate, which contains every round-trip delay the fixture produces plus
#: the 2e-8 s pulse; the 500 MHz sweep gives a 2e-9 s range cell, and with a
#: static fixture the range-migration bound is satisfied with room to spare.
def pulsed_spec(*, num_pulses: int = 2, num_samples: int = 32):
    from witwin.radar.synthesis import PulsedEchoSpec

    return PulsedEchoSpec(
        num_pulses=num_pulses,
        num_samples=num_samples,
        sample_period_s=2.0e-9,
        pri_s=1.0e-6,
        range_gate_start_s=0.0,
        pulse_kind="lfm",
        pulse_width_s=2.0e-8,
        bandwidth_hz=5.0e8,
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        max_expected_delay_rate=0.0,
        carrier_hz=0.0,
        carrier_rate_hz=geo.REFERENCE_FREQUENCY_HZ,
    )


def synthesize(kind: str, composed, spec) -> torch.Tensor:
    """Dispatch to the waveform owner by name, with no default.

    A ``dict.get`` with a fallback here would turn an unknown waveform into
    another waveform's cube, which is the shape of mistake the production
    dispatch guard in ``test_phase6_no_torch_physics.py`` exists to prevent.
    Mirroring it in the fixture keeps a typo in a parametrize list from
    silently testing FMCW three times.
    """

    from witwin.radar.synthesis import (
        synthesize_fmcw_beat,
        synthesize_ofdm_cfr,
        synthesize_pulsed_echo,
    )

    owners = {
        "fmcw": synthesize_fmcw_beat,
        "ofdm": synthesize_ofdm_cfr,
        "pulsed": synthesize_pulsed_echo,
    }
    return owners[kind](to_synthesis(composed), spec)


def make_spec(kind: str):
    """The spec that goes with ``kind``, with the same no-default rule."""

    from . import multi_endpoint_driver as drv

    builders = {
        "fmcw": lambda: drv.make_spec(num_chirps=2),
        "ofdm": ofdm_spec,
        "pulsed": pulsed_spec,
    }
    return builders[kind]()


def chain_loss(
    spike,
    kind: str,
    spec,
    *,
    sites=None,
    transmitters=None,
    receivers=None,
    response=None,
    ad_mode: str = "none",
) -> torch.Tensor:
    """Core leaf -> propagation -> RCS -> two-way -> cube -> scalar loss.

    ``sum |cube|^2`` rather than a random-target inner product, because the
    magnitude loss is real-valued without a conjugation convention of its own
    and cannot accidentally test the fixture's target instead of the chain.

    ``include_delay_rate=False``: the fixture is static, so a rate would be
    exactly zero and would only add a second term for a finite difference to
    cancel against. The Doppler half of the chain is owned by the Phase-7
    kinematics tests and by the per-variable jvp tests at the operator level.
    """

    from . import multi_endpoint_driver as drv

    composed, _, _ = spike.frame(
        sites,
        drv.make_response() if response is None else response,
        transmitters=transmitters,
        receivers=receivers,
        ad_mode=ad_mode,
        include_delay_rate=False,
    )
    cube = synthesize(kind, composed, spec)
    return cube.abs().square().sum()


__all__ = [
    "chain_loss",
    "make_spec",
    "ofdm_spec",
    "pulsed_spec",
    "synthesize",
]
