"""The receiver frontend: one chain, one order, one ADC.

Work item 5's "enable order, units, and seeds" lands here. The physical signal
chain is

    synthesis output [sqrt(W)] -> port -> phase -> thermal -> LNA -> AGC -> ADC

and it is fixed in :class:`~witwin.radar.frontend.chain.FrontendChain` rather
than in a caller. Thermal noise is INPUT-REFERRED, so it is added before the
gain; a chain that let the caller reorder those two is wrong by ``g_lna^2`` in
output noise power, silently.

Units are physical rather than raw standard deviations: a noise figure and a
system temperature and an explicit bandwidth, not a ``std``. Seeds are per
stage, so toggling one stage leaves every other stage bit-identical.
"""

from .chain import (
    FrontendChain,
    FrontendDiagnostics,
    FrontendOutput,
    frontend_block_size,
)
from .contracts import (
    AGC_MODES,
    AGC_MODE_GLOBAL,
    AGC_MODE_PER_RX,
    BOLTZMANN_J_PER_K,
    FRONTEND_STAGE_ORDER,
    REFERENCE_TEMPERATURE_K,
    STAGE_PHASE_NOISE,
    STAGE_THERMAL_NOISE,
    AdcSpec,
    AgcSpec,
    FrontendSpec,
    LnaSpec,
    NoiseSpec,
    PortSpec,
    SeedSpec,
)

__all__ = [
    "AGC_MODES",
    "AGC_MODE_GLOBAL",
    "AGC_MODE_PER_RX",
    "BOLTZMANN_J_PER_K",
    "FRONTEND_STAGE_ORDER",
    "REFERENCE_TEMPERATURE_K",
    "STAGE_PHASE_NOISE",
    "STAGE_THERMAL_NOISE",
    "AdcSpec",
    "AgcSpec",
    "FrontendChain",
    "FrontendDiagnostics",
    "FrontendOutput",
    "FrontendSpec",
    "LnaSpec",
    "NoiseSpec",
    "PortSpec",
    "SeedSpec",
    "frontend_block_size",
]
