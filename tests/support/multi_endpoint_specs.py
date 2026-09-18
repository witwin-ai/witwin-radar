"""The three waveform specs on the multi-endpoint geometry.

One FMCW, one OFDM and one pulsed spec for ``support.multi_endpoint_driver``'s
frozen fixture (2 TX x 2 RX x 2 sites). The cross-waveform invariants and the
launch ledger read them from here so that both files measure the same specs.
"""

from __future__ import annotations

from witwin.radar.synthesis import FmcwSpec, OfdmSpec

from . import multi_endpoint_geometry as geo
from . import pulsed_grid

C0 = geo.C0_M_PER_S
F_REF_HZ = geo.REFERENCE_FREQUENCY_HZ

# The fixture's own radar config has slope 60.012 MHz/us, and at a 4.4 MSPS ADC
# that puts the longest round trip's beat tone at 3.6 MHz - above the 2.2 MHz
# Nyquist limit, where a beat-frequency estimate means nothing. The slope is
# the one number lowered here, and it is lowered to keep the whole fixture
# inside the band rather than to make anything pass.
FMCW_SLOPE_HZ_PER_S = 2.5e13
FMCW_SAMPLES = 256
FMCW_SAMPLE_RATE_HZ = 4.4e6
FMCW_SAMPLE_PERIOD_S = 1.0 / FMCW_SAMPLE_RATE_HZ
FMCW_CHIRP_PERIOD_S = (7.0 + 58.0) * 1e-6
FMCW_T_START_S = 6.0e-6
FMCW_NUM_TX = 2
FMCW_NUM_RX = 2
#: ``1 / (S N T_s)``: one FFT bin expressed as a delay.
FMCW_DELAY_RESOLUTION_S = 1.0 / (FMCW_SLOPE_HZ_PER_S * FMCW_SAMPLES * FMCW_SAMPLE_PERIOD_S)

OFDM_SUBCARRIERS = 64
OFDM_DF_HZ = 120.0e3
OFDM_CYCLIC_PREFIX_S = 2.0e-6
OFDM_MAX_DELAY_S = 1.0e-6

PULSED_MAX_DELAY_RATE = 2.0 * 12.0 / C0
#: ``1 / B``: one pulsed range cell expressed as a delay.
PULSED_DELAY_RESOLUTION_S = 1.0 / pulsed_grid.BANDWIDTH_HZ


def fmcw_spec(num_chirps: int = 4) -> FmcwSpec:
    return FmcwSpec(
        num_samples=FMCW_SAMPLES,
        num_chirps=num_chirps,
        sample_period_s=FMCW_SAMPLE_PERIOD_S,
        chirp_period_s=FMCW_CHIRP_PERIOD_S,
        slope_hz_per_s=FMCW_SLOPE_HZ_PER_S,
        t_start_s=FMCW_T_START_S,
        reference_frequency_hz=F_REF_HZ,
        carrier_hz=0.0,
        carrier_rate_hz=F_REF_HZ,
        num_tx=FMCW_NUM_TX,
        num_rx=FMCW_NUM_RX,
        output_domain="beat",
    )


def ofdm_spec(num_symbols: int = 4) -> OfdmSpec:
    return OfdmSpec(
        num_subcarriers=OFDM_SUBCARRIERS,
        num_symbols=num_symbols,
        subcarrier_spacing_hz=OFDM_DF_HZ,
        cyclic_prefix_s=OFDM_CYCLIC_PREFIX_S,
        reference_frequency_hz=F_REF_HZ,
        max_expected_delay_s=OFDM_MAX_DELAY_S,
        carrier_hz=0.0,
        carrier_rate_hz=F_REF_HZ,
    )


def pulsed_spec(num_pulses: int = 4):
    return pulsed_grid.reference_spec(
        num_pulses=num_pulses,
        num_samples=512,
        reference_frequency_hz=F_REF_HZ,
        carrier_rate_hz=F_REF_HZ,
        max_expected_delay_rate=PULSED_MAX_DELAY_RATE,
    )
