"""One physical target, three waveforms, every peak on an exact bin.

The plan's cross-waveform criterion asks that FMCW, OFDM and pulsed range and
Doppler outputs agree on an analytic target. This module is the geometry and the
three waveform grids that make that a statement about EXACT bins rather than
about tolerances: one round-trip delay lands on an integer FFT bin, an integer
CIR sample and an integer matched-filter lag at once, and one round-trip delay
RATE lands on the same signed Doppler bin in all three.

Nothing here is searched. The site position is solved in closed form and the
three grids are solved from the delay it produces.

**The site.** ``TX_A`` and ``RX_A`` are two points 0.15 m apart, so the locus of
sites with a fixed round-trip path length ``L = d_in + d_out`` is the ellipse
with those two as foci. Taking the point on its minor axis gives

    ``site = ((x_tx + x_rx) / 2, sqrt((L/2)^2 - (d_tx_rx/2)^2), 0)``

which is one square root, not a search. At the length used here the site sits at
``(0.075, 2.145, 0)``: in front of the wall plane, so both lines of sight are
clear, and far enough inside the facet that the ``TX_A`` image-source reflection
also exists, which is what keeps this fixture a multi-row one.

**The delay.** ``L`` is chosen so that the FMCW beat bin is an integer:
``tau = m f_s / (S N)``. Everything else is then solved FROM ``tau``:

* OFDM: ``T_s = tau / m_cir`` fixes ``N_sc df``, hence ``df``;
* pulsed: ``T_s = tau / m_lag`` fixes the ADC grid.

**The delay rate.** A closing speed is chosen so the FMCW Doppler bin is an
integer, ``q = f_ref |tau_rate| T_slot C``, and the OFDM symbol period and the
pulsed PRI are then solved so their coherent processing intervals MATCH the
FMCW frame's. All three therefore share one velocity bin width and put the same
target on the same signed bin ``+q``, which is the sharpest available form of
the cross-waveform criterion.

**Recorded deviation: the pulsed exact-lag case uses an LFM pulse, not a rect.**
The design asks for a rectangular pulse. Measured on this fixture, a rect pulse
cannot carry an exact-lag assertion: its matched-filter output is a triangle of
half-width ``T_p``, so adjacent lag bins differ by ``1 / M_p`` - 0.2757 dB at
``M_p = 32``, measured - and the pulse support is HALF OPEN, so a delay that
float32 rounds a part in 1e8 ABOVE ``m T_s`` drops the first received sample and
moves the triangle's apex to ``m + 1``. That is a property of the rectangular
pulse and of the half-open support the spec deliberately pins, not of the
processing chain, and asserting an exact bin through it would be asserting the
last bit of a float32 delay. The LFM's compressed main lobe is ``1 / B`` wide -
2.5 samples here - so its argmax is decided by the delay and not by a rounding,
and the same one missing sample costs a measured 3.1 percent of the peak
(``31 / 32``), which is asserted rather than hidden.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from . import multi_endpoint_geometry as geo


C0 = geo.C0_M_PER_S
F_REF_HZ = geo.REFERENCE_FREQUENCY_HZ

# --- FMCW grid: the fixture's own radar block, in SI -----------------------
FMCW_SLOPE_HZ_PER_S = 60.012e12
FMCW_SAMPLE_RATE_HZ = 4.4e6
FMCW_SAMPLES = 256
FMCW_CHIRP_PERIOD_S = (7.0 + 58.0) * 1e-6
FMCW_T_START_S = 6.0e-6
FMCW_NUM_TX = 2
FMCW_NUM_RX = 2
FMCW_CHIRPS = 8

#: The FMCW range bin the target is placed on. 50 of 256 keeps the beat tone at
#: 0.86 MHz, comfortably inside the 2.2 MHz Nyquist limit of the fixture's ADC,
#: and far enough from bin 0 that a DC term could not be mistaken for it.
FMCW_RANGE_BIN = 50

#: ``tau = m f_s / (S N)``: the delay that lands exactly on that bin.
TAU_S = FMCW_RANGE_BIN * FMCW_SAMPLE_RATE_HZ / (
    FMCW_SLOPE_HZ_PER_S * FMCW_SAMPLES
)
ROUND_TRIP_M = C0 * TAU_S
RANGE_M = 0.5 * ROUND_TRIP_M

#: The signed Doppler bin every waveform must put the target on. 2 of 8 is
#: inside the FMCW frame's Nyquist edge at 4, and non-zero in both senses.
DOPPLER_BIN = 2

FMCW_SLOT_PERIOD_S = FMCW_CHIRP_PERIOD_S * FMCW_NUM_TX
#: ``q = f_ref |tau_rate| T_slot C`` solved for the speed, then
#: ``tau_rate = -2 v / c`` because the target is CLOSING.
CLOSING_SPEED_MPS = 0.5 * C0 * DOPPLER_BIN / (
    F_REF_HZ * FMCW_SLOT_PERIOD_S * FMCW_CHIRPS
)
DELAY_RATE = -2.0 * CLOSING_SPEED_MPS / C0

# --- OFDM grid solved from tau ---------------------------------------------
OFDM_SUBCARRIERS = 64
OFDM_CIR_SAMPLE = 4
OFDM_WAVEFORM_SAMPLE_PERIOD_S = TAU_S / OFDM_CIR_SAMPLE
OFDM_SPACING_HZ = 1.0 / (OFDM_SUBCARRIERS * OFDM_WAVEFORM_SAMPLE_PERIOD_S)
OFDM_SYMBOLS = 2048
#: The symbol period that puts the same closing speed on Doppler bin ``+q``.
OFDM_SYMBOL_PERIOD_S = DOPPLER_BIN / (
    F_REF_HZ * abs(DELAY_RATE) * OFDM_SYMBOLS
)
OFDM_CYCLIC_PREFIX_S = OFDM_SYMBOL_PERIOD_S - 1.0 / OFDM_SPACING_HZ
#: A configured range window, comfortably above ``tau`` and below the prefix.
OFDM_MAX_DELAY_S = 5.0e-8

# --- Pulsed grid solved from tau -------------------------------------------
PULSED_LAG_SAMPLE = 4
PULSED_SAMPLE_PERIOD_S = TAU_S / PULSED_LAG_SAMPLE
PULSED_SAMPLES = 128
PULSED_WIDTH_S = 32 * PULSED_SAMPLE_PERIOD_S
#: ``f_s = 2.5 B``, the ratio ``support/pulsed_grid.py`` records as the minimum
#: for the DISCRETE matched-filter sum to equal the continuous integral.
PULSED_BANDWIDTH_HZ = 1.0 / (2.5 * PULSED_SAMPLE_PERIOD_S)
PULSED_PULSES = 64
PULSED_PRI_S = DOPPLER_BIN / (F_REF_HZ * abs(DELAY_RATE) * PULSED_PULSES)


def site_position_m(round_trip_m: float = ROUND_TRIP_M) -> tuple[float, float, float]:
    """The point on the ``TX_A`` / ``RX_A`` ellipse's minor axis, in closed form."""

    focus = 0.5 * geo.distance_m(geo.TX_A_POSITION_M, geo.RX_A_POSITION_M)
    semi_major = 0.5 * float(round_trip_m)
    return (
        0.5 * (geo.TX_A_POSITION_M[0] + geo.RX_A_POSITION_M[0]),
        math.sqrt(semi_major * semi_major - focus * focus),
        0.0,
    )


SITE_POSITION_M = site_position_m()
SITE_STABLE_ID = geo.SITE_P_STABLE_ID


def closing_velocity_m_per_s(
    site: tuple[float, float, float] = SITE_POSITION_M,
    delay_rate: float = DELAY_RATE,
) -> tuple[float, float, float]:
    """The site velocity that produces exactly ``delay_rate``, in closed form.

    The rate is linear in the velocity, so evaluating the fixture's own float64
    rate formula at unit speed along ``-y`` and dividing is exact. ``-y`` is
    chosen because the site sits at ``+y`` between two endpoints on the ``x``
    axis, so moving along ``-y`` shortens BOTH legs and the target is
    unambiguously closing - which is the only geometry in which a Doppler SIGN
    test cannot pass by accident.
    """

    direction = (0.0, -1.0, 0.0)
    unit = geo.leg_delay_rate_s_per_s(
        site, geo.TX_A_POSITION_M, "los", direction
    ) + geo.leg_delay_rate_s_per_s(site, geo.RX_A_POSITION_M, "los", direction)
    scale = float(delay_rate) / unit
    return tuple(scale * value for value in direction)


SITE_VELOCITY_M_PER_S = closing_velocity_m_per_s()

#: The frame-invariant name of the one row every assertion is made about: the
#: reflection-free round trip ``TX_A -> site -> RX_A``.
TARGET_KEY = (
    geo.TX_A_STABLE_ID,
    SITE_STABLE_ID,
    geo.RX_A_STABLE_ID,
    "los",
    "los",
)


def fmcw_spec(num_chirps: int = FMCW_CHIRPS):
    from witwin.radar.synthesis import FmcwSpec

    return FmcwSpec(
        num_samples=FMCW_SAMPLES,
        num_chirps=num_chirps,
        sample_period_s=1.0 / FMCW_SAMPLE_RATE_HZ,
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


def ofdm_spec(num_symbols: int = OFDM_SYMBOLS):
    from witwin.radar.synthesis import OfdmSpec

    return OfdmSpec(
        num_subcarriers=OFDM_SUBCARRIERS,
        num_symbols=num_symbols,
        subcarrier_spacing_hz=OFDM_SPACING_HZ,
        cyclic_prefix_s=OFDM_CYCLIC_PREFIX_S,
        reference_frequency_hz=F_REF_HZ,
        max_expected_delay_s=OFDM_MAX_DELAY_S,
        carrier_hz=0.0,
        carrier_rate_hz=F_REF_HZ,
    )


def pulsed_spec(num_pulses: int = PULSED_PULSES, *, pulse_kind: str | None = None):
    from witwin.radar.synthesis.assembly import PULSE_KIND_LFM, PulsedSpec

    kind = PULSE_KIND_LFM if pulse_kind is None else pulse_kind
    bandwidth = (
        PULSED_BANDWIDTH_HZ if kind == PULSE_KIND_LFM else 1.0 / PULSED_WIDTH_S
    )
    return PulsedSpec(
        num_pulses=num_pulses,
        num_samples=PULSED_SAMPLES,
        sample_period_s=PULSED_SAMPLE_PERIOD_S,
        pri_s=PULSED_PRI_S,
        range_gate_start_s=0.0,
        pulse_kind=kind,
        pulse_width_s=PULSED_WIDTH_S,
        bandwidth_hz=bandwidth,
        reference_frequency_hz=F_REF_HZ,
        max_expected_delay_rate=4.0 * abs(DELAY_RATE),
        carrier_hz=0.0,
        carrier_rate_hz=F_REF_HZ,
    )


def array_spec():
    """The fixture's 2 TX x 2 RX front end, as a ``SensorArraySpec``."""

    from witwin.radar import RadarConfig
    from witwin.radar.sensors import SensorArraySpec

    return SensorArraySpec.from_radar_config(
        RadarConfig.from_dict(dict(geo.FIXTURE_RADAR_CONFIG))
    )


def make_spike():
    """One compiled scene with the analytically placed site."""

    from . import multi_endpoint_driver as drv

    return drv.MultiEndpointSpike(sites=((SITE_STABLE_ID, SITE_POSITION_M),))


def target_row(spike, composed) -> int:
    """Which composed row is ``TX_A -> site -> RX_A`` with no reflection."""

    from . import multi_endpoint_driver as drv

    return drv.composed_keys(spike, composed).index(TARGET_KEY)


def isolate(batch, row: int):
    """The same batch with every row but ``row`` masked on ``row_valid``.

    Row identity, row order, dtype, device and storage are untouched: only the
    validity mask changes, exactly as the Phase-6 cross-waveform file does it.
    The fixture's closest composed rows differ by picoseconds, so a peak finder
    handed more than one would report neither.
    """

    mask = torch.zeros(batch.path_count, dtype=torch.bool, device=batch.device)
    mask[row] = True
    existing = batch.row_valid
    combined = mask if existing is None else (mask & existing)
    return replace(batch, row_valid=combined.contiguous())


def moving_frame(spike, velocity=None):
    """One composed frame carrying the exact ``delay_rate`` of ``velocity``.

    The rate reaches the batch through a forward-AD tangent on the site
    position, which is the production seam: a rate rebuilt from Python values
    would be a number this fixture invented rather than one the propagation
    consumer produced.
    """

    import torch.autograd.forward_ad as forward_ad

    positions = spike.site_tensor()
    tangent = torch.tensor(
        [SITE_VELOCITY_M_PER_S if velocity is None else velocity],
        dtype=torch.float32,
        device=positions.device,
    )
    with forward_ad.dual_level():
        composed, _, _ = spike.frame(
            forward_ad.make_dual(positions, tangent), ad_mode="jvp"
        )
        return replace(
            composed,
            total_delay_s=composed.total_delay_s.clone(),
            delay_rate=composed.delay_rate.clone(),
            complex_transfer_ref=composed.complex_transfer_ref.clone(),
        )


__all__ = [
    "C0",
    "CLOSING_SPEED_MPS",
    "DELAY_RATE",
    "DOPPLER_BIN",
    "FMCW_CHIRPS",
    "FMCW_RANGE_BIN",
    "F_REF_HZ",
    "OFDM_CIR_SAMPLE",
    "PULSED_LAG_SAMPLE",
    "RANGE_M",
    "ROUND_TRIP_M",
    "SITE_POSITION_M",
    "SITE_STABLE_ID",
    "SITE_VELOCITY_M_PER_S",
    "TARGET_KEY",
    "TAU_S",
    "array_spec",
    "closing_velocity_m_per_s",
    "fmcw_spec",
    "isolate",
    "make_spike",
    "moving_frame",
    "ofdm_spec",
    "pulsed_spec",
    "site_position_m",
    "target_row",
]
