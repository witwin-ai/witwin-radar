"""F-5: DDM and BPM transmit multiplexing and their demultiplexer (R-ADR-048).

The oracle is the scene itself: a still point target off boresight seen by a
2 x 4 array under TDM, DDM and BPM. After :func:`~witwin.radar.processing.demultiplex`
the eight virtual channels at the target's range bin must carry the same
relative phases under all three schemes, and those phases must step by
``pi sin(theta)`` per half-wavelength element of the virtual array, the plane
wave the geometry implies.
"""

from __future__ import annotations

import math

import pytest
import torch
from support import oracle_scenes as sc

from witwin.radar import Fmcw, PointTargets
from witwin.radar.processing import demultiplex

pytestmark = pytest.mark.gpu

RANGE_M = 20.0
AZIMUTH = math.radians(17.0)


def _radar(mimo: str, chirps: int):
    waveform = Fmcw(
        slope=30e12,
        sample_rate=10e6,
        samples_per_chirp=64,
        chirps_per_frame=chirps,
        adc_start=1e-6,
        idle=5e-6,
        ramp_end=8e-6,
        output="beat",
        mimo=mimo,
    )
    from witwin.radar import Radar

    return Radar(
        carrier=sc.CARRIER_HZ,
        waveform=waveform,
        tx=((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
        rx=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
        antenna_unit="half_wavelength",
        power=10.0,
        look_at=sc.LOOK_AT,
        up=sc.UP,
    )


def _virtual(mimo: str, chirps: int):
    radar = _radar(mimo, chirps)
    # Local right is world +z for this pose, so the azimuth is in the x-z plane.
    target = PointTargets(positions=[[RANGE_M * math.cos(AZIMUTH), 0.0, RANGE_M * math.sin(AZIMUTH)]], rcs=1.0)
    result = radar.simulate(sc.empty_scene(), target, times=(0.0,), los=True, reflections=0)
    physical = result.frame(0).processing_cube()
    virtual = demultiplex(physical)
    profile = torch.fft.fft(virtual.data.to(torch.complex128), dim=-1)
    peak = int(profile.abs().sum(dim=(0, 1, 2)).argmax())
    vector = profile[..., peak].mean(dim=-1).reshape(-1)
    return result, physical, virtual, vector / vector[0]


def test_ddm_and_bpm_publish_one_transmit_channel_and_demultiplex_to_the_virtual_array():
    result, physical, virtual, _ = _virtual("ddm", 32)
    assert result.axes.mimo == "ddm" and result.cube.shape[1:3] == (1, 4)
    assert physical.data.shape == (1, 4, 32, 64)
    assert virtual.data.shape == (2, 4, 16, 64)
    assert virtual.axes.demultiplexed and virtual.axes.doppler_bin_count == 16
    assert virtual.axes.slow_time_period_s == pytest.approx(2 * result.axes.slow_time_period_s)
    _, _, bpm, _ = _virtual("bpm", 32)
    assert bpm.data.shape == (2, 4, 16, 64)


def test_ddm_and_bpm_angles_match_tdm_on_the_same_scene():
    _, _, _, tdm = _virtual("tdm", 16)
    _, _, _, ddm = _virtual("ddm", 32)
    _, _, _, bpm = _virtual("bpm", 32)
    assert float((ddm - tdm).abs().max()) < 1e-3
    assert float((bpm - tdm).abs().max()) < 1e-3
    # The plane wave the geometry implies: virtual element tx * 4 + rx sits at
    # that many half wavelengths, so consecutive elements differ by pi sin(theta).
    step = torch.angle(tdm[1:] / tdm[:-1])
    assert float(step.abs().mean()) == pytest.approx(math.pi * math.sin(AZIMUTH), rel=2e-3)
    # The residual spread is the float32 rounding of the element positions.
    assert float(step.std()) < 5e-3


def test_a_tdm_cube_demultiplexes_to_itself_and_coupling_is_refused_under_ddm():
    _, physical, virtual, _ = _virtual("tdm", 4)
    assert virtual is physical
    from witwin.radar import Coupling

    radar = _radar("ddm", 4).replace(coupling=Coupling(tx=((1.0, 0.1), (0.1, 1.0))))
    with pytest.raises(NotImplementedError, match="DDM and BPM"):
        radar.simulate(
            sc.empty_scene(),
            PointTargets(positions=[[RANGE_M, 0.0, 0.0]], rcs=1.0),
            times=(0.0,),
            los=True,
            reflections=0,
        )
