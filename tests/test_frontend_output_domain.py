"""The same physical receiver must agree before/after a range transform."""

from dataclasses import replace

import pytest
import torch
from support import multi_endpoint_driver as drv
from support.simulate_fixture import fixture_radar, static_scene

from witwin.radar import Adc, Agc, Motion, Noise, PointTargets

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("motion", [Motion.adc(), Motion.adaptive()], ids=["adc", "adaptive"])
@pytest.mark.parametrize("hardware", ["adc", "agc", "thermal", "combined"])
def test_physical_receiver_precedes_range_transform(moving, motion, hardware):
    base = fixture_radar()
    origin = torch.tensor([[2.0, 0.2, 0.0]], device=base.device)
    velocity = torch.tensor([[0.7, 0.0, 0.0]], device=base.device)
    targets = PointTargets(
        positions=origin,
        amplitude=drv.FIXTURE_AMPLITUDE,
        phase=drv.FIXTURE_PHASE_RAD,
        trajectory=(lambda t: origin + t * velocity) if moving else None,
    )

    outputs = {}
    for domain in ("beat", "spectrum"):
        # The receive chain is part of the radar, so the two output domains are
        # two radars built from the same stages rather than one radar edited
        # between calls.
        radar = base.replace(
            waveform=replace(base.waveform, samples_per_chirp=16, chirps_per_frame=2, output=domain),
            adc=Adc(bits=3, full_scale=1e-7) if hardware in ("adc", "combined") else None,
            agc=Agc(target_rms=3e-8) if hardware in ("agc", "combined") else None,
            noise=Noise(bandwidth=1e5) if hardware in ("thermal", "combined") else None,
            lna_gain=3.0,
            seed=93,
        )
        result = radar.simulate(static_scene(), targets, times=(0.0,), los=True, reflections=0, motion=motion)
        assert result.output_domain == domain
        assert result.frame_synthesis().output_domain == domain
        outputs[domain] = result.cube
    expected = torch.fft.fft(outputs["beat"], dim=-1, norm="forward")
    torch.testing.assert_close(outputs["spectrum"], expected, rtol=2e-6, atol=1e-13)
