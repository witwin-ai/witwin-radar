"""The same physical receiver must agree before/after a range transform."""

from dataclasses import replace

import pytest
import torch
from test_phase11_simulate_entry import _radar, _response, _static_scene

from witwin.radar.frontend import AdcSpec, AgcSpec, FrontendChain, FrontendSpec, LnaSpec, NoiseSpec, SeedSpec
from witwin.radar.propagation import Kinematics
from witwin.radar.simulation import ScatterSitePolicy

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("hardware", ["adc", "agc", "thermal", "combined"])
def test_physical_receiver_precedes_range_transform(moving, hardware):
    radar = _radar()
    radar.system_config = replace(
        radar.system_config, waveform=replace(radar.system_config.waveform, adc_samples=16, chirp_per_frame=2)
    )
    hardware_spec = FrontendSpec(
        adc=AdcSpec(bits=3, full_scale=1e-7) if hardware in ("adc", "combined") else None,
        agc=AgcSpec(target_rms=3e-8) if hardware in ("agc", "combined") else None,
        noise=NoiseSpec(bandwidth_hz=1e5) if hardware in ("thermal", "combined") else None,
        lna=LnaSpec(gain_db=3.0),
        seed=SeedSpec(93),
    )
    radar.frontend = FrontendChain(hardware_spec)
    origin = torch.tensor([[2.0, 0.2, 0.0]], device=radar.device)

    class Linear:
        def at(self, t):
            v = torch.tensor([[0.7, 0.0, 0.0]], device=radar.device)
            return Kinematics(origin + t * v, v)

    outputs = {}
    for domain in ("beat", "spectrum"):
        radar.system_config = replace(
            radar.system_config, waveform=replace(radar.system_config.waveform, output_domain=domain)
        )
        result = radar.simulate(
            _static_scene(),
            times=(0.0,),
            response=_response(radar),
            sites=ScatterSitePolicy.explicit(origin, trajectory=Linear() if moving else None),
            components=frozenset({"los"}),
            max_depth=0,
        )
        assert result.output_domain == domain
        assert result.frame_synthesis().output_domain == domain
        outputs[domain] = result.cube
    expected = torch.fft.fft(outputs["beat"], dim=-1, norm="forward")
    torch.testing.assert_close(outputs["spectrum"], expected, rtol=2e-6, atol=1e-13)
