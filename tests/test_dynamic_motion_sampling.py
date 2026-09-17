"""Independent geometric phase oracles for the scene-driven motion sampler."""

import math

import pytest
import torch
from test_phase11_simulate_entry import _radar, _response, _static_scene

from witwin.radar.propagation import Kinematics
from witwin.radar.simulation import ScatterSitePolicy

pytestmark = pytest.mark.gpu


class Orbit:
    def __init__(self, device, *, radius=0.12, frequency=40.0):
        self.device, self.radius, self.frequency = device, radius, frequency

    def at(self, t):
        angle = 2 * math.pi * self.frequency * t
        r, w = self.radius, 2 * math.pi * self.frequency
        return Kinematics(
            torch.tensor([[2 + r * math.cos(angle), r * math.sin(angle), 0]], device=self.device),
            torch.tensor([[-r * w * math.sin(angle), r * w * math.cos(angle), 0]], device=self.device),
        )


def test_orbit_phase_and_tdm_match_independent_geometric_delay():
    radar = _radar()
    trajectory = Orbit(radar.device)
    result = radar.simulate(
        _static_scene(),
        times=(0.0,),
        response=_response(radar),
        sites=ScatterSitePolicy.explicit(trajectory.at(0).positions_m, trajectory=trajectory),
        components=frozenset({"los"}),
        max_depth=0,
        motion_sampling="chirp",
    )
    spec = radar.system_config.waveform_spec()
    # Invert only the declared spectrum. Test the first ADC sample of each TX.
    beat = torch.fft.ifft(result.cube[0], dim=-1)
    for tx in range(spec.num_tx):
        for rx in range(spec.num_rx):
            times = [result.sample_times_s[0][c * spec.num_tx + tx] for c in range(spec.num_chirps)]
            positions = torch.cat([trajectory.at(t).positions_m for t in times]).double()
            delay = (
                (positions - radar.tx_pos[tx].double()).norm(dim=-1)
                + (positions - radar.rx_pos[rx].double()).norm(dim=-1)
            ) / 299792458.0
            cycles = spec.reference_frequency_hz * delay + spec.slope_hz_per_s * delay * (spec.t_start_s - delay / 2)
            phase_steps = torch.angle(beat[tx, rx, 1:, 0] * beat[tx, rx, :-1, 0].conj())
            expected = torch.angle(torch.exp(2j * math.pi * (cycles[1:] - cycles[:-1])))
            torch.testing.assert_close(phase_steps.double(), expected, atol=2e-3, rtol=0)
    assert result.path_set_complete
    assert result.discovery_count == spec.num_chirps * spec.num_tx


def test_dynamic_default_discovers_endpoint_born_reflections():
    radar = _radar()

    class Crossing:
        def at(self, t):
            return Kinematics(
                torch.tensor([[2.0, 2.4 - 1800 * t, 0.0]], device=radar.device),
                torch.tensor([[0.0, -1800.0, 0.0]], device=radar.device),
            )

    trajectory = Crossing()
    args = {
        "times": (0.0, 0.001),
        "motion_sampling": "chirp",
        "response": _response(radar),
        "sites": ScatterSitePolicy.explicit(trajectory.at(0).positions_m, trajectory=trajectory),
    }
    complete = radar.simulate(_static_scene(), **args)
    held = radar.simulate(_static_scene(), motion_event_period_frames=10, **args)
    assert complete.path_set_complete and not held.path_set_complete
    assert complete.discovery_count > held.discovery_count == 1
    assert not torch.allclose(complete.cube, held.cube, atol=0, rtol=1e-3)


@pytest.mark.parametrize("times", [(0.0, float("nan")), (1.0, 0.0), (0.0, 0.0)])
def test_invalid_frame_times_are_refused(times):
    radar = _radar()
    with pytest.raises(ValueError, match="finite|increasing"):
        radar.simulate(_static_scene(), times=times, response=_response(radar))
