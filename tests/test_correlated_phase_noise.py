"""Independent interval-covariance and homodyne-suppression noise oracles."""

from dataclasses import replace

import pytest
import torch
from support.reference_frontend import single_sideband_psd
from test_phase11_simulate_entry import _radar, _response, _static_scene

from witwin.radar.frontend import FrontendChain, FrontendSpec, NoiseSpec, PortSpec
from witwin.radar.simulation import ScatterSitePolicy

pytestmark = pytest.mark.gpu


def noise():
    return NoiseSpec(phase_noise_dbc_per_hz=-80.0, phase_offset_hz=1e5, phase_sample_rate_hz=5e6)


def test_delay_variance_covariance_and_zero_delay_cancellation():
    model = noise()
    times = torch.arange(16384, device="cuda", dtype=torch.float64) + 0.5
    a = model.phase_difference(times, torch.full_like(times, 1e-7), seed_base=7).double()
    b = model.phase_difference(times, torch.full_like(times, 4e-7), seed_base=7).double()
    q = 4 * torch.pi**2 * 1e10 * 1e-8
    assert float(a.var()) == pytest.approx(q * 1e-7, rel=0.04)
    assert float(b.var()) == pytest.approx(q * 4e-7, rel=0.04)
    covariance = ((a - a.mean()) * (b - b.mean())).mean()
    assert float(covariance) == pytest.approx(q * 1e-7, rel=0.06)
    zero = model.phase_difference(times, torch.zeros_like(times), seed_base=7)
    assert torch.count_nonzero(zero) == 0


def test_time_gaps_query_order_and_interval_additivity():
    model = noise()
    times = torch.arange(8192, device="cuda", dtype=torch.float64) + 0.25
    # Dyadic endpoints are exactly representable even thousands of seconds
    # from zero: the additivity oracle must compare identical physical times.
    small, large = 2.0**-20, 2.0**-10
    short = model.phase_difference(times, torch.full_like(times, small), seed_base=19)
    gap = model.phase_difference(times, torch.full_like(times, large), seed_base=19)
    assert float(gap.var() / short.var()) == pytest.approx(1024.0, rel=0.06)
    reverse = model.phase_difference(times.flip(0), torch.full_like(times, large), seed_base=19)
    torch.testing.assert_close(reverse.flip(0), gap, rtol=0, atol=0)
    after_short = model.phase_difference(times - small, torch.full_like(times, large - small), seed_base=19)
    torch.testing.assert_close(short + after_short, gap, atol=2e-7, rtol=2e-6)


def test_homodyne_phase_noise_psd_has_delay_cancellation_transfer():
    model = noise()
    fs, delay = 5e6, 1e-6
    times = torch.arange(1 << 19, device="cuda", dtype=torch.float64) / fs + 0.5
    phase = model.phase_difference(times, torch.full_like(times, delay), seed_base=53)
    frequency, measured = single_sideband_psd(phase, sample_rate_hz=fs, segment=4096)
    for target in (25000, 50000, 100000, 200000):
        index = int((frequency - target).abs().argmin())
        f = frequency[index - 4 : index + 5]
        expected = 1e-8 * (1e5 / f) ** 2 * 4 * torch.sin(torch.pi * f * delay) ** 2
        ratio = measured[index - 4 : index + 5].mean() / expected.mean()
        assert abs(float(10 * torch.log10(ratio))) < 1.0


def test_explicit_receiver_timestamps_include_idle_time():
    chain = FrontendChain(FrontendSpec(port=PortSpec(1), noise=noise()))
    times = torch.tensor([0, 1e-6, 100e-6, 101e-6], device="cuda", dtype=torch.float64)
    signal = torch.ones(4, device="cuda", dtype=torch.complex64)
    output = chain.apply(signal, times_s=times)
    expected = noise().phase_difference(times, times, seed_base=chain.spec.seed.seed_base)
    torch.testing.assert_close(output.diagnostics.phase_rad, expected, rtol=0, atol=0)


def test_delays_refuse_nonexistent_brownian_time_derivative():
    times = torch.tensor([1.0], device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="oscillator"):
        noise().phase_difference(times, torch.tensor([1e-6], device="cuda", requires_grad=True), seed_base=0)


def test_scene_path_noise_preserves_output_domain_and_frame_time():
    radar = _radar()
    radar.system_config = replace(
        radar.system_config, waveform=replace(radar.system_config.waveform, adc_samples=4, chirp_per_frame=1)
    )
    radar.frontend = FrontendChain(FrontendSpec(noise=noise()))
    outputs = {}
    for domain in ("beat", "spectrum"):
        radar.system_config = replace(
            radar.system_config, waveform=replace(radar.system_config.waveform, output_domain=domain)
        )
        outputs[domain] = radar.simulate(
            _static_scene(),
            times=(0.0, 0.1),
            response=_response(radar),
            sites=ScatterSitePolicy.explicit(torch.tensor([[2.0, 0.2, 0.0]], device=radar.device)),
            components=frozenset({"los"}),
            max_depth=0,
        ).cube
    torch.testing.assert_close(outputs["spectrum"], torch.fft.fft(outputs["beat"], dim=-1, norm="forward"))
    assert not torch.equal(outputs["beat"][0], outputs["beat"][1])
