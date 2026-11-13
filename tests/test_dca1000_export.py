"""DCA1000 raw ADC export and reader (F-8): the documented layouts, value by value, and round trips."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from witwin.radar import Adc
from witwin.radar.rawdata import read_dca1000, write_dca1000


def _codes_cube(frames: int, num_tx: int, num_rx: int, chirps: int, samples: int) -> torch.Tensor:
    """A cube whose every component is a distinct integer, so a layout error cannot cancel."""

    count = frames * num_tx * num_rx * chirps * samples
    real = torch.arange(count, dtype=torch.float32).reshape(frames, num_tx, num_rx, chirps, samples)
    return torch.complex(real, -real - 1.0)


def test_the_two_lane_complex_layout_value_by_value(tmp_path):
    """One frame, 2 TX x 2 RX x 2 chirps x 4 samples, against the layout written out by hand."""

    cube = _codes_cube(1, 2, 2, 2, 4)
    path = write_dca1000(tmp_path / "adc.bin", cube, scale=1.0)
    stream = np.fromfile(path, dtype="<i2")
    expected = []
    for chirp in range(2):
        for tx in range(2):  # TDM slot c * num_tx + t: transmitter fastest
            for rx in range(2):
                values = cube[0, tx, rx, chirp]
                for n in range(0, 4, 2):
                    expected += [values[n].real, values[n + 1].real, values[n].imag, values[n + 1].imag]
    assert stream.tolist() == [int(value) for value in expected]


def test_the_four_lane_complex_layout_value_by_value(tmp_path):
    cube = _codes_cube(1, 1, 3, 2, 3)
    stream = np.fromfile(write_dca1000(tmp_path / "adc.bin", cube, scale=1.0, layout="xwr14xx"), dtype="<i2")
    expected = []
    for chirp in range(2):
        for n in range(3):
            expected += [cube[0, 0, rx, chirp, n].real for rx in range(3)]
            expected += [cube[0, 0, rx, chirp, n].imag for rx in range(3)]
    assert stream.tolist() == [int(value) for value in expected]


@pytest.mark.parametrize("layout", ["xwr16xx", "xwr14xx"])
@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("iq_swap", [False, True])
def test_write_then_read_is_the_quantised_cube(tmp_path, layout, real, iq_swap):
    generator = torch.Generator().manual_seed(3)
    cube = torch.randn((2, 3, 4, 8, 16), generator=generator, dtype=torch.complex64)
    scale = 1000.0
    path = write_dca1000(tmp_path / "adc.bin", cube, scale=scale, layout=layout, real=real, iq_swap=iq_swap)
    back = read_dca1000(
        path, num_tx=3, num_rx=4, chirps=8, samples=16, layout=layout, real=real, iq_swap=iq_swap, scale=scale
    )
    expected = torch.round(cube.real * scale) / scale
    if real:
        assert back.dtype == torch.float32 and torch.equal(back, expected)
    else:
        expected = torch.complex(expected, torch.round(cube.imag * scale) / scale)
        assert back.dtype == torch.complex64 and torch.equal(back, expected)


def test_an_adc_quantised_cube_is_written_as_its_exact_codes(tmp_path):
    adc = Adc(bits=12, full_scale=0.5)
    codes = torch.randint(-2048, 2048, (1, 1, 2, 4, 8), generator=torch.Generator().manual_seed(1))
    cube = torch.complex(codes.float() * adc.step, -codes.float() * adc.step)
    path = write_dca1000(tmp_path / "adc.bin", cube, adc=adc)
    back = read_dca1000(path, num_tx=1, num_rx=2, chirps=4, samples=8)
    assert torch.equal(back.real, codes.float()) and torch.equal(back.imag, -codes.float())


def test_a_result_must_be_a_beat_domain_fmcw_cube(tmp_path):
    spectrum = SimpleNamespace(cube=torch.zeros((1, 1, 1, 2, 4), dtype=torch.complex64), kind="fmcw")
    spectrum.output_domain = "spectrum"
    with pytest.raises(ValueError, match="Fmcw\\(output='beat'\\)"):
        write_dca1000(tmp_path / "adc.bin", spectrum, scale=1.0)


def test_codes_outside_int16_are_refused_unless_clipped(tmp_path):
    cube = torch.full((1, 1, 1, 1, 2), 40000.0 + 0j, dtype=torch.complex64)
    with pytest.raises(ValueError, match="outside int16"):
        write_dca1000(tmp_path / "adc.bin", cube, scale=1.0)
    path = write_dca1000(tmp_path / "adc.bin", cube, scale=1.0, clip=True)
    assert np.fromfile(path, dtype="<i2").tolist() == [32767, 32767, 0, 0]


def test_exactly_one_scale_source_and_a_whole_file(tmp_path):
    cube = torch.zeros((1, 1, 1, 1, 2), dtype=torch.complex64)
    with pytest.raises(ValueError, match="exactly one"):
        write_dca1000(tmp_path / "adc.bin", cube)
    path = write_dca1000(tmp_path / "adc.bin", cube, scale=1.0)
    with pytest.raises(ValueError, match="whole number"):
        read_dca1000(path, num_tx=1, num_rx=1, chirps=1, samples=4)
