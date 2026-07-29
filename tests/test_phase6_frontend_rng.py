"""Reproducibility: per-stage streams, and a realisation the schedule cannot move.

Two properties, and both of them are about what a DIFFERENTIAL measurement
means. If enabling phase noise shifts the thermal realisation, then comparing a
run with phase noise against a run without it compares two different noise
draws while believing it isolated one stage. If the realisation depends on the
block size, then a scheduling decision - which is not a modelling decision - is
a numerical one, and two machines disagree about the answer.

Both are properties of the RNG's KEYING, not of its quality. A counter-based
generator keyed by ``(seed_base, stage_id, linear element index)`` has both by
construction; a per-thread state seeded from a thread id has neither.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

BLOCK_ENV = "WITWIN_RADAR_FRONTEND_BLOCK"


def _chain(noise, *, seed: int, lna=None):
    from witwin.radar.frontend import FrontendChain, FrontendSpec, PortSpec, SeedSpec

    return FrontendChain(FrontendSpec(port=PortSpec(1.0), noise=noise, lna=lna, seed=SeedSpec(seed_base=seed)))


def _thermal_only():
    from witwin.radar.frontend import NoiseSpec

    return NoiseSpec(noise_figure_db=6.0, bandwidth_hz=5e6)


def _thermal_and_phase():
    from witwin.radar.frontend import NoiseSpec

    return NoiseSpec(
        noise_figure_db=6.0,
        bandwidth_hz=5e6,
        phase_noise_dbc_per_hz=-85.0,
        phase_offset_hz=1e5,
        phase_sample_rate_hz=5e6,
    )


def _zeros(count: int = 1 << 16) -> torch.Tensor:
    return torch.zeros(count, dtype=torch.complex64, device="cuda")


# ---------------------------------------------------------------------------
# T4.10 - per-stage seed independence
# ---------------------------------------------------------------------------


def test_toggling_phase_noise_leaves_the_thermal_realisation_bit_identical():
    """The hazard this kills, stated as an equality.

    A single generator threaded through the chain consumes its draws in order,
    so enabling phase noise shifts every subsequent thermal draw and the two runs
    differ everywhere. With per-stage keying they differ nowhere.

    A DEVIATION from the brief, recorded rather than buried: the brief measures
    this by subtracting the phase-rotated signal from a run with a nonzero input.
    That subtraction is not exact in floating point - ``(x e^{j theta} + n) -
    x e^{j theta}`` recovers ``n`` only to rounding - so it could not be asserted
    with ``torch.equal`` and would need a tolerance, which is precisely what this
    property does not want. Running both chains on a ZERO input isolates the
    additive term exactly, and the equality is then the real statement: the same
    draws, bit for bit, with a stage toggled between them.
    """

    signal = _zeros()
    without_phase = _chain(_thermal_only(), seed=3).apply(signal).signal
    with_phase = _chain(_thermal_and_phase(), seed=3).apply(signal).signal
    assert torch.equal(without_phase, with_phase)

    # And the phase stage really was doing something in the second run, so the
    # equality above is not the trivial one.
    phase = _chain(_thermal_and_phase(), seed=3).apply(signal).diagnostics.phase_rad
    assert phase is not None
    assert float(phase.abs().max()) > 0.0


def test_toggling_the_lna_leaves_the_thermal_realisation_a_pure_scaling():
    """The gain multiplies the same draws; it does not draw different ones."""

    from witwin.radar.frontend import LnaSpec

    signal = _zeros()
    plain = _chain(_thermal_only(), seed=9).apply(signal).signal
    amplified = _chain(_thermal_only(), seed=9, lna=LnaSpec(gain_db=12.0)).apply(signal).signal
    assert torch.allclose(amplified, plain * LnaSpec(gain_db=12.0).voltage_gain, rtol=1e-6, atol=0.0)


# ---------------------------------------------------------------------------
# T4.11 - full reproducibility
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_realisation_and_a_different_seed_does_not():
    signal = _zeros()
    first = _chain(_thermal_and_phase(), seed=5).apply(signal)
    second = _chain(_thermal_and_phase(), seed=5).apply(signal)
    other = _chain(_thermal_and_phase(), seed=6).apply(signal)

    assert torch.equal(first.signal, second.signal)
    assert torch.equal(first.diagnostics.phase_rad, second.diagnostics.phase_rad)
    assert not torch.equal(first.signal, other.signal)
    assert not torch.equal(first.diagnostics.phase_rad, other.diagnostics.phase_rad)


@pytest.mark.parametrize("block", ["64", "128", "512", "1024"])
def test_the_realisation_does_not_depend_on_the_launch_configuration(monkeypatch, block):
    """The Philox-versus-per-thread-state test, run at four launch widths.

    This is the assertion that forbids a ``curand`` state-per-thread scheme. Such
    a scheme keys its stream by the thread that happens to touch an element, so
    changing the block size renumbers every stream and every realisation changes -
    silently, because the STATISTICS are unchanged and only the realisation moves.
    Counter-based keying by the element's linear index cannot do that.

    The Wiener scan is deliberately excluded from the override: its accumulation
    ORDER is part of the realisation, so it runs single-threaded at every block
    size and this test would not detect a change there. That is stated rather
    than implied because it is a real limit of what is being measured.
    """

    signal = _zeros()
    monkeypatch.delenv(BLOCK_ENV, raising=False)
    baseline = _chain(_thermal_and_phase(), seed=11).apply(signal)

    monkeypatch.setenv(BLOCK_ENV, block)
    from witwin.radar.frontend import frontend_block_size

    assert frontend_block_size() == int(block)
    resized = _chain(_thermal_and_phase(), seed=11).apply(signal)

    assert torch.equal(baseline.signal, resized.signal)
    assert torch.equal(baseline.diagnostics.phase_rad, resized.diagnostics.phase_rad)


def test_an_unusable_block_override_is_refused_rather_than_rounded(monkeypatch):
    from witwin.radar.frontend import frontend_block_size

    for value in ("0", "17", "2048", "-64"):
        monkeypatch.setenv(BLOCK_ENV, value)
        with pytest.raises(ValueError, match="power of two"):
            frontend_block_size()


def test_the_realisation_does_not_depend_on_the_signal_it_is_added_to():
    """The draws are a function of the index and the seed, and of nothing else.

    Subtracting two runs that differ only in their input recovers the same
    scaled input, which says the additive term was identical in both. A generator
    seeded from anything data-dependent - a pointer, a length, a hash of the
    input - would fail this while passing every statistical test.
    """

    generator = torch.Generator(device="cpu").manual_seed(13)
    base = (
        torch.complex(torch.randn(1 << 14, generator=generator), torch.randn(1 << 14, generator=generator))
        .to(torch.complex64)
        .cuda()
    )
    chain = _chain(_thermal_only(), seed=17)
    first = chain.apply(base).signal
    second = chain.apply(2 * base).signal
    assert torch.allclose(second - first, base, rtol=1e-5, atol=1e-7)
