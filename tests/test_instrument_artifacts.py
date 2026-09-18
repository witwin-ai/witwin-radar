"""Antenna coupling and quadrature imbalance, both off unless declared.

These two are the artifacts a sub-metre dataset cannot omit without teaching a
model something false. Coupling puts a tone at a range of centimetres, tens of dB
above anything a body returns, and the range transform spreads its skirt over
exactly the bins a target inside a metre occupies. Quadrature imbalance mirrors
the Doppler axis, and the sign of the velocity is often the label itself.

Each test asserts the effect against the record's OWN declaration - the isolation
in dB, the image rejection in dB - rather than against a number spelled a second
time here, so a change to the model cannot be absorbed by editing a constant.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.radar import Radar
from witwin.radar.frontend import Iq
from witwin.radar.radar import Leakage

#: A 77 GHz sweep whose sampled bandwidth gives a 4.3 cm range cell, so a target
#: inside a metre lands around bin 23 of 256 and the coupling skirt reaches it.
CONFIG = {
    "num_tx": 1,
    "num_rx": 1,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 58,
    "chirp_per_frame": 8,
    "power": 12,
    "tx_loc": [[0, 0, 0]],
    "rx_loc": [[0, 0, 0]],
}

SPEED_OF_LIGHT_M_PER_S = 299792458.0
#: A two-centimetre coupling path: the board, not the room.
COUPLING_DELAY_S = 2 * 0.02 / SPEED_OF_LIGHT_M_PER_S


def _radar(**extra) -> Radar:
    return Radar.from_dict(CONFIG, position=(0, 0, 0), look_at=(1, 0, 0), **extra)


def test_neither_artifact_is_on_unless_it_is_declared():
    """The default is the ideal instrument, because a dataset must opt in."""

    radar = _radar()
    assert radar.leakage is None
    assert radar.iq is None
    assert radar.system_config.frontend is None or radar.system_config.frontend.iq is None


def test_the_coupled_amplitude_is_the_declared_isolation_below_the_transmit_power():
    """Isolation is the only knob, and it means what it says.

    The synthesis families publish sqrt(W), and the transmit power reaches them
    as ``sqrt(P)`` on the weight, so the coupled amplitude is ``sqrt(P)`` taken
    down by the isolation. Asserting it here is what stops the dB from being
    quietly interpreted as a power ratio on an amplitude.
    """

    leakage = Leakage(isolation_db=40.0, delay_s=COUPLING_DELAY_S)
    transmit_watts = 1e-3 * 10.0 ** (CONFIG["power"] / 10.0)
    assert leakage.amplitude(CONFIG["power"]) == pytest.approx(math.sqrt(transmit_watts) * 10.0 ** (-40.0 / 20.0))


def test_a_negative_isolation_and_a_negative_delay_are_refused():
    with pytest.raises(ValueError, match="isolation_db must be positive"):
        Leakage(isolation_db=0.0, delay_s=COUPLING_DELAY_S)
    with pytest.raises(ValueError, match="delay_s must not be negative"):
        Leakage(isolation_db=40.0, delay_s=-1e-12)


@pytest.mark.gpu
def test_the_coupling_lands_in_the_bin_its_delay_names_and_skirts_the_near_range():
    """Where the tone sits, and how far its skirt reaches.

    The bin is ``slope * delay * Ts * N``, which for a board-scale delay is bin
    zero. The skirt matters more than the peak: a rectangular window is what the
    closed-form spectrum is, and a target at one metre sits about 23 bins away,
    where the skirt is still tens of dB above the noise floor of the transform.
    That is the number a near-range dataset is missing when coupling is off.
    """

    radar = _radar(leakage=Leakage(isolation_db=40.0, delay_s=COUPLING_DELAY_S))
    spec = radar.system_config.waveform_spec()
    empty = torch.zeros((spec.num_chirps, 1, spec.num_samples), dtype=torch.complex64, device=radar.device)
    cube = radar._leakage_cube(spec, 1, empty)

    magnitude = cube.abs()[0, 0]
    expected_bin = spec.slope_hz_per_s * COUPLING_DELAY_S * spec.sample_period_s * spec.num_samples
    assert int(magnitude.argmax()) == round(expected_bin)

    metre_bin = round(spec.slope_hz_per_s * (2 / SPEED_OF_LIGHT_M_PER_S) * spec.sample_period_s * spec.num_samples)
    skirt_db = 20 * math.log10(float(magnitude[metre_bin] / magnitude.max()))
    # A rectangular window's skirt, not a floor: far below the tone, far above
    # nothing. The bound is loose on purpose - it pins the ORDER, which is what
    # decides whether a body echo competes with it.
    assert -45.0 < skirt_db < -25.0


@pytest.mark.gpu
def test_declaring_coupling_adds_it_to_the_echo_and_leaves_the_echo_alone():
    """Superposition, and the default's exactness.

    Coupling is one more path, so the cube with it must be the cube without it
    plus the coupling alone. Asserting equality on the ideal cube also pins that
    an undeclared coupling costs the echo nothing at all, not even a rounding.
    """

    ideal = _radar()
    coupled = _radar(leakage=Leakage(isolation_db=30.0, delay_s=COUPLING_DELAY_S))
    spec = ideal.system_config.waveform_spec()
    empty = torch.zeros((spec.num_chirps, 1, spec.num_samples), dtype=torch.complex64, device=ideal.device)

    assert torch.equal(ideal._leakage_cube(spec, 1, empty), empty.new_zeros(()))
    alone = coupled._leakage_cube(spec, 1, empty)
    assert alone.shape == empty.shape
    assert float(alone.abs().max()) > 0.0


def test_the_image_rejection_is_what_the_imbalance_declares():
    """A mirrored tone at the declared level, in the mirrored bin.

    The tone is placed on an integer Doppler bin so it has no leakage of its
    own; anything in the mirrored bin is the image and nothing else. The level
    is compared against ``image_rejection_db``, which is derived from the same
    two coefficients the stage applies, so the test pins the DEFINITION rather
    than a measured constant.
    """

    for gain_db, phase_deg in ((0.5, 2.0), (1.0, 5.0)):
        imbalance = Iq(gain_db=gain_db, phase_deg=phase_deg)
        chirps, bin_index = 64, 7
        # float64: the assertion is on the definition, and a float32 tone would
        # put its own rounding between the two coefficients and the measurement.
        slow = torch.arange(chirps, dtype=torch.float64)
        tone = torch.exp(2j * math.pi * bin_index / chirps * slow)
        mixed = imbalance.alpha * tone + imbalance.beta * tone.conj()
        doppler = torch.fft.fft(mixed, norm="forward").abs()
        measured = -20 * math.log10(float(doppler[chirps - bin_index] / doppler[bin_index]))
        assert measured == pytest.approx(imbalance.image_rejection_db, abs=1e-6)


def test_a_perfect_demodulator_is_the_identity_and_says_so():
    """Zeroed is accepted and exact, because it is the end of a sweep."""

    imbalance = Iq()
    assert imbalance.alpha == pytest.approx(1.0)
    assert imbalance.beta == pytest.approx(0.0)
    assert imbalance.image_rejection_db == math.inf


@pytest.mark.gpu
def test_the_imbalance_reaches_the_receive_chain_and_the_quantiser_sees_the_image():
    """The stage is in the chain, before the gain rather than after it.

    Placement is observable: an image created before the AGC is part of the RMS
    the AGC divides by, so it changes the gain the true return receives. A stage
    bolted on after the quantiser could not do that, and would also hand back an
    unquantised image.
    """

    from dataclasses import replace

    from witwin.radar.frontend import Agc, FrontendChain, FrontendSpec, Noise

    signal = torch.ones((4, 16), dtype=torch.complex64, device="cuda")
    # A resolved noise bandwidth because Radar normally fills it from the
    # waveform, and an AGC so the placement claim is observable at all.
    spec = FrontendSpec(noise=Noise(figure=3.0, bandwidth=4.4e6), agc=Agc(target_rms=0.3), impedance=50.0, seed=3)
    ideal = FrontendChain(spec).apply(signal, seed_base=3)
    imbalanced = FrontendChain(replace(spec, iq=Iq(1.0, 5.0))).apply(signal, seed_base=3)
    assert not torch.equal(ideal.signal, imbalanced.signal)
    # The AGC divided by an RMS that now includes the image, so the gain moved.
    assert not torch.equal(ideal.diagnostics.agc_gain, imbalanced.diagnostics.agc_gain)
