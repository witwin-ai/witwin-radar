"""What each processing stage publishes, and the conventions it publishes it in.

Data only. Every record here is a frozen dataclass that a stage CONSTRUCTS
through a ``from_<producer>`` classmethod or that a stage entry returns
directly; none of them evaluates anything. The producer knows which product it
made, so a consumer never has to infer an axis, a unit, or a sign from the name
of the function that returned it.

Three conventions are carried as data rather than as documentation, because all
three change the answer and none of them is visible in a magnitude plot:

* :data:`PROCESSING_DOPPLER_CONVENTION` - a POSITIVE Doppler bin is a CLOSING
  (approaching) target, in every waveform. The FMCW beat cube is the conjugate
  of Channel's phasor, so its raw slow-time tone sits at ``+f_ref tau_rate``
  while the OFDM and pulsed tones sit at ``-f_ref tau_rate``. The two are
  reconciled EXACTLY ONCE, inside :func:`~witwin.radar.processing.range_doppler`,
  driven by :attr:`ProcessingAxes.doppler_sign`. Nothing else in this package
  may look at a phasor.
* :data:`PROCESSING_AMPLITUDE_CONVENTION` - every range profile and every
  Range-Doppler map is an AMPLITUDE estimate, normalised so that an isolated
  path row peaks at ``|C_rt|`` times the window's coherent gain. The three
  waveforms' native transforms have three different gains (``N`` for an
  unnormalised beat FFT, ``1`` for a CIR inverse transform, ``1`` for a
  unit-energy matched filter), and a facade that published all three would make
  a cross-waveform amplitude comparison a per-waveform bookkeeping exercise.
* :data:`PROCESSING_UNITS` - SI throughout, published as a mapping so a test can
  assert it against the values rather than against a docstring.

Nothing in this module reads a path batch, a topology, or a ``row_valid`` mask.
Processing consumes synthesis results; dead rows were already zeroed on the
WEIGHT before the kernel launched, so a cube arrives with their contribution
already exactly zero and there is nothing here for a row mask to do.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


#: A positive Doppler bin is a closing target. Stated once, applied once.
PROCESSING_DOPPLER_CONVENTION = "positive_doppler_bin_is_closing"

#: The amplitude normalisation every stage in this package publishes.
PROCESSING_AMPLITUDE_CONVENTION = (
    "peak = |C_rt| * window_coherent_gain; transforms are amplitude normalised"
)

#: The name of the fast-time axis per waveform, and of the slow-time axis. Read
#: by :class:`~witwin.radar.processing.axes.ProcessingAxes`; published here so a
#: reader finds the whole vocabulary in one module.
FAST_TIME_NAMES = {"fmcw": "sample", "ofdm": "subcarrier", "pulsed": "sample"}
SLOW_TIME_NAMES = {"fmcw": "chirp", "ofdm": "symbol", "pulsed": "pulse"}

#: Every published scalar and axis, with its SI unit. A quantity that is not in
#: this mapping is not part of the published metadata contract.
PROCESSING_UNITS = {
    "slow_time_period_s": "s",
    "range_bin_m": "m",
    "range_origin_m": "m",
    "max_unambiguous_range_m": "m",
    "velocity_bin_mps": "m/s",
    "max_unambiguous_speed_mps": "m/s",
    "wavelength_m": "m",
    "reference_frequency_hz": "Hz",
    "element_spacing_m": "m",
    "range_m": "m",
    "velocity_mps": "m/s",
}


def _require_complex(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if not value.is_complex():
        raise TypeError(
            f"{name} must be complex; a magnitude has already thrown away the "
            f"phase every later stage needs, got {value.dtype}"
        )
    return value


@dataclass(frozen=True, slots=True, eq=False)
class RangeProfile:
    """``[..., C, R]`` complex, one range profile per slow-time sample.

    ``R`` is the range-bin count the axes record already published, and the
    profile is asserted against it rather than defining it: a stage that could
    publish its own bin count could publish one that disagrees with the metres
    the axis is in.

    ``window_coherent_gain`` is ``mean(w)``. For FMCW and OFDM the identity
    ``peak == |C_rt| * window_coherent_gain`` is exact for an isolated on-bin
    row. For the pulsed backend the window tapers the correlation SPECTRUM, so
    the gain there is the ``|P(f)|^2``-weighted mean of the same window and the
    identity holds only for the unwindowed case. Published rather than folded
    in, so a caller that wants the raw transform can recover it.
    """

    data: torch.Tensor
    axes: object
    window: str
    window_coherent_gain: float

    def __post_init__(self) -> None:
        _require_complex("data", self.data)
        if self.data.dim() < 2:
            raise ValueError(
                "a range profile is [..., slow_time, range]; got shape "
                f"{tuple(self.data.shape)}"
            )
        if int(self.data.shape[-1]) != int(self.axes.range_bin_count):
            raise ValueError(
                f"the profile has {int(self.data.shape[-1])} range bins but its "
                f"axes record publishes {int(self.axes.range_bin_count)}"
            )

    @property
    def range_axis(self) -> torch.Tensor:
        """``[R]`` float64 metres, from the axes record and nowhere else."""

        return self.axes.range_m


@dataclass(frozen=True, slots=True, eq=False)
class RangeDopplerMap:
    """``[..., D, R]`` complex, ``fftshift``ed and in the closing-positive sign.

    The Doppler axis is :attr:`ProcessingAxes.velocity_mps`: signed metres per
    second, ascending, with the waveform's phasor reconciliation already
    applied. There is no second place in this package where a sign is decided.
    """

    data: torch.Tensor
    axes: object
    window: str
    window_coherent_gain: float

    def __post_init__(self) -> None:
        _require_complex("data", self.data)
        if self.data.dim() < 2:
            raise ValueError(
                "a Range-Doppler map is [..., doppler, range]; got shape "
                f"{tuple(self.data.shape)}"
            )
        if int(self.data.shape[-1]) != int(self.axes.range_bin_count):
            raise ValueError(
                f"the map has {int(self.data.shape[-1])} range bins but its axes "
                f"record publishes {int(self.axes.range_bin_count)}"
            )
        if int(self.data.shape[-2]) != int(self.axes.doppler_bin_count):
            raise ValueError(
                f"the map has {int(self.data.shape[-2])} Doppler bins but its "
                f"axes record publishes {int(self.axes.doppler_bin_count)}"
            )

    @property
    def range_axis(self) -> torch.Tensor:
        return self.axes.range_m

    @property
    def doppler_axis(self) -> torch.Tensor:
        """``[D]`` float64 metres per second, positive for a closing target."""

        return self.axes.velocity_mps


@dataclass(frozen=True, slots=True, eq=False)
class BeamCube:
    """``[*beam, D, R]`` complex: the beam / velocity / range cube.

    ``directions`` is the ``[*beam, 3]`` unit-vector grid the cube was formed
    over, in the array's LOCAL frame. It travels with the cube because a beam
    index means nothing without it, and because a detector that reports an angle
    has to read the same grid the former steered with.
    """

    data: torch.Tensor
    axes: object
    directions: torch.Tensor

    def __post_init__(self) -> None:
        _require_complex("data", self.data)
        if self.data.dim() < 3:
            raise ValueError(
                "a beam cube is [*beam, doppler, range]; got shape "
                f"{tuple(self.data.shape)}"
            )
        beam_shape = tuple(self.data.shape[:-2])
        if tuple(self.directions.shape) != (*beam_shape, 3):
            raise ValueError(
                f"the cube spans beams {beam_shape} but directions has shape "
                f"{tuple(self.directions.shape)}; the two are one statement"
            )

    @property
    def range_axis(self) -> torch.Tensor:
        return self.axes.range_m

    @property
    def doppler_axis(self) -> torch.Tensor:
        return self.axes.velocity_mps


__all__ = [
    "FAST_TIME_NAMES",
    "PROCESSING_AMPLITUDE_CONVENTION",
    "PROCESSING_DOPPLER_CONVENTION",
    "PROCESSING_UNITS",
    "SLOW_TIME_NAMES",
    "BeamCube",
    "RangeDopplerMap",
    "RangeProfile",
]
