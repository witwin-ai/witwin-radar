"""The attach point: a synthesis result becomes a processing cube.

``SynthesisResult`` had no consumer outside ``witwin/radar/synthesis/`` and
``radar.py`` at all. This is where the processing chain attaches to it, and the
transpose it needs already exists: ``synthesis/assembly.py::assemble_frame_cube``
converts the SINK-MAJOR composed pair rank into the TX-MAJOR virtual-antenna
index every angle estimator steers. That function is CALLED here rather than
reimplemented, because two packers for one layout is how a TX/RX swap ships
silently on a square array.

Nothing in this module knows about ``row_valid``. A dead row is masked on the
WEIGHT before the waveform kernel launches, so its contribution to the cube is a
literal zero and no downstream stage has anything to mask. That is asserted by
``tests/processing/test_cube.py`` rather than assumed here: an assumption that
nothing executes is an assumption that stops being true.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..synthesis.assembly import assemble_frame_cube


@dataclass(frozen=True, slots=True, eq=False)
class ProcessingCube:
    """``[TX, RX, C, S]`` complex, with the metadata record it was built with.

    ``C`` is the slow-time axis (chirps, symbols or pulses) and ``S`` the
    fast-time axis (samples or subcarriers); which is which is named on the
    axes record rather than inferred from the waveform.
    """

    data: torch.Tensor
    axes: object

    def __post_init__(self) -> None:
        if not isinstance(self.data, torch.Tensor):
            raise TypeError(
                f"data must be a torch.Tensor, got {type(self.data).__name__}"
            )
        if not self.data.is_complex():
            raise TypeError(
                "a processing cube is complex IQ; a magnitude has already "
                f"thrown away the phase every later stage needs, got {self.data.dtype}"
            )
        if self.data.dim() != 4:
            raise ValueError(
                "a processing cube is [tx, rx, slow_time, fast_time]; got shape "
                f"{tuple(self.data.shape)}"
            )
        expected = (self.axes.num_tx, self.axes.num_rx)
        if tuple(self.data.shape[:2]) != expected:
            raise ValueError(
                f"the cube's array axes are {tuple(self.data.shape[:2])} but the "
                f"axes record declares {expected}"
            )
        if int(self.data.shape[2]) != int(self.axes.doppler_bin_count):
            raise ValueError(
                f"the cube has {int(self.data.shape[2])} slow-time samples but "
                f"the axes record's Doppler transform is "
                f"{int(self.axes.doppler_bin_count)} long"
            )

    @property
    def device(self) -> torch.device:
        return self.data.device

    @classmethod
    def from_synthesis(cls, result, axes) -> "ProcessingCube":
        """Pack one rank-3 synthesis cube into the rank-4 array layout.

        The axes record is passed in rather than rebuilt, so that the record
        used to pack the cube and the record every later stage reads are the
        same object. Rebuilding it here would let a caller process a cube
        against a different array than the one it was assembled for.
        """

        if tuple(result.axes) != (
            axes.slow_time_name,
            "sensor_pair",
            axes.fast_time_name,
        ):
            raise ValueError(
                f"this result publishes axes {tuple(result.axes)} but the "
                f"metadata record describes ({axes.slow_time_name}, "
                f"sensor_pair, {axes.fast_time_name}); the two describe "
                "different products"
            )
        if str(result.phasor) != axes.phasor:
            raise ValueError(
                f"this result is in the {result.phasor!r} convention but the "
                f"metadata record was built for {axes.phasor!r}; the Doppler "
                "sign is derived from that string and would be reconciled "
                "backwards"
            )
        return cls(
            data=assemble_frame_cube(
                result.cube, num_tx=axes.num_tx, num_rx=axes.num_rx
            ),
            axes=axes,
        )


__all__ = ["ProcessingCube"]
