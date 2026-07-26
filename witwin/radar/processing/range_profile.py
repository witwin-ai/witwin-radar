"""One range-profile entry, three waveform backends, one owner.

There were three range FFTs in ``sigproc`` with three different windowing
choices - Hamming on fast time only, Hamming on both axes plus an unconditional
DC removal, and none at all - and no named range-profile stage at all. The OFDM
equivalent, the inverse transform of the channel frequency response, did not
exist anywhere. This module is the single owner.

The backend is selected by ``axes.waveform``, which is a STORED discriminator
read off the metadata record, not a probe and not an inference:

* **FMCW** - transform over fast time. The beat tone sits at ``S tau``, so bin
  ``k`` is ``c k f_s / (2 S N)`` metres.
* **OFDM** - INVERSE transform over subcarriers: the channel impulse response.
  ``H[n] = C exp(-j 2 pi n df tau)`` inverts to a peak at ``tau / T_s``.
* **Pulsed** - matched filter over fast time, correlating against the analytic
  replica the synthesis kernel evaluates. Same correlation as
  ``sigproc/matched_filter.py`` WITHOUT its unconditional ``complex128``
  upcast, which doubled every pulsed frame's transform cost to buy precision a
  float32 input never had.

Two arguments are explicit here that used to be implicit somewhere:

* ``window`` - one named family, applied to the input of the final transform in
  all three backends, and defaulting to ``rectangular`` so that a caller
  comparing against an analytic unwindowed transform gets the analytic answer.
* ``remove_dc`` - the fast-time mean subtraction that ``process_rd_tensor``
  applies UNCONDITIONALLY and undocumented. It is a clutter operation, so it is
  a flag, and it **defaults to off**: a component-export test that asks for the
  clutter cube and silently gets the clutter removed from it is comparing two
  different quantities.

Every backend is amplitude normalised, so an isolated on-bin row peaks at
``|C_rt| * window_coherent_gain`` in all three. The three native transforms have
three different gains and publishing all three would make a cross-waveform
amplitude comparison a per-waveform bookkeeping exercise.
"""

from __future__ import annotations

import torch

from .contracts import RangeProfile
from .cube import ProcessingCube
from .primitives import (
    DEFAULT_WINDOW,
    matched_filter,
    remove_mean,
    taper,
    window_coherent_gain,
)


def _unpack(cube, axes):
    if isinstance(cube, ProcessingCube):
        if axes is not None and axes is not cube.axes:
            raise ValueError(
                "a ProcessingCube already carries its metadata record; passing "
                "a second one is how a cube ends up processed against a "
                "different array than it was assembled for"
            )
        return cube.data, cube.axes
    if not isinstance(cube, torch.Tensor):
        raise TypeError(
            "range_profile takes a ProcessingCube or a [..., slow_time, "
            f"fast_time] tensor, got {type(cube).__name__}"
        )
    if axes is None:
        raise ValueError(
            "a bare tensor carries no metadata; pass axes= so the range axis "
            "and the backend are decided by the same record every later stage "
            "reads"
        )
    return cube, axes


def range_profile(
    cube,
    *,
    axes=None,
    window: str | None = None,
    remove_dc: bool = False,
) -> RangeProfile:
    """``[..., C, S]`` complex -> ``RangeProfile`` with ``[..., C, R]``.

    Rank generic with an arbitrary leading batch: a ``[TX, RX, C, S]`` cube and
    a ``[C, S]`` slice of it give identical results on the shared slice, with no
    Python loop over the batch anywhere.
    """

    data, record = _unpack(cube, axes)
    if not data.is_complex():
        raise TypeError(
            "a range profile is formed from complex IQ; got a real tensor of "
            f"dtype {data.dtype}"
        )
    if data.dim() < 2:
        raise ValueError(
            "the input is [..., slow_time, fast_time]; got shape "
            f"{tuple(data.shape)}"
        )
    name = DEFAULT_WINDOW if window is None else str(window)

    if remove_dc:
        data = remove_mean(data, dim=-1)

    if record.waveform == "pulsed":
        expected = record.range_bin_count // record.range_oversample
        if int(data.shape[-1]) != expected:
            raise ValueError(
                f"the fast-time axis holds {int(data.shape[-1])} samples but the "
                f"metadata record was built for {expected} at oversample "
                f"{record.range_oversample}"
            )
        profile = matched_filter(
            data,
            record.matched_filter_replica,
            sample_period_s=record.matched_filter_sample_period_s,
            oversample=record.range_oversample,
            window=name,
        )
        taper_length = int(data.shape[-1]) + int(
            record.matched_filter_replica.shape[0]
        )
    else:
        if int(data.shape[-1]) != record.range_bin_count:
            raise ValueError(
                f"the fast-time axis holds {int(data.shape[-1])} "
                f"{record.fast_time_name}s but the metadata record publishes "
                f"{record.range_bin_count} range bins"
            )
        windowed = taper(data, name, dim=-1)
        taper_length = int(data.shape[-1])
        if record.waveform == "fmcw":
            # Amplitude normalised: the unnormalised beat FFT peaks at N |C|.
            profile = torch.fft.fft(windowed, dim=-1, norm="forward")
        else:
            # The CIR. The inverse transform already carries the 1 / N_sc that
            # makes the peak the coefficient itself.
            profile = torch.fft.ifft(windowed, dim=-1)

    return RangeProfile(
        data=profile,
        axes=record,
        window=name,
        window_coherent_gain=window_coherent_gain(name, taper_length),
    )


__all__ = ["range_profile"]
