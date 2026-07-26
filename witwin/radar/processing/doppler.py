"""The slow-time transform, and the ONE place a Doppler sign is reconciled.

The transform itself is unremarkable: window slow time, transform, ``fftshift``.
What matters is the four lines in the middle.

FMCW's beat cube is the CONJUGATE of Channel's ``exp(-j k d)`` phasor, because
de-chirping multiplies the echo by the conjugate of the transmitted chirp. Its
slow-time tone therefore sits at ``+f_ref tau_rate`` while the OFDM and pulsed
tones sit at ``-f_ref tau_rate``. Nothing in this repository reconciled that:
``sigproc/microdoppler.py`` fixes a Channel-SIGNED axis and is handed FMCW cubes
by callers, so the same receding target read as approaching on one waveform and
receding on another, in a magnitude plot where the difference is invisible.

The canonical convention, stated once in
:data:`~witwin.radar.processing.contracts.PROCESSING_DOPPLER_CONVENTION` and
applied once here, is that a POSITIVE Doppler bin is a CLOSING target. A
closing radial speed ``v`` gives ``tau_rate = -2 v / c``, so the canonical
frequency is ``-f_ref tau_rate = +2 v / lambda`` and

    ``v = lambda f / 2``

is the velocity axis :class:`ProcessingAxes` publishes, ascending and
``fftshift``ed, for every waveform. The reconciliation is a FREQUENCY REVERSAL
of the raw spectrum, ``X[k] -> X[(-k) mod D]``, applied only when
``axes.doppler_sign`` is ``+1``. It is an index gather with no arithmetic, so it
is exact, and it happens BEFORE the shift because negating a frequency index is
a wrap in the unshifted order and is not the same as reversing a shifted axis
(for even ``D`` the shifted axis is asymmetric about zero, and reversing it
would move every bin by one).
"""

from __future__ import annotations

import torch

from .contracts import RangeDopplerMap, RangeProfile
from .primitives import DEFAULT_WINDOW, taper, window_coherent_gain


def range_doppler(
    profile: RangeProfile, *, window: str | None = None
) -> RangeDopplerMap:
    """``RangeProfile[..., C, R]`` -> ``RangeDopplerMap[..., D, R]``.

    Rank generic with an arbitrary leading batch, so ``[P, C, R]`` and
    ``[TX, RX, C, R]`` both work without a Python loop.

    Amplitude normalised like the range stage: the transform carries ``1 / D``
    so an isolated on-bin row peaks at its own coefficient magnitude rather than
    at the coherent-integration gain times it. The integration gain is
    recoverable exactly from :attr:`RangeDopplerMap.axes.doppler_bin_count`.
    """

    if not isinstance(profile, RangeProfile):
        raise TypeError(
            "range_doppler consumes a RangeProfile, so that the range axis it "
            "publishes is the one the range stage already decided; got "
            f"{type(profile).__name__}"
        )
    record = profile.axes
    name = DEFAULT_WINDOW if window is None else str(window)
    data = profile.data
    if int(data.shape[-2]) != int(record.doppler_bin_count):
        raise ValueError(
            f"the profile has {int(data.shape[-2])} slow-time samples but the "
            f"metadata record's transform is {int(record.doppler_bin_count)} long"
        )

    spectrum = torch.fft.fft(taper(data, name, dim=-2), dim=-2, norm="forward")
    if record.doppler_sign == 1:
        # The cube is conjugated relative to Channel, so its tone sits at the
        # negated canonical frequency. Reverse the frequency index - a gather,
        # not arithmetic - and every waveform below this line is in one sign.
        bins = int(record.doppler_bin_count)
        reversed_index = torch.remainder(
            -torch.arange(bins, device=spectrum.device), bins
        )
        spectrum = spectrum.index_select(-2, reversed_index)
    spectrum = torch.fft.fftshift(spectrum, dim=-2)

    return RangeDopplerMap(
        data=spectrum,
        axes=record,
        window=name,
        window_coherent_gain=window_coherent_gain(
            name, int(record.doppler_bin_count)
        ),
    )


__all__ = ["range_doppler"]
