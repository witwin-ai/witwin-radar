"""Raw ADC export in the TI DCA1000 capture layout, and the matching reader.

A DCA1000 capture is a flat little-endian ``int16`` stream. :func:`write_dca1000`
writes an FMCW beat cube in that layout so a pipeline written for recorded TI
data reads simulated frames unchanged, and :func:`read_dca1000` reads one
back (and reads a real capture written in the same layout).

The layout written, and the assumptions it rests on - stated because they are
reproduced from TI's published capture descriptions (the DCA1000 user guide and
the "mmWave Radar Device ADC Raw Data Capture" application report with its
``readDCA1000`` reference script), not checked against hardware here:

* The file is the POST-PROCESSED capture: the UDP packet headers the DCA1000
  streams (sequence number and byte count per packet) are already stripped,
  which is what mmWave Studio and the DCA1000 CLI write after a capture. No
  header, no padding.
* Order, outermost first: frame, chirp in transmission order, then the layout's
  receiver/sample interleave. Under TDM, chirp ``c`` of transmitter ``t`` is
  transmitted at slot ``c * num_tx + t`` (R-ADR-047's slot law), so the file's
  chirp order is ``(c, t)`` with ``t`` fastest. A DDM or BPM cube has one
  transmit channel and its chirps are written in order.
* ``layout="xwr16xx"`` (also xWR18xx, xWR68xx, AWRx843: two LVDS lanes),
  complex: per chirp, receiver by receiver, each receiver's ``N`` samples as
  groups of four values ``I[n], I[n+1], Q[n], Q[n+1]``; ``N`` must be even.
  Real: per chirp, receiver by receiver, ``N`` samples.
* ``layout="xwr14xx"`` (four LVDS lanes, at most four receivers), complex: per
  chirp, sample by sample, ``I`` of every receiver then ``Q`` of every
  receiver. Real: per chirp, sample by sample, every receiver.
* ``I`` is the cube's real part and ``Q`` its imaginary part, so the complex
  sample is ``I + jQ`` in the cube's own phasor convention (the de-chirped beat
  convention: a target at positive range is a positive beat frequency).
  ``iq_swap=True`` writes ``Q`` first, as an ``adcbufCfg`` IQ-swap setting does.
* A value is the cube's amplitude times ``scale`` codes per unit, rounded half
  to even. With ``adc=`` the radar's converter, ``scale = 1 / Adc.step`` and a
  quantised cube is written as its exact codes. A code outside ``int16`` is
  refused unless ``clip=True``, which saturates it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: The receiver/sample interleaves this module writes and reads.
DCA1000_LAYOUTS = ("xwr16xx", "xwr14xx")

_INT16 = np.iinfo(np.int16)


def _cube_of(source: Any) -> tuple[torch.Tensor, int]:
    """The ``[frame, tx, rx, chirp, sample]`` cube and its transmit-channel count, refusing a non-beat cube."""

    cube = getattr(source, "cube", source)
    if not isinstance(cube, torch.Tensor):
        raise TypeError(f"write_dca1000 takes a Result or a [frame, tx, rx, chirp, sample] tensor, got {type(source)}")
    if hasattr(source, "cube"):
        if source.kind != "fmcw" or source.output_domain != "beat":
            raise ValueError(
                f"a raw ADC capture holds FMCW beat samples; this result is {source.kind} in the "
                f"{source.output_domain!r} domain. Simulate with Fmcw(output='beat')"
            )
    if cube.dim() == 4:
        cube = cube.unsqueeze(0)
    if cube.dim() != 5:
        raise ValueError(f"the cube must be [frame, tx, rx, chirp, sample], got shape {tuple(cube.shape)}")
    return cube.detach(), int(cube.shape[1])


def _codes(values: np.ndarray, scale: float, clip: bool) -> np.ndarray:
    scaled = np.rint(values.astype(np.float64) * scale)
    outside = int(np.count_nonzero((scaled < _INT16.min) | (scaled > _INT16.max)))
    if outside and not clip:
        raise ValueError(
            f"{outside} samples fall outside int16 at scale {scale} codes per unit; lower the scale or pass clip=True"
        )
    return np.clip(scaled, _INT16.min, _INT16.max).astype("<i2")


def write_dca1000(
    path: str | os.PathLike[str],
    source: Any,
    *,
    scale: float | None = None,
    adc: Any = None,
    layout: str = "xwr16xx",
    real: bool | None = None,
    iq_swap: bool = False,
    clip: bool = False,
) -> Path:
    """Write ``source`` (a beat-domain FMCW :class:`~witwin.radar.Result`, or its cube) as a DCA1000 capture.

    Exactly one of ``scale`` (codes per unit of the cube) and ``adc`` (the
    :class:`~witwin.radar.Adc` the cube was quantised by) is given. ``real``
    writes the in-phase channel only; ``None`` is real for a real-valued cube
    and complex otherwise. Returns the file path.
    """

    if (scale is None) == (adc is None):
        raise ValueError("give exactly one of scale (codes per unit) and adc (the converter)")
    if adc is not None:
        if int(adc.bits) > 16:
            raise ValueError(f"a {adc.bits}-bit converter's codes do not fit the capture's int16 samples")
        codes_per_unit = 1.0 / float(adc.step)
    elif scale is not None and float(scale) > 0.0:
        codes_per_unit = float(scale)
    else:
        raise ValueError(f"scale must be a positive number of codes per unit, got {scale!r}")
    if layout not in DCA1000_LAYOUTS:
        raise ValueError(f"layout must be one of {DCA1000_LAYOUTS}, got {layout!r}")
    cube, _ = _cube_of(source)
    frames, num_tx, num_rx, chirps, samples = (int(size) for size in cube.shape)
    is_real = (not cube.is_complex()) if real is None else bool(real)
    # [frame, tx, rx, chirp, sample] -> [frame, chirp, tx, rx, sample]: transmission order, then receivers.
    ordered = cube.permute(0, 3, 1, 2, 4).reshape(frames * chirps * num_tx, num_rx, samples).cpu()
    if is_real:
        values = _codes((ordered.real if ordered.is_complex() else ordered).numpy(), codes_per_unit, clip)
        if layout == "xwr14xx":
            values = values.transpose(0, 2, 1)
        stream = values
    else:
        if not ordered.is_complex():
            raise ValueError("a complex capture needs a complex cube; pass real=True for the in-phase channel")
        first, second = ordered.real.numpy(), ordered.imag.numpy()
        if iq_swap:
            first, second = second, first
        first, second = _codes(first, codes_per_unit, clip), _codes(second, codes_per_unit, clip)
        if layout == "xwr16xx":
            if samples % 2:
                raise ValueError(f"the two-lane layout pairs samples; {samples} samples per chirp is odd")
            pairs = (frames * chirps * num_tx, num_rx, samples // 2, 2)
            stream = np.stack((first.reshape(pairs), second.reshape(pairs)), axis=3)
        else:
            if num_rx > 4:
                raise ValueError(f"the four-lane layout carries at most four receivers, got {num_rx}")
            stream = np.stack((first.transpose(0, 2, 1), second.transpose(0, 2, 1)), axis=2)
    target = Path(path)
    np.ascontiguousarray(stream, dtype="<i2").tofile(target)
    return target


def read_dca1000(
    path: str | os.PathLike[str],
    *,
    num_tx: int,
    num_rx: int,
    chirps: int,
    samples: int,
    layout: str = "xwr16xx",
    real: bool = False,
    iq_swap: bool = False,
    scale: float = 1.0,
) -> torch.Tensor:
    """Read a DCA1000 capture into ``[frame, tx, rx, chirp, sample]``, divided by ``scale``.

    ``num_tx`` is the number of transmit channels recorded separately (one for
    DDM or BPM) and ``chirps`` the chirps per transmitter per frame. Complex
    captures come back ``complex64``, real ones ``float32``. A file whose
    length is not a whole number of frames is refused.
    """

    if layout not in DCA1000_LAYOUTS:
        raise ValueError(f"layout must be one of {DCA1000_LAYOUTS}, got {layout!r}")
    data = np.fromfile(Path(path), dtype="<i2").astype(np.float32)
    per_frame = num_tx * num_rx * chirps * samples * (1 if real else 2)
    if data.size == 0 or data.size % per_frame:
        raise ValueError(f"{path} holds {data.size} int16 values, not a whole number of {per_frame}-value frames")
    frames = data.size // per_frame
    blocks = frames * chirps * num_tx
    # Each component is divided by the scale on its own: a complex division
    # would round differently from the real quotient the writer inverted.
    data = (data / np.float32(scale)).astype(np.float32)
    if real:
        if layout == "xwr16xx":
            cube = data.reshape(blocks, num_rx, samples)
        else:
            cube = data.reshape(blocks, samples, num_rx).transpose(0, 2, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(cube))
    else:
        if layout == "xwr16xx":
            grouped = data.reshape(blocks, num_rx, samples // 2, 2, 2)
            first, second = grouped[:, :, :, 0, :].reshape(blocks, num_rx, samples), grouped[:, :, :, 1, :]
            second = second.reshape(blocks, num_rx, samples)
        else:
            grouped = data.reshape(blocks, samples, 2, num_rx)
            first, second = grouped[:, :, 0, :].transpose(0, 2, 1), grouped[:, :, 1, :].transpose(0, 2, 1)
        if iq_swap:
            first, second = second, first
        tensor = torch.complex(
            torch.from_numpy(np.ascontiguousarray(first)), torch.from_numpy(np.ascontiguousarray(second))
        )
    shaped = tensor.reshape(frames, chirps, num_tx, num_rx, samples).permute(0, 2, 3, 1, 4).contiguous()
    return shaped


__all__ = ["DCA1000_LAYOUTS", "read_dca1000", "write_dca1000"]
