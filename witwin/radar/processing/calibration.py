"""Calibration against a known scene: channel gains, the range bias, and the background.

Three measurements a radar needs before its products mean what their axes say,
each estimated from frames of a scene the caller knows and applied to any later
product of the same radar:

* **Array calibration** (:func:`estimate_array_calibration`). A corner
  reflector at a known position ``q`` returns, on virtual channel ``v``,
  ``g_v A`` after its compressed sample is read at ``q``'s own delay and that
  delay's carrier phase is removed (:func:`element_responses`): ``A`` is the
  reflector's amplitude, common to an ideal array, and ``g_v`` is everything
  that differs between channels - the transmit and receive paths' gain and
  phase, cable and trace lengths, the element's own pattern toward ``q``. The
  estimate is ``g_v = m_v / (mean_v |m_v| exp(j arg m_ref))``: unit mean
  magnitude, zero phase on the reference channel. Applying it divides each
  channel by ``g_v``, which leaves the common level and phase alone and makes
  the array's channels agree, so the angle routes read the geometry again.
* **Range (delay) calibration** (:func:`estimate_range_calibration`). An extra
  round-trip delay ``tau_b`` in the transmit and receive paths moves every
  return by ``c tau_b / 2`` metres. It is measured as the mean offset between
  the reflector's compressed peak (parabolic interpolation of ``|X|^2`` within
  ``search_m`` of where the geometry puts it) and the geometric range, and it is
  applied to the AXES: the range origin moves by ``-c tau_b / 2``, so the bins
  are unchanged and every later stage reads the corrected metres. The carrier
  phase ``f tau_b`` that the same delay adds is common to every channel and is
  absorbed by the array calibration's reference phase.
* **Background subtraction** (:func:`subtract_background`). Transmit leakage
  and static clutter are deterministic and identical in every frame of a still
  scene, so a reference frame recorded without the target (or the mean of
  several, :func:`average_background`) is subtracted sample for sample. Thermal
  noise does not cancel: the difference carries the reference's noise too,
  which averaging ``K`` reference frames reduces to ``1 / K`` of a frame's.

None of these is a per-sample hot path: the estimates read a few samples per
channel, and the applications are one broadcast multiply, one axis
replacement and one subtraction.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from ..policy import refuse_derivative
from ..synthesis.assembly import SPEED_OF_LIGHT_M_PER_S
from .angle import BeamCube
from .imaging import Aperture, _require_aperture, element_responses
from .range_doppler import RangeDopplerMap, RangeProfile
from .signal import ProcessingAxes, ProcessingCube, _range_axis

#: The products a calibration applies to: each carries ``data [..., TX, RX, a, b]`` and its axes.
CALIBRATED_PRODUCTS = (ProcessingCube, RangeProfile, RangeDopplerMap)

_PEAK_REASON = (
    "the range bias is read at the compressed peak, an argmax over range bins, and a peak position is a discrete "
    "choice whose derivative is not the derivative of the bias."
)


def _require_product(name: str, product: object):
    if not isinstance(product, CALIBRATED_PRODUCTS):
        if isinstance(product, BeamCube):
            raise TypeError(f"{name} is a beam cube; calibrate the channels before beamforming")
        raise TypeError(
            f"{name} must be a ProcessingCube, RangeProfile or RangeDopplerMap, got {type(product).__name__}"
        )
    return product


@dataclass(frozen=True, slots=True, eq=False)
class ArrayCalibration:
    """Per virtual channel complex gain ``g[TX, RX]`` (complex128, unit mean magnitude).

    ``reference`` is the ``(tx, rx)`` channel whose phase is zero by
    definition. :meth:`apply` divides a product's channels by ``g``.
    """

    gains: torch.Tensor
    reference: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.gains, torch.Tensor) or not self.gains.is_complex() or self.gains.dim() != 2:
            raise TypeError("ArrayCalibration.gains is a complex [TX, RX] tensor")
        tx, rx = self.reference
        if not (0 <= tx < int(self.gains.shape[0]) and 0 <= rx < int(self.gains.shape[1])):
            raise ValueError(f"reference channel {self.reference} is outside the {tuple(self.gains.shape)} array")

    def apply(self, product):
        """The same product with each channel divided by its gain."""

        product = _require_product("product", product)
        data = product.data
        if data.dim() < 4 or tuple(data.shape[-4:-2]) != tuple(self.gains.shape):
            raise ValueError(
                f"the product's channel axes are {tuple(data.shape[-4:-2])} but the calibration is "
                f"{tuple(self.gains.shape)}"
            )
        correction = (1.0 / self.gains).to(device=data.device, dtype=data.dtype)
        return replace(product, data=data * correction.reshape(*self.gains.shape, 1, 1))


def estimate_array_calibration(
    profile: RangeProfile, aperture: Aperture, reflector_m, *, reference: tuple[int, int] = (0, 0)
) -> ArrayCalibration:
    """Measure the channel gains from a point reflector at the known position ``reflector_m``.

    ``profile`` is the reflector frame's range profile (zero padded, as for
    :func:`backproject`), ``aperture`` the frame's :class:`Aperture`. Any
    leading batch (several frames) is averaged. The reflector must dominate its
    own resolution cell; a channel with no response is refused.
    """

    responses = element_responses(profile, aperture, reflector_m)
    if responses.dim() > 2:
        responses = responses.reshape(-1, *responses.shape[-2:]).mean(dim=0)
    responses = responses.to(torch.complex128)
    magnitude = responses.abs()
    if not bool((magnitude > 0).all()):
        raise ValueError("a channel has no response at the reflector; check its position and the range axis")
    tx, rx = reference
    anchor = responses[tx, rx]
    gains = responses / (magnitude.mean() * (anchor / anchor.abs()))
    return ArrayCalibration(gains=gains, reference=(int(tx), int(rx)))


@dataclass(frozen=True, slots=True)
class RangeCalibration:
    """The system's extra round-trip delay ``delay_s`` (TX plus RX path), s."""

    delay_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.delay_s):
            raise ValueError(f"delay_s must be finite, got {self.delay_s}")

    @property
    def range_bias_m(self) -> float:
        """``c tau_b / 2``: how far beyond its true range every return appears, m."""

        return SPEED_OF_LIGHT_M_PER_S * self.delay_s / 2.0

    def corrected_axes(self, axes: ProcessingAxes) -> ProcessingAxes:
        """``axes`` with the range origin moved by ``-c tau_b / 2``."""

        origin = float(axes.range_origin_m) - self.range_bias_m
        return replace(
            axes,
            range_origin_m=origin,
            range_m=_range_axis(int(axes.range_bin_count), float(axes.range_bin_m), origin, axes.device),
        )

    def apply(self, product):
        """The same samples, published on the corrected range axis."""

        product = _require_product("product", product)
        return replace(product, axes=self.corrected_axes(product.axes))


def estimate_range_calibration(
    profile: RangeProfile, aperture: Aperture, reflector_m, *, search_m: float
) -> RangeCalibration:
    """Measure the round-trip delay bias from a point reflector at ``reflector_m``.

    For every channel and slow-time sample the compressed peak is sought within
    ``search_m`` metres of the reflector's geometric range (half its two-way
    path from the aperture) and refined by a parabola through the peak bin and
    its neighbours on ``|X|^2``; the bias is the mean offset. Zero pad the
    profile: a parabola on a sinc main lobe sampled at its own resolution is
    biased by up to a tenth of a bin, and by much less on an 8x grid.
    """

    samples = _require_aperture(profile, aperture)
    refuse_derivative(
        "witwin.radar.processing.calibration.estimate_range_calibration", _PEAK_REASON, profile=profile.data
    )
    axes = profile.axes
    bin_m = float(axes.range_bin_m)
    if not search_m > 0.0:
        raise ValueError(f"search_m must be positive, got {search_m}")
    reach = max(1, math.ceil(float(search_m) / bin_m))
    ranges = int(samples.shape[-1])
    point = torch.as_tensor(reflector_m, dtype=torch.float64).reshape(1, 1, 3)
    half_path = 0.5 * (
        torch.linalg.vector_norm(point - aperture.tx_m, dim=-1)
        + torch.linalg.vector_norm(point - aperture.rx_m, dim=-1)
    )  # [C, V]
    expected = ((half_path - float(axes.range_origin_m)) / bin_m).transpose(0, 1).to(samples.device)  # [V, C]
    centre = torch.round(expected).to(torch.int64)
    if bool(((centre - reach - 1) < 0).any()) or bool(((centre + reach + 1) > ranges - 1).any()):
        raise ValueError("the search window around the reflector runs off the range axis; shrink search_m")
    offsets = torch.arange(-reach - 1, reach + 2, device=samples.device)
    index = (centre.unsqueeze(-1) + offsets).unsqueeze(0).expand(int(samples.shape[0]), -1, -1, -1)
    power = samples.gather(-1, index).abs().square().to(torch.float64)  # [B, V, C, 2 reach + 3]
    peak = power[..., 1:-1].argmax(dim=-1, keepdim=True) + 1
    left, middle, right = (power.gather(-1, peak + shift).squeeze(-1) for shift in (-1, 0, 1))
    curvature = left - 2.0 * middle + right
    vertex = torch.where(curvature < 0, 0.5 * (left - right) / curvature, torch.zeros_like(curvature))
    apparent = centre + (peak.squeeze(-1) - reach - 1) + vertex
    bias_m = float(((apparent - expected) * bin_m).mean())
    return RangeCalibration(delay_s=2.0 * bias_m / SPEED_OF_LIGHT_M_PER_S)


def _same_grid(product, background) -> None:
    if type(product) is not type(background):
        raise TypeError(f"the background is a {type(background).__name__} and the product a {type(product).__name__}")
    a, b = product.axes, background.axes
    fields = ("waveform", "phasor", "range_bin_count", "doppler_bin_count", "range_bin_m", "num_tx", "num_rx")
    different = [name for name in fields if getattr(a, name) != getattr(b, name)]
    if different:
        raise ValueError(f"the background was recorded on a different grid: {', '.join(different)} differ")
    trailing = tuple(product.data.shape[product.data.dim() - background.data.dim() :])
    if trailing != tuple(background.data.shape):
        raise ValueError(
            f"the background {tuple(background.data.shape)} is not the product's trailing shape "
            f"{tuple(product.data.shape)}"
        )


def subtract_background(product, background):
    """``product - background``, sample for sample, on the product's axes.

    ``background`` is the same kind of product recorded from the same radar
    without the target (or :func:`average_background` of several); a leading
    batch on ``product`` is broadcast against it. Leakage and static clutter
    cancel exactly; noise adds.
    """

    _require_product("product", product)
    _require_product("background", background)
    _same_grid(product, background)
    return replace(product, data=product.data - background.data.to(product.data.device))


def average_background(references: Sequence):
    """The mean of several reference products of one radar, as one product."""

    references = list(references)
    if not references:
        raise ValueError("average_background needs at least one reference product")
    first = _require_product("references[0]", references[0])
    for reference in references[1:]:
        _require_product("reference", reference)
        _same_grid(first, reference)
    return replace(first, data=torch.stack([reference.data for reference in references]).mean(dim=0))
