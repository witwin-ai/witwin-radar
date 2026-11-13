"""Synthetic-aperture imaging: time-domain back-projection, SAR and ISAR apertures.

Back-projection is the reference imaging algorithm, and the only one here. For
every pixel ``p`` it sums, over the slow-time samples ``c`` and the virtual
elements ``v``, the range-compressed sample at the pixel's own round-trip
delay, with that delay's carrier phase removed:

    L_cv(p) = |p - tx_cv| + |p - rx_cv|          two-way path, m
    tau     = L_cv(p) / c0                       round-trip delay, s
    k       = (L_cv(p) / 2 - r_0) / dr           fractional bin on the profile's range axis
    I(p)    = 1/(C V) sum_{c, v} X_cv(k) exp(-j s 2 pi (f tau - a tau^2))

``X_cv(k)`` is the profile linearly interpolated between bins ``floor(k)`` and
``floor(k) + 1``; a ``k`` off the range axis contributes nothing. ``r_0`` and
``dr`` are the profile's published :class:`ProcessingAxes` range origin and
bin, so a zero-padded profile (``range_profile(range_bins=...)``) is the
interpolation grid, and the padding factor bounds the interpolation loss. The
phase law ``(s, f, a)`` is the waveform's and is held on the :class:`Aperture`:

* FMCW (beat phasor, ``s = +1``): the de-chirped tone of a delay ``tau`` has
  phase ``(f_ref + f_0) tau + S tau (t_adc - tau / 2)`` cycles at the first ADC
  sample (``cuda/fmcw_phase.cuh`` states it). The 1/N range DFT at the tone's
  own frequency returns the first sample's phase, so ``f = f_ref + f_0 +
  S t_adc`` and ``a = S / 2`` (the residual video phase).
* OFDM (Channel phasor, ``s = -1``): ``H[n] = C exp(-j 2 pi (f_ref + n df) tau)``
  inverts to a delay peak of phase ``-f_ref tau``: ``f = f_ref``, ``a = 0``.
* Pulsed (Channel phasor): the matched-filter peak is ``C exp(-j 2 pi f_ref
  tau)``: ``f = f_ref``, ``a = 0``. A stepped carrier changes ``f`` per pulse and
  is refused.

Normalisation is the package's amplitude convention: an isolated point target
whose compressed peak is ``|C|`` on every sample images at ``|C|`` times the
range window's coherent gain at its own position (less the linear
interpolation loss, ``<= 2.5 %`` at an 8x padded FMCW profile). Resolution is
the textbook one: ``c / 2B`` in range, ``lambda R / 2L`` across a straight
aperture of length ``L`` (null to peak), and ``lambda / 2 dtheta`` across a
target that turns by ``dtheta``.

Element positions are the virtual array's phase centres in metres, float64:
``tx_m[c, v]`` and ``rx_m[c, v]`` with ``v = tx * num_rx + rx`` (the TX-major
order of a :class:`ProcessingCube`), at the instant the ``c``-th sample's phase
refers to. A moving radar is the start-stop model: the elements are where they
are at that instant, and the motion inside one chirp is not modelled. The
positions are host declarations of the image geometry and carry no derivative;
the image is differentiable with respect to the complex samples in both modes.

A CUDA profile runs the native ``backprojection`` family (one launch; the
backward is its adjoint and the JVP the same linear map of the tangent). A CPU
profile runs the same equation in Torch primitives, vectorised over pixels,
samples and elements; it is the oracle the GPU tests compare against.

ISAR is the same sum in the target's body frame: a target turning by ``R(t)``
about a centre sees the radar move by ``R(t)^T``, so :func:`isar_aperture` maps
the element positions into the body frame and the image grid is the target's
own coordinates at the reference instant.
"""

import math
from dataclasses import dataclass, replace

import torch

from ..cuda import native_ops as _ops
from ..display import compact_repr
from ..policy import first_order_only, refuse_derivative
from ..synthesis.assembly import SPEED_OF_LIGHT_M_PER_S, FmcwSpec, OfdmSpec, PulsedSpec
from .range_doppler import RangeProfile
from .signal import ProcessingAxes, _doppler_sign_from_phasor, _require_complex

#: The frames an aperture and an image are stated in: the world, or the body
#: frame of a turning target at its reference instant.
IMAGE_FRAMES = ("world", "body")

_GEOMETRY_REASON = (
    "the aperture and the pixel grid are the image's declared geometry, and the back-projection family has no "
    "derivative with respect to a position: it carries derivatives of the complex samples only."
)


@dataclass(frozen=True, slots=True, eq=False)
class Aperture:
    """Where every virtual element was for every slow-time sample, and the phase law.

    ``tx_m`` and ``rx_m`` are ``[C, V, 3]`` float64 metres (``V = num_tx *
    num_rx``, TX major), ``times_s`` the ``[C, V]`` instants they hold at, s.
    ``carrier_hz`` and ``curvature_hz_per_s`` are the ``f`` and ``a`` of the
    compressed-sample phase ``f tau - a tau^2`` (cycles), ``phasor`` the cube's
    convention (its sign ``s`` is derived from it, as the Doppler sign is), and
    ``waveform`` / ``reference_frequency_hz`` are checked against the profile.
    """

    __repr__ = compact_repr

    tx_m: torch.Tensor
    rx_m: torch.Tensor
    times_s: torch.Tensor
    carrier_hz: float
    curvature_hz_per_s: float
    phasor: str
    waveform: str
    reference_frequency_hz: float
    num_tx: int
    num_rx: int
    frame: str = "world"

    def __post_init__(self) -> None:
        refuse_derivative(
            "witwin.radar.processing.imaging.Aperture",
            _GEOMETRY_REASON,
            tx_m=self.tx_m,
            rx_m=self.rx_m,
            times_s=self.times_s,
        )
        elements = self.num_tx * self.num_rx
        for name in ("tx_m", "rx_m"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
                raise TypeError(f"Aperture.{name} must be a float64 tensor: a float32 metre at 10 m is a 1 um ulp")
            if value.dim() != 3 or int(value.shape[1]) != elements or int(value.shape[2]) != 3:
                raise ValueError(f"Aperture.{name} is [slow, {elements} elements, 3], got {tuple(value.shape)}")
        if tuple(self.tx_m.shape) != tuple(self.rx_m.shape):
            raise ValueError(f"tx_m {tuple(self.tx_m.shape)} and rx_m {tuple(self.rx_m.shape)} differ")
        if not isinstance(self.times_s, torch.Tensor) or tuple(self.times_s.shape) != tuple(self.tx_m.shape[:2]):
            raise ValueError(f"Aperture.times_s is [slow, elements] = {tuple(self.tx_m.shape[:2])}")
        _doppler_sign_from_phasor(self.phasor)
        if self.frame not in IMAGE_FRAMES:
            raise ValueError(f"Aperture.frame must be one of {IMAGE_FRAMES}, got {self.frame!r}")
        if not (math.isfinite(self.carrier_hz) and math.isfinite(self.curvature_hz_per_s)):
            raise ValueError("the phase law's carrier and curvature must be finite")

    def to(self, device) -> "Aperture":
        """The same aperture with its tensors on ``device``, so an image loop copies no geometry."""

        target = torch.device(device)
        return replace(self, tx_m=self.tx_m.to(target), rx_m=self.rx_m.to(target), times_s=self.times_s.to(target))

    @property
    def slow_count(self) -> int:
        return int(self.tx_m.shape[0])

    @property
    def element_count(self) -> int:
        return int(self.tx_m.shape[1])

    @property
    def phase_sign(self) -> int:
        """``+1`` for the conjugated beat phasor, ``-1`` for Channel's."""

        return _doppler_sign_from_phasor(self.phasor)

    @classmethod
    def from_radar(cls, radar, *, time_s: float = 0.0, velocity_mps=None, pose=None) -> "Aperture":
        """The aperture one frame of ``radar`` sweeps, starting at frame time ``time_s``.

        The sample instants come from the waveform: FMCW slot ``chirp * num_tx +
        tx`` at its ramp start plus its ADC start (the first sample, whose phase
        the range transform returns), an OFDM symbol at ``l T_sym``, a pulse at
        ``l T_pri``. The authored element positions are ``radar.tx_pos`` /
        ``rx_pos``. They move in one of two ways, or not at all:

        * ``velocity_mps``: a straight track, ``x(t) = x + v t``, world m/s.
        * ``pose``: the verbs' ego-motion, ``pose(t) -> (rotation, translation)``,
          a ``(3, 3)`` local-to-world rotation about the radar's authored position
          and a ``(3,)`` world translation: ``x(t) = p0 + R(t) (x - p0) + T(t)``.
          It is called once per distinct instant, on the host.

        A chirp sequence, DDM/BPM MIMO and a stepped pulse carrier have no single
        phase law per element here and are refused by name.
        """

        if velocity_mps is not None and pose is not None:
            raise ValueError("give the radar's motion as velocity_mps or as pose, not both")
        spec = radar.waveform_spec()
        array = radar.system_config.sensors.array
        num_tx, num_rx = int(array.num_tx), int(array.num_rx)
        instants, carrier, curvature = _sample_instants_and_law(spec, num_tx)
        times = (instants + float(time_s)).repeat_interleave(num_rx, dim=1)  # [C, V], v = tx * num_rx + rx
        tx = radar.tx_pos.detach().to(device="cpu", dtype=torch.float64)
        rx = radar.rx_pos.detach().to(device="cpu", dtype=torch.float64)
        slow = int(times.shape[0])
        tx_m = tx.repeat_interleave(num_rx, dim=0).expand(slow, -1, -1)
        rx_m = rx.repeat(num_tx, 1).expand(slow, -1, -1)
        if velocity_mps is not None:
            velocity = torch.as_tensor(velocity_mps, dtype=torch.float64).reshape(1, 1, 3)
            drift = times.unsqueeze(-1) * velocity
            tx_m, rx_m = tx_m + drift, rx_m + drift
        elif pose is not None:
            tx_m, rx_m = _posed(radar, pose, times, tx_m, rx_m)
        return cls(
            tx_m=tx_m.contiguous(),
            rx_m=rx_m.contiguous(),
            times_s=times.contiguous(),
            carrier_hz=carrier,
            curvature_hz_per_s=curvature,
            phasor=str(type(spec).phasor),
            waveform=_waveform_name(spec),
            reference_frequency_hz=float(spec.reference_frequency_hz),
            num_tx=num_tx,
            num_rx=num_rx,
        )


def _waveform_name(spec) -> str:
    if isinstance(spec, FmcwSpec):
        return "fmcw"
    if isinstance(spec, OfdmSpec):
        return "ofdm"
    return "pulsed"


def _sample_instants_and_law(spec, num_tx: int) -> tuple[torch.Tensor, float, float]:
    """``[C, num_tx]`` sample instants from the frame start (s), and the phase law ``(f, a)``."""

    reference = float(spec.reference_frequency_hz)
    if isinstance(spec, FmcwSpec):
        if spec.sequence:
            raise ValueError(
                "a chirp sequence gives each slot its own slope and ADC start, so its compressed samples have no "
                "single phase law; image one profile at a time with an explicit Aperture"
            )
        if spec.simultaneous:
            raise ValueError(
                f"a {spec.mimo} cube records every transmitter on each receiver at once; its demultiplexed virtual "
                "elements' sample instants are not derived here, so build the Aperture from explicit positions"
            )
        profiles = torch.from_numpy(spec.slot_profiles())
        instants = (profiles[:, 0] + profiles[:, 3]).reshape(int(spec.num_chirps), num_tx)
        slope = float(spec.slope_hz_per_s)
        carrier = reference + float(spec.start_offset_hz) + slope * float(spec.t_start_s)
        return instants, carrier, 0.5 * slope
    if isinstance(spec, OfdmSpec):
        slots = torch.arange(int(spec.num_symbols), dtype=torch.float64) * float(spec.symbol_period_s)
        return slots.reshape(-1, 1).expand(-1, num_tx), reference, 0.0
    if isinstance(spec, PulsedSpec):
        if float(spec.frequency_step_hz) != 0.0:
            raise ValueError("a stepped pulse carrier changes the phase law from pulse to pulse and is not imaged here")
        slots = torch.arange(int(spec.num_pulses), dtype=torch.float64) * float(spec.pri_s)
        return slots.reshape(-1, 1).expand(-1, num_tx), reference, 0.0
    raise TypeError(f"no imaging phase law for a {type(spec).__name__}")


def _posed(radar, pose, times: torch.Tensor, tx_m: torch.Tensor, rx_m: torch.Tensor):
    """``p0 + R(t) (x - p0) + T(t)`` at each distinct instant of ``times``."""

    distinct, inverse = torch.unique(times, return_inverse=True)
    rotations, translations = [], []
    for instant in distinct.tolist():
        rotation, translation = pose(float(instant))
        rotations.append(torch.as_tensor(rotation, dtype=torch.float64).detach().cpu().reshape(3, 3))
        translations.append(torch.as_tensor(translation, dtype=torch.float64).detach().cpu().reshape(3))
    rotation = torch.stack(rotations)[inverse]  # [C, V, 3, 3]
    translation = torch.stack(translations)[inverse]  # [C, V, 3]
    origin = torch.as_tensor(radar.position, dtype=torch.float32).to(torch.float64)

    def move(points: torch.Tensor) -> torch.Tensor:
        return origin + torch.einsum("cvij,cvj->cvi", rotation, points - origin) + translation

    return move(tx_m), move(rx_m)


def _rodrigues(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """``[..., 3, 3]`` rotations by ``angle[...]`` rad about the unit ``axis`` (right handed)."""

    x, y, z = axis.tolist()
    cross = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float64)
    outer = torch.outer(axis, axis)
    eye = torch.eye(3, dtype=torch.float64)
    sin = torch.sin(angle)[..., None, None]
    cos = torch.cos(angle)[..., None, None]
    return cos * eye + sin * cross + (1.0 - cos) * outer


def isar_aperture(
    aperture: Aperture, *, center_m, angular_velocity_rad_s=None, rotation=None, reference_time_s: float = 0.0
) -> Aperture:
    """The same aperture seen from the body frame of a turning target.

    The target turns about ``center_m`` (world m): a body point ``x`` is at
    ``center + R(t) (x - center)`` in the world, and the body frame coincides
    with the world at ``reference_time_s``. Give the turn as one of

    * ``angular_velocity_rad_s``: a constant spin vector ``w`` (world, rad/s),
      ``R(t)`` the right-handed rotation by ``|w| (t - t_ref)`` about ``w``;
    * ``rotation``: ``R`` per sample, ``[C, 3, 3]`` or ``[C, V, 3, 3]``.

    Each element maps to ``center + R(t)^T (x - center)``. The cross-range
    resolution of the image is ``lambda / (2 dtheta)`` for a total turn
    ``dtheta``.
    """

    if not isinstance(aperture, Aperture):
        raise TypeError(f"isar_aperture takes an Aperture, got {type(aperture).__name__}")
    if aperture.frame != "world":
        raise ValueError("this aperture is already in a body frame")
    if (angular_velocity_rad_s is None) == (rotation is None):
        raise ValueError("give the target's turn as exactly one of angular_velocity_rad_s and rotation")
    refuse_derivative("witwin.radar.processing.imaging.isar_aperture", _GEOMETRY_REASON, rotation=rotation)
    center = torch.as_tensor(center_m, dtype=torch.float64).reshape(3)
    if rotation is None:
        spin = torch.as_tensor(angular_velocity_rad_s, dtype=torch.float64).reshape(3)
        rate = float(torch.linalg.norm(spin))
        if rate == 0.0:
            raise ValueError("a zero spin turns nothing and gives no cross-range aperture")
        turn = _rodrigues(spin / rate, rate * (aperture.times_s - float(reference_time_s)))
    else:
        turn = torch.as_tensor(rotation, dtype=torch.float64)
        if turn.dim() == 3:
            turn = turn.unsqueeze(1).expand(-1, aperture.element_count, -1, -1)
        if tuple(turn.shape) != (aperture.slow_count, aperture.element_count, 3, 3):
            raise ValueError(
                f"rotation is [slow, 3, 3] or [slow, elements, 3, 3] = "
                f"{(aperture.slow_count, aperture.element_count, 3, 3)}, got {tuple(turn.shape)}"
            )

    def body(points: torch.Tensor) -> torch.Tensor:
        return center + torch.einsum("cvji,cvj->cvi", turn, points - center)

    return replace(aperture, tx_m=body(aperture.tx_m).contiguous(), rx_m=body(aperture.rx_m).contiguous(), frame="body")


@dataclass(frozen=True, slots=True, eq=False)
class SarImage:
    """``[..., *grid]`` complex back-projected image over ``pixels_m [*grid, 3]``.

    Amplitude normalised: a point target images at its compressed peak
    amplitude. ``frame`` is the aperture's (``"world"``, or the turning
    target's ``"body"`` frame).
    """

    __repr__ = compact_repr

    data: torch.Tensor
    pixels_m: torch.Tensor
    axes: ProcessingAxes
    frame: str

    def __post_init__(self) -> None:
        _require_complex("data", self.data)
        grid = tuple(self.pixels_m.shape[:-1])
        if tuple(self.data.shape[self.data.dim() - len(grid) :]) != grid:
            raise ValueError(f"the image's trailing axes must be the pixel grid {grid}, got {tuple(self.data.shape)}")

    @property
    def power(self) -> torch.Tensor:
        """``|I|^2``."""

        return self.data.real.square() + self.data.imag.square()


def _law(axes: ProcessingAxes, aperture: Aperture, groups: int) -> tuple[float, ...]:
    """The scalar arguments of the native family, in its signature order."""

    per_group = aperture.element_count if groups == 1 else 1
    return (
        float(axes.range_origin_m),
        float(axes.range_bin_m),
        1.0 / SPEED_OF_LIGHT_M_PER_S,
        float(aperture.carrier_hz),
        float(aperture.curvature_hz_per_s),
        float(aperture.phase_sign),
        1.0 / (aperture.slow_count * per_group),
    )


class _Backprojection(torch.autograd.Function):
    """The native family. Complex samples cross the boundary as real views."""

    @staticmethod
    def forward(samples, tx, rx, pixels, groups, law):
        batch = int(samples.shape[0])
        image = torch.empty((batch, groups, int(pixels.shape[0])), dtype=samples.dtype, device=samples.device)
        _ops().backprojection_forward(
            torch.view_as_real(samples.resolve_conj().contiguous()),
            tx,
            rx,
            pixels,
            torch.view_as_real(image),
            groups,
            *law,
        )
        return image

    @staticmethod
    def setup_context(ctx, inputs, output):
        samples, tx, rx, pixels, ctx.groups, ctx.law = inputs
        ctx.shape, ctx.dtype = tuple(samples.shape), samples.dtype
        ctx.save_for_backward(tx, rx, pixels)
        ctx.save_for_forward(tx, rx, pixels)

    @staticmethod
    @first_order_only
    def backward(ctx, grad_image):
        tx, rx, pixels = ctx.saved_tensors
        grad = torch.zeros(ctx.shape, dtype=ctx.dtype, device=grad_image.device)
        _ops().backprojection_backward(
            torch.view_as_real(grad_image.resolve_conj().contiguous()),
            tx,
            rx,
            pixels,
            torch.view_as_real(grad),
            ctx.groups,
            *ctx.law,
        )
        return grad, None, None, None, None, None

    @staticmethod
    def jvp(ctx, tangent, *_):
        tx, rx, pixels = ctx.saved_tensors
        out = torch.empty((ctx.shape[0], ctx.groups, int(pixels.shape[0])), dtype=ctx.dtype, device=tangent.device)
        _ops().backprojection_jvp(
            torch.view_as_real(tangent.resolve_conj().contiguous()),
            tx,
            rx,
            pixels,
            torch.view_as_real(out),
            ctx.groups,
            *ctx.law,
        )
        return out


def _backproject_torch(samples, tx, rx, pixels, groups: int, law: tuple[float, ...]) -> torch.Tensor:
    """The same equation as the native family, in Torch primitives: ``[B, V, C, R] -> [B, groups, P]``."""

    origin, bin_m, inverse_speed, carrier, curvature, sign, scale = law
    ranges = int(samples.shape[-1])
    # [V, C, P] geometry, float64.
    path = torch.linalg.vector_norm(pixels - tx.transpose(0, 1).unsqueeze(2), dim=-1) + torch.linalg.vector_norm(
        pixels - rx.transpose(0, 1).unsqueeze(2), dim=-1
    )
    k = (0.5 * path - origin) / bin_m
    valid = (k >= 0.0) & (k <= ranges - 1)
    base = torch.floor(k).clamp(0, ranges - 1)
    fraction = torch.where(valid, k - base, torch.zeros_like(k))
    first = base.to(torch.int64)
    second = (first + 1).clamp(max=ranges - 1)
    tau = path * inverse_speed
    cycles = carrier * tau - curvature * tau * tau
    phasor = torch.polar(valid.to(torch.float64), (-sign * 2.0 * math.pi) * (cycles - torch.floor(cycles)))
    batch = int(samples.shape[0])
    shape = (batch, *first.shape)
    low = samples.gather(-1, first.unsqueeze(0).expand(shape))
    high = samples.gather(-1, second.unsqueeze(0).expand(shape))
    real = samples.real.dtype
    terms = (low + fraction.to(real) * (high - low)) * phasor.to(samples.dtype)
    per_element = terms.sum(dim=2)  # [B, V, P]
    if groups == 1:
        per_element = per_element.sum(dim=1, keepdim=True)
    return per_element * scale


def _require_aperture(profile: RangeProfile, aperture: Aperture) -> torch.Tensor:
    """``[B, V, C, R]`` samples, after checking the profile and the aperture describe one cube."""

    if not isinstance(profile, RangeProfile):
        raise TypeError(f"back-projection takes a RangeProfile, got {type(profile).__name__}")
    if not isinstance(aperture, Aperture):
        raise TypeError(f"back-projection takes an Aperture, got {type(aperture).__name__}")
    axes = profile.axes
    if axes.transmit_channels != axes.num_tx:
        raise ValueError("a DDM or BPM profile is imaged after processing.demultiplex separates its transmitters")
    if (axes.num_tx, axes.num_rx) != (aperture.num_tx, aperture.num_rx):
        raise ValueError(
            f"the profile is a {axes.num_tx} x {axes.num_rx} array and the aperture {aperture.num_tx} x "
            f"{aperture.num_rx}"
        )
    if axes.waveform != aperture.waveform or axes.phasor != aperture.phasor:
        raise ValueError(
            f"the profile is a {axes.waveform} cube in {axes.phasor!r} and the aperture's phase law is for a "
            f"{aperture.waveform} cube in {aperture.phasor!r}"
        )
    if float(axes.reference_frequency_hz) != float(aperture.reference_frequency_hz):
        raise ValueError("the profile and the aperture were built at different reference frequencies")
    data = profile.data
    if data.dim() < 4 or tuple(data.shape[-4:-2]) != (axes.num_tx, axes.num_rx):
        raise ValueError(f"the profile is [..., tx, rx, slow, range]; got shape {tuple(data.shape)}")
    if int(data.shape[-2]) != aperture.slow_count:
        raise ValueError(
            f"the profile holds {int(data.shape[-2])} slow-time samples and the aperture {aperture.slow_count}; a "
            "slow-time filter that shortened the profile also changed its sample instants"
        )
    return data.reshape(-1, aperture.element_count, aperture.slow_count, int(data.shape[-1]))


def _backproject(samples, aperture: Aperture, pixels, axes: ProcessingAxes, groups: int) -> torch.Tensor:
    """``[B, V, C, R]`` samples -> ``[B, groups, P]``, on the samples' device."""

    device = samples.device
    tx = aperture.tx_m.to(device).contiguous()
    rx = aperture.rx_m.to(device).contiguous()
    pixels = pixels.to(device=device, dtype=torch.float64).contiguous()
    law = _law(axes, aperture, groups)
    if samples.is_cuda:
        if samples.dtype not in (torch.complex64, torch.complex128):
            raise TypeError(f"the CUDA back-projection takes complex64 or complex128 samples, got {samples.dtype}")
        return _Backprojection.apply(samples, tx, rx, pixels, groups, law)
    return _backproject_torch(samples, tx, rx, pixels, groups, law)


def backproject(profile: RangeProfile, aperture: Aperture, pixels_m: torch.Tensor) -> SarImage:
    """Back-project a range profile onto ``pixels_m [*grid, 3]`` (metres, the aperture's frame).

    ``profile`` is ``[..., TX, RX, C, R]``, as :func:`range_profile` publishes
    it from a :class:`ProcessingCube`; any leading batch is kept. Zero pad the
    range transform (``range_profile(range_bins=8 * N)`` on an FMCW beat or OFDM
    cube, ``range_oversample`` on a pulsed one): the image reads it by linear
    interpolation, and on the transform's own grid neighbouring bins of an FMCW
    tone are almost in antiphase. Differentiable with respect to the samples in
    both modes; the geometry is refused as a derivative carrier.
    """

    samples = _require_aperture(profile, aperture)
    refuse_derivative("witwin.radar.processing.imaging.backproject", _GEOMETRY_REASON, pixels_m=pixels_m)
    if not isinstance(pixels_m, torch.Tensor) or pixels_m.dim() < 1 or int(pixels_m.shape[-1]) != 3:
        raise ValueError("pixels_m is a [..., 3] tensor of image points in metres")
    grid = tuple(pixels_m.shape[:-1])
    image = _backproject(samples, aperture, pixels_m.reshape(-1, 3), profile.axes, groups=1)
    data = image.reshape(*profile.data.shape[:-4], *grid)
    return SarImage(data=data, pixels_m=pixels_m.to(torch.float64), axes=profile.axes, frame=aperture.frame)


def element_responses(profile: RangeProfile, aperture: Aperture, point_m: torch.Tensor) -> torch.Tensor:
    """``[..., TX, RX]``: each virtual element's back-projected response at one point.

    The slow-time mean of the element's phase-compensated compressed sample at
    the point's delay: the per-element terms of :func:`backproject` before the
    element sum. For an ideal array and a point scatterer at ``point_m`` every
    entry is the same complex amplitude; what differs is the channel's own
    complex gain, which is what array calibration measures.
    """

    samples = _require_aperture(profile, aperture)
    refuse_derivative("witwin.radar.processing.imaging.element_responses", _GEOMETRY_REASON, point_m=point_m)
    point = torch.as_tensor(point_m, dtype=torch.float64).reshape(1, 3)
    response = _backproject(samples, aperture, point, profile.axes, groups=aperture.element_count)
    return response.reshape(*profile.data.shape[:-2])
