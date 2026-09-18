"""Constant-false-alarm-rate detection, batched, and the point cloud.

Every detector here takes ``[..., D, R]`` with an arbitrary leading batch and
returns :class:`Detections`, so a beam cube, a per-pair map and a single slice
are the same call with no Python loop over beams. :func:`ca_cfar_1d` is the
range-only form, over ``[..., R]``.

The threshold law is the standard cell-averaging one and is published rather
than buried: with ``N`` training cells and a design false-alarm probability
``P_fa``, the scale is

    ``alpha = N * (P_fa ** (-1 / N) - 1)``

which for an exponentially distributed power estimate gives
``P_fa = (1 + alpha / N) ** (-N)`` exactly. That identity is what the
false-alarm-rate test measures against, so the constant cannot be tuned to make
a test pass without the measured rate moving with it.

Note the units the identity holds in. It is a statement about POWER: the noise
estimate must be an average of ``|x| ** 2``. Fed a magnitude or a decibel map -
which is what every existing caller does - the detector still adapts to the
local level, but the nominal ``P_fa`` is a design parameter rather than a
prediction. That is stated here because it was not stated anywhere before.

The identity is equally a statement about a MEAN. :func:`os_cfar` scales an
ordered statistic with that same cell-averaging constant, so the rate it
achieves is NOT the ``pfa`` it is handed. Its own docstring gives the law its
rate does follow, the measured ratio, and why the constant is left alone.

**Every detector here is explicitly non-differentiable and refuses a derivative
at its entry.** This deliberately gives up a derivative that does exist: the
threshold is a ring average of the training cells, so it is a perfectly smooth
function of the map. What the stage OUTPUTS is a detection decision, and the
mask that carries it is a bool with no derivative at all; publishing a live
threshold beside a severed mask is how a caller ends up optimising the level
and believing they are optimising the detection. A differentiable-CFAR
surrogate - a soft threshold, a sigmoid mask - is a modelling decision with its
own design rather than something a detector may choose.
``docs/dev/radar-ad-capability-matrix.md`` carries the same reason as four
``REF`` rows.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..policy import refuse_derivative
from .angle import AOA_ROUTES, ArrayGeometry, tdm_compensate
from .range_doppler import RangeDopplerMap
from .signal import real_dtype_of

#: Why no detector here has a derivative. Written once and quoted by all four
#: entries, so the four cannot drift into four explanations of one decision.
_CFAR_REASON = (
    "a detection is the output of a threshold COMPARISON, a step function of "
    "the map with a zero derivative almost everywhere and an undefined one "
    "exactly where the detection changes."
)


@dataclass(frozen=True, slots=True, eq=False)
class Detections:
    """A boolean mask and the adaptive threshold that produced it.

    Both have the shape of the map they were computed on, leading batch
    included. The threshold travels with the mask because a detection without
    the level it beat is not reproducible, and because the point-cloud stage
    reports an energy relative to it.
    """

    mask: torch.Tensor
    threshold: torch.Tensor

    def __post_init__(self) -> None:
        if self.mask.dtype != torch.bool:
            raise TypeError(f"mask must be bool, got {self.mask.dtype}")
        if tuple(self.mask.shape) != tuple(self.threshold.shape):
            raise ValueError(
                f"mask is {tuple(self.mask.shape)} but threshold is "
                f"{tuple(self.threshold.shape)}; they describe the same map"
            )

    @property
    def count(self) -> torch.Tensor:
        """The number of detections, as a DEVICE scalar.

        Deliberately not an ``int``. Reading it to the host is a
        synchronization, and this package refuses to add one implicitly - a
        caller that needs the number asks for it and pays for it visibly.
        """

        return self.mask.sum()


def _alpha(n_train: int, pfa: float) -> float:
    return n_train * (float(pfa) ** (-1.0 / n_train) - 1.0)


def _real_values(data: torch.Tensor) -> torch.Tensor:
    values = torch.abs(data) if torch.is_complex(data) else data
    return values.to(real_dtype_of(data))


def _as_batch(values: torch.Tensor, rank: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    if values.dim() < rank:
        raise ValueError(
            f"the map must be [..., {'doppler, range' if rank == 2 else 'range'}]; got shape {tuple(values.shape)}"
        )
    leading = tuple(values.shape[:-rank])
    return values.reshape(-1, *values.shape[-rank:]), leading


def _replicate_pad_2d(data: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    return F.pad(data.unsqueeze(1), (pad_w, pad_w, pad_h, pad_h), mode="replicate")


def ca_cfar(
    rd_map: torch.Tensor,
    *,
    guard_cells: tuple[int, int] = (2, 3),
    training_cells: tuple[int, int] = (4, 6),
    pfa: float = 1e-3,
) -> Detections:
    """Cell-averaging CFAR over ``[..., D, R]``, from two pooled averages.

    The ring sum is the outer mean times the outer count minus the guard mean
    times the guard count, each mean one ``avg_pool2d`` launch. Edges are
    handled by replicate padding, so a cell at the border sees a ring of the
    same size rather than a smaller and therefore noisier one. The exact
    summed-area form of the same estimator is the test oracle in
    ``tests/support/reference_cfar.py``; the two agree up to float
    re-association.
    """

    refuse_derivative("witwin.radar.processing.detection.ca_cfar", _CFAR_REASON, rd_map=rd_map)
    values = _real_values(rd_map)
    flat, leading = _as_batch(values, 2)
    doppler, ranges = int(flat.shape[-2]), int(flat.shape[-1])
    gd, gr = int(guard_cells[0]), int(guard_cells[1])
    td, tr = int(training_cells[0]), int(training_cells[1])
    outer_d, outer_r = gd + td, gr + tr
    n_outer = (2 * outer_d + 1) * (2 * outer_r + 1)
    n_guard = (2 * gd + 1) * (2 * gr + 1)
    n_train = n_outer - n_guard
    if n_train < 1:
        raise ValueError(
            f"guard_cells={guard_cells} and training_cells={training_cells} leave "
            "no training cells to estimate the noise from"
        )
    alpha = _alpha(n_train, pfa)

    def pooled(size: tuple[int, int]) -> torch.Tensor:
        padded = _replicate_pad_2d(flat, size[0] // 2, size[1] // 2)
        return F.avg_pool2d(padded, kernel_size=size, stride=1).squeeze(1)

    outer_mean = pooled((2 * outer_d + 1, 2 * outer_r + 1))
    guard_mean = pooled((2 * gd + 1, 2 * gr + 1))
    noise = (outer_mean * n_outer - guard_mean * n_guard) / n_train
    threshold = (alpha * noise).reshape(*leading, doppler, ranges)
    return Detections(mask=values > threshold, threshold=threshold)


def os_cfar(
    rd_map: torch.Tensor,
    *,
    guard_cells: tuple[int, int] = (2, 3),
    training_cells: tuple[int, int] = (4, 6),
    rank_fraction: float = 0.75,
    pfa: float = 1e-3,
) -> Detections:
    """Ordered-statistic CFAR over ``[..., D, R]``.

    The noise estimate is the ``rank_fraction``-th ordered training sample
    rather than their mean, which is what makes it robust when a second target
    sits inside the training ring and would otherwise raise the threshold above
    the first.

    This is the memory outlier of the three: it materialises every cell's
    training patch, ``[batch, D * R, n_outer]``, and sorts it. The cost is
    recorded rather than hidden.

    ``pfa`` IS NOT THE ACHIEVED RATE HERE. The scale is the cell-averaging
    constant ``alpha = N (pfa ** (-1 / N) - 1)``, which inverts the false-alarm
    law of a MEAN of ``N`` exponential samples; this detector thresholds against
    the ``k``-th smallest of them instead, whose exact rate on exponential power
    is Rohling's

        ``P_fa = prod_{i=0}^{k-1} (N - i) / (N - i + alpha)``

    with ``k = min(int(rank_fraction * N), N - 1) + 1``. At the defaults that
    lands well BELOW the declared number - measured 1.88e-3 for ``pfa=1e-2`` and
    1.03e-4 for ``pfa=1e-3`` at ``N=40``, ``k=31``, ratios 0.19 and 0.10 - so
    the error is in the conservative direction and costs sensitivity rather than
    producing surprise false alarms. Read ``pfa`` on this function as a design
    constant in the cell-averaging parameterisation; when the achieved rate is
    what matters, invert the law above.

    The constant is deliberately left alone rather than re-solved for the
    ordered statistic: moving it is a numerical change that owes its own
    decision and its own golden update.
    """

    refuse_derivative(
        "witwin.radar.processing.detection.os_cfar",
        _CFAR_REASON + " The ordered statistic adds a second discrete decision on top of it: "
        "which training sample the threshold is read from is chosen by a sort.",
        rd_map=rd_map,
    )
    values = _real_values(rd_map)
    flat, leading = _as_batch(values, 2)
    batch = int(flat.shape[0])
    doppler, ranges = int(flat.shape[-2]), int(flat.shape[-1])
    gd, gr = int(guard_cells[0]), int(guard_cells[1])
    td, tr = int(training_cells[0]), int(training_cells[1])
    outer_d, outer_r = gd + td, gr + tr
    n_outer = (2 * outer_d + 1) * (2 * outer_r + 1)
    n_train = n_outer - (2 * gd + 1) * (2 * gr + 1)
    if n_train < 1:
        raise ValueError(
            f"guard_cells={guard_cells} and training_cells={training_cells} leave "
            "no training cells to estimate the noise from"
        )
    alpha = _alpha(n_train, pfa)

    # A POSITION mask, not a value threshold: the guard band is where it is, not
    # wherever the values happen to be large.
    keep = torch.ones((2 * outer_d + 1, 2 * outer_r + 1), dtype=torch.bool, device=flat.device)
    keep[td : td + 2 * gd + 1, tr : tr + 2 * gr + 1] = False

    padded = _replicate_pad_2d(flat, outer_d, outer_r)
    patches = F.unfold(padded, kernel_size=(2 * outer_d + 1, 2 * outer_r + 1), stride=1).transpose(1, 2)
    training, _ = torch.sort(patches[:, :, keep.reshape(-1)], dim=-1)
    index = min(int(rank_fraction * n_train), int(training.shape[-1]) - 1)
    threshold = (alpha * training[:, :, index]).reshape(batch, doppler, ranges)
    threshold = threshold.reshape(*leading, doppler, ranges)
    return Detections(mask=values > threshold, threshold=threshold)


def ca_cfar_1d(
    profile: torch.Tensor, *, guard_cells: int = 2, training_cells: int = 8, pfa: float = 1e-3
) -> Detections:
    """Range-only cell-averaging CFAR over ``[..., R]``.

    Same law, one axis, and the same replicate-padded ring so a detection at
    the first range bin is not systematically favoured.
    """

    refuse_derivative("witwin.radar.processing.detection.ca_cfar_1d", _CFAR_REASON, profile=profile)
    values = _real_values(profile)
    flat, leading = _as_batch(values, 1)
    ranges = int(flat.shape[-1])
    guard = int(guard_cells)
    train = int(training_cells)
    outer = guard + train
    n_train = 2 * train
    if n_train < 1:
        raise ValueError(f"training_cells={training_cells} leaves no training cells to estimate the noise from")
    alpha = _alpha(n_train, pfa)

    padded = F.pad(flat.unsqueeze(1), (outer, outer), mode="replicate")
    integral = F.pad(padded, (1, 0), mode="constant", value=0).cumsum(dim=-1)
    device = flat.device
    centre = torch.arange(ranges, device=device, dtype=torch.int64) + outer
    outer_sum = integral[..., centre + outer + 1] - integral[..., centre - outer]
    guard_sum = integral[..., centre + guard + 1] - integral[..., centre - guard]
    noise = (outer_sum - guard_sum) / n_train
    threshold = (alpha * noise).squeeze(1).reshape(*leading, ranges)
    return Detections(mask=values > threshold, threshold=threshold)


#: Why the point-cloud stage has no derivative. One statement, quoted by the
#: stage and by the ``topk`` thinning inside it. What a derivative through the
#: selection would describe is the value AT a frozen selection, not the answer
#: moving: perturb the map far enough for the selection to change and it
#: predicts nothing about the new point list, including its length.
_SELECTION_REASON = (
    "which cells become points is a discrete selection - an argwhere over a "
    "threshold mask, thinned by a topk - so the published values are values AT "
    "a frozen choice, and even the LENGTH of the answer is data dependent."
)


#: The column order of :meth:`PointCloud.as_columns`, stated as data.
POINT_CLOUD_COLUMNS = ("x", "y", "z", "velocity_mps", "energy", "range_m")


@dataclass(frozen=True, slots=True, eq=False)
class PointCloud:
    """``N`` detections placed in the array's LOCAL frame, in metres.

    ``velocity_mps`` is in the canonical closing-positive convention, the same
    one :class:`RangeDopplerMap`'s Doppler axis publishes. ``energy`` is in
    decibels relative to one unit of the map's own amplitude, which is the
    amplitude convention the range and Doppler stages publish, so a level here
    is comparable across waveforms.
    """

    xyz: torch.Tensor
    velocity_mps: torch.Tensor
    energy: torch.Tensor
    range_m: torch.Tensor

    def __post_init__(self) -> None:
        if self.xyz.dim() != 2 or int(self.xyz.shape[1]) != 3:
            raise ValueError(f"xyz must be [N, 3]; got {tuple(self.xyz.shape)}")
        count = int(self.xyz.shape[0])
        for name in ("velocity_mps", "energy", "range_m"):
            value = getattr(self, name)
            if value.dim() != 1 or int(value.shape[0]) != count:
                raise ValueError(f"{name} must be [{count}] to match xyz; got {tuple(value.shape)}")

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def device(self) -> torch.device:
        return self.xyz.device

    def as_columns(self) -> torch.Tensor:
        """``[N, 6]`` in :data:`POINT_CLOUD_COLUMNS` order.

        The flat form a fixed-size detection batch is assembled from. It is a
        VIEW-building stack, not the record's storage: the named fields are the
        contract.
        """

        return torch.stack(
            (self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2], self.velocity_mps, self.energy, self.range_m), dim=1
        )

    @classmethod
    def empty(cls, *, device: torch.device) -> "PointCloud":
        zero = torch.zeros((0,), dtype=torch.float64, device=device)
        return cls(
            xyz=torch.zeros((0, 3), dtype=torch.float64, device=device),
            velocity_mps=zero,
            energy=zero.clone(),
            range_m=zero.clone(),
        )

    def select(self, keep: torch.Tensor) -> "PointCloud":
        """The subset the boolean mask ``keep`` selects, as a new record."""

        return PointCloud(
            xyz=self.xyz[keep],
            velocity_mps=self.velocity_mps[keep],
            energy=self.energy[keep],
            range_m=self.range_m[keep],
        )


def range_gate_mask(axes, gate_m: tuple[float, float] | None) -> torch.Tensor | None:
    """``[R]`` bool: which range bins a gate in METRES admits.

    ``None`` admits everything. The gate is half open, ``lo <= r < hi``, so two
    abutting gates partition the axis exactly once.
    """

    if gate_m is None:
        return None
    low, high = float(gate_m[0]), float(gate_m[1])
    if not high > low:
        raise ValueError(f"the range gate must be (low_m, high_m) with high > low, got {gate_m!r}")
    return (axes.range_m >= low) & (axes.range_m < high)


def point_cloud(
    detections: Detections,
    rd: RangeDopplerMap,
    array: ArrayGeometry,
    *,
    route: str = "phase_comparison",
    fft_size: int = 64,
    range_gate_m: tuple[float, float] | None = None,
    max_points: int | None = None,
    positive_velocity_only: bool = False,
    energy_floor: float = 1e-6,
) -> PointCloud:
    """``Detections[D, R]`` plus ``RangeDopplerMap[TX, RX, D, R]`` -> ``PointCloud``.

    The detection mask is rank 2 because a detection is a cell of the coherently
    combined map: the AoA estimate for that cell is what needs every element,
    and estimating an angle per element per cell first would be estimating the
    angle of the noise.

    ``route`` names the angle estimator explicitly, from
    :data:`~witwin.radar.processing.angle.AOA_ROUTES`, because a front-end
    change that silently swaps the estimator is a change of answer with no
    change of call. ``range_gate_m`` is a pair of distances in METRES read
    against :attr:`ProcessingAxes.range_m`; a bin index never appears in the
    signature. Everything published is float64 and on the input device.
    """

    if not isinstance(detections, Detections):
        raise TypeError(f"detections must be a Detections record, got {type(detections).__name__}")
    if not isinstance(rd, RangeDopplerMap):
        raise TypeError(f"rd must be a RangeDopplerMap, got {type(rd).__name__}")
    if not isinstance(array, ArrayGeometry):
        raise TypeError(f"array must be an ArrayGeometry, got {type(array).__name__}")
    if route not in AOA_ROUTES:
        raise ValueError(f"route must be one of {tuple(sorted(AOA_ROUTES))}, got {route!r}")
    # Before the shape checks and before any arithmetic, so the refusal fires
    # with no PointCloud in existence and no transform paid for. The type
    # checks above are all that precede it, and only because the guard has to
    # know these are the records whose tensors it is naming.
    refuse_derivative(
        "witwin.radar.processing.detection.point_cloud",
        _SELECTION_REASON,
        rd_data=rd.data,
        detection_threshold=detections.threshold,
    )
    mask = detections.mask
    if mask.dim() != 2:
        raise ValueError(
            "the detection mask is [doppler, range]: one cell of the combined "
            f"map per detection; got shape {tuple(mask.shape)}"
        )
    data = rd.data
    if data.dim() < 3:
        raise ValueError(f"the map must be [*pair, doppler, range]; got shape {tuple(data.shape)}")
    if tuple(data.shape[-2:]) != tuple(mask.shape):
        raise ValueError(
            f"the map is {tuple(data.shape[-2:])} but the mask is {tuple(mask.shape)}; they describe different grids"
        )
    pairs = array.sensor_pair_count
    flat = data.reshape(pairs, int(mask.shape[0]), int(mask.shape[1]))
    combined = flat.sum(dim=0)
    energy_db = 20 * torch.log10(combined.abs() + float(energy_floor))

    axes = rd.axes
    gate = range_gate_mask(axes, range_gate_m)
    if gate is not None:
        mask = mask & gate.reshape(1, -1).to(mask.device)
    if max_points is not None:
        mask = _keep_strongest(mask, energy_db, int(max_points))

    # The one host observation this stage makes: a point cloud has a data
    # dependent length, so the row list cannot stay on the device.
    cells = torch.argwhere(mask)
    if int(cells.shape[0]) == 0:
        return PointCloud.empty(device=data.device)
    doppler_index = cells[:, 0]
    range_index = cells[:, 1]

    range_m = axes.range_m.to(data.device).index_select(0, range_index)
    velocity = axes.velocity_mps.to(data.device).index_select(0, doppler_index)
    energy = energy_db[doppler_index, range_index].to(torch.float64)

    aoa_input = flat[:, doppler_index, range_index]
    aoa_input = tdm_compensate(aoa_input, velocity, array, axes)
    cosines = AOA_ROUTES[route](aoa_input, array, fft_size=fft_size).to(torch.float64)

    cloud = PointCloud(
        xyz=(cosines * range_m.reshape(1, -1)).transpose(0, 1).contiguous(),
        velocity_mps=velocity,
        energy=energy,
        range_m=range_m,
    )
    # A zero boresight cosine is the estimator's way of saying the pair of
    # angles it found describes no real direction; those rows are not points.
    cloud = cloud.select(cosines[1] != 0)
    if positive_velocity_only and len(cloud) > 0:
        cloud = cloud.select(cloud.velocity_mps > 0)
    return cloud


def _keep_strongest(mask: torch.Tensor, energy: torch.Tensor, max_points: int) -> torch.Tensor:
    """Thin a detection mask to its ``max_points`` strongest cells.

    Done on the DEVICE with ``topk`` over the flattened map rather than by
    reading the detection list to the host and sorting it there.

    Guarded again even though :func:`point_cloud` already refused at its entry
    and is this function's only caller. Defence in depth is the right shape for
    a wall: a future second caller inherits the refusal instead of quietly
    reopening the hole, and the ``topk`` here is the peak selection the plan
    names by that name.
    """

    refuse_derivative("witwin.radar.processing.detection._keep_strongest", _SELECTION_REASON, energy=energy)
    if max_points < 0:
        raise ValueError(f"max_points must be non-negative, got {max_points}")
    flat_mask = mask.reshape(-1)
    total = int(flat_mask.shape[0])
    if max_points >= total:
        return mask
    scored = torch.where(flat_mask, energy.reshape(-1), torch.full_like(energy.reshape(-1), -torch.inf))
    keep = torch.zeros_like(flat_mask)
    if max_points > 0:
        keep[torch.topk(scored, max_points).indices] = True
    return (keep & flat_mask).reshape(mask.shape)


def combine_incoherent(cubes) -> torch.Tensor:
    """``sum_j |cube_j|^2``: the power sum of independently exported components.

    Returns a REAL tensor. That is the point of the operation and it is not a
    convenience: the result has no phase, cannot be fed back into a coherent
    stage, and a caller that wanted an amplitude has to say which phase it
    meant.

    Incoherent combination is a physical claim: the components have no fixed
    phase relationship, so their POWERS add and their amplitudes do not. It is
    a post-synthesis statement about an ensemble and belongs here rather than
    inside a waveform kernel, whose whole contract is that it sums complex
    amplitudes over a pair segment. The physically honest incoherent model - a
    per-realization random phase drawn into the scatter response, so that an
    ensemble of frames averages to the power sum while each frame keeps its
    speckle - needs a native RNG and a seed contract consistent with the
    frontend's, and is not what this function does.

    The magnitude is formed as ``re^2 + im^2`` rather than as ``abs()**2``
    because ``abs`` is not differentiable at the origin, and an exactly zero
    entry is the normal case here: every masked row of every component export
    contributes one.
    """

    listed = list(cubes)
    if not listed:
        raise ValueError(
            "combine_incoherent needs at least one cube; an empty sum is not a "
            "zero-power scene, it is a caller that forgot to export anything"
        )
    total = None
    for index, cube in enumerate(listed):
        if not isinstance(cube, torch.Tensor):
            raise TypeError(f"cube {index} must be a torch.Tensor, got {type(cube).__name__}")
        if cube.shape != listed[0].shape:
            raise ValueError(
                f"cube {index} has shape {tuple(cube.shape)} but cube 0 has "
                f"{tuple(listed[0].shape)}; a power sum is elementwise and the "
                "components must share their axes"
            )
        if cube.device != listed[0].device:
            raise ValueError(f"cube {index} is on {cube.device} but cube 0 is on {listed[0].device}")
        power = cube.real * cube.real + cube.imag * cube.imag if cube.is_complex() else cube * cube
        total = power if total is None else total + power
    return total


def combine_coherent(cube: torch.Tensor) -> torch.Tensor:
    """``|sum_p cube[p]|``: the coherent (boresight) sum over sensor pairs, then abs.

    ``cube`` is ``[..., D, R]`` complex with the sensor pairs in the leading
    axes; every leading axis is folded into one pair axis and summed with the
    phases intact, which is the boresight beam. Returns a REAL ``[D, R]``
    magnitude for a detector that thresholds one map across the virtual array.
    """

    return cube.reshape(-1, *cube.shape[-2:]).sum(dim=0).abs()
