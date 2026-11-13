"""Constant-false-alarm-rate detection, batched, and the point cloud.

Every detector here takes ``[..., D, R]`` with an arbitrary leading batch and
returns :class:`Detections`, so a beam cube, a per-pair map and a single slice
are the same call with no Python loop over beams. :func:`ca_cfar_1d` is the
range-only form, over ``[..., R]``.

Every detector is SQUARE LAW. A complex map is detected on ``|x|^2`` and a
real map is read as a power map; the noise model behind every threshold
constant is the one ``|x|^2`` of circular complex Gaussian noise has, an
exponential variable per cell. The constants are SOLVED from each detector's
exact false-alarm law (:func:`cfar_false_alarm_probability`) for the training
count each cell actually has, so the ``pfa`` a detector is handed is the rate
it achieves on that noise, at the edges as well as in the interior. The tests
measure it over millions of noise cells for every detector.

Edges follow the axes' own topology: Doppler is a circular DFT axis and the
training ring wraps there; range is not circular and the ring is truncated at
its ends, with the constant re-solved for the shorter ring.

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

import functools
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..cuda import native_ops as _ops
from ..display import compact_repr
from ..policy import refuse_derivative
from .angle import AOA_ROUTES, ArrayGeometry, tdm_compensate
from .range_doppler import RangeDopplerMap

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

    __repr__ = compact_repr

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


#: The detector families, by the noise-level statistic each one thresholds
#: against: the ring MEAN (``"ca"``), the GREATER or SMALLER of the two
#: half-ring means (``"go"``, ``"so"``), or an ORDERED statistic (``"os"``).
CFAR_KINDS = ("ca", "go", "so", "os")


def _power(data: torch.Tensor) -> torch.Tensor:
    """The square-law detector input: ``|x|^2`` for complex IQ, the map itself otherwise.

    Formed as ``re^2 + im^2``. A real input is read as a POWER map already;
    a magnitude or a decibel map is not one, and every false-alarm law in this
    module would then be a statement about a different distribution.
    """

    if torch.is_complex(data):
        return data.real.square() + data.imag.square()
    if not data.is_floating_point():
        raise TypeError(f"a detector input is a complex map or a real power map, got {data.dtype}")
    return data


def _as_batch(values: torch.Tensor, rank: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    if values.dim() < rank:
        raise ValueError(
            f"the map must be [..., {'doppler, range' if rank == 2 else 'range'}]; got shape {tuple(values.shape)}"
        )
    leading = tuple(values.shape[:-rank])
    return values.reshape(-1, *values.shape[-rank:]), leading


@dataclass(frozen=True, slots=True)
class _Ring:
    """The training ring of one window, as offsets from the cell under test.

    ``gd``/``gr`` are the guard half widths and ``td``/``tr`` the training
    depths along Doppler and range. The ring is split into two HALVES along
    range: the lagging half is every training cell at a negative range offset
    plus the zero-range-offset cells at negative Doppler offsets, the leading
    half is its mirror image. Each half is a union of disjoint rectangles,
    listed as ``(d0, d1, r0, r1)`` inclusive offset bounds, so a half sum is a
    sum of non-negative block sums with no cancellation - a strong return in
    the guard band cannot drown the noise estimate in rounding.
    """

    gd: int
    gr: int
    td: int
    tr: int

    @property
    def outer_d(self) -> int:
        return self.gd + self.td

    @property
    def outer_r(self) -> int:
        return self.gr + self.tr

    def blocks(self, side: int) -> tuple[tuple[int, int, int, int], ...]:
        """The rectangles of the lagging (``side=-1``) or leading (``+1``) half."""

        od, orr, gd, gr, td, tr = self.outer_d, self.outer_r, self.gd, self.gr, self.td, self.tr
        found = []
        if tr > 0:
            found.append((-od, od, -orr, -gr - 1) if side < 0 else (-od, od, gr + 1, orr))
        if td > 0 and gr > 0:
            columns = (-gr, -1) if side < 0 else (1, gr)
            found.append((-od, -gd - 1, *columns))
            found.append((gd + 1, od, *columns))
        if td > 0:
            found.append((-od, -gd - 1, 0, 0) if side < 0 else (gd + 1, od, 0, 0))
        return tuple(found)

    def column_counts(self, side: int) -> torch.Tensor:
        """``[2 * outer_r + 1]`` int64: training cells of one half per range offset."""

        counts = torch.zeros(2 * self.outer_r + 1, dtype=torch.int64)
        for d0, d1, r0, r1 in self.blocks(side):
            counts[r0 + self.outer_r : r1 + self.outer_r + 1] += d1 - d0 + 1
        return counts

    def training_offsets(self) -> torch.Tensor:
        """``[N]`` int64 positions of the training cells in the row-major window patch."""

        width = 2 * self.outer_r + 1
        keep = torch.zeros((2 * self.outer_d + 1, width), dtype=torch.bool)
        for side in (-1, 1):
            for d0, d1, r0, r1 in self.blocks(side):
                keep[d0 + self.outer_d : d1 + self.outer_d + 1, r0 + self.outer_r : r1 + self.outer_r + 1] = True
        return torch.argwhere(keep.reshape(-1)).reshape(-1)


def _ring(guard_cells, training_cells, *, rank: int) -> _Ring:
    if rank == 2:
        gd, gr = (int(value) for value in guard_cells)
        td, tr = (int(value) for value in training_cells)
    else:
        gd, gr, td, tr = 0, int(guard_cells), 0, int(training_cells)
    if min(gd, gr, td, tr) < 0:
        raise ValueError(f"guard_cells={guard_cells} and training_cells={training_cells} must be non-negative")
    ring = _Ring(gd=gd, gr=gr, td=td, tr=tr)
    if not ring.blocks(-1) and not ring.blocks(1):
        raise ValueError(
            f"guard_cells={guard_cells} and training_cells={training_cells} leave "
            "no training cells to estimate the noise from"
        )
    return ring


def _half_counts(ring: _Ring, ranges: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[R]`` int64 training-cell counts of each half, after range truncation.

    The Doppler axis wraps, so a half never loses a cell to it. The range axis
    does not: a cell near either end of the range axis keeps only the training
    cells that lie on the map, and its threshold constant is solved for that
    count rather than borrowed from the interior.
    """

    offsets = torch.arange(-ring.outer_r, ring.outer_r + 1, dtype=torch.int64)
    position = torch.arange(ranges, dtype=torch.int64).reshape(-1, 1) + offsets.reshape(1, -1)
    inside = ((position >= 0) & (position < ranges)).to(torch.int64)
    return inside @ ring.column_counts(-1), inside @ ring.column_counts(1)


def _bisect(probability, target: float, size: int) -> torch.Tensor:
    """The ``alpha >= 0`` at which a DECREASING false-alarm law equals ``target``.

    ``probability`` maps a ``[size]`` float64 tensor of scale constants to
    ``[size]`` false-alarm probabilities. Bracketed by doubling, then bisected
    to float64 resolution. Host-side scalar arithmetic on a handful of
    distinct ring sizes, run once per detector configuration and cached.
    """

    low = torch.zeros(size, dtype=torch.float64)
    high = torch.ones(size, dtype=torch.float64)
    for _ in range(200):
        above = probability(high) > target
        if not bool(above.any()):
            break
        high = torch.where(above, high * 2.0, high)
    for _ in range(56):
        middle = 0.5 * (low + high)
        above = probability(middle) > target
        low = torch.where(above, middle, low)
        high = torch.where(above, high, middle)
    return 0.5 * (low + high)


def os_rank_index(count: torch.Tensor, rank_fraction: float) -> torch.Tensor:
    """The 0-based ordered-statistic index ``min(int(rank_fraction * N), N - 1)``."""

    return torch.minimum(torch.floor(count.to(torch.float64) * float(rank_fraction)).to(torch.int64), count - 1)


def cfar_false_alarm_probability(
    kind: str, alpha: torch.Tensor, n_lag: torch.Tensor, n_lead: torch.Tensor, *, rank_fraction: float = 0.75
) -> torch.Tensor:
    """The exact false-alarm probability of a detector on exponential noise.

    The noise model is the square-law one: every cell's power is an
    independent exponential variable of a common unknown mean, which is what
    ``|x|^2`` of circular complex Gaussian noise is. ``n_lag``/``n_lead`` are
    the training-cell counts of the two half rings and ``alpha`` is the scale
    applied to the detector's noise statistic. All laws are scale free, which
    is the CFAR property itself.

    * ``"ca"``: threshold ``alpha * S / N`` on the ring sum ``S`` of
      ``N = n_lag + n_lead`` cells, ``P = (1 + alpha / N) ** -N``.
    * ``"so"``: ``alpha * min(S1 / n1, S2 / n2)``. With ``A_i = S_i / n_i``,
      ``P = E[exp(-alpha min(A1, A2))]
      = 1 - alpha sum_{i<n1, j<n2} n1^i n2^j (i+j)! / (i! j! (alpha+n1+n2)^(i+j+1))``,
      from ``P(A_i > t) = exp(-n_i t) sum_{m<n_i} (n_i t)^m / m!``.
    * ``"go"``: ``alpha * max(...)``, and ``P_go = P_ca(n1) + P_ca(n2) - P_so``
      because ``exp(-a min) + exp(-a max) = exp(-a A1) + exp(-a A2)``.
      A half with no cells on the map makes both reduce to cell averaging
      over the other half.
    * ``"os"``: ``alpha * X_(k)``, the ``k``-th smallest of the ``N`` training
      cells with ``k - 1 =`` :func:`os_rank_index`; Rohling's
      ``P = prod_{i=0}^{k-1} (N - i) / (N - i + alpha)``.
    """

    a = alpha.to(torch.float64)
    n1 = n_lag.to(torch.float64)
    n2 = n_lead.to(torch.float64)
    total = n1 + n2
    if kind == "ca":
        return torch.exp(-total * torch.log1p(a / total))
    if kind == "os":
        count = (n_lag + n_lead).to(torch.int64)
        rank = os_rank_index(count, rank_fraction) + 1
        index = torch.arange(int(rank.max()), dtype=torch.float64)
        live = index.reshape(1, -1) < rank.reshape(-1, 1)
        remaining = total.reshape(-1, 1) - index.reshape(1, -1)
        terms = torch.log(remaining.clamp(min=1.0)) - torch.log(remaining.clamp(min=1.0) + a.reshape(-1, 1))
        return torch.exp(torch.where(live, terms, torch.zeros_like(terms)).sum(dim=1))
    if kind not in ("go", "so"):
        raise ValueError(f"kind must be one of {CFAR_KINDS}, got {kind!r}")
    return _two_half_law(kind, n_lag, n_lead)(a)


def _two_half_law(kind: str, n_lag: torch.Tensor, n_lead: torch.Tensor):
    """The GO/SO false-alarm law for fixed half counts, as a function of ``alpha``.

    Everything that does not depend on ``alpha`` - the ``lgamma`` coefficients
    of the double series - is formed once here, so a bisection over ``alpha``
    evaluates one ``exp`` per series term per step.
    """

    if kind not in ("go", "so"):
        raise ValueError(f"kind must be one of {CFAR_KINDS}, got {kind!r}")
    n1 = n_lag.to(torch.float64)
    n2 = n_lead.to(torch.float64)
    total = n1 + n2
    one_sided = (n1 == 0) | (n2 == 0)
    safe1 = torch.where(n1 == 0, total, n1)
    safe2 = torch.where(n2 == 0, total, n2)
    largest = int(torch.maximum(safe1, safe2).max())
    i = torch.arange(largest, dtype=torch.float64).reshape(1, -1, 1)
    j = torch.arange(largest, dtype=torch.float64).reshape(1, 1, -1)
    live = (i < safe1.reshape(-1, 1, 1)) & (j < safe2.reshape(-1, 1, 1))
    coefficient = (
        i * torch.log(safe1).reshape(-1, 1, 1)
        + j * torch.log(safe2).reshape(-1, 1, 1)
        + torch.lgamma(i + j + 1)
        - torch.lgamma(i + 1)
        - torch.lgamma(j + 1)
    )
    coefficient = torch.where(live, coefficient, torch.full_like(coefficient, -torch.inf))
    order = (i + j + 1).expand_as(coefficient)

    def law(alpha: torch.Tensor) -> torch.Tensor:
        a = alpha.to(torch.float64)
        ca1 = torch.exp(-safe1 * torch.log1p(a / safe1))
        ca2 = torch.exp(-safe2 * torch.log1p(a / safe2))
        series = torch.exp(coefficient - order * torch.log(a + safe1 + safe2).reshape(-1, 1, 1)).sum(dim=(1, 2))
        smaller = 1.0 - a * series
        result = smaller if kind == "so" else ca1 + ca2 - smaller
        return torch.where(one_sided, torch.where(n1 == 0, ca2, ca1), result)

    return law


@functools.lru_cache(maxsize=128)
def _threshold_constants(
    kind: str, ranges: int, ring: _Ring, pfa: float, rank_fraction: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per range column: the scale ``alpha`` and the two half-mean weights, on the host.

    Returns ``(alpha, lag_weight, lead_weight)`` as ``[R]`` float64 CPU tensors
    (for ``"os"`` the two weights are the ordered-statistic index and unused).
    The half weights are ``1 / n`` of the half, or - for a half truncated to
    no cells - zero on that half and ``1 / n`` of the other, so the mean
    statistic of an empty half is the other half's mean without a branch.
    """

    n_lag, n_lead = _half_counts(ring, ranges)
    count = n_lag + n_lead
    if bool((count == 0).any()):
        raise ValueError(f"a {ranges}-bin range axis leaves a cell with no training cells inside the map")
    pairs = torch.stack((n_lag, n_lead), dim=1)
    unique, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    if kind == "ca":
        alpha_unique = unique.sum(dim=1).to(torch.float64) * (pfa ** (-1.0 / unique.sum(dim=1).to(torch.float64)) - 1.0)
    else:
        if kind == "os":
            law = lambda value: cfar_false_alarm_probability(  # noqa: E731
                kind, value, unique[:, 0], unique[:, 1], rank_fraction=rank_fraction
            )
        else:
            law = _two_half_law(kind, unique[:, 0], unique[:, 1])
        alpha_unique = _bisect(law, pfa, int(unique.shape[0]))
    alpha = alpha_unique[inverse]
    lag = n_lag.to(torch.float64)
    lead = n_lead.to(torch.float64)
    if kind == "os":
        return alpha, os_rank_index(count, rank_fraction).to(torch.float64), torch.zeros_like(alpha)
    if kind == "ca":
        weight = 1.0 / (lag + lead)
        return alpha, weight, weight
    lag_weight = torch.where(lag > 0, 1.0 / lag.clamp(min=1.0), torch.zeros_like(lag))
    lead_weight = torch.where(lead > 0, 1.0 / lead.clamp(min=1.0), torch.zeros_like(lead))
    # An empty half takes the other half's mean.
    lag_from_lead = torch.where(lag > 0, torch.zeros_like(lead), lead_weight)
    lead_from_lag = torch.where(lead > 0, torch.zeros_like(lag), lag_weight)
    return alpha, torch.stack((lag_weight, lag_from_lead)), torch.stack((lead_weight, lead_from_lag))


@functools.lru_cache(maxsize=128)
def _constants_on(kind, ranges, ring, pfa, rank_fraction, device, dtype):
    """:func:`_threshold_constants` placed on ``device`` once per configuration.

    Cached so a detector called every frame uploads its ``[R]`` constants once
    rather than once per call. The ordered-statistic index stays int64.
    """

    alpha, lag, lead = _threshold_constants(kind, ranges, ring, pfa, rank_fraction)
    if kind == "os":
        return alpha.to(device=device, dtype=dtype), lag.to(device=device, dtype=torch.int64), None
    return tuple(value.to(device=device, dtype=dtype) for value in (alpha, lag, lead))


def _pad_window(flat: torch.Tensor, ring: _Ring, *, range_fill: float) -> torch.Tensor:
    """``[B, D, R]`` -> ``[B, 1, D + 2 od, R + 2 or]``: Doppler wrapped, range filled.

    The Doppler axis of a slow-time DFT is circular - bin ``D - 1`` is the
    neighbour of bin ``0`` - so the ring wraps there. The range axis is not,
    and its padding carries ``range_fill``: zero for a block sum (the count
    tables already exclude those cells) and ``+inf`` for an ordered statistic
    (sorted past every real training cell).
    """

    doppler = int(flat.shape[-2])
    if 2 * ring.outer_d + 1 > doppler:
        raise ValueError(
            f"the Doppler window spans {2 * ring.outer_d + 1} bins (guard {ring.gd} + training {ring.td} each side "
            f"of the cell) but the map has {doppler}; the Doppler axis wraps, so a window that long would count a "
            f"cell twice. Use guard_cells[0] + training_cells[0] <= {(doppler - 1) // 2} for this map, or more "
            "slow-time samples"
        )
    padded = flat.unsqueeze(1)
    if ring.outer_d > 0:
        padded = F.pad(padded, (0, 0, ring.outer_d, ring.outer_d), mode="circular")
    return F.pad(padded, (ring.outer_r, ring.outer_r, 0, 0), mode="constant", value=range_fill)


def _half_sums(flat: torch.Tensor, ring: _Ring) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B, D, R]`` -> the lagging and leading training sums, ``[B, D, R]`` each.

    One ``avg_pool2d`` per distinct block shape over the padded map; each block
    sum is a shifted slice of its pool. Batching is along the outer axis, so a
    slice of a batched call is the same reduction as the standalone call.
    """

    doppler, ranges = int(flat.shape[-2]), int(flat.shape[-1])
    padded = _pad_window(flat, ring, range_fill=0.0)
    pooled: dict[tuple[int, int], torch.Tensor] = {}
    sums = []
    for side in (-1, 1):
        total = torch.zeros_like(flat)
        for d0, d1, r0, r1 in ring.blocks(side):
            shape = (d1 - d0 + 1, r1 - r0 + 1)
            if shape not in pooled:
                pooled[shape] = F.avg_pool2d(padded, kernel_size=shape, stride=1).squeeze(1) * (shape[0] * shape[1])
            row = ring.outer_d + d0
            column = ring.outer_r + r0
            total = total + pooled[shape][:, row : row + doppler, column : column + ranges]
        sums.append(total)
    return sums[0], sums[1]


def _mean_level_cfar(name: str, kind: str, rd_map, guard_cells, training_cells, pfa: float, rank: int) -> Detections:
    refuse_derivative(f"witwin.radar.processing.detection.{name}", _CFAR_REASON, rd_map=rd_map)
    pfa = _require_pfa(pfa)
    ring = _ring(guard_cells, training_cells, rank=rank)
    power = _power(rd_map)
    flat, leading = _as_batch(power if rank == 2 else power.unsqueeze(-2), 2)
    ranges = int(flat.shape[-1])
    alpha, lag_weight, lead_weight = _constants_on(kind, ranges, ring, pfa, 0.0, flat.device, flat.dtype)
    lag, lead = _half_sums(flat, ring)
    if kind == "ca":
        level = (lag + lead) * lag_weight
    else:
        lag_mean = lag * lag_weight[0] + lead * lag_weight[1]
        lead_mean = lead * lead_weight[0] + lag * lead_weight[1]
        level = torch.maximum(lag_mean, lead_mean) if kind == "go" else torch.minimum(lag_mean, lead_mean)
    threshold = alpha * level
    threshold = threshold.reshape(*leading, *power.shape[-rank:]) if rank == 2 else threshold.reshape(power.shape)
    return Detections(mask=power > threshold, threshold=threshold)


def _require_pfa(pfa) -> float:
    value = float(pfa)
    if not 0.0 < value < 1.0:
        raise ValueError(f"pfa must be a probability in (0, 1), got {pfa!r}")
    return value


_SQUARE_LAW = """
    The detector is SQUARE LAW: a complex map is detected on ``|x|^2``, and a
    real map is read as a power map already (pass ``|x|^2``, never ``|x|`` or
    decibels). On that input the threshold constant is solved so the
    false-alarm probability on exponential (complex Gaussian) noise is exactly
    ``pfa`` at every cell; :func:`cfar_false_alarm_probability` is the law.

    Edges: the Doppler axis WRAPS, because a slow-time DFT is circular, and the
    window must fit inside it. The range axis is truncated: a cell near either
    end keeps the training cells that lie on the map and its constant is solved
    for that count, so the edge cells hold the same ``pfa`` as the interior.

    ``threshold`` is in the input's power units and ``mask = power > threshold``.
"""


def ca_cfar(
    rd_map: torch.Tensor,
    *,
    guard_cells: tuple[int, int] = (2, 3),
    training_cells: tuple[int, int] = (4, 6),
    pfa: float = 1e-3,
) -> Detections:
    """Cell-averaging CFAR over ``[..., D, R]``: the ring mean, scaled.

    ``alpha = N (pfa ** (-1 / N) - 1)`` on the mean of the ``N`` ring cells.
    """

    return _mean_level_cfar("ca_cfar", "ca", rd_map, guard_cells, training_cells, pfa, 2)


def go_cfar(
    rd_map: torch.Tensor,
    *,
    guard_cells: tuple[int, int] = (2, 3),
    training_cells: tuple[int, int] = (4, 6),
    pfa: float = 1e-3,
) -> Detections:
    """Greatest-of CFAR over ``[..., D, R]``: the larger of the two half-ring means.

    Holds its false-alarm rate at a clutter edge along range, at a small loss
    in a homogeneous background.
    """

    return _mean_level_cfar("go_cfar", "go", rd_map, guard_cells, training_cells, pfa, 2)


def so_cfar(
    rd_map: torch.Tensor,
    *,
    guard_cells: tuple[int, int] = (2, 3),
    training_cells: tuple[int, int] = (4, 6),
    pfa: float = 1e-3,
) -> Detections:
    """Smallest-of CFAR over ``[..., D, R]``: the smaller of the two half-ring means.

    Resolves a second target in one half of the ring, at the cost of false
    alarms at a clutter edge.
    """

    return _mean_level_cfar("so_cfar", "so", rd_map, guard_cells, training_cells, pfa, 2)


def ca_cfar_1d(
    profile: torch.Tensor, *, guard_cells: int = 2, training_cells: int = 8, pfa: float = 1e-3
) -> Detections:
    """Range-only cell-averaging CFAR over ``[..., R]``: ``2 * training_cells`` ring cells."""

    return _mean_level_cfar("ca_cfar_1d", "ca", profile, guard_cells, training_cells, pfa, 1)


def os_cfar(
    rd_map: torch.Tensor,
    *,
    guard_cells: tuple[int, int] = (2, 3),
    training_cells: tuple[int, int] = (4, 6),
    rank_fraction: float = 0.75,
    pfa: float = 1e-3,
) -> Detections:
    """Ordered-statistic CFAR over ``[..., D, R]``.

    The noise statistic is the ``k``-th smallest training cell with
    ``k - 1 = min(int(rank_fraction * N), N - 1)``, which is what keeps the
    threshold down when a second target sits inside the ring. The scale is
    solved from Rohling's law for that ``k`` and ``N`` (see
    :func:`cfar_false_alarm_probability`), so ``pfa`` is the achieved rate.

    A CUDA map is ranked by the native ``cfar_order_statistic`` kernel, one
    thread per cell selecting in place, so no ``[cells, N]`` patch tensor is
    formed. A CPU map materialises the training patches with ``unfold`` and
    sorts them; both evaluate the same statistic of the same cells.
    """

    refuse_derivative(
        "witwin.radar.processing.detection.os_cfar",
        _CFAR_REASON + " The ordered statistic adds a second discrete decision on top of it: "
        "which training sample the threshold is read from is chosen by a sort.",
        rd_map=rd_map,
    )
    pfa = _require_pfa(pfa)
    if not 0.0 <= float(rank_fraction) <= 1.0:
        raise ValueError(f"rank_fraction must be in [0, 1], got {rank_fraction!r}")
    ring = _ring(guard_cells, training_cells, rank=2)
    power = _power(rd_map)
    flat, leading = _as_batch(power, 2)
    doppler, ranges = int(flat.shape[-2]), int(flat.shape[-1])
    alpha, rank_index, _ = _constants_on("os", ranges, ring, pfa, float(rank_fraction), flat.device, flat.dtype)
    level = _order_statistic(flat, ring, rank_index)
    threshold = (alpha * level).reshape(*leading, doppler, ranges)
    return Detections(mask=power > threshold, threshold=threshold)


#: The ring size the native ordered-statistic kernel holds per thread.
_NATIVE_MAX_TRAINING = 1024


def _native_ring_tables(ring: _Ring, rank_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The training offsets and per-column ranks as int32 on the map's device, cached."""

    return _native_ring_tables_cached(ring, rank_index)


@functools.lru_cache(maxsize=128)
def _native_ring_tables_cached(ring: _Ring, rank_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = rank_index.device
    return ring.training_offsets().to(device=device, dtype=torch.int32), rank_index.to(torch.int32)


def _order_statistic(flat: torch.Tensor, ring: _Ring, rank_index: torch.Tensor) -> torch.Tensor:
    """``[B, D, R]``: the ``rank_index[r]``-th smallest training cell of every cell."""

    doppler, ranges = int(flat.shape[-2]), int(flat.shape[-1])
    padded = _pad_window(flat, ring, range_fill=float("inf"))
    if flat.is_cuda:
        if flat.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"the CUDA ordered-statistic route takes a float32 or float64 map, got {flat.dtype}")
        offsets, rank = _native_ring_tables(ring, rank_index)
        if int(offsets.shape[0]) > _NATIVE_MAX_TRAINING:
            raise ValueError(
                f"the training ring holds {int(offsets.shape[0])} cells; the CUDA ordered-statistic route selects "
                f"in a {_NATIVE_MAX_TRAINING}-cell thread-local buffer"
            )
        level = torch.empty_like(flat)
        _ops().cfar_order_statistic_forward(
            padded.squeeze(1).contiguous(), offsets, rank, level, 2 * ring.outer_d + 1, 2 * ring.outer_r + 1
        )
        return level
    offsets = ring.training_offsets()
    patches = F.unfold(padded, kernel_size=(2 * ring.outer_d + 1, 2 * ring.outer_r + 1), stride=1)
    training = patches.index_select(1, offsets.to(flat.device))
    ordered, _ = torch.sort(training, dim=1)
    pick = rank_index.to(flat.device).repeat(doppler).reshape(1, 1, -1).expand(int(flat.shape[0]), 1, -1)
    return ordered.gather(1, pick).reshape(-1, doppler, ranges)


for _detector in (ca_cfar, go_cfar, so_cfar, ca_cfar_1d, os_cfar):
    _detector.__doc__ += _SQUARE_LAW


def group_peaks(detections: Detections, rd_map: torch.Tensor, *, neighbourhood: tuple[int, int] = (3, 3)) -> Detections:
    """Keep a detection only where it is the strongest cell of its neighbourhood.

    ``rd_map`` is the map the detector ran on (complex, or real power), and
    ``neighbourhood`` the odd ``(doppler, range)`` window around each cell. A
    detection survives when its power is the maximum of that window, so the
    cluster of cells one return lights up - its main lobe, its window
    leakage - collapses to the one cell at its peak. The Doppler axis wraps
    and the range axis does not, as in the detectors. A tie inside a window is
    broken by position (the first cell in row-major order wins), so two equal
    cells never both survive or both vanish.

    One ``max_pool2d`` with its argmax indices: a cell is a peak when the pool
    centred on it points back at it. The threshold is passed through.
    """

    if not isinstance(detections, Detections):
        raise TypeError(f"detections must be a Detections record, got {type(detections).__name__}")
    refuse_derivative(
        "witwin.radar.processing.detection.group_peaks",
        _SELECTION_REASON,
        rd_map=rd_map,
        detection_threshold=detections.threshold,
    )
    window_d, window_r = (int(value) for value in neighbourhood)
    if window_d < 1 or window_r < 1 or window_d % 2 == 0 or window_r % 2 == 0:
        raise ValueError(f"neighbourhood must be two odd positive sizes, got {neighbourhood!r}")
    power = _power(rd_map)
    if tuple(power.shape) != tuple(detections.mask.shape):
        raise ValueError(
            f"the map is {tuple(power.shape)} but the detections are {tuple(detections.mask.shape)}; "
            "they describe different grids"
        )
    flat, leading = _as_batch(power, 2)
    doppler, ranges = int(flat.shape[-2]), int(flat.shape[-1])
    if window_d > doppler:
        raise ValueError(f"a {window_d}-bin Doppler neighbourhood does not fit in {doppler} Doppler bins")
    half_d, half_r = window_d // 2, window_r // 2
    padded = flat.unsqueeze(1)
    if half_d > 0:
        padded = F.pad(padded, (0, 0, half_d, half_d), mode="circular")
    padded = F.pad(padded, (half_r, half_r, 0, 0), mode="constant", value=-torch.inf)
    _, argmax = F.max_pool2d(padded, kernel_size=(window_d, window_r), stride=1, return_indices=True)
    width = ranges + 2 * half_r
    own = (torch.arange(doppler, device=flat.device).reshape(-1, 1) + half_d) * width + (
        torch.arange(ranges, device=flat.device).reshape(1, -1) + half_r
    )
    peak = (argmax.squeeze(1) == own).reshape(*leading, doppler, ranges)
    return Detections(mask=detections.mask & peak, threshold=detections.threshold)


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


#: The frames a :class:`PointCloud` can be in.
POINT_CLOUD_FRAMES = ("radar", "world")


@dataclass(frozen=True, slots=True, eq=False)
class PointCloud:
    """``N`` detections, in metres, in the frame ``frame`` names.

    ``frame="radar"`` - what :func:`point_cloud` publishes - is the PROCESSING
    frame: origin at the radar's position, ``x`` to the radar's right, ``y``
    along its boresight, ``z`` up. It is NOT the radar's pose-local frame,
    which is ``(right, up, -boresight)``. :meth:`to_world` maps a radar-frame
    cloud into the world through the radar's own pose. ``frame="world"`` is
    the world frame the scene and ``Radar.position`` are declared in.

    ``velocity_mps`` is the RADIAL velocity in the canonical closing-positive
    convention, the same one :class:`RangeDopplerMap`'s Doppler axis
    publishes; it is a scalar and does not change with the frame. ``energy`` is
    in decibels relative to one unit of the map's own amplitude, which is the
    amplitude convention the range and Doppler stages publish, so a level here
    is comparable across waveforms. ``range_m`` is the measured range, the
    distance from the radar.
    """

    __repr__ = compact_repr

    xyz: torch.Tensor
    velocity_mps: torch.Tensor
    energy: torch.Tensor
    range_m: torch.Tensor
    frame: str = "radar"

    def __post_init__(self) -> None:
        if self.frame not in POINT_CLOUD_FRAMES:
            raise ValueError(f"frame must be one of {POINT_CLOUD_FRAMES}, got {self.frame!r}")
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
    def empty(cls, *, device: torch.device, frame: str = "radar") -> "PointCloud":
        zero = torch.zeros((0,), dtype=torch.float64, device=device)
        return cls(
            xyz=torch.zeros((0, 3), dtype=torch.float64, device=device),
            velocity_mps=zero,
            energy=zero.clone(),
            range_m=zero.clone(),
            frame=frame,
        )

    def select(self, keep: torch.Tensor) -> "PointCloud":
        """The subset the boolean mask ``keep`` selects, as a new record."""

        return PointCloud(
            xyz=self.xyz[keep],
            velocity_mps=self.velocity_mps[keep],
            energy=self.energy[keep],
            range_m=self.range_m[keep],
            frame=self.frame,
        )

    def to_world(self, radar) -> "PointCloud":
        """This radar-frame cloud in the world, through ``radar``'s pose.

        ``radar`` is the :class:`witwin.radar.Radar` whose array produced the
        cloud. The processing frame is mapped to the radar's pose-local frame,
        ``(x, y, z) -> (x, z, -y)`` (right, up, minus boresight), and then
        through the same world-from-local rotation and origin the radar uses to
        place its own elements, so a detection lands where the scene put the
        reflector. Velocity, energy and range are scalars and are copied.
        """

        # The pose owner is read by its interface rather than imported: the radar
        # record imports the session, which imports this package.
        if not hasattr(radar, "_world_from_local_matrix"):
            raise TypeError(f"to_world takes the witwin.radar.Radar the cloud came from, got {type(radar).__name__}")
        if self.frame != "radar":
            raise ValueError(f"this cloud is already in the {self.frame!r} frame")
        position, world_from_local = radar._world_from_local_matrix(device=self.device, dtype=torch.float64)
        xyz = self.xyz.to(torch.float64)
        local = torch.stack((xyz[:, 0], xyz[:, 2], -xyz[:, 1]), dim=1)
        world = local @ world_from_local.transpose(0, 1) + position.reshape(1, 3)
        return PointCloud(
            xyz=world.contiguous(),
            velocity_mps=self.velocity_mps,
            energy=self.energy,
            range_m=self.range_m,
            frame="world",
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
    route: str = "scan",
    fft_size: int = 64,
    range_gate_m: tuple[float, float] | None = None,
    max_points: int | None = None,
    positive_velocity_only: bool = False,
    energy_floor: float = 1e-6,
    peak_neighbourhood: tuple[int, int] | None = (3, 3),
) -> PointCloud:
    """``Detections[D, R]`` plus ``RangeDopplerMap[TX, RX, D, R]`` -> ``PointCloud``.

    The detection mask is rank 2 because a detection is a cell of the coherently
    combined map: the AoA estimate for that cell is what needs every element,
    and estimating an angle per element per cell first would be estimating the
    angle of the noise.

    ``route`` names the angle estimator explicitly, from
    :data:`~witwin.radar.processing.angle.AOA_ROUTES`, because a front-end
    change that silently swaps the estimator is a change of answer with no
    change of call. The default is ``"scan"``, the steering-vector scan that
    reads any declared layout; the other routes refuse a layout they cannot
    read by name. ``range_gate_m`` is a pair of distances in METRES read
    against :attr:`ProcessingAxes.range_m`; a bin index never appears in the
    signature. Everything published is float64 and on the input device.

    ``peak_neighbourhood`` groups the detections before any angle is read:
    :func:`group_peaks` on the coherently combined map, so the cells one return
    lights up become one point at its peak. ``None`` keeps every detected cell.

    The cloud is in the radar PROCESSING frame (``x`` right, ``y`` boresight,
    ``z`` up, origin at the radar); :meth:`PointCloud.to_world` places it in
    the world.
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
    if peak_neighbourhood is not None:
        mask = group_peaks(
            Detections(mask=mask, threshold=detections.threshold), combined, neighbourhood=peak_neighbourhood
        ).mask

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


#: Why clustering has no derivative.
_CLUSTER_REASON = (
    "a cluster label is a connected component of a thresholded distance graph, a discrete function of the "
    "point positions with no derivative."
)


@dataclass(frozen=True, slots=True, eq=False)
class PointClusters:
    """A density clustering of one :class:`PointCloud`.

    ``labels`` is ``[N]`` int64, the cluster of each point, ``-1`` for noise.
    Clusters are numbered ``0 .. count - 1`` in order of their lowest point
    index, so a clustering is reproducible from the cloud alone. ``centroid``
    is ``[count, 3]`` (the mean member position, in the cloud's frame),
    ``velocity_mps`` the mean member radial velocity and ``size`` the member
    count, all indexed by label.
    """

    __repr__ = compact_repr

    labels: torch.Tensor
    centroid: torch.Tensor
    velocity_mps: torch.Tensor
    size: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.size.shape[0])


def cluster_points(cloud: PointCloud, *, eps_m: float, min_points: int, velocity_scale_s: float = 0.0) -> PointClusters:
    """DBSCAN over the points of ``cloud``.

    Two points are neighbours when their distance is at most ``eps_m``. The
    distance is Euclidean over the position and, when ``velocity_scale_s`` is
    positive, the radial velocity times that scale - seconds, so a velocity
    difference counts as the distance it covers in that time and two targets
    passing each other at different speeds stay apart. A CORE point has at
    least ``min_points`` neighbours, itself included. Clusters are the
    connected components of the core points under the neighbour relation; a
    non-core point with a core neighbour joins the lowest-labelled cluster
    among them (DBSCAN's border points, made deterministic); everything else
    is noise.

    The components are found by min-label propagation with pointer jumping
    over the ``[N, N]`` neighbour matrix: each round is two library reductions
    and the loop runs until no label moves, which takes a number of rounds
    logarithmic-to-linear in the component diameter, never one per point.
    """

    if not isinstance(cloud, PointCloud):
        raise TypeError(f"cluster_points takes a PointCloud, got {type(cloud).__name__}")
    refuse_derivative(
        "witwin.radar.processing.detection.cluster_points",
        _CLUSTER_REASON,
        xyz=cloud.xyz,
        velocity_mps=cloud.velocity_mps,
    )
    eps = float(eps_m)
    if not eps > 0.0:
        raise ValueError(f"eps_m must be positive, got {eps_m!r}")
    if type(min_points) is not int or min_points < 1:
        raise ValueError(f"min_points must be a positive int, got {min_points!r}")
    scale = float(velocity_scale_s)
    if not scale >= 0.0:
        raise ValueError(f"velocity_scale_s must be non-negative, got {velocity_scale_s!r}")
    count = len(cloud)
    device = cloud.device
    if count == 0:
        empty = torch.zeros((0,), dtype=torch.float64, device=device)
        return PointClusters(
            labels=torch.zeros((0,), dtype=torch.int64, device=device),
            centroid=torch.zeros((0, 3), dtype=torch.float64, device=device),
            velocity_mps=empty,
            size=torch.zeros((0,), dtype=torch.int64, device=device),
        )
    features = cloud.xyz.to(torch.float64)
    if scale > 0.0:
        features = torch.cat((features, cloud.velocity_mps.to(torch.float64).reshape(-1, 1) * scale), dim=1)
    assigned = _dbscan_roots(features.contiguous(), eps, min_points)
    clustered = assigned >= 0
    roots, compact = torch.unique(assigned[clustered], sorted=True, return_inverse=True)
    labels_out = torch.full_like(assigned, -1)
    labels_out[clustered] = compact
    groups = int(roots.shape[0])
    size = torch.zeros(groups, dtype=torch.int64, device=device).index_add_(
        0, compact, torch.ones_like(compact, dtype=torch.int64)
    )
    centroid = torch.zeros((groups, 3), dtype=torch.float64, device=device).index_add_(
        0, compact, cloud.xyz[clustered].to(torch.float64)
    ) / size.reshape(-1, 1)
    velocity = (
        torch.zeros(groups, dtype=torch.float64, device=device).index_add_(
            0, compact, cloud.velocity_mps[clustered].to(torch.float64)
        )
        / size
    )
    return PointClusters(labels=labels_out, centroid=centroid, velocity_mps=velocity, size=size)


def _dbscan_roots(features: torch.Tensor, eps: float, min_points: int) -> torch.Tensor:
    """``[N]`` int64: each point's component root (its smallest core index), ``-1`` for noise.

    A CUDA cloud runs the native ``dbscan_labels`` union-find; a CPU cloud runs
    min-label propagation over the dense neighbour matrix. Both publish the
    smallest core index of the component, and a border point the smallest root
    among its core neighbours, so the two routes return the same labels.
    """

    count = int(features.shape[0])
    if features.is_cuda:
        parent = torch.empty(count, dtype=torch.int32, device=features.device)
        core = torch.empty(count, dtype=torch.int32, device=features.device)
        _ops().dbscan_labels_forward(features, parent, core, eps, min_points)
        return parent.to(torch.int64)
    difference = features.unsqueeze(1) - features.unsqueeze(0)
    neighbour = difference.square().sum(dim=-1) <= eps * eps
    core = neighbour.sum(dim=1) >= min_points
    labels = _connected_core_labels(neighbour & core.reshape(1, -1) & core.reshape(-1, 1), core)
    # A border point takes the smallest label among its core neighbours.
    sentinel = torch.full_like(labels, count)
    candidate = torch.where(neighbour & core.reshape(1, -1), labels.reshape(1, -1), sentinel.reshape(1, -1))
    border = candidate.min(dim=1).values
    return torch.where(core, labels, torch.where(border < count, border, torch.full_like(border, -1)))


def _connected_core_labels(adjacency: torch.Tensor, core: torch.Tensor) -> torch.Tensor:
    """``[N]`` int64: the smallest point index of each core point's component, ``N`` off the core.

    Min-label propagation (every core point takes the smallest label among
    itself and its core neighbours) alternated with pointer jumping
    (``label = label[label]``), to a fixed point. One host read per round
    decides termination.
    """

    count = int(adjacency.shape[0])
    index = torch.arange(count, device=adjacency.device)
    labels = torch.where(core, index, torch.full_like(index, count))
    sentinel = torch.full((1, count), count, dtype=labels.dtype, device=labels.device)
    for _ in range(count + 1):
        spread = torch.where(adjacency, labels.reshape(1, -1), sentinel).min(dim=1).values
        updated = torch.minimum(labels, spread)
        jumped = torch.where(updated < count, updated[updated.clamp(max=count - 1)], updated)
        updated = torch.minimum(updated, jumped)
        if bool(torch.equal(updated, labels)):
            return labels
        labels = updated
    raise RuntimeError("label propagation did not converge; this is a defect, not an input error")


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
    """``|sum_p cube[p]|^2``: the coherent (boresight) sum over sensor pairs, as power.

    ``cube`` is ``[..., D, R]`` complex with the sensor pairs in the leading
    axes; every leading axis is folded into one pair axis and summed with the
    phases intact, which is the boresight beam. Returns a REAL ``[D, R]``
    POWER map - the square-law input every detector here thresholds, and the
    same unit :func:`combine_incoherent` returns. Formed as ``re^2 + im^2`` so
    it stays differentiable at an exact zero.
    """

    total = cube.reshape(-1, *cube.shape[-2:]).sum(dim=0)
    return total.real.square() + total.imag.square()
