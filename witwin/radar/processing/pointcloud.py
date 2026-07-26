"""Detections plus a Range-Doppler map become points in metres.

``sigproc`` had two near-identical pipelines here - ``frame2pointcloud`` and
``_process_pc_cfar_tensor`` - that differed only in which detector produced the
mask, and both hard coded a range gate as the bin indices ``[:, :25]`` and
``[:, 125:]``. Those numbers are a 128 by 256 configuration written into the
source: change the range-bin count and the gate silently moves to a different
part of the scene.

There is one pipeline here, the detector is an ARGUMENT, and the gate is a pair
of distances in METRES read against :attr:`ProcessingAxes.range_m`. A bin index
never appears in the signature.

Everything published is float64 and on the input device. The stage performs
exactly one host observation - the ``torch.argwhere`` that turns a mask into a
row list - and that observation is unavoidable: a point cloud has a data
dependent length. It is named here so the frozen pipeline budget can attribute
it to processing rather than to the simulation half.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .aoa import AOA_ROUTES, tdm_compensate
from .beamforming import ArrayGeometry
from .cfar import Detections
from .contracts import RangeDopplerMap


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
                raise ValueError(
                    f"{name} must be [{count}] to match xyz; got "
                    f"{tuple(value.shape)}"
                )

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def device(self) -> torch.device:
        return self.xyz.device

    def as_columns(self) -> torch.Tensor:
        """``[N, 6]`` in :data:`POINT_CLOUD_COLUMNS` order.

        The flat form the legacy pipeline published and the form a fixed-size
        detection batch is assembled from. It is a VIEW-building stack, not the
        record's storage: the named fields are the contract.
        """

        return torch.stack(
            (
                self.xyz[:, 0],
                self.xyz[:, 1],
                self.xyz[:, 2],
                self.velocity_mps,
                self.energy,
                self.range_m,
            ),
            dim=1,
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
        raise ValueError(
            f"the range gate must be (low_m, high_m) with high > low, got {gate_m!r}"
        )
    return (axes.range_m >= low) & (axes.range_m < high)


def point_cloud(
    detections: Detections,
    rd: RangeDopplerMap,
    axes,
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
    :data:`~witwin.radar.processing.aoa.AOA_ROUTES`. The legacy dispatch on
    ``num_tx`` is preserved only inside the ``naive_xyz`` adapter, because a
    front-end change that silently swaps the estimator is a change of answer
    with no change of call.
    """

    if not isinstance(detections, Detections):
        raise TypeError(
            f"detections must be a Detections record, got {type(detections).__name__}"
        )
    if not isinstance(rd, RangeDopplerMap):
        raise TypeError(
            f"rd must be a RangeDopplerMap, got {type(rd).__name__}"
        )
    if not isinstance(array, ArrayGeometry):
        raise TypeError(
            f"array must be an ArrayGeometry, got {type(array).__name__}"
        )
    if route not in AOA_ROUTES:
        raise ValueError(
            f"route must be one of {tuple(sorted(AOA_ROUTES))}, got {route!r}"
        )
    mask = detections.mask
    if mask.dim() != 2:
        raise ValueError(
            "the detection mask is [doppler, range]: one cell of the combined "
            f"map per detection; got shape {tuple(mask.shape)}"
        )
    data = rd.data
    if data.dim() < 3:
        raise ValueError(
            "the map must be [*pair, doppler, range]; got shape "
            f"{tuple(data.shape)}"
        )
    if tuple(data.shape[-2:]) != tuple(mask.shape):
        raise ValueError(
            f"the map is {tuple(data.shape[-2:])} but the mask is "
            f"{tuple(mask.shape)}; they describe different grids"
        )
    pairs = array.sensor_pair_count
    flat = data.reshape(pairs, int(mask.shape[0]), int(mask.shape[1]))
    combined = flat.sum(dim=0)
    energy_db = 20 * torch.log10(combined.abs() + float(energy_floor))

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


def _keep_strongest(
    mask: torch.Tensor, energy: torch.Tensor, max_points: int
) -> torch.Tensor:
    """Thin a detection mask to its ``max_points`` strongest cells.

    Done on the DEVICE with ``topk`` over the flattened map rather than by
    reading the detection list to the host and sorting it there.
    """

    if max_points < 0:
        raise ValueError(f"max_points must be non-negative, got {max_points}")
    flat_mask = mask.reshape(-1)
    total = int(flat_mask.shape[0])
    if max_points >= total:
        return mask
    scored = torch.where(
        flat_mask, energy.reshape(-1), torch.full_like(energy.reshape(-1), -torch.inf)
    )
    keep = torch.zeros_like(flat_mask)
    if max_points > 0:
        keep[torch.topk(scored, max_points).indices] = True
    return (keep & flat_mask).reshape(mask.shape)


__all__ = [
    "POINT_CLOUD_COLUMNS",
    "PointCloud",
    "point_cloud",
    "range_gate_mask",
]
