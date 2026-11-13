"""Mesh facets as scatter sites: the facet model, its sites and one frame's geometry.

A mesh target scatters as physical-optics facets (R-ADR-042). This module owns
everything about the facets that is not the scattering coefficient itself:

* which faces of which Core structures scatter, read once per topology from
  the snapshot - the authored faces with zero-area and duplicate faces dropped,
  and the material each face carries, evaluated at the carrier;
* how faces are grouped into sites: one facet per site by default, or the
  facets of one cubic cell per site under a declared cluster size or site
  budget;
* one frame's facet corners and site positions, from the snapshot's world
  vertices, through the native ``facet_geometry`` operator and its exact
  forward and reverse companions.

The coefficient is :class:`witwin.radar.scattering.FacetScatterResponse`; the
target record is :class:`witwin.radar.targets.MeshTargets`; the binding reads
the sites through :class:`witwin.radar.binding.ScatterSitePolicy`.

World vertices follow Core's documented pose convention, the one Channel's
compiler applies to the same snapshot: a deformation replaces (or offsets) the
authored local vertices, which are recentred when the mesh says so, scaled,
rotated by the authored rotation and placed at the authored position; a rigid
motion then rotates about the world origin (Euler ``(yaw, pitch, roll)``
intrinsic Z-Y-X, or a scalar-first quaternion) and translates. That is a fixed
handful of tensor library calls per structure; every per-facet quantity is
native.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .cuda import native_ops as _ops
from .policy import first_order_only

#: ``material_params`` model column: a planar layer stack in vacuum.
MODEL_LAYERED = 0
#: ``material_params`` model column: a perfect conductor.
MODEL_PEC = 1
_VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12


@dataclass(frozen=True, slots=True, eq=False)
class FacetModel:
    """One topology's facet tables, built on the host once and uploaded once.

    Faces index the concatenation of the selected structures' vertices, in
    ascending structure-ID order. ``site_offsets``/``site_facets`` is the CSR of
    facets per site and ``facet_site`` its inverse; ``vertex_offsets``/
    ``vertex_entries`` lists each vertex's ``(facet * 3 + corner)`` incidences,
    the order the reverse pass sums in.
    """

    structure_ids: tuple[int, ...]
    vertex_counts: tuple[int, ...]
    faces: torch.Tensor
    site_offsets: torch.Tensor
    site_facets: torch.Tensor
    facet_site: torch.Tensor
    vertex_offsets: torch.Tensor
    vertex_entries: torch.Tensor
    facet_material: torch.Tensor
    material_params: torch.Tensor
    layer_offsets: torch.Tensor
    layers: torch.Tensor
    key: tuple

    @property
    def facet_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def site_count(self) -> int:
        return int(self.site_offsets.shape[0]) - 1

    @property
    def vertex_count(self) -> int:
        return int(self.vertex_offsets.shape[0]) - 1


@dataclass(frozen=True, slots=True, eq=False)
class FacetFrame:
    """One frame's facets: the static model plus this instant's geometry.

    ``corners`` is ``(F, 9)`` world metres, ``sites`` ``(S, 3)``; both carry the
    tape of the snapshot's vertices.
    """

    model: FacetModel
    corners: torch.Tensor
    sites: torch.Tensor


class _FacetGeometry(torch.autograd.Function):
    """Native facet corners and site positions, linear in the world vertices."""

    @staticmethod
    def forward(vertices, model):
        corners = torch.empty((model.facet_count, 9), dtype=torch.float32, device=vertices.device)
        sites = torch.empty((model.site_count, 3), dtype=torch.float32, device=vertices.device)
        _ops().facet_geometry_forward(vertices, model.faces, model.site_offsets, model.site_facets, corners, sites)
        return corners, sites

    @staticmethod
    def setup_context(ctx, inputs, output):
        model = inputs[1]
        ctx.model = model
        # The operator is linear, so its tape is the topology: the vertex
        # incidence CSR the reverse pass reduces over, the face table the
        # forward-mode pass gathers through.
        tables = (model.vertex_offsets, model.vertex_entries, model.facet_site, model.site_offsets)
        ctx.save_for_backward(*tables)
        ctx.save_for_forward(model.faces, model.site_offsets, model.site_facets)

    @staticmethod
    @first_order_only
    def backward(ctx, grad_corners, grad_sites):
        vertex_offsets, vertex_entries, facet_site, site_offsets = ctx.saved_tensors
        grad_vertices = torch.empty(
            (int(vertex_offsets.shape[0]) - 1, 3), dtype=torch.float32, device=grad_corners.device
        )
        _ops().facet_geometry_backward(
            vertex_offsets,
            vertex_entries,
            facet_site,
            site_offsets,
            grad_corners.contiguous(),
            grad_sites.contiguous(),
            grad_vertices,
        )
        return grad_vertices, None

    @staticmethod
    def jvp(ctx, tan_vertices, tan_model):
        faces, site_offsets, site_facets = ctx.saved_tensors
        corners = torch.empty((int(faces.shape[0]), 9), dtype=torch.float32, device=tan_vertices.device)
        sites = torch.empty((int(site_offsets.shape[0]) - 1, 3), dtype=torch.float32, device=tan_vertices.device)
        _ops().facet_geometry_jvp(tan_vertices.contiguous(), faces, site_offsets, site_facets, corners, sites)
        return corners, sites


class FacetSites:
    """The site source of a mesh target: resolves a snapshot into a :class:`FacetFrame`.

    The model is built on first use and rebuilt only when the selected
    structures' topology or materials change (a new face tensor, a different
    material object or version); a deformation or a rigid motion moves the
    vertices and keeps the model.
    """

    def __init__(
        self,
        *,
        structure_ids: tuple[int, ...] | None,
        carrier_hz: float,
        cluster_size_m: float | None,
        max_sites: int | None,
    ) -> None:
        self.structure_ids = structure_ids
        self.carrier_hz = float(carrier_hz)
        self.cluster_size_m = cluster_size_m
        self.max_sites = max_sites
        self._model: FacetModel | None = None
        self._frame: tuple[torch.device, tuple, FacetFrame] | None = None

    def resolve(self, snapshot: Any, *, device: torch.device) -> FacetFrame:
        states = _selected_states(snapshot, self.structure_ids)
        key = tuple(
            (
                int(state.structure_id),
                _faces_identity(state),
                id(state.structure.material),
                int(getattr(state.structure.material, "version", 0)),
            )
            for state in states
        )
        # The objects a frame's world vertices are a function of. When every one
        # is the same object as last time and none carries a derivative, the
        # previous frame is this frame: a still mesh costs no upload and no
        # launch per observation. The objects themselves are held, so an
        # identity comparison cannot be fooled by a recycled id.
        sources = tuple(
            (state.structure.geometry, state.rigid_motion, state.deformation, state.structure.material)
            for state in states
        )
        cached = self._frame
        if (
            cached is not None
            and cached[0] == device
            and len(cached[1]) == len(sources)
            and all(a is b for old, new in zip(cached[1], sources, strict=True) for a, b in zip(old, new, strict=True))
        ):
            return cached[2]
        vertices = [_world_vertices(state, device=device) for state in states]
        model = self._model
        if model is None or model.key != key or model.faces.device != device:
            model = self._build(states, vertices, key=key, device=device)
            self._model = model
        stacked = vertices[0] if len(vertices) == 1 else torch.cat(vertices, dim=0)
        corners, sites = _FacetGeometry.apply(stacked.contiguous(), model)
        frame = FacetFrame(model=model, corners=corners, sites=sites)
        live = stacked.requires_grad or _carries_tangent(stacked)
        self._frame = None if live else (device, sources, frame)
        return frame

    def _build(self, states, vertices, *, key: tuple, device: torch.device) -> FacetModel:
        """Host tables for one topology: one device-to-host copy of the vertices, at freeze time."""

        faces_list, materials, offsets = [], [], 0
        material_index: dict[int, int] = {}
        facet_material: list[np.ndarray] = []
        host_vertices = []
        for state, world in zip(states, vertices, strict=True):
            structure = state.structure
            faces = _authored_faces(structure)
            points = world.detach().to(device="cpu", dtype=torch.float64).numpy()
            kept = _kept_faces(points, faces)
            local = np.flatnonzero(kept)
            faces_list.append(faces[kept] + offsets)
            slot = material_index.setdefault(id(structure.material), len(materials))
            if slot == len(materials):
                materials.append(_material_rows(structure.material, self.carrier_hz))
            facet_material.append(np.full(local.shape[0], slot, dtype=np.int64))
            host_vertices.append(points)
            offsets += points.shape[0]
        faces = np.concatenate(faces_list, axis=0)
        if faces.shape[0] == 0:
            raise ValueError("the selected structures carry no face with a non-zero area; a mesh target needs one")
        points = np.concatenate(host_vertices, axis=0)
        centroids = points[faces].mean(axis=1)
        site_of = _cluster(centroids, cluster_size_m=self.cluster_size_m, max_sites=self.max_sites)
        site_facets = np.argsort(site_of, kind="stable")
        site_counts = np.bincount(site_of, minlength=int(site_of.max()) + 1)
        site_offsets = np.concatenate(([0], np.cumsum(site_counts)))
        incidence = faces.reshape(-1)
        vertex_entries = np.argsort(incidence, kind="stable")
        vertex_counts = np.bincount(incidence, minlength=points.shape[0])
        vertex_offsets = np.concatenate(([0], np.cumsum(vertex_counts)))
        params = np.asarray([[row[0], row[1]] for row in materials], dtype=np.float32)
        layer_rows = [layer for row in materials for layer in row[2]]
        layer_counts = [len(row[2]) for row in materials]
        layer_offsets = np.concatenate(([0], np.cumsum(layer_counts)))
        layers = np.asarray(layer_rows if layer_rows else [[0.0, 1.0, 0.0, 1.0]], dtype=np.float32).reshape(-1, 4)

        # Every table crosses to the device in two copies, one per dtype, and
        # is published as views of those buffers.
        integer_tables = (faces, site_offsets, site_facets, site_of, vertex_offsets, vertex_entries)
        integer_tables += (np.concatenate(facet_material), layer_offsets)
        integers = torch.from_numpy(
            np.concatenate([np.asarray(table, dtype=np.int64).reshape(-1) for table in integer_tables])
        ).to(device)
        views = torch.split(integers, [int(np.asarray(table).size) for table in integer_tables])
        floats = torch.from_numpy(np.concatenate((params.reshape(-1), layers.reshape(-1))).astype(np.float32)).to(
            device
        )
        material_params, layer_values = torch.split(floats, [params.size, layers.size])
        return FacetModel(
            structure_ids=tuple(int(state.structure_id) for state in states),
            vertex_counts=tuple(int(value.shape[0]) for value in host_vertices),
            faces=views[0].view(-1, 3),
            site_offsets=views[1],
            site_facets=views[2],
            facet_site=views[3],
            vertex_offsets=views[4],
            vertex_entries=views[5],
            facet_material=views[6],
            material_params=material_params,
            layer_offsets=views[7],
            layers=layer_values,
            key=key,
        )


def _carries_tangent(tensor: torch.Tensor) -> bool:
    import torch.autograd.forward_ad as forward_ad

    return forward_ad.unpack_dual(tensor).tangent is not None


def _faces_identity(state: Any) -> int:
    faces = getattr(state.structure.geometry, "faces", None)
    return id(faces) if faces is not None else id(state.structure.geometry)


def _selected_states(snapshot: Any, structure_ids: tuple[int, ...] | None) -> list[Any]:
    states = getattr(snapshot, "structures", None)
    if states is None:
        raise TypeError("snapshot must expose structures; pass a witwin.core SceneSnapshot")
    by_id = {int(state.structure_id): state for state in states}
    if structure_ids is None:
        selected = sorted(by_id)
    else:
        selected = sorted(int(value) for value in structure_ids)
        missing = [value for value in selected if value not in by_id]
        if missing:
            raise ValueError(
                f"MeshTargets structure_ids {missing} are not in this snapshot, which carries {sorted(by_id)}"
            )
    if not selected:
        raise ValueError("a mesh target needs at least one structure; the snapshot carries none")
    return [by_id[value] for value in selected]


def _authored_faces(structure: Any) -> np.ndarray:
    faces = getattr(structure.geometry, "faces", None)
    if not isinstance(faces, torch.Tensor):
        faces = structure.geometry.to_mesh()[1]
    faces = faces.detach().to(device="cpu", dtype=torch.int64).numpy()
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"structure {int(structure.structure_id)} faces must have shape (F, 3)")
    return faces


def _kept_faces(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Faces with a non-zero area, each distinct vertex triple once.

    A duplicate face would scatter twice; a zero-area one has no normal.
    """

    corners = points[faces]
    area2 = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    scale = max(float(np.abs(points).max()) if points.size else 0.0, 1.0)
    kept = area2 > (1e-12 * scale) ** 2
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    unique = np.zeros(faces.shape[0], dtype=bool)
    unique[first] = True
    return kept & unique


def _cluster(centroids: np.ndarray, *, cluster_size_m: float | None, max_sites: int | None) -> np.ndarray:
    """Site index per facet: identity, cubic cells of a declared edge, or the finest cells within a budget.

    Sites are numbered in ascending order of their first member facet, so the
    site order is a function of the face order alone.
    """

    count = centroids.shape[0]
    if cluster_size_m is None and (max_sites is None or count <= max_sites):
        return np.arange(count, dtype=np.int64)
    if cluster_size_m is not None:
        return _cells(centroids, float(cluster_size_m))
    span = float(np.ptp(centroids, axis=0).max()) if count > 1 else 1.0
    low, high = span * 1e-6, max(span, 1e-9) * 2.0
    best = _cells(centroids, high)
    for _ in range(48):
        middle = math.sqrt(low * high)
        labels = _cells(centroids, middle)
        if int(labels.max()) + 1 <= int(max_sites):  # type: ignore[arg-type]
            best, high = labels, middle
        else:
            low = middle
    return best


def _cells(centroids: np.ndarray, size: float) -> np.ndarray:
    keys = np.floor(centroids / size).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first, kind="stable"), kind="stable")
    return order[inverse.reshape(-1)].astype(np.int64)


def _material_rows(material: Any, carrier_hz: float) -> tuple[float, float, list[list[float]]]:
    """``(model, rms_height_m, layers)`` of one Core material at the carrier.

    A perfect conductor is the exact ``r_TE = -1``, ``r_TM = +1``. Anything else
    is its layer stack in vacuum - the authored ``layers``, or one layer of the
    material itself of ``thickness_m`` - each layer ``(thickness_m, Re eps,
    Im eps, mu_r)`` with ``eps`` the complex relative permittivity Core
    evaluates at the carrier (``Im eps <= 0``). The front roughness's rms
    height drives the coherent loss and the diffuse term.
    """

    from witwin.core import MaterialLayer

    roughness = getattr(material, "roughness_front", None)
    rms = 0.0 if roughness is None else float(roughness.rms_height_m)
    if getattr(material, "conductor_model", None) == "perfect":
        return float(MODEL_PEC), rms, []
    layers = tuple(getattr(material, "layers", ()) or ())
    if not layers:
        layers = (
            MaterialLayer(
                thickness_m=material.thickness_m,
                eps_r=material.eps_r,
                sigma_e=material.sigma_e,
                mu_r=material.mu_r,
                dispersion=material.dispersion,
            ),
        )
    rows = []
    for layer in layers:
        sample = layer.evaluate_at_frequency(carrier_hz)
        eps = complex(sample.eps_r)
        mu = complex(sample.mu_r)
        if abs(mu.imag) > 0.0:
            raise NotImplementedError("a magnetic loss tangent is not part of the facet reflection law")
        rows.append([float(layer.thickness_m), eps.real, min(eps.imag, 0.0), mu.real])
    return float(MODEL_LAYERED), rms, rows


def _world_vertices(state: Any, *, device: torch.device) -> torch.Tensor:
    """One structure's world vertices at the snapshot, float32 on ``device``."""

    from witwin.core import Mesh
    from witwin.core.math import quat_from_euler, quat_to_rotation_matrix

    geometry = state.structure.geometry
    deformation = state.deformation
    if deformation is not None:
        if not isinstance(geometry, Mesh):
            raise TypeError("a deformed mesh target requires witwin.core.Mesh geometry")
        local = deformation.vertices if deformation.vertices is not None else geometry.vertices + deformation.offsets
        local = local.to(device=device, dtype=torch.float32)
        if geometry.recenter:
            local = local - 0.5 * (local.amin(dim=0) + local.amax(dim=0))
        rotation = geometry.rotation.to(device=device, dtype=torch.float32)
        quaternion = rotation if rotation.shape == (4,) else quat_from_euler(*rotation, device=device)
        vertices = (local * geometry.scale.to(device=device, dtype=torch.float32)) @ quat_to_rotation_matrix(
            quaternion
        ).T + geometry.position.to(device=device, dtype=torch.float32)
    else:
        vertices = geometry.to_mesh()[0].to(device=device, dtype=torch.float32)
    motion = state.rigid_motion
    if motion is not None:
        if motion.rotation is not None:
            value = torch.as_tensor(motion.rotation, device=device, dtype=torch.float32)
            quaternion = value if value.shape == (4,) else quat_from_euler(value[2], value[1], value[0], device=device)
            vertices = vertices @ quat_to_rotation_matrix(quaternion).T
        if motion.translation is not None:
            vertices = vertices + motion.translation.to(device=device, dtype=torch.float32)
    return vertices


__all__: list[str] = []
