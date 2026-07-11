"""RayD-based ray tracing for radar scenes."""

from __future__ import annotations

from dataclasses import dataclass

import drjit as dr
import rayd.drjit as rd
import torch

from ._rayd_bridge import (
    Bool,
    Float,
    MultipathBuffers,
    Point3f,
    RayDMeshState,
    RayDSceneCache,
    TensorXf,
    UInt32,
    as_cuda_vertices,
    make_perspective_rays,
    wrapped_vertices_to_point3f,
)
from .material import fresnel
from .trace_result import TraceResult, empty_trace
from .types import SamplingMode


class Tracer:
    """Ray tracer for declarative radar scenes."""

    _RAY_EPSILON = 1e-4
    _VISIBILITY_TOLERANCE = 1e-3

    def __init__(
        self,
        scene,
        radar,
        resolution=128,
        epsilon_r=5.0,
        sampling: SamplingMode = "pixel",
        *,
        multipath: bool = False,
        max_reflections: int = 0,
        ray_batch_size: int = 65536,
    ):
        self.scene = scene
        self.radar = radar
        self.resolution = int(resolution)
        self.epsilon_r = float(epsilon_r)
        self.sampling = SamplingMode(sampling)
        self.multipath = bool(multipath)
        self.max_reflections = int(max_reflections)
        self.ray_batch_size = int(ray_batch_size)

        if self.max_reflections < 0:
            raise ValueError("max_reflections must be >= 0.")
        if self.ray_batch_size <= 0:
            raise ValueError("ray_batch_size must be > 0.")
        if self.multipath and self.sampling != SamplingMode.PIXEL:
            raise ValueError("multipath=True requires sampling='pixel'.")

        self._cache = RayDSceneCache(self.radar.device)

    def _empty_trace(self, *, include_tri_indices: bool = False) -> TraceResult:
        return empty_trace(self.radar.device, include_tri_indices=include_tri_indices)

    def _prepare_scene(self, renderables) -> bool:
        return self._cache.prepare(
            renderables,
            dirty_level=self.scene.dirty_level,
            dirty_full=self.scene.DIRTY_FULL,
            mark_clean=self.scene.mark_clean,
        )

    def _gen_rays_batch(self, start: int, count: int):
        return make_perspective_rays(self.radar, self.resolution, start=start, count=count)

    def _gen_rays(self):
        return self._gen_rays_batch(0, self.resolution * self.resolution)

    def trace(self, *, time: float | None = None):
        renderables = self.scene.compile_renderables(time=time)
        if not self._prepare_scene(renderables):
            return self._empty_trace(include_tri_indices=self.sampling == SamplingMode.TRIANGLE)

        if self.sampling == SamplingMode.TRIANGLE:
            return self._trace_triangles(renderables)
        if self.sampling == SamplingMode.PIXEL:
            return self._trace_pixels_multipath() if self.multipath else self._trace_pixels(renderables)
        raise AssertionError(f"Unsupported sampling mode '{self.sampling}'.")

    def match(self, a, b):
        idx_a, idx_b = self.match_indices(a, b)
        return a.points[idx_a], b.points[idx_b], a.intensities[idx_a]

    def match_indices(self, a, b):
        """Return device-resident corresponding path indices for two traces."""
        if a._tri_indices is not None and b._tri_indices is not None:
            if a._tri_indices.numel() == 0 or b._tri_indices.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=a.points.device)
                return empty, empty
            positions = torch.searchsorted(b._tri_indices, a._tri_indices)
            in_bounds = positions < b._tri_indices.numel()
            safe_positions = positions.clamp_max(b._tri_indices.numel() - 1)
            matched = in_bounds & (b._tri_indices[safe_positions] == a._tri_indices)
            idx_a = matched.nonzero(as_tuple=True)[0]
            idx_b = positions[matched]
            return idx_a, idx_b
        n = min(a.points.shape[0], b.points.shape[0])
        indices = torch.arange(n, dtype=torch.int64, device=a.points.device)
        return indices, indices

    def render_image(self, *, time: float | None = None):
        renderables = self.scene.compile_renderables(time=time)
        if not self._prepare_scene(renderables):
            return torch.zeros((self.resolution, self.resolution), dtype=torch.float32, device=self.radar.device)
        return self._trace_primary_rays(renderables, image_only=True).reshape(self.resolution, self.resolution)

    def _trace_pixels(self, renderables):
        result = self._trace_primary_rays(renderables, image_only=False)
        valid_index = (result[:, 4] > 0.5).nonzero(as_tuple=True)[0]
        if valid_index.numel() == 0:
            return self._empty_trace()
        return TraceResult(
            result[:, :3][valid_index],
            result[:, 3][valid_index],
            normals=result[:, 5:8][valid_index],
        )

    def _trace_primary_rays(self, renderables, *, image_only: bool):
        scene = self._cache.scene
        rays = self._gen_rays()
        vertex_inputs = self._cache.vertex_inputs(renderables)
        count = self.resolution * self.resolution

        @dr.wrap(source="torch", target="drjit")
        def _primary(*vertices):
            self._cache.update_from_wrapped_inputs(vertices, differentiable)
            its = scene.intersect(rays, flags=rd.RayFlags.Geometric)
            normals = its.geo_n
            cos_i = dr.abs(dr.dot(-rays.d, normals))
            reflectance = fresnel(cos_i, self._cache.lookup_eps_r(its.shape_id, self.epsilon_r))
            valid_float = dr.select(its.is_valid(), Float(1.0), Float(0.0))
            reflectance = reflectance * valid_float
            if image_only:
                return TensorXf(reflectance, shape=(count,))
            return _pack_primary_trace(its.p, normals, reflectance, valid_float, count)

        differentiable = tuple(vertex.requires_grad for vertex in vertex_inputs)
        return _primary(*vertex_inputs)

    def _trace_triangles(self, renderables):
        chunks = _TriangleChunks()
        tri_offset = 0
        for state in self._cache.dynamic_meshes():
            mesh_data = renderables[state.name]
            vertices = as_cuda_vertices(mesh_data.vertices, self.radar.device)
            result, tri_idx = self._trace_mesh_triangles(state, vertices, mesh_data.eps_r)
            if result.numel() > 0:
                chunks.append(result, tri_idx + tri_offset)
            tri_offset += state.num_faces

        if not chunks.points:
            return self._empty_trace(include_tri_indices=True)
        return TraceResult(
            torch.cat(chunks.points),
            torch.cat(chunks.intensities),
            torch.cat(chunks.tri_indices),
            normals=torch.cat(chunks.normals),
        )

    def _trace_mesh_triangles(self, state: RayDMeshState, vertices_torch: torch.Tensor, eps_r):
        if state.num_faces == 0:
            return _empty_triangle_result(self.radar.device), torch.empty((0,), dtype=torch.int64, device=self.radar.device)

        packed = self._packed_triangle_geometry(state, vertices_torch, eps_r)
        visible_index = (packed[:, 7] > 0.5).nonzero(as_tuple=True)[0]
        if visible_index.numel() == 0:
            return _empty_triangle_result(self.radar.device), visible_index.to(torch.int64)
        return packed[visible_index], visible_index.to(torch.int64)

    def _packed_triangle_geometry(self, state: RayDMeshState, vertices_torch: torch.Tensor, eps_r):
        scene = self._cache.scene
        face0, face1, face2 = state.face_indices
        origin = self.radar.position.detach().cpu().tolist()

        @dr.wrap(source="torch", target="drjit")
        def _geometry(vertices):
            positions = wrapped_vertices_to_point3f(vertices, state.num_vertices)
            face_idx = dr.arange(UInt32, state.num_faces)
            v0 = dr.gather(Point3f, positions, dr.gather(UInt32, face0, face_idx))
            v1 = dr.gather(Point3f, positions, dr.gather(UInt32, face1, face_idx))
            v2 = dr.gather(Point3f, positions, dr.gather(UInt32, face2, face_idx))

            centroid, normal, area = _triangle_geometry(v0, v1, v2)
            radar_origin = Point3f(origin[0], origin[1], origin[2])
            view_dir = dr.normalize(radar_origin - centroid)
            valid = _triangle_visible(scene, origin, centroid, normal, view_dir, state.num_faces)
            intensity = area * fresnel(dr.abs(dr.dot(view_dir, normal)), eps_r)
            return _pack_triangle_trace(centroid, normal, intensity, valid, state.num_faces)

        return _geometry(vertices_torch)

    def _trace_pixels_multipath(self):
        buffers = MultipathBuffers.empty()
        for start in range(0, self.resolution * self.resolution, self.ray_batch_size):
            count = min(self.ray_batch_size, self.resolution * self.resolution - start)
            self._trace_pixel_batch(self._gen_rays_batch(start=start, count=count), count=count, buffers=buffers)

        if not buffers.has_points:
            return self._empty_trace()
        return TraceResult(
            torch.cat(buffers.points, dim=0),
            torch.cat(buffers.intensities, dim=0),
            entry_points=torch.cat(buffers.entry_points, dim=0),
            fixed_path_lengths=torch.cat(buffers.fixed_lengths, dim=0),
            depths=torch.cat(buffers.depths, dim=0),
            normals=torch.cat(buffers.normals, dim=0),
        )

    def _trace_pixel_batch(self, rays, *, count: int, buffers: MultipathBuffers) -> None:
        state = _MultipathState.empty(count)
        for depth in range(self.max_reflections + 1):
            if not dr.any(state.active):
                break

            hit = self._multipath_hit(rays, state, depth)
            if not dr.any(hit.valid):
                break

            self._append_trace_batch(hit, depth=depth, buffers=buffers)
            if depth == self.max_reflections:
                break

            rays = rd.RayAD(hit.next_origin, hit.reflected_dir)
            state = state.next(hit)

    def _multipath_hit(self, rays, state: "_MultipathState", depth: int) -> "_MultipathHit":
        its = self._cache.scene.intersect(rays, state.active, flags=rd.RayFlags.Geometric)
        valid = state.active & its.is_valid()
        hit_points = its.p
        normals = its.geo_n
        reflectance = _surface_reflectance(
            rays.d,
            normals,
            self._cache.lookup_eps_r(its.shape_id, self.epsilon_r),
        )
        emitted_entry_points, emitted_lengths, visible = self._multipath_visibility(
            hit_points, normals, state, valid, depth
        )
        intensity = state.cumulative_reflectance * reflectance
        emitted_valid = visible & (intensity > 0)
        reflected_dir = dr.normalize(rays.d - 2.0 * dr.dot(rays.d, normals) * normals)
        offset_sign = dr.select(dr.dot(reflected_dir, normals) >= 0.0, 1.0, -1.0)
        return _MultipathHit(
            valid=valid,
            emitted_valid=emitted_valid,
            hit_points=hit_points,
            normals=normals,
            intensity=intensity,
            entry_points=emitted_entry_points,
            fixed_lengths=emitted_lengths,
            reflected_dir=reflected_dir,
            next_origin=hit_points + normals * (offset_sign * self._RAY_EPSILON),
            reflectance=reflectance,
        )

    def _multipath_visibility(self, hit_points, normals, state: "_MultipathState", valid, depth: int):
        if depth == 0:
            return hit_points, dr.zeros(Float, dr.width(valid)), valid
        segment_lengths = dr.norm(hit_points - state.prev_bounce_points)
        fixed_lengths = state.fixed_lengths + segment_lengths
        visible = valid & self._visible_from_origin(hit_points, normals, valid)
        return state.entry_points, fixed_lengths, visible

    def _visible_from_origin(self, hit_points, normals, active):
        origin = self.radar.position.detach().cpu().tolist()
        radar_origin = Point3f(origin[0], origin[1], origin[2])
        to_origin = radar_origin - hit_points
        direction = dr.normalize(to_origin)
        offset_sign = dr.select(dr.dot(direction, normals) >= 0.0, 1.0, -1.0)
        shadow_origin = hit_points + normals * (offset_sign * self._RAY_EPSILON)
        shadow = self._cache.scene.intersect(
            rd.RayAD(shadow_origin, direction),
            active,
            flags=rd.RayFlags.Geometric,
        )
        return (~shadow.is_valid()) | (shadow.t >= dr.norm(to_origin) - self._VISIBILITY_TOLERANCE)

    def _append_trace_batch(self, hit: "_MultipathHit", *, depth: int, buffers: MultipathBuffers) -> None:
        if not dr.any(hit.emitted_valid):
            return

        packed = _pack_multipath_trace(hit, depth)
        mask = TensorXf(dr.select(hit.emitted_valid, Float(1.0), Float(0.0))).torch() > 0.5
        packed = packed[mask]
        if packed.numel() == 0:
            return

        buffers.points.append(packed[:, :3])
        buffers.intensities.append(packed[:, 3])
        buffers.entry_points.append(packed[:, 4:7])
        buffers.fixed_lengths.append(packed[:, 7])
        buffers.depths.append(packed[:, 8].to(torch.int32))
        buffers.normals.append(packed[:, 9:12])


class _TriangleChunks:
    def __init__(self):
        self.points = []
        self.intensities = []
        self.normals = []
        self.tri_indices = []

    def append(self, packed: torch.Tensor, tri_indices: torch.Tensor) -> None:
        self.points.append(packed[:, :3])
        self.intensities.append(packed[:, 3])
        self.normals.append(packed[:, 4:7])
        self.tri_indices.append(tri_indices)


@dataclass
class _MultipathState:
    active: object
    entry_points: object
    prev_bounce_points: object
    fixed_lengths: object
    cumulative_reflectance: object

    @classmethod
    def empty(cls, count: int) -> "_MultipathState":
        return cls(
            active=dr.full(Bool, True, count),
            entry_points=Point3f(0.0, 0.0, 0.0),
            prev_bounce_points=Point3f(0.0, 0.0, 0.0),
            fixed_lengths=dr.zeros(Float, count),
            cumulative_reflectance=dr.full(Float, 1.0, count),
        )

    def next(self, hit: "_MultipathHit") -> "_MultipathState":
        return _MultipathState(
            active=hit.valid,
            entry_points=hit.entry_points,
            prev_bounce_points=hit.hit_points,
            fixed_lengths=hit.fixed_lengths,
            cumulative_reflectance=self.cumulative_reflectance * hit.reflectance,
        )


@dataclass
class _MultipathHit:
    valid: object
    emitted_valid: object
    hit_points: object
    normals: object
    intensity: object
    entry_points: object
    fixed_lengths: object
    reflected_dir: object
    next_origin: object
    reflectance: object


def _surface_reflectance(incoming, normals, eps_r):
    return fresnel(dr.abs(dr.dot(-incoming, normals)), eps_r)


def _triangle_geometry(v0, v1, v2):
    centroid = (v0 + v1 + v2) / 3.0
    cross = dr.cross(v1 - v0, v2 - v0)
    cross_len = dr.norm(cross)
    return centroid, cross / (cross_len + 1e-10), 0.5 * cross_len


def _triangle_visible(scene, radar_origin, centroid, normal, view_dir, count: int):
    front = dr.dot(view_dir, normal) > 0.0
    origin = Point3f(
        dr.full(Float, float(radar_origin[0]), count),
        dr.full(Float, float(radar_origin[1]), count),
        dr.full(Float, float(radar_origin[2]), count),
    )
    ray_d = dr.normalize(centroid - origin)
    its = scene.intersect(rd.RayAD(origin, ray_d), flags=rd.RayFlags.Geometric)
    return front & its.is_valid() & (its.t >= dr.norm(centroid - origin) - 0.01)


def _empty_triangle_result(device: torch.device) -> torch.Tensor:
    return torch.empty((0, 8), dtype=torch.float32, device=device)


def _pack_primary_trace(points, normals, reflectance, valid_float, count: int):
    out = dr.zeros(Float, count * 8)
    idx = dr.arange(UInt32, count)
    _scatter_vec3(out, points, idx, 8, valid_float)
    dr.scatter(out, reflectance, idx * 8 + 3)
    dr.scatter(out, valid_float, idx * 8 + 4)
    _scatter_vec3(out, normals, idx, 8, valid_float, offset=5)
    return TensorXf(out, shape=(count, 8))


def _pack_triangle_trace(centroid, normal, intensity, valid, count: int):
    out = dr.zeros(Float, count * 8)
    idx = dr.arange(UInt32, count)
    _scatter_vec3(out, centroid, idx, 8)
    dr.scatter(out, intensity, idx * 8 + 3)
    _scatter_vec3(out, normal, idx, 8, offset=4)
    dr.scatter(out, dr.select(valid, Float(1.0), Float(0.0)), idx * 8 + 7)
    return TensorXf(out, shape=(count, 8))


def _pack_multipath_trace(hit: _MultipathHit, depth: int):
    count = dr.width(hit.intensity)
    out = dr.zeros(Float, count * 12)
    idx = dr.arange(UInt32, count)
    _scatter_vec3(out, hit.hit_points, idx, 12)
    dr.scatter(out, hit.intensity, idx * 12 + 3)
    _scatter_vec3(out, hit.entry_points, idx, 12, offset=4)
    dr.scatter(out, hit.fixed_lengths, idx * 12 + 7)
    dr.scatter(out, Float(float(depth)), idx * 12 + 8)
    _scatter_vec3(out, hit.normals, idx, 12, offset=9)
    return TensorXf(out, shape=(count, 12)).torch()


def _scatter_vec3(out, vector, idx, stride: int, scale=1.0, *, offset: int = 0) -> None:
    dr.scatter(out, vector[0] * scale, idx * stride + offset)
    dr.scatter(out, vector[1] * scale, idx * stride + offset + 1)
    dr.scatter(out, vector[2] * scale, idx * stride + offset + 2)
