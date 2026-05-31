"""RayD-based ray tracing for radar scenes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import drjit as dr
import numpy as np
import rayd as rd
import torch

from .material import fresnel
from .types import SamplingMode


Float = dr.cuda.ad.Float
UInt32 = dr.cuda.UInt
Bool = dr.cuda.ad.Bool
Point3f = dr.cuda.ad.Array3f
TensorXf = dr.cuda.ad.TensorXf


@dataclass(frozen=True)
class _RayDMeshState:
    name: str
    mesh_id: int
    num_vertices: int
    num_faces: int
    eps_r: float
    dynamic: bool
    face_columns: tuple[UInt32, UInt32, UInt32]


class TraceResult:
    """Opaque trace result. Supports ``points, intensities = tracer.trace()``."""

    __slots__ = (
        "points",
        "intensities",
        "entry_points",
        "fixed_path_lengths",
        "depths",
        "normals",
        "_tri_indices",
    )

    def __init__(
        self,
        points,
        intensities,
        tri_indices=None,
        *,
        entry_points=None,
        fixed_path_lengths=None,
        depths=None,
        normals=None,
    ):
        self.points = points
        self.intensities = intensities
        self.entry_points = points if entry_points is None else entry_points
        if fixed_path_lengths is None:
            fixed_path_lengths = torch.zeros(points.shape[0], dtype=torch.float32, device=points.device)
        self.fixed_path_lengths = fixed_path_lengths
        if depths is None:
            depths = torch.zeros(points.shape[0], dtype=torch.int32, device=points.device)
        self.depths = depths
        self.normals = normals
        self._tri_indices = tri_indices

    def __iter__(self):
        yield self.points
        yield self.intensities

    def __repr__(self):
        return f"TraceResult({self.points.shape[0]} points)"


def _faces_array(faces) -> np.ndarray:
    faces_np = np.asarray(faces, dtype=np.int32)
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError("Compiled mesh faces must have shape (F, 3).")
    return np.ascontiguousarray(faces_np)


def _rayd_faces(faces: np.ndarray):
    return dr.cuda.Array3i(
        faces[:, 0].tolist(),
        faces[:, 1].tolist(),
        faces[:, 2].tolist(),
    )


def _face_columns(faces: np.ndarray) -> tuple[UInt32, UInt32, UInt32]:
    return (
        UInt32(faces[:, 0].tolist()),
        UInt32(faces[:, 1].tolist()),
        UInt32(faces[:, 2].tolist()),
    )


def _torch_vertices_to_cuda_array3f(vertices: torch.Tensor):
    values = vertices.detach().to(device="cpu", dtype=torch.float32).numpy()
    return dr.cuda.Array3f(
        values[:, 0].tolist(),
        values[:, 1].tolist(),
        values[:, 2].tolist(),
    )


def _torch_vertices_to_cuda_ad_array3f(vertices: torch.Tensor):
    values = vertices.detach().to(device="cpu", dtype=torch.float32).numpy()
    return Point3f(
        values[:, 0].tolist(),
        values[:, 1].tolist(),
        values[:, 2].tolist(),
    )


def _wrapped_vertices_to_point3f(vertices, num_vertices: int):
    flat = dr.ravel(vertices)
    idx = dr.arange(UInt32, num_vertices)
    return Point3f(
        dr.gather(Float, flat, idx * 3),
        dr.gather(Float, flat, idx * 3 + 1),
        dr.gather(Float, flat, idx * 3 + 2),
    )


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

        self._rd_scene = None
        self._mesh_states: list[_RayDMeshState] = []
        self._renderable_signature = None

    def _empty_trace(self, *, include_tri_indices: bool = False) -> TraceResult:
        device = self.radar.device
        tri_indices = None
        if include_tri_indices:
            tri_indices = torch.empty((0,), dtype=torch.int64, device=device)
        return TraceResult(
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            tri_indices,
            entry_points=torch.empty((0, 3), dtype=torch.float32, device=device),
            fixed_path_lengths=torch.empty((0,), dtype=torch.float32, device=device),
            depths=torch.empty((0,), dtype=torch.int32, device=device),
            normals=torch.empty((0, 3), dtype=torch.float32, device=device),
        )

    def _require_cuda(self) -> None:
        if self.radar.device.type != "cuda":
            raise RuntimeError("RayD tracing requires a CUDA radar device.")
        index = self.radar.device.index
        if index is not None:
            rd.set_device(int(index))

    def _signature(self, renderables):
        signature = []
        for name, mesh_data in renderables.items():
            faces = _faces_array(mesh_data.faces)
            signature.append(
                (
                    name,
                    int(mesh_data.vertices.shape[0]),
                    int(faces.shape[0]),
                    faces.tobytes(),
                )
            )
        return tuple(signature)

    def _vertex_inputs(self, renderables) -> list[torch.Tensor]:
        return [
            renderables[state.name].vertices.to(device=self.radar.device, dtype=torch.float32).contiguous()
            for state in self._mesh_states
        ]

    def _build_scene(self, renderables) -> None:
        self._require_cuda()
        rd_scene = rd.Scene()
        mesh_states: list[_RayDMeshState] = []
        for name, mesh_data in renderables.items():
            vertices = mesh_data.vertices.to(device=self.radar.device, dtype=torch.float32).contiguous()
            faces = _faces_array(mesh_data.faces)
            mesh = rd.Mesh(_torch_vertices_to_cuda_array3f(vertices), _rayd_faces(faces))
            if hasattr(mesh, "use_face_normals"):
                mesh.use_face_normals = True
            # Mark every mesh dynamic so structure transforms and differentiable
            # geometry can be refit without rebuilding the whole OptiX scene.
            mesh_id = rd_scene.add_mesh(mesh, dynamic=True)
            mesh_states.append(
                _RayDMeshState(
                    name=name,
                    mesh_id=int(mesh_id),
                    num_vertices=int(vertices.shape[0]),
                    num_faces=int(faces.shape[0]),
                    eps_r=float(mesh_data.eps_r),
                    dynamic=bool(mesh_data.dynamic),
                    face_columns=_face_columns(faces),
                )
            )
        rd_scene.build()
        self._rd_scene = rd_scene
        self._mesh_states = mesh_states
        self._renderable_signature = self._signature(renderables)
        self.scene.mark_clean()

    def _sync_scene_vertices(self, renderables) -> None:
        if self._rd_scene is None:
            return
        for state in self._mesh_states:
            vertices = renderables[state.name].vertices.to(device=self.radar.device, dtype=torch.float32).contiguous()
            self._rd_scene.update_mesh_vertices(state.mesh_id, _torch_vertices_to_cuda_ad_array3f(vertices))
        if self._mesh_states:
            self._rd_scene.sync()
        self.scene.mark_clean()

    def _prepare_scene(self, renderables) -> bool:
        if not renderables:
            self._rd_scene = None
            self._mesh_states = []
            self._renderable_signature = None
            self.scene.mark_clean()
            return False
        signature = self._signature(renderables)
        if (
            self._rd_scene is None
            or self.scene.dirty_level >= self.scene.DIRTY_FULL
            or signature != self._renderable_signature
        ):
            self._build_scene(renderables)
        else:
            self._sync_scene_vertices(renderables)
        return bool(self._mesh_states)

    def _update_scene_from_wrapped_inputs(self, vertex_inputs) -> None:
        for state, vertices in zip(self._mesh_states, vertex_inputs):
            self._rd_scene.update_mesh_vertices(
                state.mesh_id,
                _wrapped_vertices_to_point3f(vertices, state.num_vertices),
            )
        if self._mesh_states:
            self._rd_scene.sync()

    def _camera_basis(self):
        position = self.radar.position.to(dtype=torch.float32)
        target = self.radar.target.to(dtype=torch.float32)
        up = self.radar.up.to(dtype=torch.float32)
        forward = target - position
        forward = forward / torch.linalg.norm(forward)
        right = torch.cross(forward, up, dim=0)
        right = right / torch.linalg.norm(right)
        true_up = torch.cross(right, forward, dim=0)
        true_up = true_up / torch.linalg.norm(true_up)
        back = -forward
        return position.detach().cpu().tolist(), right.detach().cpu().tolist(), true_up.detach().cpu().tolist(), back.detach().cpu().tolist()

    def _gen_rays_batch(self, start: int, count: int):
        origin, right, up, back = self._camera_basis()
        idx = dr.arange(UInt32, count) + int(start)
        width = self.resolution
        height = self.resolution
        px = Float(idx % width)
        py = Float(idx // width)
        tan_half = math.tan(math.radians(float(self.radar.fov)) * 0.5)
        x = ((px + 0.5) / float(width) * 2.0 - 1.0) * tan_half
        y = (1.0 - (py + 0.5) / float(height) * 2.0) * tan_half
        z = Float(-1.0)
        direction = dr.normalize(
            Point3f(
                x * right[0] + y * up[0] + z * back[0],
                x * right[1] + y * up[1] + z * back[1],
                x * right[2] + y * up[2] + z * back[2],
            )
        )
        ray_origin = Point3f(
            dr.full(Float, origin[0], count),
            dr.full(Float, origin[1], count),
            dr.full(Float, origin[2], count),
        )
        return rd.RayAD(ray_origin, direction)

    def _gen_rays(self):
        return self._gen_rays_batch(0, self.resolution * self.resolution)

    def _lookup_eps_r(self, shape_id):
        eps_r = Float(self.epsilon_r)
        for state in self._mesh_states:
            eps_r = dr.select(shape_id == state.mesh_id, Float(state.eps_r), eps_r)
        return eps_r

    def trace(self, *, time: float | None = None):
        renderables = self.scene.compile_renderables(time=time)
        if not self._prepare_scene(renderables):
            return self._empty_trace(include_tri_indices=self.sampling == SamplingMode.TRIANGLE)

        if self.sampling == SamplingMode.TRIANGLE:
            return self._trace_triangles(renderables)
        if self.sampling == SamplingMode.PIXEL:
            if self.multipath:
                return self._trace_pixels_multipath()
            return self._trace_pixels(renderables)
        raise AssertionError(f"Unsupported sampling mode '{self.sampling}'.")

    def match(self, a, b):
        if a._tri_indices is not None and b._tri_indices is not None:
            _, idx_a, idx_b = np.intersect1d(
                a._tri_indices.detach().cpu().numpy(),
                b._tri_indices.detach().cpu().numpy(),
                return_indices=True,
            )
            return a.points[idx_a], b.points[idx_b], a.intensities[idx_a]
        n = min(a.points.shape[0], b.points.shape[0])
        return a.points[:n], b.points[:n], a.intensities[:n]

    def render_image(self, *, time: float | None = None):
        renderables = self.scene.compile_renderables(time=time)
        if not self._prepare_scene(renderables):
            return torch.zeros((self.resolution, self.resolution), dtype=torch.float32, device=self.radar.device)

        scene_ref = self._rd_scene
        rays = self._gen_rays()
        vertex_inputs = self._vertex_inputs(renderables)

        @dr.wrap(source="torch", target="drjit")
        def _image(*vertices):
            self._update_scene_from_wrapped_inputs(vertices)
            its = scene_ref.intersect(rays, flags=rd.RayFlags.Geometric)
            normals = its.geo_n
            cos_i = dr.abs(dr.dot(-rays.d, normals))
            reflectance = fresnel(cos_i, self._lookup_eps_r(its.shape_id))
            valid_float = dr.select(its.is_valid(), Float(1.0), Float(0.0))
            return TensorXf(reflectance * valid_float, shape=(self.resolution * self.resolution,))

        return _image(*vertex_inputs).reshape(self.resolution, self.resolution)

    def _get_dynamic_meshes(self):
        return [state for state in self._mesh_states if state.dynamic]

    def _trace_triangles(self, renderables):
        all_points = []
        all_intensities = []
        all_tri_indices = []
        all_normals = []
        tri_offset = 0

        for state in self._get_dynamic_meshes():
            mesh_data = renderables[state.name]
            pts, intensities, normals, tri_idx, num_faces = self._trace_mesh_triangles(
                state,
                mesh_data.vertices.to(device=self.radar.device, dtype=torch.float32).contiguous(),
                mesh_data.eps_r,
            )
            if pts.shape[0] > 0:
                all_points.append(pts)
                all_intensities.append(intensities)
                all_normals.append(normals)
                all_tri_indices.append(tri_idx + tri_offset)
            tri_offset += num_faces

        if not all_points:
            return self._empty_trace(include_tri_indices=True)
        return TraceResult(
            torch.cat(all_points),
            torch.cat(all_intensities),
            torch.cat(all_tri_indices),
            normals=torch.cat(all_normals),
        )

    def _trace_mesh_triangles(self, state: _RayDMeshState, vertices_torch: torch.Tensor, eps_r):
        if state.num_faces == 0:
            empty = (
                torch.empty((0, 3), dtype=torch.float32, device=self.radar.device),
                torch.empty((0,), dtype=torch.float32, device=self.radar.device),
                torch.empty((0, 3), dtype=torch.float32, device=self.radar.device),
                torch.empty((0,), dtype=torch.int64, device=self.radar.device),
            )
            return (*empty, 0)

        scene_ref = self._rd_scene
        face0, face1, face2 = state.face_columns
        origin = self.radar.position.detach().cpu().tolist()

        @dr.wrap(source="torch", target="drjit")
        def _geometry(vertices):
            positions = _wrapped_vertices_to_point3f(vertices, state.num_vertices)
            face_idx = dr.arange(UInt32, state.num_faces)
            i0 = dr.gather(UInt32, face0, face_idx)
            i1 = dr.gather(UInt32, face1, face_idx)
            i2 = dr.gather(UInt32, face2, face_idx)

            v0 = dr.gather(Point3f, positions, i0)
            v1 = dr.gather(Point3f, positions, i1)
            v2 = dr.gather(Point3f, positions, i2)
            centroid = (v0 + v1 + v2) / 3.0
            cross = dr.cross(v1 - v0, v2 - v0)
            cross_len = dr.norm(cross)
            area = 0.5 * cross_len
            normal = cross / (cross_len + 1e-10)

            radar_origin = Point3f(origin[0], origin[1], origin[2])
            view_dir = dr.normalize(radar_origin - centroid)
            front = dr.dot(view_dir, normal) > 0.0

            ray_d = dr.normalize(centroid - radar_origin)
            expected_t = dr.norm(centroid - radar_origin)
            ray_o = Point3f(
                dr.full(Float, origin[0], state.num_faces),
                dr.full(Float, origin[1], state.num_faces),
                dr.full(Float, origin[2], state.num_faces),
            )
            its = scene_ref.intersect(rd.RayAD(ray_o, ray_d), flags=rd.RayFlags.Geometric)
            not_occluded = its.is_valid() & (its.t >= expected_t - 0.01)
            valid = front & not_occluded

            cos_i = dr.abs(dr.dot(view_dir, normal))
            intensity = area * fresnel(cos_i, eps_r)
            valid_float = dr.select(valid, Float(1.0), Float(0.0))

            out = dr.zeros(Float, state.num_faces * 8)
            idx = dr.arange(UInt32, state.num_faces)
            dr.scatter(out, centroid[0], idx * 8)
            dr.scatter(out, centroid[1], idx * 8 + 1)
            dr.scatter(out, centroid[2], idx * 8 + 2)
            dr.scatter(out, intensity, idx * 8 + 3)
            dr.scatter(out, normal[0], idx * 8 + 4)
            dr.scatter(out, normal[1], idx * 8 + 5)
            dr.scatter(out, normal[2], idx * 8 + 6)
            dr.scatter(out, valid_float, idx * 8 + 7)
            return TensorXf(out, shape=(state.num_faces, 8))

        result = _geometry(vertices_torch)
        visible_index = (result[:, 7] > 0.5).nonzero(as_tuple=True)[0]
        if visible_index.numel() == 0:
            empty = (
                torch.empty((0, 3), dtype=torch.float32, device=self.radar.device),
                torch.empty((0,), dtype=torch.float32, device=self.radar.device),
                torch.empty((0, 3), dtype=torch.float32, device=self.radar.device),
                torch.empty((0,), dtype=torch.int64, device=self.radar.device),
            )
            return (*empty, state.num_faces)
        return (
            result[:, :3][visible_index],
            result[:, 3][visible_index],
            result[:, 4:7][visible_index],
            visible_index.to(torch.int64),
            state.num_faces,
        )

    def _trace_pixels(self, renderables):
        scene_ref = self._rd_scene
        rays = self._gen_rays()
        vertex_inputs = self._vertex_inputs(renderables)
        count = self.resolution * self.resolution

        @dr.wrap(source="torch", target="drjit")
        def _pixel(*vertices):
            self._update_scene_from_wrapped_inputs(vertices)
            its = scene_ref.intersect(rays, flags=rd.RayFlags.Geometric)
            normals = its.geo_n
            cos_i = dr.abs(dr.dot(-rays.d, normals))
            reflectance = fresnel(cos_i, self._lookup_eps_r(its.shape_id))
            valid_float = dr.select(its.is_valid(), Float(1.0), Float(0.0))
            reflectance = reflectance * valid_float

            out = dr.zeros(Float, count * 8)
            idx = dr.arange(UInt32, count)
            dr.scatter(out, its.p[0] * valid_float, idx * 8)
            dr.scatter(out, its.p[1] * valid_float, idx * 8 + 1)
            dr.scatter(out, its.p[2] * valid_float, idx * 8 + 2)
            dr.scatter(out, reflectance, idx * 8 + 3)
            dr.scatter(out, valid_float, idx * 8 + 4)
            dr.scatter(out, normals[0] * valid_float, idx * 8 + 5)
            dr.scatter(out, normals[1] * valid_float, idx * 8 + 6)
            dr.scatter(out, normals[2] * valid_float, idx * 8 + 7)
            return TensorXf(out, shape=(count, 8))

        result = _pixel(*vertex_inputs)
        valid_index = (result[:, 4] > 0.5).nonzero(as_tuple=True)[0]
        if valid_index.numel() == 0:
            return self._empty_trace()
        points = result[:, :3][valid_index]
        intensities = result[:, 3][valid_index]
        normals = result[:, 5:8][valid_index]
        return TraceResult(points, intensities, normals=normals)

    def _trace_pixels_multipath(self):
        all_points = []
        all_intensities = []
        all_entry_points = []
        all_fixed_lengths = []
        all_depths = []
        all_normals = []

        for start in range(0, self.resolution * self.resolution, self.ray_batch_size):
            count = min(self.ray_batch_size, self.resolution * self.resolution - start)
            rays = self._gen_rays_batch(start=start, count=count)
            self._trace_pixel_batch(
                rays,
                count=count,
                all_points=all_points,
                all_intensities=all_intensities,
                all_entry_points=all_entry_points,
                all_fixed_lengths=all_fixed_lengths,
                all_depths=all_depths,
                all_normals=all_normals,
            )

        if not all_points:
            return self._empty_trace()
        return TraceResult(
            torch.cat(all_points, dim=0),
            torch.cat(all_intensities, dim=0),
            entry_points=torch.cat(all_entry_points, dim=0),
            fixed_path_lengths=torch.cat(all_fixed_lengths, dim=0),
            depths=torch.cat(all_depths, dim=0),
            normals=torch.cat(all_normals, dim=0),
        )

    def _trace_pixel_batch(
        self,
        rays,
        *,
        count: int,
        all_points,
        all_intensities,
        all_entry_points,
        all_fixed_lengths,
        all_depths,
        all_normals,
    ) -> None:
        active = dr.full(Bool, True, count)
        entry_points = Point3f(0.0, 0.0, 0.0)
        prev_bounce_points = Point3f(0.0, 0.0, 0.0)
        fixed_lengths = dr.zeros(Float, count)
        cumulative_reflectance = dr.full(Float, 1.0, count)

        for depth in range(self.max_reflections + 1):
            if not dr.any(active):
                break

            its = self._rd_scene.intersect(rays, active, flags=rd.RayFlags.Geometric)
            valid = active & its.is_valid()
            if not dr.any(valid):
                break

            hit_points = its.p
            normals = its.geo_n
            incoming = rays.d
            eps_r = self._lookup_eps_r(its.shape_id)
            cos_i = dr.abs(dr.dot(-incoming, normals))
            reflectance = fresnel(cos_i, eps_r)

            if depth == 0:
                emitted_entry_points = hit_points
                emitted_fixed_lengths = dr.zeros(Float, count)
                visible = valid
            else:
                emitted_entry_points = entry_points
                segment_lengths = dr.norm(hit_points - prev_bounce_points)
                emitted_fixed_lengths = fixed_lengths + segment_lengths
                visible = valid & self._visible_from_origin(hit_points, normals, valid)

            emitted_intensities = cumulative_reflectance * reflectance
            emitted_valid = visible & (emitted_intensities > 0)

            self._append_trace_batch(
                hit_points=hit_points,
                intensities=emitted_intensities,
                entry_points=emitted_entry_points,
                fixed_lengths=emitted_fixed_lengths,
                normals=normals,
                depth=depth,
                valid=emitted_valid,
                all_points=all_points,
                all_intensities=all_intensities,
                all_entry_points=all_entry_points,
                all_fixed_lengths=all_fixed_lengths,
                all_depths=all_depths,
                all_normals=all_normals,
            )

            if depth == self.max_reflections:
                break

            reflected_dir = dr.normalize(incoming - 2.0 * dr.dot(incoming, normals) * normals)
            offset_sign = dr.select(dr.dot(reflected_dir, normals) >= 0.0, 1.0, -1.0)
            next_origin = hit_points + normals * (offset_sign * self._RAY_EPSILON)

            rays = rd.RayAD(next_origin, reflected_dir)
            entry_points = hit_points if depth == 0 else emitted_entry_points
            prev_bounce_points = hit_points
            fixed_lengths = dr.zeros(Float, count) if depth == 0 else emitted_fixed_lengths
            cumulative_reflectance = cumulative_reflectance * reflectance
            active = valid

    def _visible_from_origin(self, hit_points, normals, active):
        origin = self.radar.position.detach().cpu().tolist()
        radar_origin = Point3f(origin[0], origin[1], origin[2])
        to_origin = radar_origin - hit_points
        direction = dr.normalize(to_origin)
        offset_sign = dr.select(dr.dot(direction, normals) >= 0.0, 1.0, -1.0)
        shadow_origin = hit_points + normals * (offset_sign * self._RAY_EPSILON)
        shadow_si = self._rd_scene.intersect(rd.RayAD(shadow_origin, direction), active, flags=rd.RayFlags.Geometric)
        expected_t = dr.norm(to_origin)
        return (~shadow_si.is_valid()) | (shadow_si.t >= expected_t - self._VISIBILITY_TOLERANCE)

    def _append_trace_batch(
        self,
        *,
        hit_points,
        intensities,
        entry_points,
        fixed_lengths,
        normals,
        depth: int,
        valid,
        all_points,
        all_intensities,
        all_entry_points,
        all_fixed_lengths,
        all_depths,
        all_normals,
    ) -> None:
        if not dr.any(valid):
            return

        count = dr.width(intensities)
        idx = dr.arange(UInt32, count)
        out = dr.zeros(Float, count * 12)
        valid_float = dr.select(valid, Float(1.0), Float(0.0))
        dr.scatter(out, hit_points[0], idx * 12)
        dr.scatter(out, hit_points[1], idx * 12 + 1)
        dr.scatter(out, hit_points[2], idx * 12 + 2)
        dr.scatter(out, intensities, idx * 12 + 3)
        dr.scatter(out, entry_points[0], idx * 12 + 4)
        dr.scatter(out, entry_points[1], idx * 12 + 5)
        dr.scatter(out, entry_points[2], idx * 12 + 6)
        dr.scatter(out, fixed_lengths, idx * 12 + 7)
        dr.scatter(out, Float(float(depth)), idx * 12 + 8)
        dr.scatter(out, normals[0], idx * 12 + 9)
        dr.scatter(out, normals[1], idx * 12 + 10)
        dr.scatter(out, normals[2], idx * 12 + 11)

        packed = TensorXf(out, shape=(count, 12)).torch()
        mask = TensorXf(valid_float).torch() > 0.5
        packed = packed[mask]
        if packed.numel() == 0:
            return

        all_points.append(packed[:, :3])
        all_intensities.append(packed[:, 3])
        all_entry_points.append(packed[:, 4:7])
        all_fixed_lengths.append(packed[:, 7])
        all_depths.append(packed[:, 8].to(torch.int32))
        all_normals.append(packed[:, 9:12])
