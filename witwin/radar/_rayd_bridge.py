"""Private RayD/Dr.Jit interop helpers for radar tracing."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import drjit as dr
import numpy as np
import rayd.drjit as rd
import torch


Float = dr.cuda.ad.Float
UInt32 = dr.cuda.UInt
Bool = dr.cuda.ad.Bool
Point3f = dr.cuda.ad.Array3f
TensorXf = dr.cuda.ad.TensorXf


@dataclass(frozen=True)
class RayDMeshState:
    name: str
    mesh_id: int
    num_vertices: int
    num_faces: int
    eps_r: float
    dynamic: bool
    face_indices: tuple[object, object, object]


@dataclass(frozen=True)
class RayDCameraBasis:
    origin: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    back: tuple[float, float, float]


@dataclass
class MultipathBuffers:
    points: list[torch.Tensor]
    intensities: list[torch.Tensor]
    entry_points: list[torch.Tensor]
    fixed_lengths: list[torch.Tensor]
    depths: list[torch.Tensor]
    normals: list[torch.Tensor]

    @classmethod
    def empty(cls) -> "MultipathBuffers":
        return cls([], [], [], [], [], [])

    @property
    def has_points(self) -> bool:
        return bool(self.points)


def faces_array(faces) -> np.ndarray:
    faces_np = np.asarray(faces, dtype=np.int32)
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError("Compiled mesh faces must have shape (F, 3).")
    return np.ascontiguousarray(faces_np)


def make_rayd_faces(faces: np.ndarray):
    return dr.cuda.Array3i(
        faces[:, 0].tolist(),
        faces[:, 1].tolist(),
        faces[:, 2].tolist(),
    )


def make_face_indices(faces: np.ndarray) -> tuple[object, object, object]:
    return (
        UInt32(faces[:, 0].tolist()),
        UInt32(faces[:, 1].tolist()),
        UInt32(faces[:, 2].tolist()),
    )


def torch_vertices_to_rayd(vertices: torch.Tensor, *, ad: bool):
    array_type = Point3f if ad else dr.cuda.Array3f
    float_type = Float if ad else dr.cuda.Float
    detached = vertices.detach()
    if detached.is_cuda:
        # Zero-copy interop: Dr.Jit wraps CUDA torch tensors via DLPack, so
        # the per-iteration vertex sync never round-trips through the CPU.
        columns = detached.to(dtype=torch.float32).transpose(0, 1).contiguous()
        return array_type(
            float_type(columns[0]),
            float_type(columns[1]),
            float_type(columns[2]),
        )
    values = detached.to(device="cpu", dtype=torch.float32).numpy()
    return array_type(
        values[:, 0].tolist(),
        values[:, 1].tolist(),
        values[:, 2].tolist(),
    )


def wrapped_vertices_to_point3f(vertices, num_vertices: int):
    flat = dr.ravel(vertices)
    idx = dr.arange(UInt32, num_vertices)
    return Point3f(
        dr.gather(Float, flat, idx * 3),
        dr.gather(Float, flat, idx * 3 + 1),
        dr.gather(Float, flat, idx * 3 + 2),
    )


def renderable_signature(renderables) -> tuple:
    signature = []
    for name, mesh_data in renderables.items():
        faces = faces_array(mesh_data.faces)
        signature.append(
            (
                name,
                int(mesh_data.vertices.shape[0]),
                int(faces.shape[0]),
                faces.tobytes(),
                float(mesh_data.eps_r),
                bool(mesh_data.dynamic),
            )
        )
    return tuple(signature)


class RayDSceneCache:
    """Builds and refits a RayD scene from compiled radar renderables."""

    def __init__(self, device: torch.device):
        self.device = device
        self.scene = None
        self.mesh_states: list[RayDMeshState] = []
        self.vertex_tensors: list[torch.Tensor] = []
        self.signature = None

    def prepare(self, renderables, *, dirty_level: int, dirty_full: int, mark_clean) -> bool:
        if not renderables:
            self.scene = None
            self.mesh_states = []
            self.vertex_tensors = []
            self.signature = None
            mark_clean()
            return False

        # Topology changes always raise the scene to DIRTY_FULL, so the
        # (expensive, CPU-side) signature hash only needs to run on full
        # rebuild candidates; vertex-only updates take the refit path.
        if self.scene is None or dirty_level >= dirty_full:
            signature = renderable_signature(renderables)
            if self.scene is None or signature != self.signature:
                self._build(renderables)
            else:
                self.sync_vertices(renderables)
        elif dirty_level > 0:
            self.sync_vertices(renderables)
        else:
            self.sync_vertices(renderables, dynamic_only=True)
        mark_clean()
        return bool(self.mesh_states)

    def _build(self, renderables) -> None:
        require_cuda(self.device)
        scene = rd.Scene()
        mesh_states: list[RayDMeshState] = []
        vertex_tensors: list[torch.Tensor] = []
        for name, mesh_data in renderables.items():
            vertices = as_cuda_vertices(mesh_data.vertices, self.device)
            faces = faces_array(mesh_data.faces)
            mesh = rd.Mesh(torch_vertices_to_rayd(vertices, ad=False), make_rayd_faces(faces))
            if hasattr(mesh, "use_face_normals"):
                mesh.use_face_normals = True
            mesh_id = scene.add_mesh(mesh, dynamic=True)
            mesh_states.append(
                RayDMeshState(
                    name=name,
                    mesh_id=int(mesh_id),
                    num_vertices=int(vertices.shape[0]),
                    num_faces=int(faces.shape[0]),
                    eps_r=float(mesh_data.eps_r),
                    dynamic=bool(mesh_data.dynamic),
                    face_indices=make_face_indices(faces),
                )
            )
            vertex_tensors.append(vertices)
        scene.build()
        self.scene = scene
        self.mesh_states = mesh_states
        self.vertex_tensors = vertex_tensors
        self.signature = renderable_signature(renderables)

    def sync_vertices(self, renderables, *, dynamic_only: bool = False) -> None:
        if self.scene is None:
            return
        updated_states: list[RayDMeshState] = []
        updated_vertices: list[torch.Tensor] = []
        updated_any = False
        for state, cached_vertices in zip(self.mesh_states, self.vertex_tensors):
            mesh_data = renderables[state.name]
            should_update = not dynamic_only or state.dynamic or bool(mesh_data.dynamic)
            if should_update:
                vertices = as_cuda_vertices(mesh_data.vertices, self.device)
                self.scene.update_mesh_vertices(state.mesh_id, torch_vertices_to_rayd(vertices, ad=True))
                updated_any = True
            else:
                vertices = cached_vertices
            updated_states.append(
                replace(
                    state,
                    eps_r=float(mesh_data.eps_r),
                    dynamic=bool(mesh_data.dynamic),
                )
            )
            updated_vertices.append(vertices)
        self.mesh_states = updated_states
        self.vertex_tensors = updated_vertices
        if updated_any:
            self.scene.sync()

    def update_from_wrapped_inputs(self, vertex_inputs, differentiable) -> None:
        updated_any = False
        for state, vertices, should_update in zip(self.mesh_states, vertex_inputs, differentiable):
            if not should_update:
                continue
            self.scene.update_mesh_vertices(
                state.mesh_id,
                wrapped_vertices_to_point3f(vertices, state.num_vertices),
            )
            updated_any = True
        if updated_any:
            self.scene.sync()

    def vertex_inputs(self, renderables) -> list[torch.Tensor]:
        return [
            as_cuda_vertices(renderables[state.name].vertices, self.device)
            if renderables[state.name].vertices.requires_grad
            else cached
            for state, cached in zip(self.mesh_states, self.vertex_tensors)
        ]

    def dynamic_meshes(self) -> list[RayDMeshState]:
        return [state for state in self.mesh_states if state.dynamic]

    def lookup_eps_r(self, shape_id, default_eps_r: float):
        eps_r = Float(default_eps_r)
        for state in self.mesh_states:
            eps_r = dr.select(shape_id == state.mesh_id, Float(state.eps_r), eps_r)
        return eps_r


def require_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("RayD tracing requires a CUDA radar device.")
    if device.index is not None:
        rd.set_device(int(device.index))


def as_cuda_vertices(vertices: torch.Tensor, device: torch.device) -> torch.Tensor:
    return vertices.to(device=device, dtype=torch.float32).contiguous()


def camera_basis(radar) -> RayDCameraBasis:
    position = radar.position.to(dtype=torch.float32)
    target = radar.target.to(dtype=torch.float32)
    up = radar.up.to(dtype=torch.float32)
    forward = target - position
    forward = forward / torch.linalg.norm(forward)
    right = torch.cross(forward, up, dim=0)
    right = right / torch.linalg.norm(right)
    true_up = torch.cross(right, forward, dim=0)
    true_up = true_up / torch.linalg.norm(true_up)
    back = -forward
    return RayDCameraBasis(
        origin=tuple(position.detach().cpu().tolist()),
        right=tuple(right.detach().cpu().tolist()),
        up=tuple(true_up.detach().cpu().tolist()),
        back=tuple(back.detach().cpu().tolist()),
    )


def make_perspective_rays(radar, resolution: int, *, start: int, count: int):
    basis = camera_basis(radar)
    idx = dr.arange(UInt32, count) + int(start)
    px = Float(idx % resolution)
    py = Float(idx // resolution)
    tan_half = math.tan(math.radians(float(radar.fov)) * 0.5)
    x = ((px + 0.5) / float(resolution) * 2.0 - 1.0) * tan_half
    y = (1.0 - (py + 0.5) / float(resolution) * 2.0) * tan_half
    z = Float(-1.0)
    direction = dr.normalize(
        Point3f(
            x * basis.right[0] + y * basis.up[0] + z * basis.back[0],
            x * basis.right[1] + y * basis.up[1] + z * basis.back[1],
            x * basis.right[2] + y * basis.up[2] + z * basis.back[2],
        )
    )
    origin = Point3f(
        dr.full(Float, basis.origin[0], count),
        dr.full(Float, basis.origin[1], count),
        dr.full(Float, basis.origin[2], count),
    )
    return rd.RayAD(origin, direction)
