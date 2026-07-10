"""Differentiable SMPL geometry with optional smplpytorch dependency."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from witwin.core import GeometryBase


class _Arr(np.ndarray):
    @property
    def r(self):
        return np.asarray(self)


class _ChRecon:
    def __init__(self, *args, **kwargs):
        self._data = np.array([])
        for value in args:
            if isinstance(value, np.ndarray):
                self._data = value
                return
            if isinstance(value, _ChRecon):
                self._data = value._data
                return

    def __setstate__(self, state):
        if isinstance(state, dict):
            for value in state.values():
                if isinstance(value, np.ndarray):
                    self._data = value
                    return
                if isinstance(value, _ChRecon):
                    self._data = value._data
                    return
        elif isinstance(state, np.ndarray):
            self._data = state
        elif isinstance(state, (list, tuple)):
            for value in state:
                if isinstance(value, np.ndarray):
                    self._data = value
                    return


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChRecon
        return super().find_class(module, name)


def _ready_arguments_numpy(fname_or_dict):
    import cv2

    if not isinstance(fname_or_dict, dict):
        with open(fname_or_dict, "rb") as handle:
            data = _Unpickler(handle, encoding="latin1").load()
    else:
        data = fname_or_dict

    for key, value in list(data.items()):
        if isinstance(value, _ChRecon):
            data[key] = value._data

    want_shape_model = "shapedirs" in data
    num_pose_params = data["kintree_table"].shape[1] * 3

    if "trans" not in data:
        data["trans"] = np.zeros(3)
    if "pose" not in data:
        data["pose"] = np.zeros(num_pose_params)
    if "shapedirs" in data and "betas" not in data:
        data["betas"] = np.zeros(data["shapedirs"].shape[-1])

    if want_shape_model:
        data["v_shaped"] = data["shapedirs"].dot(data["betas"]) + data["v_template"]
        v_shaped = data["v_shaped"]
        joint_regressor = data["J_regressor"]
        data["J"] = np.column_stack(
            [
                joint_regressor.dot(v_shaped[:, 0]),
                joint_regressor.dot(v_shaped[:, 1]),
                joint_regressor.dot(v_shaped[:, 2]),
            ]
        )
        pose = data["pose"].ravel()[3:]
        rotations = np.concatenate(
            [(cv2.Rodrigues(np.array(pp, dtype=np.float64))[0] - np.eye(3)).ravel() for pp in pose.reshape((-1, 3))]
        ).ravel()
        data["v_posed"] = v_shaped + data["posedirs"].dot(rotations)
    else:
        pose = data["pose"].ravel()[3:]
        rotations = np.concatenate(
            [(cv2.Rodrigues(np.array(pp, dtype=np.float64))[0] - np.eye(3)).ravel() for pp in pose.reshape((-1, 3))]
        ).ravel()
        data["v_posed"] = data["v_template"] + data["posedirs"].dot(rotations)

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = value.view(_Arr)
    return data


def _setup_smpl_compat():
    try:
        import chumpy  # noqa: F401

        return
    except ImportError:
        pass

    try:
        import smplpytorch.native.webuser.serialization as serialization
    except ImportError:
        return

    serialization.ready_arguments = _ready_arguments_numpy


try:
    _setup_smpl_compat()
    from smplpytorch.pytorch.smpl_layer import SMPL_Layer

    _SMPL_AVAILABLE = True
except ImportError:
    _SMPL_AVAILABLE = False


_SMPL_LAYER_CACHE: dict[tuple[str, str, str], Any] = {}
_SMPL_FACES_CACHE: dict[tuple[str, str, str], np.ndarray] = {}


def _resolve_scene_device(device: str | None) -> str:
    requested = "cuda" if device is None else device
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "SMPLBody defaults to CUDA, but torch.cuda.is_available() is False. "
            "Pass device='cpu' only for scene construction or non-rendering workflows."
        )
    return str(resolved)


def _default_smpl_model_root() -> str:
    return str(Path(__file__).resolve().parents[4] / "radar" / "models" / "smpl_models")


def _to_vertex_tensor(value, *, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _get_smpl_layer(*, gender: str, model_root: str, device: str):
    if not _SMPL_AVAILABLE:
        raise ImportError("smplpytorch is required to instantiate or evaluate SMPLBody.")
    key = (str(gender), str(model_root), str(device))
    layer = _SMPL_LAYER_CACHE.get(key)
    if layer is None:
        layer = SMPL_Layer(center_idx=0, gender=gender, model_root=model_root).to(device)
        _SMPL_LAYER_CACHE[key] = layer
    return layer


def _axis_angle_matrices(rvecs: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues for (J, 3) axis-angle vectors -> (J, 3, 3)."""
    # smplpytorch's zero-rotation guard: offset the vector, not the norm, so
    # the gradient at exactly zero pose stays finite.
    safe = rvecs + 1e-8
    angle = safe.norm(dim=1, keepdim=True)
    axis = safe / angle
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=rvecs.dtype, device=rvecs.device).expand_as(skew)
    sin = torch.sin(angle).unsqueeze(-1)
    cos = torch.cos(angle).unsqueeze(-1)
    return eye + sin * skew + (1.0 - cos) * (skew @ skew)


def _fast_smpl_forward(layer, pose: torch.Tensor, betas: torch.Tensor):
    """Vectorized SMPL LBS equivalent to smplpytorch's forward (batch 1).

    smplpytorch's Python implementation costs ~21 ms per call; this batched
    version is ~2 ms and numerically matches it (verified by diag_fast_lbs).
    """
    device = pose.device
    v_template = layer.th_v_template[0]
    joints_rest = layer.th_J_regressor @ (v_template + torch.einsum("vdk,k->vd", layer.th_shapedirs, betas.reshape(-1)))
    v_shaped = v_template + torch.einsum("vdk,k->vd", layer.th_shapedirs, betas.reshape(-1))

    rotations = _axis_angle_matrices(pose.reshape(24, 3))
    eye = torch.eye(3, dtype=pose.dtype, device=device)
    pose_feature = (rotations[1:] - eye).reshape(-1)
    v_posed = v_shaped + torch.einsum("vdk,k->vd", layer.th_posedirs, pose_feature)

    parents = layer.kintree_parents
    relative = [joints_rest[0]]
    for j in range(1, 24):
        relative.append(joints_rest[j] - joints_rest[parents[j]])

    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=pose.dtype, device=device)
    transforms: list[torch.Tensor] = []
    for j in range(24):
        local = torch.cat([torch.cat([rotations[j], relative[j].reshape(3, 1)], dim=1), bottom], dim=0)
        transforms.append(local if j == 0 else transforms[parents[j]] @ local)
    global_transforms = torch.stack(transforms)

    posed_joints = global_transforms[:, :3, 3]
    corrected_t = posed_joints - torch.einsum("jab,jb->ja", global_transforms[:, :3, :3], joints_rest)
    skin_rot = torch.einsum("vj,jab->vab", layer.th_weights, global_transforms[:, :3, :3])
    skin_t = layer.th_weights @ corrected_t
    vertices = torch.einsum("vab,vb->va", skin_rot, v_posed) + skin_t

    center = posed_joints[0]
    return (vertices - center).unsqueeze(0), (posed_joints - center).unsqueeze(0)


class SMPLBody(GeometryBase):
    """Differentiable SMPL geometry with position and rotation."""

    kind = "smpl"

    def __init__(
        self,
        pose,
        shape,
        *,
        position=(0.0, 0.0, 0.0),
        gender: str = "male",
        model_root: str | None = None,
        rotation=None,
        device=None,
    ):
        super().__init__(position=position, rotation=rotation, device=device)
        tensor_device = str(self.position.device)
        self.pose = _to_vertex_tensor(pose, device=tensor_device).reshape(-1)
        self.shape = _to_vertex_tensor(shape, device=tensor_device).reshape(-1)
        self.gender = str(gender)
        self.model_root = _default_smpl_model_root() if model_root is None else str(model_root)

    def updated(self, **changes) -> "SMPLBody":
        updated = SMPLBody(
            pose=changes.pop("pose", self.pose),
            shape=changes.pop("shape", self.shape),
            position=changes.pop("position", self.position),
            gender=changes.pop("gender", self.gender),
            model_root=changes.pop("model_root", self.model_root),
            rotation=changes.pop("rotation", self.rotation),
            device=changes.pop("device", self.position.device),
        )
        if changes:
            unsupported = ", ".join(sorted(changes))
            raise TypeError(f"Unsupported SMPLBody updates: {unsupported}")
        return updated

    def _evaluate(self, *, device: str):
        layer = _get_smpl_layer(gender=self.gender, model_root=self.model_root, device=device)
        pose_tensor = self.pose.to(device=device, dtype=torch.float32).view(1, -1)
        shape_tensor = self.shape.to(device=device, dtype=torch.float32).view(1, -1)
        if shape_tensor.requires_grad:
            shape_tensor = shape_tensor + 1e-8
        vertices, joints = _fast_smpl_forward(layer, pose_tensor.reshape(-1), shape_tensor.reshape(-1))
        vertices = self._transform_mesh_verts(vertices[0])
        joints = self._transform_mesh_verts(joints[0])
        cache_key = (self.gender, self.model_root, device)
        faces = _SMPL_FACES_CACHE.get(cache_key)
        if faces is None:
            faces = np.ascontiguousarray(layer.th_faces.detach().cpu().numpy().astype(np.int32))
            _SMPL_FACES_CACHE[cache_key] = faces
        return vertices.contiguous(), faces, joints.contiguous()

    def to_mesh(self, segments=16, *, device=None):
        del segments
        resolved_device = _resolve_scene_device(device or self.position.device)
        vertices, faces, _ = self._evaluate(device=resolved_device)
        face_tensor = torch.as_tensor(faces, device=vertices.device, dtype=torch.int64)
        return vertices, face_tensor

    def joints(self, *, device=None) -> torch.Tensor:
        resolved_device = _resolve_scene_device(device or self.position.device)
        _, _, joints = self._evaluate(device=resolved_device)
        return joints
