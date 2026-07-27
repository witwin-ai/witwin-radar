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


def _carries_derivative(value) -> str | None:
    """``"requires_grad"``, ``"a forward tangent"``, or ``None``."""

    if not isinstance(value, torch.Tensor):
        return None
    import torch.autograd.forward_ad as forward_ad

    if value.requires_grad:
        return "requires_grad"
    if forward_ad.unpack_dual(value).tangent is not None:
        return "a forward tangent"
    return None


def _refuse_deformation_derivative(name: str, value, carrier: str) -> None:
    """Refuse a pose or shape derivative at the Core/Channel deformation bridge.

    ``SmplPoseDeformation`` publishes two things into the Core world model: a
    rest ``Mesh`` and a per-frame ``DeformationState``. Both cross the
    Core/Channel COMPILE boundary. Until Phase 9 the rest mesh silently
    ``detach()``ed its vertices, so a caller could mark the pose, watch the
    whole chain run, and read ``pose.grad is None`` back - a severed derivative
    with no failure, which is the defect class this phase removes.

    Whether a graph-bearing vertex tensor survives ``Mesh`` construction and a
    Channel compile is a separate, unverified question, and a half-working pose
    gradient would be worse than none. So this refuses, and the deferral is
    named: plumbing a pose derivative into the compiled scene is a design that
    has to be accepted on the Core/Channel side first, not a detach to delete.

    ``SMPLBody`` itself is untouched and still publishes differentiable
    vertices. Its pose and shape gradients reach the LEGACY radar
    ``Scene.compile_renderables()`` mesh and are a working capability there
    (measured); this refusal is about the deformation bridge only.
    """

    raise RuntimeError(
        f"{name} carries {carrier}, and SmplPoseDeformation cannot deliver that "
        "derivative: the rest Mesh and every DeformationState it publishes cross "
        "the Core/Channel compile boundary, and a pose derivative is not plumbed "
        "across it. Accepting this would sever the graph silently and hand back "
        "grad = None. What IS supported is pose_rate as a forward-AD tangent "
        "DIRECTION, which is how velocity_at produces an exact vertex velocity. "
        "For a differentiable body geometry today, mark the mesh VERTICES of a "
        "witwin.core.Mesh, which the fixed-topology reflection route does "
        "support. Plumbing a pose derivative into the compiled scene is a "
        "separate accepted design."
    )


class SMPLBody(GeometryBase):
    """Differentiable SMPL geometry with position and rotation.

    A ``requires_grad`` pose or shape reaches the vertices this publishes and,
    through the legacy radar ``Scene``, the compiled mesh. That route is left
    alone. What does NOT work is routing a pose derivative through
    :class:`SmplPoseDeformation` into a Core ``Structure``, and that one now
    refuses rather than detaching.
    """

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


class SmplPoseDeformation:
    """A posed SMPL body as a Core deformation with an analytic vertex velocity.

    This is the bridge Phase 7 needs and it replaces nothing: the legacy radar
    ``Scene`` treats an ``SMPLBody`` as a piece of geometry it re-poses and
    re-meshes per frame on the host. Here the body is instead a
    ``witwin.core.dynamics.Deformation`` over a fixed rest ``Mesh``, so the
    structure's TOPOLOGY - face indexing, primitive IDs, material assignment -
    is authored once and never moves, and only the vertices evolve. That is
    exactly the condition under which a frozen propagation topology may be
    replayed against the moved geometry: a deformation changes
    ``geometry_version`` and leaves ``topology_version`` alone.

    ``pose_rate`` is the pose derivative in radians per second, in the same
    72-element axis-angle layout as the pose. The vertex velocity is obtained by
    running the posing function itself under forward-mode AD with ``pose_rate``
    as the tangent, so it is the exact derivative of the SAME linear blend
    skinning the primal uses. It is deliberately NOT a difference of two posed
    meshes: a finite difference would carry a truncation error that grows with
    the step, and Core has no velocity descriptor on ``DeformationState``
    (recorded as Phase-7 gap C2), which is why an analytic route has to exist
    here at all.

    The body must be authored at the identity transform if the rest ``Mesh``
    is, because a deformation replaces the mesh's LOCAL vertices and the mesh's
    own scale, rotation and position are applied afterwards. :meth:`rest_mesh`
    builds a mesh that satisfies that by construction.
    """

    def __init__(
        self,
        body: "SMPLBody",
        *,
        pose_rate,
        reference_time_s: float = 0.0,
        device=None,
    ) -> None:
        if not isinstance(body, SMPLBody):
            raise TypeError(
                f"body must be an SMPLBody, got {type(body).__name__}"
            )
        # Refused here, at the earliest point the bridge exists, rather than at
        # the first rest_mesh: a caller that got to build the deformation, run a
        # whole epoch loop and only then learn the pose was never differentiable
        # has already paid for the wrong answer.
        for name, value in (("body.pose", body.pose), ("body.shape", body.shape)):
            carrier = _carries_derivative(value)
            if carrier is not None:
                _refuse_deformation_derivative(name, value, carrier)
        # pose_rate is a forward-AD tangent DIRECTION and never a leaf, the same
        # ADR-038 statement kinematics makes about a velocity: it is consumed by
        # make_dual inside velocity_at, so d(loss)/d(pose_rate) does not exist.
        rate_carrier = _carries_derivative(pose_rate)
        if rate_carrier is not None:
            raise RuntimeError(
                f"pose_rate carries {rate_carrier}, and a pose rate here is a "
                "forward-AD tangent DIRECTION rather than a leaf (ADR-038). "
                "velocity_at feeds it to make_dual as the tangent of the pose "
                "primal, so d(loss)/d(pose_rate) is structurally unavailable in "
                "both AD modes and no gradient would ever come back."
            )
        self._device = _resolve_scene_device(device or body.position.device)
        self._body = body
        rate = _to_vertex_tensor(pose_rate, device=self._device).reshape(-1)
        pose = body.pose.to(device=self._device, dtype=torch.float32).reshape(-1)
        if rate.shape != pose.shape:
            raise ValueError(
                f"pose_rate has {tuple(rate.shape)} entries and the body's "
                f"pose has {tuple(pose.shape)}; the rate must name the same "
                "joints in the same order as the pose it differentiates"
            )
        self._pose = pose
        self._pose_rate = rate
        self.reference_time_s = float(reference_time_s)

    @property
    def pose_rate(self) -> torch.Tensor:
        return self._pose_rate

    def pose_at(self, time_s: float) -> torch.Tensor:
        elapsed = float(time_s) - self.reference_time_s
        return self._pose + self._pose_rate * elapsed

    def body_at(self, time_s: float) -> "SMPLBody":
        return self._body.updated(pose=self.pose_at(time_s), device=self._device)

    def _vertices(self, pose: torch.Tensor) -> torch.Tensor:
        """The posing function, as one differentiable expression of the pose.

        Everything between the pose and the vertices is Torch, so a forward-AD
        dual on ``pose`` reaches the vertices. Rebuilding the body from Python
        values here would kill the tangent silently and publish a zero velocity,
        which is indistinguishable from a body that is holding still.
        """

        vertices, _ = self._body.updated(
            pose=pose, device=self._device
        ).to_mesh(device=self._device)
        return vertices

    def vertices_at(self, time_s: float) -> torch.Tensor:
        return self._vertices(self.pose_at(time_s))

    def at(self, time_s: float):
        """The ``witwin.core`` deformation descriptor at ``time_s``."""

        from witwin.core.dynamics import DeformationState

        return DeformationState(vertices=self.vertices_at(time_s))

    def velocity_at(self, time_s: float) -> torch.Tensor:
        """``d(vertices)/dt``, one row per SMPL vertex, in authored order."""

        import torch.autograd.forward_ad as forward_ad

        with forward_ad.dual_level():
            pose = forward_ad.make_dual(self.pose_at(time_s), self._pose_rate)
            vertices = self._vertices(pose)
            tangent = forward_ad.unpack_dual(vertices).tangent
            if tangent is None:
                raise RuntimeError(
                    "the SMPL posing function produced no vertex tangent; the "
                    "pose stopped being a differentiable expression somewhere "
                    "between make_dual and to_mesh"
                )
            return tangent.clone().contiguous()

    def rest_mesh(self, **mesh_kwargs):
        """The authored rest ``Mesh`` a ``Structure`` carries.

        ``recenter=False`` is mandatory and is not a default:
        ``witwin.core.Mesh`` otherwise subtracts the bounding-box centre from
        the authored vertices, and Channel's compiler re-applies that same
        recentring to the DEFORMED vertices with a different bounding box, so a
        limb that moved would drag the whole body with it.

        The vertices used to be ``detach()``ed here. That detach was the whole
        of the pose-gradient defect: it made the severance invisible. It is now
        a refusal. The body's pose and shape were already checked at
        construction, so what this catches is a derivative that arrived through
        the body's TRANSFORM - a ``requires_grad`` ``position`` or ``rotation``
        reaching ``_transform_mesh_verts`` - which the constructor cannot see.
        """

        from witwin.core import Mesh

        vertices, faces = self._body.updated(
            pose=self._pose, device=self._device
        ).to_mesh(device=self._device)
        carrier = _carries_derivative(vertices)
        if carrier is not None:
            _refuse_deformation_derivative(
                "the posed body's vertices (through its position or rotation)",
                vertices,
                carrier,
            )
        return Mesh(
            vertices=vertices,
            faces=faces.detach().to(dtype=torch.int64),
            recenter=False,
            fill_mode=mesh_kwargs.pop("fill_mode", "surface"),
            topology_diagnostics=mesh_kwargs.pop("topology_diagnostics", False),
            **mesh_kwargs,
        )
