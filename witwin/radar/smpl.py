"""Differentiable SMPL geometry over smplpytorch.

The SMPL model files are chumpy pickles and chumpy is not a dependency, so
:func:`_load_model` unpickles them with plain NumPy arrays standing in for the
chumpy objects and is installed as smplpytorch's model loader at first use.
smplpytorch and OpenCV form the ``smpl`` optional extra and are imported at
first use as well, so a missing one raises ``ImportError`` naming it from the
call that needed it rather than from ``import witwin.radar``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.autograd.forward_ad as forward_ad
from witwin.core import GeometryBase, Mesh
from witwin.core.dynamics import DeformationState

from .policy import _carries_derivative, resolve_device


class _Arr(np.ndarray):
    """A NumPy array with chumpy's ``.r`` accessor, which ``SMPL_Layer`` reads."""

    @property
    def r(self):
        return np.asarray(self)


class _ChRecon:
    """Stands in for any pickled chumpy object and keeps its first array payload."""

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


def _load_model(fname_or_dict):
    """smplpytorch's ``ready_arguments`` without chumpy: the same keys, plain arrays."""

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


_SMPL_LAYER_CACHE: dict[tuple[str, str, str], Any] = {}


def _default_smpl_model_root() -> str:
    return str(Path(__file__).resolve().parents[2] / "models" / "smpl_models")


def _to_vertex_tensor(value, *, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _get_smpl_layer(*, gender: str, model_root: str, device: str):
    key = (str(gender), str(model_root), str(device))
    layer = _SMPL_LAYER_CACHE.get(key)
    if layer is None:
        from smplpytorch.pytorch import smpl_layer

        # SMPL_Layer.__init__ resolves ``ready_arguments`` through its module
        # globals, so the chumpy-free loader is installed there, before any layer.
        smpl_layer.ready_arguments = _load_model
        layer = smpl_layer.SMPL_Layer(center_idx=0, gender=gender, model_root=model_root).to(device)
        # Immutable model topology stays on the same device as the blend shapes.
        layer._radar_parents = torch.tensor(layer.kintree_parents[1:], dtype=torch.int64, device=device)
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


def _fast_smpl_forward(layer, pose: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    """Vectorized SMPL linear blend skinning: ``(V, 3)`` vertices centred on joint 0.

    ``tests/test_smpl_pose_refusal.py`` pins this against ``SMPL_Layer.forward``.
    Shape blending and local transforms are evaluated once per call; only the
    parent-dependent products follow the articulated tree serially.
    """
    device = pose.device
    v_template = layer.th_v_template[0]
    v_shaped = v_template + torch.einsum("vdk,k->vd", layer.th_shapedirs, betas.reshape(-1))
    joints_rest = layer.th_J_regressor @ v_shaped

    rotations = _axis_angle_matrices(pose.reshape(24, 3))
    eye = torch.eye(3, dtype=pose.dtype, device=device)
    pose_feature = (rotations[1:] - eye).reshape(-1)
    v_posed = v_shaped + torch.einsum("vdk,k->vd", layer.th_posedirs, pose_feature)

    parents = layer.kintree_parents
    relative = torch.cat((joints_rest[:1], joints_rest[1:] - joints_rest[layer._radar_parents]))
    bottom = torch.eye(4, dtype=pose.dtype, device=device)[3:].expand(24, -1, -1)
    local = torch.cat((torch.cat((rotations, relative[:, :, None]), dim=2), bottom), dim=1)
    transforms = [local[0]]
    for j in range(1, 24):
        transforms.append(transforms[parents[j]] @ local[j])
    global_transforms = torch.stack(transforms)

    posed_joints = global_transforms[:, :3, 3]
    corrected_t = posed_joints - torch.einsum("jab,jb->ja", global_transforms[:, :3, :3], joints_rest)
    skin_rot = torch.einsum("vj,jab->vab", layer.th_weights, global_transforms[:, :3, :3])
    skin_t = layer.th_weights @ corrected_t
    vertices = torch.einsum("vab,vb->va", skin_rot, v_posed) + skin_t
    return vertices - posed_joints[0]


def _refuse_pose_tangent(name: str) -> None:
    """Refuse a forward tangent on the posed body at the deformation bridge.

    A pose or shape GRADIENT is carried (R-ADR-052): every ``DeformationState``
    this bridge publishes holds the posed vertices as a differentiable
    expression of the pose, the shape and the body transform, and the facet
    geometry, the facet response and Channel's fixed-topology replay carry the
    vertex derivative in reverse mode. A forward TANGENT cannot be: Channel
    refuses a compile whose mesh vertices carry a dual (its finite-vertex check
    has no forward-AD formula), and :meth:`SmplPoseDeformation.velocity_at`
    opens its own forward level over the pose, which a caller's dual would have
    to nest inside.
    """

    raise RuntimeError(
        f"{name} carries a forward tangent, and SmplPoseDeformation cannot deliver a forward-mode pose "
        "derivative: Channel refuses to compile mesh vertices that carry a dual, and velocity_at opens its own "
        "forward level over the pose. The pose, the shape and the body transform are reverse-mode leaves: mark "
        "them requires_grad and run the verb with ad_mode='vjp'."
    )


class SMPLBody(GeometryBase):
    """Differentiable SMPL geometry with position and rotation.

    A ``requires_grad`` pose, shape, position or rotation reaches the authored
    vertices this object publishes, and through :class:`SmplPoseDeformation`
    every deformation state of a compiled Core ``Structure`` (R-ADR-052).
    """

    kind = "smpl"

    #: Reverse-mode leaves only: Channel refuses a forward tangent on mesh
    #: vertices at compile, and every one of these reaches the vertices.
    LEAVES: dict[str, tuple[str, ...]] = dict.fromkeys(("pose", "shape", "position", "rotation"), ("vjp",))

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

    def updated(self, **changes) -> SMPLBody:
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
        pose = self.pose.to(device=device, dtype=torch.float32)
        shape = self.shape.to(device=device, dtype=torch.float32)
        vertices = self._transform_mesh_verts(_fast_smpl_forward(layer, pose, shape))
        # Own the returned topology, just as the previous host-to-device copy
        # did: a caller mutating its mesh must not corrupt the cached model.
        return vertices.contiguous(), layer.th_faces.clone()

    def to_mesh(self, segments=16, *, device=None):
        del segments
        resolved_device = str(resolve_device(device or self.position.device, owner="SMPLBody"))
        return self._evaluate(device=resolved_device)


class SmplPoseDeformation:
    """A posed SMPL body as a Core deformation with an analytic vertex velocity.

    The body is represented as a ``witwin.core.dynamics.Deformation`` over a
    fixed rest ``Mesh``. Structure topology - face indexing, primitive IDs and
    material assignment - is authored once; only vertex state evolves. That is
    exactly the condition under which a frozen propagation topology may be
    replayed against the moved geometry: a deformation changes
    ``geometry_version`` and leaves ``topology_version`` alone.

    ``pose_rate`` is the pose derivative in radians per second, in the same
    72-element axis-angle layout as the pose. The vertex velocity is obtained by
    running the posing function itself under forward-mode AD with ``pose_rate``
    as the tangent, so it is the exact derivative of the SAME linear blend
    skinning the primal uses. It is deliberately NOT a difference of two posed
    meshes: a finite difference would carry a truncation error that grows with
    the step, and Core has no velocity descriptor on ``DeformationState``,
    which is why an analytic route has to exist here at all.

    The body must be authored at the identity transform if the rest ``Mesh``
    is, because a deformation replaces the mesh's LOCAL vertices and the mesh's
    own scale, rotation and position are applied afterwards. :meth:`rest_mesh`
    builds a mesh that satisfies that by construction.
    """

    def __init__(self, body: SMPLBody, *, pose_rate, reference_time_s: float = 0.0, device=None) -> None:
        if not isinstance(body, SMPLBody):
            raise TypeError(f"body must be an SMPLBody, got {type(body).__name__}")
        # A gradient on the pose or the shape is carried into every state this
        # bridge publishes. A forward tangent is refused here, at the earliest
        # point the bridge exists, rather than at the compile it cannot cross.
        for name, value in (("body.pose", body.pose), ("body.shape", body.shape)):
            if _carries_derivative(value) == "a forward tangent":
                _refuse_pose_tangent(name)
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
        self._device = str(resolve_device(device or body.position.device, owner="SmplPoseDeformation"))
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

    def _vertices(self, pose: torch.Tensor) -> torch.Tensor:
        """The posing function, as one differentiable expression of the pose.

        Everything between the pose and the vertices is Torch, so a forward-AD
        dual on ``pose`` reaches the vertices. Rebuilding the body from Python
        values here would kill the tangent silently and publish a zero velocity,
        which is indistinguishable from a body that is holding still.
        """

        vertices, _ = self._body.updated(pose=pose, device=self._device).to_mesh(device=self._device)
        return vertices

    def vertices_at(self, time_s: float) -> torch.Tensor:
        return self._vertices(self.pose_at(time_s))

    def at(self, time_s: float):
        """The ``witwin.core`` deformation descriptor at ``time_s``."""

        return DeformationState(vertices=self.vertices_at(time_s))

    def velocity_at(self, time_s: float) -> torch.Tensor:
        """``d(vertices)/dt``, one row per SMPL vertex, in authored order."""

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

        The vertices keep their tape: a gradient on the pose, the shape or the
        body's position or rotation is carried. A forward tangent that arrived
        through the body's TRANSFORM, which the constructor cannot see, is
        refused here, before Channel's compile would.
        """

        vertices, faces = self._body.updated(pose=self._pose, device=self._device).to_mesh(device=self._device)
        if _carries_derivative(vertices) == "a forward tangent":
            _refuse_pose_tangent("the posed body's vertices (through its position or rotation)")
        return Mesh(
            vertices=vertices,
            faces=faces.detach().to(dtype=torch.int64),
            recenter=False,
            fill_mode=mesh_kwargs.pop("fill_mode", "surface"),
            topology_diagnostics=mesh_kwargs.pop("topology_diagnostics", False),
            **mesh_kwargs,
        )

    def mesh_target(
        self,
        scene: Any,
        *,
        structure_id: int,
        material: Any,
        cluster_size: float | None = None,
        max_sites: int | None = None,
        seed: int = 0,
        **mesh_kwargs,
    ) -> tuple[Any, Any]:
        """The posed body as a scattering mesh: ``(DynamicScene, MeshTargets)``.

        The body joins ``scene`` as one Core ``Structure`` of its rest mesh and
        ``material``, deformed by this object, and the returned
        :class:`~witwin.radar.targets.MeshTargets` makes its faces the scatter
        sites (R-ADR-042): each facet returns what its area, orientation and
        material give, the body occludes itself through the legs, and no cross
        section is declared. ``cluster_size`` or ``max_sites`` is the site
        budget for a body of 13 776 faces. ``material`` is required: the
        dielectric a body is made of at the carrier is a statement about the
        world, not a default.
        """

        from witwin.core import Scene, Structure
        from witwin.core.dynamics import DynamicScene

        from .targets import MeshTargets

        structure = Structure(
            geometry=self.rest_mesh(**mesh_kwargs),
            material=material,
            structure_id=int(structure_id),
            material_id=int(structure_id),
            assignment_id=int(structure_id),
            surface_id=int(structure_id),
        )
        world = Scene(structures=(*scene.structures, structure), endpoints=list(scene.endpoints))
        dynamic = DynamicScene(world, structure_deformations={int(structure_id): self})
        targets = MeshTargets(
            structure_ids=(int(structure_id),), cluster_size=cluster_size, max_sites=max_sites, seed=seed
        )
        return dynamic, targets


__all__ = ["SMPLBody", "SmplPoseDeformation"]
