"""Declarative radar scene definitions built on shared core structures."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from witwin.core import GeometryBase, Material, Mesh, SMPLBody, SceneBase, Structure
from witwin.core.math import quat_to_rotation_matrix
from .motion import RotationMotion, StructureMotion, TranslationMotion, tensor_scalar, tensor_vec3
from .sensor import Sensor


_UNSET = object()


def _resolve_scene_device(device: str | None) -> str:
    requested = "cuda" if device is None else device
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Radar scenes default to CUDA, but torch.cuda.is_available() is False. "
            "Pass device='cpu' only for scene construction or non-rendering workflows."
        )
    return str(resolved)


def _to_tensor3(value, *, device: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    if tensor.shape != (3,):
        raise ValueError("value must have shape (3,).")
    return tensor


def _to_vertex_tensor(value, *, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _to_faces_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    faces = np.asarray(value, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3).")
    return np.ascontiguousarray(faces)


def _materialize_geometry(geometry: GeometryBase | Mesh, *, device: str) -> tuple[torch.Tensor, np.ndarray]:
    if isinstance(geometry, (Mesh, SMPLBody)):
        vertices, faces = geometry.to_mesh(device=device)
    else:
        vertices, faces = geometry.to_mesh()
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.to(device=device, dtype=torch.float32)
    else:
        vertices = torch.as_tensor(vertices, device=device, dtype=torch.float32)
    faces = _to_faces_array(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("geometry.to_mesh() must return vertices with shape (V, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("geometry.to_mesh() must return faces with shape (F, 3).")
    return vertices.contiguous(), np.ascontiguousarray(faces)


def _default_material(material: Material | None) -> Material:
    return material if material is not None else Material()


def _radar_metadata(
    metadata: Mapping[str, Any] | None = None,
    *,
    bsdf: dict[str, Any] | None = None,
    dynamic: bool | None = None,
) -> dict[str, Any]:
    merged = dict(metadata or {})
    if bsdf is not None:
        merged["bsdf"] = dict(bsdf)
    if dynamic is not None:
        merged["dynamic"] = bool(dynamic)
    return merged


def _metadata_value(structure: Structure, key: str, default):
    return structure.metadata.get(key, default)


def _clone_geometry(geometry: GeometryBase, **changes):
    if isinstance(geometry, SMPLBody):
        return geometry.updated(**changes)
    if isinstance(geometry, Mesh):
        params = {
            "vertices": changes.pop("vertices", geometry._vertices_tensor),
            "faces": changes.pop("faces", geometry.faces),
            "position": changes.pop("position", geometry.position),
            "scale": changes.pop("scale", geometry.scale),
            "rotation": changes.pop("rotation", geometry.rotation),
            "recenter": changes.pop("recenter", geometry.recenter),
            "fill_mode": changes.pop("fill_mode", geometry.fill_mode),
            "surface_thickness": changes.pop("surface_thickness", geometry.surface_thickness),
            "source_path": changes.pop("source_path", geometry.source_path),
            "device": changes.pop("device", geometry.position.device),
        }
        if changes:
            unsupported = ", ".join(sorted(changes))
            raise TypeError(f"Unsupported Mesh updates: {unsupported}")
        return Mesh(**params)

    updated = copy.copy(geometry)
    if "position" in changes:
        updated.position = _to_tensor3(changes.pop("position"), device=str(geometry.position.device))
    if "rotation" in changes:
        updated.rotation = torch.as_tensor(changes.pop("rotation"), device=geometry.position.device, dtype=torch.float32)
    if changes:
        unsupported = ", ".join(sorted(changes))
        raise TypeError(f"Unsupported geometry updates for {type(geometry).__name__}: {unsupported}")
    return updated


def _identity_transform(*, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(4, device=device, dtype=dtype)


def _translation_transform(translation: torch.Tensor, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    transform = _identity_transform(device=device, dtype=dtype)
    transform[:3, 3] = translation
    return transform


def _axis_angle_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.clamp(torch.linalg.norm(axis), min=1e-12)
    x, y, z = axis[0], axis[1], axis[2]
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_minus_c = 1.0 - c
    row0 = torch.stack((c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s))
    row1 = torch.stack((y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s))
    row2 = torch.stack((z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c))
    return torch.stack((row0, row1, row2), dim=0)


def _rotation_about_origin_transform(
    origin: torch.Tensor,
    axis: torch.Tensor,
    angle: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    rotation = _axis_angle_rotation(axis.to(device=device, dtype=dtype), angle.to(device=device, dtype=dtype))
    translate_to = _translation_transform(origin.to(device=device, dtype=dtype), device=device, dtype=dtype)
    translate_back = _translation_transform(-origin.to(device=device, dtype=dtype), device=device, dtype=dtype)
    transform = _identity_transform(device=device, dtype=dtype)
    transform[:3, :3] = rotation
    return translate_to @ transform @ translate_back


def _apply_transform_to_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return points @ rotation.transpose(0, 1) + translation


def _apply_transform_to_vectors(vectors: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[:3, :3]
    return vectors @ rotation.transpose(0, 1)


def _geometry_local_to_world_points(
    geometry: GeometryBase,
    points: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    points = points.to(device=device, dtype=dtype)
    rotation = quat_to_rotation_matrix(geometry.rotation.to(device=device, dtype=dtype))
    position = geometry.position.to(device=device, dtype=dtype)
    if isinstance(geometry, Mesh):
        scale = geometry.scale.to(device=device, dtype=dtype)
        points = points * scale
    return points @ rotation.transpose(0, 1) + position


def _geometry_local_to_world_vectors(
    geometry: GeometryBase,
    vectors: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    vectors = vectors.to(device=device, dtype=dtype)
    rotation = quat_to_rotation_matrix(geometry.rotation.to(device=device, dtype=dtype))
    return vectors @ rotation.transpose(0, 1)


@dataclass
class CompiledMesh:
    name: str
    vertices: torch.Tensor
    faces: np.ndarray
    eps_r: float
    bsdf: dict[str, Any] | None = None
    dynamic: bool = False
    joints: torch.Tensor | None = None
    source_kind: str = "mesh"


class SceneModule(torch.nn.Module):
    """Module-style radar scenes that materialize to a standard Scene."""

    def to_scene(self) -> "Scene":
        raise NotImplementedError("SceneModule subclasses must implement to_scene().")


class Scene(SceneBase):
    """Radar scene built from shared core structures."""

    DIRTY_NONE = 0
    DIRTY_VERTICES = 1
    DIRTY_FULL = 2

    def __init__(
        self,
        sensor: Sensor | None = None,
        *,
        structures=None,
        structure_motions: Mapping[str, StructureMotion | Mapping[str, Any]] | None = None,
        metadata=None,
        device: str | None = "cuda",
        verbose: bool = False,
    ):
        resolved_device = _resolve_scene_device(device)
        super().__init__(
            structures=structures,
            metadata=metadata,
            device=resolved_device,
            verbose=verbose,
        )
        self.sensor = sensor or Sensor.identity()
        self._structure_motions: dict[str, StructureMotion] = {}
        self._dirty_level = self.DIRTY_FULL
        self._last_compiled_joints: dict[str, torch.Tensor] = {}
        if structure_motions:
            for name, motion in structure_motions.items():
                self.add_structure_motion(str(name), **dict(StructureMotion.from_value(motion).__dict__))

    @property
    def dirty(self) -> bool:
        return self._dirty_level > self.DIRTY_NONE

    @property
    def dirty_level(self) -> int:
        return self._dirty_level

    @property
    def sensor_origin(self) -> tuple[float, float, float]:
        return self.sensor.origin

    @property
    def sensor_target(self) -> tuple[float, float, float]:
        return self.sensor.target

    @property
    def sensor_up(self) -> tuple[float, float, float]:
        return self.sensor.up

    @property
    def fov(self) -> float:
        return self.sensor.fov

    @property
    def has_motion(self) -> bool:
        return bool(self._structure_motions)

    def mark_clean(self) -> None:
        self._dirty_level = self.DIRTY_NONE

    def _set_dirty(self, level: int) -> None:
        self._dirty_level = max(self._dirty_level, level)

    def set_sensor(
        self,
        sensor: Sensor | None = None,
        *,
        origin=None,
        target=None,
        up=None,
        fov=None,
    ) -> "Scene":
        if sensor is None:
            sensor = Sensor(
                origin=self.sensor.origin if origin is None else origin,
                target=self.sensor.target if target is None else target,
                up=self.sensor.up if up is None else up,
                fov=self.sensor.fov if fov is None else fov,
            )
        self.sensor = sensor
        self._set_dirty(self.DIRTY_FULL)
        return self

    def add_structure(self, structure: Structure) -> "Scene":
        if not isinstance(structure, Structure):
            raise TypeError("Radar Scene structures must be witwin.core.Structure instances.")
        if structure.name is None:
            raise ValueError("Radar Scene structures must define a unique name.")
        if not isinstance(structure.geometry, GeometryBase):
            raise TypeError("Radar Scene structures must wrap a GeometryBase geometry.")
        if any(existing.name == structure.name for existing in self.structures):
            raise ValueError(f"Structure '{structure.name}' already exists.")
        self.structures.append(structure)
        self._set_dirty(self.DIRTY_FULL)
        return self

    def add_mesh(
        self,
        *,
        name: str,
        vertices=None,
        faces=None,
        geometry: GeometryBase | Mesh | None = None,
        material: Material | None = None,
        bsdf: dict[str, Any] | None = None,
        dynamic: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Scene":
        if geometry is None:
            if vertices is None or faces is None:
                raise ValueError("add_mesh requires either geometry or both vertices and faces.")
            geometry = Mesh(vertices, faces, position=(0.0, 0.0, 0.0), recenter=False, device=self.device)
        elif vertices is not None or faces is not None:
            raise ValueError("add_mesh accepts either geometry or vertices/faces, not both.")
        return self.add_structure(
            Structure(
                geometry=geometry,
                material=_default_material(material),
                name=name,
                metadata=_radar_metadata(metadata, bsdf=bsdf, dynamic=True if dynamic else None),
            )
        )

    def add_smpl(
        self,
        *,
        name: str,
        pose,
        shape,
        position=(0.0, 0.0, 0.0),
        gender: str = "male",
        model_root: str | None = None,
        rotation=None,
        material: Material | None = None,
        bsdf: dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Scene":
        return self.add_structure(
            Structure(
                geometry=SMPLBody(
                    pose=pose,
                    shape=shape,
                    position=position,
                    gender=gender,
                    model_root=model_root,
                    rotation=rotation,
                    device=self.device,
                ),
                material=_default_material(material),
                name=name,
                metadata=_radar_metadata(metadata, bsdf=bsdf, dynamic=True),
            )
        )

    def get_structure_motion(self, name: str) -> StructureMotion | None:
        return self._structure_motions.get(name)

    def add_structure_motion(
        self,
        name: str,
        *,
        translation: TranslationMotion | Mapping[str, Any] | None = None,
        rotation: RotationMotion | Mapping[str, Any] | None = None,
        parent: str | None = None,
    ) -> "Scene":
        self._require_structure(name)
        motion = StructureMotion(
            translation=TranslationMotion.from_value(translation),
            rotation=RotationMotion.from_value(rotation),
            parent=parent,
        )
        self._validate_structure_motion(name, motion)
        self._structure_motions[name] = motion
        self._set_dirty(self.DIRTY_VERTICES)
        return self

    def update_structure_motion(
        self,
        name: str,
        *,
        translation: TranslationMotion | Mapping[str, Any] | None | object = _UNSET,
        rotation: RotationMotion | Mapping[str, Any] | None | object = _UNSET,
        parent: str | None | object = _UNSET,
    ) -> "Scene":
        self._require_structure(name)
        existing = self._structure_motions.get(name)
        if existing is None:
            raise KeyError(f"Structure '{name}' does not have motion configured.")
        updated = StructureMotion(
            translation=existing.translation if translation is _UNSET else TranslationMotion.from_value(translation),
            rotation=existing.rotation if rotation is _UNSET else RotationMotion.from_value(rotation),
            parent=existing.parent if parent is _UNSET else parent,
        )
        self._validate_structure_motion(name, updated)
        self._structure_motions[name] = updated
        self._set_dirty(self.DIRTY_VERTICES)
        return self

    def clear_structure_motion(self, name: str) -> "Scene":
        self._require_structure(name)
        if name not in self._structure_motions:
            raise KeyError(f"Structure '{name}' does not have motion configured.")
        self._structure_motions.pop(name, None)
        self._set_dirty(self.DIRTY_VERTICES)
        return self

    def update_structure(self, name: str, **changes) -> "Scene":
        structure_keys = {"geometry", "material", "priority", "enabled", "tags", "metadata", "name"}
        metadata_keys = {"bsdf", "dynamic"}
        for index, structure in enumerate(self.structures):
            if structure.name != name:
                continue

            structure_changes = {key: changes[key] for key in structure_keys & changes.keys()}
            metadata = dict(structure.metadata)
            for key in metadata_keys & changes.keys():
                metadata[key] = changes[key]
            geometry_changes = {
                key: value
                for key, value in changes.items()
                if key not in structure_keys and key not in metadata_keys
            }

            topology_changed = "name" in structure_changes or "geometry" in structure_changes
            updated_geometry = structure_changes.pop("geometry", structure.geometry)
            if geometry_changes:
                updated_geometry = _clone_geometry(updated_geometry, **geometry_changes)
                topology_changed = topology_changed or any(
                    key in geometry_changes for key in ("faces", "gender", "model_root")
                )

            updated = Structure(
                geometry=updated_geometry,
                material=structure_changes.pop("material", structure.material),
                name=structure_changes.pop("name", structure.name),
                priority=structure_changes.pop("priority", structure.priority),
                enabled=structure_changes.pop("enabled", structure.enabled),
                tags=structure_changes.pop("tags", structure.tags),
                metadata=structure_changes.pop("metadata", metadata),
            )
            if structure_changes:
                unsupported = ", ".join(sorted(structure_changes))
                raise TypeError(f"Unsupported Structure updates: {unsupported}")
            self.structures[index] = updated
            if structure.name != updated.name:
                motion = self._structure_motions.pop(structure.name, None)
                if motion is not None:
                    self._structure_motions[updated.name] = motion
                for child_name, child_motion in list(self._structure_motions.items()):
                    if child_motion.parent == structure.name:
                        self._structure_motions[child_name] = StructureMotion(
                            translation=child_motion.translation,
                            rotation=child_motion.rotation,
                            parent=updated.name,
                        )
            self._set_dirty(self.DIRTY_FULL if topology_changed else self.DIRTY_VERTICES)
            return self
        raise KeyError(f"Structure '{name}' not found.")

    def remove(self, name: str) -> "Scene":
        for index, structure in enumerate(self.structures):
            if structure.name == name:
                self.structures.pop(index)
                self._set_dirty(self.DIRTY_FULL)
                self._last_compiled_joints.pop(name, None)
                self._structure_motions.pop(name, None)
                return self
        raise KeyError(f"Structure '{name}' not found.")

    def clone(self, **overrides) -> "Scene":
        return Scene(
            sensor=overrides.get("sensor", self.sensor),
            structures=overrides.get("structures", list(self.structures)),
            structure_motions=overrides.get("structure_motions", dict(self._structure_motions)),
            metadata=overrides.get("metadata", dict(self.metadata)),
            device=overrides.get("device", self.device),
            verbose=overrides.get("verbose", self.verbose),
        )

    def trace(self, renderer=None, *, time: float | None = None):
        if renderer is None:
            from .renderer import Renderer

            renderer = Renderer(self)
        return renderer.trace(time=time)

    def interpolator(self, renderer=None):
        trace = self.trace(renderer=renderer)

        def _interpolator(t):
            if self.has_motion:
                return self.trace(renderer=renderer, time=t)
            del t
            return trace

        return _interpolator

    def get_joints(self, name: str, *, time: float | None = None) -> np.ndarray:
        if time is not None or name not in self._last_compiled_joints:
            self.compile_renderables(time=time)
        if name not in self._last_compiled_joints:
            raise KeyError(f"SMPL body '{name}' not found.")
        return self._last_compiled_joints[name].detach().cpu().numpy()

    def _require_structure(self, name: str) -> Structure:
        for structure in self.structures:
            if structure.name == name:
                return structure
        raise KeyError(f"Structure '{name}' not found.")

    def _validate_structure_motion(self, name: str, motion: StructureMotion) -> None:
        self._require_structure(name)
        if motion.parent is not None:
            if motion.parent == name:
                raise ValueError("Structure motion parent cannot reference itself.")
            self._require_structure(motion.parent)
            current = motion.parent
            visited = {name}
            while current is not None:
                if current in visited:
                    raise ValueError("Structure motion graph must be acyclic.")
                visited.add(current)
                parent_motion = self._structure_motions.get(current)
                current = None if parent_motion is None else parent_motion.parent

    def _resolve_structure_transform(
        self,
        name: str,
        *,
        time: float,
        cache: dict[str, torch.Tensor],
        active: tuple[str, ...],
    ) -> torch.Tensor:
        if name in cache:
            return cache[name]
        if name in active:
            raise ValueError("Structure motion graph must be acyclic.")

        structure = self._require_structure(name)
        parent_transform = _identity_transform(device=self.device, dtype=torch.float32)
        motion = self._structure_motions.get(name)
        if motion is not None and motion.parent is not None:
            parent_transform = self._resolve_structure_transform(
                motion.parent,
                time=time,
                cache=cache,
                active=active + (name,),
            )

        if motion is None:
            cache[name] = parent_transform
            return parent_transform

        self_transform = self._build_structure_motion_transform(
            structure.geometry,
            motion,
            time=time,
            parent_transform=parent_transform,
        )
        total_transform = self_transform @ parent_transform
        cache[name] = total_transform
        return total_transform

    def _build_structure_motion_transform(
        self,
        geometry: GeometryBase,
        motion: StructureMotion,
        *,
        time: float,
        parent_transform: torch.Tensor,
    ) -> torch.Tensor:
        dtype = torch.float32
        device = self.device
        transform = _identity_transform(device=device, dtype=dtype)
        translation_delta = torch.zeros(3, device=device, dtype=dtype)

        if motion.translation is not None:
            translation_delta = self._resolve_translation_delta(
                geometry,
                motion.translation,
                time=time,
                parent_transform=parent_transform,
            )
            transform = _translation_transform(translation_delta, device=device, dtype=dtype)

        if motion.rotation is None:
            return transform

        rotation_transform = self._resolve_rotation_transform(
            geometry,
            motion.rotation,
            translation_delta=translation_delta,
            parent_transform=parent_transform,
            device=device,
            dtype=dtype,
            time=time,
        )
        return rotation_transform @ transform

    def _resolve_translation_delta(
        self,
        geometry: GeometryBase,
        translation: TranslationMotion,
        *,
        time: float,
        parent_transform: torch.Tensor,
    ) -> torch.Tensor:
        dtype = torch.float32
        offset = tensor_vec3(translation.offset, device=self.device, dtype=dtype)
        velocity = tensor_vec3(translation.velocity, device=self.device, dtype=dtype)
        t_ref = tensor_scalar(translation.t_ref, device=self.device, dtype=dtype)
        delta = offset + velocity * (torch.tensor(time, device=self.device, dtype=dtype) - t_ref)
        if translation.space == "world":
            return delta
        world_delta = _geometry_local_to_world_vectors(geometry, delta.unsqueeze(0), device=self.device, dtype=dtype)[0]
        return _apply_transform_to_vectors(world_delta.unsqueeze(0), parent_transform)[0]

    def _resolve_rotation_transform(
        self,
        geometry: GeometryBase,
        rotation: RotationMotion,
        *,
        translation_delta: torch.Tensor,
        parent_transform: torch.Tensor,
        device: str,
        dtype: torch.dtype,
        time: float,
    ) -> torch.Tensor:
        axis = tensor_vec3(rotation.axis, device=device, dtype=dtype)
        if rotation.space == "local":
            axis_world = _geometry_local_to_world_vectors(geometry, axis.unsqueeze(0), device=device, dtype=dtype)[0]
            origin_local = torch.zeros(1, 3, device=device, dtype=dtype)
            if rotation.origin is not None:
                origin_local = tensor_vec3(rotation.origin, device=device, dtype=dtype).view(1, 3)
            origin_world = _geometry_local_to_world_points(geometry, origin_local, device=device, dtype=dtype)[0]
            axis_world = _apply_transform_to_vectors(axis_world.unsqueeze(0), parent_transform)[0]
            origin_world = _apply_transform_to_points(origin_world.unsqueeze(0), parent_transform)[0]
        else:
            axis_world = axis
            if rotation.origin is None:
                origin_world = geometry.position.to(device=device, dtype=dtype)
            else:
                origin_world = tensor_vec3(rotation.origin, device=device, dtype=dtype)

        origin_world = origin_world + translation_delta
        angle = tensor_scalar(rotation.angle, device=device, dtype=dtype)
        angular_velocity = tensor_scalar(rotation.angular_velocity, device=device, dtype=dtype)
        t_ref = tensor_scalar(rotation.t_ref, device=device, dtype=dtype)
        theta = angle + angular_velocity * (torch.tensor(time, device=device, dtype=dtype) - t_ref)
        return _rotation_about_origin_transform(origin_world, axis_world, theta, device=device, dtype=dtype)

    def compile_renderables(self, *, time: float | None = None) -> dict[str, CompiledMesh]:
        renderables: dict[str, CompiledMesh] = {}
        joints: dict[str, torch.Tensor] = {}
        transforms: dict[str, torch.Tensor] = {}
        time_value = 0.0 if time is None else float(time)
        for structure in self.structures:
            if not structure.enabled:
                continue
            motion_transform = self._resolve_structure_transform(
                structure.name,
                time=time_value,
                cache=transforms,
                active=(),
            )
            compiled = self._compile_structure(structure, motion_transform=motion_transform)
            renderables[compiled.name] = compiled
            if compiled.joints is not None:
                joints[compiled.name] = compiled.joints
        self._last_compiled_joints = joints
        return renderables

    def _compile_structure(self, structure: Structure, *, motion_transform: torch.Tensor) -> CompiledMesh:
        geometry = structure.geometry
        joints = None
        source_kind = "mesh"
        if isinstance(geometry, SMPLBody):
            vertices, faces, joints = geometry._evaluate(device=self.device)
            source_kind = "smpl"
        else:
            vertices, faces = _materialize_geometry(geometry, device=self.device)
        vertex_tensor = _to_vertex_tensor(vertices, device=self.device).contiguous()
        vertex_tensor = _apply_transform_to_points(vertex_tensor, motion_transform).contiguous()
        if joints is not None:
            joints = _apply_transform_to_points(
                _to_vertex_tensor(joints, device=self.device).contiguous(),
                motion_transform,
            ).contiguous()
        dynamic = bool(
            _metadata_value(structure, "dynamic", False)
            or vertex_tensor.requires_grad
            or joints is not None
            or structure.name in self._structure_motions
        )
        eps_r = float(structure.material.evaluate_static().eps_r)
        return CompiledMesh(
            name=structure.name,
            vertices=vertex_tensor,
            faces=_to_faces_array(faces),
            eps_r=eps_r,
            bsdf=_metadata_value(structure, "bsdf", None),
            dynamic=dynamic,
            joints=joints,
            source_kind=source_kind,
        )
