"""Structure motion specifications for time-varying radar scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .utils.vector import coerce_optional_vec3, coerce_scalar, coerce_vec3, vector_norm


def _normalize_space(value, *, name: str, default: str) -> str:
    if value is None:
        return default
    space = str(value).lower()
    if space not in {"local", "world"}:
        raise ValueError(f"{name} must be 'local' or 'world'.")
    return space


def tensor_vec3(value, *, device, dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.shape != (3,):
        raise ValueError("motion vector must have shape (3,).")
    return tensor


def tensor_scalar(value, *, device, dtype) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype).reshape(())


@dataclass(frozen=True)
class TranslationMotion:
    offset: Any = (0.0, 0.0, 0.0)
    velocity: Any = (0.0, 0.0, 0.0)
    space: str = "world"
    t_ref: Any = 0.0

    def __post_init__(self):
        object.__setattr__(self, "offset", coerce_vec3(self.offset, name="TranslationMotion.offset"))
        object.__setattr__(self, "velocity", coerce_vec3(self.velocity, name="TranslationMotion.velocity"))
        object.__setattr__(
            self,
            "space",
            _normalize_space(self.space, name="TranslationMotion.space", default="world"),
        )
        object.__setattr__(self, "t_ref", coerce_scalar(self.t_ref, name="TranslationMotion.t_ref"))

    @classmethod
    def from_value(cls, value) -> "TranslationMotion | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("translation must be a TranslationMotion or mapping.")
        allowed = {"offset", "velocity", "space", "t_ref"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise TypeError(f"Unsupported translation motion keys: {', '.join(unknown)}")
        return cls(
            offset=value.get("offset", (0.0, 0.0, 0.0)),
            velocity=value.get("velocity", (0.0, 0.0, 0.0)),
            space=value.get("space", "world"),
            t_ref=value.get("t_ref", 0.0),
        )


@dataclass(frozen=True)
class RotationMotion:
    axis: Any
    angular_velocity: Any
    angle: Any = 0.0
    origin: Any | None = None
    space: str = "local"
    t_ref: Any = 0.0

    def __post_init__(self):
        object.__setattr__(self, "axis", coerce_vec3(self.axis, name="RotationMotion.axis"))
        if vector_norm(self.axis) <= 1e-12:
            raise ValueError("RotationMotion.axis must be non-zero.")
        object.__setattr__(
            self,
            "angular_velocity",
            coerce_scalar(self.angular_velocity, name="RotationMotion.angular_velocity"),
        )
        object.__setattr__(self, "angle", coerce_scalar(self.angle, name="RotationMotion.angle"))
        object.__setattr__(self, "origin", coerce_optional_vec3(self.origin, name="RotationMotion.origin"))
        object.__setattr__(
            self,
            "space",
            _normalize_space(self.space, name="RotationMotion.space", default="local"),
        )
        object.__setattr__(self, "t_ref", coerce_scalar(self.t_ref, name="RotationMotion.t_ref"))

    @classmethod
    def from_value(cls, value) -> "RotationMotion | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("rotation must be a RotationMotion or mapping.")
        allowed = {"axis", "angular_velocity", "angle", "origin", "space", "t_ref"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise TypeError(f"Unsupported rotation motion keys: {', '.join(unknown)}")
        if "axis" not in value:
            raise ValueError("rotation motion requires axis.")
        if "angular_velocity" not in value:
            raise ValueError("rotation motion requires angular_velocity.")
        return cls(
            axis=value["axis"],
            angular_velocity=value["angular_velocity"],
            angle=value.get("angle", 0.0),
            origin=value.get("origin"),
            space=value.get("space", "local"),
            t_ref=value.get("t_ref", 0.0),
        )


@dataclass(frozen=True)
class StructureMotion:
    translation: TranslationMotion | None = None
    rotation: RotationMotion | None = None
    parent: str | None = None

    def __post_init__(self):
        if self.parent is not None:
            parent = str(self.parent)
            if not parent:
                raise ValueError("StructureMotion.parent must be a non-empty string.")
            object.__setattr__(self, "parent", parent)
        if self.translation is None and self.rotation is None and self.parent is None:
            raise ValueError("StructureMotion requires translation, rotation, or parent.")

    @classmethod
    def from_value(cls, value) -> "StructureMotion":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("structure motion must be a StructureMotion or mapping.")
        allowed = {"translation", "rotation", "parent"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise TypeError(f"Unsupported structure motion keys: {', '.join(unknown)}")
        return cls(
            translation=TranslationMotion.from_value(value.get("translation")),
            rotation=RotationMotion.from_value(value.get("rotation")),
            parent=value.get("parent"),
        )
