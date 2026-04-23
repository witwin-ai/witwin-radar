"""Structure motion specifications for time-varying radar scenes."""

from __future__ import annotations

import torch

from .utils.vector import optional_vec3_tensor, scalar_tensor, vec3_tensor


_ALLOWED_SPACES = {"local", "world"}


def _normalize_space(value, *, name: str, default: str) -> str:
    if value is None:
        return default
    space = str(value).lower()
    if space not in _ALLOWED_SPACES:
        raise ValueError(f"{name} must be 'local' or 'world'.")
    return space


def tensor_vec3(value, *, device, dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.shape != (3,):
        raise ValueError("motion vector must have shape (3,).")
    return tensor


def tensor_scalar(value, *, device, dtype) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype).reshape(())


class TranslationMotion:
    """Linear motion described by offset + velocity in a reference frame."""

    def __init__(
        self,
        offset=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        space: str = "world",
        t_ref=0.0,
    ) -> None:
        self.offset: torch.Tensor = vec3_tensor(offset, name="TranslationMotion.offset")
        self.velocity: torch.Tensor = vec3_tensor(velocity, name="TranslationMotion.velocity")
        self.space: str = _normalize_space(space, name="TranslationMotion.space", default="world")
        self.t_ref: torch.Tensor = scalar_tensor(t_ref, name="TranslationMotion.t_ref")


class RotationMotion:
    """Rotation described by axis + angular velocity (+ optional initial angle, pivot)."""

    def __init__(
        self,
        axis,
        angular_velocity,
        angle=0.0,
        origin=None,
        space: str = "local",
        t_ref=0.0,
    ) -> None:
        axis_t = vec3_tensor(axis, name="RotationMotion.axis")
        if torch.linalg.norm(axis_t) <= 1e-12:
            raise ValueError("RotationMotion.axis must be non-zero.")
        self.axis: torch.Tensor = axis_t
        self.angular_velocity: torch.Tensor = scalar_tensor(
            angular_velocity, name="RotationMotion.angular_velocity"
        )
        self.angle: torch.Tensor = scalar_tensor(angle, name="RotationMotion.angle")
        self.origin: torch.Tensor | None = optional_vec3_tensor(origin, name="RotationMotion.origin")
        self.space: str = _normalize_space(space, name="RotationMotion.space", default="local")
        self.t_ref: torch.Tensor = scalar_tensor(t_ref, name="RotationMotion.t_ref")


class StructureMotion:
    """Composed structure motion: translation, rotation, and/or parent linkage."""

    def __init__(
        self,
        translation: TranslationMotion | None = None,
        rotation: RotationMotion | None = None,
        parent: str | None = None,
    ) -> None:
        if parent is not None:
            parent_str = str(parent)
            if not parent_str:
                raise ValueError("StructureMotion.parent must be a non-empty string.")
            parent = parent_str
        if translation is None and rotation is None and parent is None:
            raise ValueError("StructureMotion requires translation, rotation, or parent.")
        self.translation = translation
        self.rotation = rotation
        self.parent = parent
