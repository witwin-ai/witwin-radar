"""Internal helper utilities shared across the radar package."""

from .tensor import (
    real_dtype,
    resolve_scene_device,
    to_faces_array,
    to_tensor3,
    to_vertex_tensor,
)
from .vector import (
    coerce_optional_vec3,
    coerce_scalar,
    coerce_vec3,
    cross3,
    norm3,
    normalize_rows,
    sub3,
    vec3_tuple,
    vector_norm,
)
from .geometry import (
    apply_transform_to_points,
    apply_transform_to_vectors,
    axis_angle_rotation,
    geometry_local_to_world_points,
    geometry_local_to_world_vectors,
    identity_transform,
    rotation_about_origin_transform,
    translation_transform,
)

__all__ = [
    "real_dtype",
    "resolve_scene_device",
    "to_faces_array",
    "to_tensor3",
    "to_vertex_tensor",
    "coerce_optional_vec3",
    "coerce_scalar",
    "coerce_vec3",
    "cross3",
    "norm3",
    "normalize_rows",
    "sub3",
    "vec3_tuple",
    "vector_norm",
    "apply_transform_to_points",
    "apply_transform_to_vectors",
    "axis_angle_rotation",
    "geometry_local_to_world_points",
    "geometry_local_to_world_vectors",
    "identity_transform",
    "rotation_about_origin_transform",
    "translation_transform",
]
