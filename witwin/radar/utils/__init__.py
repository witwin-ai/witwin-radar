"""Internal helper utilities shared across the radar package.

The antenna-pattern helpers used to live here and no longer do. They are a
SENSOR property, and Phase 6 gave sensors one owner: they are now
``witwin.radar.sensors.pattern``. Keeping a re-export here would defeat the
point of the move, which was to make it obvious where a pattern lookup is
supposed to come from.
"""

from .tensor import (
    real_dtype,
    resolve_scene_device,
    to_faces_array,
    to_tensor3,
    to_vertex_tensor,
)
from .vector import (
    normalize_rows,
    optional_vec3_tensor,
    scalar_tensor,
    vec3_tensor,
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
    "normalize_rows",
    "optional_vec3_tensor",
    "scalar_tensor",
    "vec3_tensor",
    "apply_transform_to_points",
    "apply_transform_to_vectors",
    "axis_angle_rotation",
    "geometry_local_to_world_points",
    "geometry_local_to_world_vectors",
    "identity_transform",
    "rotation_about_origin_transform",
    "translation_transform",
]
