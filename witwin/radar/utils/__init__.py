"""Internal helper utilities shared across the radar package.

The antenna-pattern helpers used to live here and no longer do. They are a
SENSOR property, and Phase 6 gave sensors one owner: they are now
``witwin.radar.sensors.pattern``. Keeping a re-export here would defeat the
point of the move, which was to make it obvious where a pattern lookup is
supposed to come from.

``geometry.py`` sat beside ``tensor.py`` and ``vector.py`` until Phase 11 and is
gone for a different reason. Its eight rigid-transform helpers existed to serve
the radar-owned logical ``Scene``, whose motion graph composed them; that Scene
is deleted and Core's ``RigidMotion`` / ``Trajectory`` / ``DynamicScene`` own
world motion now. They are not re-implemented here.
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
]
