"""A small synthetic linear-blend-skinning rig in the SMPL layer's shape.

No SMPL model pickles are distributed with this repository, so the pose and
shape leaves are tested on a rig that has every table ``_fast_smpl_forward``
reads - 24 joints in a kinematic tree, a template, shape and pose blend shapes,
a joint regressor and skinning weights - over a small mesh. Installed into the
layer cache under its own ``(gender, model_root)`` key, it runs through
``SMPLBody``, ``SmplPoseDeformation`` and ``mesh_target`` exactly as a real
SMPL layer does; only the numbers differ.

The mesh is a strip in the body's ``y``-``z`` plane, one column of quads per
joint along ``z``, so a pose rotation bends the strip at that joint. Joint
``j``'s parent is ``j - 1`` (a chain), and each vertex is skinned to the two
joints nearest it along the strip.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

JOINTS = 24
GENDER = "synthetic"
MODEL_ROOT = "<synthetic linear-blend-skinning rig>"


def _orient_toward_minus_x(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flip = normals[:, 0] > 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def build_rig(*, width_m: float = 0.06, segment_m: float = 0.01, seed: int = 0) -> SimpleNamespace:
    """The layer tables on the CPU, float32, in ``SMPL_Layer``'s attribute names."""

    rng = np.random.default_rng(seed)
    rows = JOINTS + 1
    z = np.arange(rows) * segment_m
    y = np.array([-0.5 * width_m, 0.0, 0.5 * width_m])
    grid_z, grid_y = np.meshgrid(z, y, indexing="ij")
    # A slight lean so no facet sits exactly at the specular point of a broadside radar.
    template = np.stack((0.15 * grid_z.ravel(), grid_y.ravel(), grid_z.ravel()), axis=1)
    columns = len(y)
    faces = []
    for i in range(rows - 1):
        for j in range(columns - 1):
            a = i * columns + j
            faces += [(a, a + 1, a + columns), (a + 1, a + columns + 1, a + columns)]
    faces = _orient_toward_minus_x(template, np.asarray(faces, dtype=np.int64))

    count = template.shape[0]
    joint_z = (np.arange(JOINTS) + 0.5) * segment_m
    weights = np.zeros((count, JOINTS))
    for vertex in range(count):
        distance = np.abs(joint_z - template[vertex, 2])
        nearest = np.argsort(distance)[:2]
        share = 1.0 / (distance[nearest] + 1e-3)
        weights[vertex, nearest] = share / share.sum()
    # Each joint is the mean of the vertices of its row and the next one.
    regressor = np.zeros((JOINTS, count))
    for joint in range(JOINTS):
        members = [joint * columns + j for j in range(columns)] + [(joint + 1) * columns + j for j in range(columns)]
        regressor[joint, members] = 1.0 / len(members)
    shapedirs = 2e-3 * rng.standard_normal((count, 3, 10))
    posedirs = 1e-3 * rng.standard_normal((count, 3, 9 * (JOINTS - 1)))
    parents = [-1] + list(range(JOINTS - 1))

    def single(value):
        return torch.as_tensor(value, dtype=torch.float32)

    return SimpleNamespace(
        th_v_template=single(template)[None],
        th_shapedirs=single(shapedirs),
        th_posedirs=single(posedirs),
        th_J_regressor=single(regressor),
        th_weights=single(weights),
        th_faces=torch.as_tensor(faces, dtype=torch.int64),
        kintree_parents=parents,
    )


def install(device: str) -> tuple[str, str]:
    """Put the rig in the SMPL layer cache for ``device``; returns ``(gender, model_root)`` for ``SMPLBody``."""

    from witwin.radar import smpl

    # The body resolves its device from a tensor, so the key is the indexed name ("cuda:0").
    device = str(torch.zeros(0, device=device).device)
    rig = build_rig()
    layer = SimpleNamespace(
        **{name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in vars(rig).items()}
    )
    layer._radar_parents = torch.tensor(layer.kintree_parents[1:], dtype=torch.int64, device=device)
    smpl._SMPL_LAYER_CACHE[(GENDER, MODEL_ROOT, str(device))] = layer
    return GENDER, MODEL_ROOT


__all__ = ["GENDER", "JOINTS", "MODEL_ROOT", "build_rig", "install"]
