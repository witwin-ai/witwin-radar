"""Worlds, radars and target recipes for the analytic oracle and survival suites.

Everything here is built through the public surfaces a user has: Core's
``Scene``/``Mesh``/``Structure``/``PhysicalMaterial`` and Radar's ``Radar``,
``Fmcw``/``Ofdm``/``Pulsed`` and ``PointTargets``. No expected value lives
here; those are in ``tests/reference/radar_oracles.py``.

Pose convention shared by every oracle: the radar sits at the world origin and
looks along world +x with world +y up, so the local frame is right = +z,
up = +y, boresight = +x, and ``polarization="up"`` is world +y. The plane
normal to that polarization is the world x-z plane.
"""

from __future__ import annotations

import math

import numpy as np
import torch

CARRIER_HZ = 77e9
LOOK_AT = (1.0, 0.0, 0.0)
UP = (0.0, 1.0, 0.0)


def empty_scene():
    from witwin.core import Scene

    return Scene(structures=(), endpoints=[])


def beat_fmcw(*, samples: int = 16, chirps: int = 1, sample_rate: float = 5e6, slope: float = 30e12):
    """A small FMCW ramp in the beat domain, idle 5 us, ramp just covering the ADC window."""

    from witwin.radar import Fmcw

    ramp = samples / sample_rate + 1e-6
    return Fmcw(
        slope=slope,
        sample_rate=sample_rate,
        samples_per_chirp=samples,
        chirps_per_frame=chirps,
        adc_start=0.0,
        idle=5e-6,
        ramp_end=ramp,
        output="beat",
    )


def radar(waveform=None, *, tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0),), power_dbm=10.0, **fields):
    """A metre-authored radar at the origin, looking along +x."""

    from witwin.radar import Radar

    return Radar(
        carrier=CARRIER_HZ,
        waveform=beat_fmcw() if waveform is None else waveform,
        tx=tx,
        rx=rx,
        power=power_dbm,
        look_at=LOOK_AT,
        up=UP,
        **fields,
    )


def plane_scene(*, corners, material, faces=((0, 1, 2), (0, 2, 3)), structure_id: int = 1):
    """One flat mesh, authored in world coordinates and never recentred."""

    from witwin.core import Mesh, Scene, Structure

    mesh = Mesh(
        vertices=torch.tensor(corners, dtype=torch.float32),
        faces=torch.tensor(faces, dtype=torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    wall = Structure(
        geometry=mesh,
        material=material,
        structure_id=structure_id,
        material_id=structure_id,
        assignment_id=structure_id,
        surface_id=structure_id,
    )
    return Scene(structures=(wall,), endpoints=[])


def side_wall_corners(*, z_m: float, x_range=(-5.0, 80.0), y_half: float = 20.0):
    """A rectangle in the plane ``z = z_m``: normal along the array axis, so
    the mirror image moves the echo in azimuth and the ``"up"`` polarization
    is perpendicular to every plane of incidence (pure TE)."""

    x0, x1 = x_range
    return ((x0, -y_half, z_m), (x1, -y_half, z_m), (x1, y_half, z_m), (x0, y_half, z_m))


# ---------------------------------------------------------------------------
# The sampled-surface recipe a user can write today
# ---------------------------------------------------------------------------
#
# N isotropic point sites spread over the illuminated surface, each carrying
# ``sigma_total / N``. This is the recipe the SMPL benchmark uses; the oracle
# asks whether it converges to the analytic cross section of the surface.


def plate_sites(n: int, *, side_m: float, range_m: float) -> np.ndarray:
    """``n x n`` cell centres of a square plate in ``x = range_m`` facing the radar."""

    step = side_m / n
    centres = -0.5 * side_m + (np.arange(n) + 0.5) * step
    y, z = np.meshgrid(centres, centres, indexing="ij")
    return np.stack((np.full(y.size, range_m), y.ravel(), z.ravel()), axis=1)


def sphere_sites(n: int, *, radius_m: float, range_m: float) -> np.ndarray:
    """``n`` equal-area points on the illuminated hemisphere, front pole at ``range_m``.

    A Fibonacci lattice in ``cos`` of the angle from the radar direction, which
    is equal-area on the hemisphere.
    """

    k = np.arange(n)
    cos_polar = 1.0 - (k + 0.5) / n
    sin_polar = np.sqrt(1.0 - cos_polar**2)
    azimuth = k * math.pi * (3.0 - math.sqrt(5.0))
    centre = range_m + radius_m
    return np.stack(
        (centre - radius_m * cos_polar, radius_m * sin_polar * np.cos(azimuth), radius_m * sin_polar * np.sin(azimuth)),
        axis=1,
    )


def trihedral_sites(n: int, *, edge_m: float, range_m: float) -> np.ndarray:
    """Centroids of a uniform triangulation of the three faces of a triangular trihedral.

    Each face is the right triangle spanned by two of the three orthogonal
    edges, split into ``n^2`` congruent triangles. The corner is placed at
    ``x = range_m`` with its symmetry axis ``(1, 1, 1)/sqrt(3)`` pointing back
    at the radar, which is the peak-RCS aspect.
    """

    upward = [((i + 1.0 / 3.0), (j + 1.0 / 3.0)) for i in range(n) for j in range(n - i)]
    downward = [((i + 2.0 / 3.0), (j + 2.0 / 3.0)) for i in range(n) for j in range(n - i - 1)]
    local = np.asarray(upward + downward, dtype=np.float64) * (edge_m / n)
    points = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        face = np.zeros((local.shape[0], 3))
        face[:, a] = local[:, 0]
        face[:, b] = local[:, 1]
        points.append(face)
    corner_frame = np.concatenate(points, axis=0)
    return corner_frame @ _rotation_taking(np.ones(3) / math.sqrt(3.0), np.array([-1.0, 0.0, 0.0])).T + np.array(
        [range_m, 0.0, 0.0]
    )


# ---------------------------------------------------------------------------
# Meshes for the facet route
# ---------------------------------------------------------------------------
#
# The same canonical shapes as triangle meshes, each face wound so that its
# right-handed normal points to the side the radar illuminates (outward on the
# sphere, into the open side of the corner reflectors).


def _orient(vertices: np.ndarray, faces: np.ndarray, toward) -> np.ndarray:
    """Flip every face whose normal disagrees with ``toward(centroids)``."""

    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flip = np.einsum("ij,ij->i", normals, toward(corners.mean(axis=1))) < 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def _grid_faces(n: int, base: int = 0) -> list[tuple[int, int, int]]:
    faces = []
    for i in range(n):
        for j in range(n):
            a = base + i * (n + 1) + j
            faces += [(a, a + 1, a + n + 1), (a + 1, a + n + 2, a + n + 1)]
    return faces


def plate_mesh(n: int, *, side_m: float, range_m: float) -> tuple[np.ndarray, np.ndarray]:
    """A square plate in ``x = range_m`` split into ``2 n^2`` triangles, facing the radar."""

    grid = np.linspace(-0.5 * side_m, 0.5 * side_m, n + 1)
    y, z = np.meshgrid(grid, grid, indexing="ij")
    vertices = np.stack((np.full(y.size, range_m), y.ravel(), z.ravel()), axis=1)
    faces = np.asarray(_grid_faces(n))
    return vertices, _orient(vertices, faces, lambda c: np.tile([-1.0, 0.0, 0.0], (len(c), 1)))


def sphere_mesh(n: int, *, radius_m: float, range_m: float) -> tuple[np.ndarray, np.ndarray]:
    """A UV sphere of ``n`` latitude bands and ``2 n`` meridians, front pole at ``range_m``.

    The poles sit on the world z axis, so the radar looks at the equator band.
    """

    centre = np.array([range_m + radius_m, 0.0, 0.0])
    vertices = [[0.0, 0.0, radius_m], [0.0, 0.0, -radius_m]]
    for band in range(1, n):
        polar = math.pi * band / n
        for meridian in range(2 * n):
            azimuth = math.pi * meridian / n
            vertices.append(
                [
                    radius_m * math.sin(polar) * math.cos(azimuth),
                    radius_m * math.sin(polar) * math.sin(azimuth),
                    radius_m * math.cos(polar),
                ]
            )

    def index(band: int, meridian: int) -> int:
        return 2 + (band - 1) * 2 * n + meridian % (2 * n)

    faces = []
    for meridian in range(2 * n):
        faces.append((0, index(1, meridian), index(1, meridian + 1)))
        faces.append((1, index(n - 1, meridian + 1), index(n - 1, meridian)))
    for band in range(1, n - 1):
        for meridian in range(2 * n):
            a, b = index(band, meridian), index(band, meridian + 1)
            c, d = index(band + 1, meridian), index(band + 1, meridian + 1)
            faces += [(a, c, b), (b, c, d)]
    local = np.asarray(vertices)
    faces = _orient(local, np.asarray(faces), lambda c: c)
    return local + centre, faces


def trihedral_mesh(n: int, *, edge_m: float, range_m: float) -> tuple[np.ndarray, np.ndarray]:
    """The triangular trihedral of :func:`trihedral_sites` as ``3 n^2`` triangles."""

    index: dict[tuple, int] = {}
    vertices: list[np.ndarray] = []

    def vertex(point: np.ndarray) -> int:
        key = tuple(np.round(point, 12))
        if key not in index:
            index[key] = len(vertices)
            vertices.append(point)
        return index[key]

    step = edge_m / n
    faces = []
    for a, b in ((0, 1), (1, 2), (2, 0)):

        def point(i: int, j: int, a: int = a, b: int = b) -> np.ndarray:
            value = np.zeros(3)
            value[a], value[b] = i * step, j * step
            return value

        for i in range(n):
            for j in range(n - i):
                faces.append((vertex(point(i, j)), vertex(point(i + 1, j)), vertex(point(i, j + 1))))
                if j < n - i - 1:
                    faces.append((vertex(point(i + 1, j)), vertex(point(i + 1, j + 1)), vertex(point(i, j + 1))))
    local = np.asarray(vertices)
    faces = _orient(local, np.asarray(faces), lambda c: np.tile([1.0, 1.0, 1.0], (len(c), 1)))
    rotation = _rotation_taking(np.ones(3) / math.sqrt(3.0), np.array([-1.0, 0.0, 0.0]))
    return local @ rotation.T + np.array([range_m, 0.0, 0.0]), faces


def dihedral_mesh(n: int, *, edge_m: float, face_m: float, range_m: float) -> tuple[np.ndarray, np.ndarray]:
    """A right-angle dihedral, edge along world y at ``x = range_m``, opening toward the radar.

    Each face is ``edge_m`` along the edge by ``face_m`` across, at 45 degrees
    to the boresight, split into ``2 n^2`` triangles.
    """

    vertices, faces = [], []
    for sign in (1.0, -1.0):
        across = np.array([-1.0, 0.0, sign]) / math.sqrt(2.0)
        base = len(vertices)
        for i in range(n + 1):
            for j in range(n + 1):
                vertices.append(np.array([range_m, edge_m * (j / n - 0.5), 0.0]) + across * (face_m * i / n))
        faces += _grid_faces(n, base)
    vertices = np.asarray(vertices)
    return vertices, _orient(vertices, np.asarray(faces), lambda c: np.tile([-1.0, 0.0, 0.0], (len(c), 1)))


def mesh_scene(vertices: np.ndarray, faces: np.ndarray, *, material, structure_id: int = 1, extra=()):
    """One world-authored mesh structure, plus any ``extra`` structures."""

    from witwin.core import Mesh, Scene, Structure

    mesh = Mesh(
        vertices=torch.tensor(vertices, dtype=torch.float32),
        faces=torch.tensor(faces, dtype=torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    target = Structure(
        geometry=mesh,
        material=material,
        structure_id=structure_id,
        material_id=structure_id,
        assignment_id=structure_id,
        surface_id=structure_id,
    )
    return Scene(structures=(target, *extra), endpoints=[])


def _rotation_taking(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The rotation matrix taking unit ``source`` onto unit ``target`` (Rodrigues)."""

    axis = np.cross(source, target)
    sine = np.linalg.norm(axis)
    cosine = float(np.dot(source, target))
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


__all__ = [
    "CARRIER_HZ",
    "LOOK_AT",
    "UP",
    "beat_fmcw",
    "dihedral_mesh",
    "empty_scene",
    "mesh_scene",
    "plane_scene",
    "plate_mesh",
    "plate_sites",
    "radar",
    "side_wall_corners",
    "sphere_mesh",
    "sphere_sites",
    "trihedral_mesh",
    "trihedral_sites",
]
