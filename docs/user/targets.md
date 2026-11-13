# Targets: what the radar is looking at

Every world verb takes a target record as its second, required argument. It
says where the scatterers are and how strongly they scatter; there is no
default target set, because any default would be an unchosen statement about
the world. Three records exist:

| Record | Scatterers | Strength |
|---|---|---|
| `PointTargets` | world positions you name, optionally moving | declared: `rcs` (m^2) or `amplitude`, optionally aspect-dependent, polarimetric, tabulated or fluctuating |
| `StructureTargets` | one point at the world anchor of each moving Core structure | declared, as above (no table) |
| `MeshTargets` | every facet of selected Core structures | computed: physical optics from each facet's area, orientation and material |

The carrier and the device are not target fields; the radar supplies them.

## Setup

The blocks below read the composed round trips directly with
`radar.trace(...).rows()`: one row per path, with its complex transfer
`complex_transfer_ref` (sqrt(W), Channel's phasor) and its delay.

```python
import dataclasses
import math

import numpy as np
import torch
from witwin.core import MaterialLayer, Mesh, PhysicalMaterial, Scene, Structure, SurfaceRoughness
from witwin.core.dynamics import DynamicScene, LinearTrajectory
from witwin.radar import (Aspect, Fluctuation, Fmcw, MeshTargets, PointTargets, Radar, RcsTable, ScatteringMatrix,
                          StructureTargets)

chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=64, chirps_per_frame=8,
             adc_start=1e-6, idle=4e-6, ramp_end=16e-6)
radar = Radar(reference_frequency_hz=77e9, waveform=chirp, power=15.0, tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0),))


def transfers(targets, scene=None, reflections=0, radar=radar):
    """The composed rows' complex transfers, complex128, in row order."""
    paths = radar.trace(Scene() if scene is None else scene, targets, times=(0.0,), reflections=reflections)
    return paths.rows().complex_transfer_ref.to(torch.complex128)


def db(ratio):
    return 20.0 * math.log10(abs(ratio))
```

## `PointTargets`

```python
pair = PointTargets(positions=[(-0.5, 0.0, -3.0), (0.5, 0.0, -3.0)], rcs=[1.0, 0.25], phase=[0.0, 1.0],
                    ids=(20, 10))
rows = transfers(pair)
print(f"amplitude ratio {abs(rows[1] / rows[0]):.3f}, phase difference {float(torch.angle(rows[1] / rows[0])):+.3f} rad")
assert abs(abs(rows[1] / rows[0]) - 2.0) < 1e-4  # rows are in ascending ID order: id 10, then id 20
```

- `positions` is `(S, 3)` world metres, a sequence of triples or a tensor. A
  tensor passes through untouched, so a `requires_grad` leaf keeps its tape. A
  float64 tensor keeps float64 path lengths ([Conventions](conventions.md#float64-positions)).
- `rcs` (m^2) or `amplitude` (the dimensionless `sqrt(4 pi rcs) / wavelength`):
  exactly one of them, one value for every target or one per target in
  `positions` order. `phase` (rad) is the scattering phase, per target or
  shared. A 0-dim tensor `rcs` is a differentiable leaf.
- `trajectory(t)` returns the same ordered points at time `t` as an `(S, 3)`
  tensor; without it the targets are static (zero Doppler). Rotation and
  articulation must move the points; a velocity label alone moves nothing.
- `ids` fixes each target's stable world ID (distinct non-negative ints);
  otherwise they are allocated. Composed rows, labels and events are ordered
  by ascending ID, which is why the two rows above come back as id 10, id 20.

Without further options each target is an isotropic, polarization-preserving
point whose echo follows the radar equation in every direction.

### `Aspect`: a lobe instead of a sphere

`Aspect(axis, exponent, coherent_interval, phase_rate=0.0)` scales a target's
strength by `max(-dir_in . axis, 0)^n * max(dir_out . axis, 0)^n`, with
`dir_in`/`dir_out` the propagation directions into and out of the target:
the return is largest when `axis` points back along the incoming wave.
`axis` is `(S, 3)` unit vectors, or a function of time (a rotating lobe, see
[Motion](motion.md#ego-motion-and-rotating-aspect)).

```python
def lobe(tilt_deg):
    tilt = math.radians(tilt_deg)
    aspect = Aspect(axis=[(math.sin(tilt), 0.0, math.cos(tilt))], exponent=2.0, coherent_interval=1.0)
    return transfers(PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0, aspect=aspect))[0]


print(f"60 degrees off the lobe: {db(lobe(60) / lobe(0)):.2f} dB")
assert abs(abs(lobe(60) / lobe(0)) - 0.5 ** 4) < 1e-4  # cos(60)^(2n), n = 2
```

### `ScatteringMatrix`: polarimetric targets

`PointTargets(scattering=ScatteringMatrix(hh, hv, vh, vv, up=...))` gives the
target a 2x2 scattering matrix (receive polarization first), relative to its
strength, in the backscatter basis `h = up x k / |up x k|`, `v = h x k`
(`k` the direction of the wave arriving at the target, `up` a world vector
not parallel to it). `ScatteringMatrix.sphere()` is the isotropic default;
`ScatteringMatrix.dihedral()` has HH and VV 180 degrees apart. The radar's
`polarization` (transmit) and `rx_polarization` (receive) pick the channel;
`"up"` and `"right"` are the pose's own axes, so a cross-polar channel is one
radar:

```python
mixer = ScatteringMatrix(hh=1.0, hv=0.5, vh=0.5, vv=1.0, up=(0.0, 1.0, 0.0))
target = (0.0, 0.0, -3.0)
co = transfers(PointTargets(positions=[target], rcs=1.0, scattering=mixer))[0]
cross = transfers(PointTargets(positions=[target], rcs=1.0, scattering=mixer), radar=radar.replace(rx_polarization="right"))[0]
sphere_cross = transfers(PointTargets(positions=[target], rcs=1.0), radar=radar.replace(rx_polarization="right"))[0]
print(f"cross/co {abs(cross / co):.3f}; isotropic target cross-pol {abs(sphere_cross):.1e}")
assert abs(abs(cross / co) - 0.5) < 1e-4 and abs(sphere_cross) == 0.0
```

A scattering matrix takes one strength and one phase shared by every target,
and does not combine with an `Aspect` or an `RcsTable`.

### `RcsTable`: a tabulated cross section

`rcs=RcsTable(axis, rcs, frequencies=None, phase=0.0)` replaces the scalar
cross section by `sigma(theta)` or `sigma(theta, f)`. `theta` is the angle
between the target's `axis` and the bisector of the directions to the
transmitter and the receiver (the monostatic aspect for a co-located radar),
sampled uniformly on `[0, pi]` by the table's last axis. A `(F, A)` table
needs ascending `frequencies` (Hz) and is interpolated to the carrier once;
the carrier must lie inside them. The table is host data and takes no
derivative.

```python
table = [1.0, 0.1, 0.01]  # m^2 at 0, 90 and 180 degrees
nose_on = transfers(PointTargets(positions=[target], rcs=RcsTable(axis=[(0.0, 0.0, 1.0)], rcs=table)))[0]
broadside = transfers(PointTargets(positions=[target], rcs=RcsTable(axis=[(1.0, 0.0, 0.0)], rcs=table)))[0]
print(f"broadside / nose-on: {db(broadside / nose_on):.2f} dB (power ratio {abs(broadside / nose_on) ** 2:.3f})")
assert abs(abs(broadside / nose_on) ** 2 - 0.1) < 1e-4
```

### `Fluctuation`: Swerling models

`fluctuation=Fluctuation(swerling, coherent_interval, seed=0)` multiplies each
target's power by an independent draw with mean one: exponential for Swerling
I and II, chi-square with four degrees of freedom for III and IV. I and III
hold one draw per `coherent_interval` (scan to scan); II and IV redraw at every
observation (pulse to pulse). Draws are a pure function of the seed, the
instant and the target, so every verb sees the same realisation. The phase is
not fluctuated.

```python
swerling_i = PointTargets(positions=[target], rcs=1.0,
                          fluctuation=Fluctuation(swerling=1, coherent_interval=1e-3, seed=3))
frames = radar.simulate(Scene(), swerling_i, times=tuple(0.01 * k for k in range(8)))
power = frames.cube.abs().square().amax(dim=(1, 2, 3, 4))
print("per-frame power / mean:", [round(float(p), 2) for p in power / power.mean()])
```

Swerling II and IV refuse `Motion.adaptive()` (interpolating between probes
would blend independent draws); a `Fluctuation` does not combine with an
`Aspect` whose axis is a function of time.

## `StructureTargets`

`StructureTargets(rcs=..., structure_ids=None)` puts one scatterer at the
world anchor Core publishes for each selected structure with rigid motion (a
`DynamicScene` trajectory), so the scatterer follows the structure. Selection
and per-target values are in ascending structure-ID order. A structure with
no rigid motion has no anchor and is refused by name. The structure itself
stays in the scene; give it a material that does not block the legs if it
should not shadow its own anchor.

```python
marker = Structure(geometry=Mesh(vertices=torch.tensor([[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0]]),
                                 faces=torch.tensor([[0, 1, 2]]), recenter=False),
                   material=PhysicalMaterial(eps_r=1.0), structure_id=7)
world = DynamicScene(Scene(structures=(marker,)),
                     structure_trajectories={7: LinearTrajectory(origin=torch.tensor([0.0, 0.0, -6.0]),
                                                                 velocity=torch.tensor([0.0, 0.0, 3.0]))})
result = radar.simulate(world, StructureTargets(rcs=1.0), times=(0.0,))
profile = result.frame(0).range_profile().data.abs()[0, 0, 0]
print(f"structure anchor at {float(result.axes.range_m[int(profile.argmax())]):.2f} m ({result.motion_sampling})")
```

## `MeshTargets`: physical-optics facets

`MeshTargets(structure_ids=None, cluster_size=None, max_sites=None, seed=0)`
makes every facet of the selected structures (all structures by default) a
scatterer. Each facet returns the physical-optics field of a flat facet of
its area and orientation to both legs, times the reflection coefficient of
its structure's own Core material at the carrier, evaluated at the facet's
local incidence angle. There is no cross section to declare. A facet scatters
from its normal side only, the side its right-handed corner order points to.
The structures stay in the scene, so a facet hidden by anything (including the
target itself) contributes nothing.

A 5 cm square plate at broadside has the analytic cross section
`4 pi A^2 / lambda^2`:

```python
def plate_scene(material, side=0.05, distance=3.0, cells=4):
    """A square plate facing the radar at z = -distance, normals toward +z."""
    grid = np.linspace(-side / 2, side / 2, cells + 1)
    y, x = np.meshgrid(grid, grid, indexing="ij")
    vertices = np.stack((x.ravel(), y.ravel(), np.full(x.size, -distance)), axis=1)
    faces = []
    for i in range(cells):
        for j in range(cells):
            a = i * (cells + 1) + j
            faces += [(a, a + 1, a + cells + 2), (a, a + cells + 2, a + cells + 1)]  # counter-clockwise seen from +z
    mesh = Mesh(vertices=torch.tensor(vertices, dtype=torch.float32), faces=torch.tensor(faces), recenter=False)
    return Scene(structures=(Structure(geometry=mesh, material=material),))


analytic = 4.0 * math.pi * (0.05 ** 2) ** 2 / radar.wavelength ** 2
reference = transfers(PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=analytic)).sum()
metal = transfers(MeshTargets(), plate_scene(PhysicalMaterial.perfect_conductor())).sum()
print(f"PEC plate vs analytic {analytic:.2f} m^2: {db(metal / reference):+.2f} dB")
assert abs(db(metal / reference)) < 0.5
```

### Materials

The facet's reflection coefficient comes from the Core `PhysicalMaterial` of
its structure: a perfect conductor, a dielectric slab (`eps_r`, `sigma_e` in
S/m, `thickness_m`), a layer stack (`layers=` of `MaterialLayer`), and the
front surface's roughness (`roughness_front=SurfaceRoughness(...)`), which
moves power from the coherent specular term (`exp(-(2 k h)^2)`) into a diffuse
term with a frozen per-facet phase drawn from `MeshTargets.seed`. Channel's
ITU-R P.2040 presets work too ([Multipath](multipath.md#scene-authoring)).

```python
materials = {
    "concrete slab": PhysicalMaterial(eps_r=5.24, sigma_e=0.0462, thickness_m=0.1),
    "two-layer stack": PhysicalMaterial(eps_r=2.5, thickness_m=0.002,
                                        layers=(MaterialLayer(thickness_m=0.002, eps_r=2.5),
                                                MaterialLayer(thickness_m=0.01, eps_r=5.0, sigma_e=0.05))),
    "rough PEC (h = lambda/16)": PhysicalMaterial.perfect_conductor(
        roughness_front=SurfaceRoughness(rms_height_m=radar.wavelength / 16, correlation_length_x_m=0.01,
                                         correlation_length_y_m=0.01)),
}
for name, material in materials.items():
    level = db(transfers(MeshTargets(), plate_scene(material)).sum() / metal)
    print(f"{name:26s} {level:+.2f} dB relative to the PEC plate")
    assert level < 0.0
```

### Multi-bounce targets need `reflections`

The last interaction of a facet path is the facet itself; the bounces before
it come from the scene's propagation. A dihedral's double bounce therefore
needs `reflections=1`, a trihedral's triple bounce `reflections=2`.

```python
def dihedral_scene(face=0.05, distance=3.0, cells=8):
    """A right-angle PEC dihedral, edge along y at z = -distance, opening toward the radar."""
    vertices, faces = [], []
    for sign in (1.0, -1.0):
        across = np.array([sign, 0.0, 1.0]) / math.sqrt(2.0)
        normal = np.array([-sign, 0.0, 1.0]) / math.sqrt(2.0)  # into the open side
        base = len(vertices)
        for i in range(cells + 1):
            for j in range(cells + 1):
                vertices.append(np.array([0.0, face * (j / cells - 0.5), -distance]) + across * (face * i / cells))
        for i in range(cells):
            for j in range(cells):
                p, q = base + i * (cells + 1) + j, base + (i + 1) * (cells + 1) + j
                for tri in ((p, p + 1, q + 1), (p, q + 1, q)):
                    a, b, c = (vertices[k] for k in tri)
                    faces.append(tri if np.dot(np.cross(b - a, c - a), normal) > 0 else (tri[0], tri[2], tri[1]))
    mesh = Mesh(vertices=torch.tensor(np.array(vertices), dtype=torch.float32), faces=torch.tensor(faces),
                recenter=False)
    return Scene(structures=(Structure(geometry=mesh, material=PhysicalMaterial.perfect_conductor()),))


corner = dihedral_scene()
analytic = 8.0 * math.pi * 0.05 ** 4 / radar.wavelength ** 2
reference = transfers(PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=analytic)).sum()
for reflections in (0, 1):
    level = db(transfers(MeshTargets(), corner, reflections=reflections).sum() / reference)
    print(f"dihedral, reflections={reflections}: {level:+.2f} dB vs analytic")
assert abs(db(transfers(MeshTargets(), corner, reflections=1).sum() / reference)) < 1.0
```

### Site budget: `cluster_size` and `max_sites`

Every facet is one scatter site, and each site costs `(N_tx + N_rx)` leg rows
and `N_tx * N_rx` composed rows (times its multipath), so a fine mesh is
expensive. `cluster_size` (m) merges the facets whose centroids share one
cubic cell of that edge into one site; `max_sites` picks the finest such cell
that keeps the site count within the budget (give one or the other). A merged
site keeps every facet's own term and phase, sits on the member facet nearest
the cell mean, and is occluded as one point. It is valid while the cell is
small against the surface's curvature and well inside the far field
(`k D^2 / (8 R)` small for a cell of extent `D` at range `R`).

```python
fine = plate_scene(PhysicalMaterial.perfect_conductor(), cells=16)  # 512 facets
exact = transfers(MeshTargets(), fine)
merged = transfers(MeshTargets(max_sites=16), fine)
print(f"{exact.numel()} rows -> {merged.numel()} rows, level change {db(merged.sum() / exact.sum()):+.3f} dB")
assert merged.numel() <= 16 and abs(db(merged.sum() / exact.sum())) < 0.2
```

### Known limitations of mesh targets

- With Channel 0.6.0, reflection discovery can return no rows off a finely
  meshed face that is not axis-aligned while the result still reports
  `complete=True`: a 5 cm PEC dihedral at 10 m with 45-degree faces loses its
  double bounce (-43 dB) at 52 or more cells per edge (10,816 faces and up);
  50 per edge works. A flat plate, upright or tilted 20 degrees, keeps its
  reflection at 18,432 faces. Use the coarsest mesh that resolves the shape,
  and check a multi-bounce target's level against a coarser mesh.
- A mesh target's outbound legs are restricted to the line of sight: a
  facet -> wall -> receiver ghost is not composed (its reciprocal
  transmitter -> wall -> facet is).
- There is no default material: the facet's material is its structure's.
- A site on or within a fraction of a millimetre of a conductor can lose its
  wall reflection (a Channel limitation).

## Polarization and element patterns: limitations

Radar owns the element pattern and polarization: each leg's field reaches the
target unprojected, the target's scattering matrix (or the facet's TE/TM
coefficients) acts on it, and the receiver reads the component along its
polarization. Two things are not modelled:

- **Cross-polar (Jones) element patterns.** A `Pattern` is a co-polar gain;
  an element that radiates a cross-polar component, in particular on
  reflected rows, is not represented.
- An isotropic, linearly polarized element has a polarization null on its own
  polarization axis; the echo there is exactly zero.

## SMPL bodies

`witwin.radar.smpl` turns an SMPL human body into a mesh target. It needs the
`smpl` extra (`smplpytorch`, OpenCV) and the SMPL model files, which are not
shipped and must be obtained from the SMPL authors. The model root defaults
to `models/smpl_models` next to the package source; pass `model_root=` to
point elsewhere. The block below therefore does not run in this
documentation's test.

```python no-run
import torch
from witwin.core import PhysicalMaterial, Scene
from witwin.radar.smpl import SMPLBody, SmplPoseDeformation

body = SMPLBody(pose=torch.zeros(72), shape=torch.zeros(10), position=(0.0, 0.0, -4.0), gender="male",
                model_root="/path/to/smpl_models")
walking = SmplPoseDeformation(body, pose_rate=torch.zeros(72))  # pose derivative, rad/s, same 72-element layout
skin = PhysicalMaterial(eps_r=6.4, sigma_e=42.0, thickness_m=0.01, name="skin")  # approximate wet skin near 77 GHz
scene, targets = walking.mesh_target(Scene(), structure_id=7, material=skin, max_sites=2048)
result = radar.simulate(scene, targets, times=(0.0,))
```

`SmplPoseDeformation.mesh_target(scene, structure_id=, material=, cluster_size=, max_sites=, seed=)`
adds the body to `scene` as one Core structure with `recenter=False`, deformed
over time by the pose and `pose_rate`, and returns `(DynamicScene, MeshTargets)`.
The body's 13,776 faces make `max_sites` or `cluster_size` the practical
choice. `material` is required. The pose, shape, position and rotation are
reverse-mode leaves (`ad_mode="vjp"`); a forward tangent is refused.

See also: [Motion](motion.md), [Multipath](multipath.md),
[Conventions](conventions.md).
