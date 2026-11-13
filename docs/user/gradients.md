# Gradients and inverse design

A Radar result is differentiable with respect to the physical inputs that
enter it smoothly: target positions and trajectories, cross sections, mesh
vertices and materials, the radar's pose, element patterns and gains,
transmit power, LNA gain and coupling. You make an input a leaf by passing a
`torch` tensor that requires a gradient (reverse mode) or carries a forward
tangent (forward mode), and you tell the world-facing verbs which mode to ask
Channel for with `ad_mode`. Everything else is refused before any work, with
an error that says why; nothing is silently detached.

## `ad_mode`

`Radar.trace`, `simulate` and `stream` (and the `Network` verbs) take
`ad_mode=`:

| `ad_mode` | Propagation (Channel) derivatives | Use |
|---|---|---|
| `"none"` (default) | none | forward simulation; Radar-side leaves (cross sections, patterns, power, LNA gain, coupling) still differentiate |
| `"vjp"` | reverse mode | `loss.backward()` / `torch.autograd.grad` |
| `"jvp"` | forward mode | `torch.autograd.forward_ad` tangents |

The mode matters for every leaf that moves geometry Channel traces: target
positions and trajectories, mesh vertices and materials, and the radar pose.
With `ad_mode="none"` a target-position leaf is not differentiated (the cube
carries no graph from it), and a radar-pose leaf is refused by name. Only
first-order derivatives are supported; a gradient of a gradient is refused.

## Which inputs are leaves

Each differentiable record lists its leaves and their modes in `LEAVES`, and
`witwin.radar.capabilities()["ad_leaves"]` collects them (plus the Core
material fields a mesh target reads):

```python
import math

import torch
import torch.autograd.forward_ad as fwAD
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure
from witwin.radar import Coupling, Fmcw, MeshTargets, Pattern, PointTargets, Radar, capabilities
from witwin.radar import processing as P

for record, leaves in capabilities()["ad_leaves"].items():
    print(f"{record:12s}", ", ".join(f"{name} ({'/'.join(modes)})" for name, modes in leaves.items()))
```

| Record | Leaf | Notes |
|---|---|---|
| `PointTargets` | `positions`, `rcs`, `amplitude`, `phase`, and tensors a `trajectory` closes over | positions and trajectories need `ad_mode` |
| `Radar` | `position`, `look_at`, `up` | need `ad_mode`, and a world-vector `polarization` |
| `Radar` | `power` (dBm), `lna_gain` (dB) | 0-dim tensors |
| `Pattern` | `x_gain`, `y_gain`, `gain_table` (the table values), `gain` (dBi) | float tensors for power, complex for field; angle axes stay host floats |
| `Coupling` | `tx`, `rx` | complex tensor matrices |
| Core `PhysicalMaterial` on a `MeshTargets` structure | `eps_r`, `sigma_e`, `thickness_m`, `mu_r`, roughness `rms_height_m` | need `ad_mode` |
| Core `Mesh` vertices (walls and `MeshTargets`) | vertex tensor | reverse and forward mode |
| `witwin.radar.smpl.SMPLBody` | `pose`, `shape`, `position`, `rotation` | reverse and forward mode |

The setup used below is a small version of the quickstart radar (four chirps,
beat output):

```python
chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=256, chirps_per_frame=4,
             adc_start=6e-6, idle=7e-6, ramp_end=65e-6, output="beat")
radar = Radar(reference_frequency_hz=77e9, waveform=chirp, power=15.0, antenna_unit="half_wavelength",
              tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
scene = Scene()


def energy(result):
    return result.cube.abs().square().sum()
```

## Target positions, cross sections and trajectories

Reverse mode: make the leaves, simulate with `ad_mode="vjp"`, and
differentiate any scalar of the result. Received power goes as the cross
section, so `d energy / d rcs = energy / rcs`:

```python
positions = torch.tensor([[0.0, 0.0, -3.0]], device="cuda", requires_grad=True)
rcs = torch.tensor([2.0], device="cuda", requires_grad=True)
e = energy(radar.simulate(scene, PointTargets(positions=positions, rcs=rcs), times=(0.0,), ad_mode="vjp"))
d_position, d_rcs = torch.autograd.grad(e, (positions, rcs))
print("d energy / d position:", [f"{v:.3e}" for v in d_position[0].tolist()])
print(f"d energy / d rcs = {float(d_rcs):.4e}, energy / rcs = {e.detach().item() / 2.0:.4e}")
assert abs(float(d_rcs) / (e.detach().item() / 2.0) - 1.0) < 1e-4
```

Forward mode carries a tangent from one input direction to the whole cube in
one pass: here, the cube's derivative with respect to moving the target along
`+z`:

```python
with fwAD.dual_level():
    moving = fwAD.make_dual(torch.tensor([[0.0, 0.0, -3.0]], device="cuda"),
                            torch.tensor([[0.0, 0.0, 1.0]], device="cuda"))
    cube = radar.simulate(scene, PointTargets(positions=moving, rcs=1.0), times=(0.0,), ad_mode="jvp").cube
    tangent = fwAD.unpack_dual(cube).tangent
print("d cube / dz:", tuple(tangent.shape), f"max |.| = {float(tangent.abs().max()):.3e}")
```

A `trajectory` is any callable `t -> [S, 3]` tensor; tensors it closes over
are leaves, so a velocity (or any motion parameter) can be fitted:

```python
speed = torch.tensor(1.0, device="cuda", requires_grad=True)


def walk(t):
    return torch.stack([torch.zeros_like(speed), torch.zeros_like(speed), -3.0 + speed * t]).reshape(1, 3)


e = energy(radar.simulate(scene, PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0, trajectory=walk),
                          times=(0.0,), ad_mode="vjp"))
(d_speed,) = torch.autograd.grad(e, speed)
print(f"d energy / d speed = {float(d_speed):.3e}")
```

## A small inverse problem: fit a target's range and cross section

The "measurement" is a simulated frame of a 2 m^2 target at 3.00 m. Starting
from a guess at 3.02 m and 0.5 m^2, gradient descent on the zero-padded range
profile's magnitude recovers both. The guess has to lie within about a range
resolution cell (4.3 cm here): further away, the two profiles do not overlap
and the gradient carries no information about where the peak is.

```python
def profile(z, log_rcs):
    position = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z]).reshape(1, 3)
    result = radar.simulate(scene, PointTargets(positions=position, rcs=log_rcs.exp().reshape(1)),
                            times=(0.0,), ad_mode="vjp")
    return P.range_profile(result.frame(0).processing_cube(), range_bins=1024).data.abs().mean(dim=(0, 1, 2))


with torch.no_grad():
    observed = profile(torch.tensor(-3.0, device="cuda"), torch.tensor(math.log(2.0), device="cuda"))
scale = float(observed.max())
z = torch.tensor(-3.02, device="cuda", requires_grad=True)
log_rcs = torch.tensor(math.log(0.5), device="cuda", requires_grad=True)
optimizer = torch.optim.Adam([{"params": [z], "lr": 0.002}, {"params": [log_rcs], "lr": 0.08}])
for step in range(60):
    optimizer.zero_grad()
    loss = ((profile(z, log_rcs) - observed) / scale).square().sum()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"step {step:2d}: loss {loss.item():.4f}, range {-z.item():.4f} m, rcs {log_rcs.exp().item():.3f} m^2")
print(f"fitted: range {-z.item():.4f} m (true 3.0000), rcs {log_rcs.exp().item():.3f} m^2 (true 2.000)")
assert abs(z.item() + 3.0) < 2e-3 and abs(log_rcs.exp().item() / 2.0 - 1.0) < 0.1
```

## Mesh vertices

A Core `Mesh` whose vertices require a gradient carries it through the facet
geometry, the facet response and Channel's replay, for a `MeshTargets`
structure and for walls alike. Author meshes in world coordinates with
`recenter=False`; the Core default moves the mesh to its bounding-box centre.
Vertices are a leaf in both modes: a gradient with `ad_mode="vjp"`, a forward
tangent with `ad_mode="jvp"`.

```python
corners = torch.tensor([[-0.05, -0.05, -2.0], [0.05, -0.05, -2.0], [0.05, 0.05, -2.0], [-0.05, 0.05, -2.0]],
                       device="cuda")


def plate_scene(vertices, material=PhysicalMaterial.perfect_conductor()):
    mesh = Mesh(vertices=vertices, faces=torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda"), recenter=False,
                fill_mode="surface", topology_diagnostics=False)
    plate = Structure(geometry=mesh, material=material, structure_id=1, material_id=1, assignment_id=1,
                      surface_id=1)
    return Scene(structures=(plate,), endpoints=[])


vertices = corners.clone().requires_grad_(True)
e = energy(radar.simulate(plate_scene(vertices), MeshTargets(), times=(0.0,), reflections=0, ad_mode="vjp"))
(d_vertices,) = torch.autograd.grad(e, vertices)
print("d energy / d vertex z:", [f"{v:.2e}" for v in d_vertices[:, 2].tolist()])
```

Forward mode on the same vertices, moving the whole plate along `+z`:

```python no-run
with fwAD.dual_level():
    dual = fwAD.make_dual(corners, torch.tensor([0.0, 0.0, 1.0], device="cuda").expand_as(corners))
    cube = radar.simulate(plate_scene(dual), MeshTargets(), times=(0.0,), reflections=0, ad_mode="jvp").cube
    tangent = fwAD.unpack_dual(cube).tangent
print("d cube / d plate z:", tuple(tangent.shape), f"max |.| = {float(tangent.abs().max()):.3e}")
```

The same plate with a dielectric whose permittivity is a leaf:

```python
eps_r = torch.tensor(4.0, requires_grad=True)
dielectric = PhysicalMaterial(eps_r=eps_r, sigma_e=0.01, thickness_m=0.01)
e = energy(radar.simulate(plate_scene(corners, dielectric), MeshTargets(), times=(0.0,), reflections=0,
                          ad_mode="vjp"))
(d_eps,) = torch.autograd.grad(e, eps_r)
print(f"d energy / d eps_r = {float(d_eps):.3e}")
```

An SMPL body is a mesh target whose vertices are a differentiable function of
the pose, shape and body transform, so those are leaves too, in both modes. It
needs the SMPL model files, so this block is not run here:

```python no-run
from witwin.radar.smpl import SMPLBody, SmplPoseDeformation

pose = torch.zeros(72, requires_grad=True)
body = SMPLBody(pose=pose, shape=torch.zeros(10), position=(0.0, 0.0, -3.0), model_root="/path/to/smpl/models")
scene, targets = SmplPoseDeformation(body, pose_rate=torch.zeros(72)).mesh_target(
    Scene(), structure_id=7, material=PhysicalMaterial(eps_r=50.0, sigma_e=1.0), cluster_size=0.05)
energy(radar.simulate(scene, targets, times=(0.0,), reflections=0, ad_mode="vjp")).backward()
print(pose.grad.shape)
```

`pose_rate` is a tangent direction for the body's motion, not a leaf.

## Radar pose

`Radar(position=, look_at=, up=)` may be tensors. They move every element and
the pattern frame, so they need `ad_mode="vjp"` or `"jvp"` and a polarization
given as a world vector (`"up"` and `"right"` turn with the pose, and Channel
publishes no derivative for an endpoint polarization). In a `pose=`
trajectory, the translation may be a leaf; the rotation may not.

```python
origin = torch.zeros(3, device="cuda", requires_grad=True)
posed = radar.replace(position=origin, polarization=(0.0, 1.0, 0.0))
e = energy(posed.simulate(scene, PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0), times=(0.0,), ad_mode="vjp"))
(d_origin,) = torch.autograd.grad(e, origin)
print("d energy / d radar position:", [f"{v:.2e}" for v in d_origin.tolist()])
```

## Patterns, gains, power and coupling

These are Radar-side leaves; they work under any `ad_mode`. A 0-dim `power`
(dBm) or `lna_gain` (dB) tensor carries the derivative `ln(10)/20` per dB in
amplitude; pattern table values and `gain` (dBi) enter linearly; coupling
matrices are complex leaves. One `pattern` serves both sides unless
`rx_pattern` is given, so its gain's derivative counts twice.

```python
power = torch.tensor(15.0, device="cuda", requires_grad=True)
gain = torch.tensor(6.0, requires_grad=True)
cuts = [float(a) for a in range(-90, 91, 10)]
table = torch.tensor([math.cos(math.radians(a)) ** 2 for a in cuts], requires_grad=True)
c_rx = torch.eye(4, dtype=torch.complex64, requires_grad=True)
leafy = radar.replace(power=power, pattern=Pattern.separable(cuts, table, cuts, [1.0] * len(cuts), gain=gain),
                      lna_gain=torch.tensor(10.0, requires_grad=True), coupling=Coupling(rx=c_rx))
off_axis = PointTargets(positions=[(1.0, 0.0, -3.0)], rcs=1.0)
e = energy(leafy.simulate(scene, off_axis, times=(0.0,)))
d_power, d_gain, d_table, d_coupling = torch.autograd.grad(e, (power, gain, table, c_rx))
value = e.detach().item()
print(f"d ln(energy) / d power = {float(d_power) / value:.4f} per dB (ln(10)/10 = {math.log(10) / 10:.4f})")
print(f"d ln(energy) / d gain  = {float(d_gain) / value:.4f} per dB (two sides: {2 * math.log(10) / 10:.4f})")
print("table gradient nonzero at", [cuts[i] for i in torch.nonzero(d_table).flatten().tolist()], "deg",
      f"(the target is {math.degrees(math.atan2(1.0, 3.0)):.1f} deg off boresight)")
assert abs(float(d_power) / value - math.log(10) / 10) < 1e-3
```

## What is refused, and why

```python
attempts = {
    "pose leaf with an 'up' polarization": lambda: radar.replace(
        position=torch.zeros(3, device="cuda", requires_grad=True)).simulate(
        scene, off_axis, times=(0.0,), ad_mode="vjp"),
    "pose leaf under ad_mode='none'": lambda: radar.replace(
        position=torch.zeros(3, device="cuda", requires_grad=True), polarization=(0.0, 1.0, 0.0)).simulate(
        scene, off_axis, times=(0.0,)),
    "polarization leaf": lambda: radar.replace(polarization=torch.tensor([0.0, 1.0, 0.0], requires_grad=True)),
    "transmission with ad_mode='vjp'": lambda: radar.simulate(
        scene, PointTargets(positions=positions, rcs=1.0), times=(0.0,), transmission=True, ad_mode="vjp"),
    "a CFAR detector on a graph": lambda: P.ca_cfar(torch.rand(32, 64, requires_grad=True)),
}
for name, attempt in attempts.items():
    try:
        attempt()
    except (RuntimeError, NotImplementedError, ValueError, TypeError) as error:
        print(f"{name}: {type(error).__name__}: {str(error)[:100]}")
    else:
        print(f"{name}: accepted")
```

* **Discrete choices have no derivative**: CFAR detection, peak grouping,
  the argmax angle estimators, point clouds, clustering and tracking; the
  ADC's rounding; the waveform and every configuration scalar (chirp timing,
  noise figure, AGC and ADC settings, `Compression` parameters, `impedance`,
  `seed`); pattern angle axes. They raise on a tensor that carries a gradient
  or a forward tangent.
* **Polarization**: a polarization leaf, an `"up"`/`"right"` polarization with
  a pose leaf, and a `pose=` rotation leaf. Channel treats an endpoint
  polarization as a constant.
* **Transmission and diffraction** exist only at their discovery instant and
  cannot be frozen for the replay a derivative goes through: run them with
  `ad_mode="none"`.
* A roughness leaf at exactly zero (the model's one-sided edge), a transmit
  power leaf inside a `Network`, and second-order derivatives.

Back-projection, range and Doppler transforms, clutter filters, range-angle
maps and micro-Doppler spectrograms are differentiable in the samples; see
[Processing](processing.md).
