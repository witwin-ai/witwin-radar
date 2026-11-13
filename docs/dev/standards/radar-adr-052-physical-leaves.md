# R-ADR-052: Physical inputs as AD leaves - patterns, gains, pose, materials and the SMPL body

Status: Accepted

## Context

R-ADR-018 sorted every Radar AD cell into `SUP`, `ZERO`, `REF` or `DECL`. Several
physically meaningful inputs sat in `REF` (or, worse, were silently converted to
host floats) only because no native slot carried them: the antenna pattern
tables and peak gains, the LNA gain, the transmit power, the radar pose, the
material of a mesh target, and the SMPL pose and shape behind the deformation
bridge. Each of them enters the physics smoothly, so its derivative is defined.
Before this record, `Radar(power=tensor)` and `Radar(lna_gain=tensor)` passed
`float()` checks and severed the tape without a word, `Pattern.dipole(gain=t)`
did the same, and a Core material's `eps_r`, `sigma_e`, `thickness_m` or
roughness height reached the facet tables through `float()`/`complex()`.

## Decision

Each input below is a leaf in the modes listed; everything else keeps its
R-ADR-018 state. The hot path stays native: Torch only assembles the small
parameter buffers once per topology and applies scalar unit factors.

### Antenna patterns (`sensor_weight.cu`, `sensors.py`)

A `Pattern` built from tensors (`separable`, `table`, `field_separable`,
`field_table`, or a 0-dim tensor `gain` on any constructor) packs its buffer on
the tape. With each element written as a complex amplitude `A` (the field `F`,
or `sqrt(max(G, 0))` for a power pattern) the row scale is
`q A_t A_r`, `q = sqrt(max(I, 0)) a_t a_r`, `a = 10^(G_dBi / 20)`. The table
values enter `A` linearly with the interpolation weights as constants of the
direction (bilinear map: four cells; separable: two knots per cut and the
product rule), so the derivative is exact; a power element takes
`dA = dG / (2 sqrt G)` for `G > 0` and 0 at the clamp, the convention the
direction derivative already uses. For a real loss and the row's scale
cotangent `Gs`, the cotangent of `A_t` is `q conj(A_r) Gs` and of `a_t` is
`sqrt(I) a_r Re(conj(Gs) A_t A_r)`.

The jvp takes optional tangent buffers (empty is zero and costs nothing). The
backward writes each row's contributions as `(key, value)` pairs over
`[tx table | rx table | frame]`; the caller stable-sorts the keys and the native
`keyed_sum_forward` adds each run in ascending row order. This is the same
determinism contract as the antenna-position reduction (no atomics), and it
runs only when a table or the frame is a leaf: one extra launch and one sort.

The angle axes stay host floats and a tensor axis is refused: a knot decides
which segment a direction lands in, so moving one is a different table.

### Pattern frame and radar pose (`sensor_weight.cu`, `radar.py`)

`Radar.position`, `look_at` and `up` may be tensors and keep their tape: the
element positions and the pattern frame are differentiable expressions of them.
The positions reach both legs through Channel's fixed-topology replay and the
two-way join (already `SUP`); the frame reaches the weight through a new
frame slot of the sensor operator: with `w_j = v . axis_j`,
`dA / daxis_j = v (dA/dw_j)` and, at the orthonormal frame the radar builds,
`dA/dw_j = (dA/dv) . axis_j`. A `pose=` trajectory's translation is a leaf the
same way.

Refused, before any work: an `"up"`/`"right"` polarization with a pose leaf
(it turns with the pose), a pose trajectory's rotation and a polarization leaf
(Channel's replay takes endpoint polarizations as primal-only constants; the
field is linear in the launched dipole but the 3x3 transport is not published),
and a pose leaf under a `grad` mode that does not ask Channel for that
derivative (`"vjp"` for a gradient, `"jvp"` for a tangent) - the legs' part
would be missing and the pattern and join parts alone would be a silently
partial derivative.

### Transmit power and LNA gain (`radar.py`, `frontend.py`)

A 0-dim tensor `Radar.power` launches Channel's legs with its host value
(endpoint powers are primal-only there) and enters as
`s_P = 10^((p - p) / 20)`, exactly one in value with derivative
`ln(10) / 20` per dB - the `sqrt(P)` amplitude law. It is folded into the
transmit element's packed peak, so every composed row carries it once through
the native table derivative, and the leakage rows take it too.

A 0-dim tensor `Radar.lna_gain` runs the chain at the host gain and applies
`s (y - dc) + dc` with `s = 10^((g - g) / 20)` after the quadrature split and
the baseband stage. Every stage between the LNA and the AGC is homogeneous of
degree one in the gain - thermal noise and compression are input-referred, the
IQ split is linear, the flicker level scales with the gain - except the DC
offset, which is why it is taken out and put back. The AGC and the ADC then
see a differentiable signal like any other (the ADC still refuses one). Cost:
two library calls over the cube, only when the gain is a leaf.

A complex tensor `Coupling.tx` or `Coupling.rx` matrix is a leaf: the coupling
stage is two cuBLAS products (`Y = C_tx^T H C_rx^T`) whose derivative in both
modes is the transposed products, so the chain carries it as given.

The other frontend scalars keep R-ADR-018's host-float rule: their
derivative needs a reparameterised noise model (`Noise`), a control-target
decision (`Agc`), or is a quantiser (`Adc`). The compression parameters
(`iip3`, `p1db`, `am_pm`) stay host floats: the stage is smooth in them except
at its saturation hold point, but the native compression operator has no
parameter slot, each parameter needs a cube-wide reduction, and no consumer has
asked; recorded as a deferral.

### Mesh-target materials (`facet_scatter.cu`, `facets.py`, `scattering.py`)

A Core `PhysicalMaterial` whose `eps_r`, `sigma_e`, `thickness_m`, `mu_r` or
front-roughness `rms_height_m` is a tensor reaches the facet tables with its
tape: Core evaluates `eps` at the carrier differentiably and the facet module
assembles the `(model, rms)` and `(thickness, Re eps, Im eps, mu)` rows with a
handful of library calls per topology. Both tables are differentiable inputs of
the facet operators. The layer stack has ONE templated expression; the
geometry-only path runs it on a one-lane jet over the cosine and chains the
slope, and a material-aware path runs it on the caller's jet with the layer
parameters as jets. The JVP carries material tangents on its lane beside the
geometry's. The VJP gives each facet's thread a second sweep over the same rows
in the same order, one pass per layer on a five-lane jet (rms on lane 0 of the
first pass, the layer's four fields on lanes 1 to 4), into a per-facet scratch
row, and a second launch inside the same operator sums those rows per material
entry in facet order. Neither runs without a material leaf.

A roughness leaf at exactly zero is refused: the diffuse amplitude grows like
`|h|` there, so the smooth surface is the edge of the model's domain with only
a one-sided derivative. The model column (conductor or layered) is a constant.

### SMPL pose, shape and transform (`smpl.py`)

`SmplPoseDeformation` no longer refuses a pose or shape gradient: every
`DeformationState` holds the vertices `_fast_smpl_forward` poses as a
differentiable expression of the pose, the shape and the body transform, and
the facet geometry, the facet response and Channel's replay carry the vertex
derivative. Forward mode stays refused at the bridge, naming why: Channel 0.6
refuses a compile whose vertices carry a dual, and `velocity_at` opens its own
forward level over the pose. `pose_rate` stays a tangent direction (ADR-038).
The layer cache is consulted before `smplpytorch` is imported, so a synthetic
linear-blend-skinning rig with the same tables runs the same path in the
tests; the real-model tests skip without the pickles.

## Consequences

- Native schema changes (ABI unchanged): `sensor_weight_jvp` takes three
  optional tangents; `sensor_weight_backward` takes the keyed-contribution
  buffers and two flags; `facet_response_jvp` takes two optional material
  tangents; `facet_response_backward` takes the material gradients, a scratch
  buffer and `max_layers`. New operator `keyed_sum_forward` (forward-only,
  listed in `no_backward`: it runs inside a backward).
- The `_SensorWeight` tape grows by the frame and the two tables (12 saved
  tensors); they are the plan's resident buffers, saved by reference.
- A radar without leaves is bit-identical: host patterns take the host packing
  route, and a tensor power or LNA gain without a derivative applies a factor
  of exactly one (pinned by `torch.equal` tests).
- `Pattern` compares and hashes its tensor fields by identity.

## Evidence

All on the GPU, against central differences of the operator and of the public
verbs, plus the VJP/JVP adjoint identity:

- `tests/test_pattern_leaves.py`: every table kind on both sides with the dBi
  gain (adjoint to 1e-4, central differences to 1 %), the pattern frame,
  deterministic reduction (`torch.equal` over two backwards), the launch budget
  (one `keyed_sum_forward` with a leaf, none without), the bit-identical host
  packing, and table and gain leaves through `Radar.simulate`.
- `tests/test_receiver_gain_leaves.py`: transmit power (isotropic, dipole,
  leakage) against central differences and the `ln(10)/20` closed form; LNA gain
  with noise, compression, IQ and a DC offset (the DC case is shown NOT to
  follow the whole-signal law); transmit and receive coupling matrices;
  bit-identity of tensors without a derivative.
- `tests/test_radar_pose_leaves.py`: `position`, `look_at`, `up` in both modes,
  a trajectory's translation, and the four refusals.
- `tests/test_facet_material_leaves.py`: the material tables at the operator
  (PEC, one and two layers, smooth and rough), Core `eps_r`, `sigma_e`,
  `thickness_m` and roughness through `Radar.trace` in both modes, the zero-rms
  refusal, and a profiler count showing the material sweep runs only when asked.
- `tests/test_smpl_pose_leaves.py`: pose, shape and body position through
  `mesh_target` and `Radar.trace` on the synthetic rig (3 %, the vertex-gradient
  test's tolerance: the facet response is float32 and a facet's visibility is a
  discrete gate), the tangent and `pose_rate` refusals.
