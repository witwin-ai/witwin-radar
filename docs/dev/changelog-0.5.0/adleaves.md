# adleaves

## Breaking changes

- `SensorWeightGeometry.FROZEN_FIELDS` no longer lists `pattern_frame`, and `SensorWeightPlan` no longer refuses a marked table: both are differentiable inputs. The native `sensor_weight_jvp`, `sensor_weight_backward`, `facet_response_jvp` and `facet_response_backward` schemas take new (optional) arguments; callers of the internal operators must pass them (empty tensors where unused).
- `SmplPoseDeformation` accepts a pose or shape that carries a gradient instead of refusing it; a forward tangent is still refused, now with a message naming Channel's compile and `grad='vjp'`.
- `Coupling.tx`/`Coupling.rx` keep a complex tensor as given instead of refusing it when it carries a derivative.

## Added

- Antenna pattern leaves: `Pattern.separable`, `table`, `field_separable` and `field_table` accept float (power) or complex (field) tensors for the cut or table values, and every constructor accepts a 0-dim tensor `gain` (dBi). The sensor-weight operator carries the derivative of the table values, the peak gains and the pattern frame natively in both modes; the reverse pass is a keyed list reduced deterministically by the new `keyed_sum_forward` operator. A tensor angle axis is refused.
- Radar pose leaves: a tensor `Radar.position`, `look_at` or `up` keeps its tape through every element position and the pattern frame, with a world-vector polarization; a `pose=` trajectory's translation is a leaf too. Run the verb with `grad="vjp"` (gradient) or `grad="jvp"` (tangent).
- `Radar.power` and `Radar.lna_gain` accept a 0-dim tensor leaf: the transmit power through every transmitted amplitude (including leakage), the LNA gain through the receive chain up to the AGC (the DC offset excepted, as physics requires). A tensor without a derivative is bit-identical to the host value.
- Mesh-target material leaves: a Core `PhysicalMaterial` whose `eps_r`, `sigma_e`, `thickness_m`, `mu_r` or `roughness_front.rms_height_m` is a tensor reaches the facet response's reflection law and roughness terms, in both modes.
- SMPL leaves: the pose, the shape and the body position/rotation of an `SMPLBody` behind `SmplPoseDeformation.mesh_target` reach the traced rows in reverse mode.
- Complex tensor `Coupling` matrices are leaves of the receive chain.

## Fixed

- `Radar(power=tensor)`, `Radar(lna_gain=tensor)`, `Pattern.dipole(gain=tensor)`/`Pattern.isotropic(gain=tensor)` and a mesh target's Core material with tensor fields silently converted the tensor to a host float and severed the gradient; each is now a leaf.

## Changed

- Refused with a named reason, before any work: a pose leaf with an `"up"`/`"right"` polarization (it turns with the pose and Channel's replay takes polarizations as constants), a polarization leaf, a pose trajectory's rotation leaf, a pose leaf under a `grad` mode that does not carry it, a roughness rms leaf at exactly zero, and an SMPL pose/shape/transform forward tangent.
