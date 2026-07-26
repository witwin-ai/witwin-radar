# R-ADR-012: The kinematics-to-dual seam owns delay rate's input

Status: Accepted (Phase 7)

## Context

Phase 4 proved the delay-rate chain end to end and Phase 5 composed the two legs
in a native kernel. What none of it ever had was an INPUT.

`witwin.core.dynamics.RigidMotion` has declared `velocity` and
`angular_velocity` since it was written, and before this decision they had zero
consumers anywhere in the platform. Channel's compiler reads only `rotation` and
`translation` (`scene/compiler.py`); Radar read neither. Every Doppler number in
the repository was produced by a test that built its own tangent from a literal
velocity. So the half of the plan's work item 4 that says "from endpoint and
target kinematics" was entirely unbuilt, and `angular_velocity` - the only route
by which a rotating target or a rotor blade can produce a Doppler SPREAD rather
than a shift - had no route into any computation at all.

Three further facts constrain the answer.

1. **A forward-AD tangent dies silently.** `make_dual(p, v)` followed by any
   rebuild of `p` from Python values yields an ordinary tensor with no tangent,
   and the chain then publishes `delay_rate = 0`. Zero is exactly what a correct
   stationary scene publishes, so the failure is invisible to inspection and to
   any test whose fixture is purely transverse.
2. **A round trip needs three tangents at once.** The inbound leg's rate is
   `d|p_site - p_tx|/dt` and the outbound leg's is `d|p_rx - p_site|/dt`. One
   `dual_level` per tensor gives each leg one moving end and one frozen end.
3. **Core has no deformation velocity.** `DeformationState` states where the
   vertices are and never how fast they are moving. Production finite
   differences are forbidden, and Core is read-only for this work.

## Decision

`witwin/radar/propagation/kinematics.py` is the SINGLE Radar owner of the
conversion from a Core snapshot to `(positions, velocities)`, and of the dual
level those tensors are covered by.

- **Endpoint velocity** is `EndpointState.rigid_motion.velocity`, verbatim.
  Position follows Core's own composition: the authored antenna position plus
  the snapshot's additional world-frame `translation`. An endpoint's `rotation`
  is orientation and does not move its phase centre.
- **Rigid-body site velocity** is `v(p) = v_cm + omega x (p - c)`. This is the
  only place `angular_velocity` enters.
- **The rotation centre is the CURRENT TRANSLATION, not the authored pose
  position.** Channel composes a moved structure as `vertices @ R.T + t`: the
  authored world vertices are rotated about the world ORIGIN and translated
  afterwards, so `dp/dt = omega x (p - t) + t_dot`. The intuitive answer is
  wrong in a way that hides, because it adds an `omega x (t - pose)` offset that
  is uniform over the body and therefore reads as a platform velocity.
  `rotation_centre_m` states the correct value once.
- **Deformation velocity is analytic and caller-supplied** through the
  `DeformationVelocity` protocol, `velocity_at(time_s) -> (V, 3)`. The Core gap
  is RECORDED, not patched. If Core later grows a velocity descriptor on
  `DeformationState`, an adapter implementing this protocol over it is the whole
  migration.
- **`two_way_duals` covers the transmitter, site and receiver tensors in ONE
  `dual_level`**, and slot replication inside that level is `replicate_slots`,
  an `index_select` on the dual tensor.
- Output is `float32`, contiguous, and on one device, which is the Channel
  endpoint contract restated where the tensors are BUILT rather than where they
  are rejected.

The seam owns no physics. Every number downstream of it is produced by a native
Channel kernel or by the native two-way join. Building a `(N, 3)` tangent in
Torch is dual construction and metadata, and it is on the orchestration side of
the single-backend policy for the same reason endpoint batching is.

## The retardation approximation, stated

`delay_rate = rate_in + rate_out` evaluates BOTH legs at the same world instant
`t`. The exact two-way rate evaluates the outbound leg at `t + tau_in`, where
the target has moved on, and carries a `(1 - v_r/c)` factor from the same
retardation. The relative error is therefore `O(v/c)`: about `4e-8` at 12 m/s,
five orders of magnitude below the float32 delay quantisation these rows are
published at.

This is now stated in `RadarPathBatch.delay_rate` and in
`TwoWayComposer.compose` rather than left implicit, because it is the one
approximation in the composition that a VELOCITY rather than a geometry can make
visible: an acceptance test driven at an absurd velocity would otherwise
disagree with a relativistic oracle and read as a defect.

## Consequences

- Every ADR-038 fixture in the dynamics phase must carry a non-zero RADIAL
  component. A transverse-only fixture cannot distinguish a dead tangent from a
  correct zero, and `SITE_P_RADIAL_VELOCITY_M_PER_S` exists for exactly that.
  `test_a_dead_tangent_is_detectable` reproduces the PARTIAL kill - one of the
  three tensors rebuilt, the others live - because the total kill is already
  refused by the adapter and the partial one is not.
- `multi_endpoint_driver` accepts live transmitter and receiver tensors. It used
  to rebuild both from Python tuples on every call, which made them
  undualisable and silently limited every Doppler test to a moving target.
- The closed-form oracle generalises with them: `leg_delay_rate_s_per_s` is
  stated once for a moving end and a fixed end, and a leg with both ends moving
  is the SUM of two calls. Nothing about a moving platform needs a second
  closed form.
- Galilean equivalence between "endpoints move at `u`" and "target moves at
  `-u`" holds EXACTLY for a line-of-sight row at any `u`, and for a reflection
  row only when `u` is parallel to the reflecting plane. The mirror does not
  move with the boost, so for a wall-normal `u` a double-reflection row's rate
  is the exact NEGATIVE of its reciprocal. That is physics, it is pinned, and it
  is a sharper statement than the equality it replaces.

## Recorded, not patched

- **Core C2.** `DeformationState` (`core/witwin/core/dynamics.py:116-147`) has
  no velocity descriptor. Minimal repro: construct a `DynamicScene` with any
  `structure_deformations` entry, take two snapshots, and observe that the only
  route from them to `d(vertices)/dt` is a difference quotient. This ADR routes
  around it with `DeformationVelocity` and does not patch Core.
- **Core C1.** A `DynamicScene` with endpoint trajectories and a completely
  static wall rebuilds the RayD scene and BVH on every distinct `time_s`,
  because `time_s` is hashed into `geometry_version`. This seam routes around it
  by construction: endpoint and target motion goes into the endpoint TENSORS and
  never into a recompile.
