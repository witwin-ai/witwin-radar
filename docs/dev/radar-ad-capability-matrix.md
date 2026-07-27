# Radar AD capability matrix

The authoritative statement of which Radar AD cells are supported, which are
structurally zero, which are refused, and which are declared non-differentiable
outputs. A cell is a `(route, leaf-or-output, mode)` triple.

This document is machine parsed. Adding a row without a resolvable test node id,
or in a state or mechanism outside the closed vocabularies below, fails the
matrix test.

## Vocabulary

Four target states, and **SILENT is not one of them**. Removing the silent class
is the point of Phase 9: a cell that answers with a severed derivative, a
`grad = None`, or a plausible zero and does not fail is a defect, not a state.

| State | Meaning | Required evidence |
|---|---|---|
| `SUP` | Supported. A nonzero derivative is published and it is correct. | A named test at the boundary that publishes it, validated against finite differences, an independent float64 oracle, an analytic closed form, or a jvp/vjp adjoint identity. |
| `ZERO` | Structurally zero. The leaf genuinely does not enter this physics, and exact zero is the complete answer. | A named test asserting EXACT zero, plus a falsifier showing the zero is a fact about the physics rather than a severed wire. |
| `REF` | Refused. Fails loudly BEFORE any numerical work and before any result object exists. | A named test asserting the raise, its owner, and that no partial result was produced. |
| `DECL` | Declared non-differentiable OUTPUT. The published tensor deliberately carries no graph and no tangent. | The capability record names the field and the route, a test pins the declaration against observed behaviour, and the contract document states it. Legal for outputs only, never for inputs. |

`mechanism` is one of `native-companion`, `native-declared`, `torch-orchestration`
or `host-declaration`. `torch-orchestration` is legal only outside the hot path -
result assembly, scalar construction, refusal checks - and is never physics.

`validation` is one of `fd`, `oracle-f64`, `analytic`, `adjoint`, `declaration`
or `refusal`.

`mode` is `jvp`, `vjp` or `both`.

Test node ids are given as `path::function` relative to the repository root.

## Mirrored Channel rows

Radar's acceptance cannot depend on reading another repository's file at test
time, so the Channel rows Radar actually consumes are mirrored here and pinned
against the live `witwin.channel.propagation.consumer.capabilities()` record by
`tests/test_phase4_import_boundary.py::test_the_consumer_contract_is_the_version_this_spike_was_built_against`.
The authority is Channel's own
`docs/dev/propagation-ad-capability-matrix.md` (ADR-043, consumer
`CONTRACT_VERSION` 6).

Radar requests exactly one response, `scalar_transport`, with
`components <= {los, reflection}`, always on the fixed-topology route. Every
mirrored row below is therefore a row Radar can actually reach.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| reevaluate/prepared | endpoint positions | both | SUP | native-companion | witwin/radar/propagation/channel_consumer.py:493 | tests/test_phase4_spike_e2e.py::test_reverse_mode_loss_gradients_match_the_oracle, tests/test_phase5_reflection_ad.py::test_reverse_mode_site_gradients_match_finite_differences | fd |
| reevaluate/prepared | mesh vertices | vjp | SUP | native-companion | tests/support/multi_endpoint_world.py:21 | tests/test_phase9_scene_leaf_ad.py::test_a_mesh_vertex_gradient_reaches_a_synthesized_fmcw_loss | fd |
| reevaluate/prepared | mesh vertices | jvp | SUP | native-companion | tests/support/multi_endpoint_world.py:21 | tests/test_phase9_scene_leaf_ad.py::test_a_forward_tangent_on_the_wall_matches_the_reverse_gradient | adjoint |
| reevaluate/prepared | mesh vertices, in-plane components | both | ZERO | native-companion | tests/support/multi_endpoint_world.py:21 | tests/test_phase9_scene_leaf_ad.py::test_the_in_plane_vertex_gradient_is_exactly_zero_and_that_is_correct | analytic |
| reevaluate/prepared | material eps_r | vjp | SUP | native-companion | tests/support/multi_endpoint_world.py:21 | tests/test_phase9_scene_leaf_ad.py::test_a_material_permittivity_gradient_reaches_a_synthesized_fmcw_loss | fd |
| reevaluate/prepared | material eps_r, alone | jvp | REF | torch-orchestration | witwin/radar/propagation/channel_consumer.py:718 | tests/test_phase9_scene_leaf_ad.py::test_a_material_only_forward_dual_is_refused_by_the_dead_tangent_guard | refusal |
| reevaluate/prepared | material eps_r, beside an endpoint tangent | jvp | SUP | native-companion | tests/support/multi_endpoint_world.py:21 | tests/test_phase9_scene_leaf_ad.py::test_vertices_permittivity_and_endpoints_are_live_in_one_call | fd |
| reevaluate/prepared | out:field_direction | both | SUP | native-companion | witwin/radar/propagation/channel_consumer.py:571 | tests/test_phase9_aspect_direction_ad.py::test_the_frozen_leg_publishes_a_graph_bearing_field_direction | declaration |
| discovery | out:field_direction | both | DECL | native-declared | witwin/radar/propagation/channel_consumer.py:395 | tests/test_phase4_import_boundary.py::test_the_consumer_contract_is_the_version_this_spike_was_built_against | declaration |
| reevaluate/prepared | sources.powers_w, endpoint polarizations | both | REF | host-declaration | witwin/radar/propagation/channel_consumer.py:493 | tests/test_phase4_adapter.py::test_differentiable_power_is_rejected_before_any_native_work, tests/test_phase4_adapter.py::test_differentiable_polarization_is_rejected | refusal |
| any | component diffraction, component transmission | both | REF | host-declaration | witwin/radar/propagation/channel_consumer.py:248 | tests/test_phase4_adapter.py::test_adapter_rejects_unfreezable_components | refusal |

## Propagation-adjacent Radar cells

### Aspect scatter response (`witwin/radar/scattering/aspect.py`)

`field_direction` is the only geometry input this response has, so a detached
direction takes its endpoint gradient to exactly zero. That was the state of the
tree before ADR-043 and it is now the falsifier.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| aspect/evaluate_rows | endpoint positions through dir_in (reflection row) | vjp | SUP | native-companion | witwin/radar/scattering/aspect.py:436 | tests/test_phase9_aspect_direction_ad.py::test_a_reverse_aspect_gradient_reaches_the_site_positions, tests/test_phase9_aspect_direction_ad.py::test_a_reverse_aspect_gradient_reaches_the_transmitter_through_a_reflection_row | fd |
| aspect/evaluate_rows | endpoint positions through dir_in (line-of-sight row) | vjp | SUP | native-companion | witwin/radar/scattering/aspect.py:436 | tests/test_phase9_aspect_direction_ad.py::test_a_reverse_aspect_gradient_reaches_the_site_positions | fd |
| aspect/evaluate_rows | endpoint positions through dir_out | vjp | SUP | native-companion | witwin/radar/scattering/aspect.py:436 | tests/test_phase9_aspect_direction_ad.py::test_a_reverse_aspect_gradient_reaches_the_site_positions | fd |
| aspect/evaluate_rows | endpoint positions through dir_in and dir_out | jvp | SUP | native-companion | witwin/radar/scattering/aspect.py:436 | tests/test_phase9_aspect_direction_ad.py::test_a_forward_tangent_on_the_sites_reaches_the_aspect_response | adjoint |
| aspect/compose/fmcw | endpoint positions to a synthesized cube | both | SUP | native-companion | witwin/radar/scattering/aspect.py:436 | tests/test_phase9_aspect_direction_ad.py::test_the_aspect_direction_gradient_reaches_a_synthesized_fmcw_loss, tests/test_phase9_aspect_direction_ad.py::test_the_direction_term_is_load_bearing_in_the_synthesized_loss | fd |
| aspect/evaluate_rows | outbound leg above depth zero | both | REF | host-declaration | witwin/radar/scattering/aspect.py:464 | tests/test_phase7_scatter_response_kernel.py::test_the_response_refuses_a_higher_order_outbound_leg | refusal |

### Two-way join and the wideband band (`witwin/radar/paths/two_way.py`)

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| join/_compose_band | endpoint positions through every column | vjp | SUP | native-companion | witwin/radar/paths/two_way.py:754 | tests/test_phase9_wideband_join_ad.py::test_a_reverse_endpoint_gradient_reaches_every_wideband_column | fd |
| join/_compose_band | endpoint positions through every column | jvp | SUP | native-companion | witwin/radar/paths/two_way.py:754 | tests/test_phase9_wideband_join_ad.py::test_a_forward_tangent_on_an_endpoint_reaches_the_band | adjoint |
| join/_compose_band | site and transmitter positions together | vjp | SUP | native-companion | witwin/radar/paths/two_way.py:754 | tests/test_phase9_wideband_join_ad.py::test_a_combined_endpoint_perturbation_equals_the_sum_of_its_parts | fd |
| join/_compose_band | out:frequency_response per-column derivative | vjp | SUP | native-companion | witwin/radar/paths/two_way.py:754 | tests/test_phase9_wideband_join_ad.py::test_each_wideband_column_carries_a_different_endpoint_derivative, tests/test_phase9_wideband_join_ad.py::test_the_reference_column_gradient_equals_the_narrowband_join_gradient | analytic |
| join/_compose_band | out:autograd context aliasing | vjp | SUP | torch-orchestration | witwin/radar/paths/two_way.py:754 | tests/test_phase9_wideband_join_ad.py::test_the_band_loop_keeps_one_join_context_per_column_and_aliases_its_tables | declaration |

### Kinematics (`witwin/radar/propagation/kinematics.py`)

Under ADR-038 a velocity is a forward-AD tangent DIRECTION, never a leaf.
`d(loss)/d(velocity)` does not exist in either mode.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| kinematics/Kinematics | velocities_m_per_s | vjp | REF | torch-orchestration | witwin/radar/propagation/kinematics.py:113 | tests/test_phase9_velocity_leaf_refusal.py::test_a_kinematics_velocity_that_requires_grad_is_refused | refusal |
| kinematics/Kinematics | velocities_m_per_s | jvp | REF | torch-orchestration | witwin/radar/propagation/kinematics.py:113 | tests/test_phase9_velocity_leaf_refusal.py::test_a_kinematics_velocity_carrying_a_forward_dual_is_refused | refusal |
| kinematics/LinearDeformation | velocities_m_per_s | both | REF | torch-orchestration | witwin/radar/propagation/kinematics.py:373 | tests/test_phase9_velocity_leaf_refusal.py::test_a_linear_deformation_velocity_that_requires_grad_is_refused | refusal |
| kinematics/deformation_kinematics | DeformationVelocity.velocity_at | both | REF | torch-orchestration | witwin/radar/propagation/kinematics.py:323 | tests/test_phase9_velocity_leaf_refusal.py::test_a_custom_deformation_velocity_is_refused_and_blamed_by_name | refusal |
| kinematics/rigid_site_velocities | velocity derived from a grad-carrying position | both | REF | torch-orchestration | witwin/radar/propagation/kinematics.py:113 | tests/test_phase9_velocity_leaf_refusal.py::test_a_velocity_derived_from_a_grad_carrying_position_is_refused | refusal |
| kinematics/two_way_duals | velocity as a tangent direction | jvp | SUP | native-companion | witwin/radar/propagation/kinematics.py:561 | tests/test_phase9_velocity_leaf_refusal.py::test_the_same_velocity_as_a_tangent_direction_stays_supported, tests/test_phase7_doppler_scenarios.py::test_a_radially_moving_site_matches_the_projection_formula | analytic |
| kinematics/two_way_duals | position leaf beside a velocity tangent | both | SUP | native-companion | witwin/radar/propagation/kinematics.py:561 | tests/test_phase9_velocity_leaf_refusal.py::test_a_position_leaf_and_a_velocity_tangent_coexist_in_one_dual | declaration |
| kinematics/structure_site_kinematics | stationary structure velocity | both | ZERO | torch-orchestration | witwin/radar/propagation/kinematics.py:296 | tests/test_phase9_velocity_leaf_refusal.py::test_a_stationary_structure_state_still_builds_exact_zeros | analytic |

### SMPL geometry (`witwin/radar/geometry/smpl.py`)

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| smpl/SmplPoseDeformation | body pose, body shape | vjp | REF | torch-orchestration | witwin/radar/geometry/smpl.py:348 | tests/test_phase9_smpl_pose_refusal.py::test_a_pose_or_shape_derivative_is_refused_at_the_deformation_bridge | refusal |
| smpl/SmplPoseDeformation | body pose | jvp | REF | torch-orchestration | witwin/radar/geometry/smpl.py:348 | tests/test_phase9_smpl_pose_refusal.py::test_a_pose_carrying_a_forward_dual_is_refused_too | refusal |
| smpl/SmplPoseDeformation | pose_rate | both | REF | torch-orchestration | witwin/radar/geometry/smpl.py:348 | tests/test_phase9_smpl_pose_refusal.py::test_a_pose_rate_that_requires_grad_is_refused_by_adr_038 | refusal |
| smpl/rest_mesh | body position, body rotation | both | REF | torch-orchestration | witwin/radar/geometry/smpl.py:475 | tests/test_phase9_smpl_pose_refusal.py::test_a_transform_derivative_is_refused_by_rest_mesh | refusal |
| smpl/velocity_at | pose_rate as a tangent direction | jvp | SUP | torch-orchestration | witwin/radar/geometry/smpl.py:458 | tests/test_phase9_smpl_pose_refusal.py::test_the_analytic_vertex_velocity_still_comes_from_the_pose_rate | fd |
| legacy-scene/SMPLBody | pose | vjp | SUP | torch-orchestration | witwin/radar/geometry/smpl.py:319 | tests/test_phase9_smpl_pose_refusal.py::test_the_smpl_body_itself_still_publishes_a_pose_gradient | declaration |

## Deferred

Each entry names the work and the reason. A deferral is a `REF` or a `DECL` cell
above, never a silent one.

- **`field_direction` on transmission, wedge and coupled diffraction.** Channel
  forwards those families to `rayd::torch`, which publishes no direction
  cotangent or tangent. Channel declares them `DECL` in ADR-043 with a deferral
  to a RayD ADR. Radar never requests those components, so no Radar cell exists.
- **Discovery-route geometry liveness.** The derivative of a discovery result is
  only defined between selection boundaries and Channel deliberately publishes
  no subgradient at one. The supported differentiable route is
  `prepare_fixed_topology` + `reevaluate`, which is the route Radar runs per
  frame. Channel owns the deferral.
- **A pose derivative into the compiled scene.** `SmplPoseDeformation` publishes
  a rest `Mesh` and per-frame `DeformationState`s that cross the Core/Channel
  compile boundary, and a pose derivative is not plumbed across it. Whether a
  graph-bearing vertex tensor survives `Mesh` construction and a Channel compile
  is unverified, and a half-working pose gradient would be worse than none, so
  the bridge refuses. The supported differentiable geometry today is a
  `witwin.core.Mesh` vertex tensor. Plumbing a pose derivative across the
  boundary is a separate accepted design.
- **A material-only forward tangent.** The adapter's dead-tangent guard requires
  a `delay_s` tangent under `ad_mode='jvp'`, because a dead tangent publishes
  `delay_rate = 0`, which is indistinguishable from a correct stationary answer.
  A permittivity moves the coefficient and not the delay. Loosening the guard to
  accept a coefficient-only tangent is a decision about what `delay_rate = None`
  means to a caller and is not taken here; the cell is available today whenever
  the same call also carries an endpoint tangent.
