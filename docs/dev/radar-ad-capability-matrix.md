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
| reevaluate/prepared | endpoint positions | both | SUP | native-companion | witwin/radar/propagation/channel_consumer.py:516 | tests/test_phase4_spike_e2e.py::test_reverse_mode_loss_gradients_match_the_oracle, tests/test_phase5_reflection_ad.py::test_reverse_mode_site_gradients_match_finite_differences | fd |
| reevaluate/prepared | mesh vertices | vjp | SUP | native-companion | tests/support/multi_endpoint_world.py:21 | tests/test_phase9_scene_leaf_ad.py::test_a_mesh_vertex_gradient_reaches_a_synthesized_fmcw_loss | fd |
| reevaluate/prepared | mesh vertices | jvp | SUP | native-companion | tests/support/multi_endpoint_world.py:21 | tests/test_phase9_scene_leaf_ad.py::test_a_forward_tangent_on_the_wall_matches_the_reverse_gradient | adjoint |
| reevaluate/prepared | mesh vertices, in-plane components | both | ZERO | native-companion | tests/support/multi_endpoint_world.py:21 | tests/test_phase9_scene_leaf_ad.py::test_the_in_plane_vertex_gradient_is_exactly_zero_and_that_is_correct | analytic |
| reevaluate/prepared | material eps_r | vjp | SUP | native-companion | tests/support/multi_endpoint_world.py:21 | tests/test_phase9_scene_leaf_ad.py::test_a_material_permittivity_gradient_reaches_a_synthesized_fmcw_loss | fd |
| reevaluate/prepared | material eps_r, alone | jvp | REF | torch-orchestration | witwin/radar/propagation/channel_consumer.py:742 | tests/test_phase9_scene_leaf_ad.py::test_a_material_only_forward_dual_is_refused_by_the_dead_tangent_guard | refusal |
| reevaluate/prepared | material eps_r, beside an endpoint tangent | jvp | SUP | native-companion | tests/support/multi_endpoint_world.py:21 | tests/test_phase9_scene_leaf_ad.py::test_vertices_permittivity_and_endpoints_are_live_in_one_call | fd |
| reevaluate/prepared | out:field_direction | both | SUP | native-companion | witwin/radar/propagation/channel_consumer.py:594 | tests/test_phase9_aspect_direction_ad.py::test_the_frozen_leg_publishes_a_graph_bearing_field_direction | declaration |
| discovery | out:field_direction | both | DECL | native-declared | witwin/radar/propagation/channel_consumer.py:418 | tests/test_phase4_import_boundary.py::test_the_consumer_contract_is_the_version_this_spike_was_built_against | declaration |
| reevaluate/prepared | sources.powers_w, endpoint polarizations | both | REF | host-declaration | witwin/radar/propagation/channel_consumer.py:516 | tests/test_phase4_adapter.py::test_differentiable_power_is_rejected_before_any_native_work, tests/test_phase4_adapter.py::test_differentiable_polarization_is_rejected | refusal |
| any | component diffraction, component transmission | both | REF | host-declaration | witwin/radar/propagation/channel_consumer.py:271 | tests/test_phase4_adapter.py::test_adapter_rejects_unfreezable_components | refusal |

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

## Synthesis, frontend and sensor cells

### The host-float rule (`witwin/radar/host_parameters.py`)

Every configuration scalar on every waveform spec and every frontend stage spec
is a host float and refuses a tensor at construction. The rule is on the TYPE
rather than on `requires_grad`, because a tensor that does not require grad
today is the input that starts requiring grad tomorrow, and `float()` on a
device tensor is a host synchronisation as well. All 85 spec construction sites
in the tree were checked and the whole suite re-run, default and `--gpu`: no
caller was displaced.

Each spec's row covers every scalar it declares; the per-field cases are the
parametrized ids inside the cited test.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| frontend/PortSpec | reference_impedance_ohm | both | REF | host-declaration | witwin/radar/frontend/contracts.py:72 | tests/test_phase9_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| frontend/NoiseSpec | noise_figure_db, antenna_temperature_k, bandwidth_hz, phase_noise_dbc_per_hz, phase_offset_hz, phase_sample_rate_hz | both | REF | host-declaration | witwin/radar/frontend/contracts.py:81 | tests/test_phase9_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| frontend/LnaSpec | gain_db | both | REF | host-declaration | witwin/radar/frontend/contracts.py:95 | tests/test_phase9_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| frontend/AgcSpec | target_rms, min_gain_db, max_gain_db | both | REF | host-declaration | witwin/radar/frontend/contracts.py:105 | tests/test_phase9_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| frontend/AdcSpec | bits, full_scale | both | REF | host-declaration | witwin/radar/frontend/contracts.py:115 | tests/test_phase9_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| frontend/SeedSpec | seed_base | both | REF | host-declaration | witwin/radar/frontend/contracts.py:404 | tests/test_phase9_host_float_refusal.py::test_the_seed_is_refused_by_its_own_older_type_rule | refusal |
| synthesis/FmcwBeatSpec | every waveform scalar | both | REF | host-declaration | witwin/radar/synthesis/contracts.py:230 | tests/test_phase9_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| synthesis/OfdmCfrSpec | every waveform scalar | both | REF | host-declaration | witwin/radar/synthesis/contracts.py:469 | tests/test_phase9_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| synthesis/PulsedEchoSpec | every waveform scalar | both | REF | host-declaration | witwin/radar/synthesis/contracts.py:722 | tests/test_phase9_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| synthesis/DirichletSpectrumSpec | every waveform scalar | both | REF | host-declaration | witwin/radar/synthesis/dirichlet_spectrum.py:98 | tests/test_phase9_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| any spec | an unmarked tensor, the tomorrow case | both | REF | host-declaration | witwin/radar/host_parameters.py:40 | tests/test_phase9_host_float_refusal.py::test_every_configuration_scalar_refuses_an_unmarked_tensor_too | refusal |
| any spec | a forward dual | jvp | REF | host-declaration | witwin/radar/host_parameters.py:40 | tests/test_phase9_host_float_refusal.py::test_a_forward_dual_is_refused_as_well | refusal |

### Sensor weight (`witwin/radar/sensors/weights.py`)

Six differentiable inputs and thirteen frozen constants. The constants used to
return `None` from `backward` by construction, with no refusal anywhere; they now
fail at `SensorWeightGeometry` / `SensorWeightPlan` construction, before
`validate` and before any launch.

One caller in the tree was displaced and it was FIXED rather than the rule
softened: `tests/solvers/test_mimo_cross.py` marked a velocity and handed it to
`Radar.mimo_from_trace`. That request was measured, before the change, to run a
whole frame and return `velocities.grad is None` while the position gradient
came back correctly, so no capability was removed. The call site now detaches,
with the reason written there.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| sensor_weight/evaluate | tx_pos, rx_pos, site_in, site_out | jvp | SUP | native-companion | witwin/radar/sensors/weights.py:332 | tests/test_phase6_sensor_weight.py::test_the_jvp_matches_a_central_finite_difference | fd |
| sensor_weight/evaluate | tx_pos, rx_pos, site_in, site_out | vjp | SUP | native-companion | witwin/radar/sensors/weights.py:332 | tests/test_phase6_sensor_weight.py::test_the_vjp_is_the_adjoint_of_the_jvp | adjoint |
| sensor_weight/evaluate | intensity, weight real and imaginary parts | both | SUP | native-companion | witwin/radar/sensors/weights.py:332 | tests/test_phase6_sensor_weight.py::test_the_vjp_is_the_adjoint_of_the_jvp | adjoint |
| sensor_weight/evaluate | antenna position reduction order | vjp | SUP | native-companion | witwin/radar/sensors/weights.py:332 | tests/test_phase6_sensor_weight.py::test_the_antenna_gradient_is_a_deterministic_reduction | declaration |
| sensor_weight/SensorWeightGeometry | tx_velocity, rx_velocity, site_velocity | both | REF | host-declaration | witwin/radar/sensors/weights.py:195 | tests/test_phase9_sensor_constant_refusal.py::test_a_marked_frozen_geometry_field_is_refused, tests/test_phase9_sensor_constant_refusal.py::test_a_velocity_refusal_names_the_reason_a_velocity_is_never_a_leaf | refusal |
| sensor_weight/SensorWeightGeometry | fixed_length_m, normals, pol_tx, pol_rx, local_axes | vjp | REF | host-declaration | witwin/radar/sensors/weights.py:195 | tests/test_phase9_sensor_constant_refusal.py::test_a_marked_frozen_geometry_field_is_refused | refusal |
| sensor_weight/SensorWeightGeometry | fixed_length_m, normals, pol_tx, pol_rx, local_axes | jvp | REF | host-declaration | witwin/radar/sensors/weights.py:195 | tests/test_phase9_sensor_constant_refusal.py::test_a_dual_carrying_frozen_geometry_field_is_refused | refusal |
| sensor_weight/SensorWeightPlan | pattern tables | both | REF | host-declaration | witwin/radar/sensors/weights.py:239 | tests/test_phase9_sensor_constant_refusal.py::test_a_marked_pattern_table_is_refused | refusal |
| sensor_weight/round_trip | a marked velocity from a production entry point | both | REF | host-declaration | witwin/radar/sensors/weights.py:195 | tests/test_phase9_sensor_constant_refusal.py::test_a_marked_field_on_the_production_geometry_is_still_refused, tests/test_phase9_sensor_constant_refusal.py::test_the_production_geometry_carries_no_marked_frozen_field | refusal |
| sensor_weight/evaluate | out:pattern_gain | both | DECL | native-declared | witwin/radar/sensors/weights.py:410 | tests/test_phase6_sensor_weight.py::test_the_kernel_pattern_gain_equals_the_torch_pattern_gain | declaration |
| sensor_weight/round_trip | site positions to a composed round-trip weight | vjp | SUP | native-companion | witwin/radar/sensors/round_trip.py:293 | tests/test_phase11_antenna_pattern_route.py::test_the_reverse_gradient_of_a_site_matches_a_central_difference | fd |
| sensor_weight/round_trip | transmit element positions to a composed round-trip weight | vjp | SUP | native-companion | witwin/radar/sensors/round_trip.py:293 | tests/test_phase11_antenna_pattern_route.py::test_the_reverse_gradient_of_a_transmit_element_matches_a_central_difference | fd |
| sensor_weight/round_trip | site positions to a composed round-trip weight | jvp | SUP | native-companion | witwin/radar/sensors/round_trip.py:293 | tests/test_phase11_antenna_pattern_route.py::test_the_forward_tangent_of_a_site_matches_a_central_difference | fd |

The last three rows are Phase 11's PRODUCTION route for this family. Until
`RoundTripPatternStage` existed the family's only importer was
`sensors/legacy_paths.py`, on the Dirichlet route that Phase 11 deletes, so
every row above described a capability no production entry point could reach.
The stage reaches all three companions through `evaluate_sensor_weights` and
adds no `torch.autograd.Function` of its own, so the tape ledger's ten owners
are still ten and the boundary budget is unchanged.

### Scatter response (`witwin/radar/scattering/rcs.py`)

The one capability ADD of this stage. A radar cross section is scene state and
the canonical inverse-design leaf, which is why it sits on the other side of the
host-float rule from every spec scalar above. The `sqrt` is result construction,
off the per-path loop, so `torch-orchestration` is the legal mechanism; every
per-path product downstream of it is still a native kernel.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| rcs/from_rcs | sigma_m2 to a synthesized FMCW loss | vjp | SUP | torch-orchestration | witwin/radar/scattering/rcs.py:44 | tests/test_phase9_rcs_sigma_ad.py::test_the_cross_section_gradient_equals_the_exact_closed_form, tests/test_phase9_rcs_sigma_ad.py::test_the_cross_section_gradient_matches_a_central_difference | analytic |
| rcs/from_rcs | sigma_m2 through the square-root law | vjp | SUP | torch-orchestration | witwin/radar/scattering/rcs.py:44 | tests/test_phase9_rcs_sigma_ad.py::test_the_gradient_is_the_amplitude_gradient_through_the_square_root | analytic |
| rcs/from_rcs | sigma_m2 to a synthesized FMCW loss | jvp | SUP | torch-orchestration | witwin/radar/scattering/rcs.py:44 | tests/test_phase9_rcs_sigma_ad.py::test_the_forward_tangent_matches_the_reverse_gradient | adjoint |
| rcs/from_values | amplitude, phase_rad | both | SUP | torch-orchestration | witwin/radar/scattering/rcs.py:132 | tests/test_phase4_spike_e2e.py::test_reverse_mode_loss_gradients_match_the_oracle, tests/test_phase4_spike_e2e.py::test_forward_mode_loss_tangent_matches_the_oracle | fd |
| rcs/from_rcs | sigma_m2 with requires_grad=True as well | both | REF | torch-orchestration | witwin/radar/scattering/rcs.py:147 | tests/test_phase9_rcs_sigma_ad.py::test_marking_the_derived_amplitude_as_well_is_refused | refusal |
| rcs/from_rcs | a non-scalar sigma_m2 | both | REF | torch-orchestration | witwin/radar/scattering/rcs.py:44 | tests/test_phase9_rcs_sigma_ad.py::test_a_non_scalar_cross_section_is_refused | refusal |

### FMCW beat synthesis (`witwin/radar/synthesis/fmcw_beat.py`)

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| synthesis/fmcw_beat | tau_rt | jvp | SUP | native-companion | witwin/radar/synthesis/fmcw_beat.py:260 | tests/test_phase9_fmcw_variable_jvp.py::test_the_jvp_of_each_differentiable_input_matches_a_central_difference | oracle-f64 |
| synthesis/fmcw_beat | tau_rate | jvp | SUP | native-companion | witwin/radar/synthesis/fmcw_beat.py:260 | tests/test_phase9_fmcw_variable_jvp.py::test_the_jvp_of_each_differentiable_input_matches_a_central_difference, tests/test_phase9_fmcw_variable_jvp.py::test_the_rate_derivative_is_not_the_delay_derivative_scaled_by_slow_time | analytic |
| synthesis/fmcw_beat | weight real and imaginary parts | jvp | SUP | native-companion | witwin/radar/synthesis/fmcw_beat.py:260 | tests/test_phase9_fmcw_variable_jvp.py::test_the_jvp_of_each_differentiable_input_matches_a_central_difference | oracle-f64 |
| synthesis/fmcw_beat | tau_rt, tau_rate, weight | vjp | SUP | native-companion | witwin/radar/synthesis/fmcw_beat.py:260 | tests/test_phase4_fmcw_beat_ad.py::test_native_vjp_matches_the_oracle, tests/test_phase4_fmcw_beat_ad.py::test_multi_segment_vjp_matches_the_oracle | oracle-f64 |
| synthesis/fmcw_beat | tau_rate at the first chirp | jvp | ZERO | native-companion | witwin/radar/synthesis/fmcw_beat.py:260 | tests/test_phase9_fmcw_variable_jvp.py::test_a_rate_only_tangent_is_not_the_zero_tangent | analytic |

### End-to-end waveform chains (`tests/support/waveform_chains.py`)

The FMCW chain has been covered since Phase 4. These are the OFDM and pulsed
equivalents: a Core endpoint leaf through the Channel consumer, the two-way
join, the scatter response and the waveform kernel to a scalar loss, in both
modes, against a fourth-order difference of the whole production chain.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| chain/ofdm | site, transmitter and receiver positions | vjp | SUP | native-companion | witwin/radar/synthesis/ofdm_cfr.py:305 | tests/test_phase9_waveform_chain_ad.py::test_the_endpoint_gradient_matches_a_fourth_order_difference | fd |
| chain/ofdm | site, transmitter and receiver positions | jvp | SUP | native-companion | witwin/radar/synthesis/ofdm_cfr.py:305 | tests/test_phase9_waveform_chain_ad.py::test_the_forward_tangent_reproduces_the_reverse_directional_derivative | adjoint |
| chain/pulsed | site, transmitter and receiver positions | vjp | SUP | native-companion | witwin/radar/synthesis/pulsed_echo.py:283 | tests/test_phase9_waveform_chain_ad.py::test_the_endpoint_gradient_matches_a_fourth_order_difference | fd |
| chain/pulsed | site, transmitter and receiver positions | jvp | SUP | native-companion | witwin/radar/synthesis/pulsed_echo.py:283 | tests/test_phase9_waveform_chain_ad.py::test_the_forward_tangent_reproduces_the_reverse_directional_derivative | adjoint |
| chain/ofdm, chain/pulsed | the silent transmitter TX_B | vjp | ZERO | native-companion | witwin/radar/paths/two_way.py:754 | tests/test_phase9_waveform_chain_ad.py::test_the_silent_transmitter_has_an_exactly_zero_gradient | analytic |
| chain/ofdm, chain/pulsed | the out-of-plane endpoint component | vjp | ZERO | native-companion | witwin/radar/paths/two_way.py:754 | tests/test_phase9_waveform_chain_ad.py::test_the_out_of_plane_gradient_is_exactly_zero | analytic |
| chain/pulsed | tau_rt with a RECTANGULAR pulse | both | ZERO | native-companion | witwin/radar/synthesis/pulsed_echo.py:283 | tests/test_phase6_pulsed_ad.py::test_the_rectangular_envelope_has_exactly_zero_delay_gradient | analytic |
| chain/any | a forward dual covering one endpoint set only | jvp | REF | torch-orchestration | witwin/radar/propagation/channel_consumer.py:742 | tests/test_phase9_waveform_chain_ad.py::test_a_transmitter_only_forward_dual_is_refused_by_the_dead_tangent_guard | refusal |
| chain/any | one dual level over all three endpoint sets | jvp | SUP | native-companion | witwin/radar/propagation/kinematics.py:561 | tests/test_phase9_waveform_chain_ad.py::test_one_dual_level_over_all_three_endpoint_sets_is_the_supported_shape | declaration |

### Frontend chain (`witwin/radar/frontend/chain.py`)

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| frontend/adc | the signal | both | REF | host-declaration | witwin/radar/frontend/chain.py:94 | tests/test_phase6_frontend_chain.py::test_the_quantizer_refuses_a_differentiable_input | refusal |

## Processing: the non-differentiability wall

The wall sits at the first DISCRETE DECISION, not at "post-processing". Above it
the signal chain is linear or smooth and stays live; below it every stage
refuses at its entry, in both modes, before any device operation runs.
`witwin/radar/sigproc/*` are deprecated re-export shims over these owners, so
there is exactly one implementation of each guard and no second copy under
`sigproc/`.

`witwin/radar/ad_contracts.py::refuse_derivative` is the single owner of every
`REF` row here and of the frontend ADC row below. "Before any result exists" is
measured rather than asserted: each refusal test replaces the nine device
operations these stages are built from with counting stand-ins and asserts the
count is exactly zero, and the instrument is calibrated against the same stages
running normally.

Over-refusing is the opposite mistake and just as easy to make, so three of
the `REF` rows below carry a second test that asserts the guard changed
nothing else: the detector still detects and gives bitwise the same threshold
on a detached map, the point cloud still publishes its point, and the
estimator's own contract checks still fire.

### Above the wall (`SUP`)

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| processing/range_profile | the frame cube | vjp | SUP | torch-orchestration | witwin/radar/processing/range_profile.py:77 | tests/test_phase9_processing_wall.py::test_the_range_and_doppler_transforms_stay_differentiable | analytic |
| processing/range_doppler | the frame cube | vjp | SUP | torch-orchestration | witwin/radar/processing/doppler.py:40 | tests/test_phase9_processing_wall.py::test_the_range_and_doppler_transforms_stay_differentiable | analytic |
| processing/beam_cube | the frame cube | vjp | SUP | torch-orchestration | witwin/radar/processing/beam_cube.py:30 | tests/test_phase9_processing_wall.py::test_the_beam_cube_stays_differentiable | analytic |
| processing/matched_filter | the received train | vjp | SUP | torch-orchestration | witwin/radar/processing/matched_filter.py:85 | tests/test_phase9_processing_wall.py::test_the_matched_filter_stays_differentiable | analytic |
| processing/tdm_compensate | the virtual-antenna column | vjp | SUP | torch-orchestration | witwin/radar/processing/aoa.py:109 | tests/test_phase9_processing_wall.py::test_the_tdm_compensation_stays_differentiable | analytic |
| processing/music_spectrum | the angle snapshots | vjp | SUP | torch-orchestration | witwin/radar/processing/aoa.py:409 | tests/test_phase9_processing_wall.py::test_the_music_pseudo_spectrum_is_differentiable_and_matches_a_difference | fd |
| processing/music_image | the range profile | vjp | SUP | torch-orchestration | witwin/radar/processing/aoa.py:513 | tests/test_phase9_processing_wall.py::test_music_image_carries_the_same_live_derivative_as_the_spectrum | analytic |

### Below the wall (`REF`)

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| processing/ca_cfar | the Range-Doppler map | vjp | REF | host-declaration | witwin/radar/processing/cfar.py:162 | tests/test_phase9_processing_wall.py::test_a_cfar_detector_refuses_a_gradient_before_any_compute | refusal |
| processing/ca_cfar | the Range-Doppler map | jvp | REF | host-declaration | witwin/radar/processing/cfar.py:162 | tests/test_phase9_processing_wall.py::test_a_cfar_detector_refuses_a_forward_dual_before_any_compute | refusal |
| processing/ca_cfar_fast | the Range-Doppler map | vjp | REF | host-declaration | witwin/radar/processing/cfar.py:214 | tests/test_phase9_processing_wall.py::test_a_cfar_detector_refuses_a_gradient_before_any_compute , tests/test_phase9_processing_wall.py::test_the_detectors_still_detect_and_the_guard_changed_no_value | refusal |
| processing/ca_cfar_fast | the Range-Doppler map | jvp | REF | host-declaration | witwin/radar/processing/cfar.py:214 | tests/test_phase9_processing_wall.py::test_a_cfar_detector_refuses_a_forward_dual_before_any_compute | refusal |
| processing/os_cfar | the Range-Doppler map | vjp | REF | host-declaration | witwin/radar/processing/cfar.py:285 | tests/test_phase9_processing_wall.py::test_a_cfar_detector_refuses_a_gradient_before_any_compute | refusal |
| processing/os_cfar | the Range-Doppler map | jvp | REF | host-declaration | witwin/radar/processing/cfar.py:285 | tests/test_phase9_processing_wall.py::test_a_cfar_detector_refuses_a_forward_dual_before_any_compute | refusal |
| processing/ca_cfar_1d | the range profile | both | REF | host-declaration | witwin/radar/processing/cfar.py:340 | tests/test_phase9_processing_wall.py::test_the_range_only_detector_refuses_both_modes_before_any_compute | refusal |
| processing/point_cloud | the Range-Doppler map | vjp | REF | host-declaration | witwin/radar/processing/pointcloud.py:199 | tests/test_phase9_processing_wall.py::test_the_point_cloud_refuses_a_gradient_before_any_compute , tests/test_phase9_processing_wall.py::test_the_point_cloud_still_produces_its_one_point | refusal |
| processing/point_cloud | the Range-Doppler map | jvp | REF | host-declaration | witwin/radar/processing/pointcloud.py:199 | tests/test_phase9_processing_wall.py::test_the_point_cloud_refuses_a_forward_dual_before_any_compute | refusal |
| processing/point_cloud | the detection threshold | vjp | REF | host-declaration | witwin/radar/processing/pointcloud.py:199 | tests/test_phase9_processing_wall.py::test_the_point_cloud_refuses_a_live_detection_threshold | refusal |
| processing/_keep_strongest | the energy map | both | REF | host-declaration | witwin/radar/processing/pointcloud.py:278 | tests/test_phase9_processing_wall.py::test_the_peak_selection_refuses_a_gradient_before_any_topk | refusal |
| processing/phase_comparison_aoa | the virtual-antenna column | both | REF | host-declaration | witwin/radar/processing/aoa.py:232 | tests/test_phase9_processing_wall.py::test_an_argmax_angle_estimator_refuses_both_modes_before_any_compute | refusal |
| processing/fft2_aoa | the virtual-antenna column | both | REF | host-declaration | witwin/radar/processing/aoa.py:315 | tests/test_phase9_processing_wall.py::test_an_argmax_angle_estimator_refuses_both_modes_before_any_compute , tests/test_phase9_processing_wall.py::test_the_fft2_route_still_enforces_its_own_contract | refusal |
| processing/DetectionFrame | xyz, velocity_mps, energy | both | REF | host-declaration | witwin/radar/processing/tracking.py:81 | tests/processing/test_tracking.py::test_the_handoff_is_explicitly_non_differentiable_and_refuses_a_gradient, tests/processing/test_tracking.py::test_the_handoff_also_refuses_a_forward_dual_which_it_used_to_accept | refusal |
| processing/any guarded stage | the wall speaks with one voice | both | REF | host-declaration | witwin/radar/ad_contracts.py:71 | tests/test_phase9_processing_wall.py::test_every_wall_refusal_comes_from_the_one_owner | refusal |
| processing/any guarded stage | no result object is produced | both | REF | host-declaration | witwin/radar/ad_contracts.py:71 | tests/test_phase9_processing_wall.py::test_the_no_partial_result_instrument_is_not_vacuous | refusal |

## Higher order: first derivatives only, everywhere

`witwin/radar/ad_contracts.py::first_order_only` decorates all ten registered
`backward` methods in the package and is the only implementation of the rule.
It matches Channel's ADR-043 convention exactly, including the half Channel
DROPPED: `jvp` beside `requires_grad` is a legitimate first-order request under
ADR-038 and is not refused on either side of the boundary.

The decorator wraps `once_differentiable` from OUTSIDE, and the ordering is load
bearing: `once_differentiable` runs the backward body inside `torch.no_grad()`,
so a grad-mode check written inside the body sees grad mode off even under
`create_graph=True`. That is measured, not assumed.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| higher-order/two_way | grad of grad, create_graph=True | vjp | REF | torch-orchestration | witwin/radar/paths/two_way.py:244 | tests/test_phase9_higher_order_refusal.py::test_a_grad_of_grad_request_fails_loudly_and_names_the_owner | refusal |
| higher-order/fmcw | grad of grad, create_graph=True | vjp | REF | torch-orchestration | witwin/radar/synthesis/fmcw_beat.py:149 | tests/test_phase9_higher_order_refusal.py::test_a_grad_of_grad_request_fails_loudly_and_names_the_owner | refusal |
| higher-order/ofdm | grad of grad, create_graph=True | vjp | REF | torch-orchestration | witwin/radar/synthesis/ofdm_cfr.py:122 | tests/test_phase9_higher_order_refusal.py::test_a_grad_of_grad_request_fails_loudly_and_names_the_owner | refusal |
| higher-order/pulsed | grad of grad, create_graph=True | vjp | REF | torch-orchestration | witwin/radar/synthesis/pulsed_echo.py:155 | tests/test_phase9_higher_order_refusal.py::test_a_grad_of_grad_request_fails_loudly_and_names_the_owner | refusal |
| higher-order/sensor_weight | grad of grad, create_graph=True | vjp | REF | torch-orchestration | witwin/radar/sensors/weights.py:413 | tests/test_phase9_higher_order_refusal.py::test_a_grad_of_grad_request_fails_loudly_and_names_the_owner | refusal |
| higher-order/frontend | grad of grad, create_graph=True | vjp | REF | torch-orchestration | witwin/radar/frontend/chain.py:265 | tests/test_phase9_higher_order_refusal.py::test_a_grad_of_grad_request_fails_loudly_and_names_the_owner | refusal |
| higher-order/every boundary | a cotangent carrying a forward tangent | jvp | REF | torch-orchestration | witwin/radar/ad_contracts.py:100 | tests/test_phase9_higher_order_refusal.py::test_a_cotangent_carrying_a_forward_tangent_is_refused | refusal |
| higher-order/every boundary | a cotangent that itself requires grad | vjp | REF | torch-orchestration | witwin/radar/ad_contracts.py:100 | tests/test_phase9_higher_order_refusal.py::test_a_cotangent_that_itself_requires_grad_is_refused | refusal |
| higher-order/every boundary | no partial second-order gradient is left behind | vjp | REF | torch-orchestration | witwin/radar/ad_contracts.py:100 | tests/test_phase9_higher_order_refusal.py::test_no_gradient_survives_the_refusal | refusal |
| higher-order/every boundary | the FIRST-order request over the same graph | vjp | SUP | native-companion | witwin/radar/ad_contracts.py:100 | tests/test_phase9_higher_order_refusal.py::test_the_first_order_request_over_the_same_graph_still_works | adjoint |
| higher-order/all ten backwards | the decorator is applied at every site | both | REF | torch-orchestration | witwin/radar/ad_contracts.py:100 | tests/test_phase9_higher_order_refusal.py::test_every_registered_backward_is_decorated_by_the_one_owner, tests/test_phase9_higher_order_refusal.py::test_the_package_names_no_second_higher_order_rule | refusal |
| higher-order/nested forward levels | a second dual_level | jvp | REF | torch-orchestration | witwin/radar/ad_contracts.py:100 | tests/test_phase9_higher_order_refusal.py::test_nested_forward_levels_stay_torch_owned | refusal |
| higher-order/decorator ordering | once_differentiable cannot replace the check | vjp | REF | torch-orchestration | witwin/radar/ad_contracts.py:128 | tests/test_phase9_higher_order_refusal.py::test_once_differentiable_cannot_replace_the_grad_mode_check | refusal |

## The combined-input matrix: one scenario, one frozen topology

Every row above drives ONE leaf. These drive all eight supported leaves through
the same `PreparedFixedTopology`, in all three AD modes, for each of the three
waveform families. The scenario owner is `tests/support/ad_matrix.py`.

Two facts the scenario is built around, both measured rather than assumed:

- the combined gradient equals the single-leaf gradient **bitwise** on all eight
  leaves and all three waveforms, so the equality is asserted exactly;
- `sum |cube|^2` is exactly invariant under the response phase, because one
  `ScalarRcsResponse` multiplies every composed row. The scenario's loss adds
  `sum Re(cube^2)`, which a global rotation multiplies by `exp(2 j theta)`, and
  a test pins the invariance of the magnitude half so the choice cannot be
  mistaken for decoration.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| chain/fmcw, chain/ofdm, chain/pulsed | all eight supported leaves at once | vjp | SUP | native-companion | tests/support/ad_matrix.py:151 | tests/test_phase9_combined_ad_matrix.py::test_every_supported_leaf_is_live_in_one_combined_backward, tests/test_phase9_combined_ad_matrix.py::test_each_combined_gradient_equals_its_single_leaf_gradient | analytic |
| chain/fmcw, chain/ofdm, chain/pulsed | all eight supported leaves at once, against a difference | vjp | SUP | native-companion | tests/support/ad_matrix.py:151 | tests/test_phase9_combined_ad_matrix.py::test_the_combined_difference_equals_the_sum_of_the_single_leaf_differences | fd |
| reevaluate/prepared | material sigma_e | vjp | SUP | native-companion | tests/support/multi_endpoint_world.py:21 | tests/test_phase9_combined_ad_matrix.py::test_the_conductivity_gradient_matches_a_central_difference | fd |
| rcs/from_rcs | phase_rad to a synthesized cube, all three waveforms | vjp | SUP | torch-orchestration | witwin/radar/scattering/rcs.py:132 | tests/test_phase9_combined_ad_matrix.py::test_the_response_phase_gradient_matches_a_central_difference | fd |
| rcs/from_rcs | phase_rad under a magnitude-only loss | vjp | ZERO | torch-orchestration | witwin/radar/scattering/rcs.py:214 | tests/test_phase9_combined_ad_matrix.py::test_a_magnitude_only_loss_cannot_see_the_response_phase | analytic |
| chain/fmcw, chain/ofdm, chain/pulsed | endpoint positions, full-cube cotangent | both | SUP | native-companion | tests/support/ad_matrix.py:151 | tests/test_phase9_combined_ad_matrix.py::test_the_jvp_is_the_adjoint_of_the_vjp_on_one_frozen_topology | adjoint |
| reevaluate/prepared | out:compact row identity across none, jvp and vjp | both | SUP | native-declared | witwin/radar/propagation/channel_consumer.py:516 | tests/test_phase9_combined_ad_matrix.py::test_the_three_ad_modes_publish_the_same_compact_rows | declaration |
| reevaluate/prepared | out:primal, bitwise across the three modes | both | SUP | native-companion | witwin/radar/propagation/channel_consumer.py:516 | tests/test_phase9_combined_ad_matrix.py::test_the_primal_is_bitwise_identical_in_all_three_ad_modes | analytic |
| reevaluate/prepared | out:every published tensor under ad_mode none | both | DECL | native-declared | witwin/radar/propagation/channel_consumer.py:516 | tests/test_phase9_combined_ad_matrix.py::test_ad_mode_none_publishes_no_graph_and_no_tangent | declaration |
| reevaluate/prepared | out:topology identity, scene-leaf compile against the shared one | both | SUP | native-declared | tests/support/ad_matrix.py:151 | tests/test_phase9_combined_ad_matrix.py::test_the_scene_leaf_scenario_is_the_same_topology_as_the_shared_one | declaration |
| chain/fmcw, chain/ofdm, chain/pulsed | out:the scenario shape the matrix rests on | both | DECL | native-declared | tests/support/ad_matrix.py:151 | tests/test_phase9_combined_ad_matrix.py::test_the_three_waveforms_share_one_frozen_topology | declaration |

## Row validity: a row that stops existing

`row_valid` covers `{los, reflection}` and is the sole authority. A row that
stops existing at the perturbed endpoints is a COMPLETE answer, not a failure:
exact zeros on the payload, an exact zero contribution to every gradient, and an
exactly zero forward tangent. The falsifier for the whole group is that the
waveform kernels gate on `row_valid` rather than on a weight that happens to be
zero, which is shown by overwriting a dead row's payload with a value four
orders of magnitude above every live one and asserting the cube is bitwise
unchanged.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| reevaluate/prepared | out:payload of a dead row | both | ZERO | native-companion | witwin/radar/propagation/channel_consumer.py:587 | tests/test_phase9_row_validity_ad.py::test_a_dead_row_publishes_an_exactly_zero_payload, tests/test_phase9_row_validity_ad.py::test_the_base_configuration_has_no_dead_row_at_all | analytic |
| reevaluate/prepared | endpoint positions through a dead row | vjp | ZERO | native-companion | witwin/radar/propagation/channel_consumer.py:587 | tests/test_phase9_row_validity_ad.py::test_the_dead_rows_contribute_exactly_zero_to_the_gradient | analytic |
| reevaluate/prepared | endpoint positions through a dead row | jvp | ZERO | native-companion | witwin/radar/propagation/channel_consumer.py:587 | tests/test_phase9_row_validity_ad.py::test_a_dead_row_carries_an_exactly_zero_forward_tangent | analytic |
| synthesis/any | out:cube contribution of a dead row | both | ZERO | native-companion | witwin/radar/synthesis/contracts.py:1225 | tests/test_phase9_row_validity_ad.py::test_a_poisoned_dead_row_cannot_change_the_cube | analytic |
| reevaluate/prepared | endpoint positions, frozen replay against a fresh discovery | vjp | SUP | native-companion | witwin/radar/propagation/channel_consumer.py:516 | tests/test_phase9_row_validity_ad.py::test_the_frozen_replay_and_a_fresh_discovery_agree_bit_for_bit | analytic |
| reevaluate/prepared | endpoint positions of a fully occluded site | vjp | ZERO | native-companion | witwin/radar/propagation/channel_consumer.py:587 | tests/test_phase9_row_validity_ad.py::test_a_fully_occluded_site_answers_with_an_exactly_zero_gradient, tests/test_phase9_row_validity_ad.py::test_moving_one_site_kills_exactly_two_rows_and_keeps_nine | analytic |
| join/freeze | a declared site with no outbound row | both | REF | torch-orchestration | witwin/radar/paths/two_way.py:527 | tests/test_phase9_row_validity_ad.py::test_a_fresh_freeze_at_the_occluded_geometry_refuses_instead | refusal |
| reevaluate/prepared | out:a stale discovery-time answer | both | REF | native-companion | witwin/radar/propagation/channel_consumer.py:516 | tests/test_phase9_row_validity_ad.py::test_the_replay_never_answers_with_the_geometry_it_was_frozen_at | analytic |

## Refused tangents, driven through the whole chain

Every row here is asserted to fail BEFORE a cube exists: the three waveform
owners are replaced by counting stand-ins and the count must be exactly zero,
with the instrument calibrated against the same chain running normally.

`RadarEndpointSpec` carries exactly `stable_ids`, `positions_m`,
`polarizations` and `powers_w`, so there is no polarization BASIS a Radar caller
could mark. That refusal is Channel's and lives in Channel's matrix.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| reevaluate/prepared | an ad_mode outside the vocabulary | both | REF | host-declaration | witwin/radar/propagation/channel_consumer.py:525 | tests/test_phase9_refused_tangents.py::test_an_ad_mode_outside_the_vocabulary_is_refused_before_any_cube, tests/test_phase9_refused_tangents.py::test_the_no_cube_instrument_counts_a_real_synthesis | refusal |
| adapter/frequency_offsets_hz | the whole grid as a tensor | both | REF | host-declaration | witwin/radar/propagation/channel_consumer.py:139 | tests/test_phase9_refused_tangents.py::test_a_tensor_band_is_refused_at_adapter_construction | refusal |
| adapter/frequency_offsets_hz | a sequence whose entries are tensors | both | REF | host-declaration | witwin/radar/propagation/channel_consumer.py:139 | tests/test_phase9_refused_tangents.py::test_a_band_whose_ENTRIES_are_tensors_is_refused_too, tests/test_phase9_refused_tangents.py::test_a_host_float_band_is_still_accepted | refusal |
| reevaluate/prepared | sources.powers_w | both | REF | host-declaration | witwin/radar/propagation/channel_consumer.py:516 | tests/test_phase9_refused_tangents.py::test_a_primal_only_endpoint_input_is_refused_in_both_modes | refusal |
| reevaluate/prepared | endpoint polarizations | both | REF | host-declaration | witwin/radar/propagation/channel_consumer.py:516 | tests/test_phase9_refused_tangents.py::test_a_primal_only_endpoint_input_is_refused_in_both_modes | refusal |
| reevaluate/prepared | there is no polarization basis to mark | both | DECL | host-declaration | witwin/radar/propagation/contracts.py:1 | tests/test_phase9_refused_tangents.py::test_the_endpoint_spec_carries_no_polarization_basis_to_mark | declaration |
| adapter/components | diffraction, transmission | both | REF | host-declaration | witwin/radar/propagation/channel_consumer.py:271 | tests/test_phase9_refused_tangents.py::test_an_unfreezable_component_is_refused_before_any_discovery | refusal |
| synthesis/any spec | a tensor waveform scalar, from the chain | both | REF | host-declaration | witwin/radar/host_parameters.py:40 | tests/test_phase9_refused_tangents.py::test_a_tensor_waveform_scalar_is_refused_before_the_spec_exists | refusal |
| higher-order/whole chain | grad of grad over compile to cube | vjp | REF | torch-orchestration | witwin/radar/ad_contracts.py:100 | tests/test_phase9_refused_tangents.py::test_a_grad_of_grad_request_through_the_whole_chain_is_refused, tests/test_phase9_refused_tangents.py::test_the_first_order_request_over_the_same_chain_still_works | refusal |
| higher-order/every boundary | a cotangent carrying a forward tangent | jvp | REF | torch-orchestration | witwin/radar/ad_contracts.py:100 | tests/test_phase9_refused_tangents.py::test_a_cotangent_carrying_a_forward_tangent_is_refused_at_every_boundary | refusal |

## The four chains that had no AD coverage

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| frontend/chain | endpoint positions behind a real cube | vjp | SUP | native-companion | witwin/radar/frontend/chain.py:419 | tests/test_phase9_chain_coverage.py::test_the_noiseless_frontend_scales_the_endpoint_gradient_by_exactly_r_g_squared | analytic |
| frontend/chain | endpoint positions behind a real cube, noise live | vjp | SUP | native-companion | witwin/radar/frontend/chain.py:419 | tests/test_phase9_chain_coverage.py::test_the_noisy_frontend_gradient_matches_a_fourth_order_difference | fd |
| frontend/chain | endpoint positions behind a real cube | jvp | SUP | native-companion | witwin/radar/frontend/chain.py:419 | tests/test_phase9_chain_coverage.py::test_the_frontend_forward_tangent_reproduces_the_reverse_gradient | adjoint |
| frontend/agc | endpoint positions through a global AGC | vjp | ZERO | native-companion | witwin/radar/frontend/chain.py:228 | tests/test_phase9_chain_coverage.py::test_a_global_agc_makes_a_magnitude_loss_exactly_constant | analytic |
| frontend/noise | out:the Philox realisation under AD | vjp | SUP | native-companion | witwin/radar/frontend/chain.py:140 | tests/test_phase9_chain_coverage.py::test_the_same_seed_replays_a_bitwise_identical_gradient, tests/test_phase9_chain_coverage.py::test_the_physics_chain_itself_has_no_noise_to_reproduce | declaration |
| reevaluate_slots/prepared | endpoint positions over a whole slot-major frame | vjp | SUP | native-companion | witwin/radar/propagation/channel_consumer.py:483 | tests/test_phase9_chain_coverage.py::test_a_slot_batched_replay_carries_the_single_frame_gradient_exactly | analytic |
| reevaluate_slots/prepared | endpoint positions over a whole slot-major frame, against a difference | vjp | SUP | native-companion | witwin/radar/propagation/channel_consumer.py:483 | tests/test_phase9_chain_coverage.py::test_the_slot_batched_gradient_matches_a_fourth_order_difference | fd |
| join/_compose_band | endpoint positions to a synthesized wideband cube | vjp | SUP | native-companion | witwin/radar/paths/two_way.py:754 | tests/test_phase9_chain_coverage.py::test_a_wideband_endpoint_gradient_reaches_a_synthesized_cube, tests/test_phase9_chain_coverage.py::test_the_wideband_cube_is_not_the_narrowband_one | fd |
| sensor_weight/round_trip | site position to a synthesized waveform cube | vjp | SUP | native-companion | witwin/radar/sensors/round_trip.py:293 | tests/test_phase9_chain_coverage.py::test_a_sensor_weight_gradient_reaches_a_synthesized_waveform_cube | fd |

## Tape ownership and the budget pins

Every autograd context in the package, its reverse companion, and the structural
statement that pins it. The full ledger - saved tensor names, symbolic byte
formulas, measured bytes, launch counts, backward wall times and, above all,
context LIFETIMES - is `docs/dev/ad-tape-and-budget-ledger.md`. These rows are
the matrix's index into it.

A tape row is `SUP` with `validation = declaration` on purpose. What is being
claimed is that the reverse companion exists, is native, and costs one launch -
a structural fact proved by a structural test. The NUMERICAL correctness of each
of these companions is claimed by the finite-difference rows in the sections
above; a tape row does not restate it.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| tape/two_way | out:join context, 10 saved tensors, one launch each way | vjp | SUP | native-companion | witwin/radar/paths/two_way.py:240 | tests/test_phase9_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/aspect | out:aspect context, 9 saved tensors, one launch each way | vjp | SUP | native-companion | witwin/radar/scattering/aspect.py:159 | tests/test_phase9_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/fmcw_beat | out:beat context, backward saves segment where forward saves offsets | both | SUP | native-companion | witwin/radar/synthesis/fmcw_beat.py:141 | tests/test_phase9_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/ofdm_cfr | out:cfr context, same forward/backward asymmetry | both | SUP | native-companion | witwin/radar/synthesis/ofdm_cfr.py:118 | tests/test_phase9_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/pulsed_echo | out:echo context, same forward/backward asymmetry | both | SUP | native-companion | witwin/radar/synthesis/pulsed_echo.py:151 | tests/test_phase9_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/dirichlet_spectrum | out:chunked and MIMO-linear contexts, two variants | both | SUP | native-companion | witwin/radar/synthesis/dirichlet_spectrum.py:159 | tests/test_phase9_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/sensor_weight | out:weight context, 9 saved tensors, one launch each way | vjp | SUP | native-companion | witwin/radar/sensors/weights.py:405 | tests/test_phase9_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/frontend | out:noise and AGC contexts, two owners in one call | vjp | SUP | native-companion | witwin/radar/frontend/chain.py:173 | tests/test_phase9_backward_budget.py::test_the_frontend_costs_one_backward_launch_per_forward_stage | declaration |
| tape/compose_band | out:tape bytes as a linear law in the band column count | vjp | SUP | native-companion | witwin/radar/paths/two_way.py:754 | tests/test_phase9_backward_budget.py::test_the_band_loop_tape_obeys_its_predicted_linear_law, tests/test_phase9_backward_budget.py::test_the_band_loop_tape_law_holds_at_a_width_it_was_not_fitted_on | declaration |
| tape/any | out:no tape reaches a public result record | both | SUP | torch-orchestration | witwin/radar/propagation/contracts.py:186 | tests/test_phase9_tape_non_leak.py::test_the_leg_batch_and_the_composed_batch_carry_no_tape, tests/test_phase9_tape_non_leak.py::test_the_synthesis_and_sensor_results_carry_no_tape | declaration |
| tape/any | out:no module outside an owner reads a context | both | SUP | native-companion | witwin/radar/ad_contracts.py:104 | tests/test_phase9_tape_non_leak.py::test_every_context_read_sits_inside_a_tape_owner, tests/test_phase9_tape_non_leak.py::test_the_context_scan_is_not_vacuous | declaration |
| reevaluate/prepared | out:ad_companion_launches, ad_tape_bytes | vjp | SUP | native-declared | witwin/radar/propagation/channel_consumer.py:588 | tests/test_phase9_backward_budget.py::test_the_channel_reevaluate_publishes_its_ad_launches_and_tape_bytes, tests/test_phase9_backward_budget.py::test_a_primal_only_reevaluate_builds_no_tape_at_all | declaration |

The wall-time and peak-memory budgets are not matrix rows - they are statements
about a machine rather than about a cell - and live in the ledger's own budget
table with their measured values and headroom.

## Deferred

Each entry names the work and the reason. A deferral is a `REF` or a `DECL` cell
above, never a silent one.

- **`field_direction` on transmission, wedge and coupled diffraction.** Channel
  forwards those families to `rayd::torch`, which publishes no direction
  cotangent or tangent. Channel declares them `DECL` in ADR-043 with a deferral
  to a RayD ADR. Radar never requests those components, so no Radar cell exists. Follow-up owner: Channel, in the RayD ADR that ADR-043 defers to; Radar has no cell to open until it lands.
- **Discovery-route geometry liveness.** The derivative of a discovery result is
  only defined between selection boundaries and Channel deliberately publishes
  no subgradient at one. The supported differentiable route is
  `prepare_fixed_topology` + `reevaluate`, which is the route Radar runs per
  frame. Channel owns the deferral. Follow-up owner: Channel, ADR-043 section on `differentiable_geometry_outputs`.
- **A pose derivative into the compiled scene.** `SmplPoseDeformation` publishes
  a rest `Mesh` and per-frame `DeformationState`s that cross the Core/Channel
  compile boundary, and a pose derivative is not plumbed across it. Whether a
  graph-bearing vertex tensor survives `Mesh` construction and a Channel compile
  is unverified, and a half-working pose gradient would be worse than none, so
  the bridge refuses. The supported differentiable geometry today is a
  `witwin.core.Mesh` vertex tensor. Plumbing a pose derivative across the
  boundary is a separate accepted design. Follow-up owner: Radar `geometry/smpl.py` plus `witwin.core` Mesh construction, as one accepted design covering the compile boundary.
- **A material-only forward tangent.** The adapter's dead-tangent guard requires
  a `delay_s` tangent under `ad_mode='jvp'`, because a dead tangent publishes
  `delay_rate = 0`, which is indistinguishable from a correct stationary answer.
  A permittivity moves the coefficient and not the delay. Loosening the guard to
  accept a coefficient-only tangent is a decision about what `delay_rate = None`
  means to a caller and is not taken here; the cell is available today whenever
  the same call also carries an endpoint tangent. Follow-up owner: Radar `propagation/channel_consumer.py`, whose dead-tangent guard owns the decision about what `delay_rate = None` means to a caller.
- **The LNA voltage gain as a leaf.** It is the one frontend scalar whose
  derivative would be perfectly well defined - a smooth multiplicative factor on
  the whole signal - and it is refused anyway, because the native frontend
  operator carries no tangent or gradient slot for it and no consumer asks for
  one. Adding the slot is a self-contained native change: one extra input on the
  fused phase/thermal/LNA operator's backward and jvp. It is deferred rather
  than done because a device gain is not scene state and nothing in the Phase-9
  acceptance set needs it. Follow-up owner: Radar `frontend/`, one extra tangent and gradient slot on the fused phase/thermal/LNA operator.
- **Waveform-parameter optimisation.** A spec scalar changes the SHAPE of the
  output as often as its value - `num_samples`, `num_subcarriers` and
  `pulse_width_s` all move a sampling grid - and a derivative taken across a
  grid change is not the derivative of a fixed function. Deciding which subset
  is safely continuous, and what a sampling-grid derivative means, is a
  modelling decision with its own ADR rather than a slot to open quietly. Follow-up owner: Radar `synthesis/`, as a new R-ADR deciding which spec scalars are continuous and what a sampling-grid derivative means.
- **A pathwise derivative through the noise realisation.** Every `NoiseSpec`
  scalar parameterises a counter-based Philox draw. A reparameterised noise
  model, where the realisation is a smooth function of a fixed standard normal,
  is the shape that would make these leaves meaningful, and it is a separate
  design with its own accuracy and reproducibility questions. Follow-up owner: Radar `frontend/`, as a reparameterised noise-model R-ADR with its own accuracy and reproducibility evidence.
- **A sensor pattern table as a leaf.** The tables are a resident lookup and the
  kernel interpolates them; the pattern's real contribution to the derivative is
  already carried by the weight, through the positions that decide which angle
  is looked up. Optimising the tabulated VALUES - antenna design rather than
  scene reconstruction - would need a gradient slot on the interpolation and a
  decision about what a derivative at a knot means. Follow-up owner: Radar `sensors/`, gated on a consumer asking for antenna-design gradients rather than scene gradients.
- **The Channel diffraction primal defect.** `evaluate` with
  `components={"diffraction"}` raises `IndexError` inside Channel's own
  enumerated diffraction stage at every AD mode including `none`, because the
  solver scene it builds carries no transmitters or receivers to index. ADR-043
  narrowed `component_ad_modes["diffraction"]` to `{"none"}` so the AD column is
  refused pre-compute rather than advertised and unreachable, and deliberately
  did NOT fix the primal: repairing the plumbing would silently re-open an AD
  column nobody has validated. Radar refuses the component at adapter
  construction and never reaches it. Follow-up owner: Channel, as a primal
  reachability fix with its own evidence, separately from any AD work.

## Acceptance record

The plan's seven Phase-9 acceptance criteria, each mapped to the tests that
prove it, and each marked with what was actually achieved. "Partially proved" is
used where it is true; an honest partial is worth more than a claimed pass.

| criterion | proved by | verdict |
|---|---|---|
| Capability-advertised geometry, material, frequency, target-state, RCS, waveform and receiver jvp/vjp matrix passes; unsupported cells have pre-compute failure tests | this document, 165 rows in four states with no empty test cell, enforced by `tests/test_phase9_capability_matrix.py` (18 tests, including `::test_a_supported_numerical_row_carries_a_real_oracle`, which allowlists the structural `SUP` rows so that a `DECL` row flipped to `SUP` on a declaration-style test fails instead of passing); the refusals themselves by `tests/test_phase9_host_float_refusal.py`, `tests/test_phase9_refused_tangents.py`, `tests/test_phase9_processing_wall.py`, `tests/test_phase9_velocity_leaf_refusal.py`, `tests/test_phase9_smpl_pose_refusal.py`, `tests/test_phase9_sensor_constant_refusal.py` | proved |
| Tests-only finite differences or independent references validate first order | every `SUP` row carries `fd`, `oracle-f64`, `analytic`, `adjoint` or - for one of the 21 allowlisted structural claims - `declaration`; `tests/test_phase9_capability_matrix.py::test_a_supported_row_is_never_justified_by_a_refusal`, `::test_a_supported_numerical_row_carries_a_real_oracle` and `::test_the_structural_allowlist_has_no_stale_entry` enforce the vocabulary; no production finite difference exists, pinned by `tests/test_phase6_no_torch_physics.py` | proved |
| Primal, jvp and vjp share compact path identity, row mapping and numerical convention | `tests/test_phase9_combined_ad_matrix.py::test_the_three_ad_modes_publish_the_same_compact_rows`, `::test_the_primal_is_bitwise_identical_in_all_three_ad_modes`, `::test_the_jvp_is_the_adjoint_of_the_vjp_on_one_frozen_topology`, `::test_the_three_waveforms_share_one_frozen_topology`, all on ONE frozen topology per scenario | proved |
| Topology discovery, hard pruning, ADC, CFAR and tracking AD requests fail before any partial result | `tests/test_phase9_processing_wall.py` (25 tests, with a `_ComputeWatch` instrument that measures that nothing was computed and is calibrated against the same stages running normally), `tests/test_phase9_refused_tangents.py::test_an_unfreezable_component_is_refused_before_any_discovery`, `::test_a_primal_only_endpoint_input_is_refused_in_both_modes` | proved |
| No production finite difference, detach, silent zero-gradient or Torch physics fallback | `tests/test_phase6_no_torch_physics.py` (12 tests, extended this phase to the guard owners and to `scattering/rcs.py` by equality), `tests/test_phase9_row_validity_ad.py::test_a_poisoned_dead_row_cannot_change_the_cube` for the zero-gradient half, `tests/test_phase4_import_boundary.py` for the host-observation half | proved |
| Tape does not leak into public results or get parsed across owners | `tests/test_phase9_tape_non_leak.py` (8 tests), with a calibration that plants a context in a record and checks the walker objects; scan limits stated in the module docstring | proved |
| Forward and backward time, launch and memory budgets met | `tests/test_phase9_backward_budget.py` (21 tests, three of which parse `docs/dev/ad-tape-and-budget-ledger.md` and pin its bytes formula against a live measurement and its budget tables against this module's own constants) and `docs/dev/ad-tape-and-budget-ledger.md`; the pre-existing forward pins in `tests/test_phase8_pipeline_budget.py` and `tests/test_phase5_budget.py` are unchanged and pass on an idle device, with the measured cold-clock caveat recorded in the ledger rather than absorbed into a wider factor | partially proved |

**Why the last criterion is "partially proved" and not "proved".** Every budget
this phase introduced is measured, pinned and green. The two pre-existing
Phase-8 wall-time pins are also green on an idle device - measured three
consecutive times while closing the phase, at 2.7421-2.9225 ms against a 2.8990
ms budget and 4.2097-4.9246 ms against a 5.0440 ms budget - but the first run of
a cold session lands within one percent of the first budget, which is inside the
measurement's own noise rather than comfortably clear of it. Calling that
"proved" would overstate it. The number to fix is the device state, not the
factor; the ledger records the measurement and the reason.
