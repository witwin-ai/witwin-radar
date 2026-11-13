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
`tests/test_import_boundary.py::test_the_consumer_contract_is_the_version_this_adapter_was_built_against`.
The authority is Channel ADR-043 and consumer `CONTRACT_VERSION` 7 (Channel
0.6.0). Version 7 added the reflection discovery source and the completeness
flag; every row mirrored below reads the same in Channel 0.6.0's propagation AD
capability matrix as it did at version 6.

A forward-mode tangent on mesh vertices compiles and propagates on Channel 0.6.0 (tag 699fd8c): its
finite-vertex check reads the primal, so the `aminmax` reduction no longer sees the dual. The release built
from the earlier 0.6.0 tag refused it at compile; that build was never published.

Radar requests exactly one response, `scalar_transport`, with
`components <= {los, reflection}`, always on the fixed-topology route. Every
mirrored row below is therefore a row Radar can actually reach.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| reevaluate/prepared | endpoint positions | both | SUP | native-companion | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_single_site_end_to_end.py::test_reverse_mode_loss_gradients_match_the_oracle, tests/test_reflection_ad.py::test_reverse_mode_site_gradients_match_finite_differences | fd |
| reevaluate/prepared | mesh vertices | vjp | SUP | native-companion | tests/support/multi_endpoint_world.py::make_scene | tests/test_scene_leaf_ad.py::test_a_mesh_vertex_gradient_reaches_a_synthesized_fmcw_loss | fd |
| reevaluate/prepared | mesh vertices | jvp | SUP | native-declared | tests/support/multi_endpoint_world.py::make_scene | tests/test_scene_leaf_ad.py::test_a_forward_tangent_on_the_wall_matches_the_reverse_gradient | adjoint |
| reevaluate/prepared | mesh vertices, in-plane components | both | ZERO | native-companion | tests/support/multi_endpoint_world.py::make_scene | tests/test_scene_leaf_ad.py::test_the_in_plane_vertex_gradient_is_exactly_zero_and_that_is_correct | analytic |
| reevaluate/prepared | material eps_r | vjp | SUP | native-companion | tests/support/multi_endpoint_world.py::make_scene | tests/test_scene_leaf_ad.py::test_a_material_permittivity_gradient_reaches_a_synthesized_fmcw_loss | fd |
| reevaluate/prepared | material eps_r, beside an endpoint tangent | jvp | SUP | native-companion | tests/support/multi_endpoint_world.py::make_scene | tests/test_scene_leaf_ad.py::test_vertices_permittivity_and_endpoints_are_live_in_one_call | fd |
| reevaluate/prepared | out:field_direction | both | SUP | native-companion | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_aspect_direction_ad.py::test_the_frozen_leg_publishes_a_graph_bearing_field_direction | declaration |
| discovery | out:field_direction | both | DECL | native-declared | witwin/radar/channel.py::ChannelPropagationAdapter.refreeze | tests/test_import_boundary.py::test_the_consumer_contract_is_the_version_this_adapter_was_built_against | declaration |
| reevaluate/prepared | sources.powers_w, endpoint polarizations | both | REF | host-declaration | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_channel_adapter.py::test_differentiable_power_is_rejected_before_any_native_work, tests/test_channel_adapter.py::test_differentiable_polarization_is_rejected | refusal |
| any | component diffraction, component transmission | both | REF | host-declaration | witwin/radar/channel.py::compile_scene | tests/test_channel_adapter.py::test_adapter_rejects_unfreezable_components | refusal |

## Propagation-adjacent Radar cells

### Aspect scatter response (`witwin/radar/scattering.py`)

`field_direction` is the only geometry input this response has, so a detached
direction takes its endpoint gradient to exactly zero. That was the state of the
tree before ADR-043 and it is now the falsifier.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| aspect/evaluate_rows | endpoint positions through dir_in (reflection row) | vjp | SUP | native-companion | witwin/radar/scattering.py::_AspectResponse | tests/test_aspect_direction_ad.py::test_a_reverse_aspect_gradient_reaches_the_site_positions, tests/test_aspect_direction_ad.py::test_a_reverse_aspect_gradient_reaches_the_transmitter_through_a_reflection_row | fd |
| aspect/evaluate_rows | endpoint positions through dir_in (line-of-sight row) | vjp | SUP | native-companion | witwin/radar/scattering.py::_AspectResponse | tests/test_aspect_direction_ad.py::test_a_reverse_aspect_gradient_reaches_the_site_positions | fd |
| aspect/evaluate_rows | endpoint positions through dir_out | vjp | SUP | native-companion | witwin/radar/scattering.py::_AspectResponse | tests/test_aspect_direction_ad.py::test_a_reverse_aspect_gradient_reaches_the_site_positions | fd |
| aspect/evaluate_rows | endpoint positions through dir_in and dir_out | jvp | SUP | native-companion | witwin/radar/scattering.py::_AspectResponse | tests/test_aspect_direction_ad.py::test_a_forward_tangent_on_the_sites_reaches_the_aspect_response | adjoint |
| aspect/compose/fmcw | endpoint positions to a synthesized cube | both | SUP | native-companion | witwin/radar/scattering.py::_AspectResponse | tests/test_aspect_direction_ad.py::test_the_aspect_direction_gradient_reaches_a_synthesized_fmcw_loss, tests/test_aspect_direction_ad.py::test_the_direction_term_is_load_bearing_in_the_synthesized_loss | fd |
| aspect/evaluate_rows | outbound reflection segment endpoints | both | SUP | native-companion | witwin/radar/scattering.py::_ScatterDirection | tests/test_multipath_directions.py::test_native_scatter_direction_primal_vjp_and_jvp, tests/test_multipath_directions.py::test_outbound_reflection_aspect_uses_departure_and_carries_gradients | fd |
| aspect/evaluate_rows | missing outbound arrival segment | both | REF | host-declaration | witwin/radar/scattering.py::AspectScatterResponse | tests/test_scatter_response_kernel.py::test_a_reflected_outbound_row_reads_its_departure_off_the_arrival_segment | refusal |

### SMPL authoring (`witwin/radar/smpl.py`)

R-ADR-052 makes the pose, the shape and the body transform reverse-mode leaves:
every `DeformationState` the bridge publishes holds the posed vertices as a
differentiable expression of them, and the facet geometry, the facet response
and Channel's fixed-topology replay carry the vertex derivative. The end-to-end
evidence runs on a synthetic 24-joint linear-blend-skinning rig through the same
code path (`tests/support/synthetic_smpl.py`); the real-model tests skip without
the SMPL pickles. Forward mode stays refused: `velocity_at` opens its own
forward level over the pose, and PyTorch's forward AD cannot nest a caller's
level around it. Mesh vertices themselves take a forward tangent.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| smpl/SmplPoseDeformation | body pose to the traced rows | vjp | SUP | native-companion | witwin/radar/smpl.py::SmplPoseDeformation | tests/test_smpl_pose_leaves.py::test_the_pose_gradient_matches_central_differences | fd |
| smpl/SmplPoseDeformation | body shape to the traced rows | vjp | SUP | native-companion | witwin/radar/smpl.py::SmplPoseDeformation | tests/test_smpl_pose_leaves.py::test_the_shape_gradient_matches_central_differences | fd |
| smpl/SmplPoseDeformation | body position to the traced rows | vjp | SUP | native-companion | witwin/radar/smpl.py::SmplPoseDeformation.rest_mesh | tests/test_smpl_pose_leaves.py::test_the_body_position_is_a_leaf_too | fd |
| smpl/SmplPoseDeformation | body pose, body shape | jvp | REF | torch-orchestration | witwin/radar/smpl.py::SmplPoseDeformation | tests/test_smpl_pose_leaves.py::test_a_pose_tangent_is_refused_before_any_compile, tests/test_smpl_pose_refusal.py::test_a_pose_carrying_a_forward_dual_is_refused_too | refusal |
| smpl/rest_mesh | body position, body rotation | jvp | REF | torch-orchestration | witwin/radar/smpl.py::SmplPoseDeformation.rest_mesh | tests/test_smpl_pose_refusal.py::test_a_transform_tangent_is_refused_by_rest_mesh | refusal |
| smpl/SmplPoseDeformation | pose_rate | both | REF | torch-orchestration | witwin/radar/smpl.py::SmplPoseDeformation | tests/test_smpl_pose_leaves.py::test_a_pose_rate_leaf_is_still_refused, tests/test_smpl_pose_refusal.py::test_a_pose_rate_that_requires_grad_is_refused_by_adr_038 | refusal |
| smpl/velocity_at | pose_rate as a tangent direction | jvp | SUP | torch-orchestration | witwin/radar/smpl.py::SmplPoseDeformation.velocity_at | tests/test_smpl_pose_refusal.py::test_the_analytic_vertex_velocity_still_comes_from_the_pose_rate | fd |
| legacy-scene/SMPLBody | pose | vjp | SUP | torch-orchestration | witwin/radar/smpl.py::SMPLBody._evaluate | tests/test_smpl_pose_refusal.py::test_the_smpl_body_itself_still_publishes_a_pose_gradient | declaration |

### Radar pose (`witwin/radar/radar.py`)

`Radar.position`, `look_at` and `up` are leaves (R-ADR-052): every element
position and the pattern frame are differentiable expressions of them, carried
by Channel's replay, the two-way join and the sensor-weight frame slot. A
polarization the pose would turn (`"up"`, `"right"`, a pose trajectory's
rotation) is refused for a pose leaf, because Channel's replay takes endpoint
polarizations as constants; a world-vector polarization does not turn and the
derivative is then complete.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| radar/pose | position, look_at, up to the simulated cube | both | SUP | native-companion | witwin/radar/radar.py::Radar._place_antennas | tests/test_radar_pose_leaves.py::test_a_pose_leaf_matches_central_differences_in_both_modes | fd |
| radar/pose | a pose trajectory's translation to the simulated cube | vjp | SUP | native-companion | witwin/radar/simulation.py::WorldHalf._rotation_at | tests/test_radar_pose_leaves.py::test_a_pose_trajectory_translation_is_a_leaf | fd |
| radar/pose | a pose leaf with an "up" or "right" polarization | both | REF | host-declaration | witwin/radar/radar.py::Radar._require_pose_leaf_polarization | tests/test_radar_pose_leaves.py::test_a_pose_leaf_with_a_pose_derived_polarization_is_refused | refusal |
| radar/pose | a polarization leaf | both | REF | host-declaration | witwin/radar/radar.py::Radar._place_antennas | tests/test_radar_pose_leaves.py::test_a_polarization_leaf_is_refused | refusal |
| radar/pose | a pose trajectory's rotation | both | REF | host-declaration | witwin/radar/simulation.py::WorldHalf._rotation_at | tests/test_radar_pose_leaves.py::test_a_pose_trajectory_rotation_leaf_is_refused | refusal |
| radar/pose | a pose leaf under a grad mode that does not carry it | both | REF | host-declaration | witwin/radar/radar.py::Radar._require_pose_ad_mode | tests/test_radar_pose_leaves.py::test_a_pose_leaf_in_the_wrong_ad_mode_is_refused_before_any_work | refusal |

### Leg endpoints and scene leaves under the verb's `ad_mode` (`witwin/radar/policy.py`)

A scatter-site position is an endpoint of both legs, and a wall's vertices and
material enter every row that meets the wall. Channel's fixed-topology replay
carries those derivatives only in the mode the verb asks for (`vjp` for a
gradient, `jvp` for a forward tangent); under `none` every row is discovery's
and carries none, so a marked input came back with no derivative or with the
Radar-side stages' part alone. `policy.require_leg_ad_mode` refuses it, the
same rule as the pose rows above. A site position is admitted where it is
formed: a `PointTargets` tensor when the session's site policy is built, before
any compile; a trajectory's output, a structure anchor and a mesh target's world
vertices when the frame resolves them, before the facet operator or any
discovery. Scene leaves are admitted on each compiled scene, before any
discovery, unless the request is line-of-sight rows alone, which read a wall
only through occlusion. Radar-side leaves (cross section, patterns, power, LNA
gain, coupling) are unaffected and differentiate under `none`.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| sites/explicit | PointTargets.positions under none, a gradient under jvp or a tangent under vjp | both | REF | host-declaration | witwin/radar/policy.py::require_leg_ad_mode | tests/test_site_position_ad_mode.py::test_a_marked_point_target_position_is_refused_before_any_work, tests/test_site_position_ad_mode.py::test_every_site_source_refuses_a_derivative_its_mode_does_not_carry | refusal |
| sites/trajectory | a PointTargets trajectory's output under a mode that does not carry it | both | REF | host-declaration | witwin/radar/policy.py::require_leg_ad_mode | tests/test_site_position_ad_mode.py::test_a_marked_point_target_trajectory_is_refused_before_any_discovery, tests/test_site_position_ad_mode.py::test_every_site_source_refuses_a_derivative_its_mode_does_not_carry | refusal |
| sites/structure_anchor | a StructureTargets anchor under a mode that does not carry it | both | REF | host-declaration | witwin/radar/policy.py::require_leg_ad_mode | tests/test_site_position_ad_mode.py::test_every_site_source_refuses_a_derivative_its_mode_does_not_carry | refusal |
| sites/facets | MeshTargets world vertices under a mode that does not carry them | both | REF | host-declaration | witwin/radar/facets.py::FacetSites.resolve | tests/test_site_position_ad_mode.py::test_marked_mesh_target_vertices_are_refused_before_the_facet_operator | refusal |
| scene/compiled | wall vertices and material tables under a mode that does not carry them, with a non-line-of-sight component | both | REF | host-declaration | witwin/radar/simulation.py::WorldHalf._compile | tests/test_site_position_ad_mode.py::test_marked_wall_vertices_are_refused_before_any_discovery, tests/test_site_position_ad_mode.py::test_a_marked_wall_material_under_none_is_refused_too | refusal |

## Synthesis, frontend and sensor cells

### The host-float rule (`witwin/radar/policy.py`)

Every configuration scalar on every waveform spec and every frontend stage spec
is a host float and refuses a tensor at construction. The rule is on the TYPE
rather than on `requires_grad`, because a tensor that does not require grad
today is the input that starts requiring grad tomorrow, and `float()` on a
device tensor is a host synchronisation as well. All 85 spec construction sites
in the tree were checked and the whole suite re-run, default and `--gpu`: no
caller was displaced.

Each spec's row covers every scalar it declares; the per-field cases are the
parametrized ids inside the cited test.

The port impedance, the LNA gain and the seed no longer have records of their
own: each was one number, so each is a field of `FrontendSpec` and of the radar
that builds it. The impedance and the seed are refused at the same
host-declaration boundary. The LNA gain and the transmit power are leaves
(R-ADR-052): smooth amplitude factors whose derivative is exact - the chain
runs at the host value and a unit factor carries the leaf.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| frontend/FrontendSpec | impedance | both | REF | host-declaration | witwin/radar/frontend.py::FrontendSpec | tests/test_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| frontend/FrontendSpec | lna, through noise, compression, IQ and the DC offset | both | SUP | torch-orchestration | witwin/radar/frontend.py::FrontendChain.apply | tests/test_receiver_gain_leaves.py::test_the_lna_gain_is_a_leaf_of_the_receive_chain | fd |
| radar/power | transmit power to the simulated cube | both | SUP | native-companion | witwin/radar/radar.py::Radar._transmit_amplitude_scale | tests/test_receiver_gain_leaves.py::test_the_transmit_power_is_a_leaf_with_the_amplitude_law | fd |
| radar/power | transmit power to the leakage rows | both | SUP | torch-orchestration | witwin/radar/radar.py::Radar._leakage_cube | tests/test_receiver_gain_leaves.py::test_the_transmit_power_reaches_the_leakage_rows | fd |
| frontend/coupling | Coupling.tx, Coupling.rx complex matrices | both | SUP | torch-orchestration | witwin/radar/frontend.py::FrontendChain._couple | tests/test_receiver_gain_leaves.py::test_a_coupling_matrix_is_a_leaf | fd |
| frontend/Noise | figure, antenna_temperature, bandwidth, phase_density, phase_offset, phase_sample_rate | both | REF | host-declaration | witwin/radar/frontend.py::Noise | tests/test_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| frontend/Agc | target_rms, min_gain, max_gain | both | REF | host-declaration | witwin/radar/frontend.py::Agc | tests/test_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| frontend/Adc | bits, full_scale | both | REF | host-declaration | witwin/radar/frontend.py::Adc | tests/test_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| frontend/FrontendSpec | seed | both | REF | host-declaration | witwin/radar/frontend.py::FrontendSpec | tests/test_host_float_refusal.py::test_the_seed_is_refused_by_its_own_older_type_rule | refusal |
| synthesis/FmcwSpec | every waveform scalar | both | REF | host-declaration | witwin/radar/synthesis/assembly.py::FmcwSpec | tests/test_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| synthesis/OfdmCfrSpec | every waveform scalar | both | REF | host-declaration | witwin/radar/synthesis/assembly.py::OfdmSpec | tests/test_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| synthesis/PulsedEchoSpec | every waveform scalar | both | REF | host-declaration | witwin/radar/synthesis/assembly.py::PulsedSpec | tests/test_host_float_refusal.py::test_every_configuration_scalar_refuses_a_marked_tensor | refusal |
| any spec | an unmarked tensor, the tomorrow case | both | REF | host-declaration | witwin/radar/policy.py::require_host_float | tests/test_host_float_refusal.py::test_every_configuration_scalar_refuses_an_unmarked_tensor_too | refusal |
| any spec | a forward dual | jvp | REF | host-declaration | witwin/radar/policy.py::require_host_float | tests/test_host_float_refusal.py::test_a_forward_dual_is_refused_as_well | refusal |

### Sensor weight (`witwin/radar/sensors.py`)

Nine differentiable inputs - the six position, intensity and weight inputs, the
pattern frame and the two packed element patterns (table values and peak
amplitudes, R-ADR-052) - and frozen velocities and fixed lengths. The frozen
constants fail at `SensorWeightGeometry` construction, before `validate` and
before any launch.

The former cached MIMO caller exposed a silent velocity-gradient hole. The
breaking cutover removed that route; current velocity support and refusals are
represented only by the rows below.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| sensor_weight/evaluate | tx_pos, rx_pos, site_in, site_out | jvp | SUP | native-companion | witwin/radar/sensors.py::evaluate_sensor_weights | tests/test_sensor_weight.py::test_the_jvp_matches_a_central_finite_difference | fd |
| sensor_weight/evaluate | tx_pos, rx_pos, site_in, site_out | vjp | SUP | native-companion | witwin/radar/sensors.py::evaluate_sensor_weights | tests/test_sensor_weight.py::test_the_vjp_is_the_adjoint_of_the_jvp | adjoint |
| sensor_weight/evaluate | intensity, weight real and imaginary parts | both | SUP | native-companion | witwin/radar/sensors.py::evaluate_sensor_weights | tests/test_sensor_weight.py::test_the_vjp_is_the_adjoint_of_the_jvp | adjoint |
| sensor_weight/evaluate | antenna position reduction order | vjp | SUP | native-companion | witwin/radar/sensors.py::evaluate_sensor_weights | tests/test_sensor_weight.py::test_the_antenna_gradient_is_a_deterministic_reduction | declaration |
| sensor_weight/SensorWeightGeometry | tx_velocity, rx_velocity, site_velocity | both | REF | host-declaration | witwin/radar/sensors.py::SensorWeightGeometry | tests/test_sensor_constant_refusal.py::test_a_marked_frozen_geometry_field_is_refused, tests/test_sensor_constant_refusal.py::test_a_velocity_refusal_names_the_reason_a_velocity_is_never_a_leaf | refusal |
| sensor_weight/SensorWeightGeometry | fixed_length_m | vjp | REF | host-declaration | witwin/radar/sensors.py::SensorWeightGeometry | tests/test_sensor_constant_refusal.py::test_a_marked_frozen_geometry_field_is_refused | refusal |
| sensor_weight/SensorWeightGeometry | fixed_length_m | jvp | REF | host-declaration | witwin/radar/sensors.py::SensorWeightGeometry | tests/test_sensor_constant_refusal.py::test_a_dual_carrying_frozen_geometry_field_is_refused | refusal |
| sensor_weight/evaluate | pattern_frame | both | SUP | native-companion | witwin/radar/sensors.py::_SensorWeight | tests/test_pattern_leaves.py::test_the_pattern_frame_matches_central_differences_in_both_modes | fd |
| sensor_weight/evaluate | packed table values and peak gain, every kind, both sides | both | SUP | native-companion | witwin/radar/sensors.py::_SensorWeight | tests/test_pattern_leaves.py::test_table_values_and_gain_match_central_differences_in_both_modes | fd |
| sensor_weight/evaluate | table and frame reduction order | vjp | SUP | native-companion | witwin/radar/sensors.py::_keyed_parameter_gradients | tests/test_pattern_leaves.py::test_the_parameter_reduction_is_deterministic | declaration |
| sensor_weight/Pattern | Pattern.table / field_table values to the simulated cube | both | SUP | native-companion | witwin/radar/sensors.py::Pattern | tests/test_pattern_leaves.py::test_a_pattern_table_leaf_reaches_the_simulated_cube | fd |
| sensor_weight/Pattern | Pattern.gain to the simulated cube | vjp | SUP | native-companion | witwin/radar/sensors.py::Pattern | tests/test_pattern_leaves.py::test_a_pattern_gain_leaf_reaches_the_simulated_cube | fd |
| sensor_weight/Pattern | an angle axis | both | REF | host-declaration | witwin/radar/sensors.py::Pattern | tests/test_pattern_leaves.py::test_a_host_pattern_packs_bit_identically_and_a_tensor_axis_is_refused | refusal |
| sensor_weight/round_trip | a marked velocity from a production entry point | both | REF | host-declaration | witwin/radar/sensors.py::SensorWeightGeometry | tests/test_sensor_constant_refusal.py::test_a_marked_field_on_the_production_geometry_is_still_refused, tests/test_sensor_constant_refusal.py::test_the_production_geometry_carries_no_marked_frozen_field | refusal |
| sensor_weight/evaluate | out:pattern_gain | both | DECL | native-declared | witwin/radar/sensors.py::evaluate_sensor_weights | tests/test_sensor_weight.py::test_the_kernel_pattern_gain_equals_the_torch_pattern_gain | declaration |
| sensor_weight/round_trip | site positions to a composed round-trip weight | vjp | SUP | native-companion | witwin/radar/sensors.py::RoundTripPatternStage.apply | tests/test_antenna_pattern_route.py::test_the_reverse_gradient_of_a_site_matches_a_central_difference | fd |
| sensor_weight/round_trip | transmit element positions to a composed round-trip weight | vjp | SUP | native-companion | witwin/radar/sensors.py::RoundTripPatternStage.apply | tests/test_antenna_pattern_route.py::test_the_reverse_gradient_of_a_transmit_element_matches_a_central_difference | fd |
| sensor_weight/round_trip | site positions to a composed round-trip weight | jvp | SUP | native-companion | witwin/radar/sensors.py::RoundTripPatternStage.apply | tests/test_antenna_pattern_route.py::test_the_forward_tangent_of_a_site_matches_a_central_difference | fd |

The last three rows are Phase 11's PRODUCTION route for this family. Until
`RoundTripPatternStage` existed the family's only importer was
`sensors/legacy_paths.py`, on the Dirichlet route Phase 11 deleted, so every
row above described a capability no production entry point could reach. The
stage reaches all three companions through `evaluate_sensor_weights` and adds
no `torch.autograd.Function` of its own, so it changed no tape owner and no
boundary budget.

### Scatter response (`witwin/radar/scattering.py`)

The one capability ADD of this stage. A radar cross section is scene state and
the canonical inverse-design leaf, which is why it sits on the other side of the
host-float rule from every spec scalar above. The `sqrt` is result construction,
off the per-path loop, so `torch-orchestration` is the legal mechanism; every
per-path product downstream of it is still a native kernel.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| rcs/from_rcs | sigma_m2 to a synthesized FMCW loss | vjp | SUP | torch-orchestration | witwin/radar/scattering.py::ScalarRcsResponse.from_rcs | tests/test_rcs_sigma_ad.py::test_the_cross_section_gradient_equals_the_exact_closed_form, tests/test_rcs_sigma_ad.py::test_the_cross_section_gradient_matches_a_central_difference | analytic |
| rcs/from_rcs | sigma_m2 through the square-root law | vjp | SUP | torch-orchestration | witwin/radar/scattering.py::ScalarRcsResponse.from_rcs | tests/test_rcs_sigma_ad.py::test_the_gradient_is_the_amplitude_gradient_through_the_square_root | analytic |
| rcs/from_rcs | sigma_m2 to a synthesized FMCW loss | jvp | SUP | torch-orchestration | witwin/radar/scattering.py::ScalarRcsResponse.from_rcs | tests/test_rcs_sigma_ad.py::test_the_forward_tangent_matches_the_reverse_gradient | adjoint |
| rcs/from_values | amplitude, phase_rad | both | SUP | torch-orchestration | witwin/radar/scattering.py::ScalarRcsResponse.from_values | tests/test_single_site_end_to_end.py::test_reverse_mode_loss_gradients_match_the_oracle, tests/test_single_site_end_to_end.py::test_forward_mode_loss_tangent_matches_the_oracle | fd |
| rcs/from_rcs | sigma_m2 with requires_grad=True as well | both | REF | torch-orchestration | witwin/radar/scattering.py::ScalarRcsResponse.from_rcs | tests/test_rcs_sigma_ad.py::test_marking_the_derived_amplitude_as_well_is_refused | refusal |
| rcs/from_rcs | a rank-2 sigma_m2 | both | REF | torch-orchestration | witwin/radar/scattering.py::ScalarRcsResponse.from_rcs | tests/test_rcs_sigma_ad.py::test_a_matrix_cross_section_is_refused | refusal |


### Mesh-target facets and tabulated responses (`witwin/radar/facets.py`, `witwin/radar/scattering.py`)

The facet and table responses are native per-row families whose companions
are exact forward-mode jets (R-ADR-042, R-ADR-045). A facet material's
front-roughness rms and every layer's thickness, permittivity and permeability
are leaves (R-ADR-052); the material model, the polarization, the diffuse seed
and a response table select the law and are not. A forward tangent on the vertices of a mesh that is also in the scene
reaches both the facet families and Channel's compiled scene.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| scatter/facet_response | leg directions, facet corners, float64 site positions (2x2 matrix per row) | vjp | SUP | native-companion | witwin/radar/scattering.py::_FacetResponse | tests/test_facet_scatter_kernel.py::test_both_ad_modes_match_finite_differences_and_each_other | fd |
| scatter/facet_response | leg directions, facet corners, float64 site positions (2x2 matrix per row) | jvp | SUP | native-companion | witwin/radar/scattering.py::_FacetResponse | tests/test_facet_scatter_kernel.py::test_both_ad_modes_match_finite_differences_and_each_other | adjoint |
| facets/facet_geometry | world vertices | both | SUP | native-companion | witwin/radar/facets.py::_FacetGeometry | tests/test_facet_scatter_kernel.py::test_facet_geometry_and_its_two_companions | adjoint |
| mesh/trace | authored mesh vertices to the composed rows | vjp | SUP | native-companion | witwin/radar/facets.py::_FacetGeometry | tests/test_mesh_targets.py::test_the_vertex_gradient_matches_finite_differences | fd |
| scatter/facet_response | material tables: rms, layer thickness, Re eps, Im eps, mu | both | SUP | native-companion | witwin/radar/scattering.py::_FacetResponse | tests/test_facet_material_leaves.py::test_the_material_tables_match_central_differences_in_both_modes | fd |
| mesh/trace | Core material eps_r, sigma_e, thickness_m, roughness rms to the composed rows | both | SUP | native-companion | witwin/radar/facets.py::_live_material_tables | tests/test_facet_material_leaves.py::test_a_core_material_leaf_reaches_the_traced_rows_in_both_modes | fd |
| mesh/trace | a roughness rms leaf at exactly zero | both | REF | host-declaration | witwin/radar/facets.py::_material_rows | tests/test_facet_material_leaves.py::test_a_roughness_leaf_at_exactly_zero_is_refused | refusal |
| scatter/table_response | leg directions, table axis | both | SUP | native-companion | witwin/radar/scattering.py::_TableResponse | tests/test_target_fluctuation_and_tables.py::test_the_table_operator_matches_finite_differences_and_its_adjoint | fd |
### FMCW beat synthesis (`witwin/radar/synthesis/fmcw.py`)

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| synthesis/fmcw_beat | tau_rt | jvp | SUP | native-companion | witwin/radar/synthesis/fmcw.py::_FmcwSynthesis | tests/test_fmcw_variable_jvp.py::test_the_jvp_of_each_differentiable_input_matches_a_central_difference | oracle-f64 |
| synthesis/fmcw_beat | tau_rate | jvp | SUP | native-companion | witwin/radar/synthesis/fmcw.py::_FmcwSynthesis | tests/test_fmcw_variable_jvp.py::test_the_jvp_of_each_differentiable_input_matches_a_central_difference, tests/test_fmcw_variable_jvp.py::test_the_rate_derivative_is_not_the_delay_derivative_scaled_by_slow_time | analytic |
| synthesis/fmcw_beat | weight real and imaginary parts | jvp | SUP | native-companion | witwin/radar/synthesis/fmcw.py::_FmcwSynthesis | tests/test_fmcw_variable_jvp.py::test_the_jvp_of_each_differentiable_input_matches_a_central_difference | oracle-f64 |
| synthesis/fmcw_beat | tau_rt, tau_rate, weight | vjp | SUP | native-companion | witwin/radar/synthesis/fmcw.py::_FmcwSynthesis | tests/test_fmcw_beat_ad.py::test_native_vjp_matches_the_oracle, tests/test_fmcw_beat_ad.py::test_multi_segment_vjp_matches_the_oracle | oracle-f64 |
| synthesis/fmcw_beat | tau_rate at the first chirp | jvp | ZERO | native-companion | witwin/radar/synthesis/fmcw.py::_FmcwSynthesis | tests/test_fmcw_variable_jvp.py::test_a_rate_only_tangent_is_not_the_zero_tangent | analytic |
| synthesis/fmcw_observation | refreshed delay and complex weight | both | SUP | native-companion | witwin/radar/synthesis/fmcw.py::_FmcwObservations | tests/test_fmcw_observations.py::test_ragged_observations_match_complex_oracle_and_ad | oracle-f64 |
| synthesis/fmcw_observation | ADC time | both | REF | native-companion | witwin/radar/synthesis/fmcw.py::synthesize_fmcw_observations | tests/test_fmcw_observations.py::test_empty_observations_and_time_derivative_refusal | refusal |
| synthesis/fmcw_adaptive | compact probe delay and complex transfer | both | SUP | native-companion | witwin/radar/synthesis/fmcw.py::_AdaptiveFmcw | tests/test_adaptive_fmcw_fusion.py::test_compact_fmcw_matches_complex_oracle_and_first_order_ad | oracle-f64 |
| synthesis/fmcw_adaptive | partition basis and ADC clock | both | REF | native-companion | witwin/radar/synthesis/fmcw.py::synthesize_adaptive_fmcw | tests/test_adaptive_fmcw_fusion.py::test_compact_fmcw_empty_rows_and_derivative_refusals | refusal |

### End-to-end waveform chains (`tests/support/waveform_chains.py`)

The FMCW chain has been covered since Phase 4. These are the OFDM and pulsed
equivalents: a Core endpoint leaf through the Channel consumer, the two-way
join, the scatter response and the waveform kernel to a scalar loss, in both
modes, against a fourth-order difference of the whole production chain.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| chain/ofdm | site, transmitter and receiver positions | vjp | SUP | native-companion | witwin/radar/synthesis/ofdm.py::synthesize_ofdm | tests/test_waveform_chain_ad.py::test_the_endpoint_gradient_matches_a_fourth_order_difference | fd |
| chain/ofdm | site, transmitter and receiver positions | jvp | SUP | native-companion | witwin/radar/synthesis/ofdm.py::synthesize_ofdm | tests/test_waveform_chain_ad.py::test_the_forward_tangent_reproduces_the_reverse_directional_derivative | adjoint |
| chain/pulsed | site, transmitter and receiver positions | vjp | SUP | native-companion | witwin/radar/synthesis/pulsed.py::synthesize_pulsed | tests/test_waveform_chain_ad.py::test_the_endpoint_gradient_matches_a_fourth_order_difference | fd |
| chain/pulsed | site, transmitter and receiver positions | jvp | SUP | native-companion | witwin/radar/synthesis/pulsed.py::synthesize_pulsed | tests/test_waveform_chain_ad.py::test_the_forward_tangent_reproduces_the_reverse_directional_derivative | adjoint |
| chain/ofdm, chain/pulsed | the silent transmitter TX_B | vjp | ZERO | native-companion | witwin/radar/paths.py::TwoWayComposer.compose | tests/test_waveform_chain_ad.py::test_the_silent_transmitter_has_an_exactly_zero_gradient | analytic |
| chain/ofdm, chain/pulsed | the out-of-plane endpoint component | vjp | ZERO | native-companion | witwin/radar/paths.py::TwoWayComposer.compose | tests/test_waveform_chain_ad.py::test_the_out_of_plane_gradient_is_exactly_zero | analytic |
| chain/pulsed | tau_rt with a RECTANGULAR pulse | both | ZERO | native-companion | witwin/radar/synthesis/pulsed.py::synthesize_pulsed | tests/test_pulsed_ad.py::test_the_rectangular_envelope_has_exactly_zero_delay_gradient | analytic |
| chain/any | one dual level over all three endpoint sets | jvp | SUP | native-companion | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_waveform_chain_ad.py::test_one_dual_level_over_all_three_endpoint_sets_is_the_supported_shape | declaration |

### Frontend chain (`witwin/radar/frontend.py`)

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| frontend/adc | the signal | both | REF | host-declaration | witwin/radar/frontend.py::FrontendChain._quantize | tests/test_frontend_chain.py::test_the_quantizer_refuses_a_differentiable_input | refusal |
| frontend/oscillator-time | timestamps and path delays | both | REF | host-declaration | witwin/radar/frontend.py::Noise.phase_difference | tests/test_correlated_phase_noise.py::test_delays_refuse_nonexistent_brownian_time_derivative | refusal |

### Adaptive path interpolation (`witwin/radar/paths.py`)

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| paths/adaptive-interpolation | endpoint delays and complex transfers at a fixed time partition | both | SUP | native-companion | witwin/radar/paths.py::interpolate_path_rows | tests/test_path_interpolation.py::test_interpolation_preserves_carrier_wraps_and_native_derivatives | oracle-f64 |

Adaptive topology and refinement decisions are held fixed under AD. Probe residuals are detached
only to choose that discrete partition; the returned cube uses the native interpolant on live
endpoint payloads. The public-scene adjoint is tested in
`tests/test_adaptive_motion.py::test_adaptive_recompiles_moving_reflectors_and_preserves_scene_adjoint`.

## Processing: the non-differentiability wall

The wall sits at the first DISCRETE DECISION, not at "post-processing". Above it
the signal chain is linear or smooth and stays live; below it every stage
refuses at its entry, in both modes, before any device operation runs.
There is one processing implementation per concept under `witwin/radar/processing/`; no mirror package or adapter copy exists.

`witwin/radar/policy.py::refuse_derivative` is the single owner of every
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
| processing/range_profile | the frame cube | vjp | SUP | torch-orchestration | witwin/radar/processing/range_doppler.py::range_profile | tests/test_processing_differentiability_wall.py::test_the_range_and_doppler_transforms_stay_differentiable | analytic |
| processing/range_doppler_map | the frame cube | vjp | SUP | torch-orchestration | witwin/radar/processing/range_doppler.py::range_doppler_map | tests/test_processing_differentiability_wall.py::test_the_range_and_doppler_transforms_stay_differentiable | analytic |
| processing/beam_cube | the frame cube | vjp | SUP | torch-orchestration | witwin/radar/processing/angle.py::beam_cube | tests/test_processing_differentiability_wall.py::test_the_beam_cube_stays_differentiable | analytic |
| processing/matched_filter | the received train | vjp | SUP | torch-orchestration | witwin/radar/processing/range_doppler.py::matched_filter | tests/test_processing_differentiability_wall.py::test_the_matched_filter_stays_differentiable | analytic |
| processing/tdm_compensate | the virtual-antenna column | vjp | SUP | torch-orchestration | witwin/radar/processing/angle.py::tdm_compensate | tests/test_processing_differentiability_wall.py::test_the_tdm_compensation_stays_differentiable | analytic |
| processing/music_spectrum | the angle snapshots | vjp | SUP | torch-orchestration | witwin/radar/processing/angle.py::music_spectrum | tests/test_processing_differentiability_wall.py::test_the_music_pseudo_spectrum_is_differentiable_and_matches_a_difference | fd |
| processing/music_image | the range profile | vjp | SUP | torch-orchestration | witwin/radar/processing/angle.py::music_image | tests/test_processing_differentiability_wall.py::test_music_image_carries_the_same_live_derivative_as_the_spectrum | analytic |
| processing/backproject | the range profile | vjp | SUP | native-companion | witwin/radar/processing/imaging.py::backproject | tests/processing/test_imaging.py::test_the_native_vjp_and_jvp_are_the_cpu_routes_derivatives_and_match_finite_differences , tests/processing/test_imaging.py::test_the_image_keeps_a_leading_batch_and_differentiates_through_the_cpu_route | fd |
| processing/backproject | the range profile | jvp | SUP | native-companion | witwin/radar/processing/imaging.py::backproject | tests/processing/test_imaging.py::test_the_native_vjp_and_jvp_are_the_cpu_routes_derivatives_and_match_finite_differences | fd |

### Below the wall (`REF`)

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| processing/ca_cfar | the Range-Doppler map | vjp | REF | host-declaration | witwin/radar/processing/detection.py::ca_cfar | tests/test_processing_differentiability_wall.py::test_a_cfar_detector_refuses_a_gradient_before_any_compute , tests/test_processing_differentiability_wall.py::test_the_detectors_still_detect_and_the_guard_changed_no_value | refusal |
| processing/ca_cfar | the Range-Doppler map | jvp | REF | host-declaration | witwin/radar/processing/detection.py::ca_cfar | tests/test_processing_differentiability_wall.py::test_a_cfar_detector_refuses_a_forward_dual_before_any_compute | refusal |
| processing/os_cfar | the Range-Doppler map | vjp | REF | host-declaration | witwin/radar/processing/detection.py::os_cfar | tests/test_processing_differentiability_wall.py::test_a_cfar_detector_refuses_a_gradient_before_any_compute | refusal |
| processing/os_cfar | the Range-Doppler map | jvp | REF | host-declaration | witwin/radar/processing/detection.py::os_cfar | tests/test_processing_differentiability_wall.py::test_a_cfar_detector_refuses_a_forward_dual_before_any_compute | refusal |
| processing/go_cfar | the Range-Doppler map | vjp | REF | host-declaration | witwin/radar/processing/detection.py::go_cfar | tests/test_processing_differentiability_wall.py::test_a_cfar_detector_refuses_a_gradient_before_any_compute | refusal |
| processing/go_cfar | the Range-Doppler map | jvp | REF | host-declaration | witwin/radar/processing/detection.py::go_cfar | tests/test_processing_differentiability_wall.py::test_a_cfar_detector_refuses_a_forward_dual_before_any_compute | refusal |
| processing/so_cfar | the Range-Doppler map | vjp | REF | host-declaration | witwin/radar/processing/detection.py::so_cfar | tests/test_processing_differentiability_wall.py::test_a_cfar_detector_refuses_a_gradient_before_any_compute | refusal |
| processing/so_cfar | the Range-Doppler map | jvp | REF | host-declaration | witwin/radar/processing/detection.py::so_cfar | tests/test_processing_differentiability_wall.py::test_a_cfar_detector_refuses_a_forward_dual_before_any_compute | refusal |
| processing/group_peaks | the Range-Doppler map | both | REF | host-declaration | witwin/radar/processing/detection.py::group_peaks | tests/test_processing_differentiability_wall.py::test_peak_grouping_and_clustering_refuse_both_modes_before_any_compute | refusal |
| processing/cluster_points | the point positions | both | REF | host-declaration | witwin/radar/processing/detection.py::cluster_points | tests/test_processing_differentiability_wall.py::test_peak_grouping_and_clustering_refuse_both_modes_before_any_compute | refusal |
| processing/scan_aoa | the virtual-antenna column | both | REF | host-declaration | witwin/radar/processing/angle.py::scan_aoa | tests/test_processing_differentiability_wall.py::test_an_argmax_angle_estimator_refuses_both_modes_before_any_compute | refusal |
| processing/ca_cfar_1d | the range profile | both | REF | host-declaration | witwin/radar/processing/detection.py::ca_cfar_1d | tests/test_processing_differentiability_wall.py::test_the_range_only_detector_refuses_both_modes_before_any_compute | refusal |
| processing/point_cloud | the Range-Doppler map | vjp | REF | host-declaration | witwin/radar/processing/detection.py::point_cloud | tests/test_processing_differentiability_wall.py::test_the_point_cloud_refuses_a_gradient_before_any_compute , tests/test_processing_differentiability_wall.py::test_the_point_cloud_still_produces_its_one_point | refusal |
| processing/point_cloud | the Range-Doppler map | jvp | REF | host-declaration | witwin/radar/processing/detection.py::point_cloud | tests/test_processing_differentiability_wall.py::test_the_point_cloud_refuses_a_forward_dual_before_any_compute | refusal |
| processing/point_cloud | the detection threshold | vjp | REF | host-declaration | witwin/radar/processing/detection.py::point_cloud | tests/test_processing_differentiability_wall.py::test_the_point_cloud_refuses_a_live_detection_threshold | refusal |
| processing/_keep_strongest | the energy map | both | REF | host-declaration | witwin/radar/processing/detection.py::_keep_strongest | tests/test_processing_differentiability_wall.py::test_the_peak_selection_refuses_a_gradient_before_any_topk | refusal |
| processing/backproject | the aperture and pixel geometry | both | REF | host-declaration | witwin/radar/processing/imaging.py::backproject | tests/processing/test_imaging.py::test_the_geometry_carries_no_derivative_and_mismatches_are_refused | refusal |
| processing/estimate_range_calibration | the range profile | both | REF | host-declaration | witwin/radar/processing/calibration.py::estimate_range_calibration | tests/processing/test_calibration.py::test_a_round_trip_delay_bias_is_measured_and_moves_the_range_axis | refusal |
| processing/phase_comparison_aoa | the virtual-antenna column | both | REF | host-declaration | witwin/radar/processing/angle.py::phase_comparison_aoa | tests/test_processing_differentiability_wall.py::test_an_argmax_angle_estimator_refuses_both_modes_before_any_compute | refusal |
| processing/fft2_aoa | the virtual-antenna column | both | REF | host-declaration | witwin/radar/processing/angle.py::fft2_aoa | tests/test_processing_differentiability_wall.py::test_an_argmax_angle_estimator_refuses_both_modes_before_any_compute , tests/test_processing_differentiability_wall.py::test_the_fft2_route_still_enforces_its_own_contract | refusal |
| processing/DetectionFrame | xyz, velocity_mps, energy | both | REF | host-declaration | witwin/radar/processing/tracking.py::DetectionFrame | tests/processing/test_tracking.py::test_the_handoff_is_explicitly_non_differentiable_and_refuses_a_gradient, tests/processing/test_tracking.py::test_the_handoff_also_refuses_a_forward_dual_which_it_used_to_accept | refusal |
| processing/any guarded stage | the wall speaks with one voice | both | REF | host-declaration | witwin/radar/policy.py::refuse_derivative | tests/test_processing_differentiability_wall.py::test_every_wall_refusal_comes_from_the_one_owner | refusal |
| processing/any guarded stage | no result object is produced | both | REF | host-declaration | witwin/radar/policy.py::refuse_derivative | tests/test_processing_differentiability_wall.py::test_the_no_partial_result_instrument_is_not_vacuous | refusal |

## Higher order: first derivatives only, everywhere

`witwin/radar/policy.py::first_order_only` decorates all ten registered
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
| higher-order/two_way | grad of grad, create_graph=True | vjp | REF | torch-orchestration | witwin/radar/paths.py::_TwoWayJoin.backward | tests/test_higher_order_refusal.py::test_a_grad_of_grad_request_fails_loudly_and_names_the_owner | refusal |
| higher-order/fmcw | grad of grad, create_graph=True | vjp | REF | torch-orchestration | witwin/radar/synthesis/fmcw.py::_FmcwSynthesis.backward | tests/test_higher_order_refusal.py::test_a_grad_of_grad_request_fails_loudly_and_names_the_owner | refusal |
| higher-order/ofdm | grad of grad, create_graph=True | vjp | REF | torch-orchestration | witwin/radar/synthesis/ofdm.py::_OfdmCfrSynthesis.backward | tests/test_higher_order_refusal.py::test_a_grad_of_grad_request_fails_loudly_and_names_the_owner | refusal |
| higher-order/pulsed | grad of grad, create_graph=True | vjp | REF | torch-orchestration | witwin/radar/synthesis/pulsed.py::_PulsedEchoSynthesis.backward | tests/test_higher_order_refusal.py::test_a_grad_of_grad_request_fails_loudly_and_names_the_owner | refusal |
| higher-order/sensor_weight | grad of grad, create_graph=True | vjp | REF | torch-orchestration | witwin/radar/sensors.py::_SensorWeight.backward | tests/test_higher_order_refusal.py::test_a_grad_of_grad_request_fails_loudly_and_names_the_owner | refusal |
| higher-order/frontend | grad of grad, create_graph=True | vjp | REF | torch-orchestration | witwin/radar/frontend.py::_FrontendAgc.backward | tests/test_higher_order_refusal.py::test_a_grad_of_grad_request_fails_loudly_and_names_the_owner | refusal |
| higher-order/every boundary | a cotangent carrying a forward tangent | jvp | REF | torch-orchestration | witwin/radar/policy.py::first_order_only | tests/test_higher_order_refusal.py::test_a_cotangent_carrying_a_forward_tangent_is_refused | refusal |
| higher-order/every boundary | a cotangent that itself requires grad | vjp | REF | torch-orchestration | witwin/radar/policy.py::first_order_only | tests/test_higher_order_refusal.py::test_a_cotangent_that_itself_requires_grad_is_refused | refusal |
| higher-order/every boundary | no partial second-order gradient is left behind | vjp | REF | torch-orchestration | witwin/radar/policy.py::first_order_only | tests/test_higher_order_refusal.py::test_no_gradient_survives_the_refusal | refusal |
| higher-order/every boundary | the FIRST-order request over the same graph | vjp | SUP | native-companion | witwin/radar/policy.py::first_order_only | tests/test_higher_order_refusal.py::test_the_first_order_request_over_the_same_graph_still_works | adjoint |
| higher-order/all ten backwards | the decorator is applied at every site | both | REF | torch-orchestration | witwin/radar/policy.py::first_order_only | tests/test_higher_order_refusal.py::test_every_registered_backward_is_decorated_by_the_one_owner, tests/test_higher_order_refusal.py::test_the_package_names_no_second_higher_order_rule | refusal |
| higher-order/nested forward levels | a second dual_level | jvp | REF | torch-orchestration | witwin/radar/policy.py::first_order_only | tests/test_higher_order_refusal.py::test_nested_forward_levels_stay_torch_owned | refusal |
| higher-order/decorator ordering | once_differentiable cannot replace the check | vjp | REF | torch-orchestration | witwin/radar/policy.py::first_order_only | tests/test_higher_order_refusal.py::test_once_differentiable_cannot_replace_the_grad_mode_check | refusal |

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
| chain/fmcw, chain/ofdm, chain/pulsed | all eight supported leaves at once | vjp | SUP | native-companion | tests/support/ad_matrix.py::frame | tests/test_combined_ad_matrix.py::test_every_supported_leaf_is_live_in_one_combined_backward, tests/test_combined_ad_matrix.py::test_each_combined_gradient_equals_its_single_leaf_gradient | analytic |
| chain/fmcw, chain/ofdm, chain/pulsed | all eight supported leaves at once, against a difference | vjp | SUP | native-companion | tests/support/ad_matrix.py::frame | tests/test_combined_ad_matrix.py::test_the_combined_difference_equals_the_sum_of_the_single_leaf_differences | fd |
| reevaluate/prepared | material sigma_e | vjp | SUP | native-companion | tests/support/multi_endpoint_world.py::make_scene | tests/test_combined_ad_matrix.py::test_the_conductivity_gradient_matches_a_central_difference | fd |
| rcs/from_rcs | phase_rad to a synthesized cube, all three waveforms | vjp | SUP | torch-orchestration | witwin/radar/scattering.py::ScalarRcsResponse.from_values | tests/test_combined_ad_matrix.py::test_the_response_phase_gradient_matches_a_central_difference | fd |
| rcs/from_rcs | phase_rad under a magnitude-only loss | vjp | ZERO | torch-orchestration | witwin/radar/scattering.py::ScalarRcsResponse.evaluate | tests/test_combined_ad_matrix.py::test_a_magnitude_only_loss_cannot_see_the_response_phase | analytic |
| chain/fmcw, chain/ofdm, chain/pulsed | endpoint positions, full-cube cotangent | both | SUP | native-companion | tests/support/ad_matrix.py::frame | tests/test_combined_ad_matrix.py::test_the_jvp_is_the_adjoint_of_the_vjp_on_one_frozen_topology | adjoint |
| reevaluate/prepared | out:compact row identity across none, jvp and vjp | both | SUP | native-declared | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_combined_ad_matrix.py::test_the_three_ad_modes_publish_the_same_compact_rows | declaration |
| reevaluate/prepared | out:primal, bitwise across the three modes | both | SUP | native-companion | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_combined_ad_matrix.py::test_the_primal_is_bitwise_identical_in_all_three_ad_modes | analytic |
| reevaluate/prepared | out:every published tensor under ad_mode none | both | DECL | native-declared | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_combined_ad_matrix.py::test_ad_mode_none_publishes_no_graph_and_no_tangent | declaration |
| reevaluate/prepared | out:topology identity, scene-leaf compile against the shared one | both | SUP | native-declared | tests/support/ad_matrix.py::frame | tests/test_combined_ad_matrix.py::test_the_scene_leaf_scenario_is_the_same_topology_as_the_shared_one | declaration |
| chain/fmcw, chain/ofdm, chain/pulsed | out:the scenario shape the matrix rests on | both | DECL | native-declared | tests/support/ad_matrix.py::frame | tests/test_combined_ad_matrix.py::test_the_three_waveforms_share_one_frozen_topology | declaration |

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
| reevaluate/prepared | out:payload of a dead row | both | ZERO | native-companion | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_row_validity_ad.py::test_a_dead_row_publishes_an_exactly_zero_payload, tests/test_row_validity_ad.py::test_the_base_configuration_has_no_dead_row_at_all | analytic |
| reevaluate/prepared | endpoint positions through a dead row | vjp | ZERO | native-companion | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_row_validity_ad.py::test_the_dead_rows_contribute_exactly_zero_to_the_gradient | analytic |
| reevaluate/prepared | endpoint positions through a dead row | jvp | ZERO | native-companion | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_row_validity_ad.py::test_a_dead_row_carries_an_exactly_zero_forward_tangent | analytic |
| synthesis/any | out:cube contribution of a dead row | both | ZERO | native-companion | witwin/radar/synthesis/assembly.py::SynthesisPathBatch.from_radar_paths | tests/test_row_validity_ad.py::test_a_poisoned_dead_row_cannot_change_the_cube | analytic |
| reevaluate/prepared | endpoint positions, frozen replay against a fresh discovery | vjp | SUP | native-companion | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_row_validity_ad.py::test_the_frozen_replay_and_a_fresh_discovery_agree_bit_for_bit | analytic |
| reevaluate/prepared | endpoint positions of a fully occluded site | vjp | ZERO | native-companion | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_row_validity_ad.py::test_a_fully_occluded_site_answers_with_an_exactly_zero_gradient, tests/test_row_validity_ad.py::test_moving_one_site_kills_exactly_two_rows_and_keeps_nine | analytic |
| join/freeze | a declared site with no row at a discovery (occluded) | both | ZERO | torch-orchestration | witwin/radar/paths.py::TwoWayComposer.freeze | tests/test_row_validity_ad.py::test_a_fresh_freeze_at_the_occluded_geometry_composes_the_site_to_zero | analytic |
| join/freeze | an unknown declared site ID | both | REF | torch-orchestration | witwin/radar/paths.py::TwoWayComposer.freeze | tests/test_row_validity_ad.py::test_an_unknown_site_id_is_still_refused | refusal |
| join/compose | complex3 fields, element and site positions, interaction points, Channel length, scalar response | both | SUP | native-companion | witwin/radar/paths.py::_TwoWayJoin | tests/test_join_geometry_ad.py::test_reverse_mode_matches_float64_autograd, tests/test_join_geometry_ad.py::test_forward_mode_satisfies_the_adjoint_identity | oracle-f64 |
| join/compose | scattering matrix and final-segment directions (site basis) | both | SUP | native-companion | witwin/radar/paths.py::_TwoWayJoin | tests/test_join_geometry_ad.py::test_reverse_mode_matches_float64_autograd, tests/test_join_geometry_ad.py::test_forward_mode_satisfies_the_adjoint_identity | oracle-f64 |
| join/compose | out:total_delay_residual_s, the float64 delay residual | both | DECL | native-declared | witwin/radar/paths.py::_TwoWayJoin | tests/test_join_geometry_ad.py::test_forward_matches_the_float64_reference | declaration |
| join/compose | site-less rows: complex3 field, transmitter, receiver, interaction points, Channel length | both | SUP | native-companion | witwin/radar/paths.py::_TwoWayJoin | tests/test_join_geometry_ad.py::test_site_less_rows_match_float64_autograd_in_reverse_mode, tests/test_join_geometry_ad.py::test_site_less_rows_satisfy_the_adjoint_identity_in_forward_mode | oracle-f64 |
| synthesis/fmcw, ofdm, pulsed | in:total_delay_residual_s, the float64 delay residual | both | DECL | native-declared | witwin/radar/synthesis/assembly.py::delay_residual | tests/test_delay_residual_synthesis.py::test_the_fmcw_beat_reads_the_float64_delay | declaration |
| scatter/matrix | a scattering matrix's site up axis | both | REF | host-declaration | witwin/radar/scattering.py::ScatteringMatrixResponse | tests/test_polarimetric_scattering.py::test_the_matrix_response_refuses_a_derivative_on_its_site_axis | refusal |
| reevaluate/prepared | out:a stale discovery-time answer | both | REF | native-companion | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_row_validity_ad.py::test_the_replay_never_answers_with_the_geometry_it_was_frozen_at | analytic |

## Refused tangents, driven through the whole chain

Every row here is asserted to fail BEFORE a cube exists: the three waveform
owners are replaced by counting stand-ins and the count must be exactly zero,
with the instrument calibrated against the same chain running normally.

`RadarEndpointSpec` carries exactly `stable_ids`, `positions_m`,
`polarizations` and `powers_w`, so there is no polarization BASIS a Radar caller
could mark. That refusal is Channel's and lives in Channel's matrix.

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| reevaluate/prepared | an ad_mode outside the vocabulary | both | REF | host-declaration | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_refused_tangents.py::test_an_ad_mode_outside_the_vocabulary_is_refused_before_any_cube, tests/test_refused_tangents.py::test_the_no_cube_instrument_counts_a_real_synthesis | refusal |
| reevaluate/prepared | sources.powers_w | both | REF | host-declaration | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_refused_tangents.py::test_a_primal_only_endpoint_input_is_refused_in_both_modes | refusal |
| reevaluate/prepared | endpoint polarizations | both | REF | host-declaration | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_refused_tangents.py::test_a_primal_only_endpoint_input_is_refused_in_both_modes | refusal |
| reevaluate/prepared | there is no polarization basis to mark | both | DECL | host-declaration | witwin/radar/propagation.py::RadarEndpointSpec | tests/test_refused_tangents.py::test_the_endpoint_spec_carries_no_polarization_basis_to_mark | declaration |
| adapter/components | diffraction, transmission | both | REF | host-declaration | witwin/radar/channel.py::compile_scene | tests/test_refused_tangents.py::test_an_unfreezable_component_is_refused_before_any_discovery | refusal |
| synthesis/any spec | a tensor waveform scalar, from the chain | both | REF | host-declaration | witwin/radar/policy.py::require_host_float | tests/test_refused_tangents.py::test_a_tensor_waveform_scalar_is_refused_before_the_spec_exists | refusal |
| higher-order/whole chain | grad of grad over compile to cube | vjp | REF | torch-orchestration | witwin/radar/policy.py::first_order_only | tests/test_refused_tangents.py::test_a_grad_of_grad_request_through_the_whole_chain_is_refused, tests/test_refused_tangents.py::test_the_first_order_request_over_the_same_chain_still_works | refusal |
| higher-order/every boundary | a cotangent carrying a forward tangent | jvp | REF | torch-orchestration | witwin/radar/policy.py::first_order_only | tests/test_refused_tangents.py::test_a_cotangent_carrying_a_forward_tangent_is_refused_at_every_boundary | refusal |

## The four chains that had no AD coverage

| route | leaf-or-output | mode | state | mechanism | owner | test | validation |
|---|---|---|---|---|---|---|---|
| frontend/chain | endpoint positions behind a real cube | vjp | SUP | native-companion | witwin/radar/frontend.py::FrontendChain.apply | tests/test_ad_chain_coverage.py::test_the_noiseless_frontend_scales_the_endpoint_gradient_by_exactly_r_g_squared | analytic |
| frontend/chain | endpoint positions behind a real cube, noise live | vjp | SUP | native-companion | witwin/radar/frontend.py::FrontendChain.apply | tests/test_ad_chain_coverage.py::test_the_noisy_frontend_gradient_matches_a_fourth_order_difference | fd |
| frontend/chain | endpoint positions behind a real cube | jvp | SUP | native-companion | witwin/radar/frontend.py::FrontendChain.apply | tests/test_ad_chain_coverage.py::test_the_frontend_forward_tangent_reproduces_the_reverse_gradient | adjoint |
| frontend/agc | endpoint positions through a global AGC | vjp | ZERO | native-companion | witwin/radar/frontend.py::_FrontendAgc | tests/test_ad_chain_coverage.py::test_a_global_agc_makes_a_magnitude_loss_exactly_constant | analytic |
| frontend/noise | out:the Philox realisation under AD | vjp | SUP | native-companion | witwin/radar/frontend.py::_FrontendNoise | tests/test_ad_chain_coverage.py::test_the_same_seed_replays_a_bitwise_identical_gradient, tests/test_ad_chain_coverage.py::test_the_physics_chain_itself_has_no_noise_to_reproduce | declaration |
| reevaluate_slots/prepared | endpoint positions over a whole slot-major frame | vjp | SUP | native-companion | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_ad_chain_coverage.py::test_a_slot_batched_replay_carries_the_single_frame_gradient_exactly | analytic |
| reevaluate_slots/prepared | endpoint positions over a whole slot-major frame, against a difference | vjp | SUP | native-companion | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_ad_chain_coverage.py::test_the_slot_batched_gradient_matches_a_fourth_order_difference | fd |
| sensor_weight/round_trip | site position to a synthesized waveform cube | vjp | SUP | native-companion | witwin/radar/sensors.py::RoundTripPatternStage.apply | tests/test_ad_chain_coverage.py::test_a_sensor_weight_gradient_reaches_a_synthesized_waveform_cube | fd |

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
| tape/two_way | out:join context, 15 saved tensors, one launch each way | vjp | SUP | native-companion | witwin/radar/paths.py::_TwoWayJoin.setup_context | tests/test_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/aspect | out:aspect context, 9 saved tensors, one launch each way | vjp | SUP | native-companion | witwin/radar/scattering.py::_AspectResponse.setup_context | tests/test_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/fmcw_beat | out:beat context, backward saves segment where forward saves offsets | both | SUP | native-companion | witwin/radar/synthesis/fmcw.py::_FmcwSynthesis.setup_context | tests/test_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/ofdm_cfr | out:cfr context, same forward/backward asymmetry | both | SUP | native-companion | witwin/radar/synthesis/ofdm.py::_OfdmCfrSynthesis.setup_context | tests/test_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/pulsed_echo | out:echo context, same forward/backward asymmetry | both | SUP | native-companion | witwin/radar/synthesis/pulsed.py::_PulsedEchoSynthesis.setup_context | tests/test_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/sensor_weight | out:weight context, 12 saved tensors, one launch each way | vjp | SUP | native-companion | witwin/radar/sensors.py::_SensorWeight.setup_context | tests/test_backward_budget.py::test_each_boundary_costs_one_backward_launch_per_forward_launch | declaration |
| tape/sensor_weight | out:a table or frame leaf adds one keyed_sum launch, none otherwise | vjp | SUP | native-companion | witwin/radar/sensors.py::_keyed_parameter_gradients | tests/test_pattern_leaves.py::test_the_parameter_backward_costs_one_reduction_and_only_when_asked | declaration |
| tape/facet_response | out:the material sweep and reduction run only for a material leaf | vjp | SUP | native-companion | witwin/radar/scattering.py::_FacetResponse.backward | tests/test_facet_material_leaves.py::test_the_material_reverse_pass_runs_only_when_asked | declaration |
| tape/frontend | out:noise and AGC contexts, two owners in one call | vjp | SUP | native-companion | witwin/radar/frontend.py::_FrontendNoise.setup_context | tests/test_backward_budget.py::test_the_frontend_costs_one_backward_launch_per_forward_stage | declaration |
| tape/any | out:no tape reaches a public result record | both | SUP | torch-orchestration | witwin/radar/propagation.py::RadarLegBatch | tests/test_tape_containment.py::test_the_leg_batch_and_the_composed_batch_carry_no_tape, tests/test_tape_containment.py::test_the_synthesis_and_sensor_results_carry_no_tape | declaration |
| tape/any | out:no module outside an owner reads a context | both | SUP | native-companion | witwin/radar/policy.py::first_order_only | tests/test_tape_containment.py::test_every_context_read_sits_inside_a_tape_owner, tests/test_tape_containment.py::test_the_context_scan_is_not_vacuous | declaration |
| reevaluate/prepared | out:ad_companion_launches, ad_tape_bytes | vjp | SUP | native-declared | witwin/radar/channel.py::ChannelPropagationAdapter.reevaluate_slots | tests/test_backward_budget.py::test_the_channel_reevaluate_publishes_its_ad_launches_and_tape_bytes, tests/test_backward_budget.py::test_a_primal_only_reevaluate_builds_no_tape_at_all | declaration |

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
- **The compression parameters as leaves.** `Compression.iip3`, `p1db` and
  `am_pm` are smooth except at the saturation hold point, but each is a scalar
  of the whole cube (a cube-wide reduction per parameter) and the native
  compression operator has no parameter slot; no consumer has asked. They stay
  host floats under the host-float rule. Follow-up owner: Radar `frontend.py`, a parameter slot on `frontend_compress_{backward,jvp}` with a deterministic block reduction.
- **Waveform-parameter optimisation.** A spec scalar changes the SHAPE of the
  output as often as its value - `num_samples`, `num_subcarriers` and
  `pulse_width_s` all move a sampling grid - and a derivative taken across a
  grid change is not the derivative of a fixed function. Deciding which subset
  is safely continuous, and what a sampling-grid derivative means, is a
  modelling decision with its own ADR rather than a slot to open quietly. Follow-up owner: Radar `synthesis/`, as a new R-ADR deciding which spec scalars are continuous and what a sampling-grid derivative means.
- **A pathwise derivative through the noise realisation.** Every `Noise`
  scalar parameterises a counter-based Philox draw. A reparameterised noise
  model, where the realisation is a smooth function of a fixed standard normal,
  is the shape that would make these leaves meaningful, and it is a separate
  design with its own accuracy and reproducibility questions. Follow-up owner: Radar `frontend/`, as a reparameterised noise-model R-ADR with its own accuracy and reproducibility evidence.
- **Forward-mode pose, shape and body-transform derivatives of a mesh body.**
  The reverse mode is supported (R-ADR-052). A forward tangent on mesh vertices
  is refused by Channel 0.6.0 at compile (its finite-vertex check has no
  forward-AD formula), so a pose tangent cannot reach the legs, and
  `SmplPoseDeformation.velocity_at` opens its own forward level over the pose.
  Follow-up owner: Channel, a primal-only finite-vertex check at compile; then Radar `smpl.py` evaluates the vertex velocity outside the caller's forward level.
- **A pose-dependent polarization.** An `"up"`/`"right"` polarization and a
  pose trajectory's rotation turn the polarization every leg is launched with,
  and Channel's fixed-topology replay takes endpoint polarizations as
  primal-only constants (its field is linear in the launched dipole, but the
  3x3 transport that maps it is not published). Refused for a pose leaf.
  Follow-up owner: Channel, a polarization tangent and cotangent on the replay (or the unprojected 3x3 transport); Radar then opens the rotation and the alias polarizations.
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
