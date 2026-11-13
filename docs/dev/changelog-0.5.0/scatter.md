# scatter

## Breaking changes

- `ScatterResponse.evaluate` takes `site_count` (the join's site count) instead of `row_count`; it always returned one value per site. Implementations of the protocol rename the parameter.
- The sampled-surface recipe of the RCS oracle (`N` isotropic `PointTargets` of `sigma / N`) is gone, because it is not a surface model and does not converge. A surface is a `MeshTargets` now; there is no migration that keeps the old numbers.
- `tools/benchmark_smpl_walk.py` scatters from the body's facets (`MeshTargets`, skin dielectric, self-occlusion) instead of area-sampled point sites of `1 / N` cross section; `--sites` is now the site budget. Its results are not comparable with earlier runs.

## Added

- `MeshTargets(structure_ids=None, cluster_size_m=None, max_sites=None, seed=0)`: the faces of Core structures are the scatterers. Each facet is a physical-optics facet (exact flat-facet form factor, Kirchhoff obliquity, `4 pi A / lambda^2` normalisation) extracted natively per frame from the snapshot's deformed world vertices, occluded by the scene through the legs, culled on its back side, and differentiable with respect to the vertices in reverse mode. Multi-bounce on the target comes from the scene's reflections (`reflections=1` for a dihedral, `2` for a trihedral). `cluster_size_m` or `max_sites` merges facets into fewer sites for large meshes (R-ADR-042).
- A facet's return depends on its structure's Core material: the layer-stack reflection coefficient at the incidence angle, co-polar for the radar's polarization (R-ADR-043), and, with `roughness_front`, a coherent roughness loss plus an energy-consistent Lambertian diffuse term with a seeded per-facet phase (R-ADR-044).
- `SmplPoseDeformation.mesh_target(scene, structure_id=..., material=..., ...)` returns the scene with the posed body in it and the `MeshTargets` that makes the body scatter from its facets.
- `Fluctuation(swerling, coherent_interval, seed)` on `PointTargets` and `StructureTargets`: Swerling I-IV power fluctuation per site, held for a coherent interval (I, III) or redrawn every observation (II, IV), seeded and keyed like the frontend's noise (R-ADR-045). Swerling II and IV refuse `Motion.adaptive()`, which would interpolate between independent draws.
- `RcsTable(axis, rcs_m2, frequencies_hz=None, phase_rad=0.0)` as a `PointTargets.rcs`: a tabulated `sigma(aspect, frequency)`, reduced to the carrier and interpolated per composed row at the bistatic aspect angle, differentiable in the axis and the leg directions (R-ADR-045).
- `FacetScatterResponse`, `TableScatterResponse` and `FluctuatingResponse` in `witwin.radar.scattering`; `Fluctuation`, `MeshTargets` and `RcsTable` at the package root.

## Fixed

- `PointTargets` and `StructureTargets` accept per-target `rcs`, `amplitude` and `phase` vectors without an `Aspect`; `ScalarRcsResponse` carries one value per site (PH-4). A per-target vector declared against caller IDs that are not ascending is permuted into the join's site order, and so are an `Aspect`'s axes, strengths and phases, which were previously applied in declaration order.
- The RCS of a surface now converges to its analytic value as the mesh is refined: a 5 cm PEC plate reads -0.01 dB of `4 pi A^2 / lambda^2` from two triangles, a sphere within 0.3 dB of `pi a^2`, a dihedral within 0.1 dB of `8 pi a^2 b^2 / lambda^2`, where `N` isotropic sites read up to +36 dB (PH-2, PH-3). A trihedral's triple bounce composes correctly but reads about -8.5 dB until the legs carry polarization (strict xfail, PH-5).

## Performance

- A still mesh target's facets are placed once per session and reused; each composition adds exactly one native launch (`facet_response_forward`) and no host synchronization or device copy over point targets at the same sites, pinned by a budget test.
