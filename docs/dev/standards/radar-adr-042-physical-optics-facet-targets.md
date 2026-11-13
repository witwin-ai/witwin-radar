# R-ADR-042: Mesh targets scatter as physical-optics facets in Radar's native layer

Status: Accepted

## Context

R-ADR-020 section 4 refused every rule that derives a scatter site from geometry and
recorded the deferral: a sampling rule is a geometry algorithm, and a geometry
algorithm on the production path must not be a Torch expression. The audit
measured what that left a user with (PH-2, PH-3): a surface's echo had to be
hand-built from `N` isotropic point sites of `sigma / N`, which reads `N sigma`
for a broadside plate (+24 to +36 dB at 16^2 to 64^2 sites), and a
sampling-dependent interference sum for a sphere or a trihedral. The SMPL
walking benchmark used exactly that recipe.

The owner decided where the extraction lives (plan 4.11, decision 1): in
Radar's native layer, with visibility through Channel's consumer, so the
Channel contract does not grow. Radar may read `witwin.core` mesh geometry
directly and reaches Channel only through `witwin/radar/channel.py`.

## Decision

### 1. Owners

| Concern | Owner |
|---|---|
| target vocabulary: `MeshTargets(structure_ids, cluster_size_m, max_sites, seed)` | `witwin/radar/targets.py` |
| which faces scatter, their materials, how facets group into sites, one frame's facet geometry | `witwin/radar/facets.py` |
| the facet scattering coefficient, `FacetScatterResponse` | `witwin/radar/scattering.py` |
| the `facets` site source and the per-binding facet frame | `witwin/radar/binding.py` |
| both native families, `facet_geometry` and `facet_response`, with their VJP and JVP | `witwin/radar/cuda/facet_scatter.cu` |

The target structures stay in the Core scene, so Channel sees them as
occluders and reflectors. A frame's facets are computed from the snapshot's
world vertices (deformation, authored pose, rigid motion, in Core's documented
convention) by one `facet_geometry` launch; nothing per facet is Torch.

### 2. The facet model

For a composed row through site `s`, the response is the coherent sum over the
site's facets `f`:

```
S = sum_f [ j (k^2/pi) A_f ob_f chi_f Gamma_f F_f  +  k sqrt(sigma_d,f / pi) exp(-j psi_f) ] exp(j k q.(c_f - x_s))
```

* `k = 2 pi / lambda` at the carrier, metres; `u_in = -dir_in` points from the
  facet toward the wave's source (the inbound leg's arrival direction reversed),
  `u_out = dir_out` toward the receiver side (the outbound leg's departure
  direction); `q = u_in + u_out`.
* `A_f`, `n_f`, `c_f`: facet area, unit normal by the right-handed corner order,
  centroid. `ci = n.u_in`, `co = n.u_out`; a facet with either not positive
  contributes zero (backface culling, in the kernel): a facet is lit and seen
  from its normal side only.
* `ob = (ci + co) / 2`, the scalar Kirchhoff obliquity. It is `cos theta` when
  monostatic and `cos theta_i` at the bistatic specular direction.
* `F_f = (1/A) int_T exp(j k q.(r - c_f)) dA`, the exact form factor of a flat
  triangle in the plane-wave approximation over the facet, evaluated in closed
  form as `2 g[p0, p1, p2]`, the second divided difference of `-exp(jx)` at the
  centred phase nodes (Hermite-Genocchi), with a Taylor series for close nodes.
  A flat surface is therefore exact at any facet size; refinement is needed only
  to follow curvature and illumination boundaries.
* `Gamma_f`, `chi_f`, `sigma_d,f`: the material's co-polar coefficient at the incidence angle `ci`
  (R-ADR-043) and the rough-surface terms (R-ADR-044); `Gamma = -1`, `chi = 1`,
  `sigma_d = 0` for a smooth conductor.
* `exp(j k q.(c_f - x_s))` moves each facet's phase reference from the site
  position `x_s`, which is what the legs were evaluated to, to its own
  centroid. It is exactly 1 for a site of one facet.

**Normalisation.** The two-way join multiplies `C_in S C_out` with
`C = lambda / (4 pi d) exp(-j k d)` per leg and the site excited at 1 W, so a
dimensionless `S` reproduces the bistatic radar equation when
`|S|^2 = 4 pi sigma / lambda^2`. A broadside plate's facets give
`S = 4 pi A / lambda^2 = (k^2 / pi) A` and `sigma = 4 pi A^2 / lambda^2`.

**Phase.** `j` is the Huygens kernel's phase in the `exp(+j omega t)`,
`exp(-j k d)` convention Channel uses. A stationary-phase evaluation over a
convex surface contributes `-j`, so a PEC sphere returns `-sqrt(pi) a` with the
phase of its front point and a broadside PEC plate `-j 4 pi A / lambda^2`.
`tests/oracle/test_rcs_convergence.py` pins both phases.

### 3. Validity domain

* **Physical optics.** High frequency: radii of curvature, edge lengths and
  facet-to-facet spacing of illuminated features much larger than `lambda`
  (the sphere oracle has `k a = 81`). No edge diffraction (the PO edge
  contribution is present, the PTD fringe correction is not), no creeping waves,
  no surface waves, no shadow-boundary smoothing: a facet is lit or not.
* **Plane wave over a facet.** `F_f` assumes the leg's direction is constant
  across the facet: a facet of extent `D` at leg length `d` needs
  `k D^2 / (8 d) << 1`. Near-field targets are handled by refining the mesh,
  since every facet carries its own leg geometry to its site.
* **Facet size versus wavelength.** None for flat parts. On a curved surface the
  faceting error is the sagitta: a facet whose sagitta `delta` has
  `2 k delta` of order 1 shifts the specular phase. The sphere oracle is within
  1 dB from 3968 faces (about `lambda/2` facets on a `k a = 81` sphere).
* **Sites and clusters.** A merged site (`cluster_size_m`, `max_sites`) keeps
  every facet's own term and phase, which is exact while the cluster of extent
  `D` is in the far field of both legs, `k D^2 / (8 R) << 1`. The cluster is
  occluded as one point: its centroid's leg validity gates all its facets.
* **Occlusion.** By the legs: a facet whose site has no valid line of sight
  (hidden by the rest of the mesh or the world) contributes nothing. The join
  accepts such a site as an empty contribution for a mesh target. A facet is in
  or out as a whole.
* **Polarization.** Scalar and co-polar (R-ADR-043 section 3). Each leg is
  projected onto the site's dipole at the radar's polarization, so the
  polarization rotation of a reflected inbound leg is lost at the facet.

### 4. Composition with the two-way legs

Facets are ordinary scatter sites, so legs, element positions, multipath
ghosts, Doppler and every synthesis route are unchanged. Multi-bounce on the
target is the scene's own reflection: a dihedral's double bounce is an inbound
leg with one reflection (`reflections >= 1`), a trihedral's triple bounce an
inbound leg with two (`reflections >= 2`).

A facet scatters once, with physical optics, at the **last** interaction of the
wave (the shooting-and-bouncing-rays rule: geometric optics before, PO at the
final surface, free space after). The join therefore composes a mesh target's
facets only with line-of-sight outbound rows. Without the rule every
multi-bounce would be composed once per facet it visits: the dihedral read
+6 dB, twice the amplitude, before it. The leg topology names interactions by
compiled face index rather than by Core primitive, so the rule cannot single out
the target's own surfaces and also drops reflections off the rest of the world
after the facet (a facet-wall-receiver ghost). Its reciprocal ordering,
transmitter-wall-facet, is composed. Restricting the rule to self-bounces needs
Channel to publish the Core primitive (or structure) ID of each interaction in
the consumer topology; that is a Channel request, not part of this release.
Discovery still produces the outbound reflection rows the join drops; a
line-of-sight-only outbound leg would avoid that work and is left for a later
change because it splits the one adapter per epoch.

### 5. Cost and the site budget

Leg discovery and replay are per site and per element, `(N_tx + N_rx) S` leg
rows and `N_tx N_rx S` composed rows (times the multipath rows). The response
is one launch per frame, `O(rows x facets per site)`; facet geometry is one
launch. A mesh of `F` faces costs what `F` point targets cost on the legs, which
is the expensive half. `cluster_size_m` and `max_sites` are the declared
coarsening: they divide the leg cost by the cluster size and keep the response
exact within the far-field-of-the-cluster condition above. The site tables are
built on the host once per topology (one device-to-host copy of the vertices at
freeze time); a deformation or a rigid motion keeps them.

### 6. Differentiation

Exact first derivatives in both modes. The kernels carry forward-mode jets
(value plus `N` directional derivatives) through the same expression the
forward evaluates; the two special functions (form factor, layer stack) are
evaluated on small inner jets over their own arguments and chained. The VJP
owns one gradient slot per thread over four owner families (inbound leg rows,
outbound leg rows, facets, sites) and sums in the join's frozen CSR order: no
atomics. `facet_geometry` is linear, its JVP is itself on the tangent and its
VJP a per-vertex CSR reduction. A vertex derivative reaches the cube through
three routes: the site positions (through both legs), the leg directions, and
the facet corners in the response. Reverse mode is supported end to end; forward
mode is supported by both facet families, while Channel 0.6 refuses a forward
vertex tangent on the compiled scene at compile time, which is where a mesh in
the scene stops it. Material constants, the polarization and the seed select the
law and take no derivative.

## Evidence

`tests/oracle/test_rcs_convergence.py` (5 cm PEC shapes at 10 m, 77 GHz; error
against the analytic RCS):

| Shape | Before (N isotropic sites of sigma/N) | After (`MeshTargets`) |
|---|---|---|
| plate, 2 / 32 / 512 faces | +24.07 / +30.10 / +36.12 dB (16^2 / 32^2 / 64^2 sites) | -0.01 / -0.01 / -0.01 dB |
| sphere, 3968 / 16128 / 36480 faces | -11.02 / -5.06 / +0.97 dB (1024 / 4096 / 16384 sites) | -0.05 / +0.27 / -0.13 dB |
| dihedral, 128 / 512 / 2048 faces, `reflections=1` | not measured | +0.10 / -0.03 / -0.08 dB |
| trihedral, 48 / 192 / 768 faces, `reflections=2` | +11.72 / +3.91 / +8.21 dB (16 / 32 / 64 per edge) | -8.9 / -8.1 / -8.5 dB (strict xfail, PH-5) |

The trihedral's composition is correct and its level is the polarization
projection of section 3: after two perpendicular reflections the field is the
incident field rotated by 180 degrees about the third face's normal `n_C`, its
projection onto the site dipole is `2 (p.n_C)^2 - 1`, and the six orderings
average to `-1/3`, -9.5 dB. Measured -8.1 to -8.9 dB. The dihedral is the same
composition with pure TE incidence and converges. The trihedral passes when the
legs deliver the unprojected transport and the site carries a 2x2 scattering
matrix (M4-4).

`tests/test_facet_scatter_kernel.py` checks the response against a brute-force
PO quadrature per triangle, the material law against the independent Airy slab
coefficient, both AD modes against central differences and against each other,
and the geometry operator's adjoint identity.
