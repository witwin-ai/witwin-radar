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

Each facet scatters as its vector physical-optics dyadic, and a composed row
through site `s` carries the 2x2 scattering matrix (R-ADR-041's polarimetric
response interface, `matrix_width = 4`) that is the projection of the coherent
sum over the site's facets `f` onto the join's site bases:

```
D_f = j (k^2/pi) A_f chi_f F_f G_f exp(j k q.(c_f - x_s))  +  k sqrt(sigma_d,f / pi) exp(-j psi_f) exp(j k q.(c_f - x_s)) I
G_f = -(r_TE R te te^T + r_TM R tm tm^T),      R x = (n.k_i) x - k_i (n.x)
S_ab = b_r,a . (sum_f D_f) . b_i,b             [HH, HV, VH, VV], receive index first
```

* `k = 2 pi / lambda` at the carrier, metres. `k_i` and `k_r` are the inbound
  and outbound legs' final-segment directions INTO the site (both legs run from
  a radar element toward the site, R-ADR-039); `q = -(k_i + k_r)`.
* `A_f`, `n_f`, `c_f`: facet area, unit normal by the right-handed corner order,
  centroid. `ci = -n.k_i`, `co = -n.k_r`; a facet with either not positive
  contributes zero (backface culling, in the kernel): a facet is lit and seen
  from its normal side only.
* `R` is the radiation of the surface current `2 n x H` the incident field `x`
  induces (the PEC physical-optics current), taken transverse to the scattered
  direction by the receive basis; monostatic it is `(n.k_i) I = -cos(theta) I`.
  `te = (n x k_i)/|n x k_i|` and `tm = te x k_i` are the incident TE and TM
  directions; `r_TE` and `r_TM` the material's tangential-field reflection
  coefficients at the incidence angle `ci` (R-ADR-043), both `-1` on a
  conductor, where `G = R` on the incident transverse plane.
* `F_f = (1/A) int_T exp(j k q.(r - c_f)) dA`, the exact form factor of a flat
  triangle in the plane-wave approximation over the facet, evaluated in closed
  form as `2 g[p0, p1, p2]`, the second divided difference of `-exp(jx)` at the
  centred phase nodes (Hermite-Genocchi), with a Taylor series for close nodes.
  A flat surface is therefore exact at any facet size; refinement is needed only
  to follow curvature and illumination boundaries.
* `chi_f`, `sigma_d,f`: the rough-surface terms (R-ADR-044); `chi = 1`,
  `sigma_d = 0` for a smooth surface. The diffuse term scatters as the identity.
* `exp(j k q.(c_f - x_s))` moves each facet's phase reference from the site
  position `x_s`, which is what the legs were evaluated to, to its own
  centroid. It is exactly 1 for a site of one facet.
* `b_i = (h, v)` for `k_i` and `b_r` for `k_r` are the join's own
  backscatter-aligned bases with the up axis the response publishes per row,
  world `z` (and the join's fallback where `z` is parallel to `k`). The join
  rebuilds `sum_ab S_ab b_r,a b_i,b^T`, the projection of `D` onto the two
  transverse planes, which does not depend on the up axis: it is bookkeeping,
  not physics, and carries no derivative.

**Normalisation.** The join's scalar response `s` is the dyadic `s I`, and a
dimensionless `s` reproduces the bistatic radar equation when
`|s|^2 = 4 pi sigma / lambda^2`. A broadside conductor plate's facets give
`D = -j (k^2 / pi) A I = -j (4 pi A / lambda^2) I` and
`sigma = 4 pi A^2 / lambda^2`.

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
  occluded as one point: the site sits at the centroid of its anchor facet (the
  member nearest the members' mean, so the site stays on the surface; the mean
  itself lies inside a closed curved body, where every leg is occluded) and that
  point's leg validity gates all its facets. Facets of one cell that face away
  from each other are only partly lit, so a cell should be small against the
  surface's radius of curvature as well.
* **Occlusion.** By the legs: a facet whose site has no valid line of sight
  (hidden by the rest of the mesh or the world) contributes nothing; the join
  composes such a site to exactly zero (the empty-segment rule, `leg_site_ids`,
  that every site policy shares). A facet is in or out as a whole.
* **Polarization.** Full 2x2 in the legs' unprojected fields: the polarization
  rotation of a reflected inbound leg reaches the facet, which is what makes a
  trihedral's triple bounce come out right. The bistatic law is the vector PO
  current with each incident component weighted by its own Fresnel
  coefficient, exact for a conductor and at specular incidence, approximate off
  specular for a dielectric (the tangent-plane approximation).

### 4. Composition with the two-way legs

Facets are ordinary scatter sites, so legs, element positions, multipath
ghosts, Doppler and every synthesis route are unchanged. Multi-bounce on the
target is the scene's own reflection: a dihedral's double bounce is an inbound
leg with one reflection (`reflections >= 1`), a trihedral's triple bounce an
inbound leg with two (`reflections >= 2`).

A facet scatters once, with physical optics, at the **last** interaction of the
wave (the shooting-and-bouncing-rays rule: geometric optics before, PO at the
final surface, free space after). The join therefore composes a mesh target's
facets only with line-of-sight outbound rows (`outbound_line_of_sight_only`;
the outbound leg runs from the receiver to the site, so its depth is the
interactions between the facet and the receiver). Without the rule every
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

A site lies on its own surface, and Channel can return an inbound reflection
whose interaction point is the site itself: the facet's own plane mirroring the
wave onto the facet, a zero-length last segment the short-segment rule misses at
float32 resolution. That is not a path; the facet already is that surface's
physical-optics interaction. The kernel composes such a row to exactly zero
when the inbound leg's last point lies within `1 um + 1e-6 |x_s|` of the site
(a discrete gate, no derivative). Without it a 4096-face dihedral read -0.81 dB
from 3648 such rows.

**Slot stride.** On a slot-major batch the facet response, like the aspect and
tabulated responses, is evaluated over every slot from ONE slot's index tables
and a slot count, with its sites shared by the slots; the reverse pass walks
each owner's per-slot CSR once per slot, in slot order, so the strided route is
bit-identical to per-slot table copies (`tests/test_response_slot_stride.py`).

**Float64 sites.** The facet sites are written in float64 (the anchor centroid
of float32 vertices, summed in double), so the join's float64 path length
(R-ADR-040) reads the precise site; Channel discovers against its float32
rounding.

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
the scene stops it. The polarization and the seed select the law and take no
derivative; the material parameters are leaves since R-ADR-052.

## Evidence

`tests/oracle/test_rcs_convergence.py` (5 cm PEC shapes at 10 m, 77 GHz; error
against the analytic RCS):

| Shape (analytic) | Before: `N` isotropic sites of `sigma / N` | After: `MeshTargets` |
|---|---|---|
| plate (7.14 dBsm) | +24.07 / +30.10 / +36.12 dB at 16^2 / 32^2 / 64^2 sites | -0.00 / -0.01 / -0.01 dB at 2 / 32 / 512 faces |
| sphere (-21.05 dBsm) | -11.02 / -5.06 / +0.97 dB at 1024 / 4096 / 16384 sites | -0.03 / +0.28 / -0.12 dB at 3968 / 16128 / 36480 faces |
| dihedral (10.15 dBsm), `reflections=1` | not measured | -0.09 / -0.09 / -0.40 dB at 256 / 1024 / 4096 faces |
| trihedral (2.37 dBsm), `reflections=2` | +11.72 / +3.91 / +8.21 dB at 16 / 32 / 64 per edge | +0.68 / +0.40 / +0.51 dB at 768 / 1728 / 3072 faces |

Before the response was polarimetric, the trihedral read -8.1 to -8.9 dB:
with each leg projected onto a site dipole at the radar polarization, the
180-degree rotation of two perpendicular reflections was lost at the facet
(the six orderings averaged to `-1/3`, -9.5 dB predicted). With the
unprojected legs and the facet's 2x2 matrix it converges like the dihedral.

`tests/test_facet_scatter_kernel.py` checks the response against a brute-force
vector PO quadrature per triangle projected on the join's bases, the material law against the independent Airy slab
coefficient, both AD modes against central differences and against each other,
and the geometry operator's adjoint identity.
