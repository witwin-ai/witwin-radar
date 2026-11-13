# R-ADR-043: A facet's reflectivity is its Core material's layer-stack coefficient

Status: Accepted

## Context

MAT-1: a target's material did not affect its backscatter. Core and Channel
materials acted only on the environment reflections inside a leg; a point
target's strength was a declared cross section. A mesh target (R-ADR-042) is
made of the structures' own faces, and each face carries a Core
`PhysicalMaterial`, so its return must depend on that material the way a wall
bounce does.

Channel owns the Fresnel and layer-stack equations for its legs, but it does
not expose them through the consumer facade (`witwin.channel.propagation.consumer`),
and the coefficient is needed per composed row inside the facet kernel, at each
row's own incidence angle. It is therefore implemented natively in Radar, as
the same characteristic-matrix form, and Channel's own coefficient for the same
material through a wall bounce is the oracle.

## Decision

### 1. What is read from the material

`witwin/radar/facets.py` evaluates each distinct material of the selected
structures once per topology, at the carrier:

* `conductor_model == "perfect"`: the exact conductor, `r_TE = r_TM = -1`.
* otherwise the layer stack in vacuum: the authored `layers`, or one layer of
  the material itself of `thickness_m` (Channel's default for an unlayered
  material), each layer `(thickness_m, Re eps, Im eps, mu_r)` with `eps` the
  complex relative permittivity Core's `evaluate_at_frequency` returns
  (`eps_r - j sigma_e / (omega eps_0)`, so `Im eps <= 0`, dispersion included).
  A complex `mu_r` is refused by name.
* `roughness_front.rms_height_m`, the input of R-ADR-044.

`geometry_mode` does not change the coefficient, matching Channel's leg law,
which applies the same stack to a thin sheet and to a closed volume's face.

### 2. The coefficient

`stack_reflection` in `facet_scatter.cu`, in relative units:
`k_z = k0 sqrt(eps mu - sin^2 theta)` on the passive branch (non-positive
imaginary part under `exp(+j omega t)`), admittances `y_TE = k_z / mu` and
`y_TM = eps / k_z` (the common factors cancel in `r`), one characteristic
matrix per layer with phase thickness `delta = k_z d`, each scaled by
`exp(Im delta)` so a thick lossy layer cannot overflow (`r` is invariant to the
scale), vacuum outside and behind. Both coefficients are in the tangential-field
convention those admittances define, so they are equal at normal incidence;
`r_TM` is the negative of the textbook magnetic-field TM coefficient. Its slope in `cos theta` is carried by a one-lane
jet and chained into the facet's derivative.

### 3. The co-polar scalar

Until the site carries a 2x2 scattering matrix (M4-4), a facet returns one
co-polar number:

```
Gamma = w r_TE + (1 - w) r_TM,    w = (p . s)^2 / |p_perp|^2
```

with `p` the radar's world polarization, `s = n x u_in` the normal of the plane
of incidence and `p_perp` the part of `p` transverse to `u_in`. Both terms are
tangential-field coefficients, equal at normal incidence (where `w` is
undefined and set to `1/2`), and both `-1` on a conductor. The angle is the
incidence angle on the facet, `cos theta_i = n . u_in`: the tangent-plane
(Kirchhoff) approximation, in which each facet reflects the incident wave as its
own infinite plane would. At a bistatic specular point that is half the
bistatic angle; a monostatic return from a tilted facet reads the coefficient at
the tilt, although the facet's PO term there is a sidelobe rather than a
stationary point.

When the transport task's 2x2 site matrix exists, `r_TE` and `r_TM` become its
diagonal in the facet's TE/TM basis and `w` disappears; the stack is unchanged.

## Evidence

* `tests/test_facet_scatter_kernel.py`: the kernel's response for a
  20 mm dielectric layer equals a brute-force PO integral times the independent
  Airy coefficient (`tests/reference/radar_oracles.py`) with the same weight, to
  `2e-4`.
* `tests/test_mesh_targets.py::test_metal_and_dielectric_plates_differ_by_the_slab_coefficient`
  (the M4-3 acceptance): a broadside plate of each MAT-3 material over the same
  plate in PEC is `-Gamma(normal incidence)` of the Airy formula, to 0.01 dB and
  5 mrad.
* `tests/test_mesh_targets.py::test_the_facet_coefficient_is_channels_wall_coefficient`:
  at 35 degrees incidence, TE and TM, a plate at the bistatic specular point
  over the same plate in PEC equals Channel's own coefficient for that material
  read from a MAT-3 wall bounce, to `2e-3` in magnitude and 10 mrad in phase.
