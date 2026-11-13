# R-ADR-044: Rough facets lose coherent power to an energy-consistent Lambertian floor

Status: Accepted

## Context

MAT-2: no target had a diffuse return. Channel's Kirchhoff roughness kernel
lives inside its own solver and has no consumer contract, so a Radar-side model
was asked for, without a Channel change, driven by the Core material's
`roughness_front`.

## Decision

Each facet of a mesh target whose material declares `roughness_front` with rms
height `h` (m) contributes two terms (`facet_scatter.cu`, R-ADR-042's
symbols):

1. **Coherent.** The physical-optics term is multiplied by the Ament (Kirchhoff,
   Gaussian height) factor `chi = exp(-(k h (ci + co))^2 / 2)`: in power,
   `exp(-g)` with `g = (2 k h cos theta)^2` when monostatic, the Rayleigh
   criterion's specular loss. The lobe keeps the facet's PO shape.
2. **Diffuse.** A Lambertian bistatic cross section
   `sigma_d = 4 A f ci co` with `f = (1 - exp(-(2 k h ci)^2)) |Gamma(ci)|^2`,
   carried as the amplitude `k sqrt(sigma_d / pi)` (so that
   `|S|^2 = 4 pi sigma_d / lambda^2`) with a phase `psi_f` drawn once per facet
   from `MeshTargets.seed` by a counter-based hash (splitmix64 of the seed and
   the facet index). The phase is a property of the surface realisation: it does
   not change from frame to frame, and a moving surface keeps its speckle.

**Energy.** Integrated over the hemisphere, the Lambertian term scatters
`(I / 4 pi) int sigma_d dOmega = I A ci f`, exactly the power
`I A ci (1 - exp(-g_i)) |Gamma|^2` that the coherent term loses at the specular
direction, so a rough facet scatters the same total power as a smooth one, for
every `h`. The diffuse fraction depends on the incidence angle only, which is
what makes the balance exact.

**Validity.** Gaussian heights with `h` small compared with the facet and the
correlation length large compared with `lambda` (the Kirchhoff regime); the
Lambertian law is the simplest energy-consistent diffuse lobe, not a fitted
material model, and the correlation lengths are not used. The diffuse sum over
`N` facets is a speckle realisation: its power fluctuates like an exponential
variable about `sum sigma_d`, as a real rough surface's does for one
realisation.

**Derivatives.** Both terms are differentiated exactly with respect to the leg
directions, the facet corners and the site positions. The seed selects the law
and takes no derivative; `h` is a leaf since R-ADR-052, refused at exactly zero. `sqrt(sigma_d)` is not differentiable at
`sigma_d = 0`; the diffuse term is skipped there (grazing, or `h = 0`).

## Evidence

`tests/test_rough_facets.py` on a 512-facet PEC plate at 10 m, 16 seeds, for
`h = 0, lambda/16, lambda/8`:

* the coherent broadside return (mean over seeds) is `sigma_0 exp(-(2 k h)^2)`
  to 0.2 dB, and falls with `h`;
* the diffuse power (variance over seeds) is `4 A (1 - exp(-g)) cos^2` within
  the seed-sampling tolerance, at broadside and at 40 degrees off, and rises with `h`;
* coherent plus diffuse never exceeds the smooth plate's return;
* the hemisphere integral of the closed form plus the coherent loss equals the
  smooth facet's power.
