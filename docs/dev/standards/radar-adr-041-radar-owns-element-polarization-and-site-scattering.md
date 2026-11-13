# R-ADR-041: Radar owns element polarization and site scattering; Channel transports the field vector

Status: Accepted (0.5.0)

## Context

Until this record every leg requested Channel's `scalar_transport`, which
projects the transported field onto the sink's polarization, and the site was
given the radar's polarization. Channel's short-dipole projection is
unnormalised at both ends of each leg, so a round trip carried `sin^4` of the
angle to the polarization in amplitude: -5.0 dB at 30 degrees off the plane
normal to the polarization, -12.0 dB at 45, -24.1 at 60, even for
`Pattern.isotropic()`, and `Pattern.dipole()` counted the E-plane cut a second
time (PH-5, measured by R-ADR-032's oracle). The scatterer itself was a real
scalar times a phase: no HH/VV, no cross-polar term, no scattering matrix
(PH-6, MP-9). Owner decision 2: Radar requests the unprojected
`complex3_transport` and applies element pattern and polarization itself.

Channel 0.6.0 (`capabilities()`): `complex3_transport` replays under a fixed
topology for `{los, reflection}` with AD modes `{none, vjp, jvp}`;
`polarimetric_transport` (the 2x2 Jones transport) is line of sight only.

## Decision

**Response.** Every replay requests `complex3_transport`
(`witwin.radar.channel.FIELD_RESPONSE`). The adapter refuses, by name, a
Channel that cannot replay it for the requested components or AD mode; there
is no scalar fallback. Discovery keeps `scalar_transport`, which only decides
which rows exist. `RadarLegBatch.coefficient` is replaced by `field`, the
`complex64[rows, 3]` field at the site.

**Orientation.** Both legs run from a radar element toward the site
(R-ADR-039), so both fields are AT THE SITE: `E` launched by the transmitter,
`H` launched by the receiver.

**Element factor.** Channel launches from a dipole `p`; its field carries
`|P_perp(d) p|`, `d` the departure direction. The join multiplies each leg by
`a = 1/|P_perp(d) p|` (`TwoWayComposer.freeze(element_polarization=(p_tx,
p_rx))`), so the launched field is a unit co-polar vector times the transport
and the pattern stage applies the declared element pattern once. On the
element's own axis `a` is published as 0: no tangent field is non-zero on a
whole sphere, so a linearly polarized "isotropic" element has a null there.

**Site scattering.** `V0 = H^T M E`. A width-1 response is a scalar `s` with
`M = s I` (the default `ScalarRcsResponse` and the per-row `Aspect` lobe): a
polarization-preserving point scatterer, cross-polar echo exactly zero. A
width-4 response (`ScatteringMatrixResponse`, public `ScatteringMatrix`) is
`[HH, HV, VH, VV]` (receive index first) in the site's local BSA basis
`h = (u x k)/|u x k|`, `v = h x k`, with `k` the leg's final-segment
direction into the site and `u` the site's up axis:
`V0 = sum_ab S_ab (H . b_r,a)(b_i,b . E)`. `S = I` equals the width-1 `s = 1`
on a monostatic geometry. The interface a polarimetric response implements:
`matrix_width = 4`, `evaluate(sites, device) -> complex64[sites, 4]` and
`site_up(sites, device) -> float32[sites, 3]`; `scatter`'s facet-derived and
per-site responses produce it.

**Native.** All of it is in the one join launch (forward, VJP, JVP), next to
the float64 path length of R-ADR-040. Launch count is unchanged: the fields
cross as `view_as_real` views and the composed value comes back as a
`view_as_complex` view, so the composer now spends fewer launches than when
it split and recombined scalar coefficients. Per row the kernel reads two
complex 3-vectors instead of two complex scalars (48 bytes instead of 16).
The basis derivative of a width-4 response is carried analytically; the site
axis is a frame declaration and is refused as a derivative input.

**Patterns with cross-polar components.** Not implemented. Channel transports
one column per leg row - the field the element's co-polar dipole launches.
On a line-of-sight row the cross-polar column is `d x` that field, but on a
reflected row it needs the Fresnel s/p split, which only Channel owns and
which `polarimetric_transport` provides for line of sight only. The Jones
pattern record extension of R-ADR-030 is left for when Channel publishes the
2x2 transport for reflection rows (Channel request in the transport report).

## Evidence

- `tests/oracle/test_radar_equation.py`: all 16 cases pass, isotropic and
  dipole, E- and H-plane, 0 to 60 degrees, to 1e-3 in power (6 strict xfails
  removed); the H-plane cases were and stay exact.
- `tests/test_polarimetric_scattering.py`: a dihedral's VV/HH ratio is -1 to
  1e-5; a sphere's cross-polar echo is below 1e-6 of its co-polar echo (scalar
  and matrix); `ScatteringMatrix.sphere()` equals the scalar isotropic echo to
  1e-5 in both polarizations; an HV element is read only by the cross-polar
  channel; a derivative on the site axis is refused.
- `tests/test_join_geometry_ad.py`: forward, every VJP input gradient (fields,
  directions, positions, lengths, the matrix) and the JVP adjoint identity,
  for width-1 and width-4 responses, against an independent float64 Torch
  statement of the law.

## Consequences

Every number off the polarization plane changes (the sin^4 loss is gone); on
it, the co-polar numbers are unchanged to float32 rounding. `Pattern.dipole()`
now attenuates by its declared cut once. Baselines that pinned off-plane
levels move with this record.
