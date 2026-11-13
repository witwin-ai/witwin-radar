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
is no scalar fallback. A discovery that keeps its rows (the discovery route of
R-ADR-033) requests `complex3_transport` too, so those rows compose exactly as
a replay's; a discovery that only decides which rows exist takes
`scalar_transport`. `RadarLegBatch.coefficient` is replaced by `field`, the
`complex64[rows, 3]` field at the sink, and a wideband replay's band by
`field_band`, `complex64[rows, F, 3]` (Channel's `field_offsets`), laid out as
outer slots by `band_as_slots` exactly like the reference column.

**Receivers in two roles.** Channel refuses a sink that carries power, so the
binding publishes the receivers twice over one set of ID, position and
polarization tensors: `receivers` (no power: the site-less leg's sinks) and
`receiver_sources` (1 W: the outbound leg's sources, R-ADR-039). The receive
polarization is `Radar.rx_polarization` when declared, the transmit one
otherwise, in both roles.

**Orientation.** Both legs run from a radar element toward the site
(R-ADR-039), so both fields are AT THE SITE: `E` launched by the transmitter,
`H` launched by the receiver.

**Element factor.** Channel launches from a dipole `p`; its field carries
`|P_perp(d) p|`, `d` the departure direction. The join multiplies each leg by
`a = 1/|P_perp(d) p|`, so the launched field is a unit co-polar vector times
the transport and the pattern stage applies the declared element pattern once.
`p` is each leg row's own launch polarization (`RadarLegBatch.source_polarization`,
the source endpoint's polarization gathered by the adapter), not a freeze-time
constant: a radar pose (`pose=`) turns every element's polarization per
instant, Channel launches the turned one, and the factor reads the same. On the
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

**Site-less rows.** A transmitter-to-receiver row (the direct leg of R-ADR-033:
a wall echo, line of sight between distinct elements) is the join's fourth
row family: `V = a_t a_r (p_r . E) exp(j psi)`, `E` the transmitter's field at
the receiver, `p_r` the receiver's polarization (`sink_polarization`), `a_t` on
the leg's first segment and `a_r = 1/|P_perp(d_last) p_r|` on its last, and the
float64 length of R-ADR-040. The kernel writes each site row and each
site-less row straight to its frozen composed position (`place`, the inverse
of the composer's `merge`), so no gather follows the launch; the VJP gives the
site-less rows a fourth owner family in the same grid (each such row owns its
own gradient slot).

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
  statement of the law, with a different launch polarization on every row;
  the same three for the site-less family at its composed positions.
- `tests/test_polarimetric_scattering.py::test_the_element_factor_reads_the_polarization_the_pose_turned`:
  a 60 degree yaw leaves the isotropic echo unchanged to 1e-5 (a factor read
  from the authored polarization would leave a quarter).

## Consequences

Every number off the polarization plane changes (the sin^4 loss is gone); on
it, the co-polar numbers are unchanged to float32 rounding. `Pattern.dipole()`
now attenuates by its declared cut once. Baselines that pinned off-plane
levels move with this record.
