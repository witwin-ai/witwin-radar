# R-ADR-039: The outbound leg is evaluated from the receiver toward the site

Status: Accepted (0.5.0)

## Context

A round trip is two Channel legs joined at a scatter site. Until this record
the outbound leg was evaluated as Channel sees it physically: the site is a
1 W source and the receive element is the sink. That orientation has one
consequence that blocks polarimetric scattering (R-ADR-041): Channel's
`complex3_transport` transports ONE field column per leg row, the field the
source's declared polarization launches, unprojected at the sink. With the
site as source, the column is `G_out p_site` for the site's fixed endpoint
polarization. A site that scatters the incident field `E_in` re-radiates
`S E_in`, a row-dependent vector, and on a REFLECTED outbound row the
transport of an arbitrary vector needs the Fresnel s/p split that Channel
owns; one column cannot rebuild it. `polarimetric_transport` (the 2x2 Jones
transport) is line-of-sight only in Channel 0.6.

The alternative that keeps the orientation evaluates the outbound leg three
times (sites as sources polarized along x, y and z), which triples outbound
discovery and replay rows. Efficiency is the release's core requirement, so it
was rejected.

## Decision

Both legs are evaluated FROM a radar element TOWARD the sites. The inbound
leg's source is a transmitter (at the transmit power) and its sink the site;
the outbound leg's source is a receive element, excited at
`SITE_EXCITATION_POWER_W` (1 W), and its sink is the same site. By Lorentz
reciprocity of the transport, `p_rx . G(rx <- site) p_site =
p_site . G(site <- rx) p_rx`, so the scalar a round trip composes is
unchanged, and with `complex3_transport` both legs deliver a full field vector
AT THE SITE, where the site's scattering matrix applies:
`V = h_rx . S . E_in`.

This is still an explicit, separate evaluation of the outbound leg. R-ADR-003
(no reciprocity shortcut that reuses the inbound leg for the outbound one) is
untouched: reciprocity is used to choose which end of the outbound leg Channel
launches from, never to skip a leg.

What moved:

- `RadarWorldBinding` holds `transmitters`, `receivers` and `sites`; the
  receivers carry the outbound excitation as `powers_w` and the sites are a
  sink of both legs (one spec, one position tensor both legs differentiate
  through).
- `TwoWayComposer.freeze` joins the outbound leg on its SINK (the site) and
  reads the receive element from its SOURCE.
- The antenna-pattern stage reads the receive element's first segment from the
  outbound leg's `departure_target_m`.
- `AspectScatterResponse` reads the site's departure bearing off the outbound
  leg's arrival segment, `site -> arrival_origin_m`, at every depth. Its
  refusal of a reflected outbound leg is gone, and so is
  `TwoWayComposer.outbound_max_depth`, which existed only for that refusal.
- `RadarLegBatch` publishes `arrival_target_m` (the sink),
  `interaction_positions_m` and `path_length_m` beside the existing segment
  endpoints.

## Evidence

Reciprocity of Channel 0.6.0's transport, measured on the tagged build
(`radar-rel`), one element and one site, `{los, reflection}` at depth 1, a
side wall at `z = 3 m`, element polarization `+y` and site polarization
`(0.3, 0.8, 0.5)/|.|` (different on purpose, so the check is a vector
identity and not a co-polar coincidence). Geometries: broadside (6 m),
oblique, and grazing (element and site 0.1-0.15 m from the wall plane, 70 m
apart: about 0.2 degrees from grazing). Walls: 10 cm concrete slab, 6 mm glass
slab, lossy half-space (`eps_r = 5`, `sigma = 1 S/m`) and a near-PEC
half-space (`sigma = 1e7 S/m`). Site->RX against RX->site:

| Row class | worst scalar relative difference | worst complex3 (projected) relative difference | delay difference |
|---|---|---|---|
| line of sight | 1.0e-7 | 5.4e-8 | 0 (bit-identical) |
| reflection | 1.3e-7 | 1.1e-7 | 0 (bit-identical) |

The differences are float32 rounding of the two evaluation orders (about two
float32 ulps), with no growth at grazing or with loss or layering. The scalar
co-polar results on the polarization plane therefore move by at most 1.3e-7
relative (about 1e-6 dB) per row; the oracle suites (`tests/oracle/
test_radar_equation.py`, `test_image_multipath.py`,
`test_material_multipath.py`) pass unchanged.

## Consequences

Budgets: a slot-major replay group stacks the site endpoints once instead of
twice (one site spec). Channel refuses a sink that carries power, so the
receivers appear in two roles over one set of tensors: `receivers` (no power,
the site-less leg's sinks) and `receiver_sources` (1 W, this leg's sources);
the second role uploads nothing. The per-frame launch counts in
`tests/test_session_budget.py` are re-pinned over the merged release.

Row identity: the outbound leg's interaction sequence and diffracting edge are
recorded in the leg's own orientation, receiver to site. A composed row's key
(`TwoWayComposer.identity_host`, the sequences and `edge_host`, which the path
event log and the adaptive plane groups compare) is therefore the reverse of
the site-to-receiver sequence for multi-bounce outbound rows. Every key is
compared only with keys built the same way, so matching across epochs and
probes is unaffected; a consumer that reads a sequence as an ordered route
reads the outbound one from the receiver.
The `multipath` work that resolves TX and RX toward the sites in one request
(M3-5) builds on this orientation.
