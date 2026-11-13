# Multipath and the propagation adapter

## Breaking changes

- `FrozenLegTopology` (internal Channel adapter record) no longer has a `prepared` field: it carries the discovery
  `topology`, and `prepared`, `components` and the three `prepare_*` counters are properties that prepare the replay
  partition on first use. `ChannelPropagationAdapter.discover` is the new one-call discovery; `freeze` is discovery
  plus preparation, as before.
- `RadarPathTopology` gains `direct_row`, and a composed row with `site_id == -1` has `inbound_row == outbound_row ==
  -1`. Code that indexes legs with `inbound_row` must skip site-less rows (they exist only with `environment=True` or
  `coupling=True`).
- `Result`, `Paths` and per-frame records gain `discovery_complete` and `transmission_dropped_rows`; `Paths` gains
  `transmission`, `diffraction`, `environment` and `coupling`. `path_set_complete` is now also false when Channel's own
  discovery was not provably complete.

## Added

- `transmission=True` and `diffraction=True` on `trace`, `simulate` and `stream`: paths through walls and
  edge-diffracted paths. Both require `grad="none"` and a discovery at every evaluated instant (the default), because
  Channel cannot freeze them; other routes refuse them by name.
- `environment=True` adds site-less transmitter-to-receiver rows for the scene's own returns (a wall facing the radar
  echoes at `2d/c`); `coupling=True` adds the elements' straight-segment coupling and is refused for co-located
  transmit and receive elements. Both are off by default.
- `Paths.component_index()` classifies composed rows as target, environment clutter, direct leakage or
  multi-interaction, and `Paths.select(name)` keeps one class, so `radar.echo(paths.select("environment_clutter"))`
  synthesizes the clutter alone.
- `discovery=` (`"auto"`, `"exhaustive"`, `"rays"`) selects Channel's reflection candidate source; `max_paths=` bounds
  one discovery's rows (Channel refuses on overflow); `max_round_trips=` bounds one sensor pair's composed rows and is
  refused by name, naming the pair, before the join is built.
- `wideband=True` for OFDM: each subcarrier's transport is evaluated by Channel (spreading, material and layer response
  across the band) instead of shifting the reference coefficient. Refused for FMCW and pulsed radars.

## Fixed

- A scatter site on (or within a fraction of a millimetre of) an array element is refused by name instead of returning
  a finite, unbounded echo; the same check runs once per discovery on every leg that includes the line of sight.
- `path_set_complete` no longer claims completeness when Channel's reflection discovery fell back to rays.

## Performance

- A `grad="none"` run reads each rediscovered instant's rows straight from the discovery: one Channel call per leg per
  rediscovery instead of three (discover, prepare, replay), and a static world never prepares or replays. The numbers
  are bit-identical to the replay route.
