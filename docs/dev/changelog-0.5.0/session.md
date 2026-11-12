# Scene session

## Breaking changes

- `witwin.radar.simulation` no longer exports `SensorEndpointIds`; import it from `witwin.radar.binding`. `Motion`,
  `Result`, `Frame` and `Paths` stay exported from `witwin.radar`; their owner modules are now
  `witwin.radar.motion` (`Motion`) and `witwin.radar.results` (`Result`, `Frame`).
- `AdaptiveMotionSpec` is removed. Use `Motion.adaptive(...)`, whose fields are the same tolerances; they are now
  validated when the `Motion` is built, for every kind, instead of when a session opens.
- The internal records moved with their concepts: `StableIdAllocator`, `ScatterSitePolicy` and `bind_radar_world`
  live in `witwin.radar.binding`, the adaptive controller in `witwin.radar.adaptive`. The private `Radar` synthesis
  methods are gone; the synthesis dispatch is the session's, and `Radar.echo` remains the only public entry to it.
- The waveform kind constants (`WAVEFORM_FMCW`, `WAVEFORM_OFDM`, `WAVEFORM_PULSED`, `WAVEFORM_KINDS`) are owned by
  `witwin.radar.synthesis.assembly`.

## Performance

- The two-way join is built with vectorized host code and one device copy each way instead of nested Python loops,
  and a rediscovery that finds the same topology reuses the previous join and antenna-pattern tables.
- A world that does not move evaluates its paths and synthesizes its waveform once per session; later frames only run
  the receive chain.
- `simulate` writes frames into one preallocated cube when no gradient is being taken, and no longer keeps every
  frame's intermediate results.
- The default isotropic antenna pattern costs no kernel launch.
- Chirp- and ADC-refreshed FMCW sessions replay topology-identical instants of a frame together and synthesize each
  group in one launch (one per ADC sample index on the ADC route) over only the active transmitter's sensor pairs.
  `Motion.batch_observations` now bounds these groups for every kind, and therefore the rows a fused run holds at once.

All of these are bit-identical to 0.4.1.

## Changed

- The adaptive controller compares packed topology identities, and the session's slot-batched table cache is bounded.
