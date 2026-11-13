# Scene session

## Breaking changes

- `witwin.radar.simulation` no longer exports `SensorEndpointIds`; import it from `witwin.radar.binding`. `Motion`,
  `Result`, `Frame` and `Paths` stay exported from `witwin.radar`; their owner modules are now
  `witwin.radar.motion` (`Motion`) and `witwin.radar.results` (`Result`, `Frame`).
- `AdaptiveMotionSpec` is removed. Use `Motion.adaptive(...)`, whose fields are the same tolerances; they are now
  validated when the `Motion` is built, for every kind, instead of when a session opens.
- The internal records moved with their concepts: `StableIdAllocator`, `ScatterSitePolicy` and `bind_radar_world`
  live in `witwin.radar.binding`, the adaptive controller in `witwin.radar.adaptive`.

## Performance

- The two-way join is built with vectorized host code and one device copy each way instead of nested Python loops,
  and a rediscovery that finds the same topology reuses the previous join and antenna-pattern tables.
- A world that does not move evaluates its paths and synthesizes its waveform once per session; later frames only run
  the receive chain.
- `simulate` writes frames into one preallocated cube when no gradient is being taken, and no longer keeps every
  frame's intermediate results.
- Chirp- and ADC-refreshed FMCW sessions replay topology-identical instants of a frame together and synthesize each
  group in one launch over only the active transmitter's sensor pairs; on the ADC route each segment carries its own
  ADC instant through a new per-segment window start on the native beat family.
  `Motion.batch_observations` now bounds these groups for every kind, and therefore the rows a fused run holds at once.
- A slot-batched replay is composed by the native join striding one slot's tables across the slots, instead of
  copying every routing table per slot, whenever the response is per site and the antenna pattern is isotropic.

All of these produce the same bits as the session they replace.

## Changed

- `Frame.points` defaults to `route="scan"`, matching `processing.point_cloud`; pass `route="phase_comparison"` to keep
  the previous estimator on a layout it supports.
- The adaptive controller compares packed topology identities, and the session's slot-batched table cache is bounded.
