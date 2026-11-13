# Multi-radar networks and interference

## Added

- `witwin.radar.Network(radars, links=(), clock_offsets=(), if_bandwidths=())`: several radars in one scene and one
  call. `Network.simulate`, `stream`, `trace` and `echo` mirror the radar's verbs and publish one `Result` per radar,
  in `radars` order. The scene is compiled once and every leg family is discovered once per topology epoch for all
  radars' elements, so the Channel cost of a network is that of one radar. Each radar takes its own ego-motion through
  `poses=(pose_0, pose_1, ...)` (`None` holds a radar still). A radar's own cube is the one `Radar.simulate` gives
  (bit-identical with isotropic elements; to float32 rounding with a directional pattern).
- Bistatic and networked radar: `links=((i, j), ...)` makes radar `j` hear radar `i`'s transmitters through the
  targets, along the straight segment between them (`direct=True`, the default) and off the environment
  (`environment=True`). Each link is synthesized with the receiver's own dechirp reference against the transmitter's
  ramp, so slope mismatch, start-time offset (`clock_offsets`) and start-frequency offset (a one-profile
  `Chirp(start_offset=...)` sequence) appear in the beat, and the receiver's IF band (`if_bandwidths`, default half
  the ADC rate) gates it. The result passes through the receiving radar's receive chain with its own echoes.
- Mutual interference: the classic FMCW signatures follow from the same rows: a same-slope interferer offset in time
  is a ghost target at `c (tau + dT) / 2`, a slope mismatch is a broadband burst while the beat crosses the IF band.
- Native operators `fmcw_cross_forward`, `fmcw_cross_backward`, `fmcw_cross_jvp`: the cross-radar beat, with
  analytic reverse and forward derivatives; a link is differentiable in `grad="vjp"` and `grad="jvp"`.

## Changed

- `Motion.auto()` in a network resolves to `Motion.adc()` when anything moves; adaptive motion and
  `Motion(rediscover_every_chirps=...)` are refused for a network. A link requires FMCW radars with a uniform TDM ramp
  or a one-profile sequence; OFDM, pulsed, DDM/BPM and multi-profile links, a non-radar emitter, oscillator phase
  noise on a receiving radar and wideband OFDM are refused by name (R-ADR-051).
