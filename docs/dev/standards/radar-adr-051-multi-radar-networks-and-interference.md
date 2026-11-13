# R-ADR-051: Multi-radar networks, bistatic links and mutual interference

Status: Accepted

## Context

One call simulated one radar (F-3). Several radars in one scene had to be simulated one after another, each paying
the whole world half (compile, discovery, replay, join) again, and no radar could hear another: there was no bistatic
or networked radar and no mutual interference. The building blocks existed: the site-less TX->RX leg (R-ADR-034),
the receiver-oriented outbound leg (R-ADR-039), per-radar `pose=` trajectories (R-ADR-038), the waveform families
(R-ADR-046..048) and the receiver noise contract (R-ADR-029).

## Decision

### 1. API: a `Network` record with the four verbs

`witwin.radar.Network(radars, links=(), clock_offsets=(), if_bandwidths=())` holds the radars. Its verbs mirror the
radar's: `trace(scene, targets, times=, poses=...) -> NetworkPaths`, `echo(paths) -> tuple[Result, ...]`,
`simulate(...) -> tuple[Result, ...]` and `stream(...) -> Iterator[tuple[Result, ...]]`, one `Result` per radar in
`radars` order. `poses` is one `pose(t)` (or `None`) per radar, the same contract as `Radar.simulate(pose=)`; the
other keywords are the radar verbs' (`los`, `reflections`, `transmission`, `diffraction`, `environment`, `coupling`,
`discovery`, `max_paths`, `max_round_trips`, `motion`, `grad`) plus `direct` (below). A record rather than
`Radar.simulate` on a tuple, because links, clocks and IF bands are properties of the set, not of any one radar, and
because the per-radar results are a tuple a caller indexes like the `radars` it passed.

`links` lists `(transmitting radar, receiving radar)` pairs. Every radar always hears its own transmitters; a link adds
another radar's. `clock_offsets[k]` is radar `k`'s frame start relative to `times` (its frame `f` starts at
`times[f] + clock_offsets[k]`). `if_bandwidths[k]` is radar `k`'s IF band, default half its ADC rate.

### 2. One world half for every radar

The network session binds every radar's elements as ONE endpoint set: transmitters and receivers are concatenated in
radar order with stable IDs `1_000_000 + rank` and `2_000_000 + rank`, each radar's elements carried by its own pose
(`carry_binding`, the single-radar pose equation, now a module function). The epoch loop, discovery, replay, join and
pattern stage are the single-radar `WorldHalf`'s (`NetworkWorldHalf` overrides only the binding, the pattern frame
and the separation groups). So per topology epoch there is one discovery per leg family (inbound, outbound and, when
composed, the site-less leg) for all radars, one replay per leg family per observation group, and one join over every
transmitter-receiver pair. Channel's per-call cost is flat in the endpoint count (R-ADR-033's measurement), so the
world half costs about what one radar's does. `tests/test_network.py::test_the_world_half_is_paid_once_whatever_the_radar_count`
pins it: 1, 2 and 3 radars make the same Channel calls (3 evaluations for a still world, 24 for a moving one
with 8 merged instants per frame and 2 frames), and the join is frozen once.

Measured A/B, same process, contended GPU, minimum of 4 runs, 4 frames, radars of 2 TX x 4 RX, a PEC wall and three
point targets, `reflections=1`, no links (Channel `evaluate` calls / wall time):

| world | radars | network | the radars simulated one by one |
|---|---|---|---|
| still | 2 | 2 / 30.8 ms | 4 / 47.8 ms |
| still | 3 | 2 / 31.0 ms | 6 / 58.9 ms |
| every radar walking, `Motion.chirp()`, 32 slots a frame | 2 | 256 / 2037 ms | 512 / 5288 ms |
| every radar walking, `Motion.chirp()`, 32 slots a frame | 3 | 256 / 2406 ms | 768 / 7741 ms |

The radars' observation instants are merged into one schedule (`np.unique` of every radar's offsets plus its clock
offset). Each radar then reads its own instants out of the frame, so a radar observes the world exactly where it would
alone. A frame's observations are materialised once per frame (every radar reads them), which is the retention of
`Radar.trace`, not of the fused route; `stream` still holds one frame.

Constraints that follow from one world: one carrier (`Radar.carrier`, the frequency the world is evaluated at;
another start frequency is a one-profile `Chirp(start_offset=...)` sequence), one device, and one element pattern
pair (one join applies one pattern table; each element still sees its directions in its own radar's frame). A radar
whose orientation differs from another's is handled by turning each element's points into its own radar frame before
the stage, with the stage's frame the identity: one gather and one batched 3x3 product per point set per group.

### 3. A radar's own rows are its standalone rows

The receiving radar's own pairs are gathered out of the shared join (one `index_select` per row tensor per replay
group, tables built once per epoch on the host) into the composed batch a standalone session builds, in the same
order; that batch runs through the radar's own, unchanged instrument half. With isotropic elements a network radar's
cube is `torch.equal` to `Radar.simulate` in a still world and under each radar's own pose (`adc`, `chirp`): the
Channel rows of a pair do not depend on the other endpoints in the batch. With a directional pattern the cube agrees
to float32 rounding (measured 1e-7 relative): the standalone route turns a direction by the pose and then by the
authored frame inside the kernel, the network by the composed frame before it, and the two orders round differently.
A radar's composed rows carry the network's stable IDs, and `Result.last_propagation` the shared legs.

### 4. Cross-radar rows: the native cross beat family

A link's rows are the shared join's rows whose transmitter belongs to the transmitting radar and whose receiver to the
receiving one: site rows (bistatic target returns, transmitter -> site -> receiver through the receiver-oriented
outbound leg) and site-less rows of the third leg (`direct=True`: the straight segment between the two radars, the
direct interference path; `environment=True`: the environment's returns between them). They are synthesized by
`fmcw_cross_{forward,backward,jvp}` (in `fmcw_beat.cu`; phase owner `fmcw_cross_terms` in `fmcw_phase.cuh`), one
launch per replay group per receiving radar, reading the shared rows in place: each receive cell (slot, receive
element) reads its receive element's whole pair block, and a per-row link table (`row_link`, `-1` for a row not heard
this way, including the radar's own) selects the rows and the transmitter's ramp. No row is copied and no Python loop
runs over rows, cells or samples. The backward reads an inverse cell CSR (one thread per row, no atomics); the jvp is
the forward with tangents. The result is added to the receiver's beat frame before its receive chain
(`echo_frame(..., interference=)`), so the receiver's noise, LNA, AGC and ADC act on the sum.

The phase, for a row of round-trip delay `tau` heard at receive time `t` in the receiver's frame (`v` into the
receiver's ramp, `w` into the transmitter's ramp at the emission instant `t - tau - dT`, `dT` the transmitter's frame
start minus the receiver's):

```
cycles = phase0 + d_B t - d_A (t - tau) + S_B v^2 / 2 - S_A w^2 / 2,    beat = conj(C) exp(+j 2 pi cycles)
phase0 = frac((d_B - d_A) F_B)
```

with `S` the slopes [Hz/s], `d` the start offsets from the shared carrier [Hz], `F_B` the receiver's frame start in
world time. Every radar's oscillator runs at its start frequency from world time zero and each ramp adds its
quadratic over its own ramp time; the Channel weight `conj(C)` carries `+f_c tau`, so the carrier terms cancel
exactly. On one radar and one slot it is the monostatic beat `d tau + S tau (v - tau/2)`. The phase convention is a
declared model: a chirp generator whose phase accumulates the ramp's integral from chirp to chirp would add a
constant per chirp that this model does not carry.

Gates (discrete; the derivative is the almost-everywhere one, as for a pulse gate):

- the emission instant falls on a ramp of the transmitter's same-index frame, `0 <= w < ramp_end` (the transmitter is
  off during idle);
- that slot is the row's transmitter's TDM slot;
- the instantaneous beat frequency `d_B - d_A + S_B v - S_A w` lies within the receiver's IF band. This is the
  anti-alias filter as a gate on the instantaneous frequency (stationary phase of a linear-FM beat); the filter's
  transition ripple is not modelled. Monostatic rows are not gated (unchanged).

Oracles (`tests/oracle/test_fmcw_interference.py`, from the two signals in long double with the full carrier):
same-slope offset ghost at `S (tau + dT)` with zero Doppler; slope-mismatch burst exactly where
`|(S_B - S_A) v + S_A (tau + dT)| <= f_s/2`, `f_s / |S_A - S_B|` samples long and spread over the spectrum;
start-frequency offset shifting the ghost by `-d_A` and rotating it by `-2 pi d_A T_c` per chirp; synchronized
radars hearing a point target by the bistatic radar equation at beat `S (R_1 + R_2) / c`. Kernel forward, VJP and
jvp against the same signal oracle with a numerical `tau` derivative (`tests/test_fmcw_cross.py`); an end-to-end
bistatic VJP/JVP against a central difference (`tests/test_network.py`).

### 5. The site-less leg serves every pair; rows are kept per pair class

One site-less leg is discovered for all transmitter-receiver pairs. Its straight-segment rows are kept for a radar's
own pairs only with `coupling=True` and for a link's pairs only with `direct=True` (default true); its interacting rows
with `environment=True`. A straight row is one of depth 0, so `transmission` (straight rows with depth) is refused
together with any site-less leg in a network. Channel's endpoint-separation check runs only on the pair groups whose
straight rows are kept (`ChannelPropagationAdapter.discover(separation=)`): a monostatic radar with co-located
elements may join a network with `direct=True`, its own coincident pair evaluates to Channel's finite value and is
dropped.

## Not done (refused by name)

- OFDM and pulsed receivers or transmitters on a link: their cross responses need their own families (an OFDM
  receiver demodulating another radar's symbols, a pulsed gate against a foreign pulse). A link requires FMCW.
- DDM/BPM, multi-profile chirp sequences and ramp phase errors on a link.
- A non-radar emitter (a declared baseband waveform at a position): not cheap on this route, because the emitter is
  an endpoint without a radar record (no array, pose or frame) and the cross family reads ramps, not sample tables.
- Adaptive motion and `Motion.rediscover_every_chirps` in a network; `Motion.auto()` resolves to `adc` when anything
  moves.
- Oscillator phase noise on a receiving radar, wideband OFDM and Core-endpoint `endpoints=` in a network.
- A transmitter's previous or next frame is not heard: only its same-index frame.
