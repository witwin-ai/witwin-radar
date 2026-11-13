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

`witwin.radar.Network(radars, links=(), clock_offsets=(), if_bandwidths=(), emitters=(), carrier_offsets=())`
holds the radars and any declared emitters (section 6). Its verbs mirror the
radar's: `trace(scene, targets, times=, poses=...) -> NetworkPaths`, `echo(paths) -> tuple[Result, ...]`,
`simulate(...) -> tuple[Result, ...]` and `stream(...) -> Iterator[tuple[Result, ...]]`, one `Result` per radar in
`radars` order. `poses` is one `pose(t)` (or `None`) per radar, the same contract as `Radar.simulate(pose=)`; the
other keywords are the radar verbs' (`los`, `reflections`, `transmission`, `diffraction`, `environment`, `coupling`,
`discovery`, `max_paths`, `max_round_trips`, `motion`, `grad`) plus `direct` (below). A record rather than
`Radar.simulate` on a tuple, because links, clocks and IF bands are properties of the set, not of any one radar, and
because the per-radar results are a tuple a caller indexes like the `radars` it passed.

`links` lists `(transmitting radar, receiving radar)` pairs. Every radar always hears its own transmitters; a link adds
another radar's. `clock_offsets[k]` is radar `k`'s frame start relative to `times` (its frame `f` starts at
`times[f] + clock_offsets[k]`). `if_bandwidths[k]` is an FMCW radar `k`'s IF band, default half its ADC rate.
`carrier_offsets[k]` is a pulsed or OFDM radar's oscillator offset from the shared carrier (an FMCW radar declares a
start offset as a one-profile `Chirp(start_offset=...)` instead, which its own rows carry too; a nonzero network entry
for it is refused). A carrier offset acts on cross rows only; the radar's own rows keep its nominal carrier, a phase
error of `offset * tau` per row (1e-3 cycle for 10 kHz at 100 ns).

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

### 4. Cross-radar rows: the FMCW cross beat family

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
cycles = phase0 + d_B t - d_A (t - tau) + S_B v^2 / 2 - tx_cycles(w),    beat = conj(C) exp(+j 2 pi cycles)
phase0 = frac((d_B - d_A) F_B)
```

with `S` the slopes [Hz/s], `d` the start offsets from the shared carrier [Hz], `F_B` the receiver's frame start in
world time and `tx_cycles` the transmitter's own modulation phase at the emission instant (`S_A w^2 / 2` for a ramp). Every radar's oscillator runs at its start frequency from world time zero and each ramp adds its
quadratic over its own ramp time; the Channel weight `conj(C)` carries `+f_c tau`, so the carrier terms cancel
exactly. On one radar and one slot it is the monostatic beat `d tau + S tau (v - tau/2)`. The phase convention is a
declared model: a chirp generator whose phase accumulates the ramp's integral from chirp to chirp would add a
constant per chirp that this model does not carry.

Gates (discrete; the derivative is the almost-everywhere one, as for a pulse gate):

- the emission instant falls on a ramp of the transmitter's same-index frame, `0 <= w < ramp_end` (the transmitter is
  off during idle);
- that slot is the row's transmitter's TDM slot;
- the instantaneous beat frequency `d_B - d_A + S_B v - nu_A` lies within the receiver's IF band (`nu_A = S_A w` for a
  ramp, the table's phase-step frequency for an emitter, section 6). This is the
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

### 6. Sampled receivers, OFDM and pulsed transmitters, and emitters

The routing, the three launches and the chain rule of every cross family are one owner, `cuda/cross_routing.cuh`:
a family supplies only a phase functor that, for one row at one sample, says whether the row is heard and returns its
unit-weight contribution `u` and `du/dtau`; the cell value is `sum W(C) u` with `W` the conjugate (FMCW beat) or the
identity (Channel convention, sampled receivers). The receive schedule carries each cell's slot start, first-sample
offset and the receiver's own LO offset (a stepped-frequency pulse train's `l * step`). The Python bridge is one
autograd function, `synthesis.assembly.synthesize_cross`, with each family's operators.

A pulsed or OFDM receiver samples what it hears on its own grid (pulses with their range gate, OFDM post-prefix
samples at `1/(N df)`) and mixes it down with its own LO; its matched filter or OFDM demodulator then works against
its own replica or symbols downstream, unchanged. For a transmitter of offset `d_A` (pulse `q` at `d_A + q step_A`):

```
y = C s_A(w) exp(+j 2 pi cycles),    cycles = (d_A + f_q)(F_B + t - tau) - (d_B + lo)(F_B + t) + arg s_A(w)
```

`s_A` is the transmitter's own baseband on its own clock: `pulsed_cross_*` (in `pulsed_echo.cu`, envelope owner
`pulse_envelope`, now shared with the echo, which is bit-identical) evaluates a pulse train (rect, LFM, coded, stepped,
periodic) or an emitter table; `ofdm_cross_*` (in `ofdm_time.cu`, envelope owner `envelope_at`) the transmitter's
symbol stream from its own symbol table, so a timing offset against the prefix (ISI) and a carrier offset (ICI) follow
from the samples. With `A = B` both reduce to the echo kernels' `-(carrier + f_l) tau`. A hearing OFDM radar is
synthesized in the time domain (its declared CFR is then the demodulated estimate). A sampled receiver has no IF
model, so a transmitter whose band exceeds its complex sample rate, or whose carrier is offset by more than half of
it, is refused by name, and a hearing sampled radar has one transmitter (its several transmitters' channels are
modelled as ideally orthogonal, which another radar's signal is not). An FMCW transmitter into a sampled receiver and
a pulsed or OFDM transmitter into an FMCW receiver are refused by name: either needs the receiver's filter response to
a foreign sweep.

An `Emitter(position, samples, rate, power, carrier_offset, start, repeat, polarization)` is a non-radar transmitter:
a declared complex baseband (each sample at most the full amplitude) at a fixed position, its oscillator at the
carrier plus `carrier_offset` from world time zero, sample 0 leaving at `start`, repeated or sent once. Its table is
the arbitrary pulse family's chip table with each chip's frequency set to its phase step to the next sample
(magnitude held, phase linear between samples), so a sampled tone is a tone and the FMCW IF gate has an instantaneous
frequency to read. Emitters are appended to the shared transmitter list (one endpoint each, isotropic, so a network
with emitters needs isotropic radars), heard by every radar through sites, the direct line of sight (`direct`) and
the environment, and add no Channel call: they are rows of the same three legs. An FMCW receiver hears an emitter
through `fmcw_cross` (link kind 1), a sampled receiver through `pulsed_cross` (a coded pulse of one table length,
unbounded when it repeats).

Oracles (`tests/oracle/test_cross_waveform_interference.py`, at the Friis level): an OFDM receiver demodulating an
identical OFDM radar offset 0.1 us inside its prefix gives `C exp(-j 2 pi n df (tau + dT))` on every subcarrier and
symbol (2e-3 of the amplitude); a 0.2-subcarrier carrier offset gives the signal-to-ICI ratio
`|I0|^2 / (1 - |I0|^2)` within 0.5 dB and the signal at `sqrt(|I0|^2)` within 2 %; a pulsed receiver samples an offset
LFM pulse `C s(t - tau - dT)` sample by sample; an FMCW receiver hears a CW emitter as a 10-sample burst exactly where
`|S v - d_E| <= f_s/2`; a pulsed receiver hears an emitter's sampled tone at `f_0 + d_E` (1e-5). Kernels:
`tests/test_cross_families.py` (pulsed LFM with stepped carrier and coded chips, OFDM with ISI and a carrier offset,
an FMCW receiver hearing an emitter table through its IF band) against long-double signal oracles, forward, VJP and
JVP. Routing: sampled receivers under `adc` and slow-time schedules equal their still-world cube; emitters and links
add no Channel call; an end-to-end pulsed bistatic VJP/JVP against a central difference.

## Not done (refused by name)

- An FMCW transmitter heard by a pulsed or OFDM receiver, and a pulsed or OFDM transmitter or another waveform heard
  by an FMCW receiver (both need the receiver's filter response to a foreign sweep).
- A sampled receiver hearing a band wider than its sample rate or offset by more than half of it; a hearing sampled
  radar with several transmitters.
- DDM/BPM, multi-profile chirp sequences and ramp phase errors on an FMCW link.
- Emitters with directional radar elements; moving emitters.
- Adaptive motion and `Motion.rediscover_every_chirps` in a network; `Motion.auto()` resolves to `adc` when anything
  moves.
- Oscillator phase noise on a receiving radar, wideband OFDM and Core-endpoint `endpoints=` in a network.
- A transmitter's previous or next frame is not heard: only its same-index frame.
