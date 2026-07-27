# Radar Pipeline Guide

The scene-driven entry point, end to end. This document replaces
`docs/fast_mimo_api.md`, whose whole subject - `mimo_from_trace`,
`path_cache_from_trace`, `mimo_from_paths`, `MimoPathCache`, `TraceResult` - was
deleted in the Phase-11 cutover. See
`docs/dev/migration/phase11-cutover-migration-note.md` for the break list.

```text
witwin.core.Scene / DynamicScene
        |
        |  Radar.simulate(scene, times=..., response=..., sites=...)
        v
    CompiledScene  ->  propagation legs  ->  two-way join  ->  waveform synthesis
        |
        v
RadarSimulationResult          witwin.radar.processing
   .cube [frame, TX, RX, slow, fast]   -> range_profile -> range_doppler
   .last_snapshot                      -> ca_cfar / os_cfar -> point_cloud
   .last_compiled_scene                -> music_image, beam_cube, microdoppler
   .last_propagation
   .last_radar_paths
```

Everything below runs; the three files under `examples/` are the executable
version of this document.

---

## 1. The world

The logical world is owned by `witwin.core`, not by Radar. There is no radar
`Scene` any more.

```python
import torch
from witwin.core import AntennaState, Mesh, PhysicalMaterial, Scene, Structure
from witwin.core.identity import reserve_antenna_id

mesh = Mesh(
    vertices=torch.tensor(((-2.0, -2.0, -5.0), (2.0, -2.0, -5.0),
                           (2.0, 2.0, -5.0), (-2.0, 2.0, -5.0)), dtype=torch.float32),
    faces=torch.tensor(((0, 1, 2), (0, 2, 3)), dtype=torch.int64),
    recenter=False,            # MANDATORY when the vertices are world coordinates
    fill_mode="surface",
    topology_diagnostics=False,
)
wall = Structure(
    geometry=mesh,
    material=PhysicalMaterial(name="concrete", eps_r=5.24, sigma_e=0.0462),
    structure_id=1, material_id=1, assignment_id=1, surface_id=1,
)
scene = Scene(
    structures=(wall,),
    endpoints=[AntennaState(reserve_antenna_id(770101), "tx",
                            torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32))],
)
```

`Mesh` defaults `recenter=True` and silently subtracts the bounding-box centre
from authored vertices. Nothing raises; the wall simply is not where you wrote
it. Always pass `recenter=False` for world-frame geometry.

A `Scene` with `structures=()` is legal and is the right world for a pure
point-target experiment: a scatter site is a declared endpoint, not geometry.

For a moving world wrap it in `witwin.core.dynamics.DynamicScene` and hand THAT
to `simulate`; a static `Scene` is wrapped automatically.

---

## 2. `Radar.simulate`

```python
result = radar.simulate(
    scene,
    times=(0.0, 1e-3, 2e-3),   # frame instants in seconds, non-empty
    response=response,         # REQUIRED scatter response
    sites=sites,               # ScatterSitePolicy; None -> structure_anchor()
    components=None,           # frozenset override for THIS solve
    max_depth=None,            # int override for THIS solve
    polarization=None,         # None -> (0, 0, 1)
    slow_time_mode=None,       # None -> FROZEN_WEIGHT_WITH_CARRIER_RATE
    ad_mode="none",            # "none" | "vjp" | "jvp"
    world_motion="frozen_world",
    motion_event_period_frames=None,
    ids=None,                  # StableIdAllocator
    antenna_pattern=None,      # AntennaPatternSpec; None applies no pattern
) -> RadarSimulationResult
```

`Radar.simulate_group` does not exist. `Radar.solve` has never existed.

### `response` has no default

The two-way join multiplies the round trip by the target's complex response, and
every possible default - unit amplitude, unit RCS, zero phase - is a statement
about how strongly the target scatters.

```python
from witwin.radar.scattering import ScalarRcsResponse

response = ScalarRcsResponse.from_rcs(
    1.0, reference_frequency_hz=radar.config.fc, device=radar.device
)
```

`from_rcs` is the only constructor that knows what a square metre is worth; it
applies `S = sqrt(4 pi sigma) / lambda`. `from_values` authors the dimensionless
strength directly. A 0-dim tensor cross section makes `sigma` itself an autograd
leaf. `AspectScatterResponse` is the aspect-dependent alternative.

### `sites` is a declaration, never a search

```python
from witwin.radar import ScatterSitePolicy

# Explicit world positions. A live tensor is passed through UNTOUCHED, so a
# requires_grad leaf keeps its graph into BOTH propagation legs.
sites = ScatterSitePolicy.explicit(
    torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32, device=radar.device)
)

# One site per selected structure, at the world anchor Core publishes for it.
sites = ScatterSitePolicy.structure_anchor(structure_ids=(1,))
```

`explicit` refuses a wrong dtype, device, shape or stride rather than silently
copying: a copy would leave you holding a tensor that is no longer the one the
legs differentiate through. `structure_anchor` reads
`StructureState.rigid_motion.translation`, so a structure with no rigid motion
has no Core-owned anchor and is refused by name. Deriving sites by SAMPLING a
mesh is a named deferral (R-ADR-020): a sampling rule is a geometry algorithm
and does not belong in a Torch expression in Radar.

### `components` and `max_depth`

`components` is a subset of `{"los", "reflection", "diffraction",
"transmission"}` and `max_depth` is at most 5, but **`simulate` freezes a
topology and replays it**, and only `{"los", "reflection"}` are freezable
today. Asking for `diffraction` or `transmission` here fails at freeze time with
a message naming the freezable set - it does not silently drop them.

Both keywords override the radar's stored propagation block for ONE solve
through `RadarSystemConfig.with_propagation`, which returns a new configuration
rather than mutating the radar's. The defaults are
`frozenset({"los", "reflection"})` and `max_depth=1`.

### `polarization`

Channel's endpoint polarization is a WORLD-frame vector that native material
evaluation projects the field onto. The default is `(0, 0, 1)`.

**A polarization parallel to the propagation direction radiates nothing and
every transport comes back exactly zero, with nothing raised.** A radar that
looks along world `z` - which is what `target=(0, 0, -1)`, the depth-camera
convention, gives you - must declare something else:

```python
result = radar.simulate(scene, ..., polarization=(0.0, 1.0, 0.0))
```

### Waveform kind

`RadarSystemConfig` supports `fmcw`, `ofdm` and `pulsed`, and
`RadarSimulationResult` derives its axis names from whichever one synthesized
the cube. The `Radar` OBJECT is still FMCW only - `Radar.__init__` builds the
FMCW axes record unconditionally - so a non-FMCW `Radar` is unconstructible.
Driving OFDM or pulsed synthesis today means calling
`witwin.radar.synthesis.synthesize_ofdm_cfr` / `synthesize_pulsed_echo` on a
composed batch directly.

---

## 3. `RadarSimulationResult`

Frozen dataclass, data only.

| member | meaning |
| --- | --- |
| `cube` | `[frame, TX, RX, slow, fast]` complex64, the product |
| `times_s` | the instants that were asked for |
| `kind` / `axes` | `"fmcw"` and `("frame", "tx", "rx", "chirp", "sample")` |
| `phasor` / `time_dependence` / `reference_frequency_hz` | the waveform owner's conventions |
| `epochs` | the topology epoch each frame ran in |
| `rediscovery_reasons` | per frame: `"first_frame"`, `"structure_motion"`, `"motion_event_cadence"`, `"source_mutation"` or `None` |
| `compile_count` / `discovery_count` | what the run actually paid for |
| `last_*` | the four diagnostics, see below |
| `frame_count` | property |

The pair axis of the cube is this array's TX x RX grid; the composer's own rank
is SINK major (`pair = rx_rank * num_tx + tx_rank`) and `assemble_frame_cube`
transposes it into the TX-major virtual-element layout every angle estimator
steers.

---

## 4. The four typed diagnostics

```python
radar.last_snapshot        # witwin.core.SceneSnapshot
radar.last_compiled_scene  # witwin.channel.scene.CompiledScene
radar.last_propagation     # witwin.radar.RadarPropagationLegs
radar.last_radar_paths     # witwin.radar.paths.RadarPathBatch
radar.last_result          # the RadarSimulationResult itself
```

They are `None` before the first `simulate` and are cleared by a failed one - a
stale world claiming to describe this radar is worse than nothing. They describe
the LAST frame, not the sequence: a compiled scene and a leg pair are per-epoch
and per-frame objects, and retaining every frame's would retain every frame's
device memory.

`last_propagation` is a typed `RadarPropagationLegs` with `.inbound` and
`.outbound`, not a tuple and not a dict, so a swapped leg is a refusal rather
than a mirrored answer.

**Retention.** These alias the frame's own device tensors and, when `ad_mode`
asked for a graph, that frame's autograd graph. None of them holds a tape;
`tests/test_phase9_tape_non_leak.py` walks all four to keep it that way.

---

## 5. Frozen topology replay - what supersedes `MimoPathCache`

`MimoPathCache` cached per-path range/amplitude tables so a frame could be
regenerated without re-tracing. The pipeline does the same thing structurally,
and better, without a cache object: `SceneEpochLoop` discovers the discrete path
topology ONCE per epoch and every later frame REEVALUATES the frozen rows at the
current endpoint positions.

`world_motion` is the declaration that decides the cadence, and it is an
assertion about the world, not a performance switch:

| `world_motion` | meaning |
| --- | --- |
| `"frozen_world"` (default) | a moved structure retires every frozen handle; each moving frame is its own epoch |
| `"fixed_winner_replay"` | the discrete winner set is held fixed while the geometry moves: one epoch, one freeze pair, per-frame recompile |

Replay is SUBTRACTIVE. A reflection row whose specular point walks off its facet
is published through the `row_valid` mask as a complete answer; a row that would
be BORN by the motion cannot be found by a replay at all, which is what
`motion_event_period_frames` exists to bound.

Measured on the six-frame still world in `tests/test_phase11_simulate_entry.py`:
`compile_count == 1`, `discovery_count == 1`, two freezes (one per leg). Under
`fixed_winner_replay` with a moving wall: three compiles, one discovery, two
freezes.

---

## 6. Processing

`witwin.radar.processing` is the post-processing facade and is PyTorch by owner
directive. `witwin.radar.sigproc` survives only as deprecation adapters that
reproduce the legacy conventions - a symmetric unnormalised window, an
unreconciled Doppler sign - and they warn. New code uses the facade.

Every stage reads ONE metadata record, `ProcessingAxes`. Building one from a
simulation result takes a detour today:

```python
from witwin.radar.processing import ArrayGeometry, ProcessingAxes, range_doppler, range_profile
from witwin.radar.synthesis import SlowTimeMode

# ProcessingAxes is built from a rank-3 SynthesisResult, and the simulation
# result publishes the ASSEMBLED rank-5 cube. Re-synthesizing the last frame's
# composed rows is the public route to one; the record carries shapes and
# conventions, which are the waveform spec's and are the same for every frame.
synthesis = radar.synthesize(
    radar.last_radar_paths, slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE
)
axes = ProcessingAxes.from_synthesis(
    synthesis, radar.system_config.waveform_spec(), radar.system_config.sensors.array
)
geometry = ArrayGeometry.from_axes(axes)

profile = range_profile(result.cube[0], axes=axes, window="hann")
rd = range_doppler(profile, window="hann")
```

Then detections and a point cloud:

```python
from witwin.radar.processing import ca_cfar_fast, point_cloud

combined = rd.data.reshape(geometry.sensor_pair_count, *rd.data.shape[-2:]).sum(dim=0)
detections = ca_cfar_fast(combined.abs(), guard_cells=(2, 4), training_cells=(4, 8), pfa=1e-4)
cloud = point_cloud(detections, rd, axes, geometry, route="phase_comparison", max_points=64)
```

`route` is explicit. The legacy dispatch on `num_tx` survives only inside the
`naive_xyz` adapter, because a change of front end that silently swaps the angle
estimator is a change of answer with no change of call.

Two conventions are carried as data rather than as documentation:

* `PROCESSING_DOPPLER_CONVENTION` - a positive Doppler bin is a CLOSING target,
  in every waveform. The FMCW beat cube is the conjugate of Channel's phasor and
  is reconciled exactly once, inside `range_doppler`.
* `PROCESSING_AMPLITUDE_CONVENTION` - maps are AMPLITUDE estimates: an isolated
  path row peaks at `|C_rt|` times the window's coherent gain. This is order
  `1e-7` sqrt(W) for a small target at a few metres. A decibel floor tuned to
  the old unnormalised transform (`sigproc` used `1e-6`) clips such a map to a
  uniform blank.

---

## 7. Differentiability

```python
sites = ScatterSitePolicy.explicit(positions.requires_grad_(True))
result = radar.simulate(scene, times=(0.0,), response=response, sites=sites, ad_mode="vjp")
result.cube.abs().square().sum().backward()
positions.grad          # reaches through BOTH legs
```

`ad_mode` is forwarded to every replay: `"none"` builds no graph, `"vjp"` makes
the cube differentiable, `"jvp"` drives forward mode. The site tensor is the
sink of the inbound leg and the source of the outbound one and the binding hands
the SAME object to both, which is what lets one leaf accumulate from both.
Scene-owned leaves (mesh vertices, `eps_r`, `sigma_e`) and the response
parameters are differentiable too; the full matrix is
`docs/dev/radar-ad-capability-matrix.md`.

---

## 8. Known limitations

These are named deferrals, not bugs to work around silently.

**Intra-frame Doppler is zero.** `simulate` composes the round trip ONCE per
frame, so the slow-time axis of one frame carries no motion and every return
lands in the zero-Doppler bin. Frame-to-frame motion IS fully modelled: each
frame re-resolves the world at its own instant. Unpacking a `delay_rate`
requires a forward-AD velocity dual authored from Core kinematics
(`witwin.radar.propagation.kinematics.two_way_duals`), and `simulate` does not
open one. Two consequences a caller must know:

* slow-time (static) clutter removal subtracts the WHOLE signal, because the
  chirps of one frame are identical;
* a MUSIC covariance built over slow-time snapshots is rank deficient, and its
  noise subspace comes from receiver noise plus spatial smoothing rather than
  from slow time.

**The flat configuration format does not carry the `frontend` block.**
`RadarConfig.frontend` exists and `Radar` honours it, but
`validate_radar_config` ignores a `"frontend"` key in the mapping accepted by
`RadarConfig.from_dict`. Attach the spec after validation:

```python
import dataclasses
from witwin.radar.frontend import FrontendSpec, NoiseSpec, SeedSpec

config = dataclasses.replace(
    RadarConfig.from_dict(CONFIG),
    frontend=FrontendSpec(
        noise=NoiseSpec(noise_figure_db=10.0, bandwidth_hz=4.4e6),
        seed=SeedSpec(20260727),
    ),
)
```

The legacy `noise_model` / `receiver_chain` pair is the only thing the flat
format can express, and configuring it alongside `frontend` is refused.

**Only `los` and `reflection` can be frozen**, so those are the only components
`simulate` accepts. Diffraction and transmission are available through the
consumer's non-frozen entry.

**Scatter sites cannot be derived from a mesh** (R-ADR-020); declare them.

---

## 9. Worked examples

| file | what it shows |
| --- | --- |
| `examples/single_point.py` | one target plus a wall; transport checked against the radar equation, range peak and multipath peak against closed forms, then CFAR and a point cloud |
| `examples/music_imaging.py` | 20 x 20 UPA, two targets, MUSIC image checked against the analytic target angles |
| `examples/rgbd_range_doppler.py` | a depth sequence back-projected into scatter sites, one `simulate` per output frame, Range-Doppler PNGs |

Each has a notebook twin. `examples/preprocess_rfgen_rd.py` converts a recording
into the `.npz` layout the RGBD example reads; no depth asset ships with this
repository.

```bash
python -m examples.single_point
python -m examples.music_imaging
python -m examples.rgbd_range_doppler --input path/to/depths.npz
```
