# Multipath and the environment

Radar asks Channel for one-way propagation from each transmitter to each
scatter site and from each receiver to each site, then joins the two legs into
round trips through the site's scattering response. The world verbs choose
which propagation mechanisms the legs may use, and whether the environment's
own returns (a wall that echoes with no target on it) are composed at all.

| Keyword | Default | Meaning |
|---|---|---|
| `los` | `True` | the straight segment, where it is not blocked |
| `reflections` | `1` | specular reflections per leg (0 disables them) |
| `transmission` | `False` | paths through walls on the straight segment |
| `diffraction` | `False` | edge-diffracted paths |
| `environment` | `False` | add site-less transmitter -> environment -> receiver rows |
| `coupling` | `False` | add the elements' own straight-segment coupling (site-less) |
| `discovery` | `"auto"` | Channel's reflection candidate source: `"auto"`, `"exhaustive"` or `"rays"` |
| `max_paths` | `None` | bound on one Channel discovery's rows |
| `max_round_trips` | `None` | bound on one sensor pair's composed rows |
| `wideband` | `False` | per-subcarrier transport (OFDM CFR only) |

At least one of `los`, `reflections > 0`, `transmission` or `diffraction` must
be requested.

## Setup: a wall from an OBJ file and an ITU-R material

Scenes are plain `witwin.core` objects. Channel's importers read Mitsuba/Sionna
XML, PLY, OBJ and glTF files into Core structures, and its presets give
ITU-R P.2040 materials as Core `PhysicalMaterial`s; neither needs anything
from Radar. Importers keep the file's coordinates (`Mesh(recenter=False)`).

```python
import pathlib
import tempfile

import torch
from witwin.channel.importers import load_mesh, load_mitsuba
from witwin.channel.presets import itu_material, itu_material_names
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure
from witwin.radar import Fmcw, Motion, Ofdm, PointTargets, Radar

chirp = Fmcw(slope=60e12, sample_rate=4.4e6, samples_per_chirp=256, chirps_per_frame=16,
             adc_start=6e-6, idle=7e-6, ramp_end=65e-6)
radar = Radar(reference_frequency_hz=77e9, waveform=chirp, power=15.0, tx=((0.0, 0.0, 0.0),), rx=((0.0, 0.0, 0.0),))

# A 6 m x 4 m wall 6 m down the boresight, facing the radar, written as an OBJ file.
folder = pathlib.Path(tempfile.mkdtemp())
(folder / "wall.obj").write_text("o back_wall\nusemtl concrete\n"
                                 "v -3 -2 -6\nv 3 -2 -6\nv 3 2 -6\nv -3 2 -6\nf 1 2 3\nf 1 3 4\n")
concrete = itu_material("concrete", 77e9)  # eps_r and sigma from the ITU-R P.2040 power law at 77 GHz
imported = load_mesh(folder / "wall.obj", concrete)
scene = Scene(structures=imported.structures)
print("ITU materials:", ", ".join(itu_material_names()))
print(imported.report[0])
target = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0)


def strongest(result, count):
    """[(range m, power)] of the `count` strongest range bins of frame 0, strongest first."""
    profile = result.frame(0).range_profile().data.abs().square().sum(dim=(0, 1, 2))
    top = torch.topk(profile, count)
    return [(round(float(result.axes.range_m[i]), 2), float(p)) for p, i in zip(top.values, top.indices)]
```

`load_mesh(path, materials)` takes one material for the whole file or a
mapping from the file's material names (OBJ `usemtl`, glTF material names;
`None` is the default entry). `itu_material(name, frequency_hz)` must be
inside the ITU validity range of that material; with `dispersive=True` (the
default) it carries the ITU power law and is evaluated at the compile
frequency.

A Mitsuba 3 / Sionna RT XML scene works the same way; `rectangle` and `cube`
shapes need no mesh file, and an `itu-radio-material` bsdf becomes the ITU
preset:

```python
(folder / "room.xml").write_text("""<scene version="2.1.0">
  <bsdf type="itu-radio-material" id="mat-concrete"><string name="type" value="concrete"/></bsdf>
  <shape type="rectangle" id="back_wall">
    <transform name="to_world"><scale x="3" y="2"/><translate z="-6"/></transform>
    <ref id="mat-concrete"/>
  </shape>
</scene>""")
room = load_mitsuba(folder / "room.xml", frequency_hz=77e9)
print([structure.name for structure in room.structures], room.warnings)
```

## Target returns and the environment's own returns

By default only round trips through a declared target are composed: with
`reflections=1` the target's echo may bounce off the wall, but the wall
alone returns nothing. `environment=True` adds a third, site-less leg,
transmitter -> environment -> receiver, for the requested interacting
components (reflections and diffraction); its rows carry `site_id == -1`.

```python
targets_only = radar.simulate(scene, target, times=(0.0,), reflections=1)
with_wall = radar.simulate(scene, target, times=(0.0,), reflections=1, environment=True)
print("target only:      ", [r for r, _ in strongest(targets_only, 1)])
print("environment=True: ", [r for r, _ in strongest(with_wall, 2)])
assert strongest(with_wall, 1)[0][0] == 6.01  # the wall's own echo is the strongest return
```

`coupling=True` adds the straight segment between the radar's own transmit and
receive elements (and transmission through whatever lies on it, when
requested). It is refused for co-located elements, where free-space transport
is undefined:

```python
try:
    radar.simulate(scene, target, times=(0.0,), coupling=True)
except ValueError as error:
    print("refused:", str(error)[:100], "...")
separated = radar.replace(rx=((0.01, 0.0, 0.0),))
print(separated.trace(Scene(), target, times=(0.0,), reflections=0, coupling=True).component_index().counts)
```

### Component classes

`Paths.component_index(frame=0, observation=-1)` classifies each composed row
as `target`, `environment_clutter`, `direct_leakage` or `multi_interaction`
(deeper than one interaction per leg). `Paths.select(name)` makes every other
row inert, so `radar.echo(paths.select(name))` synthesizes one class alone,
through the same kernels; because synthesis is linear in the rows, the class
echoes sum to the full echo. A `ComponentDeclaration`
(`witwin.radar.paths`) reclassifies sites and material slots.
`Paths.select` refuses an adaptive trace, whose rows live in one
interpolation table.

```python
paths = radar.trace(scene, target, times=(0.0,), reflections=1, environment=True)
index = paths.component_index()
print(dict(zip(index.names, index.counts)))
parts = [radar.echo(paths.select(name)).cube for name in index.names]
full = radar.echo(paths).cube
assert torch.allclose(sum(parts), full, rtol=1e-5, atol=1e-12)
print("clutter alone:", [r for r, _ in strongest(radar.echo(paths.select("environment_clutter")), 1)])
```

## Transmission and diffraction

`transmission=True` adds paths through walls on the straight segment, with
each wall's slab (or layer stack) transmission coefficient. `diffraction=True`
adds edge-diffracted paths (uniform theory of diffraction). Channel's single
depth bound also limits how many walls a transmitted path crosses and how many
edges a diffracted one meets, so these keep at least one interaction even with
`reflections=0`.

Channel evaluates both only at discovery and cannot freeze them for replay, so
they require:

- `ad_mode="none"` (refused otherwise: `components ['transmission'] exist only
  at their discovery instant and carry no derivative ...`), and
- a discovery at every evaluated instant, which is the default; a declared
  discovery cadence (`Motion.adc(rediscover_every_frames=...)` and the others)
  is refused, and so is `wideband=True`.

```python
hidden = PointTargets(positions=[(0.0, 0.0, -8.0)], rcs=1.0)  # 2 m behind the wall
blocked = radar.simulate(scene, hidden, times=(0.0,), reflections=0)
through = radar.simulate(scene, hidden, times=(0.0,), reflections=0, transmission=True)
print(f"behind the wall: {float(blocked.cube.abs().max()):.1e} without transmission, "
      f"peak at {strongest(through, 1)[0][0]} m with it")
assert float(blocked.cube.abs().max()) == 0.0 and strongest(through, 1)[0][0] == 7.99
print("transmission_dropped_rows:", through.transmission_dropped_rows)

edge_wall = Structure(geometry=Mesh(vertices=torch.tensor([[-3.0, -3.0, -3.0], [0.0, -3.0, -3.0],
                                                           [0.0, 3.0, -3.0], [-3.0, 3.0, -3.0]]),
                                    faces=torch.tensor([[0, 1, 2], [0, 2, 3]]), recenter=False),
                      material=PhysicalMaterial.perfect_conductor())
shadowed = PointTargets(positions=[(-0.3, 0.0, -6.0)], rcs=1.0)  # in the shadow of the wall's edge x = 0
for diffraction in (False, True):
    rows = radar.trace(Scene(structures=(edge_wall,)), shadowed, times=(0.0,), reflections=0,
                       diffraction=diffraction).rows()
    print(f"diffraction={diffraction}: {rows.path_count} rows")
try:
    radar.simulate(scene, hidden, times=(0.0,), reflections=0, transmission=True, ad_mode="vjp")
except NotImplementedError as error:
    print("refused:", str(error)[:90], "...")
```

`Result.transmission_dropped_rows` is the largest number of source-sink pairs
any one discovery left without a transmission path because their straight
segment crosses more walls than the depth bound admits; it is 0 for a run
without transmission. Raise `reflections` to admit more walls.

## Discovery source and completeness

`discovery=` is Channel's reflection candidate source. `"exhaustive"` builds
the complete image tree; `"rays"` uses ray-launched candidates, which can miss
paths; `"auto"` (default) uses the image tree within a cost budget and falls
back to rays beyond it. Any discovery that is not provably complete clears
`Result.discovery_complete`, and `path_set_complete` is ANDed with it (see
[Motion](motion.md#two-completeness-statements) for the other half).

```python
for source in ("auto", "exhaustive", "rays"):
    result = radar.simulate(scene, target, times=(0.0,), reflections=1, environment=True, discovery=source)
    print(f"{source:10s} discovery_complete={result.discovery_complete} path_set_complete={result.path_set_complete} "
          f"semantics={result.semantics}")
```

Completeness is always relative to the request: complete means no path within
the requested components, depth and discovery policy was missed, not that
every conceivable propagation mechanism was modelled. The mesh-target
limitation in [Targets](targets.md#known-limitations-of-mesh-targets) is a
case where Channel reports a complete discovery that lost rows.

## Bounds: `max_paths` and `max_round_trips`

Both are refusals, never truncations:

- `max_paths=n` bounds one Channel discovery; a discovery with more rows raises
  a `RuntimeError` from Channel.
- `max_round_trips=n` bounds the rows one sensor pair would compose (one row
  per inbound row times outbound row through the same site); it is checked on
  the host before the join and names the sensor pair.

```python
for keyword in ({"max_paths": 1}, {"max_round_trips": 1}):
    try:
        radar.simulate(scene, target, times=(0.0,), reflections=1, **keyword)
    except (RuntimeError, ValueError) as error:
        print(f"{keyword}: {type(error).__name__}: {str(error)[:90]} ...")
```

## Wideband OFDM

By default every row's transfer is evaluated at the reference frequency and
the waveform applies the exact delay phase across its band. With
`wideband=True` an OFDM CFR radar asks Channel for each subcarrier's transport
(`f_ref + n * df`), so spreading loss and the material and layer response
across the band are exact. It replays every frozen row at every subcarrier,
so it costs about one transport evaluation per subcarrier. It is refused for
FMCW, pulsed and time-domain OFDM (`output="time"` or a modulated stream), and
with a time-varying `Aspect.axis` or `pose=`.

```python
ofdm = Ofdm(subcarrier_spacing=4e6, num_subcarriers=128, cyclic_prefix=0.1e-6, num_symbols=8,
            max_expected_delay=0.08e-6)
ofdm_radar = radar.replace(waveform=ofdm)
slab = Structure(geometry=Mesh(vertices=torch.tensor([[-3.0, -3.0, -5.0], [3.0, -3.0, -5.0],
                                                      [3.0, 3.0, -5.0], [-3.0, 3.0, -5.0]]),
                               faces=torch.tensor([[0, 1, 2], [0, 2, 3]]), recenter=False),
                 material=PhysicalMaterial(eps_r=4.0, thickness_m=0.02))
near = PointTargets(positions=[(0.0, 0.0, -2.0)], rcs=1.0)
for wideband in (False, True):
    cfr = ofdm_radar.simulate(Scene(structures=(slab,)), near, times=(0.0,), environment=True, wideband=wideband)
    edge = cfr.cube[0, 0, 0, 0, [0, 127]].abs()
    print(f"wideband={wideband}: |CFR| at subcarriers 0 and 127 = {edge[0]:.3e}, {edge[1]:.3e}")
try:
    radar.simulate(scene, target, times=(0.0,), wideband=True)
except NotImplementedError as error:
    print("FMCW refused:", str(error)[:80], "...")
```

The two runs agree at subcarrier 0, which is the reference frequency; they
differ across the band because the 2 cm slab's reflection coefficient varies
with frequency.

See also: [Targets](targets.md) for mesh targets, whose multi-bounce returns
need `reflections`; [Motion](motion.md) for discovery cadences;
[Waveforms](waveforms.md) for OFDM.
