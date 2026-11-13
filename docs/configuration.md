# Configuration, saved data and the doctor

This page covers everything that moves a radar or its output in or out of a
Python process: the flat configuration format, TI mmWave profiles and board
presets, saved results, datasets with ground-truth labels, raw ADC captures in
the DCA1000 layout, and the installation self-check.

## The flat configuration

`Radar.from_dict(mapping, **overrides)` builds a radar and `radar.to_dict()`
writes one; `Radar.from_json(path)` / `radar.to_json(path)` do the same through
a file. The two directions round-trip exactly:
`Radar.from_dict(radar.to_dict(), device=radar.device) == radar`.

The waveform is declared one of two ways, never both:

* the TI-style FMCW keys in vendor units (`slope` MHz/us, `sample_rate` ksps,
  `adc_samples`, `chirp_per_frame`, `adc_start_time`, `idle_time`,
  `ramp_end_time` in microseconds, optional `output_domain`), which is the
  format to write by hand;
* a `waveform` block, `{"kind": "fmcw" | "ofdm" | "pulsed", ...}` whose other
  keys are the `Fmcw` / `Ofdm` / `Pulsed` field names in SI units. This is what
  `to_dict` writes, because SI values round-trip exactly; an FMCW chirp
  sequence is a list of `Chirp` field mappings under `sequence`.

Every other `Radar` field is a key of its own name. A receive-chain record is a
block of its field names (`{"figure": 8.0}` for `Noise(figure=8.0)`), a complex
number is `{"re": x, "im": y}`, and a vector is a three-element list. `device`
is not configuration and is always a keyword. Keyword overrides use the SI
field names and are applied last. `to_dict` leaves out fields at their default.

`witwin.radar.config.config_schema()` returns the JSON Schema (draft 2020-12),
generated from the same field table the reader checks keys against:

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `format` | string |  | optional format tag; 'witwin.radar.config/1' when written by Radar.to_dict |
| `num_tx` | integer |  | transmit element count; must equal len(tx_loc) |
| `num_rx` | integer |  | receive element count; must equal len(rx_loc) |
| `fc` | number | Hz | carrier (reference frequency) |
| `power` | number | dBm | transmit power |
| `tx_loc` | array | antenna_unit | transmit element positions, [x, y, z] each, radar-local (right, up, -boresight) |
| `rx_loc` | array | antenna_unit | receive element positions, as tx_loc |
| `antenna_unit` | string |  | 'half_wavelength' (default) or 'm' |
| `slope` | number | MHz/us | TI-style FMCW: ramp slope |
| `sample_rate` | number | ksps | TI-style FMCW: ADC sample rate |
| `adc_samples` | integer |  | TI-style FMCW: samples per chirp |
| `chirp_per_frame` | integer |  | TI-style FMCW: chirps per frame per transmitter |
| `adc_start_time` | number | us | TI-style FMCW: ramp start to first ADC sample |
| `idle_time` | number | us | TI-style FMCW: idle time between ramps |
| `ramp_end_time` | number | us | TI-style FMCW: ramp end time |
| `output_domain` | string |  | TI-style FMCW: 'spectrum' (default) or 'beat' |
| `waveform` | object | SI | {'kind': 'fmcw'|'ofdm'|'pulsed', <the waveform record's fields>} |
| `antenna_pattern` | object | deg, dBi | transmit (and default receive) element pattern |
| `rx_antenna_pattern` | object | deg, dBi | receive element pattern; absent is antenna_pattern |
| `noise` | object | SI | Noise record fields |
| `lna_gain` | number | dB | LNA voltage gain |
| `agc` | object | SI | Agc record fields |
| `adc` | object | SI | Adc record fields |
| `iq` | object | dB, rad | Iq record fields |
| `leakage` | object | dB, s, rad | Leakage record fields; scalars or [num_tx][num_rx] tables |
| `compression` | object | dBm, deg | Compression record fields |
| `dc_offset` | number or object or array | V | one complex value or one per receiver |
| `coupling` | object |  | Coupling record: tx and/or rx complex matrices |
| `impedance` | number | ohm | receive port reference impedance |
| `seed` | integer |  | receive-chain base seed |
| `position` | array | m | radar origin, world frame |
| `look_at` | array | m | boresight target point, world frame |
| `up` | array |  | world up vector completing the pose |
| `polarization` | string or array |  | 'up', 'right' or a world vector |
| `rx_polarization` | string or array or null |  | receive polarization; absent is polarization |

Element patterns (`antenna_pattern`, `rx_antenna_pattern`) are
`{"kind": "separable", "x_angles_deg", "x_values", "y_angles_deg", "y_values"}` or
`{"kind": "map", "x_angles_deg", "y_angles_deg", "values"}` (values `[y][x]`),
with optional `quantity` (`"power"`, or `"field"` for complex values) and
`gain` (dBi).

## TI mmWave profiles and board presets

`witwin.radar.config.read_ti_cfg(path_or_text)` reads a TI mmWave SDK `.cfg`
profile (`channelCfg`, `adcCfg`, `profileCfg`, `chirpCfg`, `frameCfg`,
`bpmCfg`; other commands are processing or GUI settings and are ignored) into a
`TiConfig`. `TiConfig.to_radar(tx=..., rx=..., power=12.0, **overrides)` builds
the radar once the board's element positions are given, indexed by channel
number; the `.cfg` names channels, not positions.

* Under TDM each chirp slot enables one transmitter, and the radar's
  transmitters are those in SLOT order: a loop firing TX0, TX2, TX1 becomes
  `tx = (tx[0], tx[2], tx[1])`.
* `bpmCfg` with every slot enabling the same two (or four) transmitters becomes
  `Fmcw(mimo="bpm")`.
* Per-chirp variations or several profiles become an `Fmcw.sequence`; a uniform
  frame has none. `adcCfg` output format 0 is `sampling="real"`.
* `TiConfig.frame_times()` gives the `frameCfg` frame instants.
* The `.cfg` does not state the transmit power; `power` is in dBm.

`witwin.radar.config.preset(name, **overrides)` builds an evaluation board:
`"awr1843boost"` (77 GHz) and `"iwr6843isk"` (60 GHz). Both use the boards'
3 TX x 4 RX layout (receivers half a wavelength apart, TX0 and TX2 two
wavelengths apart, TX1 midway and half a wavelength up: an 8-element azimuth
row and a 4-element elevated row) and a representative 256-sample, 64-loop TDM
profile; they are not TI's demo profiles.

## Saving results

`result.save(path)` writes a directory and `Result.load(path, device="cpu")`
reads it back; the cube and the processing axes come back bit for bit.

    path/metadata.json        format "witwin.radar.result", version, producer, every scalar
    path/axes.pt              the axes record's tensors
    path/frames/000000.pt     {"cube": [tx, rx, slow, fast], "sample_times_s": float64[...]}

Each `.pt` file holds plain tensors only and is read with
`torch.load(weights_only=True)`; no Radar, Channel or Core object is pickled.
The closing frame's world state (`last_snapshot`, `last_compiled_scene`,
`last_propagation`, `last_radar_paths`) and the path events are not saved; a
loaded result carries `None` and `()` there. `metadata.json` is written last.

## Ground truth and datasets

`witwin.radar.labels.ground_truth(radar, scene, targets, times=...)` returns
`TargetLabels`: per frame and per target, the world and radar-frame position,
range, closing-positive radial velocity, azimuth, elevation, monostatic cross
section and target ID, computed from the simulated geometry at each frame
instant (`GroundTruth.at(t)` for one instant). The radar frame is the point
cloud's (`x` right, `y` boresight, `z` up). A `MeshTargets` structure reports
its area-weighted facet centroid, its area and its facet range extent, and NaN
for the cross section. `site_id` / `site_target` map a composed row's site to
its target.

`witwin.radar.labels.path_summary(paths, frame, observation)` lists one traced
observation's composed rows: Radar component class, Channel leg components,
interaction depth, float64 delay, Doppler and amplitude.

`witwin.radar.dataset.write_dataset(root, radar, [Episode(scene, targets, times,
seed=...), ...], **stream_options)` streams every episode frame by frame and
writes one saved result per episode, each frame file also carrying its labels,
plus `root/dataset.json` (the radar's `to_dict`, the options and the episode
index). `load_episode(root / name)` returns `(Result, TargetLabels)`.

## Raw ADC captures (DCA1000)

`witwin.radar.rawdata.write_dca1000(path, result, adc=radar.adc)` (or
`scale=` codes per unit) writes a beat-domain FMCW result as a post-processed
DCA1000 capture: little-endian int16, no packet headers, frames then chirps in
transmission order (TDM slot `c * num_tx + t`), then

* `layout="xwr16xx"` (two LVDS lanes; xWR16xx/18xx/68xx), complex: receiver by
  receiver, samples in groups `I[n], I[n+1], Q[n], Q[n+1]`; real: receiver by
  receiver, `N` samples;
* `layout="xwr14xx"` (four lanes), complex: sample by sample, `I` of each
  receiver then `Q` of each receiver; real: sample by sample, each receiver.

`read_dca1000(path, num_tx=, num_rx=, chirps=, samples=, ...)` reads that layout
into `[frame, tx, rx, chirp, sample]`. The layouts follow TI's published capture
descriptions and `readDCA1000` reference script; they have not been checked
against hardware here.

## The doctor

`witwin-radar-doctor` (or `python -m witwin.radar doctor`) reports the Python,
PyTorch, CUDA and driver identity, the native library's build identity and
whether it carries SASS for the active GPU, the optional Channel installation
and its own doctor verdict, and runs a one-target simulate on the GPU. It prints
a verdict with the fix for every problem and exits 1 when Radar cannot run.
`--json` prints the whole record; `--no-smoke` skips the GPU simulate.
