# R-ADR-055: User tooling - saved results, ground truth, configuration I/O, raw export and the doctor

Status: Accepted (0.5.0)

## Context

A `Result` could not be saved, its `repr` printed every sample of the cube,
`witwin.radar.__version__` was `None` (U-9). There was no ground truth to
score a detector against, no summary of the composed paths, and every dataset
loop was hand-written (U-10). The flat configuration was FMCW-only and one-way:
no `to_dict`, no schema, no TI `.cfg` reader (U-13). There was no CLI or
installation check, no device preset and no raw-data export (F-8). Channel
0.6 ships `witwin-channel-doctor`; the family should have one shape.

## Decision

**One module per concept, all outside the hot path.** `storage` (saved-result
format), `labels` (ground truth, path summary), `dataset` (streaming writer),
`config` (TI profiles, presets, schema), `rawdata` (DCA1000), `doctor`
(+ `__main__`), `display` (compact repr). None of them computes physics: labels
read the same site policy, Core snapshot and pose the session reads, the path
summary reads the composed rows the session published. Every per-target or
per-row quantity is a fixed number of batched tensor calls per instant.
`results.py` gains thin facades only (`Result.save`, `Result.load`, `__repr__`);
`simulation.Paths` gains `row_identity` and `observation_times`, two read-outs
of state it already holds.

**Saved results.** A directory: `metadata.json` (format
`"witwin.radar.result"`, integer `version`, producer version, every scalar),
`axes.pt` and one `frames/NNNNNN.pt` per frame. Every `.pt` is a flat mapping of
plain tensors read back with `torch.load(weights_only=True)`, so no Radar,
Channel or Core object is pickled and a file cannot run code when read. One
file per frame is what lets a writer stream; `metadata.json` is written last
through a rename, so its absence marks an interrupted write. The closing
frame's world handles (`last_*`) and the path events are live state, not data,
and are not saved: a loaded result carries `None`/`()` (the `last_propagation`
and `last_radar_paths` annotations became optional). Overwrite is refused
unless asked, and never clears a directory the format did not write. `Paths`
is not saved: it holds the session (compiled scenes, frozen joins), so a saved
form would be a second session format rather than a cheap addition.

**Compact repr.** `display.compact_repr` prints a tensor as
`dtype[shape] device`, a long tuple by its ends and length, a nested record by
its class name; it reads host metadata only and never synchronises. `Result`,
`Frame`, `Paths`, `SynthesisResult` and every processing product use it.

**Version.** `witwin.radar.__version__ = deployment.package_version()`: a
source checkout is read from its own `pyproject.toml` (the metadata on
`sys.path` may describe a different installed copy), a wheel from its
distribution metadata.

**Ground truth.** `GroundTruth(radar, scene, targets, pose=).at(t)` /
`ground_truth(..., times=)` publish `TargetLabels`, `[frame, target]`, float64:
world and radar-frame position, range, radial velocity, azimuth, elevation,
monostatic RCS, target ID, and `site_id -> site_target` so a composed row's
`site_id` resolves to a target. Conventions are the point cloud's: radar frame
`(right, boresight, up)`, `azimuth = atan2(x, y)`, `elevation =
atan2(z, hypot(x, y))`, range one-way from the radar position, radial velocity
`-d(range)/dt` closing positive. The velocity is the central difference over
`t +- velocity_step_s` (1 ms default) from two more world snapshots, so the
radar's pose motion is in it; with float32 positions it is good to about
`1e-6 m / 2 ms = 5e-4 m/s`. RCS: a scalar as declared, an `Aspect` lobe at
the target-to-radar direction (`sigma * max(u.a, 0)^(4n)`), an `RcsTable` by
interpolating the complex amplitude in theta exactly as the scatter response
does (R-ADR-045) and squaring, a `Fluctuation` at its mean. A mesh target
aggregates per structure: area-weighted facet centroid, area, facet range
extent; its RCS is NaN, because the facets' coherent sum is the simulation.

**Path summary.** `path_summary(paths, frame, observation)`: per composed row
the Radar component class, the two legs' Channel component IDs, the summed
interaction depth (from the composer's host identity table - no device read),
float64 delay (with the R-ADR-040 residual), amplitude and phase, and Doppler
`-f_ref d(tau)/dt` from the frame's first and last observations when they share
one frozen topology. A once-evaluated (`static`) frame is held still and has
exactly zero Doppler; an adaptive frame retains one observation and reports NaN.

**Dataset writer.** `write_dataset(root, radar, episodes, **stream_options)`
runs each `Episode(scene, targets, times, seed, name, pose, options)` through
`Radar.stream`, writing each frame (cube, observation instants, labels) as it
arrives; each episode directory is a saved result, `root/dataset.json` (the
radar's `to_dict`, the options, the index) is written last. A seed is
`Radar.replace(seed=)`. `grad` is refused.

**Configuration I/O.** The flat format stays owned by `radar.py` and now has
both directions. The waveform is the TI-style vendor-unit FMCW keys (hand
written) or a `waveform` block of the record's SI fields (what `to_dict`
writes, because SI round-trips exactly); every other `Radar` field is a key of
its own name, records as blocks of their field names, complex numbers as
`{"re", "im"}`. `device` stays a keyword. `CONFIG_FIELDS` is the one table the
reader's key check and `config_schema()` (JSON Schema 2020-12) are generated
from; `docs/configuration.md` carries it as the field table. The old refusal
of a `waveform` key becomes a refusal of a waveform declared twice.

**TI profiles and presets.** `read_ti_cfg` reads `channelCfg`, `adcCfg`,
`profileCfg`, `chirpCfg`, `frameCfg`, `bpmCfg` and refuses advanced frames;
`TiConfig.to_radar(tx=, rx=, power=)` maps a TDM loop's slots to the radar's
transmitters in SLOT order (the radar's slot `s` is its transmitter `s`,
R-ADR-047), a BPM loop (same mask every chirp, power-of-two count) to
`mimo="bpm"` with `chirps_per_frame = loops x slots`, per-chirp variations to a
`Chirp` sequence, `adcCfg` format 0 to `sampling="real"`. Units are converted
there and nowhere else. `preset("awr1843boost" | "iwr6843isk")` is a board
layout (RX at 0..3 half wavelengths, TX0/TX2 4 apart, TX1 midway one up) plus a
representative profile run through the same importer. The `.cfg` gives no
transmit power; it is a keyword (12 dBm default).

**DCA1000.** Post-processed capture (packet headers stripped), little-endian
int16, frame then chirp in transmission order (TDM slot `c * num_tx + t`),
then: two-lane devices (xWR16xx/18xx/68xx) complex as per-receiver
`I[n], I[n+1], Q[n], Q[n+1]` groups, real as per-receiver samples; four-lane
xWR14xx complex as per-sample `I` of every receiver then `Q`, real as
per-sample receivers. `I + jQ` is the cube's own (beat) phasor; `iq_swap`
writes Q first. Scale is codes per unit, or `1 / Adc.step` so a quantised cube
is its exact codes; out-of-range codes are refused unless `clip=True`. These
layouts are reproduced from TI's published capture descriptions and the
`readDCA1000` reference script, not verified against hardware here.

**Doctor.** `doctor.py` (not `deployment.py`, which the package root imports
for the version; a smoke test there would put the whole simulator in the root
import graph) reports Python/PyTorch/CUDA/driver identity, the native library's
validated identity and SASS coverage (`deployment.runtime_failures`, which
`require_supported_runtime` now also uses), the optional Channel version
against the `channel` extra's range and Channel's own doctor verdict (run in a
child process: Radar never imports Channel outside `channel.py`, and a Channel
that aborts must not take the radar report with it), and a one-target simulate
on the GPU checked against its range bin. Each failure carries its fix; exit 1
when Radar cannot run. The report shape follows `witwin-channel-doctor`.

## Evidence

* `tests/test_result_storage.py`: FMCW, BPM (transmit code) and pulsed
  (matched-filter replica) results round-trip field for field, cube and axes
  `torch.equal`; every `.pt` loads with `weights_only=True`; overwrite and
  version refusals; repr length and no tensor dumps; a simulated chirp-sampled
  result round-trips on the GPU and its RD map is bit-identical.
* `tests/test_labels.py`: moving point targets (range, radial velocity,
  azimuth, elevation within float32 position precision; aspect-lobe RCS), a
  translating and rotating radar pose, an `RcsTable` off a knot, a structure on
  a `LinearTrajectory`, a mesh plate's area and centroid, and the path summary's
  delay (2R/c) and Doppler (2 v_r f / c, rel 2e-3) against the labels.
* `tests/test_dataset_writer.py`: two episodes with different seeds; frame
  `i - 1` is on disk when frame `i` is requested; cubes `torch.equal` to
  `radar.replace(seed=).simulate(...)`; labels equal `ground_truth`.
* `tests/test_config_io.py`: every `Radar` field, every waveform kind, DDM and
  a chirp sequence round-trip through JSON text to an equal radar; schema
  covers every written key; TI TDM slot order, slope variation, real ADC, BPM;
  refusals; presets' 8 + 4 virtual array.
* `tests/test_dca1000_export.py`: both layouts value by value against
  hand-written orderings; write/read round trips (complex, real, IQ swap);
  exact ADC codes; clip refusal.
* `tests/test_doctor.py`: report and verdict, the fix text per failure, the
  Channel range equals the extra's; a live `python -m witwin.radar doctor
  --json` run on the GPU passes its smoke test.

## Consequences

* `Result.last_propagation` / `last_radar_paths` are `Optional` in the type.
* The flat configuration accepts every `Radar` field; a mapping with both a
  `waveform` block and TI keys, or an unknown key, is refused by name.
* No hot-path code changed; no native operator was added.
