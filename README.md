# WiTwin Radar

WiTwin Radar is a GPU-accelerated, differentiable radar simulator. A simulation consumes a `witwin.core` world and one-way propagation from `witwin.channel`, composes round trips, applies scattering and sensor/frontend effects, synthesizes radar waveforms, and produces typed signal-processing products.

The repository uses a breaking, concept-axis architecture. Compatibility modules and deprecated aliases are intentionally not retained.

For dynamic FMCW, `Radar.simulate(..., motion=Motion.adaptive(...))`
uses phase-controlled temporal interpolation and batched propagation. Import `Motion` from
`witwin.radar`. Per-ADC sampling remains the exhaustive reference and is what the default
`Motion.auto()` resolves to whenever anything moves; adaptive topology
probes cannot certify arbitrarily brief events between observations. Receiver effects operate on beat samples
before the requested range transform, and shared oscillator noise uses actual timestamps and path delays.
Adaptive replay reuses a complete LOS topology in an empty world and batches propagation, joins,
antenna weighting and ADC synthesis. The measured 128-sample × 1024-chirp rotor scene now takes
about 0.61 s on RTX 5080; see [performance evidence and limits](PERFORMANCE.md).

## Installation and runtime

```bash
pip install witwin-radar[channel]
```

CUDA is required for propagation and native waveform synthesis. CPU construction and PyTorch signal-processing workflows remain useful without a GPU. Linux and Windows are supported.

Release policy is explicit and conservative:

- CPython 3.10-3.14;
- Linux wheels target `manylinux_2_28_x86_64`;
- one packaged `_radar_native` library plus its identity sidecars;
- exact Torch/CUDA/ABI runtime identity;
- no JIT fallback and no success-by-loader-refusal path.

The release build matrix currently uses Torch 2.10 with CUDA 12.8 for each supported Python version. A different Torch or CUDA identity requires a separately built and validated artifact.

## Architecture

```text
witwin.core Scene/DynamicScene
        |
        v
Radar simulation session -> Channel propagation -> round-trip path composition
        |                                      |
        +-> scattering -> sensor/frontend -----+
                                               v
                       FMCW / OFDM / pulsed synthesis
                                               v
                          typed processing products
```

Production ownership is intentionally shallow:

- `witwin/radar/radar.py` — the flat `Radar` record, its waveforms, its pose, and the flat-configuration loader;
- `witwin/radar/targets.py` — the target vocabulary: positions, cross section or strength, trajectory, and identity;
- `witwin/radar/simulation.py` — scene-session execution, motion sampling, and frame results;
- `witwin/radar/channel.py` — the only production Channel importer;
- `witwin/radar/propagation.py` and `witwin/radar/paths.py` — propagation policy and round-trip composition;
- `witwin/radar/scattering.py`, `sensors.py`, and `frontend.py` — radar physics around the path;
- `witwin/radar/synthesis/` — native waveform synthesis;
- `witwin/radar/processing/` — range, Doppler, angle, detection, and tracking products;
- `witwin/radar/cuda/` — the native runtime and kernels.

## FMCW: spectrum first

FMCW synthesis directly generates the Dirichlet range spectrum in native CUDA by default. The default is `output_domain="spectrum"` in both the typed FMCW spec and the flat radar configuration. Set `output_domain="beat"` only when a caller explicitly needs a synthesized time-domain beat signal.

The result carries named axes and an output-domain field. Downstream processing uses that metadata, so it does not apply an extra range FFT to a spectrum or omit the FFT for beat samples.

## Main API

`Radar` is one flat immutable record in SI units. A sub-record appears only where a field has variants (the waveform, the antenna pattern, the motion sampling) or where several numbers are coupled (`Noise`, `Agc`, `Adc`); the array, the pose and the one-number receive stages are fields.

```python
from witwin.radar import Motion, Noise, PointTargets, Radar

radar = Radar.from_dict(CONFIG, noise=Noise(figure=10.0), seed=20260727, look_at=(0.0, 0.0, -3.0))
targets = PointTargets(positions=[(0.0, 0.0, -3.0)], rcs=1.0)
result = radar.simulate(scene, targets, times=(0.0, 0.1, 0.2), los=True, reflections=1, motion=Motion.auto())
```

`Radar.from_dict` reads the flat FMCW configuration format examples and config files use, in its vendor units, and is the only place those units are read. Every other field is a keyword override in SI, so a receive chain and a pose attach in the same call. `Radar.from_json` loads the same mapping from a file. Keys nothing consumes are refused by name rather than stored.

`Radar.simulate(...)` is the scene-driven entry point. It accepts a Core scene, a required `PointTargets` or `StructureTargets` record that says where the scatterers are and how strongly they scatter, the frame times, the propagation request as `los` and `reflections`, the in-frame resampling as `motion`, and the differentiation mode as `grad`. It returns `RadarSimulationResult`, whose cube is organized as `[frame, TX, RX, slow, fast]` and whose metadata states the waveform, the fast-axis domain and the last frame's typed diagnostics.

`Radar.stream(...)` runs the same session and yields each frame as its own one-frame `RadarSimulationResult`. Use it when the sequence is longer than the stacked cube can be held in device memory; the frames are bit-exact against `simulate` and peak allocation no longer scales with the frame count.

A radar is never edited in place: `Radar.replace(...)` and `Radar.to(device)` return a new one, and a radar retains nothing from a run, so a result is the only thing that can say which frame a diagnostic describes.

Signal processing is exported through `witwin.radar.processing`; typed products include processing cubes, range profiles, Range-Doppler maps, beam cubes, detections, and point clouds.

See `docs/pipeline_guide.md` for the full contract and `examples/single_point.py` for a maintained end-to-end example.

## Tests and governance

```bash
pytest tests/
pytest tests/ --gpu
python ci/run_ci_tier.py quick
```

The quick tier includes canonical Ruff formatting/lint, exact-clone detection, architectural, public-surface, documentation, release-claim, workflow-reference, and compatibility-removal gates. Required-Channel workflows install the Channel extra, record its build fingerprint, and permit zero skips caused by a missing Channel runtime.

No benchmark, GPU result, wheel load, or remote workflow is claimed as executed merely because its command exists. Current performance evidence and outstanding measurements are documented in `PERFORMANCE.md`.

## Examples

```bash
python -m examples.single_point
python -m examples.music_imaging
python -m examples.rgbd_range_doppler --input path/to/depths.npz
```

All maintained scene-driven examples require CUDA and Channel.

## Documentation

- Pipeline: `docs/pipeline_guide.md`
- Development standard: `docs/dev/standards/radar-adr-021-code-layout-comments-and-mathematical-ownership.md`
- Consolidation plan: `docs/dev/plans/radar-concept-axis-layout-and-module-consolidation-plan.md`
- Governance inventory: `docs/dev/audit/radar-governance-debt-and-drift-inventory.md`
- AD capability matrix: `docs/dev/radar-ad-capability-matrix.md`
- AD tape/budget ledger: `docs/dev/ad-tape-and-budget-ledger.md`

## License and citation

WiTwin Radar uses the WiTwin dual-license model. See the [WiTwin licensing page](https://witwin.ai/license). The simulator is derived from [RF-Genesis](https://github.com/Asixa/RF-Genesis); cite the RF-Genesis SenSys 2023 paper when that prior work is relevant.
Dynamic FMCW simulation refreshes the scene at each ADC observation by default, including TDM timing. `Motion.chirp()` selects the faster stop-and-hop approximation. See `docs/pipeline_guide.md` for trajectories, path completeness, and typed micro-Doppler timestamps.
