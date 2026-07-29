# WiTwin Radar

WiTwin Radar is a GPU-accelerated, differentiable radar simulator. A simulation consumes a `witwin.core` world and one-way propagation from `witwin.channel`, composes round trips, applies scattering and sensor/frontend effects, synthesizes radar waveforms, and produces typed signal-processing products.

The repository uses a breaking, concept-axis architecture. Compatibility modules and deprecated aliases are intentionally not retained.

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

- `witwin/radar/radar.py` — configuration, pose, and the `Radar` facade;
- `witwin/radar/simulation.py` — scene-session execution and frame results;
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

`Radar.simulate(...)` is the scene-driven entry point. It accepts a Core scene, frame times, an explicit scatter response, and an explicit or policy-defined set of scatter sites. It returns `RadarSimulationResult`, whose cube is organized as `[frame, TX, RX, slow, fast]` and whose metadata states the waveform and fast-axis domain.

`Radar.synthesize(...)` is the lower-level path-to-waveform entry. It dispatches from the stored waveform kind and requires an explicit slow-time mode.

Signal processing is exported through `witwin.radar.processing`; typed products include processing cubes, range profiles, Range-Doppler maps, beam cubes, detections, and point clouds.

See `docs/pipeline_guide.md` for the full contract and `examples/single_point.py` for a maintained end-to-end example.

## Tests and governance

```bash
pytest tests/
pytest tests/ --gpu
python ci/run_ci_tier.py quick
```

The quick tier includes architectural, public-surface, documentation, release-claim, workflow-reference, and compatibility-removal gates. Required-Channel workflows install the Channel extra, record its build fingerprint, and permit zero skips caused by a missing Channel runtime.

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
- Consolidation plan: `docs/dev/plans/radar-concept-axis-layout-and-module-consolidation-plan.md`
- Governance inventory: `docs/dev/audit/radar-governance-debt-and-drift-inventory.md`
- AD capability matrix: `docs/dev/radar-ad-capability-matrix.md`
- AD tape/budget ledger: `docs/dev/ad-tape-and-budget-ledger.md`

## License and citation

WiTwin Radar uses the WiTwin dual-license model. See the [WiTwin licensing page](https://witwin.ai/license). The simulator is derived from [RF-Genesis](https://github.com/Asixa/RF-Genesis); cite the RF-Genesis SenSys 2023 paper when that prior work is relevant.