# Radar Pipeline Guide

This is the current scene-to-product route after the breaking concept-axis consolidation. There is one production orchestration path and no legacy fallback path.

## 1. Ownership boundaries

`witwin.core` owns scenes, geometry, materials, structure identity, and motion. `witwin.channel` owns one-way electromagnetic propagation. Radar consumes those results and owns:

1. simulation-session policy;
2. round-trip path composition;
3. scattering and sensor/frontend effects;
4. FMCW, OFDM, and pulsed synthesis;
5. radar signal processing.

Only `witwin/radar/channel.py` imports Channel in production. The boundary publishes Radar-owned records to `propagation.py` and `paths.py`, so Channel implementation types do not spread through the package.

## 2. Configure a radar

`RadarConfig.from_dict(...)` accepts the flat configuration used by examples and config files. `Radar` converts it into conceptual blocks for waveform, sensors, propagation, processing, and frontend. Pose is set at construction or with `Radar.set_pose(...)`.

FMCW configuration includes `output_domain`:

- omitted or `"spectrum"` — direct native Dirichlet range spectrum;
- `"beat"` — explicit synthesized time-domain beat samples.

The default is spectrum. The beat route is an opt-in output domain, not a fallback.

## 3. Build the world and scatter model

Pass a `witwin.core.Scene` or `DynamicScene` to `Radar.simulate(...)`. World geometry should remain in authored world coordinates; a caller must not silently recenter geometry that is already positioned.

The scatter response is required because target reflectivity is a physical choice. Scatter sites are also explicit: use `ScatterSitePolicy.explicit(...)` for authored locations or another supported policy whose meaning is declared. Radar does not infer an undocumented mesh-sampling policy.

## 4. Execute the simulation session

`witwin/radar/simulation.py` owns the frame loop:

1. sample the Core world at the requested frame time;
2. compile or reuse the Channel scene epoch;
3. discover or reevaluate one-way topology according to policy;
4. compose direct or two-way round trips;
5. evaluate scattering and optional sensor-pattern weights;
6. synthesize the configured waveform;
7. apply the declared frontend;
8. assemble the typed frame result.

`Radar.simulate(...)` clears its last-result diagnostic before work begins. A failed call therefore cannot leave a previous result pretending to describe the failed simulation.

The returned `RadarSimulationResult.cube` has axes `[frame, TX, RX, slow, fast]`. Its metadata also includes frame times, waveform kind, named axes, phasor/time convention, reference frequency, epoch information, and last-frame typed diagnostics.

## 5. FMCW synthesis domains

`witwin/radar/synthesis/fmcw.py` is the sole FMCW synthesis owner. It consumes compact path rows and conjugates the Channel transfer coefficient exactly once into the beat convention.

### Default spectrum route

The native spectrum kernel evaluates each path as a Dirichlet contribution directly at range bins. It does not synthesize ADC samples followed by a Python-side FFT. Forward, reverse-mode, and JVP all dispatch to matching spectrum operators.

The fast axis of the result is `range`, and its length is the configured FMCW sample/bin count.

### Explicit beat route

With `output_domain="beat"`, the native beat kernel evaluates time-domain ADC samples. The fast axis is `sample`. This route exists for callers that need beat samples themselves or want to exercise a particular time-domain processing chain.

Both domains preserve the same TDM slow-time timing and compact sensor-pair ordering.

## 6. Processing without domain guessing

Build processing metadata from the synthesis result and radar array configuration. Functions under `witwin.radar.processing` consume named axes:

- range-profile construction performs no second range FFT for spectrum input;
- beat input is transformed along its sample axis;
- Range-Doppler processing transforms slow time and preserves the range axis;
- angle, beamforming, CFAR, point-cloud, and tracking stages consume typed products rather than unlabelled tensors.

A tensor shape alone is not sufficient to choose a processing route; spectrum and beat outputs can have the same rank and fast-axis length.

## 7. Differentiation contract

Native hot paths provide explicit forward, analytical backward, and JVP companions where the AD capability matrix marks support. Unsupported host observations, higher-order derivatives, or semantically dead tangents are refused at a named boundary instead of returning a plausible detached answer.

See `docs/dev/radar-ad-capability-matrix.md` for the row-level capability contract and `docs/dev/ad-tape-and-budget-ledger.md` for saved-tensor and launch accounting.

## 8. Reproducibility and Channel coverage

Required-Channel CI installs the Channel dependency, imports it before tests, records `build_info()["build_fingerprint"]`, and has a missing-Channel skip budget of zero. A release or regression record should retain that fingerprint alongside Radar's native build identity.

## 9. Maintained examples

From the repository root:

```bash
python -m examples.single_point
python -m examples.music_imaging
python -m examples.rgbd_range_doppler --input path/to/depths.npz
```

`examples/single_point.py` is the primary end-to-end reference. The other examples cover MUSIC imaging and depth-sequence Range-Doppler processing. They require CUDA and a working Channel runtime.

## 10. Validation commands

```bash
pytest tests/
pytest tests/ --gpu
python ci/run_ci_tier.py quick
python ci/check_required_channel_coverage.py
python ci/check_workflow_references.py
```

The commands above define how to obtain evidence; they are not themselves evidence that a GPU job, wheel load, or remote workflow has run.