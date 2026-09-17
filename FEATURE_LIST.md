# Radar Feature List

This file describes the current surface after the breaking concept-axis consolidation. It does not list deleted compatibility APIs.

Scene-driven FMCW phase noise uses one continuous-time Wiener oscillator shared by all paths, evaluated as the delayed transmit/receive phase difference at absolute ADC timestamps. Idle gaps and inter-path delay correlation are preserved. This white-frequency-noise model does not represent a complete multi-region device mask. Timestamp and delay derivatives of Wiener samples are explicitly refused.

## Simulation and propagation

- `Radar.simulate(...)` is the scene-driven production entry for `witwin.core.Scene` and `DynamicScene` worlds.
- `Radar.stream(...)` runs the identical session and yields one-frame results, so a sequence longer than device memory can hold as a stacked cube is still producible. Frames are bit-exact against `simulate`; peak allocation stops tracking the frame count.
- Parameter JVP seeds do not change the simulated primal or masquerade as physical velocities.
- `RadarSimulationResult` returns a typed `[frame, TX, RX, slow, fast]` cube with waveform, axes, phasor convention, reference frequency, epoch, and last-frame diagnostic metadata.
- `witwin/radar/channel.py` is the single production importer of `witwin.channel`; the rest of Radar consumes Radar-owned adapter contracts.
- Direct and multipath one-way legs are composed into round-trip paths with explicit join mode, identity, delay, delay rate, transfer provenance, and row validity.
- Dynamic FMCW scenes default to geometry and transport refreshed at every ADC timestamp, including TDM slot offsets. Explicit `motion_sampling="chirp"` selects a stop-and-hop approximation.
- Dynamic paths are rediscovered at every observation by default. Longer discovery cadences set `path_set_complete=False`; row-index finite differences are never used for velocity.
- `motion_sampling="adaptive"` with `AdaptiveMotionSpec` selects carrier-aware temporal interpolation, batched propagation/synthesis, quarter-point error probes and topology refinement. Per-path phase/amplitude tolerances, maximum probe interval, discovery budget and batch size are configurable. Exhausted budgets raise. Unsampled brief events are not certified absent; `adaptive_diagnostics` and `path_set_complete` expose that limit. ADC remains the exhaustive reference default.
- `AdaptiveMotionSpec.interpolation_nodes` sets how many sampled instants carry one accepted interval and therefore the polynomial order of the interpolated delay; the default 2 is the linear rule and 5 is a quartic. The native interpolant takes K nodes with caller-supplied weights that must sum to one and publishes no weight derivative. Raise it where the phase test is what shortens an interval, which is micro-Doppler at a high carrier; it cannot lengthen an interval the probe-spacing bound already fixed.
- `max_interval_s` is the probe-spacing floor and is enforced unconditionally. Every adaptive tolerance is checked by sampling, so motion periodic at the probe grid's step is invisible to all of them; this bound is what sets that step and is therefore an accuracy control as much as a topology one. A topological completeness proof does not relax it. Within it the run starts from the coarsest partition it allows instead of bisecting down to it.
- `path_set_complete` and `motion_sampling_exhaustive` are published separately: no path birth can have been missed, and no observation was interpolated. A certified empty-world LOS family reports the first without the second, and `adaptive_diagnostics` carries `topology_proved_complete` per frame.
- Complete empty-world LOS families reuse frozen topology while retaining phase/validity probes. Static-world replay batches joins and antenna patterns; arbitrary ADC times share bounded native CSR synthesis with VJP/JVP. Scenes with reflectors retain discovery probes; moving geometry still follows Channel compilation/retirement rules.
- `ScatterSitePolicy.explicit(..., trajectory=...)` preserves authored material-point order across translation, rotation, or articulation. `SensorEndpointIds` maps moving Core phase centres into array order.
- Scatter sites are declared explicitly or by a supported policy. Radar does not silently derive a different physical target set from mesh geometry.

## Waveform synthesis

- FMCW, OFDM, and pulsed synthesis have separate owners under `witwin/radar/synthesis/` and share typed path/result assembly.
- FMCW outputs a normalized range spectrum by default: native Dirichlet evaluation for stationary rows, native quadratic-phase summation for linearly moving rows, and the processing-owned range transform for ADC-refreshed scenes.
- The linear-delay native model includes ADC start time and fast-time motion, with matching analytic VJP/JVP.
- `FmcwSpec.output_domain="spectrum"` is the default; `output_domain="beat"` explicitly selects synthesized time-domain beat samples.
- The FMCW spectrum and beat paths each expose native forward, analytical backward, and JVP operators through the one Radar native runtime.
- TDM slot timing is derived from the transmitter index of each sensor-pair segment.
- Synthesis validates weight provenance before launch so carrier phase, spreading, transmit power, and slow-time refresh are not silently counted twice.
- Output-domain metadata is preserved in `SynthesisResult`, preventing downstream processing from guessing whether a range FFT is required.

## Radar physics

- Scalar-RCS and aspect-dependent scatter responses, including reflected outbound paths with differentiable departure bearings.
- Round-trip antenna pattern and transmit-power weighting using actual first/last path segments.
- Receiver frontend contracts for LNA, noise, AGC, ADC, port mapping, and deterministic seeds.
- FMCW receiver stages run on beat samples before the output range FFT, so beat/spectrum selections share one physical ADC and gain/noise realization.
- Radar-owned SMPL authoring layered on Core geometry.
- Explicit AD/host-observation policy, first-order reverse mode, and forward-mode JVP coverage for native hot paths.

## Signal processing

The `witwin.radar.processing` facade exports typed products and algorithms for:

- signal cube normalization and processing axes;
- range profiles and Range-Doppler maps;
- matched filtering and `SlowTimeSignal`-based micro-Doppler with explicit timestamps and phasor, physical Doppler sign, and rejection of frame gaps;
- phase-comparison and FFT AoA;
- conventional, MVDR, and MUSIC beamforming/imaging;
- CA-CFAR and OS-CFAR detection;
- point-cloud generation and nearest-neighbour association.

The processing layer consumes named-axis metadata. For FMCW spectrum input, range-profile construction is an identity-domain conversion rather than another FFT; explicit beat input takes the FFT route.

## Public API and architecture governance

- Public exports are declared in `ci/public-api-manifest.json` and signature-pinned by `ci/public-api-snapshot.json`.
- Concept owners and permitted module topology are declared in `ci/architecture-manifest.json`.
- There is no maximum-file-line policy. Consolidation favors fewer files and a shallow layout when one concept remains cohesive.
- Compatibility aliases, fallback imports, deprecation shims, and legacy package mirrors are forbidden by static gates.
- Living documentation is checked against deleted paths and retired current-surface names.

## Packaging and CI policy

- CPython 3.10-3.14 on Linux x86_64 and Windows x86_64.
- Linux release artifacts target `manylinux_2_28_x86_64`.
- Release wheels contain exactly one `_radar_native` binary and its build identity sidecars.
- The loader enforces exact Torch/CUDA/ABI identity; refusal is a release failure.
- JIT fallback is not part of the packaged runtime contract.
- Required-Channel quality and GPU workflows install the Channel extra, consume its build fingerprint, and permit zero skips caused by an absent Channel runtime.
- GPU regression is manually dispatched on the named GPU runner exception and publishes no wheel.

Commands in CI are evidence only after the corresponding job actually runs. The checked-in workflow is policy, not proof of a successful remote execution.
