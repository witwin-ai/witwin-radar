# Radar Feature List

This file describes the current surface after the breaking concept-axis consolidation. It does not list deleted compatibility APIs.

## Simulation and propagation

- `Radar.simulate(...)` is the scene-driven production entry for `witwin.core.Scene` and `DynamicScene` worlds.
- `RadarSimulationResult` returns a typed `[frame, TX, RX, slow, fast]` cube with waveform, axes, phasor convention, reference frequency, epoch, and last-frame diagnostic metadata.
- `witwin/radar/channel.py` is the single production importer of `witwin.channel`; the rest of Radar consumes Radar-owned adapter contracts.
- Direct and multipath one-way legs are composed into round-trip paths with explicit join mode, identity, delay, delay rate, transfer provenance, and row validity.
- Fixed-topology reuse and rediscovery are separate policies; dynamic scenes can be sampled per requested frame time.
- Scatter sites are declared explicitly or by a supported policy. Radar does not silently derive a different physical target set from mesh geometry.

## Waveform synthesis

- FMCW, OFDM, and pulsed synthesis have separate owners under `witwin/radar/synthesis/` and share typed path/result assembly.
- FMCW directly generates a Dirichlet range spectrum in native CUDA by default.
- `FmcwSpec.output_domain="spectrum"` is the default; `output_domain="beat"` explicitly selects synthesized time-domain beat samples.
- The FMCW spectrum and beat paths each expose native forward, analytical backward, and JVP operators through the one Radar native runtime.
- TDM slot timing is derived from the transmitter index of each sensor-pair segment.
- Synthesis validates weight provenance before launch so carrier phase, spreading, transmit power, and slow-time refresh are not silently counted twice.
- Output-domain metadata is preserved in `SynthesisResult`, preventing downstream processing from guessing whether a range FFT is required.

## Radar physics

- Scalar-RCS and aspect-dependent scatter responses.
- Round-trip antenna pattern and transmit-power weighting.
- Receiver frontend contracts for LNA, noise, AGC, ADC, port mapping, and deterministic seeds.
- Radar-owned SMPL authoring layered on Core geometry.
- Explicit AD/host-observation policy, first-order reverse mode, and forward-mode JVP coverage for native hot paths.

## Signal processing

The `witwin.radar.processing` facade exports typed products and algorithms for:

- signal cube normalization and processing axes;
- range profiles and Range-Doppler maps;
- matched filtering and micro-Doppler;
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