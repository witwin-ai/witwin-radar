# Radar Feature List

This file describes the current surface after the breaking concept-axis consolidation. It does not list deleted compatibility APIs.

Scene-driven FMCW phase noise uses one continuous-time Wiener oscillator shared by all paths, evaluated as the delayed transmit/receive phase difference at absolute ADC timestamps. Idle gaps and inter-path delay correlation are preserved. This white-frequency-noise model does not represent a complete multi-region device mask. Timestamp and delay derivatives of Wiener samples are explicitly refused.

## Configuring a radar

- `Radar` is one flat immutable record in SI units. The carrier, the waveform, the two antenna layouts and the transmit power are required; the pattern, the four receive stages, the port impedance, the seed, the pose and the device have defaults. A sub-record appears only where a field has variants (`Fmcw`/`Ofdm`/`Pulsed`, `Pattern`, `Motion`) or where several numbers are coupled (`Noise`, `Agc`, `Adc`); the array, the pose and the one-number stages are fields. No public field name carries a unit suffix, and every field states its unit on its docstring row.
- `antenna_unit` reads `tx` and `rx` in metres or in half wavelengths, converted once against the carrier, so the same layout means the same beam at any carrier.
- `Radar.from_dict` and `Radar.from_json` load the flat FMCW configuration format in its vendor units — MHz/µs, kSPS, microseconds, dBm, half wavelengths — and are the only place those units are read. Other fields are keyword overrides in SI names. Any key outside the format, including the processing-grid keys `frame_per_second`, `num_doppler_bins`, `num_range_bins` and `num_angle_bins`, is refused by name.
- A radar is immutable and holds no run state. `Radar.replace(...)` and `Radar.to(device)` return a new radar, in-place assignment raises, and a pose built from a tensor with a tape keeps that tape. The four typed per-frame diagnostics live on the result that published them, so a call that raised part way through leaves nothing behind to misread.

## Simulation and propagation

- `Radar.simulate(...)` is the scene-driven production entry for `witwin.core.Scene` and `DynamicScene` worlds. It takes the target set positionally, the propagation request as `los` and `reflections`, the in-frame resampling as `motion`, the differentiation mode as `grad`, and a moving mount's Core phase-centre IDs as `endpoints`. A request with neither the line of sight nor any reflection is refused rather than solved as nothing.
- `Radar.stream(...)` runs the identical session and yields one-frame results, so a sequence longer than device memory can hold as a stacked cube is still producible. Frames are bit-exact against `simulate`; peak allocation stops tracking the frame count.
- The pipeline's two halves are separately callable. `Radar.trace(...)` takes `simulate`'s arguments, runs the world half — world sampling, Channel epochs, topology, round-trip composition, scattering and antenna weighting — and returns `Paths`; `Radar.echo(paths)` runs the instrument half — waveform synthesis, the receive chain, the output domain and the processing axes — and returns `Result`. All four verbs share one session loop and one synthesis route, so `echo(trace(...))` is bit-identical to `simulate(...)` across every motion kind, both FMCW output domains, with and without a receive chain, and with oscillator phase noise.
- One `Paths` can be echoed by several receive chains, seeds or FMCW output domains without re-solving the world. A radar the rows do not describe is refused by name: a different carrier, a different sensor-pair partition, a different waveform behind a sampled schedule, or a phase-noise receiver against paths traced without ADC instants. `Paths.frame(i)` slices one frame and echoes to that frame's cube; `Paths.rows(...)` publishes an evaluated observation as a typed batch.
- The split costs retention, and `Paths` states the cost: it holds every evaluated observation's rows, roughly twenty bytes per live row summed over every evaluated observation of every frame, so an ADC-refreshed sequence is expensive. `simulate` and `stream` never pay it, because a frame's observations stay a generator they drain one at a time. An adaptive trace retains its accepted partition rather than its schedule, so its retained row count is the probes and not the observations. `path_set_complete` and `motion_sampling_exhaustive` belong to the trace and are published by both the `Paths` and the `Result`.
- `PointTargets` names world positions outright and `StructureTargets` places one scatterer at each Core-published world anchor of a moving structure. Positions and strength are one record, so a caller no longer repeats the radar's own carrier and device back at it. `Aspect` makes the strength depend on the geometry of each round trip; `trajectory` and `ids` carry motion and stable identity.
- A target declares its strength as `rcs` in m² or as the dimensionless `amplitude`, exactly one of the two: they are the same quantity on either side of `amplitude = sqrt(4 pi rcs) / wavelength`, so both leaves which one wins undefined and neither says nothing at all. A live tensor in either field keeps its tape, which is what an optimiser treating the strength itself as the leaf needs in order to avoid a square root in its gradient.
- `Motion` replaces the four sampling knobs that had to agree with each other, so the combinations that used to raise are now unwritable. `Motion.static()`, `chirp()`, `adc()` and `adaptive()` name the kind; `rediscover_every_frames` and `world` are advanced fields on all of them.
- `Motion.auto()` is the default. It resolves to per-ADC sampling when anything in the session moves — a structure trajectory or deformation, an endpoint trajectory, a target trajectory — or when the receiver carries oscillator phase noise, which needs ADC-time observations; otherwise it resolves to one observation per frame. `Motion.static()` is refused by name for a world that moves within a frame, because it would publish no Doppler.
- Parameter JVP seeds do not change the simulated primal or masquerade as physical velocities.
- `Result` returns a typed `[frame, TX, RX, slow, fast]` cube with waveform, phasor convention, reference frequency, epoch, and last-frame diagnostic metadata. `Result.axis_names` names the cube's axes and `Result.axes` is the `ProcessingAxes` record every processing stage reads, built once from the waveform spec and the array that produced the cube, so no caller assembles it from a re-viewed synthesis result.
- `witwin/radar/channel.py` is the single production importer of `witwin.channel`; the rest of Radar consumes Radar-owned adapter contracts.
- Direct and multipath one-way legs are composed into round-trip paths with explicit join mode, identity, delay, delay rate, transfer provenance, and row validity.
- Dynamic FMCW scenes default to geometry and transport refreshed at every ADC timestamp, including TDM slot offsets. Explicit `Motion.chirp()` selects a stop-and-hop approximation.
- Dynamic paths are rediscovered at every observation by default. Longer discovery cadences set `path_set_complete=False`; row-index finite differences are never used for velocity.
- `Motion.adaptive(...)` selects carrier-aware temporal interpolation, batched propagation/synthesis, midpoint error probes and topology refinement. An accepted interval probes `2 * (nodes - 1) + 1` instants: even positions become interpolation nodes, odd positions are where delay, phase, amplitude, leg identity and row validity are tested, which at the default two nodes is a single midpoint test. Per-path phase/amplitude tolerances, maximum probe interval, discovery budget and batch size are configurable on the same record, so a tolerance can no longer be set without the sampling kind that reads it. Exhausted budgets raise. Unsampled brief events are not certified absent; `adaptive_diagnostics` and `path_set_complete` expose that limit. ADC remains the exhaustive reference default.
- `Motion.adaptive(nodes=...)` sets how many sampled instants carry one accepted interval and therefore the polynomial order of the interpolated delay; the default 2 is the linear rule and 5 is a quartic. The native interpolant takes K nodes with caller-supplied weights that must sum to one and publishes no weight derivative. Raise it where the phase test is what shortens an interval, which is micro-Doppler at a high carrier; it cannot lengthen an interval the probe-spacing bound already fixed.
- `Motion.adaptive(max_interval=...)` is the maximum probe spacing in seconds and is enforced unconditionally, so the grid is never coarser than `max_interval / (2 * (nodes - 1))` and lowering it refines the grid. Every adaptive tolerance is checked by sampling, so motion periodic at the probe grid's step is invisible to all of them; this bound is what sets that step and is therefore an accuracy control as much as a topology one. A topological completeness proof does not relax it. Within it the run starts from the coarsest partition it allows instead of bisecting down to it.
- `path_set_complete` and `motion_sampling_exhaustive` are published separately: no path birth can have been missed, and no observation was interpolated. A certified empty-world LOS family reports the first without the second, and `adaptive_diagnostics` carries `topology_proved_complete` per frame.
- Complete empty-world LOS families reuse frozen topology while retaining phase/validity probes. Static-world replay batches joins and antenna patterns; arbitrary ADC times share bounded native CSR synthesis with VJP/JVP. Scenes with reflectors retain discovery probes; moving geometry still follows Channel compilation/retirement rules.
- `PointTargets(trajectory=...)` preserves authored material-point order across translation, rotation, or articulation; it is a plain callable returning positions, because the delay rate comes from the propagation solve rather than from a site velocity. `SensorEndpointIds`, passed as `endpoints`, maps moving Core phase centres into array order.
- Scatter sites are declared, by position or by structure anchor, and `targets` has no default. Radar derives no scatterer from mesh geometry: a sampling rule is a geometry algorithm and stays with Channel's native geometry owner.

## Waveform synthesis

- FMCW, OFDM, and pulsed synthesis have separate owners under `witwin/radar/synthesis/` and share typed path/result assembly.
- FMCW outputs a normalized range spectrum by default: native Dirichlet evaluation, in closed form, for a delay that holds the whole chirp. A walking delay is synthesized in the beat domain and reaches its spectrum through the processing-owned range transform, as do ADC-refreshed scenes; the spectrum family takes no delay rate and refuses one by name.
- The linear-delay native model includes ADC start time and fast-time motion, with matching analytic VJP/JVP.
- `FmcwSpec.output_domain="spectrum"` is the default; `output_domain="beat"` explicitly selects synthesized time-domain beat samples.
- The FMCW spectrum and beat paths each expose native forward, analytical backward, and JVP operators through the one Radar native runtime.
- TDM slot timing is derived from the transmitter index of each sensor-pair segment.
- Synthesis validates weight provenance before launch so carrier phase, spreading, transmit power, and slow-time refresh are not silently counted twice.
- Output-domain metadata is preserved in `SynthesisResult`, preventing downstream processing from guessing whether a range FFT is required.

## Radar physics

- Scalar-RCS and aspect-dependent scatter responses, including reflected outbound paths with differentiable departure bearings.
- Round-trip antenna pattern and transmit-power weighting using actual first/last path segments.
- The element pattern is isotropic by default. The earlier default was an unchosen half-wave dipole cut, which attenuated every off-boresight return by a number the caller never asked for; `Pattern.dipole()` restores it as a choice, and `Pattern.separable(...)` and `Pattern.table(...)` take a measured cut or map. Values are normalised linear power gain and the gain outside the tabulated support is exactly zero rather than the nearest tabulated value.
- `Radar.polarization` defaults to `"up"`: the pose's own up axis, transverse to the boresight whichever way the radar points. `"right"` is the other derived alias and a world vector is still accepted. A polarization parallel to the boresight radiates nothing and is refused at construction, where the earlier fixed default was parallel to the default boresight and published a cube of exact zeros with nothing raised.
- Receiver frontend contracts for LNA, noise, AGC, ADC, port mapping, and deterministic seeds, held as fields of the radar: `noise`, `lna_gain`, `agc`, `adc`, `impedance` and `seed`. A stage of one number is a keyword and a stage whose numbers are coupled is a record; the stage order and the shared Philox seed base are facts of the chain rather than something a caller assembles.
- `Noise.bandwidth` and `Noise.phase_sample_rate` default to the waveform's own sampling bandwidth and sample rate, resolved once when the radar is built. An explicit value overrides and an explicit `0.0` is refused, which is what a bandwidth of zero used to do silently: add no noise at all.
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

`Result.frame(i)` returns the `Frame` that pairs one frame's cube with that metadata — a pairing a caller used to assemble from a re-viewed synthesis result, the radar's waveform spec and the array, and could therefore assemble against a different array than the cube came from. `Frame.processing_cube()`, `range_profile()`, `range_doppler()`, `array()` and `points()` are facades over `witwin.radar.processing` and add no arithmetic of their own. `range_profile` defaults to a rectangular window because the FMCW spectrum output has already run the range transform and a taper there is refused; `range_doppler` tapers the slow axis with Hann by default because that axis is never pre-transformed. `points()` is named for the product rather than the stage, because the detector, angle-estimator and beamformer names are fenced out of every module outside the processing package.

## Public API and architecture governance

- Public exports are declared in `ci/public-api-manifest.json` and signature-pinned by `ci/public-api-snapshot.json`.
- The package root exports sixteen names, the whole happy path: `Adc`, `Agc`, `Aspect`, `Fmcw`, `Frame`, `Motion`, `Noise`, `Ofdm`, `Paths`, `Pattern`, `PointTargets`, `Pulsed`, `Radar`, `Result`, `StructureTargets` and `processing`. Advanced records stay importable from their owner modules and are not re-exported here.
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
