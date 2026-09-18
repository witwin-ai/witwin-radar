# Radar Pipeline Guide

This is the current scene-to-product route after the breaking concept-axis consolidation. There is one production orchestration path and no legacy fallback path.

`Radar.simulate(...)` runs a session to completion and stacks every frame, so peak device
allocation scales with the sequence: measured at about three times the published cube for a
128-frame run, because the per-frame list and `torch.stack` are both live. `Radar.stream(...)` runs the identical session
and yields each frame as a one-frame `RadarSimulationResult`, which keeps peak allocation
independent of the frame count. Both consume one frame generator, so the physics, the epoch
loop and the synthesis route have a single owner. A streamed result aliases that frame's device
tensors through its four `last_*` members exactly as the stacked result does: holding every
yielded frame costs more than stacking, not less. Those four members belong to the result and
only to it; a radar retains nothing from a run, so the record that answers "which frame is this"
is the one the call returned. Generator arguments are validated when iteration starts.

Dynamic FMCW can select `motion=Motion.adaptive(...)`, with `Motion` imported from
`witwin.radar`. Controls are `phase_error`, `relative_amplitude_error`,
`max_interval`, `nodes`, `max_evaluations`, and `batch_observations`.
`nodes` is the number of sampled instants one accepted interval interpolates
through, so the default 2 is the linear rule and 5 is a quartic; each interval probes a grid of
`2 * (nodes - 1) + 1` instants and tests the error at the ones between the nodes, which at the
default two nodes is a single midpoint test. Lowering `max_interval` refines the grid. A higher order
buys a longer interval, which it cannot do where the probe-spacing bound already fixes the
length, so raise it only where the phase test is what shortens an interval.
`max_interval` is the maximum probe spacing and is enforced unconditionally, including for a
family certified complete for all time. Every tolerance here is checked by sampling and therefore
cannot see motion periodic at the probe grid's step; this bound is what sets that step. The
initial partition is the coarsest one it allows. Native interpolation moves each
endpoint coefficient to the query's carrier phase before blending. The grid's interior probes
test delay, complex phase, amplitude, full leg identity, and validity; mismatches subdivide.
The native interpolant supports VJP/JVP for a fixed accepted partition. Discovery and
refinement decisions are discrete. Budget exhaustion raises instead of returning an unchecked cube.

`adaptive_diagnostics` records evaluation counts, tested errors, topology refinements and native
synthesis batch counts. `result.discovery_count` separately counts topology discoveries.
`path_set_complete` and `motion_sampling_exhaustive` are two separate statements.
The first is true when no path birth can have been missed, either because every ADC
instant was evaluated or because the candidate family was certified complete for all
time. The second is true only when no observation's transport was interpolated. An
adaptive run in a certifiable world therefore reports completeness without
exhaustiveness. Probe spacing is not a proof that a shorter occlusion or oscillation was
absent in a world that cannot be certified. Use `Motion.adc()`, which is what the default
`Motion.auto()` resolves to for a moving world, for exhaustive comparison; `Motion.chirp()` is an
explicit stop-and-hop approximation.

An empty authored world with LOS-only propagation and all endpoint pairs already present permits
topology reuse. Source mutations, retired handles and version changes still invalidate it. Worlds
with geometry keep discovery at the error-control probes, including path births and disappearances.
Static geometry permits slot-major propagation, composition and antenna weighting in batches.
ADC synthesis gathers cached path rows; it no longer creates a delay/clock tensor for every sample.
Each synthesis batch is bounded by `batch_observations * num_samples` observations and by a row
budget that scales with the node-table width, so raising the order does not raise peak allocation
(a single larger observation is indivisible). Probe batches still use `batch_observations` directly.
The phase/amplitude tolerances and maximum probe spacing are unchanged by these scheduling choices.

Receiver-enabled FMCW first synthesizes beat samples, applies receiver hardware, then computes
the requested normalized range spectrum. Common-oscillator noise is applied per path before
coherent summation as a delayed Wiener phase difference at absolute ADC time, including idle gaps.
Time/delay derivatives of this nowhere-differentiable noise are refused; fixed-query signal
derivatives remain available. A multi-region device phase-noise spectrum is not implemented.

## 1. Ownership boundaries

`witwin.core` owns scenes, geometry, materials, structure identity, and motion. `witwin.channel` owns one-way electromagnetic propagation. Radar consumes those results and owns:

1. simulation-session policy;
2. round-trip path composition;
3. scattering and sensor/frontend effects;
4. FMCW, OFDM, and pulsed synthesis;
5. radar signal processing.

Only `witwin/radar/channel.py` imports Channel in production. The boundary publishes Radar-owned records to `propagation.py` and `paths.py`, so Channel implementation types do not spread through the package.

## 2. Configure a radar

`Radar` is one flat immutable record in SI units: the carrier, the waveform, the two antenna layouts and the transmit power are required, and the pattern, the four receive stages, the port impedance, the seed, the pose and the device have defaults. Internally it derives four conceptual blocks — waveform, sensors, propagation, frontend. There is no processing block: the four keys that used to build one (`frame_per_second`, `num_doppler_bins`, `num_range_bins`, `num_angle_bins`) were read by nothing, and the loader now refuses them by name rather than storing a number that decides nothing.

`Radar.from_dict(...)` accepts the flat configuration used by examples and config files, in the vendor units it is written in — slope in MHz/µs, sample rate in kSPS, the three timings in microseconds, power in dBm, element positions in half wavelengths. This is the only place those units are read; every other field is a keyword override in SI names, so a receive chain and a pose attach in the same call. Any key the record does not consume is refused.

A radar is never edited in place. `Radar.replace(position=..., look_at=...)` returns a new radar with those fields changed, which is what keeps a radar captured in a closure or held by a result from changing underneath it, and what lets a pose built from a tensor with a tape keep that tape.

Two defaults are physical choices rather than conveniences:

- The element pattern is `Pattern.isotropic()`. The previous default was an unchosen half-wave dipole cut, which attenuated every off-boresight return by a number the caller never asked for. `Pattern.dipole()` restores it, and `Pattern.separable(...)` and `Pattern.table(...)` take a measured cut or map.
- `Radar.polarization` is `"up"`: the pose's own up axis, and therefore transverse to the boresight whichever way the radar points. Channel projects the transmitted field onto a world vector, and a vector parallel to the boresight radiates nothing; the previous fixed default was parallel to the default boresight, so every transport came back exactly zero with nothing raised. A world vector is still accepted, and one parallel to the boresight is now refused at construction instead of publishing a cube of zeros. `"right"` is the other derived alias.

The FMCW output domain is `Fmcw.output` on the typed record and `output_domain` in the flat mapping:

- omitted or `"spectrum"` — normalized range spectrum;
- `"beat"` — explicit synthesized time-domain beat samples.

The default is spectrum. The beat route is an opt-in output domain, not a fallback.

## 3. Build the world and scatter model

Pass a `witwin.core.Scene` or `DynamicScene` to `Radar.simulate(...)`. World geometry should remain in authored world coordinates; a caller must not silently recenter geometry that is already positioned.

Where the scatterers are and how strongly they scatter are one statement, so they are one record, passed as the required positional `targets` argument:

- `PointTargets(positions=..., rcs=...)` names world positions outright. `trajectory` moves the same ordered material points, `ids` fixes stable world identity, and `aspect=Aspect(...)` makes the strength depend on the geometry of each round trip instead of being isotropic.
- `StructureTargets(rcs=..., structure_ids=...)` puts one scatterer at each world anchor Core publishes for a moving structure.

A target declares its strength as `rcs` in m², or as the dimensionless `amplitude` for a caller who holds the strength itself — exactly one of the two, because they are the same quantity on either side of `amplitude = sqrt(4 pi rcs) / wavelength` and accepting both would leave which one wins undefined. Neither is defaulted: target reflectivity is a physical choice. The device and the carrier are not fields of a target record; the radar supplies them when the session starts.

There is no default target set, and Radar derives no scatterer from geometry — no surface sample, no centroid, no bounding-box centre. Every one of those is a geometry algorithm and belongs to Channel's native geometry owner; `docs/dev/standards/radar-adr-020-scene-binding-and-site-policy.md` records the deferral and names what closing it would need.

## 4. Execute the simulation session

`witwin/radar/simulation.py` owns the frame loop:

1. sample the Core world at waveform observation times within each requested frame;
2. compile or reuse the Channel scene epoch;
3. discover or reevaluate one-way topology according to policy;
4. compose direct or two-way round trips;
5. evaluate scattering and optional sensor-pattern weights;
6. synthesize the configured waveform;
7. apply the declared frontend;
8. assemble the typed frame result.

A radar holds no run state, so there is nothing for a call to clear and nothing a failed call can leave behind: a result exists only if the call that built it returned, and the four typed diagnostics are members of that result.

The returned `RadarSimulationResult.cube` has axes `[frame, TX, RX, slow, fast]`. Its metadata also includes frame times, waveform kind, named axes, phasor/time convention, reference frequency, epoch information, and last-frame typed diagnostics.

`motion` selects how often the world is resampled inside a frame and defaults to `Motion.auto()`, which resolves to per-ADC sampling when anything in the session moves — a structure trajectory or deformation, an endpoint trajectory, a target trajectory — or when the receiver carries oscillator phase noise, which needs ADC-time observations to place its delayed phase difference. Otherwise it resolves to one observation per frame. `Motion.static()` asks for that single observation explicitly and is refused by name for a world that moves within a frame, because it would publish no Doppler at all.

Dynamic FMCW therefore evaluates the world at
`frame_time + (chirp*num_tx+tx)*chirp_period + adc_start + sample*sample_period`.
The result records `sample_times_s` and `motion_sampling`; diagnostics describe
the last observation. This handles moving reflectors as well as moving sites,
without subtracting rows from different discovered path sets. It uses the
quasistatic Channel model at each observation, not relativistic retarded-time
moving-boundary propagation.

For a faster declared stop-and-hop approximation, select `motion=Motion.chirp()`.
Its geometry/weight is frozen within each chirp. OFDM refreshes per symbol and
pulsed simulation per pulse; those remain block-frozen models.
A custom site trajectory is a plain callable, `trajectory(time)`, returning the same
ordered material points at that instant as an `(S, 3)` tensor in metres. It publishes
positions only: a site velocity is never differenced into physics, because the delay
rate comes from the propagation solve at each observation instant.
Rotation/articulation must change those positions: an angular-velocity label alone
cannot move an authored point.
Parameter JVPs differentiate the trajectory; they never become physical velocity.
Low-level synthesis `delay_rate` explicitly means physical `d(tau)/dt`.

Full discovery runs at each dynamic observation by default. An explicitly longer
`Motion.<kind>(rediscover_every_frames=n)` trades completeness for speed and publishes
`path_set_complete=False` unless structure motion already forces discovery.
Visibility changes are real discontinuities: no smoothing or velocity clipping
is applied across them. Complete means complete within the requested supported
Channel components and discovery policy, not all conceivable propagation physics.

## 5. FMCW synthesis domains

`witwin/radar/synthesis/fmcw.py` is the sole FMCW synthesis owner. It consumes compact path rows and conjugates the Channel transfer coefficient exactly once into the beat convention.

### Default spectrum route

The native spectrum kernel evaluates stationary rows as Dirichlet contributions. A nonzero physical delay rate makes fast-time phase quadratic, so moving rows use an exact finite DFT sum in native CUDA. The common phase owner is `cuda/fmcw_phase.cuh`; backward and JVP differentiate that same continuous-delay equation. ADC-refreshed scenes evaluate propagation and native beat synthesis at each observation, then call the processing-owned normalized range transform.

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

`microdoppler_spectrogram` consumes `SlowTimeSignal(samples, times_s, phasor)`.
Use one sensor pair and range gate per sequence. Timestamps must be uniform;
segment frame gaps before STFT. Frequencies use `f_D=-f_ref*d(tau)/dt`
(receding negative), independent of the source beat/Channel convention.

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
