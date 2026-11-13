# Radar Repository Guide

## Purpose

WiTwin Radar is a differentiable radar simulator. `witwin.core` owns world and geometry state, `witwin.channel` owns one-way propagation, and Radar owns round-trip path composition, scattering, sensor/frontend effects, waveform synthesis, and DSP products.

The breaking concept-axis consolidation is complete. Do not add compatibility modules, aliases, fallback imports, deprecated entry points, or warning-only shims. Stable APIs may change. There is no maximum-file-line rule: prefer one clear conceptual owner over artificial file splitting.

## Concept-axis layout

- `witwin/radar/radar.py` — the flat immutable `Radar` record, the `Fmcw`/`Ofdm`/`Pulsed` waveforms, the pose transforms, and the flat-configuration loader.
- `witwin/radar/targets.py` — what the radar is looking at: `PointTargets`, `StructureTargets`, `MeshTargets`, `Aspect`, `RcsTable`, `Fluctuation`, and the split of one target record into a site policy and a scatter response. The scattering coefficient stays in `scattering.py`, the site binding in `binding.py` and the facet extraction in `facets.py`; nothing here computes any of them.
- `witwin/radar/facets.py` — a mesh target's facets: which faces scatter, their materials at the carrier, how facets group into sites, and each frame's facet corners and site positions through the native `facet_geometry` operator (R-ADR-042).
- `witwin/radar/simulation.py` — scene-session orchestration: the typed `Session` record, the `WorldHalf` that owns the epoch loop and every Channel call, the instrument half (`echo_frame` and the one synthesis dispatch), and the `Paths` the world half retains.
- `witwin/radar/motion.py` — the `Motion` sampling record and its adaptive tolerances.
- `witwin/radar/binding.py` — binding one radar to one snapshot: stable IDs, the scatter-site policy, `SensorEndpointIds`, and the endpoint specs the legs consume.
- `witwin/radar/adaptive.py` — adaptive motion control: the host-side probe refinement and the interpolated echo of its accepted partition.
- `witwin/radar/results.py` — the `Result` and `Frame` records the instrument half publishes.
- `witwin/radar/channel.py` — the only production importer of `witwin.channel`; compile, propagation, topology, and kinematics adapters.
- `witwin/radar/propagation.py` — Radar-owned propagation policies and epoch logic.
- `witwin/radar/paths.py` — two-way round-trip path contracts and composition.
- `witwin/radar/scattering.py` — scattering responses: scalar, aspect lobe, physical-optics facets, tabulated, and Swerling fluctuation.
- `witwin/radar/sensors.py` — array geometry, transmit power, antenna patterns, and round-trip weighting.
- `witwin/radar/frontend.py` — receiver chain, noise, ADC, AGC, and deterministic seed contracts.
- `witwin/radar/smpl.py` — Radar-specific SMPL authoring on Core geometry.
- `witwin/radar/policy.py` — AD and host-observation rules.
- `witwin/radar/synthesis/` — waveform synthesis. `witwin/radar/synthesis/fmcw.py` owns FMCW; `ofdm.py` and `pulsed.py` own the other waveforms; `assembly.py` owns shared batches and result assembly.
- `witwin/radar/processing/` — typed processing axes/products split by signal, range-Doppler, angle, detection, and tracking concepts.
- `witwin/radar/cuda/` — the single native runtime boundary and native kernel sources.

The executable architecture contract is `ci/architecture-manifest.json`; the public surface is `ci/public-api-manifest.json` plus `ci/public-api-snapshot.json`. Regenerate the snapshot with `python ci/write_public_api_snapshot.py` when a public signature changes deliberately.

## Code style and duplication

Ruff is the canonical formatter and linter. The width is 120 columns, and a signature or call that fits stays on one line; constructs longer than 120 columns use Ruff's hanging indentation. Do not preserve one-argument-per-line layouts with trailing commas or manually align parameters.

Do not redesign an API merely for formatting. If many parameters form one domain concept and evolve together, introduce a typed concept-owned spec/options dataclass. Do not hide unrelated parameters in a generic bag.

Production equations and non-trivial helpers have one owner. A second implementation is allowed only as an explicitly named independent test/reference oracle. Run:

```bash
python -m ruff format --check witwin/radar tests examples tools ci scripts
python -m ruff check witwin/radar tests examples tools ci scripts
python ci/check_duplicate_code.py
```

## FMCW output contract

FMCW output defaults to a normalized range spectrum. The spectrum family evaluates the 1/N DFT in closed form, as a Dirichlet kernel, which exists only because one delay holds for the whole chirp; it therefore takes no delay rate and refuses one by name. A walking delay is synthesized by the beat family, which evaluates the same linear-delay model in N per chirp, and its spectrum is the processing-owned range transform in N log N. ADC-refreshed scenes synthesize samples natively and take that same range transform; this is what the verbs default to for a dynamic scene, because a constant delay rate does not capture trajectory curvature, moving reflectors, amplitude change, or path births. `FmcwSpec.output_domain` and the flat configuration field `output_domain` default to `"spectrum"`. Use `output_domain="beat"` only when a caller explicitly needs the synthesized time-domain beat signal. Processing must use the `SynthesisResult.axes` metadata instead of inferring the domain from tensor shape.

## Public entry points

There are four verbs, and they are two halves and their fusion.

- `Radar.trace(scene, targets, times=...)` runs the world half — world sampling, Channel epochs, topology, round-trip composition, scattering and antenna weighting — and returns `Paths`. `Radar.echo(paths)` runs the instrument half — waveform synthesis, the receive chain, the output domain and the processing axes — and returns `Result`. `Radar.simulate(...)` fuses the two and stacks every frame; `Radar.stream(...)` fuses them and yields one frame at a time. The three world-facing verbs take the target set as a required positional argument, the propagation request as `los`/`reflections`, the in-frame sampling as `motion`, and the differentiation mode as `grad`.
- All four share one session loop and one synthesis route, so `echo(trace(...))` is bit-identical to `simulate(...)`. `tests/test_trace_echo_split.py` asserts it with `torch.equal` on every motion kind, in both FMCW output domains, with and without a receive chain, and with oscillator phase noise. Do not add a second synthesis route for either half; a cut that moves a number in the last bit is a cut through an equation rather than through a seam.
- `Radar.echo(...)` is the public entry to synthesis over traced rows, and it is the only one. No synthesis function is exported, dispatch on the stored waveform kind stays a private method of `Radar`, and a caller reaches synthesized samples only through the `Result` a verb returns. What `echo` lets vary is the receive chain, the seed and the FMCW output domain; anything the observation schedule was derived from — the carrier, the sensor-pair partition, the waveform behind a sampled schedule — is refused by name, as is a phase-noise receiver against paths traced without ADC instants.
- `Paths` retains every evaluated observation's rows and says so in its docstring; the fused verbs retain one replay group - at most `Motion.batch_observations` observations - at a time. That asymmetry is the contract. Making `simulate` or `stream` materialise a whole frame's observations is a retention regression, not a refactor.
- `Radar` is a flat immutable record. `Radar.from_dict` and `Radar.from_json` load the flat FMCW configuration format, and `Radar.replace(...)` and `Radar.to(device)` return a new radar; nothing edits one in place. A radar holds no run state, so the per-frame typed diagnostics are read from the result that published them.
- Processing is reached from a result, not assembled by a caller: `Result.axes` is the `ProcessingAxes` the echo builds once from the waveform spec and the array that produced the cube, `Result.axis_names` is the cube's axis-name tuple, and `Result.frame(i)` is the `Frame` that pairs one frame's cube with that metadata. `Frame`'s methods are facades that compute nothing. Public processing functions and typed products are exported from `witwin.radar.processing`.
- Channel integration is internal; callers do not import internal adapter objects through the Radar facade.

## Tests and static gates

From the repository root:

```bash
pytest tests/
pytest tests/ --gpu
pytest tests/processing/ -v
python ci/run_ci_tier.py quick
```

Important static checks include:

```bash
python ci/check_architecture.py
python ci/check_duplicate_code.py
python ci/check_documentation_surface.py
python ci/check_native_bindings.py
python ci/check_public_api_manifest.py
python ci/check_workflows.py
python -m mypy --config-file ci/mypy.ini
```

`ci/native-binding-manifest.json` is generated: after adding, renaming or removing a native operator, run `python ci/check_native_bindings.py --write`. Only its `no_backward` and `error_owners` sections are written by hand.

`ci/mypy.ini` types the scene-session modules (`simulation`, `motion`, `binding`, `adaptive`, `results`, `paths`, `propagation`) strictly; widen its `files` list as other modules gain annotations.

Required-Channel workflows install the Channel extra, record `build_info()["build_fingerprint"]`, and allow zero missing-Channel skips. Do not describe a skipped integration suite as passing.

## Platform and release policy

Simulation and native synthesis require CUDA. CPU construction and most processing tests remain available for configuration and DSP work. Linux and Windows are supported.

Release artifacts use exact Torch/CUDA/ABI runtime identity. A loader refusal is a failed release cell, never successful compatibility evidence. Linux wheels target `manylinux_2_28`. Current executable policy is `ci/release-policy.json`.

## Documentation discipline

`README.md`, `FEATURE_LIST.md`, `PERFORMANCE.md`, `docs/pipeline_guide.md`, the AD capability/ledger documents, the release deferral register, and the development standards are living documents; `ci/documentation-manifest.json` lists them. Finished plans, phase records and superseded measurements live in `docs/dev/archive/`, indexed by its `README.md`; the dated reports in `docs/dev/audit/` are historical too. Historical records must be labeled historical and must not be treated as current instructions. Never claim a benchmark, GPU suite, wheel load, or remote workflow passed unless that exact action was executed and its evidence was retained.

Comments explain invariants, ownership, units, sign/phasor conventions, normalization, numerical choices, and refusal boundaries; they do not narrate syntax. Equations must state their symbols and SI units, sign convention, normalization, validity domain, factor owners, and a test oracle. Channel owns one-way transport; Radar owns round-trip composition, scattering, sensors/frontend, waveform synthesis, and processing. Processing consumes result metadata and never infers a waveform domain from tensor shape.

The normative development standard is `docs/dev/standards/radar-adr-021-code-layout-comments-and-mathematical-ownership.md`. What governance tests and gates are for, and which kinds are not kept, is `docs/dev/standards/radar-adr-026-generated-governance-and-trimmed-gates.md`: behaviour tests and the public-API snapshot are never weakened; a test that only pins source text, layout, line numbers, hashes, counts or milestone history is replaced by a derived check or deleted. Absolute wall-clock budgets run only with `pytest --gpu --benchmark`.