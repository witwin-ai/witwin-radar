# Radar Repository Guide

## Purpose

WiTwin Radar is a differentiable radar simulator. `witwin.core` owns world and geometry state, `witwin.channel` owns one-way propagation, and Radar owns round-trip path composition, scattering, sensor/frontend effects, waveform synthesis, and DSP products.

The breaking concept-axis consolidation is complete. Do not add compatibility modules, aliases, fallback imports, deprecated entry points, or warning-only shims. Stable APIs may change. There is no maximum-file-line rule: prefer one clear conceptual owner over artificial file splitting.

## Concept-axis layout

- `witwin/radar/radar.py` — the flat immutable `Radar` record, the `Fmcw`/`Ofdm`/`Pulsed` waveforms, the pose transforms, and the flat-configuration loader.
- `witwin/radar/targets.py` — what the radar is looking at: `PointTargets`, `StructureTargets`, `Aspect`, and the split of one target record into a site policy and a scatter response. The scattering coefficient stays in `scattering.py` and the site binding in `simulation.py`; nothing here computes either.
- `witwin/radar/simulation.py` — scene-session orchestration, the `Motion` sampling record, and `RadarSimulationResult`.
- `witwin/radar/channel.py` — the only production importer of `witwin.channel`; compile, propagation, topology, and kinematics adapters.
- `witwin/radar/propagation.py` — Radar-owned propagation policies and epoch logic.
- `witwin/radar/paths.py` — direct and two-way round-trip path contracts and composition.
- `witwin/radar/scattering.py` — scalar and aspect-dependent scattering responses.
- `witwin/radar/sensors.py` — array geometry, transmit power, antenna patterns, and round-trip weighting.
- `witwin/radar/frontend.py` — receiver chain, noise, ADC, AGC, and deterministic seed contracts.
- `witwin/radar/smpl.py` — Radar-specific SMPL authoring on Core geometry.
- `witwin/radar/policy.py` — AD and host-observation rules.
- `witwin/radar/synthesis/` — waveform synthesis. `witwin/radar/synthesis/fmcw.py` owns FMCW; `ofdm.py` and `pulsed.py` own the other waveforms; `assembly.py` owns shared batches and result assembly.
- `witwin/radar/processing/` — typed processing axes/products split by signal, range-Doppler, angle, detection, and tracking concepts.
- `witwin/radar/cuda/` — the single native runtime boundary and native kernel sources.

The executable architecture contract is `ci/architecture-manifest.json`; the public surface is `ci/public-api-manifest.json` plus `ci/public-api-snapshot.json`.

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

FMCW output defaults to a normalized range spectrum. Stationary paths use native CUDA Dirichlet evaluation; linearly moving paths use native continuous-delay phase summation; ADC-refreshed scenes use native sample synthesis and the processing-owned range transform. `FmcwSpec.output_domain` and the flat configuration field `output_domain` default to `"spectrum"`. Use `output_domain="beat"` only when a caller explicitly needs the synthesized time-domain beat signal. Processing must use the `SynthesisResult.axes` metadata instead of inferring the domain from tensor shape.

## Public entry points

- `Radar.simulate(scene, targets, times=...)` is the scene-driven production entry and returns a typed frame result. `Radar.stream(...)` runs the identical session and yields one frame at a time. Both take the target set as a required positional argument, the propagation request as `los`/`reflections`, the in-frame sampling as `motion`, and the differentiation mode as `grad`.
- `Radar` is a flat immutable record. `Radar.from_dict` and `Radar.from_json` load the flat FMCW configuration format, and `Radar.replace(...)` and `Radar.to(device)` return a new radar; nothing edits one in place. A radar holds no run state, so the per-frame typed diagnostics are read from the result that published them.
- Waveform synthesis has no public entry point. Dispatch on the stored waveform kind is a private method of `Radar`, and a caller reaches synthesized samples only through the result a verb returns.
- Public processing functions and typed products are exported from `witwin.radar.processing`.
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
python ci/check_governance_inventory.py
python ci/check_no_compatibility.py
python ci/check_public_api_manifest.py
python ci/check_release_claims.py
python ci/check_required_channel_coverage.py
python ci/check_workflow_references.py
```

Required-Channel workflows install the Channel extra, record `build_info()["build_fingerprint"]`, and allow zero missing-Channel skips. Do not describe a skipped integration suite as passing.

## Platform and release policy

Simulation and native synthesis require CUDA. CPU construction and most processing tests remain available for configuration and DSP work. Linux and Windows are supported.

Release artifacts use exact Torch/CUDA/ABI runtime identity. A loader refusal is a failed release cell, never successful compatibility evidence. Linux wheels target `manylinux_2_28`. Current executable policy is `ci/release-policy.json`.

## Documentation discipline

`README.md`, `FEATURE_LIST.md`, `PERFORMANCE.md`, `docs/pipeline_guide.md`, the AD capability/ledger documents, the completed consolidation record, and the development standards are living documents. Historical records must be labeled historical and must not be treated as current instructions. Never claim a benchmark, GPU suite, wheel load, or remote workflow passed unless that exact action was executed and its evidence was retained.

Comments explain invariants, ownership, units, sign/phasor conventions, normalization, numerical choices, and refusal boundaries; they do not narrate syntax. Equations must state their symbols and SI units, sign convention, normalization, validity domain, factor owners, and a test oracle. Channel owns one-way transport; Radar owns round-trip composition, scattering, sensors/frontend, waveform synthesis, and processing. Processing consumes result metadata and never infers a waveform domain from tensor shape.

The completed consolidation record is `docs/dev/plans/radar-concept-axis-layout-and-module-consolidation-plan.md`. The normative development standard is `docs/dev/standards/radar-adr-021-code-layout-comments-and-mathematical-ownership.md`. Governance debt and closure evidence are tracked in `docs/dev/audit/radar-governance-debt-and-drift-inventory.md`.