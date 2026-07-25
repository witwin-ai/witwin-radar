# R-ADR-007: Single production backend, Dr.Jit prohibition, Torch/DSP exception

Status: Accepted (Phase 4)

## Context

Radar's production boundary runs from a `SceneSnapshot` / `CompiledScene` plus a
radar configuration to IQ samples and the processing input downstream of them.
Inside that boundary there is exactly one numerical backend. The interesting
question is not "is Torch banned" -- it plainly is not, since Torch is the tensor
API -- but where the line sits.

## Decision

### The hot loop is native

Per-path, per-sample waveform evaluation runs in a CUDA kernel. The FMCW beat
sum is the Phase-4 instance: `witwin/radar/cuda/kernels/fmcw_beat.cu` owns it,
and the Python facade contains no loop, no comprehension, and no
`torch.exp`/`sin`/`cos` over paths.

### The Torch allowlist

Torch may be used for:

- contract validation and typed-contract construction;
- orchestration and dispatch, including autograd dispatch to native companions;
- metadata-only packing, views, gathers, and result assembly;
- Core scene authoring;
- cuFFT-backed `torch.fft` and comparable DSP primitives;
- reference oracles under `tests/`, never imported by production code.

`ScalarRcsResponse` sits inside this allowlist as a per-target broadcast
parameter scale: one complex number per target, broadcast across its rows. It is
not per-path physics. Its aspect-dependent, material-informed, and polarimetric
successors DO vary per path, and those go native. The `ScatterResponse` protocol
carries `is_geometry_dependent` so the distinction is a checked property rather
than a comment, and `TwoWayComposer.compose` refuses a geometry-dependent
response outright.

### Dr.Jit

No new production path may reference Dr.Jit. The legacy edge
`witwin/radar/__init__.py -> trace.py -> drjit` still exists and is scheduled for
deletion; it is measured, not tolerated silently.

Because that edge makes a strict process-global `sys.modules` assertion
unachievable today, the Phase-4 gate is the STATIC AST CLOSURE form: no module
added by this spike names Dr.Jit, directly or transitively, excluding exactly the
one package-root edge, which is named. The strict process-global form is promised
at Phase-5 exit, when `trace.py` is deleted, and the baseline-delta test carries
an assertion that fails loudly at that point telling the next author to tighten
it.

### No finite differences in production

Production derivatives come from registered native forward/JVP/VJP companions.
Finite differences appear only in `tests/`, as oracles.

## Consequences

The scan that enforces this uses the AST, not text: every forbidden token also
appears in these modules' docstrings, where it documents the rule rather than
breaking it. A text scan flagged all of them and would have to be weakened to
pass, which is exactly the wrong direction.

## Acceptance evidence

- `tests/test_phase4_import_boundary.py::test_the_synthesis_hot_loop_is_native_not_torch`
- `tests/test_phase4_import_boundary.py::test_no_drjit_reference_of_any_kind_in_the_new_modules`
- `tests/test_phase4_import_boundary.py::test_the_spike_adds_no_drjit_or_rayd_over_the_radar_baseline`
- `tests/test_phase4_two_way.py::test_a_geometry_dependent_response_is_refused`
