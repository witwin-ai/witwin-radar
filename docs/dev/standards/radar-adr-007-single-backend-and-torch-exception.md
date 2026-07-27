# R-ADR-007: Single production backend, Dr.Jit prohibition, Torch/DSP exception

Status: Accepted (Phase 4), amended (Phase 5), deviation closed (Phase 11)

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

The per-frame two-way join is the Phase-5 instance:
`witwin/radar/cuda/kernels/two_way_join.cu` owns it. The AST scan for that one
is scoped to the `compose` FUNCTION rather than to the module, because `freeze`
legitimately iterates - once per frozen topology, on the host, after the
consumer has already synchronized. Scanning the whole file would either forbid
that or force a blanket exemption that stops meaning anything.

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

### Dr.Jit is gone (Phase 5)

The promise made in the Phase-4 revision of this ADR is kept. `trace.py`,
`material.py`, and `_rayd_bridge.py` - the only three production files that
imported Dr.Jit, on five lines - are deleted, and the gate is now the strict
PROCESS-GLOBAL form: after `import witwin.radar`, neither `drjit` nor `rayd`
appears in `sys.modules` by any route. The static AST closure is retained as a
second layer and no longer has to exclude the package root by name. The
baseline-delta assertion that was written to fail loudly at this moment did
exactly that, and was converted rather than deleted.

`render_depth` was deleted with `trace.py` per owner directive: no port, no
preservation, no deprecation shim.

The removal is hard, not a deprecation window. `Tracer`, `fresnel`,
`Radar.simulate`, and `Radar.simulate_group` raise with a message naming their
replacement, and a module-level `__getattr__` makes
`from witwin.radar import Tracer` produce that message rather than a bare
`ImportError`. Nothing falls back, because the replacement is a different
contract rather than a drop-in: a shim returning numbers under the old name
would be returning numbers from a different model. `pyproject.toml` drops the
`drjit` and `rayd-drjit` pins in the same change, since leaving them would keep
forcing Dr.Jit into every install after the modules are gone.

### Recorded deviation: `solvers/common.py` path physics stays (Phase 5) - CLOSED in Phase 11

**This section is superseded and its central claim is now false.** It said that
`end_to_end_caller` was "in every case a `DirichletSolver` method". There is no
`DirichletSolver`: Phase 11 deleted `witwin/radar/solvers/` entirely, together
with `synthesis/dirichlet_spectrum.py`, `sensors/legacy_paths.py`,
`cuda/kernels/dirichlet.cu` and the nine `dirichlet_spectrum` ABI symbols. The
manifest holds 25 operators, `RADAR_ABI_VERSION` is 2, and every remaining
symbol names a production caller.

The deviation is closed rather than merely removed. Its own argument was that
deleting `solvers/common.py` while its native replacement was out of scope would
orphan six ABI symbols; the replacement exists (the `sensor_weight` family, with
a production caller on the scene-driven route since Phase 11's antenna-pattern
stage), so the whole route could be deleted in one commit and no symbol passed
through a caller-free state. The Torch expressions themselves survive only as
the independent float64 oracle `tests/reference/path_math.py`, which is where a
CPU reference belongs.

The original text follows, for the record.

> Per-path geometry and amplitude math in `witwin/radar/solvers/common.py` is
> Torch, is production, and is NOT removed here. The reason is not convenience.
> `solver_dirichlet.py` consumes six of its helpers, and six of the nine
> manifested native symbols are the `dirichlet_spectrum` family whose
> `end_to_end_caller` is in every case a `DirichletSolver` method reached
> through `Radar.mimo` / `chirp` / `frame` / `mimo_from_paths` /
> `path_cache_from_trace`. Deleting the only production owner of that math
> while its native replacement is explicitly out of scope would orphan six ABI
> symbols and force the binding manifest and the CUDA sources into an
> architecture-cleanup commit. That is not a scope reduction; it is a
> contradiction.

What did move is the pair of helpers with ZERO production callers,
`pytorch_chirp_reference` and `pytorch_mimo_from_samples`. A CPU/Torch reference
oracle belongs under `tests/`, and those two were shipping inside the wheel;
they are now `tests/reference/dsp_oracles.py`, unchanged, so every comparison
that used them still means the same thing.

The residual surface was FROZEN by test:
`test_the_residual_torch_path_surface_is_frozen` enumerated exactly the names
`solvers/common.py` could define, so "recorded deviation" could not quietly
become "growing exception". That test is gone with the module it froze, and
`tests/test_phase5_removed_entry_points.py` now asserts the stronger claim that
`witwin/radar/solvers/` does not exist.

### No finite differences in production

Production derivatives come from registered native forward/JVP/VJP companions.
Finite differences appear only in `tests/`, as oracles.

### What the AD is verified against, per leg component (Phase 5)

Stated by component, because "the chain is FD-verified" was true of the
line-of-sight chain and was being read as a statement about all of it.

- **The join's own AD**: float64 Torch oracle, itself validated by float64
  central differences before anything is compared against it, then production
  float32 in both modes. `tests/test_phase5_two_way_join_ad.py`.
- **Line-of-sight legs, end to end**: against the independent float64 chain in
  `tests/support/reference_chain.py`, both modes.
  `tests/test_phase4_spike_e2e.py`.
- **Reflection legs, end to end**: against a finite difference of the
  PRODUCTION chain at perturbed positions, both modes, with the loss weighted
  to zero on the one composed row that joins two line-of-sight legs so the
  whole gradient is the reflection rows'. `tests/test_phase5_reflection_ad.py`.
  A reimplemented oracle is not available here and should not be built: it
  would duplicate Channel's lossy-dielectric Fresnel coefficient, which is a
  Channel/RayD numerical owner.

The remaining gap, stated so it is not read as covered: the transfer's site
dependence is dominated by the specular-point propagation phase, so these tests
verify the reflection TRANSPORT derivative rather than the Fresnel
coefficient's own derivative in isolation. Separating the two needs an oracle
for the coefficient, which is the duplication above.

## Consequences

The scan that enforces this uses the AST, not text: every forbidden token also
appears in these modules' docstrings, where it documents the rule rather than
breaking it. A text scan flagged all of them and would have to be weakened to
pass, which is exactly the wrong direction.

## Acceptance evidence

- `tests/test_phase4_import_boundary.py::test_no_drjit_or_rayd_in_the_process_after_importing_witwin_radar`
  (the strict process-global assertion)
- `tests/test_phase4_import_boundary.py::test_the_synthesis_hot_loop_is_native_not_torch`
- `tests/test_phase4_import_boundary.py::test_the_two_way_join_hot_loop_is_native_not_torch`
- `tests/test_phase4_import_boundary.py::test_no_drjit_reference_of_any_kind_in_the_new_modules`
- `tests/test_phase4_import_boundary.py::test_the_spike_adds_no_drjit_rayd_or_channel_internals`
- `tests/test_phase4_two_way.py::test_a_geometry_dependent_response_is_refused`
- `tests/test_phase5_removed_entry_points.py` (removed names raise and name
  their replacement; the residual Torch surface is frozen; the packaging
  metadata no longer pulls in Dr.Jit)
- `tests/test_phase5_reflection_ad.py` (reverse and forward mode through a
  reflection leg against finite differences of the production chain, with a
  term-level control that isolates the transfer from the delay)
