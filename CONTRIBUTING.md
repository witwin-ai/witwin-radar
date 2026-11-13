# Contributing to WiTwin Radar

This file says how to set up a development checkout, build the native library, run the tests and gates, and which
engineering rules a change must follow. The repository memory is [`AGENTS.md`](AGENTS.md) (module ownership, verbs,
gates); the normative code standard is
[R-ADR-021](docs/dev/standards/radar-adr-021-code-layout-comments-and-mathematical-ownership.md). Both are binding.

## Development environment

You need Linux or Windows, an NVIDIA GPU, CUDA 12.8 (`nvcc`), Python 3.10-3.13, `torch==2.10.0`, `witwin` (Core)
0.4 and `witwin-channel` 0.6 (`pip install -e ".[channel,dev]"` from a checkout installs the Python dependencies and
the pinned `ruff` and `mypy`).

On Linux two scripts set up a conda-based checkout:

```bash
source scripts/dev_env.sh    # activates the conda env (RADAR_DEV_CONDA_ENV, default radar-rel), pins the GPU
                             # (CUDA_VISIBLE_DEVICES, default 1) and puts this checkout first on PYTHONPATH
scripts/dev_build.sh         # builds the native library into witwin/radar/cuda/prebuilt/ (CPU only)
```

Read the header of each script for its variables (`RADAR_DEV_CONDA_ENV`, `RADAR_DEV_TOOLS_BIN`,
`RADAR_DEV_CUDA_ARCH`). Run Python and pytest from the repository root: `tests/conftest.py` puts the checkout first on
`sys.path`, and `witwin` is a namespace package, so the checkout's `witwin/radar` is the one imported.

## Building the native library

```bash
TORCH_CUDA_ARCH_LIST=8.6 python scripts/build_radar_cuda_prebuilt.py   # or scripts/dev_build.sh on Linux
```

The build compiles `witwin/radar/cuda/*` into `witwin/radar/cuda/prebuilt/` (the binary plus two identity sidecars,
all gitignored). It always runs as its own process, never inside a test or a user process. The loader refuses a
binary whose recorded source fingerprint differs from the sources, so **rebuild after any edit under
`witwin/radar/cuda/`**. The build directory is keyed by a digest of the sources, so several checkouts do not collide.
A build killed mid-way leaves a torch `FileBaton` lock; `scripts/dev_build.sh` refuses to start on one and names it.

After adding, renaming or removing a native operator, regenerate the binding manifest:

```bash
python ci/check_native_bindings.py --write    # only no_backward and error_owners are written by hand
```

## Tests and gates

```bash
pytest tests/ -q                     # CPU suite; GPU tests skip
pytest tests/ -q --gpu               # GPU suite (needs CUDA, Channel and a built native library)
pytest tests/ -q --gpu --benchmark   # also the absolute wall-clock budgets
python ci/run_ci_tier.py quick       # static gates + CPU suite: what CI runs on every change
python ci/run_ci_tier.py cuda        # quick + GPU suite + loader contract + extension boundary
python ci/run_ci_tier.py nightly     # cuda + native build, wheel build, wheel smoke, coexistence
python tools/run_doc_snippets.py     # run every Python block of README.md and docs/user/*.md (GPU)
```

The quick tier runs, in order: Ruff format and lint (`witwin/radar tests examples tools ci scripts`, 120 columns),
duplicate-code detection, `mypy` over the whole package (`ci/mypy.ini`), the generated native binding manifest,
production dependencies, oracle isolation, workflow lint, orphan modules, the architecture manifest, the public-API
manifest, the documentation surface (living documents name only existing paths and no retired names), and the CPU
suite with coverage. `python ci/run_ci_tier.py quick --dry-run` lists the exact commands.

**Testing cadence.** While working, run only the tests you wrote and the ones that exercise the code you changed. Run
the quick tier and the GPU tests for your area once, when the change is finished, and say exactly what you ran and
the result. The full GPU suite is run once per integration, not after every edit. Never describe a skipped suite as
passing, and never claim a benchmark, GPU run or wheel load that was not executed.

On a shared GPU, run one test process per device slot, keep each kernel launch short (well under 0.5 s; shrink a
scene rather than launch a multi-second kernel), and run anything that may crash the CUDA context in a fresh
subprocess with a timeout.

## Engineering rules

**One native backend, efficient by construction.** Every hot-path computation (anything per row, path, site, sample,
observation or cell) is a native CUDA kernel in `witwin/radar/cuda/` with a hand-written analytic backward (VJP) and
forward-mode (JVP) companion, registered through the one native runtime. Torch is orchestration only: allocation,
views, a small constant number of library calls (cuFFT/cuBLAS through `torch.fft`/`torch.matmul`), never per-element
math chains, Python loops over rows, sites or observations, or host round trips in a hot loop. There is no CPU or
Torch numerical fallback for simulation: fail loudly before native work. Channel is reached only through
`witwin/radar/channel.py`.

**Differentiability is preserved.** A derivative that is genuinely undefined (a discrete choice, a quantiser) is
refused loudly by name, never silently zeroed or detached. Every new native family gets forward, VJP and JVP tests
against finite differences or an analytic oracle, and a launch/synchronization budget test.

**Performance evidence is counts.** Kernel launches, host synchronizations, device copies, Python loop trips and
allocations per frame, pinned by budget tests (`tests/test_session_budget.py` and the per-family budgets). A change
that moves a count re-pins it in the same commit and says why; a performance change tightens it. Timing evidence is a
same-process A/B labelled contended, minimum of at least four runs, raw numbers reported.

**Numbers.** A numerical change for correctness or physics gets a short design record under `docs/dev/standards/`
(`radar-adr-NNN-<slug>.md`) and evidence from an independent oracle (analytic, float64 reference, image method). A
bit-level baseline that the change legitimately moves is regenerated, and the commit message says which and why.
Structural and performance changes are bit-identical (`torch.equal`) unless the design record explains why not.

**Tests.** Every behaviour change gets a test that fails before and passes after. GPU tests are marked `gpu` and skip
cleanly without CUDA. Behaviour tests and the public-API snapshot are never weakened; a test that only pins source
text, layout, line numbers, hashes or counts of files is replaced by a generated check or deleted (R-ADR-026). The
analytic oracle and survival suites live in `tests/oracle/` (R-ADR-032); a fix for a defect one of them marks
`xfail(strict=True, reason="<finding> ...")` removes that marker in the same commit. Beware `pytest.approx`: its
default absolute tolerance is 1e-12 and received powers are about 1e-13 W, so compare ratios or decibels, or pass
`abs=0`.

**Public API.** No compatibility modules, aliases or deprecated entry points. A public signature change regenerates
the snapshot (`python ci/write_public_api_snapshot.py`) and updates `ci/public-api-manifest.json`, and gets a line in
`CHANGELOG.md` under `[Unreleased]` (with the migration for a breaking change). Names follow R-ADR-049: Channel's name
for a quantity Channel also publishes, no unit suffix otherwise.

**Code and comments.** Ruff formats and lints at 120 columns; a signature that fits stays on one line. Each equation
has one owner; a second implementation exists only as a named test oracle. Comments and docstrings state invariants,
units, sign and phasor conventions, normalisation and refusal boundaries; they describe current behaviour only - no
history, no benchmark numbers, no plan item IDs. Error messages name the record and field that is wrong.

**Documentation.** User guides live in `docs/user/` and every Python block in them must run
(`tools/run_doc_snippets.py`; a block that needs files the repository does not ship is fenced `python no-run` and
says why). Living documents are listed in `ci/documentation-manifest.json`; finished plans and superseded
measurements move to `docs/dev/archive/`.

## Commits

Small logical commits, one per work item or sub-item, with the repository's prefixes: `feat(radar):`, `fix(radar):`,
`perf(radar):`, `refactor(radar):`, `test(radar):`, `docs(radar):`, `ci(radar):`, `build(radar):`. A numerical change
names the baselines it regenerates. Never commit the native binary or its sidecars.
