"""List or execute the radar repository's four fail-fast CI tiers.

Modelled on `channel/ci/run_ci_tier.py`, which exists for a reason worth
restating: before it, "did CI pass?" meant reading three workflow files and
hoping the local commands matched them. A tier is one name that a person and a
workflow can both run, and the workflows here now call these tiers rather than
restating their steps.

    quick    static gates plus the CPU suite. No GPU, no built wheel, no
             native binary required - this is what `quality.yml` runs on a
             hosted CPU runner and what a developer runs before pushing.
    cuda     quick plus the GPU suite, the loader contract, and the extension
             boundary audit, which reads the shipped binary. `gpu-regression.yml`
             runs this.
    nightly  cuda plus the packaging chain: build the native library, build the
             wheel, smoke the wheel from a fresh isolated install, and prove the
             three wheels coexist.
    release  nightly plus the complete release SASS verification.

Two properties are deliberate.

`nightly.native-prebuild` runs `scripts/build_radar_cuda_prebuilt.py` as its own
process. R-ADR-019 forbids compiling inside a test or user process - the MSVC
environment that step prepares breaks a library loaded later in the same
process - and a separate gate is exactly one separate process.

`nightly.coexistence-smoke` needs a Channel wheel, which this repository cannot
build: Channel source-links RayD and owns that build entirely. The gate reads
`artifacts/nightly/wheels/channel/`, and fails loudly naming that directory when
it is empty. A radar tier that quietly skipped the cross-package check would be
worse than one that asks for the artifact.

The four production static gates (Phase-10 work item 7: forbidden runtimes,
oracle isolation, raw native access, the Torch-physics allowlist) sit in
`quick`, next to `quick.native-bindings`. None of them imports the package or
needs a GPU, so the cheapest tier is the one that should catch them; the only
half that cannot run there is the oracle gate's wheel-member check, which is a
separate `nightly` gate against the wheel the tier just built.

`quick.orphan-modules` joined them in Phase 11 for the same reason - it parses
the tree and imports nothing - and closes the dead-code half of that phase's
acceptance criterion 8. Its sibling, the frozen public-API snapshot, is a test
rather than a gate script (`tests/test_public_api_snapshot.py`), so it runs
inside `quick.cpu-tests`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Gate:
    id: str
    args: tuple[str, ...]

    def argv(self, python: str) -> tuple[str, ...]:
        return (python, *self.args)


WHEEL_ROOT = "artifacts/nightly/wheels"

QUICK_GATES = (
    Gate(
        "quick.format",
        ("-m", "ruff", "format", "--check", "witwin/radar", "tests", "examples", "tools", "ci", "scripts"),
    ),
    Gate("quick.ruff", ("-m", "ruff", "check", "witwin/radar", "tests", "examples", "tools", "ci", "scripts")),
    Gate("quick.duplicate-code", ("ci/check_duplicate_code.py",)),
    Gate("quick.native-bindings", ("ci/check_native_bindings.py",)),
    # The four production static gates (Phase-10 work item 7). They are here
    # rather than in `cuda` on purpose: none of them imports the package,
    # loads the extension, or needs a GPU, so the cheapest tier is the one
    # that should catch a forbidden import or a widened allowlist.
    Gate("quick.production-dependencies", ("ci/check_production_dependencies.py",)),
    Gate("quick.oracle-isolation", ("ci/check_test_oracle_isolation.py",)),
    Gate("quick.raw-native-access", ("ci/check_raw_native_access.py",)),
    Gate("quick.torch-physics-allowlist", ("ci/check_torch_physics_allowlist.py",)),
    Gate("quick.workflow-policy", ("ci/check_workflow_policy.py",)),
    # Dead code, Phase-11 work item 7. Ruff reports an unused IMPORT; nothing
    # reported an unused MODULE, which is how `witwin/radar/timeline.py`
    # survived four phases after its last consumer went away.
    Gate("quick.orphan-modules", ("ci/check_orphan_modules.py",)),
    # Importing the package must not load the loader, the compiler, or CUDA.
    # The lazy __getattr__ in witwin/radar/__init__.py is what makes that true
    # and this is the gate that notices when an eager import creeps back in.
    Gate(
        "quick.import-no-native",
        (
            "-c",
            "import sys; import torch, witwin.radar; "
            "assert 'witwin.radar.cuda.runtime' not in sys.modules; "
            "assert 'torch.utils.cpp_extension' not in sys.modules; "
            "assert not torch.cuda.is_initialized()",
        ),
    ),
    Gate("quick.cpu-tests", ("-m", "coverage", "run", "-m", "pytest", "tests", "-q")),
    Gate("quick.cpu-coverage", ("-m", "coverage", "report", "--fail-under=50")),
)

CUDA_GATES = (
    # Under coverage with the higher floor, because that is exactly what
    # `gpu-regression.yml` asserted before it called this tier. A tier that
    # replaced a 75% gate with a 50% one would be a silent weakening dressed as
    # a consolidation. The GPU run overwrites the CPU run's `.coverage`, which
    # is the pre-existing behaviour: the 75% floor describes the GPU suite.
    Gate("cuda.gpu-tests", ("-m", "coverage", "run", "-m", "pytest", "tests", "--gpu", "-q")),
    Gate("cuda.gpu-coverage", ("-m", "coverage", "report", "--fail-under=75")),
    Gate("cuda.loader-contract", ("-m", "pytest", "-q", "tests/test_phase10_loader_contract.py")),
    Gate("cuda.extension-boundary", ("ci/check_extension_boundary.py",)),
)

NIGHTLY_GATES = (
    Gate("nightly.core-wheel-build", ("ci/build_core_wheel.py", "--outdir", f"{WHEEL_ROOT}/core", "--no-isolation")),
    Gate("nightly.native-prebuild", ("scripts/build_radar_cuda_prebuilt.py", "--developer")),
    Gate("nightly.wheel-build", ("-m", "build", "--wheel", "--no-isolation", "--outdir", f"{WHEEL_ROOT}/radar")),
    # `quick.oracle-isolation` can only check the CONFIGURATION - what the
    # build was asked for. This checks what came out, on the wheel that was
    # just built, which is a different question once a build hook is involved.
    Gate("nightly.oracle-isolation-wheel", ("ci/check_test_oracle_isolation.py", "--wheel", f"{WHEEL_ROOT}/radar")),
    Gate(
        "nightly.wheel-smoke",
        (
            "ci/wheel_smoke.py",
            f"{WHEEL_ROOT}/radar",
            "--core-wheel",
            f"{WHEEL_ROOT}/core",
            "--output",
            "artifacts/nightly/radar-wheel-smoke.v1.json",
        ),
    ),
    Gate(
        "nightly.coexistence-smoke",
        (
            "ci/coexistence_smoke.py",
            "--core-wheel",
            f"{WHEEL_ROOT}/core",
            "--channel-wheel",
            f"{WHEEL_ROOT}/channel",
            "--radar-wheel",
            f"{WHEEL_ROOT}/radar",
            "--output",
            "artifacts/nightly/coexistence.v1.json",
        ),
    ),
)

RELEASE_GATES = (
    # The default expectation is the complete release SASS set plus
    # compute_120 PTX. A developer build restricted to the local architecture
    # FAILS here, which is the correct answer: only a release-matrix binary can
    # pass it, and that is what makes it a release gate.
    Gate(
        "release.arch-verification",
        ("scripts/verify_cuda_binary_arches.py", "--stem", "_radar_native", "witwin/radar/cuda/prebuilt"),
    ),
)


def _tier(name: str, *gate_groups: tuple[Gate, ...]) -> tuple[Gate, ...]:
    del name
    return tuple(gate for group in gate_groups for gate in group)


TIER_GATES = {
    "quick": _tier("quick", QUICK_GATES),
    "cuda": _tier("cuda", QUICK_GATES, CUDA_GATES),
    "nightly": _tier("nightly", QUICK_GATES, CUDA_GATES, NIGHTLY_GATES),
    "release": _tier("release", QUICK_GATES, CUDA_GATES, NIGHTLY_GATES, RELEASE_GATES),
}


def format_gate(gate: Gate, python: str) -> str:
    return subprocess.list2cmdline(gate.argv(python))


def run_gates(gates: tuple[Gate, ...], *, python: str, root: Path, dry_run: bool = False) -> int:
    for gate in gates:
        command = format_gate(gate, python)
        prefix = "DRY-RUN" if dry_run else "RUN"
        print(f"[{prefix}] {gate.id}: {command}", flush=True)
        if dry_run:
            continue
        completed = subprocess.run(gate.argv(python), cwd=root, check=False)
        if completed.returncode:
            print(f"[FAIL] {gate.id}: exit code {completed.returncode}", file=sys.stderr, flush=True)
            return completed.returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tier", choices=sorted(TIER_GATES))
    parser.add_argument("--list", action="store_true", help="print the gates and exit")
    parser.add_argument("--dry-run", action="store_true", help="print each command")
    parser.add_argument("--python", default=sys.executable)
    arguments = parser.parse_args(argv)

    gates = TIER_GATES[arguments.tier]
    root = Path(__file__).resolve().parents[1]

    if arguments.list:
        for gate in gates:
            print(f"{gate.id}: {format_gate(gate, arguments.python)}")
        return 0
    return run_gates(gates, python=arguments.python, root=root, dry_run=arguments.dry_run)


if __name__ == "__main__":
    sys.exit(main())
