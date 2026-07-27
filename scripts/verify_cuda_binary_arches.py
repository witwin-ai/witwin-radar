from __future__ import annotations

import argparse
import subprocess
import tempfile
import zipfile
from pathlib import Path


EXPECTED_SASS = ("70", "75", "80", "86", "87", "89", "90", "100", "101", "120")
EXPECTED_PTX_TARGET = "sm_120"

#: The reduced set the prebuild policy allows for an opt-in pull-request smoke
#: build. It is a COMPILE gate, never a release artifact, and the workflow that
#: selects it also refuses to upload a wheel. Naming it here means the reduced
#: profile is verified rather than unverified: without an expectation to pass,
#: "we built fewer architectures" and "the build silently dropped one" look the
#: same. Channel's verifier already takes explicit expectations; this brings
#: Radar's to the same shape.
SMOKE_SASS = ("87", "120")

#: Radar ships exactly one native artifact, so the stem has a default. Passing
#: ``--stem`` still overrides it, which is what verifying a foreign binary
#: needs; omitting it no longer means "verify nothing".
DEFAULT_STEM = "_radar_native"


def _matches(path: Path, stems: tuple[str, ...]) -> bool:
    return path.suffix in {".pyd", ".so"} and any(path.name.startswith(stem) for stem in stems)


def _collect_binaries(inputs: list[Path], stems: tuple[str, ...], extract_root: Path) -> list[Path]:
    binaries: list[Path] = []
    for input_path in inputs:
        if input_path.suffix == ".whl":
            with zipfile.ZipFile(input_path) as wheel:
                for name in wheel.namelist():
                    member = Path(name)
                    if not _matches(member, stems):
                        continue
                    wheel.extract(name, extract_root)
                    binaries.append(extract_root / member)
            continue
        if input_path.is_dir():
            binaries.extend(path for path in input_path.rglob("*") if _matches(path, stems))
            continue
        if _matches(input_path, stems):
            binaries.append(input_path)
    return sorted(set(path.resolve() for path in binaries))


def _cuobjdump(flag: str, binary: Path) -> str:
    result = subprocess.run(
        ["cuobjdump", flag, str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return f"{result.stdout}\n{result.stderr}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CUDA SASS and PTX targets in release binaries.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--stem", action="append", default=None)
    parser.add_argument(
        "--expected-sass",
        default=",".join(EXPECTED_SASS),
        help=(
            "comma-separated SASS architectures that must be present. Defaults "
            "to the complete release set; the reduced pull-request smoke set is "
            f"{','.join(SMOKE_SASS)}."
        ),
    )
    parser.add_argument(
        "--expected-ptx",
        default=EXPECTED_PTX_TARGET.removeprefix("sm_"),
        help="compute capability whose PTX must be present, without a dot.",
    )
    args = parser.parse_args()
    stems = tuple(args.stem) if args.stem else (DEFAULT_STEM,)
    expected_sass = tuple(
        entry.strip() for entry in args.expected_sass.split(",") if entry.strip()
    )
    if not expected_sass:
        raise SystemExit("--expected-sass must name at least one architecture.")
    expected_ptx_target = f"sm_{args.expected_ptx.strip()}"

    with tempfile.TemporaryDirectory(prefix="witwin_cuda_arch_verify_") as temp_dir:
        binaries = _collect_binaries(args.inputs, stems, Path(temp_dir))
        if not binaries:
            raise SystemExit(f"No native binaries matching {list(stems)!r} were found.")

        for binary in binaries:
            elf_listing = _cuobjdump("--list-elf", binary)
            missing_sass = [arch for arch in expected_sass if f"sm_{arch}" not in elf_listing]
            if missing_sass:
                raise SystemExit(f"{binary} is missing SASS targets: {', '.join(missing_sass)}")

            ptx_dump = _cuobjdump("--dump-ptx", binary)
            if f".target {expected_ptx_target}" not in ptx_dump:
                raise SystemExit(
                    f"{binary} is missing {expected_ptx_target.replace('sm_', 'compute ')} PTX."
                )

            print(
                f"Verified CUDA architectures in {binary}: SASS "
                f"{','.join(expected_sass)} plus {expected_ptx_target} PTX"
            )


if __name__ == "__main__":
    main()
