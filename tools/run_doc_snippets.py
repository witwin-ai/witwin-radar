"""Run the Python code blocks of the user documentation, one fresh process per document.

Every fenced block that opens with exactly three backticks and ``python`` in a
Markdown file is executed; the blocks of one file run in order in one
namespace, so a later block may use names an earlier block defined. A block
that is illustration only (it needs files the repository does not ship, such
as SMPL model files or a capture) opens with ``python no-run`` instead and is
skipped. The scripts are written under the ignored ``build/doc_snippets/`` and removed
after the run. Scene-driven snippets need CUDA and Channel, so run this on a GPU:

    python tools/run_doc_snippets.py                    # README.md and docs/user/*.md
    python tools/run_doc_snippets.py docs/user/motion.md

The exit status is the number of documents whose blocks failed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = ("README.md", "docs/user/*.md")
BLOCK = re.compile(r"^```python[ \t]*(?P<tag>no-run)?[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)


def blocks(path: Path) -> list[str]:
    """The runnable blocks of one document, in order."""

    text = path.read_text(encoding="utf-8")
    return [match["body"] for match in BLOCK.finditer(text) if match["tag"] is None]


def documents(arguments: list[str]) -> list[Path]:
    patterns = arguments or list(DEFAULT_DOCUMENTS)
    found: list[Path] = []
    for pattern in patterns:
        matches = sorted(REPO_ROOT.glob(pattern)) if any(c in pattern for c in "*?[") else [REPO_ROOT / pattern]
        found.extend(path for path in matches if path.is_file())
    return found


def run(path: Path, timeout_s: float) -> tuple[bool, float, str]:
    """Execute one document's blocks in a fresh interpreter rooted at the repository."""

    source = "\n\n".join(blocks(path))
    if not source.strip():
        return True, 0.0, "no runnable blocks"
    scratch = REPO_ROOT / "build" / "doc_snippets"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=f"_{path.stem}.py", dir=scratch, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(source)
        script = handle.name
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, script],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(filter(None, (str(REPO_ROOT), os.environ.get("PYTHONPATH")))),
            },
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - start, f"timed out after {timeout_s:.0f} s"
    finally:
        Path(script).unlink(missing_ok=True)
    elapsed = time.perf_counter() - start
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, elapsed, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("documents", nargs="*", help="Markdown files or globs relative to the repository root")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per document")
    parser.add_argument("--verbose", action="store_true", help="print each document's output")
    options = parser.parse_args()
    failures = 0
    for path in documents(options.documents):
        ok, elapsed, output = run(path, options.timeout)
        label = path.relative_to(REPO_ROOT)
        print(f"{'ok  ' if ok else 'FAIL'} {label} ({len(blocks(path))} blocks, {elapsed:.1f} s)")
        if not ok or options.verbose:
            print("    " + output.replace("\n", "\n    "))
        failures += not ok
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
