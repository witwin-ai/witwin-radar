from __future__ import annotations

import subprocess
import sys
import textwrap


def test_trace_module_imports_without_mitsuba_runtime_dependency():
    script = r"""
import importlib.abc
import json
import sys


class BlockMitsuba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mitsuba" or fullname.startswith("mitsuba."):
            raise ModuleNotFoundError("blocked mitsuba import")
        return None


sys.meta_path.insert(0, BlockMitsuba())
import witwin.radar.trace as trace

print(json.dumps({
    "tracer": trace.Tracer.__name__,
    "mitsuba_loaded": "mitsuba" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=".",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"tracer": "Tracer"' in result.stdout
    assert '"mitsuba_loaded": false' in result.stdout
