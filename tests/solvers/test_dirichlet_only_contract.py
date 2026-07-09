"""Public contract for the Dirichlet-only radar solver."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


def test_solver_backend_selector_is_not_public_api():
    import inspect

    import witwin.radar as wr

    assert "SolverBackend" not in wr.__all__
    assert "backend" not in inspect.signature(wr.Radar).parameters


def test_radar_rejects_backend_keyword(minimal_config):
    from witwin.radar import Radar

    with pytest.raises(TypeError, match="backend"):
        Radar(minimal_config, backend="dirichlet", device="cpu")


def test_runtime_dependencies_do_not_include_slangtorch():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    assert not any(dependency.split("[", 1)[0].split(">=", 1)[0] == "slangtorch" for dependency in dependencies)


def test_native_cuda_extension_sources_are_packaged():
    from witwin.radar.cuda import build

    source_names = {path.name for path in build.extension_sources()}
    assert "extension.cpp" in source_names
    assert "dirichlet.cu" in source_names
    assert build.prebuilt_root().name == "prebuilt"
