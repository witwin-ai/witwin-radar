"""Public contract for the Dirichlet-only radar solver."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


def test_solver_backend_public_api_is_dirichlet_only():
    from witwin.radar import SolverBackend

    assert [member.value for member in SolverBackend] == ["dirichlet"]
    assert SolverBackend.DIRICHLET.value == "dirichlet"
    assert not hasattr(SolverBackend, "PYTORCH")
    assert not hasattr(SolverBackend, "SLANG")


def test_radar_rejects_removed_backend_names(minimal_config):
    from witwin.radar import Radar

    for backend in ("pytorch", "slang"):
        with pytest.raises(ValueError, match="Only the 'dirichlet' backend is supported"):
            Radar(minimal_config, backend=backend, device="cpu")


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
