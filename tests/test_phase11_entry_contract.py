"""The public entry-point contract after the Phase-11 cutover.

This is where ``tests/solvers/test_dirichlet_only_contract.py`` landed. That file
froze "there is exactly one solver and no way to pick another", which was a real
statement while a ``SolverBackend`` enum existed and a second backend had just
been removed. Two of its four assertions still mean something once the Dirichlet
route is gone and they are here; the other two are recorded in
``wf13/handoff-test-deletions.md`` with what replaced them.

What is frozen here is narrower and permanent: the scene-driven entry has no
backend selector, no ``simulate_group``, and no dependency on the solver
toolchain the removed route needed. A selector re-appearing is how a fallback
gets reintroduced without anyone deciding to add one.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import tomllib


def test_solver_backend_selector_is_not_public_api():
    import witwin.radar as wr

    assert "SolverBackend" not in wr.__all__
    assert "backend" not in inspect.signature(wr.Radar).parameters


def test_radar_rejects_backend_keyword(minimal_config):
    from witwin.radar import Radar

    with pytest.raises(TypeError, match="backend"):
        Radar(minimal_config, backend="dirichlet", device="cpu")


def test_the_simulation_entry_has_no_backend_or_solver_keyword():
    """The same statement about the entry the cutover made public.

    ``Radar.simulate`` is the surface a caller reaches now, so the "no selector"
    claim has to be made about IT and not only about the constructor. A
    ``backend=`` or ``solver=`` keyword here would be the selector coming back
    one level down.
    """

    from witwin.radar import Radar

    parameters = inspect.signature(Radar.simulate).parameters
    for name in ("backend", "solver", "engine"):
        assert name not in parameters, (name, tuple(parameters))
    assert not hasattr(Radar, "simulate_group")


def test_runtime_and_optional_dependencies_do_not_include_slangtorch():
    """Kept deliberately, though it is not one of the two named survivors.

    ``slangtorch`` was the second-backend toolchain, and this is the only
    assertion in the tree that the distribution never grows a dependency on it
    again. Letting it die with its file would have been a silent loss of a
    dependency gate rather than a migration.
    """

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    groups = [pyproject["project"]["dependencies"]]
    groups.extend(pyproject["project"].get("optional-dependencies", {}).values())
    assert not any(
        dependency.split("[", 1)[0].split(">=", 1)[0] == "slangtorch" for group in groups for dependency in group
    )
