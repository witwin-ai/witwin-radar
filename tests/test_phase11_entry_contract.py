"""The public entry-point contract after the Phase-11 cutover.

This is where ``tests/solvers/test_dirichlet_only_contract.py`` landed. That file
froze "there is exactly one solver and no way to pick another", which was a real
statement while a ``SolverBackend`` enum existed and a second backend had just
been removed. Two of its four assertions still mean something once the Dirichlet
route is gone and they are here; its fourth test,
``test_native_cuda_extension_sources_are_packaged``, made two claims that were
dropped rather than moved, and each has a better owner now:

* ``"dirichlet.cu" in extension_sources()`` named one kernel of the deleted
  route. Its replacement is
  ``test_phase4_binding_manifest.py::test_every_manifested_source_is_a_build_input``,
  which asserts SET EQUALITY between ``extension_sources()`` and
  ``ci/native-binding-manifest.json``, so the whole build input is checked
  rather than two names a contributor happened to list.
* ``prebuilt_root().name == "prebuilt"`` asserted a directory name.
  ``test_phase10_wheel_packaging.py`` asserts the thing that name was standing
  in for: the wheel declares the prebuilt artifacts and sidecars, and the
  smoke hashes every compiled translation unit.

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

#: Resolved from ``__file__`` rather than from the process CWD: the suite is run
#: from the repository root and from ``radar/`` both, and a relative read would
#: turn a directory choice into a test failure.
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_solver_backend_selector_is_not_public_api():
    import witwin.radar as wr

    assert "SolverBackend" not in wr.__all__
    assert "backend" not in inspect.signature(wr.Radar).parameters


def test_radar_rejects_backend_keyword():
    """The constructor takes SI fields, and a selector is not one of them.

    Built field by field rather than from a shared fixture mapping: the claim
    is about the keyword surface of ``Radar.__init__`` itself, so the test that
    makes it should not depend on what a configuration loader happens to fill in.
    """

    from witwin.radar import Fmcw, Radar

    waveform = Fmcw(
        slope=60.012e12,
        sample_rate=4.4e6,
        samples_per_chirp=256,
        chirps_per_frame=2,
        adc_start=0.0,
        idle=7e-6,
        ramp_end=58e-6,
    )
    with pytest.raises(TypeError, match="backend"):
        Radar(
            carrier=77e9,
            waveform=waveform,
            tx=[[0.0, 0.0, 0.0]],
            rx=[[0.0, 0.0, 0.0]],
            power=12.0,
            device="cpu",
            backend="dirichlet",
        )


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

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = [pyproject["project"]["dependencies"]]
    groups.extend(pyproject["project"].get("optional-dependencies", {}).values())
    assert not any(
        dependency.split("[", 1)[0].split(">=", 1)[0] == "slangtorch" for group in groups for dependency in group
    )
