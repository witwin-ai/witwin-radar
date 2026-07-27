"""The removed Dr.Jit entry points fail loudly and name their replacement.

Deleting a backend is only half of the job. The other half is that the names
which used to reach it do not come back as a bare ``ImportError`` or, worse, as
a shim that quietly routes somewhere else: the replacement is a different
contract, not a drop-in, so anything returning numbers under the old name would
be returning numbers from a different model.

None of this is a deprecation window. There is no flag, no environment
variable, and no fallback path that restores the old behaviour.
"""

from __future__ import annotations

import pathlib

import pytest

import witwin.radar as wr


RADAR_ROOT = pathlib.Path(wr.__file__).resolve().parent
TESTS_ROOT = pathlib.Path(__file__).resolve().parent


def test_the_dr_jit_modules_are_gone_from_the_source_tree():
    for removed in ("trace.py", "material.py", "_rayd_bridge.py"):
        assert not (RADAR_ROOT / removed).exists(), removed
    # `trace_result.py` survived the Dr.Jit removal because it was Torch-only.
    # Phase 11 deleted it anyway: what it held was the payload of an
    # interpolator, and the scene-driven entry has no interpolator.
    assert not (RADAR_ROOT / "trace_result.py").exists()
    assert not (RADAR_ROOT / "path_cache.py").exists()
    assert not (RADAR_ROOT / "solvers").exists()


@pytest.mark.parametrize("name", ["Tracer", "fresnel"])
def test_a_removed_export_names_its_replacement(name):
    with pytest.raises(AttributeError) as raised:
        getattr(wr, name)
    message = str(raised.value)
    assert "removed with the Dr.Jit ray tracer" in message
    assert name not in wr.__all__


def test_an_unknown_attribute_still_gets_an_ordinary_error():
    """The __getattr__ hook must not swallow genuine typos."""

    with pytest.raises(AttributeError, match="has no attribute"):
        wr.definitely_not_a_real_name


def test_simulate_is_the_scene_driven_entry_and_no_longer_a_refusal():
    """Phase 11 work item 1 turned this refusal into the entry point.

    This block used to assert the ``_SIMULATE_REPLACEMENT`` message, which said
    that "a scene-driven entry point that assembles those steps for a whole
    Scene is separate work and does not exist yet". It exists;
    ``tests/test_phase11_simulate_entry.py`` is what it does. What survives here
    is the shape of the surface: ``simulate`` is a bound method taking a scene
    and a declared frame sequence, and its four typed diagnostics answer
    ``None`` on a radar that has not run.
    """

    import inspect

    assert callable(wr.Radar.simulate)
    parameters = inspect.signature(wr.Radar.simulate).parameters
    assert "scene" in parameters
    for name in ("times", "response", "sites"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name

    radar = object.__new__(wr.Radar)
    for name in (
        "last_snapshot",
        "last_compiled_scene",
        "last_propagation",
        "last_radar_paths",
    ):
        assert getattr(radar, name) is None, name


def test_simulate_group_is_gone_rather_than_permanently_refusing():
    """A classmethod that can only ever raise is itself a legacy shim.

    Approved public break (Phase 11, D3). Simulating several radars over one
    world is a loop over ``Radar.simulate``; there was no Radar-owned batching
    behind the old name, so keeping a refusal would have been advertising a
    capability that never existed.
    """

    assert not hasattr(wr.Radar, "simulate_group")
    assert not hasattr(wr.Radar, "_SIMULATE_REPLACEMENT")


def test_the_dirichlet_entry_points_are_gone_and_name_their_replacement():
    """This block used to assert that the SAME six methods were untouched.

    Its argument was that the ``dirichlet_spectrum`` family's callers had
    nothing to do with the tracer, so removing them alongside it would orphan
    six native symbols for no reason. Phase 11 removes them for a reason: the
    scene-driven entry point exists, so the whole route - nine symbols, its
    translation unit, its solver and its path cache - goes at once, and no
    symbol passes through a caller-free state.

    ``Radar`` gets a plain ``AttributeError`` for a deleted method because a
    class attribute has no ``__getattr__`` hook to route through; the package
    root does have one, so the four deleted module-level names answer with a
    message that says where to go instead.
    """

    for name in (
        "mimo",
        "mimo_from_trace",
        "mimo_from_paths",
        "path_cache_from_trace",
        "chirp",
        "frame",
        "waveform",
        "solver",
    ):
        assert not hasattr(wr.Radar, name), name

    for name, expected in (
        ("Solver", "no backend selector"),
        ("TraceResult", "ScatterSitePolicy"),
        ("MimoPathCache", "frozen topology"),
        ("SamplingMode", "interpolator contract"),
        ("MotionSampling", "interpolator contract"),
    ):
        with pytest.raises(AttributeError) as raised:
            getattr(wr, name)
        assert expected in str(raised.value), name
        assert name not in wr.__all__, name


def test_the_torch_dsp_oracles_are_gone_from_the_package_and_from_tests():
    """A CPU reference oracle belongs under tests/ - or nowhere.

    Until Phase 11 this asserted the first half only: the two float64 chirp and
    MIMO references had moved out of ``solvers/common.py`` and into
    ``tests/reference/dsp_oracles.py``. They checked the Dirichlet family and
    nothing else, so they went with it. ``tests/reference/path_math.py``
    survives on purpose - it is the independent oracle for the LIVE
    ``sensor_weight`` family - which is why this asserts the two files
    separately rather than asserting that the reference package is empty.
    """

    assert not (RADAR_ROOT / "solvers").exists()
    assert not (TESTS_ROOT / "reference" / "dsp_oracles.py").exists()
    assert (TESTS_ROOT / "reference" / "path_math.py").exists()


def test_the_packaging_metadata_no_longer_pulls_in_dr_jit():
    text = (RADAR_ROOT.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "drjit" not in text
    assert "rayd-drjit" not in text
