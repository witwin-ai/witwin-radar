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


def test_the_dr_jit_modules_are_gone_from_the_source_tree():
    for removed in ("trace.py", "material.py", "_rayd_bridge.py"):
        assert not (RADAR_ROOT / removed).exists(), removed
    # trace_result.py is Torch-only and survives; TraceResult is still exported.
    assert (RADAR_ROOT / "trace_result.py").exists()
    assert wr.TraceResult is not None


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


def test_the_surviving_solver_entry_points_are_untouched():
    """The Dirichlet family keeps every production caller it had.

    Six of the nine manifested native symbols are the dirichlet_spectrum
    family, and every one of their end-to-end callers is a Radar method that
    never constructed a Tracer. Removing them alongside the tracer would have
    orphaned those symbols and forced the manifest and the CUDA sources to
    change in the same commit, for no reason.
    """

    for name in (
        "mimo",
        "mimo_from_trace",
        "mimo_from_paths",
        "path_cache_from_trace",
        "chirp",
        "frame",
    ):
        assert callable(getattr(wr.Radar, name)), name


def test_the_torch_dsp_oracles_no_longer_ship_inside_the_package():
    """A CPU reference oracle belongs under tests/, not in the wheel."""

    from witwin.radar.solvers import common

    for name in ("pytorch_chirp_reference", "pytorch_mimo_from_samples"):
        assert not hasattr(common, name), name

    from reference import dsp_oracles

    assert callable(dsp_oracles.pytorch_chirp_reference)
    assert callable(dsp_oracles.pytorch_mimo_from_samples)


def test_the_residual_torch_path_surface_is_frozen():
    """solvers/common.py is a recorded deviation, so it must not grow.

    Phase 6 work item 8 SHRANK this set. When this test was written its
    docstring said the six geometry and amplitude helpers were not removed
    because "the native evaluator that replaces them is out of scope"; that
    evaluator is the ``sensor_weight`` family and it now exists, so
    ``compute_total_path_lengths``, ``compute_antenna_pattern_gains``,
    ``compute_polarization_amplitudes``, ``compute_path_amplitudes``, and
    ``compute_slot_path_tensors`` are gone, and ``collect_interpolated_samples``
    moved to its single caller.

    The set below is what is LEFT, and none of it is physics: a contract, dtype
    and device glue, a predicate, and structural packing. The assertion is an
    equality rather than a subset for the same reason it always was - a subset
    check passes when something is added back.
    """

    from witwin.radar.solvers import common

    # Defined here, not merely reachable: `dir` also returns imported names.
    public = {
        name
        for name, value in vars(common).items()
        if not name.startswith("_")
        and callable(value)
        and getattr(value, "__module__", None) == common.__name__
    }
    assert public == {
        "PathSample",
        "normalize_interpolated_sample",
        "samples_require_grad",
    }, sorted(public)
    # `_stack_slot_samples` is private and therefore outside the set above, but
    # it is the fourth survivor and it must still be here: it is the padding
    # step the slot route hands to the native owner.
    assert callable(common._stack_slot_samples)


def test_the_migrated_helpers_are_gone_from_the_shared_module():
    """Named one by one, because "the set shrank" is not the same statement.

    A helper that came back under a new name would keep the set above at three
    entries only by accident. These five are the Torch expressions plan work
    item 8 migrated, and they must not exist here under any spelling.
    """

    from witwin.radar.solvers import common

    for name in (
        "compute_total_path_lengths",
        "compute_antenna_pattern_gains",
        "compute_polarization_amplitudes",
        "compute_path_amplitudes",
        "compute_slot_path_tensors",
        "_slot_polarization_factors",
        "_normalize_vectors",
        "collect_interpolated_samples",
    ):
        assert not hasattr(common, name), name


def test_the_packaging_metadata_no_longer_pulls_in_dr_jit():
    text = (RADAR_ROOT.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "drjit" not in text
    assert "rayd-drjit" not in text
