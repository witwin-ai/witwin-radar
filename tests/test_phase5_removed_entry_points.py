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


def test_simulate_and_simulate_group_raise_with_the_replacement_route():
    radar = object.__new__(wr.Radar)
    for call in (
        lambda: wr.Radar.simulate(radar, object()),
        lambda: wr.Radar.simulate_group(object(), radars=[]),
    ):
        with pytest.raises(NotImplementedError) as raised:
            call()
        message = str(raised.value)
        assert "ChannelPropagationAdapter" in message
        assert "TwoWayComposer" in message
        assert "synthesize_fmcw_beat" in message
        # And it says what still works, so the error is actionable rather than
        # merely a refusal.
        assert "mimo_from_paths" in message


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

    Its remaining helpers are per-path geometry and amplitude math in Torch,
    consumed by the Dirichlet backend. They are NOT removed in this change: the
    native evaluator that replaces them is out of scope, and deleting the only
    production owner while its replacement does not exist would orphan six
    manifested native symbols rather than reduce scope. Freezing the surface is
    what stops "recorded deviation" from turning into "growing exception".
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
        "collect_interpolated_samples",
        "compute_antenna_pattern_gains",
        "compute_path_amplitudes",
        "compute_polarization_amplitudes",
        "compute_slot_path_tensors",
        "compute_total_path_lengths",
        "normalize_interpolated_sample",
        "samples_require_grad",
    }, sorted(public)


def test_the_packaging_metadata_no_longer_pulls_in_dr_jit():
    text = (RADAR_ROOT.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "drjit" not in text
    assert "rayd-drjit" not in text
