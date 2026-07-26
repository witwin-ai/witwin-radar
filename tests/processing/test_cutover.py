"""The cutover, verified at the moment it happens.

Two claims, both static and both over the SOURCE rather than over a run, because
a run only visits the branch it happened to take:

* the FACADE FENCE - after the cutover no ``torch.fft``, no CFAR, no angle
  estimator and no beamformer expression appears anywhere in ``witwin/radar/``
  outside ``witwin/radar/processing/``;
* DELETION COMPLETENESS - each of the ten cutover items has an assertion that
  the symbol or the code path is gone, or that a flag that used to be implicit
  is now explicit.

Plus one runtime claim the fence cannot make: the whole chain runs end to end.
"""

from __future__ import annotations

import ast
import inspect
import io
import pathlib
import tokenize

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RADAR_ROOT = REPO_ROOT / "witwin" / "radar"
PROCESSING = RADAR_ROOT / "processing"

#: The DSP expressions the processing facade owns. A production occurrence
#: outside it is what the plan criterion forbids.
FENCED_CALLS = (
    "torch.fft.fft",
    "torch.fft.ifft",
    "torch.fft.fft2",
    "torch.fft.fftshift",
    "torch.fft.ifftshift",
    "torch.fft.fftfreq",
    "torch.linalg.eigh",
    "torch.linalg.solve",
)

#: Named allowances, each with a reason. This is a list of SIMULATION owners,
#: not of processing stages that got away.
#:
#: The legacy Dirichlet solver and the spectrum helper it names both invert a
#: SYNTHESIZED spectrum into time samples: that transform is part of producing
#: the received signal, not part of reading it, and it predates the processing
#: chain entirely. ``test_phase6_no_torch_physics`` already records it as the
#: allowlisted DSP exception and asserts it is still CALLED, so it cannot become
#: a scan that passes because the package is empty.
FENCE_ALLOWANCES = {
    "witwin/radar/solvers/solver_dirichlet.py": "legacy waveform synthesis",
}


def _code(source: str) -> str:
    """``source`` with every comment and every string literal removed.

    Half of this file scans for text that must be ABSENT, and the modules under
    scan explain at length what they deleted - naming the very tokens the scan
    forbids. A raw text scan would reject the explanation along with the act,
    which is the same reason ``test_phase6_no_torch_physics`` strips its
    comments before scanning.
    """

    pieces = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        pieces.append(token.string)
    return " ".join(pieces)


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _modules_outside_processing() -> list[pathlib.Path]:
    return sorted(
        path
        for path in RADAR_ROOT.rglob("*.py")
        if PROCESSING not in path.parents and path != PROCESSING
    )


# ---------------------------------------------------------------------------
# The fence
# ---------------------------------------------------------------------------


def test_no_dsp_expression_survives_outside_the_processing_facade():
    """Criterion: no scattered production Torch DSP outside the facade."""

    offenders: list[tuple[str, str]] = []
    for path in _modules_outside_processing():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in FENCE_ALLOWANCES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _dotted(node.func)
                if name in FENCED_CALLS:
                    offenders.append((relative, name))
    assert offenders == [], offenders


def test_the_named_allowance_is_a_simulation_owner_and_still_calls_its_transform():
    """A scan that passes because the code vanished proves nothing."""

    for relative in FENCE_ALLOWANCES:
        source = _code((REPO_ROOT / relative).read_text(encoding="utf-8"))
        assert "torch . fft . ifft" in source, relative
        # And it is an INVERSE transform building a signal, not a forward one
        # reading a spectrum: no forward FFT, no shift, no detector.
        assert "torch . fft . fft (" not in source, relative
        assert "fftshift" not in source, relative


def test_no_detector_angle_estimator_or_beamformer_lives_outside_the_facade():
    """By NAME, because these would arrive as a function definition."""

    forbidden = {
        "ca_cfar",
        "ca_cfar_2d",
        "ca_cfar_fast",
        "os_cfar",
        "os_cfar_2d",
        "music_spectrum",
        "naive_xyz",
        "phase_comparison_aoa",
        "fft2_aoa",
        "conventional_steering",
        "mvdr_weights",
        "beam_cube",
        "point_cloud",
        "tdm_compensate",
    }
    offenders: list[tuple[str, str]] = []
    for path in _modules_outside_processing():
        relative = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in forbidden:
                    offenders.append((relative, node.name))
    assert offenders == [], offenders


def test_the_whole_sigproc_package_is_re_export_only():
    """Not one expression: every legacy name resolves into the facade.

    The adapters live inside ``witwin/radar/processing/`` on purpose. That is
    what lets the fence above be a statement about a DIRECTORY rather than a
    list of exceptions.
    """

    for path in sorted((RADAR_ROOT / "sigproc").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            allowed = (
                ast.Import,
                ast.ImportFrom,
                ast.Assign,
                ast.Expr,
                ast.AnnAssign,
            )
            assert isinstance(node, allowed), (path.name, type(node).__name__)
        for node in ast.walk(tree):
            assert not isinstance(node, ast.FunctionDef), path.name
            assert not isinstance(node, ast.ClassDef), path.name


# ---------------------------------------------------------------------------
# Deletion completeness, item by item
# ---------------------------------------------------------------------------


def test_item_1_frame_config_no_longer_reads_raw_configuration_fields():
    from witwin.radar.processing.adapters import FrameConfig

    source = _code(inspect.getsource(FrameConfig))
    assert "radar.config" not in source
    assert "cfg." not in source
    # It publishes the two new records instead, which is the migration.
    assert "axes_from_radar" in source


def test_item_2_there_is_one_legacy_transform_owner_and_not_three():
    from witwin.radar.processing import adapters

    source = _code((PROCESSING / "adapters.py").read_text(encoding="utf-8"))
    assert source.count("torch . fft . fft (") == 2, source
    assert hasattr(adapters, "legacy_range_transform")
    assert hasattr(adapters, "legacy_doppler_transform")
    # Both legacy entry points and the MUSIC image go through them.
    for name in ("range_fft", "process_rd_tensor", "radar_image"):
        assert name in source


def test_item_3_frame_reshape_is_gone():
    import witwin.radar.sigproc.pointcloud as legacy

    assert not hasattr(legacy, "frame_reshape")
    with pytest.raises(ImportError):
        from witwin.radar.sigproc.pointcloud import frame_reshape  # noqa: F401


def test_item_4_the_second_point_cloud_pipeline_is_gone():
    import witwin.radar.processing.adapters as adapters
    import witwin.radar.sigproc.pointcloud as legacy

    assert not hasattr(legacy, "_process_pc_cfar_tensor")
    assert not hasattr(adapters, "_process_pc_cfar_tensor")
    # One body, and the detector is an argument to it.
    signature = inspect.signature(adapters._legacy_point_cloud)
    assert "detector" in signature.parameters


def test_item_5_reg_data_has_no_numpy_random_path():
    from witwin.radar.processing import adapters

    source = _code(inspect.getsource(adapters.reg_data))
    assert "np . random" not in source
    assert "np . zeros" not in source
    # It still RETURNS a host array, which is the legacy contract, but the
    # sampling is the detection contract's helper in torch.
    assert "as_fixed_size" in source


def test_item_6_the_magic_range_gate_bins_are_gone_from_the_source():
    # By AST, not by text: the constants survive as a NAMED pair that is
    # converted into metres, and what must be gone is the literal SLICE.
    tree = ast.parse((PROCESSING / "adapters.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Slice):
            continue
        for bound in (node.lower, node.upper):
            if isinstance(bound, ast.Constant):
                assert bound.value not in (25, 125), ast.dump(node)
    from witwin.radar.processing.adapters import LEGACY_RANGE_CUT_BINS

    # They survive only as a NAMED pair converted into metres, which is what
    # makes the gate a statement about the scene rather than about a bin count.
    assert LEGACY_RANGE_CUT_BINS == (25, 125)
    from witwin.radar.processing.adapters import _legacy_range_gate_db

    assert "range_bin_m" in _code(inspect.getsource(_legacy_range_gate_db))


def test_item_7_the_hard_coded_half_wavelength_spacing_is_gone():
    from witwin.radar.processing import aoa

    source = _code((PROCESSING / "aoa.py").read_text(encoding="utf-8"))
    assert "0.5" not in source
    assert "array . spacing_wavelengths" in source
    assert "spacing_wavelengths" in _code(inspect.getsource(aoa.upa_steering))


def test_item_8_no_numpy_survives_in_the_processing_facade():
    for path in sorted(PROCESSING.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("numpy"), path.name
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("numpy"), path.name
            if isinstance(node, ast.Attribute) and _dotted(node).startswith("np."):
                assert False, (path.name, _dotted(node))


def test_item_9_the_tdm_compensation_has_no_python_transmitter_loop():
    from witwin.radar.processing import tdm_compensate

    tree = ast.parse(inspect.getsource(tdm_compensate))
    for node in ast.walk(tree):
        assert not isinstance(node, ast.For), "a Python loop over transmitters"
        assert not isinstance(node, ast.While)
    source = _code(inspect.getsource(tdm_compensate))
    assert "clone" not in source
    assert "*=" not in source


def test_item_10_the_matched_filter_precision_is_an_explicit_argument():
    from witwin.radar.processing.matched_filter import matched_filter

    signature = inspect.signature(matched_filter)
    assert "dtype" in signature.parameters
    assert signature.parameters["dtype"].default is None
    source = _code(inspect.getsource(matched_filter))
    assert "torch . complex128" not in source


# ---------------------------------------------------------------------------
# The chain, end to end
# ---------------------------------------------------------------------------


def test_the_whole_chain_runs_from_a_synthesis_layout_to_a_detection_frame():
    """synthesize -> cube -> range profile -> RD -> beam cube -> CFAR -> AoA ->
    point cloud -> handoff, in one call sequence with no legacy name in it."""

    import math

    from conftest import MockRadar
    from support.legacy_golden import GOLDEN_CONFIG
    from witwin.radar.processing import (
        ArrayGeometry,
        DetectionFrame,
        ProcessingCube,
        TrackHandoff,
        beam_cube,
        ca_cfar_fast,
        conventional_steering,
        point_cloud,
        range_doppler,
        range_profile,
    )
    from witwin.radar.processing.adapters import axes_from_radar
    from witwin.radar.synthesis.assembly import assemble_frame_cube

    config = {
        **GOLDEN_CONFIG,
        "tx_loc": [[0, 0, 0], [4, 0, 0], [0, 0, 1]],
        "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
    }
    axes = axes_from_radar(MockRadar(config))
    array = ArrayGeometry.from_axes(axes)

    generator = torch.Generator().manual_seed(2026)
    rank3 = torch.complex(
        torch.randn(
            (axes.doppler_bin_count, array.sensor_pair_count, axes.range_bin_count),
            generator=generator,
        ),
        torch.randn(
            (axes.doppler_bin_count, array.sensor_pair_count, axes.range_bin_count),
            generator=generator,
        ),
    ).to(torch.complex64)

    cube = ProcessingCube(
        data=assemble_frame_cube(rank3, num_tx=axes.num_tx, num_rx=axes.num_rx),
        axes=axes,
    )
    profile = range_profile(cube, window="hann")
    rd = range_doppler(profile, window="hann")

    directions = torch.tensor(
        [
            [math.sin(angle), math.cos(angle), 0.0]
            for angle in (-0.4, -0.2, 0.0, 0.2, 0.4)
        ],
        dtype=torch.float64,
    )
    weights = conventional_steering(array, directions)
    beams = beam_cube(rd, weights, directions=directions)
    assert tuple(beams.data.shape) == (
        5,
        axes.doppler_bin_count,
        axes.range_bin_count,
    )

    # The batched detector runs on the whole beam cube at once, which is the
    # capability that did not exist: three rank-2 detectors and a loop.
    detected = ca_cfar_fast(
        beams.data.abs(), guard_cells=(1, 2), training_cells=(2, 3), pfa=1e-2
    )
    assert tuple(detected.mask.shape) == tuple(beams.data.shape)

    combined = rd.data.reshape(array.sensor_pair_count, *rd.data.shape[-2:]).sum(dim=0)
    cells = ca_cfar_fast(
        combined.abs(), guard_cells=(1, 2), training_cells=(2, 3), pfa=1e-2
    )
    cloud = point_cloud(cells, rd, axes, array, max_points=8)
    handoff = TrackHandoff()
    assignment = handoff.push(
        DetectionFrame.from_point_cloud(cloud, time_s=0.0, frame_index=0)
    )
    assert int(assignment.shape[0]) == len(cloud)
    assert handoff.track_count == len(cloud)
