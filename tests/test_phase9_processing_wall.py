"""The non-differentiability wall, pinned from BOTH sides.

Phase-9 item 4 names ADC rounding, CFAR, peak selection and tracking as
explicitly non-differentiable and requires them to fail BEFORE producing a
partial result. Before this file only the ADC did. Everything else on the far
side of the wall published a live derivative and said nothing, which is the
exact defect this phase exists to remove: not a wrong number, a derivative that
describes a frozen discrete choice and arrives looking like an answer.

The numbers, all measured on the fixture below before the guards were added:

* ``ca_cfar_fast``: ``d(threshold)/d(power)`` summed to **1.51e4**, and under a
  forward dual the threshold tangent was live with the same value.
* ``point_cloud``: ``cloud.xyz`` and ``cloud.energy`` both came back with
  ``requires_grad=True`` and ``d(energy)/d(cube)`` had abs-sum **58.36**.
* ``phase_comparison_aoa``: the published direction cosines carried a
  ``StackBackward0``.
* ``DetectionFrame``: refused ``requires_grad`` and accepted a forward dual.

**Where the wall is.** At the first DISCRETE DECISION, not at "post-processing".
That distinction is the content of this file: a matched filter, a range
transform, a Doppler transform, a beam cube and a MUSIC pseudo-spectrum are all
smooth functions of their input, and pinning them LIVE is what stops the wall
from creeping upwards into differentiable DSP the next time someone reads
"post-processing is not differentiable" and reaches for a detach.

**"No partial result" is measured, not asserted by inspection.** Every refusal
test runs with the eight device operations these stages are built from - the
transforms, the pooling, the cumulative sums, the sorts, the argmaxes and the
argwhere - replaced by counting stand-ins, and asserts the count is exactly
zero. A stage that raised after computing its threshold would fail that even
though it raised.

The guards live only in the canonical processing owners, with no secondary surface.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.autograd.forward_ad as forward_ad
import torch.nn.functional as F
from conftest import PROCESSING_CONFIG, make_processing_axes

from witwin.radar.processing import (
    ArrayGeometry,
    ProcessingCube,
    beam_cube,
    ca_cfar,
    ca_cfar_fast,
    conventional_steering,
    os_cfar,
    point_cloud,
    range_doppler_map,
    range_profile,
)
from witwin.radar.processing.angle import fft2_aoa, music_spectrum, phase_comparison_aoa, tdm_compensate
from witwin.radar.processing.detection import Detections, _keep_strongest, ca_cfar_1d
from witwin.radar.processing.tracking import DetectionFrame

#: Three transmitters, four receivers: enough for the phase-comparison route,
#: and the same array the point-cloud tests are written against so a failure
#: here and a failure there describe one geometry.
CONFIG = {
    **PROCESSING_CONFIG,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [0, 0, 1]],
    "rx_loc": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
}

RANGE_BIN = 19
DOPPLER_BIN = 3
AZIMUTH_COSINE = 0.25
COEFFICIENT = complex(0.7, -0.4)

#: The detector settings the point-cloud fixture uses. A tight ring and a very
#: small design false-alarm rate, so the single injected target is the only
#: detection and the point cloud has exactly one row.
DETECTOR = {"guard_cells": (1, 2), "training_cells": (3, 4), "pfa": 1e-6}


# ---------------------------------------------------------------------------
# Fixture: one target on three exact bins
# ---------------------------------------------------------------------------


def _records():
    axes = make_processing_axes(CONFIG)
    return axes, ArrayGeometry.from_axes(axes)


def _cube(axes, array) -> torch.Tensor:
    """``[TX, RX, C, S]``: one target on three exact bins plus a noise floor.

    The same construction ``tests/processing/test_pointcloud.py`` uses, kept
    here rather than imported because a wall test that depends on another test
    module's private helper is a wall test that moves when that helper does.
    """

    samples = int(axes.range_bin_count)
    chirps = int(axes.doppler_bin_count)
    direction = torch.tensor([[AZIMUTH_COSINE, math.sqrt(1.0 - AZIMUTH_COSINE**2), 0.0]], dtype=torch.float64)
    manifold = conventional_steering(array, direction, normalize=False, dtype=torch.complex64).reshape(
        array.num_tx, array.num_rx, 1, 1
    )
    chirp_period_s = axes.slow_time_period_s / axes.num_tx
    closing = DOPPLER_BIN * axes.velocity_bin_mps
    slot_phase = (
        -array.phase_sign
        * 4.0
        * math.pi
        * closing
        * torch.arange(array.num_tx, dtype=torch.float64)
        * chirp_period_s
        / array.wavelength_m
    )
    slot = torch.polar(torch.ones_like(slot_phase), slot_phase).to(torch.complex64).reshape(array.num_tx, 1, 1, 1)
    fast = torch.arange(samples, dtype=torch.float64)
    slow = torch.arange(chirps, dtype=torch.float64)
    tone = torch.polar(torch.ones(samples, dtype=torch.float64), 2.0 * math.pi * RANGE_BIN * fast / samples).to(
        torch.complex64
    )
    walk = torch.polar(torch.ones(chirps, dtype=torch.float64), -2.0 * math.pi * DOPPLER_BIN * slow / chirps).to(
        torch.complex64
    )
    generator = torch.Generator().manual_seed(606)
    floor = (
        torch.complex(
            torch.randn((array.num_tx, array.num_rx, chirps, samples), generator=generator),
            torch.randn((array.num_tx, array.num_rx, chirps, samples), generator=generator),
        ).to(torch.complex64)
        * 1e-3
    )
    signal = manifold * slot * walk.reshape(1, 1, -1, 1) * tone.reshape(1, 1, 1, -1)
    return signal * COEFFICIENT + floor


class _Scene:
    """Axes, array, cube, and everything the transforms produce from them.

    The transformed products are built ONCE, in the fixture, and never inside a
    test body. That is not tidiness: the "no partial result" instrument below
    counts every transform in the process, so a test that built its own power
    map would count its own setup and could never assert zero.
    """

    __slots__ = ("axes", "array", "cube", "rd", "power", "detected")

    def __init__(self, axes, array, cube) -> None:
        self.axes = axes
        self.array = array
        self.cube = cube
        self.rd = range_doppler_map(range_profile(ProcessingCube(cube, axes)))
        combined = self.rd.data.reshape(array.sensor_pair_count, *self.rd.data.shape[-2:]).sum(dim=0)
        self.power = combined.abs().to(torch.float64)
        self.detected = ca_cfar_fast(self.power, **DETECTOR)


@pytest.fixture(scope="module")
def scene():
    axes, array = _records()
    return _Scene(axes, array, _cube(axes, array))


def _remap(rd, data):
    """The same Range-Doppler record over a different data tensor."""

    return type(rd)(data=data, axes=rd.axes, window=rd.window, window_coherent_gain=rd.window_coherent_gain)


# ---------------------------------------------------------------------------
# The "no partial result" instrument
# ---------------------------------------------------------------------------


#: Every device operation the guarded stages are built from. If a refusal fires
#: at the entry, not one of these runs. Named as data so a stage that grows a
#: new primitive is a deliberate addition here rather than a silent hole.
_INSTRUMENTED = (
    (torch.fft, "fft"),
    (torch.fft, "fft2"),
    (F, "avg_pool2d"),
    (F, "unfold"),
    (torch, "sort"),
    (torch, "topk"),
    (torch, "argmax"),
    (torch, "argwhere"),
)


class _ComputeWatch:
    """Counts calls to the primitives a stage would make if it got past entry."""

    def __init__(self, monkeypatch) -> None:
        self.calls: list[str] = []
        for owner, name in _INSTRUMENTED:
            original = getattr(owner, name)
            monkeypatch.setattr(owner, name, self._wrap(name, original))
        original_cumsum = torch.Tensor.cumsum
        monkeypatch.setattr(torch.Tensor, "cumsum", self._wrap("cumsum", original_cumsum))

    def _wrap(self, name, original):
        def wrapped(*args, **kwargs):
            self.calls.append(name)
            return original(*args, **kwargs)

        return wrapped


@pytest.fixture
def watch(monkeypatch):
    return _ComputeWatch(monkeypatch)


def _dual_of(tensor: torch.Tensor) -> torch.Tensor:
    return forward_ad.make_dual(tensor.detach(), torch.ones_like(tensor.detach()))


def test_the_no_partial_result_instrument_is_not_vacuous(scene, watch):
    """The counters fire when a stage DOES run.

    Every refusal test below asserts an empty call list. That assertion is only
    worth anything if the list would have been non-empty, so the instrument is
    calibrated once against the same detector and the same point-cloud stage
    running normally.
    """

    detections = ca_cfar_fast(scene.power, **DETECTOR)
    assert "avg_pool2d" in watch.calls
    point_cloud(detections, scene.rd, scene.axes, scene.array, max_points=64)
    assert "argwhere" in watch.calls
    assert "topk" in watch.calls


# ---------------------------------------------------------------------------
# 1. ABOVE the wall: the declared-differentiable stages stay live
# ---------------------------------------------------------------------------


def test_the_range_and_doppler_transforms_stay_differentiable(scene):
    """Linear DSP is on the SIMULATION side of the wall and must stay there.

    The wall is at the first discrete decision, so a transform - which is a
    matrix multiply by another name - keeps its gradient. This is half the
    reason the guards below are placed per function rather than per module: a
    reader who takes "post-processing is not differentiable" literally would
    detach here and lose a derivative that is exact.
    """

    live = scene.cube.detach().clone().requires_grad_(True)
    profile = range_profile(ProcessingCube(live, scene.axes))
    assert profile.data.requires_grad
    rd = range_doppler_map(profile)
    assert rd.data.requires_grad

    (grad,) = torch.autograd.grad(rd.data.abs().square().sum(), live)
    assert torch.isfinite(grad).all()
    # Not vacuous: a severed graph would give exactly zero, and a Parseval
    # argument says this gradient is large.
    assert float(grad.abs().sum()) > 1.0


def test_the_beam_cube_stays_differentiable(scene):
    live = scene.cube.detach().clone().requires_grad_(True)
    rd = range_doppler_map(range_profile(ProcessingCube(live, scene.axes)))
    directions = torch.tensor([[0.0, 1.0, 0.0], [0.25, math.sqrt(1 - 0.0625), 0.0]], dtype=torch.float64)
    steering = conventional_steering(scene.array, directions, dtype=torch.complex64)
    cubes = beam_cube(rd, steering, directions=directions)
    assert cubes.data.requires_grad
    (grad,) = torch.autograd.grad(cubes.data.abs().square().sum(), live)
    assert float(grad.abs().sum()) > 1.0


def test_the_matched_filter_stays_differentiable():
    """A correlation is a product of transforms; it has an exact derivative."""

    from witwin.radar.processing.range_doppler import matched_filter
    from witwin.radar.synthesis.assembly import PulsedSpec

    spec = PulsedSpec(
        num_pulses=1,
        num_samples=64,
        sample_period_s=2.0e-9,
        pri_s=1.0e-6,
        range_gate_start_s=0.0,
        pulse_kind="lfm",
        pulse_width_s=2.0e-8,
        bandwidth_hz=5.0e8,
        reference_frequency_hz=77.0e9,
        max_expected_delay_rate=0.0,
        carrier_hz=0.0,
        carrier_rate_hz=77.0e9,
    )
    generator = torch.Generator().manual_seed(9)
    signal = (
        torch.complex(torch.randn(64, generator=generator), torch.randn(64, generator=generator))
        .to(torch.complex64)
        .requires_grad_(True)
    )
    compressed = matched_filter(signal, spec)
    assert compressed.requires_grad
    (grad,) = torch.autograd.grad(compressed.abs().square().sum(), signal)
    assert float(grad.abs().sum()) > 0.0


def test_the_tdm_compensation_stays_differentiable(scene):
    """One broadcast multiply. It is above the wall and it is pinned there."""

    generator = torch.Generator().manual_seed(17)
    rows = scene.array.sensor_pair_count
    virtual = (
        torch.complex(torch.randn((rows, 2), generator=generator), torch.randn((rows, 2), generator=generator))
        .to(torch.complex64)
        .requires_grad_(True)
    )
    velocity = torch.tensor([1.5, -2.0], dtype=torch.float64)
    out = tdm_compensate(virtual, velocity, scene.array, scene.axes)
    assert out.requires_grad
    (grad,) = torch.autograd.grad(out.abs().square().sum(), virtual)
    assert float(grad.abs().sum()) > 0.0


def _music_inputs():
    generator = torch.Generator().manual_seed(11)
    data = torch.complex(
        torch.randn((1, 4, 4, 8), generator=generator), torch.randn((1, 4, 4, 8), generator=generator)
    ).to(torch.complex64)
    elevation = torch.linspace(-0.4, 0.4, 5, dtype=torch.float32)
    azimuth = torch.linspace(-0.4, 0.4, 7, dtype=torch.float32)
    return data, elevation, azimuth


def test_the_music_pseudo_spectrum_is_differentiable_and_matches_a_difference(scene):
    """The MUSIC SPECTRUM is above the wall, and the reason is MEASURED.

    Its ``topk`` orders EIGENVALUES to split the signal subspace from the noise
    subspace. That is a permutation, not a peak pick: away from an eigenvalue
    crossing the published spectrum is a smooth function of the covariance, and
    a central difference agrees with autograd to 0.2 percent in a float32
    pipeline.

    This is the boundary that stops the wall from being drawn per module.
    ``music_image`` calls this function and is equally live; the peak pick a
    MUSIC image is usually followed by is the CALLER's, which is exactly why
    ``music_image`` refuses to auto-detect its range bins.
    """

    data, elevation, azimuth = _music_inputs()

    def loss(values: torch.Tensor) -> torch.Tensor:
        spectrum = music_spectrum(
            values, scene.array, elevation_rad=elevation, azimuth_rad=azimuth, num_signals=2, spatial_smooth=1
        )
        return spectrum.abs().sum()

    live = data.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(loss(live), live)
    assert grad is not None
    assert float(grad.abs().sum()) > 1.0

    index = (0, 1, 2, 3)
    step = 1.0e-2
    plus, minus = data.clone(), data.clone()
    plus[index] = plus[index] + complex(step, 0.0)
    minus[index] = minus[index] - complex(step, 0.0)
    measured = (float(loss(plus)) - float(loss(minus))) / (2.0 * step)
    analytic = float(grad[index].real)
    assert abs(analytic) > 1.0e-3
    # 3 percent, against a 0.2 percent measured agreement: the tolerance is a
    # float32 finite-difference allowance, not a claim about the derivative.
    assert abs(measured - analytic) <= 0.03 * abs(analytic)


def test_music_image_carries_the_same_live_derivative_as_the_spectrum(scene):
    """The image is an ``index_select`` and a ``permute`` around the spectrum.

    Recorded here because the phase design listed ``music_image`` as a peak-pick
    site to guard and it is not one: it contains no ``argmax`` and no ``topk``,
    and it REQUIRES the caller to supply ``range_bins`` precisely so the peak
    pick stays outside it. Guarding it would have refused a derivative that is
    defined, correct and measured one test above.
    """

    from witwin.radar.processing.angle import music_image
    from witwin.radar.processing.range_doppler import RangeProfile

    generator = torch.Generator().manual_seed(23)
    bins = int(scene.axes.range_bin_count)
    data = (
        torch.complex(
            torch.randn((4, 4, 8, bins), generator=generator), torch.randn((4, 4, 8, bins), generator=generator)
        )
        .to(torch.complex64)
        .requires_grad_(True)
    )
    profile = RangeProfile(data=data, axes=scene.axes, window="rectangular", window_coherent_gain=1.0)
    image = music_image(
        profile,
        scene.array,
        elevation_rad=torch.linspace(-0.3, 0.3, 3, dtype=torch.float32),
        azimuth_rad=torch.linspace(-0.3, 0.3, 3, dtype=torch.float32),
        range_bins=torch.tensor([1], dtype=torch.int64),
        num_signals=2,
        spatial_smooth=1,
        num_snapshots=8,
    )
    assert image.requires_grad
    (grad,) = torch.autograd.grad(image.abs().sum(), data)
    assert float(grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# 2. BELOW the wall: CFAR
# ---------------------------------------------------------------------------


CFAR_ENTRIES = {"ca_cfar": ca_cfar, "ca_cfar_fast": ca_cfar_fast, "os_cfar": os_cfar}


@pytest.mark.parametrize("entry", sorted(CFAR_ENTRIES))
def test_a_cfar_detector_refuses_a_gradient_before_any_compute(scene, watch, entry):
    """Refused at the entry, with the owner named and nothing computed.

    Before Phase 9 there was no refusal at all here and the derivative was
    real: ``d(threshold)/d(power)`` summed to 1.51e4 on this fixture. The
    detector gives that derivative up deliberately - its OUTPUT is a detection
    decision, and the mask beside the threshold carries no derivative at all.
    """

    live = scene.power.clone().requires_grad_(True)
    with pytest.raises(RuntimeError, match=f"witwin.radar.processing.detection.{entry} is not differentiable"):
        CFAR_ENTRIES[entry](live, **DETECTOR)
    assert watch.calls == []


@pytest.mark.parametrize("entry", sorted(CFAR_ENTRIES))
def test_a_cfar_detector_refuses_a_forward_dual_before_any_compute(scene, watch, entry):
    """The half a ``requires_grad`` check misses, and it was live too."""

    with forward_ad.dual_level():
        with pytest.raises(RuntimeError, match="a forward tangent"):
            CFAR_ENTRIES[entry](_dual_of(scene.power), **DETECTOR)
    assert watch.calls == []


def test_the_range_only_detector_refuses_both_modes_before_any_compute(scene, watch):
    profile = scene.power[0]
    with pytest.raises(RuntimeError, match="witwin.radar.processing.detection.ca_cfar_1d is not differentiable"):
        ca_cfar_1d(profile.clone().requires_grad_(True))
    with forward_ad.dual_level():
        with pytest.raises(RuntimeError, match="a forward tangent"):
            ca_cfar_1d(_dual_of(profile))
    assert watch.calls == []


def test_the_detectors_still_detect_and_the_guard_changed_no_value(scene):
    """Over-refusing is the opposite mistake and is just as easy to make.

    The guard raises or it does nothing: a detached clone of a grad-carrying map
    gives bitwise the same detection as the plain map. The broader
    no-regression evidence is the bitwise legacy goldens in
    ``tests/processing/test_cutover.py``, which still pass.
    """

    detached = ca_cfar_fast(scene.power.clone().requires_grad_(True).detach(), **DETECTOR)
    assert torch.equal(scene.detected.mask, detached.mask)
    assert torch.equal(scene.detected.threshold, detached.threshold)
    assert int(scene.detected.mask.sum()) > 0


# ---------------------------------------------------------------------------
# 3. BELOW the wall: the point cloud and its topk
# ---------------------------------------------------------------------------


def test_the_point_cloud_refuses_a_gradient_before_any_compute(scene, watch):
    """No ``argwhere``, no ``topk``, no ``PointCloud``.

    Before Phase 9 this stage published ``cloud.xyz`` and ``cloud.energy`` with
    ``requires_grad=True`` and ``d(energy)/d(cube)`` abs-sum 58.36 - the
    derivative of a value at a frozen ``argmax``, which predicts nothing about
    the point list once the selection moves, including its length.
    """

    live = _remap(scene.rd, scene.rd.data.detach().clone().requires_grad_(True))
    with pytest.raises(RuntimeError, match="witwin.radar.processing.detection.point_cloud is not differentiable"):
        point_cloud(scene.detected, live, scene.axes, scene.array, max_points=64)
    assert watch.calls == []


def test_the_point_cloud_refuses_a_forward_dual_before_any_compute(scene, watch):
    with forward_ad.dual_level():
        dual = _remap(scene.rd, _dual_of(scene.rd.data))
        with pytest.raises(RuntimeError, match="a forward tangent"):
            point_cloud(scene.detected, dual, scene.axes, scene.array, max_points=64)
    assert watch.calls == []


def test_the_point_cloud_refuses_a_live_detection_threshold(scene, watch):
    """The other door: a hand-built ``Detections`` whose threshold carries a tape.

    The detectors refuse at their own entry, so this shape can only be reached
    by constructing the record directly - which is exactly why the point-cloud
    stage checks the threshold as well as the map.
    """

    live = Detections(
        mask=scene.detected.mask, threshold=scene.detected.threshold.detach().clone().requires_grad_(True)
    )
    with pytest.raises(RuntimeError, match="detection_threshold carries a gradient"):
        point_cloud(live, scene.rd, scene.axes, scene.array, max_points=64)
    assert watch.calls == []


def test_the_peak_selection_refuses_a_gradient_before_any_topk(scene, watch):
    """``_keep_strongest`` is guarded even though its only caller already is."""

    energy = torch.zeros_like(scene.detected.threshold).requires_grad_(True)
    with pytest.raises(RuntimeError, match="_keep_strongest is not differentiable"):
        _keep_strongest(scene.detected.mask, energy, 8)
    assert watch.calls == []


def test_the_point_cloud_still_produces_its_one_point(scene):
    cloud = point_cloud(scene.detected, scene.rd, scene.axes, scene.array, max_points=64)
    assert len(cloud) == 1
    assert not cloud.xyz.requires_grad
    assert not cloud.energy.requires_grad


# ---------------------------------------------------------------------------
# 4. BELOW the wall: the argmax angle estimators
# ---------------------------------------------------------------------------


AOA_ENTRIES = {"phase_comparison_aoa": phase_comparison_aoa, "fft2_aoa": fft2_aoa}


@pytest.mark.parametrize("entry", sorted(AOA_ENTRIES))
def test_an_argmax_angle_estimator_refuses_both_modes_before_any_compute(scene, watch, entry):
    """The bin index is discrete; the cosine derived from it is a staircase."""

    generator = torch.Generator().manual_seed(41)
    rows = scene.array.sensor_pair_count
    virtual = torch.complex(
        torch.randn((rows, 2), generator=generator), torch.randn((rows, 2), generator=generator)
    ).to(torch.complex64)
    estimator = AOA_ENTRIES[entry]

    with pytest.raises(RuntimeError, match=f"witwin.radar.processing.angle.{entry} is not differentiable"):
        estimator(virtual.clone().requires_grad_(True), scene.array, fft_size=64)
    with forward_ad.dual_level():
        with pytest.raises(RuntimeError, match="a forward tangent"):
            estimator(_dual_of(virtual), scene.array, fft_size=64)
    assert watch.calls == []


def test_the_fft2_route_still_enforces_its_own_contract(scene):
    """The refusal did not displace the estimator's own checks.

    ``fft2_aoa`` needs four transmitter rows and this array has three, so the
    guard runs BEFORE that check without swallowing it: a detached input still
    reaches the original ``ValueError``.
    """

    virtual = torch.zeros((scene.array.sensor_pair_count, 1), dtype=torch.complex64)
    with pytest.raises(ValueError, match="at least four transmitter rows"):
        fft2_aoa(virtual, scene.array, fft_size=64)


# ---------------------------------------------------------------------------
# 5. BELOW the wall: tracking, the second door
# ---------------------------------------------------------------------------


def test_the_detection_frame_refuses_both_modes_and_shares_one_wording():
    """Unreachable in the normal flow now, and kept anyway.

    ``point_cloud`` refuses at its entry, so a ``DetectionFrame`` built from a
    real cloud can no longer carry a derivative. This guard is the second door,
    and Phase 9 gave it the forward-dual half it never had. Its own module test
    owns the detail; what is asserted here is that it speaks with the SAME
    voice as the rest of the wall.
    """

    xyz = torch.zeros((1, 3), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="is not differentiable"):
        DetectionFrame(
            time_s=0.0,
            xyz=xyz.clone().requires_grad_(True),
            velocity_mps=torch.zeros(1, dtype=torch.float64),
            energy=torch.zeros(1, dtype=torch.float64),
            frame_index=0,
        )
    with forward_ad.dual_level():
        with pytest.raises(RuntimeError, match="a forward tangent"):
            DetectionFrame(
                time_s=0.0,
                xyz=_dual_of(xyz),
                velocity_mps=torch.zeros(1, dtype=torch.float64),
                energy=torch.zeros(1, dtype=torch.float64),
                frame_index=0,
            )


# ---------------------------------------------------------------------------
# 6. One owner, one wording
# ---------------------------------------------------------------------------


def test_every_wall_refusal_comes_from_the_one_owner():
    """One owner, one wording, one behaviour - asserted, not intended.

    The frontend guard predates the wall and its wording was the model the one
    owner was generalised from. Asserting the single import here is what stops
    a future stage from growing a second voice, and with it a second behaviour:
    the old tracking guard checked ``requires_grad`` only, and that is exactly
    how a forward dual walked through it for two phases.
    """

    import ast
    import pathlib

    from witwin.radar import policy

    assert policy.refuse_derivative.__module__ == "witwin.radar.policy"

    root = pathlib.Path(__file__).resolve().parents[1] / "witwin" / "radar"
    guarded = ("processing/detection.py", "processing/angle.py", "processing/tracking.py", "frontend.py")
    for relative in guarded:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("policy")
            for alias in node.names
        }
        assert "refuse_derivative" in imported, relative
        # And no second implementation: the deleted hand-rolled guard does not
        # come back under its old name.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                assert node.name != "_refuse_gradient", relative
