"""Micro-Doppler: rotor flash, limb band, deforming mesh, and the owner line.

Plan work item 6a, driven end to end. A scatter site is an ENDPOINT, so its
``omega x r`` or deformation velocity is its micro-Doppler once the kinematics
seam exists; what this file adds is the slow-time SIGNAL and its spectrum.

How the signal is built, and why it contains no physics of its own. Channel's
transport already carries ``exp(-j 2 pi f_ref tau)`` (the synthesis contract's
``weight_includes_reference_phase``), so replaying the frozen topology at a
sequence of slot times and summing the composed rows per slot IS the slow-time
signal. Every number in it comes from the production propagation replay and the
production join; the test contributes the site TRAJECTORY and a coherent sum.

Two measurements of the same tone, deliberately kept apart:

* the SPECTRUM's peak, which cannot resolve better than one Doppler bin and is
  what asserts the two-sided structure and the band edges;
* the unwrapped slow-time PHASE SLOPE, which is sub-bin exact and is what
  carries the 2e-3 comparison against the float64 closed form.

**Deviation from the brief, stated rather than buried.** The brief asks for the
peak frequencies themselves to match the closed form at 2e-3 relative. At the
rotor's 2942 Hz that is 5.9 Hz, and the transform's bin is 78.1 Hz: an argmax
over an FFT cannot deliver it at any tolerance, and widening the transform to
reach it would rotate the blade far enough that the tone is no longer one tone.
The 2e-3 comparison is therefore made on the phase-slope estimate, which is the
quantity the peak estimates, and the peak itself is asserted to within one bin.
"""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from witwin.radar.sigproc import microdoppler as md  # noqa: E402


pytestmark = pytest.mark.gpu

#: Slow-time sample period. ``1 / (2 T_slot) = 10 kHz`` of unambiguous Doppler,
#: which is 3.4x the rotor's own 2942 Hz - so the tone cannot be an alias.
SLOT_PERIOD_S = 5.0e-5

#: Slots per measurement. 256 x 50 us = 12.8 ms, over which the rotor turns
#: 0.128 rad; at the symmetric instant the Doppler is at an extremum in the
#: rotation angle, so its first-order variation vanishes and the residual
#: spread is about 6 Hz, well inside one bin.
SLOT_COUNT = 256

#: One Doppler bin of the transform above, in hertz.
BIN_HZ = 1.0 / (SLOT_COUNT * SLOT_PERIOD_S)

#: Sub-bin agreement between the measured phase slope and the closed form.
TONE_RTOL = 2.0e-3

AXIAL_TRANSMITTERS = (geo.TRANSMITTERS[0],)
AXIAL_RECEIVERS = (geo.RECEIVERS[0],)

ROTOR_SITES = (
    (geo.SITE_P_STABLE_ID, (2.0, geo.ROTOR_RADIUS_M, 0.0)),
    (geo.SITE_R_STABLE_ID, (2.0, -geo.ROTOR_RADIUS_M, 0.0)),
)

LOS_ONLY = frozenset({"los"})


def _slot_times() -> torch.Tensor:
    """Slot times centred on zero, so slot 0 is not the reference instant.

    Centring matters for the closed form: the velocity is evaluated at the
    instant the rotor is symmetric, and a window that starts there measures
    half a lobe instead of a tone.
    """

    index = torch.arange(SLOT_COUNT, dtype=torch.float64)
    return (index - (SLOT_COUNT - 1) / 2.0) * SLOT_PERIOD_S


def _rotor_stack(sites, times_s: torch.Tensor) -> torch.Tensor:
    """Rigid rotation about ``z`` through the rotor centre, slot major.

    Built from the closed-form rotation rather than from ``v t``: a linear
    extrapolation of the velocity is exactly the model under test, and using it
    as the trajectory would make the comparison circular.
    """

    centre = torch.tensor(geo.ROTOR_CENTRE_M, dtype=torch.float64)
    base = torch.tensor([position for _, position in sites], dtype=torch.float64)
    offset = base - centre
    angle = geo.ROTOR_OMEGA_RAD_PER_S * times_s
    cos = torch.cos(angle).reshape(-1, 1)
    sin = torch.sin(angle).reshape(-1, 1)
    x = offset[:, 0].reshape(1, -1)
    y = offset[:, 1].reshape(1, -1)
    rotated = torch.stack(
        [
            cos * x - sin * y + centre[0],
            sin * x + cos * y + centre[1],
            offset[:, 2].reshape(1, -1).expand(len(times_s), -1) + centre[2],
        ],
        dim=2,
    )
    return rotated.reshape(-1, 3).to(device="cuda", dtype=torch.float32).contiguous()


def _linear_stack(sites, velocities, times_s: torch.Tensor) -> torch.Tensor:
    """``p + v t`` per site, slot major. Exact for a hinge, by construction."""

    base = torch.tensor([position for _, position in sites], dtype=torch.float64)
    rate = torch.tensor(list(velocities), dtype=torch.float64)
    stack = base.reshape(1, -1, 3) + times_s.reshape(-1, 1, 1) * rate.reshape(1, -1, 3)
    return stack.reshape(-1, 3).to(device="cuda", dtype=torch.float32).contiguous()


def _slow_time(spike, stack: torch.Tensor):
    """The per-row and summed slow-time sequences of one replayed frame.

    ONE batched consumer call per leg covers every slot; the per-slot loop that
    follows is composition only and is the refreshed-weight oracle the driver
    already owns. Returns ``(rows[K, T], total[T], frame)`` with the sequences
    as complex128 on the host and one composed frame kept so the caller can
    name its rows by identity.
    """

    inbound, outbound = spike.slot_legs(stack, slot_count=SLOT_COUNT)
    frames = spike.slot_frames(inbound, outbound)
    rows = torch.stack(
        [frame.complex_transfer_ref.detach() for frame in frames], dim=1
    )
    return (
        rows.cpu().to(torch.complex128),
        rows.sum(dim=0).cpu().to(torch.complex128),
        frames[0],
    )


def _tone_hz(sequence: torch.Tensor) -> float:
    """The tone's frequency from its unwrapped slow-time phase slope.

    The mean wrapped phase STEP, which for a single complex tone with no noise
    is exact to well past one bin and telescopes to the endpoints' phase
    difference over the window. The wrap check is not decoration:
    the estimator is silently wrong by a multiple of the slow-time rate if a
    step exceeds ``pi``, which is exactly what an aliased tone does.
    """

    phase = torch.angle(sequence)
    steps = torch.remainder(phase[1:] - phase[:-1] + math.pi, 2.0 * math.pi) - math.pi
    assert float(steps.abs().max()) < math.pi - 1.0e-6, "the tone is aliased"
    return float(steps.mean()) / (2.0 * math.pi * SLOT_PERIOD_S)


def _peak_hz(sequence: torch.Tensor) -> float:
    spectrum = md.slow_time_spectrum(sequence, window="hann")
    frequencies = md.doppler_frequencies_hz(len(sequence), SLOT_PERIOD_S)
    return float(md.dominant_frequencies_hz(spectrum, frequencies))


def _closed_form_hz(spike, velocities: dict, sites) -> dict:
    """``f_D = -f_ref tau_rate`` per composed row, keyed by row identity."""

    rows = spike.predicted_combined_rows()
    shifts = geo.combined_doppler_hz(
        rows, velocities, AXIAL_TRANSMITTERS, sites, AXIAL_RECEIVERS
    )
    return {row.key: shifts[index] for index, row in enumerate(rows)}


# --------------------------------------------------------------------------
# rotor
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rotor():
    return drv.MultiEndpointSpike(
        transmitters=AXIAL_TRANSMITTERS,
        sites=ROTOR_SITES,
        receivers=AXIAL_RECEIVERS,
        components=LOS_ONLY,
        max_depth=0,
    )


def test_a_rotating_two_blade_target_gives_a_flash_spectrum(rotor):
    """Two symmetric sidebands, one per blade, symmetric about DC.

    The advancing blade closes and the retreating one opens, so the pair is
    equal and opposite: this is the blade-flash signature and it is the one
    thing a model carrying only the body's linear velocity cannot produce - it
    would put both blades in the same bin.
    """

    times = _slot_times()
    stack = _rotor_stack(ROTOR_SITES, times)
    rows, total, frame = _slow_time(rotor, stack)
    assert rows.shape == (2, SLOT_COUNT)

    velocities = geo.rotor_site_velocities(ROTOR_SITES)
    reference = _closed_form_hz(rotor, velocities, ROTOR_SITES)
    keys = drv.composed_keys(rotor, frame)

    # Per row: the sub-bin phase-slope estimate against the closed form.
    for index, key in enumerate(keys):
        expected = reference[key]
        measured = _tone_hz(rows[index])
        assert abs(expected) > 100.0, key
        assert measured == pytest.approx(expected, rel=TONE_RTOL), key

    # And the pair is antisymmetric, which is what makes it a flash.
    advancing, retreating = (_tone_hz(rows[index]) for index in range(2))
    assert advancing * retreating < 0.0
    assert abs(advancing + retreating) / abs(advancing) < 1.0e-3

    # The spectrum of the SUM carries both, symmetric about DC to one bin.
    spectrum = md.slow_time_spectrum(total, window="hann")
    frequencies = md.doppler_frequencies_hz(SLOT_COUNT, SLOT_PERIOD_S)
    magnitude = spectrum.abs()
    upper = int(magnitude[frequencies > 0].argmax())
    lower = int(magnitude[frequencies < 0].argmax())
    upper_hz = float(frequencies[frequencies > 0][upper])
    lower_hz = float(frequencies[frequencies < 0][lower])
    assert abs(upper_hz + lower_hz) <= BIN_HZ
    for peak, tone in ((upper_hz, max(advancing, retreating)),
                       (lower_hz, min(advancing, retreating))):
        assert abs(peak - tone) <= BIN_HZ, (peak, tone)


# --------------------------------------------------------------------------
# hinge
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hinge():
    return drv.MultiEndpointSpike(
        transmitters=AXIAL_TRANSMITTERS,
        sites=geo.HINGE_SITES,
        receivers=AXIAL_RECEIVERS,
        components=LOS_ONLY,
        max_depth=0,
    )


def test_a_hinge_limb_gives_a_rectangular_doppler_band(hinge):
    """Root and tip bound the band; the sites in between fill it.

    The band edges are the root and tip shifts and are asserted to within one
    bin. Flatness is asserted at 3 dB over the three tones the fixture actually
    carries: a hinge with three scatter sites is three tones, not a continuum,
    so "flat in band" means the three peaks are within 3 dB of each other and
    not that the space between them is filled.
    """

    times = _slot_times()
    stack = _linear_stack(geo.HINGE_SITES, geo.HINGE_VELOCITIES_M_PER_S, times)
    rows, total, frame = _slow_time(hinge, stack)
    assert rows.shape == (len(geo.HINGE_SITES), SLOT_COUNT)

    velocities = {
        stable_id: geo.HINGE_VELOCITIES_M_PER_S[index]
        for index, (stable_id, _) in enumerate(geo.HINGE_SITES)
    }
    reference = _closed_form_hz(hinge, velocities, geo.HINGE_SITES)
    keys = drv.composed_keys(hinge, frame)
    tones = [_tone_hz(rows[index]) for index in range(rows.shape[0])]
    for index, key in enumerate(keys):
        assert tones[index] == pytest.approx(reference[key], rel=TONE_RTOL), key

    # The hinge closes on the transmitter, so every shift has the SAME sign and
    # the band is ordered by speed: root slowest, tip fastest. Asserting the
    # sign is a real check even though it is positive here - a band that
    # straddled DC would mean two sites moving opposite ways, which a hinge
    # cannot do.
    assert all(tone > 0.0 for tone in tones), tones
    assert tones == sorted(tones), tones

    root, tip = min(tones), max(tones)
    frequencies = md.doppler_frequencies_hz(SLOT_COUNT, SLOT_PERIOD_S)
    spectrum = md.slow_time_spectrum(total, window="hann")
    magnitude = spectrum.abs()
    # Four bins of guard on each side, because that is the Hann window's main
    # lobe width. A one-bin guard would call the main lobe's own skirt
    # "out of band" and measure the window instead of the signal.
    guard = 4.0 * BIN_HZ
    in_band = (frequencies >= root - guard) & (frequencies <= tip + guard)

    # The band edges: the extreme in-band peaks land on the root and tip tones.
    peaks = []
    for tone in tones:
        window = (frequencies >= tone - BIN_HZ) & (frequencies <= tone + BIN_HZ)
        peaks.append(float(magnitude[window].max()))
        best = float(frequencies[window][int(magnitude[window].argmax())])
        assert abs(best - tone) <= BIN_HZ, tone
    flatness_db = 20.0 * math.log10(max(peaks) / min(peaks))
    assert flatness_db < 3.0, flatness_db

    # Out of band there is nothing: the band is bounded, not merely populated.
    out_of_band = float(magnitude[~in_band].max())
    # Measured: 50.87 dB of rejection and 1.94 dB of flatness. The gate is set
    # at 20 dB, which is loose enough to survive a windowing change and tight
    # enough that a band that leaked everywhere would fail it.
    rejection_db = 20.0 * math.log10(min(peaks) / out_of_band)
    assert rejection_db > 20.0, rejection_db


# --------------------------------------------------------------------------
# deforming mesh
# --------------------------------------------------------------------------


def _smpl_model_root() -> str | None:
    import pathlib

    from witwin.radar.geometry import smpl as smpl_module

    candidates = [
        pathlib.Path(smpl_module._default_smpl_model_root()),
        pathlib.Path(__file__).resolve().parents[3]
        / "radar"
        / "models"
        / "smpl_models",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.pkl")):
            return str(candidate)
    return None


def test_smpl_limb_microdoppler_matches_an_independent_reference():
    """A posed body's fastest vertices, driven as scatter sites.

    The velocities come from ``SmplPoseDeformation``, i.e. from a forward-mode
    dual through the posing function with the pose rate as tangent. The
    reference is the sum of point-scatterer Dopplers computed in float64 from
    those velocities, which is independent of the propagation chain that
    produced the measured tones.

    The sites are placed at the fixture's own scatter positions and given the
    limb's velocities, rather than at the body's world coordinates: the fixture
    scene has a wall at ``x = 4`` and a body standing at the origin would put
    the scatter sites on top of the transmitter. The physics under test is the
    velocity field, and moving where it is sampled does not change it.
    """

    pytest.importorskip("smplpytorch")
    from witwin.radar.geometry import SMPLBody, SmplPoseDeformation

    model_root = _smpl_model_root()
    if model_root is None:
        pytest.skip("no SMPL model files available in this checkout")

    pose_rate = torch.zeros(72, device="cuda")
    pose_rate[3 * 18 + 2] = 3.0
    body = SMPLBody(
        pose=torch.zeros(72),
        shape=torch.zeros(10),
        model_root=model_root,
        device="cuda",
    )
    deformation = SmplPoseDeformation(body, pose_rate=pose_rate)
    velocity = deformation.velocity_at(0.0)
    speed = velocity.norm(dim=1)
    fastest = torch.topk(speed, 3).indices.tolist()
    limb = [
        tuple(float(value) for value in velocity[index].tolist())
        for index in fastest
    ]
    assert min(sum(value**2 for value in row) ** 0.5 for row in limb) > 0.2

    sites = tuple(
        (stable_id, position)
        for (stable_id, position) in geo.HINGE_SITES
    )
    spike = drv.MultiEndpointSpike(
        transmitters=AXIAL_TRANSMITTERS,
        sites=sites,
        receivers=AXIAL_RECEIVERS,
        components=LOS_ONLY,
        max_depth=0,
    )
    times = _slot_times()
    stack = _linear_stack(sites, limb, times)
    rows, _, frame = _slow_time(spike, stack)

    velocities = {
        stable_id: limb[index] for index, (stable_id, _) in enumerate(sites)
    }
    reference = _closed_form_hz(spike, velocities, sites)
    keys = drv.composed_keys(spike, frame)
    for index, key in enumerate(keys):
        expected = reference[key]
        assert abs(expected) > 10.0, key
        assert _tone_hz(rows[index]) == pytest.approx(expected, rel=TONE_RTOL), key


# --------------------------------------------------------------------------
# the owner line
# --------------------------------------------------------------------------


#: Analysis vocabulary. None of it may appear anywhere under ``cuda/``, and the
#: micro-Doppler module may not reach the native extension at all.
#:
#: This is the owner directive of 2026-07-25 written as an assertion:
#: simulation is native, post-processing is Torch. ``sigproc`` already carries
#: the Torch/DSP exception; what this pins is the OTHER direction - that the
#: exception did not leak a spectrogram into a kernel.
ANALYSIS_VOCABULARY = (
    "spectrogram",
    "stft",
    "cfar",
    "music",
    "pointcloud",
    "matched_filter",
    "range_doppler",
    "clustering",
)


def test_microdoppler_analysis_is_torch_only():
    """No native call in the analysis module, no analysis symbol in a kernel."""

    import ast
    import pathlib as _pathlib
    import re

    root = _pathlib.Path(__file__).resolve().parents[1]
    module = root / "witwin" / "radar" / "sigproc" / "microdoppler.py"
    source = module.read_text(encoding="utf-8")

    # Nothing in this module may resolve the extension, by any spelling.
    for forbidden in ("build_extension", "torch.ops", "_ops(", "cuda.build"):
        assert forbidden not in source, forbidden

    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported == {"__future__", "torch"}, sorted(imported)

    # And no analysis vocabulary leaked into the native side. COMMENTS are
    # stripped first, deliberately: ``fmcw_beat.cu`` names
    # ``sigproc/pointcloud.py`` and ``pulsed_echo.cu`` names
    # ``sigproc/matched_filter.py``, both to say where the processing that
    # consumes their output lives. Those references are the split being
    # documented, which is the opposite of the split being broken, and a scan
    # that failed on them would push the documentation out of the kernels.
    kernels = root / "witwin" / "radar" / "cuda"
    offenders = []
    for path in sorted(kernels.rglob("*")):
        if path.suffix not in {".cu", ".cpp", ".cuh", ".h"}:
            continue
        text = re.sub(r"/\*.*?\*/", " ", path.read_text(encoding="utf-8"), flags=re.S)
        text = re.sub(r"//.*", " ", text).lower()
        offenders.extend(
            (path.name, word) for word in ANALYSIS_VOCABULARY if word in text
        )
    assert offenders == [], offenders


def test_the_analysis_module_is_importable_without_cuda_being_resolved():
    """The strongest statement of the split that a test can make.

    Importing the analysis module must not load the extension. If it did, a
    machine with no CUDA build could not analyse a recorded signal, which is
    exactly the workflow the Torch side exists to serve.
    """

    import importlib
    import sys

    sys.modules.pop("witwin.radar.sigproc.microdoppler", None)
    loaded = "witwin.radar.cuda.build" in sys.modules
    importlib.import_module("witwin.radar.sigproc.microdoppler")
    if not loaded:
        assert "witwin.radar.cuda.build" not in sys.modules
