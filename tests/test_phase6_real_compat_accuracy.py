"""Criterion A4, measured against the exact answer rather than against a binary.

The real-compatibility criterion says the real path stays the existing Radar
baseline. Work item 8 moved the two ``torch.cdist`` distance fields, the
spreading term, the pattern lookup, and the polarization projection into the
``sensor_weight`` kernel, and the migrated route does NOT reproduce the old
cube bit for bit: the same scene differs by about 1.4e-3 relative. That number
invites the wrong question. "Is it the same as the old binary" cannot be
answered by a test at all once the old binary is gone, and it is not the
property anyone wants; the property anyone wants is "is it the right answer".

So this file measures BOTH routes' successor against the exact value. The
closed form both the base and the migrated route evaluate is written out in
float64 from the same float32 scene inputs, and the native cube is required to
sit inside the float32 phase floor of it. The floor is derived, not fitted:
the Dirichlet family accumulates the absolute phase ``2 pi fc tau`` in float32,
so a delay carrying one ulp of relative error moves that phase by
``2 pi fc tau eps`` radians, which at 77 GHz and a 20 m path is about 4e-3 rad
and is the whole of the observed deviation.

Measured with this oracle on the stage's capture scene (3 TX x 4 RX, 40 paths,
report wf7/05):

    base (pre-migration binary) vs float64 oracle: 1.88e-3
    tip  (migrated route)       vs float64 oracle: 1.65e-3
    base vs tip                                  : 1.43e-3

The migrated route is CLOSER to the exact value than the route it replaced, and
the two differ by less than either one's distance from the truth. A4 holds in
the only sense that survives the old binary: the real path reproduces the
baseline to within the baseline's own numerical accuracy. The remaining floor
is ``dirichlet.cu``'s float32 phase resolution, which is recorded debt with its
own future decision (accumulate the cycle count in double and wrap before
``sincosf``, as ``fmcw_beat.cu`` already does).

The closed form, verbatim from ``dirichlet.cu``:

    tau  = L / c0            L = |entry - tx| + fixed + |point - rx|
    phi0 = 2 pi (fc tau + S tau (t_start - tau/2))
    k0   = tau * k0_per_second
    x    = 2 pi (bin - k0) / n_fft
    D(x) = [sin((n + 0.5) x) / sin(x/2)] exp(-j n x)
    S[bin] = sum_i w_i D(x_i) exp(+j phi0_i)

``tau = L / c0`` rather than ``2 L / c0`` because ``L`` is already the full
two-way path: the legacy surface feeds the kernel ``L / 2`` metres and the
kernel doubles it, and the migrated route feeds the identical delay directly.
"""

from __future__ import annotations

import math

import pytest
import torch


CONFIG = dict(
    num_tx=3,
    num_rx=4,
    fc=77e9,
    slope=60.012,
    adc_samples=256,
    adc_start_time=6,
    sample_rate=4400,
    idle_time=7,
    ramp_end_time=58,
    chirp_per_frame=2,
    frame_per_second=30,
    num_doppler_bins=2,
    num_range_bins=256,
    num_angle_bins=64,
    power=12,
    tx_loc=((0, 0, 0), (2, 0, 0), (4, 0, 0)),
    rx_loc=((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)),
    polarization={
        "tx": [[1.0, 0.0, 0.0]] * 3,
        "rx": [[1.0, 0.0, 0.0]] * 4,
        "reflection_flip": True,
    },
)

NUM_PATHS = 24
SCENE_SEED = 23


class _Trace:
    """The smallest object ``Radar.mimo_from_trace`` reads."""

    def __init__(self, points, intensities, entry_points, fixed_path_lengths, normals):
        self.points = points
        self.intensities = intensities
        self.entry_points = entry_points
        self.fixed_path_lengths = fixed_path_lengths
        self.depths = torch.zeros(points.shape[0], dtype=torch.int32, device=points.device)
        self.normals = normals


def _scene(device):
    generator = torch.Generator().manual_seed(SCENE_SEED)
    points = torch.rand(NUM_PATHS, 3, generator=generator) * torch.tensor(
        [6.0, 3.0, 6.0]
    ) + torch.tensor([-3.0, -1.5, -9.0])
    entry = points + (torch.rand(NUM_PATHS, 3, generator=generator) - 0.5) * 0.4
    intensities = torch.rand(NUM_PATHS, generator=generator) * 0.9 + 0.1
    fixed = torch.rand(NUM_PATHS, generator=generator) * 0.5
    normals = torch.nn.functional.normalize(
        torch.randn(NUM_PATHS, 3, generator=generator), dim=-1
    )

    def to(tensor):
        return tensor.to(device=device, dtype=torch.float32).contiguous()

    return _Trace(to(points), to(intensities), to(entry), to(fixed), to(normals))


def _f64(tensor):
    return tensor.detach().to(dtype=torch.float64, device="cpu")


def _normalize(vectors):
    return vectors / torch.clamp(
        torch.linalg.norm(vectors, dim=-1, keepdim=True), min=1e-12
    )


def _path_lengths_and_weights(radar, trace):
    """Float64 ``(TX, RX, N)`` path length and real amplitude for the scene.

    Written here rather than imported from ``reference.path_math`` because that
    module is the verbatim float32 copy of the expression under test; this one
    is the same physics evaluated in double from the same float32 inputs, which
    is what makes it an accuracy reference rather than a repeat.
    """

    from witwin.radar.sensors.pattern import evaluate_antenna_pattern_vectors

    tx = _f64(radar.tx_pos)
    rx = _f64(radar.rx_pos)
    entry = _f64(trace.entry_points)
    points = _f64(trace.points)
    fixed = _f64(trace.fixed_path_lengths)
    intensities = _f64(trace.intensities)
    normals = _f64(trace.normals)

    lengths = (
        torch.linalg.norm(entry.unsqueeze(0) - tx.unsqueeze(1), dim=-1).unsqueeze(1)
        + fixed.view(1, 1, -1)
        + torch.linalg.norm(points.unsqueeze(0) - rx.unsqueeze(1), dim=-1).unsqueeze(0)
    )

    wavelength = float(radar.c0) / float(radar.config.fc)
    spreading = wavelength / (4.0 * math.pi * torch.clamp(lengths, min=1e-6))

    _, world_from_local = radar._world_from_local_matrix(
        device="cpu", dtype=torch.float64
    )

    def pattern(world_vectors):
        return evaluate_antenna_pattern_vectors(
            radar.antenna_pattern_kind,
            _f64(radar.antenna_pattern_x_angles_deg),
            _f64(radar.antenna_pattern_y_angles_deg),
            _f64(radar.antenna_pattern_x_values),
            _f64(radar.antenna_pattern_y_values),
            None,
            world_vectors @ world_from_local,
        )

    tx_gain = pattern(entry.unsqueeze(0) - tx.unsqueeze(1)).unsqueeze(1)
    rx_gain = pattern(points.unsqueeze(0) - rx.unsqueeze(1)).unsqueeze(0)
    power = torch.clamp(intensities, min=0.0).view(1, 1, -1) * torch.clamp(
        tx_gain * rx_gain, min=0.0
    )
    weights = torch.sqrt(power) * spreading

    polarization = radar.polarization
    unit_normals = _normalize(normals)
    tx_world = _normalize(_f64(polarization.tx_world))
    rx_world = _normalize(_f64(polarization.rx_world))
    reflected = tx_world.unsqueeze(1)
    if polarization.reflection_flip:
        reflected = reflected - 2.0 * (
            reflected * unit_normals.unsqueeze(0)
        ).sum(dim=-1, keepdim=True) * unit_normals.unsqueeze(0)
    reflected = _normalize(reflected)
    projection = (
        reflected.unsqueeze(1) * rx_world.view(1, rx_world.shape[0], 1, 3)
    ).sum(dim=-1)
    return lengths, weights * projection


def _oracle_cube(radar, trace, *, delay_scale: float = 1.0):
    """The float64 value of the static MIMO cube.

    ``delay_scale`` exists so a test can prove the comparison has teeth: a
    perturbation far below the tolerated amplitude error is a huge phase error
    at 77 GHz, and the gate must reject it.
    """

    solver = radar.solver
    lengths, weights = _path_lengths_and_weights(radar, trace)
    tau = lengths * delay_scale / float(radar.c0)

    fc = float(radar.config.fc)
    slope = float(radar.config.slope) * 1e12
    t_start = float(radar.config.adc_start_time) * 1e-6
    n = float(solver.n)
    n_fft = int(solver.mimo_N_fft)
    num_bins = int(solver.mimo_num_bins)

    phi0 = 2.0 * math.pi * (fc * tau + slope * tau * (t_start - 0.5 * tau))
    k0 = tau * float(solver.mimo_k0_per_second)
    bins = torch.arange(num_bins, dtype=torch.float64).view(-1, 1, 1, 1)
    x = 2.0 * math.pi * (bins - k0.unsqueeze(0)) / n_fft
    dirichlet = (torch.sin((n + 0.5) * x) / torch.sin(0.5 * x)) * torch.exp(
        -1j * n * x
    )
    spectra = (
        weights.unsqueeze(0) * dirichlet * torch.exp(1j * phi0.unsqueeze(0))
    ).sum(dim=-1)
    cube = torch.fft.ifft(spectra.permute(1, 2, 0).contiguous(), dim=-1)
    return cube.unsqueeze(2).expand(-1, -1, radar.config.chirp_per_frame, -1)


def _phase_floor(radar, lengths) -> float:
    """``2 pi fc tau eps``: one float32 ulp of delay, expressed as a phase."""

    tau_max = float(lengths.max()) / float(radar.c0)
    eps = float(torch.finfo(torch.float32).eps)
    return 2.0 * math.pi * float(radar.config.fc) * tau_max * eps


@pytest.fixture(scope="module")
def radar_and_trace():
    from conftest import make_radar_or_skip

    radar = make_radar_or_skip(CONFIG)
    return radar, _scene(radar.device)


@pytest.mark.gpu
def test_the_migrated_real_cube_sits_inside_the_float32_phase_floor(radar_and_trace):
    """A4, restated as accuracy: the native cube is the exact cube to 1 ulp of phase.

    This is the assertion the removed bit-comparison against the old binary
    cannot make. It fails if the migration ever changes the physics - a lost
    polarization sign, a doubled delay, a dropped pattern gain - because none of
    those is a float32 rounding, while it tolerates exactly the rounding that
    ``dirichlet.cu``'s float32 phase accumulation actually produces.
    """

    radar, trace = radar_and_trace
    cube = radar.mimo_from_trace(trace).detach().to(torch.complex128).cpu()
    oracle = _oracle_cube(radar, trace)
    lengths, _ = _path_lengths_and_weights(radar, trace)

    scale = float(oracle.abs().max())
    assert scale > 0.0
    error = float((cube - oracle).abs().max()) / scale
    assert error <= _phase_floor(radar, lengths), (error, _phase_floor(radar, lengths))


@pytest.mark.gpu
def test_the_accuracy_gate_rejects_a_delay_far_below_its_amplitude_tolerance(
    radar_and_trace,
):
    """The gate has teeth: it bounds the path length to a few micrometres.

    A tolerance of a few 1e-3 on a complex cube sounds loose. It is not, because
    the quantity it bounds is a phase at 77 GHz. The floor is one float32 ulp of
    delay, which on a 17 m path is about 2 micrometres of path length; scaling
    every delay by ``1 + 1e-6``, i.e. moving each site by 17 micrometres and
    changing the cube AMPLITUDE by nothing a range profile could see, is
    rejected by an order of magnitude.
    """

    radar, trace = radar_and_trace
    cube = radar.mimo_from_trace(trace).detach().to(torch.complex128).cpu()
    perturbed = _oracle_cube(radar, trace, delay_scale=1.0 + 1e-6)
    lengths, _ = _path_lengths_and_weights(radar, trace)

    scale = float(perturbed.abs().max())
    error = float((cube - perturbed).abs().max()) / scale
    assert error > _phase_floor(radar, lengths), error
