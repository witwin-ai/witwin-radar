"""Benchmark the Radar processing chain, stage by stage and end to end.

This is the measurement behind Phase-8 work item 7. The owner directive is that
Phase 8 ships NO native DSP; this tool exists to justify that default with
numbers rather than to look for permission to break it. It is run and recorded
regardless of the expected answer.

Timing convention, deliberately identical to the deleted
``tools/benchmark_dirichlet_cuda.py``
so the repository has ONE: :func:`cuda_time` is CUDA events with an explicit
``torch.cuda.synchronize``, :func:`peak_memory_mb` resets the allocator's peak
statistics around a single call, results are medians, and ``--json`` prints the
machine-readable form after the table.

**The honest caveat.** A synchronization inside a native kernel is invisible from
Python. ``torch.fft.fft`` can synchronize inside cuFFT plan creation and the
dispatch ledger will not see it. Every wall time below is therefore measured
with CUDA events; NOTHING here infers a time from a counter. The ledger counts
dispatches and host-visible observations, which is a different claim.

Sizes. Two per stage: the FROZEN FIXTURE size, which is what the acceptance
tests run at, and one REALISTIC size (128 chirps, 12 virtual elements, 256
samples). Reporting both is what separates "launch bound" from "work bound" -
if a 48x larger cube costs the same wall time, the stage is dispatch bound and
the answer to a cuFFT wrapper is still no, because a wrapper would replace one
dispatch with another.

The end-to-end pipeline runs against the REAL multi-endpoint Channel fixture,
extended to a 3 TX x 4 RX front end so the angle-of-arrival and point-cloud
stages are exercised rather than skipped. The declared array is a nominal
half-wavelength MIMO array while the fixture's Channel endpoints are its own
physical positions, so the ANGLES this pipeline produces are meaningless. That
is deliberate: this is a cost measurement, and angular correctness is asserted
by ``tests/processing/`` against analytic targets.

Example:
    python tools/benchmark_processing.py --runs 200 --json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The fixture orchestration lives under ``tests/`` because it is a fixture, not
# a production owner. The deleted ``benchmark_dirichlet_cuda.py`` reached for
# its reference oracle the same way.
if str(REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests"))


FIXTURE = {
    "chirps": 8,
    "num_tx": 2,
    "num_rx": 2,
    "pairs": 4,
    "samples": 256,
    "label": "fixture",
}
REALISTIC = {
    "chirps": 128,
    "num_tx": 3,
    "num_rx": 4,
    "pairs": 12,
    "samples": 256,
    "label": "realistic",
}

#: The 3 TX x 4 RX front end the end-to-end pipeline declares. Twelve virtual
#: elements is the smallest array on which both angle routes and the point cloud
#: are real calls rather than refusals.
PIPELINE_NUM_TX = 3
PIPELINE_NUM_RX = 4


# ---------------------------------------------------------------------------
# Timing primitives, verbatim from the Dirichlet benchmark
# ---------------------------------------------------------------------------


def cuda_time(fn, *, warmup: int, runs: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times), times


def peak_memory_mb(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / (1024.0 * 1024.0)


def host_observations(fn) -> dict[str, int]:
    """Dispatch and host-observation counters for one call of ``fn``."""

    from support.dsp_ledger import DspLedger

    with DspLedger() as ledger:
        fn()
        return {"transforms": ledger.transform_count, **ledger.host}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _complex_noise(shape, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    real = torch.randn(shape, generator=generator, device="cuda")
    imag = torch.randn(shape, generator=generator, device="cuda")
    return torch.complex(real, imag).to(torch.complex64)


def _array_spec(num_tx: int, num_rx: int):
    """The nominal half-wavelength array, from the one fixture owner."""

    from support.pipeline_chain import array_spec

    return array_spec(num_tx, num_rx)


def _fmcw_case(size, *, num_tx: int | None = None, num_rx: int | None = None):
    """A complete FMCW cube, axes record and array, at one size.

    Overriding the front end overrides the PAIR count too: the axes record
    refuses a cube whose pair axis does not match ``num_tx * num_rx``, which is
    the check that makes a wider array a different measurement rather than the
    same one with a relabelled axis.
    """

    from support import exact_bin_grid as grid
    from witwin.radar.processing import ArrayGeometry, ProcessingAxes, ProcessingCube
    from witwin.radar.synthesis.assembly import SynthesisResult

    spec = grid.fmcw_spec(size["chirps"])
    from dataclasses import replace

    num_tx = size["num_tx"] if num_tx is None else num_tx
    num_rx = size["num_rx"] if num_rx is None else num_rx
    pairs = num_tx * num_rx
    spec = replace(spec, num_tx=num_tx, num_rx=num_rx)
    cube = _complex_noise((size["chirps"], pairs, size["samples"]), seed=11)
    result = SynthesisResult.from_fmcw(cube, spec)
    axes = ProcessingAxes.from_synthesis(result, spec, _array_spec(num_tx, num_rx))
    return {
        "result": result,
        "axes": axes,
        "cube": ProcessingCube.from_synthesis(result, axes),
        "array": ArrayGeometry.from_axes(axes),
        "raw": cube,
    }


def _ofdm_case(size):
    from support import exact_bin_grid as grid
    from witwin.radar.processing import ProcessingAxes, ProcessingCube
    from witwin.radar.synthesis.assembly import SynthesisResult

    spec = grid.ofdm_spec(size["chirps"])
    cube = _complex_noise((size["chirps"], size["pairs"], 64), seed=12)
    result = SynthesisResult.from_ofdm(cube, spec)
    axes = ProcessingAxes.from_synthesis(
        result, spec, _array_spec(size["num_tx"], size["num_rx"])
    )
    return {"axes": axes, "cube": ProcessingCube.from_synthesis(result, axes)}


def _pulsed_case(size):
    from support import exact_bin_grid as grid
    from witwin.radar.processing import ProcessingAxes, ProcessingCube
    from witwin.radar.synthesis.assembly import SynthesisResult

    spec = grid.pulsed_spec(size["chirps"])
    cube = _complex_noise((size["chirps"], size["pairs"], 128), seed=13)
    result = SynthesisResult.from_pulsed(cube, spec)
    axes = ProcessingAxes.from_synthesis(
        result, spec, _array_spec(size["num_tx"], size["num_rx"])
    )
    return {
        "axes": axes,
        "spec": spec,
        "cube": ProcessingCube.from_synthesis(result, axes),
    }


# ---------------------------------------------------------------------------
# The pipeline itself: ONE owner, shared with the budget test
# ---------------------------------------------------------------------------


def _pipeline_fixture():
    """One real Channel frame at 3 TX x 4 RX, frozen outside the timed region."""

    from support.pipeline_chain import pipeline_inputs

    return pipeline_inputs(num_chirps=FIXTURE["chirps"])


# ---------------------------------------------------------------------------
# Stage groups
# ---------------------------------------------------------------------------


def group_transforms(size, args) -> list[dict]:
    from witwin.radar.processing import fft2_aoa, range_doppler_map, range_profile
    from witwin.radar.processing.range_doppler import matched_filter
    from witwin.radar.processing.signal import taper

    fmcw = _fmcw_case(size)
    ofdm = _ofdm_case(size)
    pulsed = _pulsed_case(size)
    profile = range_profile(fmcw["cube"], window="hann")

    rows = []

    def add(name, fn, *, note=""):
        median, _ = cuda_time(fn, warmup=args.warmup, runs=args.runs)
        rows.append(
            {
                "group": "transforms",
                "size": size["label"],
                "stage": name,
                "median_ms": median,
                "peak_mb": peak_memory_mb(fn),
                "host": host_observations(fn),
                "note": note,
            }
        )

    add("range_profile.fmcw", lambda: range_profile(fmcw["cube"], window="hann"))
    add("range_profile.ofdm", lambda: range_profile(ofdm["cube"], window="hann"))
    add("range_profile.pulsed", lambda: range_profile(pulsed["cube"], window="hann"))
    add("range_doppler_map", lambda: range_doppler_map(profile, window="hann"))

    # The two halves of a range profile, separated: is the cost the transform or
    # the window multiply?
    data = fmcw["cube"].data
    add("window_multiply_only", lambda: taper(data, "hann", dim=-1))
    add(
        "transform_only",
        lambda: torch.fft.fft(data, dim=-1, norm="forward"),
    )

    # The matched filter's two transforms, with and without the complex128
    # upcast S4 deleted. This is what that deletion bought.
    pdata = pulsed["cube"].data
    pspec = pulsed["spec"]
    add("matched_filter.float32", lambda: matched_filter(pdata, pspec))
    add(
        "matched_filter.complex128",
        lambda: matched_filter(pdata, pspec, dtype=torch.complex128),
        note="the deleted unconditional upcast",
    )

    # The 2-D AoA fft2, on a virtual array wide enough for it.
    wide = _fmcw_case(size, num_tx=6, num_rx=4)
    virtual = _complex_noise((24, 32), seed=17)
    add("fft2_aoa", lambda: fft2_aoa(virtual, wide["array"], fft_size=64))

    # The micro-Doppler framing copy, against the transform it feeds.
    slow = _complex_noise((size["chirps"] * 8,), seed=19)
    frame_length, hop = 32, 8
    add(
        "microdoppler.framing_copy",
        lambda: slow.unfold(-1, frame_length, hop).contiguous(),
    )
    frames = slow.unfold(-1, frame_length, hop).contiguous()
    add("microdoppler.transform", lambda: torch.fft.fft(frames, dim=-1))

    return rows


def group_cfar(size, args) -> list[dict]:
    from witwin.radar.processing import ca_cfar, ca_cfar_1d, ca_cfar_fast, os_cfar

    rows = []
    doppler = max(size["chirps"], 8)
    ranges = size["samples"]
    for batch_shape, label in (((), "single"), ((size["pairs"],), "batched")):
        magnitude = torch.rand(
            (*batch_shape, doppler, ranges), device="cuda", dtype=torch.float32
        )
        for name, fn in (
            ("ca_cfar", ca_cfar),
            ("ca_cfar_fast", ca_cfar_fast),
            ("os_cfar", os_cfar),
        ):
            def call(_fn=fn, _m=magnitude):
                return _fn(_m)

            median, _ = cuda_time(call, warmup=args.warmup, runs=args.runs)
            rows.append(
                {
                    "group": "cfar",
                    "size": size["label"],
                    "stage": f"{name}.{label}",
                    "shape": tuple(magnitude.shape),
                    "median_ms": median,
                    "peak_mb": peak_memory_mb(call),
                    "host": host_observations(call),
                    "note": "",
                }
            )
    profile = torch.rand((size["pairs"], ranges), device="cuda", dtype=torch.float32)
    def call():
        return ca_cfar_1d(profile)

    median, _ = cuda_time(call, warmup=args.warmup, runs=args.runs)
    rows.append(
        {
            "group": "cfar",
            "size": size["label"],
            "stage": "ca_cfar_1d",
            "shape": tuple(profile.shape),
            "median_ms": median,
            "peak_mb": peak_memory_mb(call),
            "host": host_observations(call),
            "note": "",
        }
    )
    return rows


def group_aoa(size, args) -> list[dict]:
    from witwin.radar.processing import (
        fft2_aoa,
        music_spectrum,
        phase_comparison_aoa,
        tdm_compensate,
    )

    rows = []

    def add(name, fn, *, note=""):
        median, _ = cuda_time(fn, warmup=args.warmup, runs=args.runs)
        rows.append(
            {
                "group": "aoa",
                "size": size["label"],
                "stage": name,
                "median_ms": median,
                "peak_mb": peak_memory_mb(fn),
                "host": host_observations(fn),
                "note": note,
            }
        )

    detections = max(size["pairs"], 8)
    narrow = _fmcw_case(size, num_tx=3, num_rx=4)
    wide = _fmcw_case(size, num_tx=6, num_rx=4)
    virtual12 = _complex_noise((12, detections), seed=23)
    virtual24 = _complex_noise((24, detections), seed=29)
    velocities = torch.rand(detections, device="cuda", dtype=torch.float32)

    add(
        "tdm_compensate.vectorized",
        lambda: tdm_compensate(
            virtual12, velocities, narrow["array"], narrow["axes"]
        ),
    )

    # The deleted form, reconstructed here and nowhere else: a Python loop over
    # transmitters with an in-place multiply on a clone.
    def legacy_tdm():
        array = narrow["array"]
        out = virtual12.clone()
        chirp_period = float(narrow["axes"].slow_time_period_s) / array.num_tx
        for index in range(1, array.num_tx):
            phase = (
                array.phase_sign
                * 4
                * math.pi
                * velocities
                * index
                * chirp_period
                / array.wavelength_m
            )
            out[index * array.num_rx : (index + 1) * array.num_rx] *= torch.exp(
                1j * phase
            )
        return out

    add("tdm_compensate.python_loop", legacy_tdm, note="the deleted form")
    add(
        "phase_comparison_aoa",
        lambda: phase_comparison_aoa(virtual12, narrow["array"], fft_size=64),
    )
    add("fft2_aoa", lambda: fft2_aoa(virtual24, wide["array"], fft_size=64))

    bins, rows_m, cols_n, snapshots = 8, 6, 8, 16
    angle_data = _complex_noise((bins, rows_m, cols_n, snapshots), seed=31)
    elevation = torch.linspace(-0.4, 0.4, 17, device="cuda", dtype=torch.float32)
    azimuth = torch.linspace(-0.6, 0.6, 25, device="cuda", dtype=torch.float32)
    add(
        "music_spectrum",
        lambda: music_spectrum(
            angle_data,
            wide["array"],
            elevation_rad=elevation,
            azimuth_rad=azimuth,
            num_signals=3,
            spatial_smooth=3,
        ),
    )

    # The two halves of MUSIC, separated: the eigen-decomposition against the
    # smoothing construction the list comprehension used to build.
    smoothing = 3
    unfolded = angle_data.unfold(1, rows_m - smoothing, 1).unfold(
        2, cols_n - smoothing, 1
    )
    add("music.smoothing_unfold", lambda: unfolded.contiguous())

    def legacy_smoothing():
        return torch.stack(
            [
                angle_data[
                    :, i : i + rows_m - smoothing, j : j + cols_n - smoothing, :
                ]
                for i in range(smoothing + 1)
                for j in range(smoothing + 1)
            ]
        )

    add("music.smoothing_stack", legacy_smoothing, note="the deleted form")
    elements = (rows_m - smoothing) * (cols_n - smoothing)
    covariance = _complex_noise((bins, elements, elements), seed=37)
    hermitian = covariance + covariance.transpose(-1, -2).conj()
    add("music.eigh", lambda: torch.linalg.eigh(hermitian))
    return rows


def group_cube(size, args) -> list[dict]:
    from witwin.radar.processing import (
        ProcessingCube,
        beam_cube,
        conventional_steering,
        range_doppler_map,
        range_profile,
    )
    from witwin.radar.synthesis.assembly import assemble_frame_cube

    case = _fmcw_case(size, num_tx=PIPELINE_NUM_TX, num_rx=PIPELINE_NUM_RX)
    raw = case["raw"]
    axes = case["axes"]
    array = case["array"]
    rd = range_doppler_map(range_profile(case["cube"], window="hann"), window="hann")
    directions = torch.stack(
        [
            torch.tensor(
                [math.sin(angle), math.cos(angle), 0.0], dtype=torch.float64
            )
            for angle in torch.linspace(-0.6, 0.6, 64).tolist()
        ]
    )
    weights = conventional_steering(array, directions)

    rows = []

    def add(name, fn, *, note=""):
        median, _ = cuda_time(fn, warmup=args.warmup, runs=args.runs)
        rows.append(
            {
                "group": "cube",
                "size": size["label"],
                "stage": name,
                "median_ms": median,
                "peak_mb": peak_memory_mb(fn),
                "host": host_observations(fn),
                "note": note,
            }
        )

    add(
        "assemble_frame_cube",
        lambda: assemble_frame_cube(
            raw, num_tx=PIPELINE_NUM_TX, num_rx=PIPELINE_NUM_RX
        ),
        note="permute/reshape/permute/contiguous, a full copy",
    )
    add(
        "ProcessingCube.from_synthesis",
        lambda: ProcessingCube.from_synthesis(case["result"], axes),
    )
    add(
        "conventional_steering",
        lambda: conventional_steering(array, directions),
        note="scene static; the element table is built once per array",
    )
    add(
        "beam_cube",
        lambda: beam_cube(rd, weights, directions=directions),
        note="the second full copy",
    )
    return rows


def group_pipeline(args) -> list[dict]:
    from support.pipeline_chain import run_pipeline

    batch, spec, spec_array = _pipeline_fixture()

    rows = []
    for detector in ("ca_cfar_fast", "ca_cfar", "os_cfar"):
        def call(_d=detector):
            return run_pipeline(batch, spec, spec_array, detector=_d)

        median, samples = cuda_time(call, warmup=args.warmup, runs=args.runs)
        rows.append(
            {
                "group": "pipeline",
                "size": "fixture",
                "stage": f"full_pipeline.{detector}",
                "median_ms": median,
                "min_ms": min(samples),
                "max_ms": max(samples),
                "peak_mb": peak_memory_mb(call),
                "host": host_observations(call),
                "note": "synthesize -> cube -> range -> RD -> CFAR -> AoA -> point cloud",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["transforms", "cfar", "aoa", "cube", "pipeline"],
        help="Which stage groups to measure.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON after the table."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the processing benchmark.")

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"runs: {args.runs}, warmup: {args.warmup}")
    print()

    rows: list[dict] = []
    for size in (FIXTURE, REALISTIC):
        if "transforms" in args.groups:
            rows += group_transforms(size, args)
        if "cfar" in args.groups:
            rows += group_cfar(size, args)
        if "aoa" in args.groups:
            rows += group_aoa(size, args)
        if "cube" in args.groups:
            rows += group_cube(size, args)
    if "pipeline" in args.groups:
        rows += group_pipeline(args)

    print(
        f"{'group':<11} {'size':<10} {'stage':<32} {'median_ms':>10} {'peak_mb':>9} "
        f"{'fft':>4} {'host':>5}  note"
    )
    print("-" * 122)
    for row in rows:
        host = row.get("host", {})
        transforms = int(host.get("transforms", 0))
        observations = sum(
            int(value) for name, value in host.items() if name != "transforms"
        )
        print(
            f"{row['group']:<11} {row['size']:<10} {row['stage']:<32} "
            f"{row['median_ms']:10.4f} {row['peak_mb']:9.2f} {transforms:4d} "
            f"{observations:5d}  {row.get('note', '')}"
        )

    if args.json:
        print()
        print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
