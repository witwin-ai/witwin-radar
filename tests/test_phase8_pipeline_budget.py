"""Criterion 8: the frozen full-pipeline latency and memory budgets.

Every number below was MEASURED with ``tools/benchmark_processing.py`` and is
written into this file with the headroom factor beside it, so a reader can see
both the measurement and the slack rather than a bare constant.

Measurement conditions, recorded because a wall-time budget is a statement about
a machine: NVIDIA GeForce RTX 5080, CUDA events, median of 200 runs after 20
warm-up calls, 3 TX x 4 RX front end, 8 chirps, 256 samples, the real
multi-endpoint Channel fixture. Process-to-process medians for the pipeline
spanned 2.19 to 2.34 ms over four independent runs, so the 1.30 factor leaves
24 percent over the WORST observed median and not merely over the best.

Two of the pins are wall times and are therefore device specific. That is
deliberate and it is what "frozen budget" means: if this fails on other
hardware, the number in the failure message is the report, and the correct
response is to record the new measurement on purpose - never to widen the
factor so a run goes green.

The counting pins - host observations, transform dispatches, D2H copies,
synchronizations, join launches - are device INDEPENDENT and are the ones that
catch an architectural regression. A stage that starts reading a device value to
the host fails those on any machine.

The honest caveat, restated from ``support/dsp_ledger.py``: a synchronization
inside cuFFT plan creation is invisible from Python. The counters below count
DISPATCHES and HOST-VISIBLE observations. Wall time is measured with CUDA
events, never inferred from a counter.
"""

from __future__ import annotations

import statistics

import pytest
import torch

from support.dsp_ledger import DspLedger
from support.pipeline_chain import pipeline_inputs, run_pipeline

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# The measured numbers, and the headroom
# ---------------------------------------------------------------------------

#: Measured median of the full pipeline with the default detector, in ms.
MEASURED_PIPELINE_MS = 2.23

#: Frozen at ``measured * 1.30``.
PIPELINE_LATENCY_HEADROOM = 1.30
PIPELINE_LATENCY_BUDGET_MS = MEASURED_PIPELINE_MS * PIPELINE_LATENCY_HEADROOM

#: Measured peak ALLOCATION DELTA of one pipeline call, in MB. Deterministic:
#: the same 1.128 MB on every one of four independent runs, because the
#: allocator replays the same sequence of exact sizes.
MEASURED_PIPELINE_PEAK_MB = 1.13
PIPELINE_MEMORY_HEADROOM = 1.25
PIPELINE_PEAK_BUDGET_MB = MEASURED_PIPELINE_PEAK_MB * PIPELINE_MEMORY_HEADROOM

#: Exact integers, attributed. One host observation - the ``torch.argwhere``
#: inside ``point_cloud``, which IS the stage because a point cloud has a
#: data-dependent length - and seven ``torch.fft`` dispatches: one range
#: transform, two for the Doppler stage (transform plus shift), two building the
#: velocity axis (``fftfreq`` plus ``fftshift``), and two inside the phase
#: comparison.
PIPELINE_HOST_OBSERVATIONS = 1
PIPELINE_TRANSFORM_DISPATCHES = 7

#: Measured median of one simulation frame - two leg reevaluations plus one
#: composition, no synthesis - in ms.
#:
#: The Phase-7 report recorded 2.30 ms/frame for this quantity. It is NOT
#: reproducible in this environment, and the reason is not a Phase-8 regression:
#: measured at the Phase-8 BASE commit ``4bb059a`` in the same session, on the
#: same fixture, the same call costs 3.911 ms, against 3.880 ms at HEAD. The
#: ratio HEAD/base is 0.992. The 2.30 ms figure describes a different machine
#: state; the portable claim is the ratio, and it says nothing regressed.
MEASURED_SIMULATION_FRAME_MS = 3.88
SIMULATION_FRAME_BUDGET_MS = MEASURED_SIMULATION_FRAME_MS * 1.30

#: ``os_cfar`` is the memory outlier of the three detectors and its cost is
#: pinned rather than discovered. One ``[128, 256]`` magnitude map: 138.0 MB
#: against ``ca_cfar_fast``'s 0.62 MB, a factor of 222. It materialises
#: ``[batch, D * R, n_outer]`` training patches and sorts them.
MEASURED_OS_CFAR_PEAK_MB = 138.0
OS_CFAR_PEAK_BUDGET_MB = MEASURED_OS_CFAR_PEAK_MB * 1.25


def _cuda_time(fn, *, warmup: int = 20, runs: int = 100) -> float:
    """``tools/benchmark_processing.py``'s convention, not a second one."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def _peak_mb(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / (1024.0 * 1024.0)


@pytest.fixture(scope="module")
def inputs():
    pytest.importorskip("witwin.channel")
    return pipeline_inputs()


# ---------------------------------------------------------------------------
# Latency and memory
# ---------------------------------------------------------------------------


def test_the_full_pipeline_meets_the_frozen_latency_budget(inputs, capsys):
    """``measured 2.23 ms * 1.30 = 2.90 ms``, CUDA events, median of 100."""

    batch, spec, spec_array = inputs
    median = _cuda_time(lambda: run_pipeline(batch, spec, spec_array))
    with capsys.disabled():
        print(
            f"\nfull pipeline: {median:.4f} ms median "
            f"(budget {PIPELINE_LATENCY_BUDGET_MS:.4f} ms, "
            f"{PIPELINE_LATENCY_HEADROOM:.2f}x of {MEASURED_PIPELINE_MS:.2f} ms)"
        )
    assert median <= PIPELINE_LATENCY_BUDGET_MS, (
        median,
        PIPELINE_LATENCY_BUDGET_MS,
    )


def test_the_full_pipeline_meets_the_frozen_peak_memory_budget(inputs, capsys):
    """``measured 1.13 MB * 1.25 = 1.41 MB`` of peak allocation delta."""

    batch, spec, spec_array = inputs
    peak = _peak_mb(lambda: run_pipeline(batch, spec, spec_array))
    with capsys.disabled():
        print(
            f"\nfull pipeline peak delta: {peak:.4f} MB "
            f"(budget {PIPELINE_PEAK_BUDGET_MB:.4f} MB)"
        )
    assert peak <= PIPELINE_PEAK_BUDGET_MB, (peak, PIPELINE_PEAK_BUDGET_MB)


def test_the_ordered_statistic_detector_stays_inside_its_recorded_memory_cost(capsys):
    """The outlier, pinned. It is 222x ``ca_cfar_fast`` and that is the point.

    A pipeline that swaps the detector swaps this in, so the number belongs in
    the budget file rather than in a report: 138 MB for ONE ``[128, 256]`` map,
    which is not a per-beam cost anyone should pay by accident.
    """

    from witwin.radar.processing import ca_cfar_fast, os_cfar

    magnitude = torch.rand((128, 256), device="cuda", dtype=torch.float32)
    ordered = _peak_mb(lambda: os_cfar(magnitude))
    pooled = _peak_mb(lambda: ca_cfar_fast(magnitude))
    with capsys.disabled():
        print(
            f"\nos_cfar peak {ordered:.2f} MB vs ca_cfar_fast {pooled:.2f} MB "
            f"({ordered / max(pooled, 1e-9):.0f}x); budget "
            f"{OS_CFAR_PEAK_BUDGET_MB:.2f} MB"
        )
    assert ordered <= OS_CFAR_PEAK_BUDGET_MB, (ordered, OS_CFAR_PEAK_BUDGET_MB)
    assert ordered > 50.0 * pooled, (ordered, pooled)


def test_the_simulation_frame_cost_has_not_regressed(capsys):
    """Two leg reevaluations plus one composition, wall clock.

    The evidence for "no regression" is in this module's docstring and is a
    measurement at the Phase-8 BASE commit rather than a comparison against a
    number recorded on another machine: base 3.911 ms, HEAD 3.880 ms.
    """

    import time

    from support import multi_endpoint_driver as drv

    pytest.importorskip("witwin.channel")
    spike = drv.MultiEndpointSpike()
    spike.frame()
    for _ in range(10):
        spike.frame()
    torch.cuda.synchronize()
    samples = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        spike.frame()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1.0e3)
    median = statistics.median(samples)
    with capsys.disabled():
        print(
            f"\nsimulation frame: {median:.4f} ms median "
            f"(budget {SIMULATION_FRAME_BUDGET_MS:.4f} ms)"
        )
    assert median <= SIMULATION_FRAME_BUDGET_MS, (
        median,
        SIMULATION_FRAME_BUDGET_MS,
    )


# ---------------------------------------------------------------------------
# The counting pins: device independent, and the ones that catch a regression
# ---------------------------------------------------------------------------


def test_the_pipeline_costs_exactly_one_host_observation_and_seven_transforms(
    inputs, capsys
):
    """Exact integers, attributed to processing.

    Attribution is the point. Processing runs AFTER synthesis, so a
    synchronization here is allowed - but if it is not counted and named, the
    frozen pipeline budget gets blamed on the simulation half.
    """

    batch, spec, spec_array = inputs
    run_pipeline(batch, spec, spec_array)  # resolve every lazy import first
    with DspLedger() as ledger:
        run_pipeline(batch, spec, spec_array)
    with capsys.disabled():
        print(f"\npipeline ledger: {ledger.live()}")
    assert ledger.transform_count == PIPELINE_TRANSFORM_DISPATCHES, ledger.launches
    assert ledger.host_observation_count == PIPELINE_HOST_OBSERVATIONS, ledger.host
    # And the one observation is the point cloud's, not a stray ``.cpu()``.
    assert ledger.host["argwhere"] == 1, ledger.host
    assert ledger.host["item"] == 0, ledger.host
    assert ledger.host["cpu"] == 0, ledger.host
    assert ledger.host["tolist"] == 0, ledger.host
    assert ledger.host["numpy"] == 0, ledger.host
    assert ledger.host["synchronize"] == 0, ledger.host


def test_the_profiling_instrumentation_does_not_change_the_output(inputs):
    """Measuring must not perturb. Bitwise, with the ledger absent and present."""

    batch, spec, spec_array = inputs
    plain = run_pipeline(batch, spec, spec_array)
    with DspLedger():
        instrumented = run_pipeline(batch, spec, spec_array)
    after = run_pipeline(batch, spec, spec_array)
    for reference, candidate in ((plain, instrumented), (plain, after)):
        assert torch.equal(reference.xyz, candidate.xyz)
        assert torch.equal(reference.velocity_mps, candidate.velocity_mps)
        assert torch.equal(reference.energy, candidate.energy)
        assert torch.equal(reference.range_m, candidate.range_m)


# ---------------------------------------------------------------------------
# The wideband budget, flat in the frequency column count
# ---------------------------------------------------------------------------


WIDEBAND_COLUMN_COUNTS = (1, 8, 64)


@pytest.fixture(scope="module")
def narrowband_spike():
    pytest.importorskip("witwin.channel")
    from support import multi_endpoint_driver as drv

    return drv.MultiEndpointSpike()


def _banded_spike(narrow, count: int | None):
    """The same compiled scene, seen through a banded adapter.

    Sharing the compiled scene removes the question of whether a difference came
    from the band or from a recompile: a frequency-only recompile leaves every
    world version domain untouched, so two scenes would still be legal, but one
    scene makes the attribution exact.
    """

    from support import multi_endpoint_driver as drv
    from witwin.radar.propagation.channel_consumer import ChannelPropagationAdapter
    from support import multi_endpoint_geometry as geo

    if count is None:
        return narrow
    offsets = tuple(float(1.0e6 * (index + 1)) for index in range(count))
    adapter = ChannelPropagationAdapter(
        narrow.compiled,
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        components=drv.MULTIPATH_COMPONENTS,
        max_depth=1,
        frequency_offsets_hz=offsets,
    )
    return drv.MultiEndpointSpike(compiled=narrow.compiled, adapter=adapter)


def test_the_wideband_budget_is_flat_in_the_frequency_column_count(
    narrowband_spike, capsys
):
    """ADR-032 at 1 copy and 1 synchronization, independent of ``F``.

    The ``(1 + F) * buckets`` NATIVE launch law is Channel's own and is asserted
    where the launches happen, in
    ``tests/propagation/consumer/test_wideband_offsets.py::
    test_the_launch_count_follows_the_published_law``. What Radar can and must
    assert is the half it owns: the column loop lives inside Channel, so ONE
    consumer call per leg answers a whole band, and the row gather that owns the
    validation copy runs once above that loop.
    """

    from witwin.channel.propagation import consumer

    reported = {}
    for count in (None, *WIDEBAND_COLUMN_COUNTS):
        spike = _banded_spike(narrowband_spike, count)
        calls = {"count": 0}
        original = consumer.reevaluate

        def counting(*args, _original=original, **kwargs):
            calls["count"] += 1
            return _original(*args, **kwargs)

        consumer.reevaluate = counting
        try:
            inbound, outbound = spike.legs()
        finally:
            consumer.reevaluate = original

        for leg in (inbound, outbound):
            diagnostics = leg.diagnostics
            assert diagnostics.validation_d2h_copies == 1, (count, diagnostics)
            assert diagnostics.validation_sync_count == 1, (count, diagnostics)
            assert diagnostics.compact_count_d2h_copies == 0, (count, diagnostics)
            assert diagnostics.compact_sync_count == 0, (count, diagnostics)
            assert diagnostics.frequency_column_count == (count or 1), (
                count,
                diagnostics,
            )
        assert calls["count"] == 2, (count, calls)
        reported[count] = (
            calls["count"],
            inbound.diagnostics.validation_d2h_copies,
            inbound.diagnostics.validation_sync_count,
            inbound.diagnostics.frequency_column_count,
        )

    with capsys.disabled():
        print("\nwideband budget, per leg")
        for count, values in reported.items():
            print(
                f"  F={str(count):<5} reevaluate={values[0]} d2h={values[1]} "
                f"sync={values[2]} columns={values[3]}"
            )


def test_the_two_way_join_costs_one_launch_per_column(narrowband_spike, capsys):
    """``1 + F`` join launches, at ``F`` in {1, 8, 64}, plus the narrowband 1.

    S1 pinned this at one band width. The budget claim is that it is LINEAR in
    the column count with the reference column as the constant - a strided
    native join would change this number, and it needs its own decision record
    and a measurement that beats the recorded F-loop cost.
    """

    import witwin.radar.paths.two_way as two_way
    from support import multi_endpoint_driver as drv
    from witwin.radar.cuda import build

    operators = build.build_extension()
    original = operators.two_way_join_forward
    reported = {}
    for count in (None, *WIDEBAND_COLUMN_COUNTS):
        spike = _banded_spike(narrowband_spike, count)
        launches = {"count": 0}

        def counting(*args, _original=original, **kwargs):
            launches["count"] += 1
            return _original(*args, **kwargs)

        class _Patched:
            def __getattr__(self, name):
                if name == "two_way_join_forward":
                    return counting
                return getattr(operators, name)

        saved = two_way._OPS
        two_way._OPS = _Patched()
        try:
            spike.frame(response=drv.make_response(), include_delay_rate=False)
        finally:
            two_way._OPS = saved
        reported[count] = launches["count"]
        assert launches["count"] == 1 + (count or 0), (count, launches)

    with capsys.disabled():
        print(f"\ntwo-way join launches: {reported}")
