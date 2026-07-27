"""Criterion 7: the BACKWARD budgets - launches, tape bytes, time and memory.

Every millisecond and every megabyte pinned in this tree before Phase 9 is a
forward number. ``test_phase8_pipeline_budget.py`` times a primal pipeline,
``test_phase5_budget.py`` sweeps a primal slot loop, and the one backward
statement anywhere - ``test_phase6_launch_budget.py``'s one-companion-launch
rule - covers three synthesis families and nothing else. A reverse pass that
doubled its launches, retained a per-column tape, or spent ten times the
forward's time would have failed no test in the tree.

Five pins, and no more than five. A pinned wall-time number is maintenance debt
and the cheapest way to make a budget suite useless is to pin everything, so
what is pinned is what is budget critical:

1. the ``_compose_band`` tape as a function of column count, as a PREDICTED
   linear law rather than a total, so it fails on a change to what the join
   saves regardless of the band width used;
2. one backward launch per forward launch, at every autograd boundary in the
   package rather than at three of them;
3. the full FMCW pipeline's backward wall time - the first backward time budget
   in the tree;
4. the same pipeline's backward peak allocation;
5. the Channel ``reevaluate`` inner loop Radar actually runs - its reverse cost
   as a RATIO to its forward cost, plus its ADR-043 companion launches and tape
   bytes by exact value.

**Why one of the five is a ratio and not a wall time.** The absolute medians of
the Channel two-leg call drift over a 1.5x range on this machine between
processes, and the forward and the reverse drift together, so an absolute
budget on either is a pin on the session rather than on the code - measured,
and then measured again, before the constant was replaced. The quotient taken
back-to-back in one process is stable and is the statement worth making anyway:
the reverse pass rides the topology the forward already solved, so it is a
surcharge and cannot approach a second solve. The observed absolute ranges are
recorded beside the constant so the information is not lost.

**Measurement conditions, recorded because a wall-time budget is a statement
about a machine.** NVIDIA GeForce RTX 5080, CUDA events, median of 50 after 10
warm-up calls, the real multi-endpoint Channel fixture, an otherwise idle GPU
(``nvidia-smi`` at 2-3 percent utilization and 4.7-5.2 GB in use by desktop
processes). Each wall-time constant below is the WORST median observed over four
independent processes, and the factor is applied on top of that - so the
headroom is over the worst measurement rather than over the luckiest one.

**The device is cold at the start of a session.** Measured while writing this
file: an idle RTX 5080 sits at 877 MHz and the first ``pytest`` invocation of a
session can miss a wall-time budget by about one percent purely on clock ramp,
with the second and third runs passing comfortably. Twenty warm-up calls do not
boost the clock from idle. That is a property of the device, not of the code,
and the correct response to a single first-run miss is to re-run - never to
widen a factor.
"""

from __future__ import annotations

import statistics

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import ad_boundaries as ab  # noqa: E402
from support import ad_matrix as mx  # noqa: E402
from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import waveform_chains as wc  # noqa: E402

from witwin.radar.propagation.channel_consumer import (  # noqa: E402
    ChannelPropagationAdapter,
)


pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# The measured numbers, and the headroom
# ---------------------------------------------------------------------------

#: Measured median backward of the full FMCW pipeline - replay both frozen
#: legs, compose, synthesize, two-term scalar loss, ``backward()`` - in ms.
#: Four independent process medians: 1.816, 1.925, 2.153, 2.684. The constant is
#: the worst.
MEASURED_PIPELINE_BACKWARD_MS = 2.68
BACKWARD_TIME_HEADROOM = 1.30
PIPELINE_BACKWARD_BUDGET_MS = MEASURED_PIPELINE_BACKWARD_MS * BACKWARD_TIME_HEADROOM

#: Measured peak ALLOCATION DELTA over the whole forward-plus-backward of that
#: pipeline, in MB. Deterministic: exactly 149504 bytes on every one of four
#: independent runs, because the allocator replays the same sequence of exact
#: sizes. The forward alone peaks at 43008 bytes, so the reverse pass costs
#: 3.5x the forward's peak and that ratio is the number worth watching.
MEASURED_PIPELINE_BACKWARD_MB = 149504 / (1024.0 * 1024.0)
BACKWARD_MEMORY_HEADROOM = 1.25
PIPELINE_BACKWARD_PEAK_BUDGET_MB = (
    MEASURED_PIPELINE_BACKWARD_MB * BACKWARD_MEMORY_HEADROOM
)

#: The Channel ``reevaluate`` inner loop is pinned as a RATIO rather than as two
#: wall times, and that is a measurement result rather than a preference.
#:
#: Absolute medians of the two-leg call drift far more than the reverse
#: surcharge does. Twelve samples in three processes on an idle device:
#: forward 3.636 to 5.542 ms, forward-plus-backward 5.326 to 7.071 ms. Both
#: quantities drift together within a process - allocator state and clock ramp
#: move them the same way - so an absolute budget on either one is a pin on the
#: session and not on the code, and one set at the tightest observation fails
#: on the first cold run of a session. Measured, twice, before writing this.
#:
#: The RATIO, sampled ALTERNATELY in one loop, runs 1.334 to 1.523 over six
#: independent processes. Taking the two medians one after the other instead
#: gives 1.342 to 1.841 - measured, and the reason the sampling is interleaved
#: rather than sequential.
#:
#: The ratio is also the statement worth making: a reverse pass costs a
#: surcharge on the forward, and a backward that started re-solving geometry
#: could not come in under 2x. The budget is 2.0 for exactly that reason - it
#: is a structural threshold, not a measured number with a factor bolted on.
MEASURED_REEVALUATE_VJP_RATIO_RANGE = (1.334, 1.523)
REEVALUATE_VJP_RATIO_BUDGET = 2.0

#: Recorded for the ledger, not asserted. See above for why.
OBSERVED_REEVALUATE_FORWARD_MS_RANGE = (3.636, 5.542)
OBSERVED_REEVALUATE_VJP_MS_RANGE = (5.326, 7.071)

#: ADR-043 accounting at the pinned fixture, per leg, under ``ad_mode='vjp'``.
#: Two companion launches each; the tape differs between the legs because the
#: inbound one is line-of-sight only and the outbound one carries reflections.
REEVALUATE_AD_LAUNCHES_PER_LEG = 2
REEVALUATE_INBOUND_TAPE_BYTES = 200
REEVALUATE_OUTBOUND_TAPE_BYTES = 496

#: The wideband band the tape law is measured over.
SUBCARRIER_SPACING_HZ = 25.0e6
BAND_WIDTHS = (1, 2, 4, 8)


def _cuda_ms(fn, *, warmup: int = 10, runs: int = 50) -> float:
    """``test_phase8_pipeline_budget.py``'s convention, not a second one."""

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


def _cuda_ms_paired(first, second, *, warmup: int = 10, runs: int = 50):
    """Two medians, sampled ALTERNATELY in one loop.

    Timing them one after the other leaves the second measurement on a
    different clock and allocator state than the first, and on this device that
    drift is larger than the difference being measured. Interleaving puts both
    quantities under the same drift, so their quotient is a property of the two
    calls rather than of when each one happened to run.
    """

    for _ in range(warmup):
        first()
        second()
    torch.cuda.synchronize()
    left, right = [], []
    for _ in range(runs):
        for call, samples in ((first, left), (second, right)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            torch.cuda.synchronize()
            samples.append(float(start.elapsed_time(end)))
    return statistics.median(left), statistics.median(right)


# ---------------------------------------------------------------------------
# 1. The band loop's tape law
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def narrow():
    return drv.MultiEndpointSpike()


def _banded(narrow, columns: int):
    adapter = ChannelPropagationAdapter(
        narrow.compiled,
        reference_frequency_hz=geo.REFERENCE_FREQUENCY_HZ,
        components=drv.MULTIPATH_COMPONENTS,
        max_depth=1,
        frequency_offsets_hz=tuple(
            float(index * SUBCARRIER_SPACING_HZ) for index in range(columns)
        ),
    )
    return drv.MultiEndpointSpike(compiled=narrow.compiled, adapter=adapter)


def _band_tape(spike, columns: int):
    """``(contexts, total bytes, distinct-storage bytes)`` for one banded frame."""

    from witwin.radar.paths import two_way

    saved = []
    original = two_way._TwoWayJoin.setup_context

    def recording(ctx, inputs, output):
        original(ctx, inputs, output)
        # Read the identities NOW: the storage is released with the graph, so a
        # pointer collected afterwards would be a use-after-free dressed up as a
        # measurement.
        saved.append(
            tuple(
                (tensor.data_ptr(), tensor.numel() * tensor.element_size())
                for tensor in ctx.to_save
            )
        )

    two_way._TwoWayJoin.setup_context = staticmethod(recording)
    try:
        sites = spike.site_tensor(requires_grad=True)
        composed, _, _ = spike.frame(
            sites, drv.make_response(), ad_mode="vjp", include_delay_rate=False
        )
        assert composed.frequency_response.shape[1] == columns
    finally:
        two_way._TwoWayJoin.setup_context = staticmethod(original)

    total = sum(size for context in saved for _, size in context)
    distinct = {}
    for context in saved:
        for pointer, size in context:
            distinct[pointer] = size
    return len(saved), total, sum(distinct.values())


def test_the_band_loop_tape_obeys_its_predicted_linear_law(narrow):
    """Tape bytes as a LAW in the column count, not as one total.

    ``_compose_band`` calls ``_TwoWayJoin.apply`` once per frequency column plus
    once for the reference column, and every call retains its own ten-tensor
    context. The survey read that as "a 64-subcarrier band holds 64 copies of
    the join tape", which is an exact statement about CONTEXTS and a five-fold
    overestimate of the MEMORY: six of the ten saved tensors are the same
    storage in every context, because the response is evaluated once above the
    loop and the join's index tables are frozen. So the marginal retained tape
    per column is the four coefficient slices and nothing else.

    Both halves are predicted from the fixture's own row counts rather than
    written down as measured constants, so this fails when the join changes WHAT
    it saves at any band width, and does not have to be re-measured when the
    fixture changes SIZE.
    """

    rows_in = narrow.composer.inbound_row_count
    rows_out = narrow.composer.outbound_row_count
    sites = narrow.composer.site_count
    paths = narrow.composer.path_count

    #: float32 per leg coefficient component (4 tensors of 4 B), plus the
    #: response pair, plus int32 ``row_valid`` and three int64 index tables.
    per_context = 8 * rows_in + 8 * rows_out + 8 * sites + 28 * paths
    shared = 8 * sites + 28 * paths
    per_column = 8 * (rows_in + rows_out)

    for columns in BAND_WIDTHS:
        contexts, total, distinct = _band_tape(_banded(narrow, columns), columns)
        assert contexts == columns + 1, (columns, contexts)
        assert total == contexts * per_context, (columns, total, per_context)
        assert distinct == shared + per_column * contexts, (columns, distinct)

    # And the law is not vacuous: the two slopes must differ, or the aliasing
    # this test exists to bound would be doing nothing.
    assert 0 < per_column < per_context


def test_the_band_loop_tape_law_holds_at_a_width_it_was_not_fitted_on(narrow):
    """The prediction, checked once at a width outside the fitted set.

    A law fitted on 1, 2, 4 and 8 and never evaluated anywhere else is a
    restatement of four measurements. Sixteen is the falsifier.
    """

    rows_in = narrow.composer.inbound_row_count
    rows_out = narrow.composer.outbound_row_count
    columns = 16
    contexts, total, distinct = _band_tape(_banded(narrow, columns), columns)
    assert contexts == columns + 1
    expected_distinct = (
        8 * narrow.composer.site_count
        + 28 * narrow.composer.path_count
        + 8 * (rows_in + rows_out) * contexts
    )
    assert distinct == expected_distinct, (distinct, expected_distinct)
    assert total == contexts * (
        8 * rows_in + 8 * rows_out + 8 * narrow.composer.site_count
        + 28 * narrow.composer.path_count
    )


# ---------------------------------------------------------------------------
# 2. One backward launch per forward launch, at every boundary
# ---------------------------------------------------------------------------


#: Every operator any of the nine boundaries can launch, with the family it
#: belongs to. A boundary is allowed exactly one forward and one backward from
#: its own family and nothing at all from another's.
BOUNDARY_OPERATORS = {
    "two_way": ("two_way_join_forward", "two_way_join_backward", "two_way_join_jvp"),
    "aspect": (
        "scatter_response_aspect_forward",
        "scatter_response_aspect_backward",
        "scatter_response_aspect_jvp",
    ),
    "fmcw": ("fmcw_beat_forward", "fmcw_beat_backward", "fmcw_beat_jvp"),
    "ofdm": ("ofdm_cfr_forward", "ofdm_cfr_backward", "ofdm_cfr_jvp"),
    "pulsed": ("pulsed_echo_forward", "pulsed_echo_backward", "pulsed_echo_jvp"),
    "dirichlet": ("forward_chunked", "backward_batched", "dirichlet_jvp"),
    "mimo_linear": (
        "forward_mimo_linear_chunked",
        "mimo_linear_backward",
        "mimo_linear_jvp",
    ),
    "sensor_weight": (
        "sensor_weight_forward",
        "sensor_weight_backward",
        "sensor_weight_jvp",
    ),
}


class _Launches:
    """Count launches of the named operators while active."""

    def __init__(self, operators, names) -> None:
        self.counts = dict.fromkeys(names, 0)
        self._operators = operators
        self._saved = []

    def __enter__(self):
        for name in self.counts:
            original = getattr(self._operators, name)
            self._saved.append((name, original))

            def counting(*args, _name=name, _original=original, **kwargs):
                self.counts[_name] += 1
                return _original(*args, **kwargs)

            setattr(self._operators, name, counting)
        return self

    def __exit__(self, *exc):
        for name, original in self._saved:
            setattr(self._operators, name, original)
        return False


def _operators():
    from witwin.radar.cuda import build

    return build.build_extension()


@pytest.mark.parametrize("name", sorted(BOUNDARY_OPERATORS))
def test_each_boundary_costs_one_backward_launch_per_forward_launch(name):
    """R-ADR-004's shape, at EVERY boundary rather than at three of them.

    ``test_phase6_launch_budget.py`` makes this statement for the three
    synthesis families it introduced. The two-way join, the aspect response, the
    two Dirichlet variants and the sensor weight owner had no backward launch
    budget at all, and a companion that launched twice - a separate reduction
    pass, a second kernel for a gradient family - would have been invisible.
    """

    operators = _operators()
    forward, backward, jvp = BOUNDARY_OPERATORS[name]
    boundary = ab.boundary(name)
    leaf = boundary.leaf.detach().clone().requires_grad_(True)

    with _Launches(operators, (forward, backward, jvp)) as ledger:
        loss = boundary.loss(leaf)
        loss.backward()

    assert ledger.counts[forward] == 1, ledger.counts
    assert ledger.counts[backward] == 1, ledger.counts
    assert ledger.counts[jvp] == 0, ledger.counts


def test_the_frontend_costs_one_backward_launch_per_forward_stage():
    """The frontend is two owners in one call, so it gets its own statement."""

    operators = _operators()
    names = (
        "frontend_noise_forward",
        "frontend_noise_backward",
        "frontend_agc_forward",
        "frontend_agc_backward",
        "frontend_quantize_forward",
    )
    boundary = ab.boundary("frontend")
    leaf = boundary.leaf.detach().clone().requires_grad_(True)
    with _Launches(operators, names) as ledger:
        boundary.loss(leaf).backward()
    assert ledger.counts["frontend_noise_forward"] == 1, ledger.counts
    assert ledger.counts["frontend_noise_backward"] == 1, ledger.counts
    assert ledger.counts["frontend_agc_forward"] == 1, ledger.counts
    assert ledger.counts["frontend_agc_backward"] == 1, ledger.counts
    # The quantizer is on the far side of the wall and this chain has no ADC.
    assert ledger.counts["frontend_quantize_forward"] == 0, ledger.counts


def test_the_launch_ledger_covers_every_tape_owner_in_the_package():
    """The set of budgeted boundaries equals the set of autograd owners.

    Without this, adding a tenth ``Function`` would add an unbudgeted backward
    and every test above would still pass.
    """

    budgeted = set(BOUNDARY_OPERATORS) | {"frontend"}
    assert budgeted == set(ab.BOUNDARY_NAMES), budgeted ^ set(ab.BOUNDARY_NAMES)


# ---------------------------------------------------------------------------
# 3. The full FMCW pipeline, backward
# ---------------------------------------------------------------------------


#: The five per-frame leaves. The three scene leaves are inside the compile and
#: a per-frame replay cannot reach them, which is what makes this the INNER LOOP
#: rather than a whole optimisation step.
PIPELINE_LEAVES = ("sites", "transmitters", "receivers", "sigma_m2", "phase_rad")


@pytest.fixture(scope="module")
def pipeline():
    spike = mx.build_spike(mx.base_values(drv.MultiEndpointSpike()))
    return spike, mx.base_values(spike)


def _pipeline_loss(spike, values):
    live = mx.marked(values, PIPELINE_LEAVES)
    composed = mx.replay(spike, live, ad_mode="vjp")
    cube = wc.synthesize("fmcw", composed, wc.make_spec("fmcw"))
    return mx.combined_loss(cube)


def test_the_full_fmcw_pipeline_backward_meets_its_time_budget(pipeline, capsys):
    """The tree's first backward wall-time budget.

    The graph is built OUTSIDE the timed region on every sample, so the number
    is the reverse pass and not the forward plus the reverse. See the module
    docstring for the measurement conditions and for the cold-clock caveat.
    """

    spike, values = pipeline

    def timed() -> float:
        loss = _pipeline_loss(spike, values)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))

    for _ in range(10):
        timed()
    median = statistics.median(timed() for _ in range(50))
    with capsys.disabled():
        print(
            f"\nfmcw pipeline backward: {median:.4f} ms "
            f"(budget {PIPELINE_BACKWARD_BUDGET_MS:.4f} ms, "
            f"{BACKWARD_TIME_HEADROOM:.2f}x of {MEASURED_PIPELINE_BACKWARD_MS} ms)"
        )
    assert median < PIPELINE_BACKWARD_BUDGET_MS, (
        f"{median:.4f} ms exceeds {PIPELINE_BACKWARD_BUDGET_MS:.4f} ms"
    )


def test_the_full_fmcw_pipeline_backward_meets_its_peak_memory_budget(
    pipeline, capsys
):
    """Peak ALLOCATION over forward plus backward, which is what a tape costs.

    Allocation rather than reservation: reserved memory is an allocator
    decision and depends on what ran before, while the allocated peak is the
    same sequence of exact sizes on every run and is therefore a statement about
    the code.
    """

    spike, values = pipeline
    _pipeline_loss(spike, values).backward()  # warm the allocator's pools

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    _pipeline_loss(spike, values).backward()
    torch.cuda.synchronize()
    peak_mb = (torch.cuda.max_memory_allocated() - before) / (1024.0 * 1024.0)

    with capsys.disabled():
        print(
            f"\nfmcw pipeline backward peak: {peak_mb:.4f} MB "
            f"(budget {PIPELINE_BACKWARD_PEAK_BUDGET_MB:.4f} MB)"
        )
    assert peak_mb < PIPELINE_BACKWARD_PEAK_BUDGET_MB, peak_mb


def test_the_backward_peak_is_larger_than_the_forward_peak(pipeline):
    """Calibration for the pin above, and a fact worth pinning on its own.

    A memory budget that a forward-only run would also satisfy is not a backward
    budget. The reverse pass costs about 3.5x the forward's peak here, and a
    change that made the two equal would mean the tape stopped being retained -
    which is either a leak of correctness or a very interesting optimisation,
    and either way not something to discover by accident.
    """

    spike, values = pipeline

    def peak(build) -> float:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        build()
        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated() - before)

    forward = peak(
        lambda: wc.synthesize(
            "fmcw", mx.replay(spike, values, ad_mode="none"), wc.make_spec("fmcw")
        )
    )
    reverse = peak(lambda: _pipeline_loss(spike, values).backward())
    assert reverse > 2.0 * forward, (reverse, forward)


# ---------------------------------------------------------------------------
# 4. The Channel inner loop, as Radar runs it
# ---------------------------------------------------------------------------


def _legs(spike, values, *, ad_mode: str, leaf=None):
    sites = values["sites"] if leaf is None else leaf
    return spike.legs(
        sites,
        transmitters=values["transmitters"],
        receivers=values["receivers"],
        ad_mode=ad_mode,
    )


def test_the_channel_reevaluate_reverse_pass_is_a_surcharge_not_a_second_solve(
    pipeline, capsys
):
    """The Channel inner loop's reverse cost, as a ratio taken in one process.

    Measured on the Radar side rather than in Channel's own
    ``tests/ad/test_ad_budgets.py``, which covers the deterministic solver and
    not the fixed-topology route Radar's per-frame loop actually runs.

    The two medians are taken back-to-back in the same process, so the clock
    and allocator drift that makes an absolute pin unreliable here cancels out
    of the quotient. The claim is structural: the reverse pass rides the frozen
    topology the forward already solved, so it is a surcharge. A backward that
    re-solved geometry would be at least a second forward and could not come in
    under 2x.
    """

    spike, values = pipeline

    def forward():
        _legs(spike, values, ad_mode="none")

    def reverse():
        sites = values["sites"].clone().requires_grad_(True)
        inbound, outbound = _legs(spike, values, ad_mode="vjp", leaf=sites)
        loss = (
            inbound.delay_s.sum()
            + outbound.delay_s.sum()
            + inbound.coefficient.abs().sum()
            + outbound.coefficient.abs().sum()
        )
        loss.backward()

    forward_ms, reverse_ms = _cuda_ms_paired(forward, reverse)
    ratio = reverse_ms / forward_ms
    with capsys.disabled():
        print(
            f"\nchannel reevaluate: forward {forward_ms:.4f} ms, "
            f"forward+backward {reverse_ms:.4f} ms, ratio {ratio:.4f} "
            f"(budget {REEVALUATE_VJP_RATIO_BUDGET:.2f}, "
            f"measured range {MEASURED_REEVALUATE_VJP_RATIO_RANGE})"
        )
    assert ratio > 1.0, (
        f"ratio {ratio:.4f}: a reverse pass that cost no more than the forward "
        "would mean the backward is not running"
    )
    assert ratio < REEVALUATE_VJP_RATIO_BUDGET, (
        f"ratio {ratio:.4f} exceeds {REEVALUATE_VJP_RATIO_BUDGET}: the reverse "
        "pass is no longer a surcharge on the frozen topology"
    )


def test_the_channel_reevaluate_publishes_its_ad_launches_and_tape_bytes(pipeline):
    """ADR-043's accounting, read where Radar receives it.

    ``PropagationDiagnostics`` gained ``ad_companion_launches`` and
    ``ad_tape_bytes`` at ``CONTRACT_VERSION`` 6 specifically so this ledger
    could exist. Pinned by exact value at a pinned fixture: an inexact
    assertion here would accept a companion that started launching twice.
    """

    spike, values = pipeline
    sites = values["sites"].clone().requires_grad_(True)
    inbound, outbound = _legs(spike, values, ad_mode="vjp", leaf=sites)

    assert inbound.diagnostics.ad_companion_launches == REEVALUATE_AD_LAUNCHES_PER_LEG
    assert outbound.diagnostics.ad_companion_launches == REEVALUATE_AD_LAUNCHES_PER_LEG
    assert inbound.diagnostics.ad_tape_bytes == REEVALUATE_INBOUND_TAPE_BYTES
    assert outbound.diagnostics.ad_tape_bytes == REEVALUATE_OUTBOUND_TAPE_BYTES
    # The two legs must not report the same tape: the inbound leg is
    # line-of-sight only and the outbound one carries reflections, so an equal
    # pair would mean the ledger is reporting a constant.
    assert inbound.diagnostics.ad_tape_bytes != outbound.diagnostics.ad_tape_bytes


def test_a_primal_only_reevaluate_builds_no_tape_at_all(pipeline):
    """The vjp-only tape gate, reproduced at the consumer boundary.

    A ledger that forwarded the raw sidecar number would report a tape for a
    primal solve, which contradicts the ledger's own contract. Zero is the
    complete answer here, and it is asserted as EXACT zero rather than as small.
    """

    spike, values = pipeline
    inbound, outbound = _legs(spike, values, ad_mode="none")
    for leg in (inbound, outbound):
        assert leg.diagnostics.ad_companion_launches == 0
        assert leg.diagnostics.ad_tape_bytes == 0
