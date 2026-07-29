"""Target and clutter, exported independently and recombined coherently.

Plan Phase-8 work item 2, from the consuming side. The claim under test is that
a radar frame's target return and its environment return can be separated,
looked at on their own, and put back together WITHOUT the separation touching
propagation row identity or the AD contract - because the separation is a mask
on the synthesis weight over one frozen topology, not a second discovery, a
second join mode, or a second kernel.

Four things this file establishes, in the order they matter:

1. the taxonomy is a PARTITION of the composed rows - exact integer counts, no
   row unclassified and no row counted twice;
2. every component export shares the same ``RadarPathTopology`` OBJECT, which is
   the direct expression of the acceptance criterion;
3. the per-component cubes sum to the full cube to a stated float tolerance,
   with the measured residual recorded here rather than in a report;
4. a masked row is inert in the primal and carries no gradient, which is what
   makes an export a statement about the scene rather than a post-hoc edit of
   the answer.

Everything runs against REAL Channel rows from the multi-endpoint fixture: 11
composed round trips over four sensor pairs, two of which are empty, four
line-of-sight target rows and seven rows that touch the wall.
"""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("witwin.channel")

from support import clutter_scenes as cs  # noqa: E402
from support import multi_endpoint_driver as drv  # noqa: E402
from support import multi_endpoint_geometry as geo  # noqa: E402
from support import reference_chain as ref  # noqa: E402
from support.synthesis_batch import to_synthesis  # noqa: E402

from witwin.radar.paths import (  # noqa: E402
    COMPONENT_NAMES,
    DIRECT_LEAKAGE,
    ENVIRONMENT_CLUTTER,
    MULTI_INTERACTION,
    TARGET,
    ComponentDeclaration,
    RadarComponentIndex,
)
from witwin.radar.processing import combine_incoherent  # noqa: E402
from witwin.radar.synthesis import select_component, synthesize_fmcw  # noqa: E402

pytestmark = pytest.mark.gpu

#: The fixture's partition, written down rather than derived from the code
#: under test. Four line-of-sight round trips through the two sites, seven that
#: bounce off the wall on one leg or both, no direct route in a two-way join and
#: nothing deeper than one interaction per leg.
EXPECTED_COUNTS = {TARGET: 4, ENVIRONMENT_CLUTTER: 7, DIRECT_LEAKAGE: 0, MULTI_INTERACTION: 0}


@pytest.fixture(scope="module")
def spike():
    return drv.MultiEndpointSpike()


@pytest.fixture(scope="module")
def index(spike):
    return cs.component_index(spike)


def _cube(batch, spec):
    return synthesize_fmcw(batch, spec)


def _component_cubes(batch, index, spec):
    return [_cube(select_component(batch, index, name), spec) for name in index.names]


# --------------------------------------------------------------------------
# 1. The partition
# --------------------------------------------------------------------------


def test_the_component_masks_are_a_partition_of_every_composed_row(spike, index):
    """Disjoint, complete, and the counts are the ones the geometry predicts.

    Exact integers, not a tolerance. The union assertion is what makes the
    coherent recombination law below meaningful: a row that belonged to no
    class would vanish from the sum and the residual would be a whole row
    rather than a rounding artefact.
    """

    assert index.names == COMPONENT_NAMES
    assert index.row_count == spike.composer.path_count == 11

    masks = {name: index.mask(name) for name in index.names}
    for name, expected in EXPECTED_COUNTS.items():
        assert index.count(name) == expected, name
        assert int(masks[name].sum()) == expected, name

    stacked = torch.stack([masks[name] for name in index.names])
    per_row = stacked.to(torch.int32).sum(dim=0)
    assert torch.equal(per_row, torch.ones_like(per_row))

    # And the classification is the one the fixture's own row keys say it is,
    # cross-checked through the geometry oracle rather than through class_id.
    composed, _, _ = spike.frame()
    keys = drv.composed_keys(spike, composed)
    for row, key in enumerate(keys):
        touches_wall = "reflection" in key[3:]
        expected = ENVIRONMENT_CLUTTER if touches_wall else TARGET
        assert index.names[int(index.class_id[row])] == expected, (row, key)


def test_a_site_left_out_of_the_declaration_is_a_loud_error(spike):
    """An unclassified row is refused at build, never silently dropped.

    This is the failure the completeness assertion exists for: declaring only
    site P leaves every site-Q round trip belonging to no class, and a build
    that shrugged would produce exports whose coherent sum quietly omitted
    them.
    """

    with pytest.raises(ValueError, match="exactly one component class"):
        cs.component_index(
            spike,
            ComponentDeclaration(
                target_site_ids={geo.SITE_P_STABLE_ID}, clutter_material_slots={geo.REFLECTION_MATERIAL_SLOT}
            ),
        )


def test_a_site_declared_both_target_and_clutter_is_refused(spike):
    """The declaration contradicts itself before it can classify anything."""

    with pytest.raises(ValueError, match="both target and clutter"):
        ComponentDeclaration(target_site_ids={geo.SITE_P_STABLE_ID}, clutter_site_ids={geo.SITE_P_STABLE_ID})


def test_the_direct_route_separates_leakage_from_a_wall_return(spike):
    """``direct_leakage`` is not merely "no scatter site".

    Four direct rows on this world: three lines of sight and one
    transmitter-to-wall-to-receiver reflection. The reflection has no site
    either, and filing it as leakage would put the strongest environment return
    in the frame under the name of the antenna coupling term.
    """

    composer, leg, index = cs.direct_route(spike)
    assert composer.path_count == 4
    assert index.count(DIRECT_LEAKAGE) == 3
    assert index.count(ENVIRONMENT_CLUTTER) == 1
    assert index.count(TARGET) == 0

    wall_row = int(torch.nonzero(index.mask(ENVIRONMENT_CLUTTER))[0])
    frozen_row = int(composer.topology.inbound_row[wall_row])
    assert int(leg.component_id[frozen_row]) == geo.REFLECTION_COMPONENT_ID
    assert int(leg.material_sequence[frozen_row][0]) == geo.REFLECTION_MATERIAL_SLOT


def test_a_deeper_leg_becomes_its_own_class(spike, index):
    """The Phase-5 ``hybrid`` third class, expressed as a depth declaration.

    Lowering ``multi_interaction_depth`` to zero reclassifies every
    single-bounce row - the whole of this fixture's clutter - as
    multi-interaction, which is the same rows under a different declaration and
    proves the depth predicate is live rather than dead configuration. The join
    mode is untouched and stays ``multipath``.
    """

    deep = cs.component_index(
        spike,
        ComponentDeclaration(
            target_site_ids={geo.SITE_P_STABLE_ID, geo.SITE_Q_STABLE_ID},
            clutter_material_slots={geo.REFLECTION_MATERIAL_SLOT},
            multi_interaction_depth=0,
        ),
    )
    assert deep.count(MULTI_INTERACTION) == EXPECTED_COUNTS[ENVIRONMENT_CLUTTER]
    assert deep.count(ENVIRONMENT_CLUTTER) == 0
    assert deep.count(TARGET) == EXPECTED_COUNTS[TARGET]
    composed, _, _ = spike.frame()
    assert composed.join_mode == "multipath"


# --------------------------------------------------------------------------
# 2. Row identity
# --------------------------------------------------------------------------


def test_every_component_export_shares_one_topology_object(spike, index):
    """The acceptance criterion, asserted with ``is``.

    Object identity, not equality. A rebuilt-but-equal topology would pass an
    elementwise comparison and would still mean that the export layer had
    respecified row identity, which is exactly what the criterion forbids.
    """

    composed, _, _ = spike.frame()
    batch = to_synthesis(composed)
    assert batch.topology is composed.topology
    assert index.topology is composed.topology
    for name in index.names:
        selected = select_component(batch, index, name)
        assert selected.topology is composed.topology
        assert selected.path_count == batch.path_count
        assert selected.sensor_pair_index is batch.sensor_pair_index
        assert selected.pair_offsets is batch.pair_offsets
        assert selected.total_delay_s is batch.total_delay_s
        assert selected.row_valid is batch.row_valid
        assert selected.join_mode == batch.join_mode
        assert selected.complex_transfer_ref.dtype == torch.complex64
        assert selected.complex_transfer_ref.device == batch.complex_transfer_ref.device


def test_an_index_from_another_topology_is_refused(spike, index):
    """Two topologies of the same length is the commonest way to mask wrongly."""

    composed, _, _ = spike.frame()
    batch = to_synthesis(composed)
    other = RadarComponentIndex(
        topology=spike.composer.topology.__class__(
            radar_source_id=composed.topology.radar_source_id.clone(),
            site_id=composed.topology.site_id.clone(),
            radar_sink_id=composed.topology.radar_sink_id.clone(),
            inbound_row=composed.topology.inbound_row.clone(),
            outbound_row=composed.topology.outbound_row.clone(),
        ),
        class_id=index.class_id,
        names=index.names,
        counts=index.counts,
        declaration=index.declaration,
    )
    with pytest.raises(ValueError, match="different topology object"):
        select_component(batch, other, TARGET)


def test_composing_is_unaffected_by_building_an_index(spike, index):
    """The sidecar observes; it does not participate.

    Building the index must not perturb the composed frame in any way, which is
    a real risk for anything that reads a frozen handle. Bitwise equality,
    because nothing between the two calls is allowed to be numerical.
    """

    first, _, _ = spike.frame()
    cs.component_index(spike)
    second, _, _ = spike.frame()
    assert torch.equal(first.complex_transfer_ref, second.complex_transfer_ref)
    assert torch.equal(first.total_delay_s, second.total_delay_s)
    assert first.topology is second.topology


# --------------------------------------------------------------------------
# 3. Coherent recombination
# --------------------------------------------------------------------------


def test_the_component_cubes_sum_to_the_full_cube(spike, index):
    """``sum_j cube(component_j) == cube(all rows)``, to a derived tolerance.

    NOT ``torch.equal``. The kernel accumulates over the rows of a pair segment
    in the same order in every selection, and a masked row contributes a literal
    ``0.0`` in its own slot - but ``(a + 0 + c) + (0 + b + 0)`` is not
    ``(a + b + c)`` in float32, so the law holds up to re-association of the
    partial sums and no further.

    The tolerance is derived, not tuned: ``8 * eps_f32 * K * max|per-row
    contribution|`` bounds the accumulated re-association error over ``K`` rows,
    and the measured residual is printed so the margin is a number rather than
    a claim.
    """

    composed, _, _ = spike.frame()
    batch = to_synthesis(composed)
    spec = drv.make_spec()

    full = _cube(batch, spec)
    parts = _component_cubes(batch, index, spec)
    total = parts[0]
    for part in parts[1:]:
        total = total + part

    eps = float(torch.finfo(torch.float32).eps)
    per_row = float(batch.complex_transfer_ref.abs().max())
    atol = 8.0 * eps * batch.path_count * per_row
    residual = float((total - full).abs().max())
    scale = float(full.abs().max())
    print(
        f"\ncoherent recombination residual: {residual:.6e} absolute, "
        f"{residual / scale:.3e} relative, against atol {atol:.6e}"
    )
    assert residual <= atol, (residual, atol)
    assert torch.allclose(total, full, rtol=1.0e-6, atol=atol)
    # Non-vacuity: the components really are different frames, so the sum is
    # doing work rather than adding zeros to the answer.
    assert float((parts[0] - full).abs().max()) > 0.1 * scale
    assert float(parts[1].abs().max()) > 0.1 * scale


def test_a_component_export_costs_one_synthesis_and_no_host_observation(spike, index, monkeypatch):
    """What the separation actually costs, measured.

        Two claims. The first is a contract: masking a weight and launching the
        waveform kernel again reads NO device value, so exporting components adds
        nothing to the frame's host-observation budget no matter how many classes
        are declared. The second is a number and is reported rather than asserted,
        because it is a property of this machine: exporting `
    `` components costs
        `
    `` synthesis launches against the one an unseparated frame pays, and the
        measured multiplier is what the pipeline budget has to absorb.
    """

    import time

    composed, _, _ = spike.frame()
    batch = to_synthesis(composed)
    spec = drv.make_spec()
    _cube(batch, spec)  # warm the operator table

    counts = dict.fromkeys(("item", "cpu", "tolist", "numpy"), 0)
    for name in counts:
        original = getattr(torch.Tensor, name)

        def observing(tensor, *args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, observing)
    syncs = {"count": 0}
    original_sync = torch.cuda.synchronize

    def counting_sync(*args, **kwargs):
        syncs["count"] += 1
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", counting_sync)

    _component_cubes(batch, index, spec)
    assert counts == dict.fromkeys(counts, 0), counts
    assert syncs["count"] == 0

    monkeypatch.undo()

    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1.0e3 / 20.0

    populated = [name for name in index.names if index.count(name)]
    whole_ms = timed(lambda: _cube(batch, spec))
    parts_ms = timed(lambda: _component_cubes(batch, index, spec))
    two_ms = timed(lambda: [_cube(select_component(batch, index, name), spec) for name in populated])
    print(
        f"\nsynthesis: {whole_ms:.4f} ms unseparated; "
        f"{two_ms:.4f} ms for the {len(populated)} populated components "
        f"({two_ms / whole_ms:.2f}x); "
        f"{parts_ms:.4f} ms for all {len(index.names)} "
        f"({parts_ms / whole_ms:.2f}x)"
    )
    # An empty component is not free: it still launches and still allocates a
    # full-size cube of zeros, which is why a caller that only wants target and
    # clutter should export those two rather than iterate the whole taxonomy.
    assert len(populated) < len(index.names)
    empty = _cube(select_component(batch, index, DIRECT_LEAKAGE), spec)
    assert empty.shape == _cube(batch, spec).shape
    assert torch.equal(empty, torch.zeros_like(empty))


def test_the_recombination_is_not_bitwise_and_that_is_expected(spike, index):
    """The measurement behind the tolerance, stated as its own claim.

    If this ever becomes bitwise the tolerance above is over-generous and
    should be tightened deliberately, so the fact is asserted rather than
    assumed. It is a property of float32 re-association, not of the masking.
    """

    composed, _, _ = spike.frame()
    batch = to_synthesis(composed)
    spec = drv.make_spec()
    full = _cube(batch, spec)
    parts = _component_cubes(batch, index, spec)
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    assert not torch.equal(total, full)


# --------------------------------------------------------------------------
# 4. A masked row is inert
# --------------------------------------------------------------------------


def test_a_masked_row_contributes_exactly_zero_and_no_gradient(spike, index):
    """Inert in the primal AND on the tape, which are two separate claims.

    The weight is masked with ``torch.where`` BEFORE the kernel, exactly as a
    dead row already is, so a masked row's contribution is a literal ``0.0``
    and its gradient path is severed. Zeroing the cube afterwards would leave a
    live gradient running back through a row the export says is not there.
    """

    response = drv.make_response(requires_grad=True)
    composed, _, _ = spike.frame(response=response)
    batch = to_synthesis(composed)
    spec = drv.make_spec()

    target_only = select_component(batch, index, TARGET)
    mask = index.mask(TARGET)
    weight = target_only.complex_transfer_ref
    assert torch.equal(weight[~mask], torch.zeros_like(weight[~mask]))
    assert torch.equal(weight[mask], batch.complex_transfer_ref[mask])

    # The clutter rows carry the target response too, so a live gradient there
    # would show up in the amplitude's grad. Masking them out must remove
    # exactly their share and nothing else.
    def amplitude_grad(selected):
        response.amplitude.grad = None
        cube = _cube(selected, spec)
        loss = cube.real.square().sum() + cube.imag.square().sum()
        # The composed batch is one graph shared by every selection, so each
        # backward has to keep it alive for the next.
        loss.backward(retain_graph=True)
        return float(response.amplitude.grad)

    whole = amplitude_grad(batch)
    part = amplitude_grad(target_only)
    assert whole != 0.0 and part != 0.0
    assert abs(part) < abs(whole)

    # And a component with NO rows carries no gradient at all: exactly zero,
    # not merely small. `direct_leakage` is empty in a two-way join.
    empty = select_component(batch, index, DIRECT_LEAKAGE)
    assert index.count(DIRECT_LEAKAGE) == 0
    empty_cube = _cube(empty, spec)
    assert torch.equal(empty_cube, torch.zeros_like(empty_cube))
    assert amplitude_grad(empty) == 0.0


# --------------------------------------------------------------------------
# 5. The analytic target-only cube
# --------------------------------------------------------------------------


def test_the_target_only_cube_matches_the_float64_oracle(spike, index):
    """The clutter is gone and what is left is the closed form.

    ``support.reference_chain.beat_samples`` is the independent float64 CPU
    oracle the Phase-4/6 acceptance tests already use; it reimplements the beat
    sum from ``fmcw_beat.cu``'s own expression and knows nothing about
    components. Feeding it the MASKED weights is what makes this a check on the
    export rather than on the kernel: the oracle sums every row, so the target
    rows must be exactly the ones the mask left alive.
    """

    composed, _, _ = spike.frame()
    batch = to_synthesis(composed)
    spec = drv.make_spec()
    selected = select_component(batch, index, TARGET)
    measured = _cube(selected, spec).cpu().to(torch.complex128)

    from witwin.radar.synthesis.assembly import pair_tx_index

    tx_index = pair_tx_index(
        num_tx=spec.num_tx, num_rx=spec.num_rx, sensor_pair_count=batch.sensor_pair_count, device=batch.device
    )
    weight = ref.channel_to_beat(selected.complex_transfer_ref.detach().cpu())
    # This frame is composed without a Doppler rate, so the oracle is handed the
    # same exact zeros the kernel is - not a rate it invented.
    assert selected.delay_rate is None
    expected = ref.beat_samples(
        selected.total_delay_s.detach().cpu(),
        torch.zeros(batch.path_count, dtype=torch.float32),
        weight,
        selected.pair_offsets.cpu(),
        spec,
        segment_tx_index=[int(value) for value in tx_index.cpu().tolist()],
    )
    error = float((measured - expected).abs().max())
    scale = float(expected.abs().max())
    print(f"\ntarget-only cube vs float64 oracle: {error / scale:.3e} relative")
    assert error / scale < 1.0e-5

    # Non-vacuity: the full cube is a materially different frame, so agreeing
    # with the oracle here is a statement about the target rows and not about
    # the kernel being self-consistent.
    full = _cube(batch, spec).cpu().to(torch.complex128)
    assert float((full - expected).abs().max()) / scale > 0.1


# --------------------------------------------------------------------------
# 6. Incoherent combination
# --------------------------------------------------------------------------


def test_incoherent_combination_adds_power_and_says_so(spike, index):
    """``sum_j |cube_j|^2``, and it is NOT ``|sum_j cube_j|^2``.

    The second half is the whole semantic. Where two components overlap in
    delay their fields interfere and the coherent magnitude differs from the
    power sum by the cross term; asserting that difference is what makes the
    choice between the two visible instead of a naming preference.
    """

    composed, _, _ = spike.frame()
    batch = to_synthesis(composed)
    spec = drv.make_spec()
    target = _cube(select_component(batch, index, TARGET), spec)
    clutter = _cube(select_component(batch, index, ENVIRONMENT_CLUTTER), spec)

    power = combine_incoherent([target, clutter])
    assert power.dtype == torch.float32
    assert power.shape == target.shape
    reference = target.real.square() + target.imag.square() + clutter.real.square() + clutter.imag.square()
    assert torch.allclose(power, reference, rtol=1.0e-6, atol=0.0)

    coherent = (target + clutter).abs().square()
    cross = float((coherent - power).abs().max())
    print(f"\nincoherent vs coherent cross term: {cross:.6e}")
    assert cross > 1.0e-3 * float(power.max())

    # A single component is its own power, exactly.
    alone = combine_incoherent([target])
    assert torch.allclose(alone, target.real.square() + target.imag.square(), rtol=0.0, atol=0.0)


def test_incoherent_combination_of_disjoint_delays_is_the_magnitude_sum(spike, index):
    """Two components with no overlap: the cross term is what it should be.

    Delay-separating the two components with a synthetic frame would be a
    different fixture, so the disjointness is created where it is checkable -
    in the RANGE PROFILE, by transforming each component's fast time and
    comparing the summed power against the power of the sum bin by bin. The
    residual is the interference the components genuinely have, and it is
    reported rather than assumed away.
    """

    composed, _, _ = spike.frame()
    batch = to_synthesis(composed)
    spec = drv.make_spec()
    target = _cube(select_component(batch, index, TARGET), spec)
    clutter = _cube(select_component(batch, index, ENVIRONMENT_CLUTTER), spec)

    profiles = [torch.fft.fft(cube, dim=-1) for cube in (target, clutter)]
    power = combine_incoherent(profiles)
    coherent = (profiles[0] + profiles[1]).abs().square()

    # The exact identity everywhere: the two laws differ by the cross term
    # ``2 Re(a conj(b))``, which is bounded by ``2 |a| |b|``.
    cross = (coherent - power).abs()
    bound = 2.0 * profiles[0].abs() * profiles[1].abs()
    assert bool((cross <= bound * (1.0 + 1.0e-4) + 1.0e-12).all())

    # Where one component owns the bin outright, the two laws agree, and the
    # rate at which they agree is derived from the dominance rather than tuned:
    # at a power ratio ``r`` the cross term is at most ``2 / sqrt(r)`` of the
    # dominant power.
    ratio = 1.0e4
    dominant = profiles[0].abs().square() > ratio * profiles[1].abs().square()
    assert bool(dominant.any())
    relative = float((cross[dominant] / power[dominant]).max())
    print(f"\ndominant-bin relative cross term: {relative:.3e}")
    assert relative <= 2.0 / math.sqrt(ratio) * (1.0 + 1.0e-3)

    # Where they overlap they genuinely do not agree, which is the semantic
    # this function exists to make visible.
    overlap = ~dominant
    difference = float(cross[overlap].max())
    print(f"\noverlapping-bin cross term: {difference:.6e}")
    assert difference > 1.0e-3 * float(power.max())
    assert math.isfinite(difference)
