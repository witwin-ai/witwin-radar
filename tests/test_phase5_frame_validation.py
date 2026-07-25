"""A frame that is not this frozen topology's frame must be refused, not gathered.

Both composers hold index tables frozen against a specific leg. Those indices
are handed to a native kernel that gathers through raw pointers: the forward and
JVP entries are never told the leg counts, and their length checks only tie the
inputs to each other. A leg batch of the wrong length therefore used to compose
without complaint and publish a full frame of plausible delays and transfers
read from whatever sat past the end of the buffer - the exact "plausible wrong
answer" the fail-loud policy exists to prevent.

The counts are all host ints already in hand at the facade, so refusing costs
nothing and observes nothing. These tests pin that the refusal is unconditional:
the previous accidental guard was ``index_select`` inside the ``row_valid``
path, which only fires when a leg happens to carry a mask.
"""

from __future__ import annotations

import pytest
import torch

from witwin.radar.paths import DirectComposer, TwoWayComposer

from reference.two_way_torch import PerSiteResponse  # noqa: E402
from support import join_fixture as fx  # noqa: E402


pytestmark = pytest.mark.gpu

SOURCES = [10, 11]
SINKS = [30]
SITES = [20, 21]
COMPONENTS = [0, 1]


def _composer() -> TwoWayComposer:
    return TwoWayComposer.freeze(
        fx.frozen_leg(fx.leg_rows(SOURCES, SITES, COMPONENTS)),
        fx.frozen_leg(fx.leg_rows(SITES, SINKS, COMPONENTS)),
        torch.tensor(SITES, dtype=torch.int64, device="cuda"),
        radar_source_ids=SOURCES,
        radar_sink_ids=SINKS,
        reference_frequency_hz=77.0e9,
    )


def _batch(rows: int, *, seed: int, row_valid: torch.Tensor | None = None):
    delay, rate, coefficient = fx.payload(rows, seed=seed)
    return fx.leg_batch(
        delay.to(torch.float32).contiguous(),
        coefficient.to(torch.complex64).contiguous(),
        rate=rate.to(torch.float32).contiguous(),
        row_valid=row_valid,
    )


def _response(composer, *, sites: int | None = None) -> PerSiteResponse:
    _, _, value = fx.payload(composer.site_count if sites is None else sites, seed=303)
    return PerSiteResponse(value.to(torch.complex64))


def _legs(composer, *, inbound_rows=None, outbound_rows=None, row_valid=None):
    valid_in = valid_out = None
    if row_valid is not None:
        valid_in, valid_out = row_valid
    inbound = _batch(
        composer.inbound_row_count if inbound_rows is None else inbound_rows,
        seed=101,
        row_valid=valid_in,
    )
    outbound = _batch(
        composer.outbound_row_count if outbound_rows is None else outbound_rows,
        seed=102,
        row_valid=valid_out,
    )
    return inbound, outbound


def test_the_matching_frame_still_composes():
    """The control. Every refusal below has to be the mismatch talking."""

    composer = _composer()
    inbound, outbound = _legs(composer)
    composed = composer.compose(inbound, outbound, _response(composer))
    assert composed.path_count == composer.path_count
    assert torch.isfinite(composed.total_delay_s).all()
    assert float(composed.total_delay_s.abs().min()) > 0.0


@pytest.mark.parametrize("delta", [-1, 1])
def test_an_inbound_leg_of_the_wrong_length_is_refused(delta):
    composer = _composer()
    inbound, outbound = _legs(
        composer, inbound_rows=composer.inbound_row_count + delta
    )
    with pytest.raises(ValueError, match="does not belong to this frozen topology"):
        composer.compose(inbound, outbound, _response(composer))


@pytest.mark.parametrize("delta", [-1, 1])
def test_an_outbound_leg_of_the_wrong_length_is_refused(delta):
    composer = _composer()
    inbound, outbound = _legs(
        composer, outbound_rows=composer.outbound_row_count + delta
    )
    with pytest.raises(ValueError, match="does not belong to this frozen topology"):
        composer.compose(inbound, outbound, _response(composer))


def test_the_refusal_names_the_leg_and_both_counts():
    """A caller has to be able to tell WHICH leg drifted, and by how much."""

    composer = _composer()
    inbound, outbound = _legs(composer, inbound_rows=3)
    with pytest.raises(ValueError) as caught:
        composer.compose(inbound, outbound, _response(composer))
    message = str(caught.value)
    assert "inbound" in message
    assert "3" in message
    assert str(composer.inbound_row_count) in message


def test_the_refusal_does_not_depend_on_a_leg_carrying_row_valid():
    """The hole opened exactly when neither leg carried a mask.

    With a mask present, ``index_select`` bounds-checked incidentally. That made
    the guard a property of the data rather than of the contract, so both cases
    are pinned.
    """

    composer = _composer()
    short = composer.inbound_row_count - 1
    for mask in (
        None,
        (
            torch.ones(short, dtype=torch.bool, device="cuda"),
            torch.ones(composer.outbound_row_count, dtype=torch.bool, device="cuda"),
        ),
    ):
        inbound, outbound = _legs(composer, inbound_rows=short, row_valid=mask)
        assert (inbound.row_valid is None) == (mask is None)
        with pytest.raises(
            ValueError, match="does not belong to this frozen topology"
        ):
            composer.compose(inbound, outbound, _response(composer))


@pytest.mark.parametrize("sites", [1, 3])
def test_a_response_of_the_wrong_length_is_refused(sites):
    """``ScatterResponse`` is an extension point, so its length is checked too.

    The forward kernel's only check on the response is against itself, so a
    third-party response returning the wrong count gathers out of bounds the
    same way a mismatched leg does.
    """

    composer = _composer()
    inbound, outbound = _legs(composer)

    class _WrongLength:
        """A protocol-satisfying response that ignores the asked-for count."""

        is_geometry_dependent = False

        def evaluate(self, row_count, device):
            _, _, value = fx.payload(sites, seed=404, device=str(device))
            return value.to(torch.complex64)

    with pytest.raises(ValueError, match="frozen against"):
        composer.compose(inbound, outbound, _WrongLength())


def test_a_response_that_is_not_a_tensor_is_refused():
    composer = _composer()
    inbound, outbound = _legs(composer)

    class _NotATensor:
        is_geometry_dependent = False

        def evaluate(self, row_count, device):
            return [1.0 + 0.0j] * row_count

    with pytest.raises(TypeError, match="torch.Tensor"):
        composer.compose(inbound, outbound, _NotATensor())


# --------------------------------------------------------------------------
# The direct composer holds the same kind of frozen index table
# --------------------------------------------------------------------------


def _direct_composer() -> DirectComposer:
    return DirectComposer.freeze(
        fx.frozen_leg(fx.leg_rows(SOURCES, SINKS, COMPONENTS)),
        radar_source_ids=SOURCES,
        radar_sink_ids=SINKS,
        reference_frequency_hz=77.0e9,
    )


def test_a_direct_leg_of_the_wrong_length_is_refused():
    """A gather is not self-validating either.

    A longer batch gathers in-range but wrong rows and publishes a plausible
    frame; a shorter one trips a device-side assert launches later, in a place
    that names nothing useful.
    """

    composer = _direct_composer()
    assert composer.path_count == len(SOURCES) * len(SINKS) * len(COMPONENTS)
    composer.compose(_batch(composer.path_count, seed=201))
    for rows in (composer.path_count - 1, composer.path_count + 1):
        with pytest.raises(
            ValueError, match="does not belong to this frozen topology"
        ):
            composer.compose(_batch(rows, seed=201))
