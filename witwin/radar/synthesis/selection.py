"""Export one scene component from a batch, without touching row identity.

Exporting the clutter alone, or the target alone, is the same operation the
waveform kernels already perform on a dead row: zero its WEIGHT before the
launch. ``synthesize_fmcw_beat`` does exactly that with ``torch.where``, and it
does it on the weight rather than on the output precisely so the row is inert
in the primal AND carries no gradient back to anything it was built from. A
component mask is that operation with a different predicate.

Three consequences, all of them load bearing:

* the exported batch shares the SAME
  :class:`~witwin.radar.paths.contracts.RadarPathTopology` OBJECT, so "row
  identity is unchanged" is assertable with ``is`` rather than with an
  elementwise comparison that a rebuilt-but-equal topology would also pass;
* row order, dtype, device and ``row_valid`` pass through untouched, so every
  export has the same shape and the same pair partition and the per-component
  cubes are addable;
* the waveform kernel accumulates over the rows of a pair segment in the same
  order regardless of masking, and a masked row contributes a literal ``0.0``
  in its own slot. Summing the per-component cubes therefore reproduces the
  full cube up to float re-association of the partial sums, and NOT bitwise:
  ``(a + 0 + c) + (0 + b + 0)`` is not ``(a + b + c)`` in float32. The
  acceptance test states a tolerance and records the measured residual.

There is no kernel change here and there is no incoherent mode. A component is
a row subset of one topology, evaluated by the same launches; combining two
components in POWER is a post-synthesis operation and lives in
:mod:`witwin.radar.processing`.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..paths.components import RadarComponentIndex
from .contracts import SynthesisPathBatch


def select_component(
    batch: SynthesisPathBatch, index: RadarComponentIndex, name: str
) -> SynthesisPathBatch:
    """The same batch with every row outside ``name`` made inert.

    ``index`` must have been built from the batch's own topology. That is
    checked by OBJECT IDENTITY, not by row count: two topologies of the same
    length are the commonest way to mask the wrong rows, and the resulting cube
    is a perfectly plausible frame of the wrong scene.

    A composed band is masked along with the reference column. Leaving it
    unmasked would publish a batch whose narrowband weight says the row is
    absent and whose wideband columns say it is present, and the waveform owner
    that consumed the band would silently disagree with the one that did not.
    """

    if not isinstance(batch, SynthesisPathBatch):
        raise TypeError(
            f"select_component needs a SynthesisPathBatch, got {type(batch).__name__}"
        )
    if not isinstance(index, RadarComponentIndex):
        raise TypeError(
            "select_component needs a RadarComponentIndex, got "
            f"{type(index).__name__}"
        )
    if index.topology is not batch.topology:
        raise ValueError(
            "the component index was built from a different topology object "
            "than this batch carries; a component mask addresses rows by "
            "position within ONE frozen topology, and masking with another "
            "topology's classification would publish a plausible cube of the "
            "wrong scene"
        )
    mask = index.mask(name)
    weight = torch.where(
        mask, batch.complex_transfer_ref, torch.zeros_like(batch.complex_transfer_ref)
    )
    band = (
        None
        if batch.frequency_response is None
        else torch.where(
            mask.unsqueeze(1),
            batch.frequency_response,
            torch.zeros_like(batch.frequency_response),
        )
    )
    return replace(batch, complex_transfer_ref=weight, frequency_response=band)


__all__ = ["select_component"]
