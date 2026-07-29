"""The refreshed-weight slow-time cube, built from the batched producer.

``SlowTimeMode.REFRESHED_WEIGHT_NO_RATE`` has been named since Phase 6 and has
never had a producer. This module is that producer, and it is deliberately
test-side: the production inner loop stays
``FROZEN_WEIGHT_WITH_CARRIER_RATE``, which does a whole frame in one kernel
launch from one composed batch, and the refreshed mode exists here as an
INDEPENDENT ORACLE for it. Two models of the same physics that agree is
evidence; one model asserted against itself is not.

What makes it independent:

* the frozen cube gets ``tau(c, p) = tau_rt + tau_rate * t_slot`` from the beat
  kernel, i.e. a first-order extrapolation of the delay from the frame origin;
* the refreshed cube gets ``tau(c, p)`` by REEVALUATING the propagation at the
  world state of every slot, so its delay is the exact geometric one.

They therefore differ by the second-order term
``0.5 * d2(tau)/dt2 * t_slot^2``, and by nothing else. That is the whole
content of the comparison.

The slot times and the slot index come from
``witwin.radar.synthesis.assembly``, which is the Phase-6 TDM owner, so this
oracle cannot drift onto a second slow-time slot table.
"""

from __future__ import annotations

from dataclasses import replace

import torch


def slot_of_each_row(composed, *, num_chirps: int, num_tx: int, num_rx: int):
    """``int64[num_chirps, path_count]``: which TDM slot each row sits in.

    A composed row belongs to exactly one sensor pair, and a sensor pair is
    driven by exactly one transmitter, so a row's slow-time slot in chirp ``c``
    is ``c * num_tx + tx(pair_of_row)``. This resolves it through
    :func:`witwin.radar.synthesis.assembly.pair_slot_index`, so the oracle reads the
    same table the beat kernel does.
    """

    from witwin.radar.synthesis.assembly import pair_slot_index

    table = pair_slot_index(
        num_chirps=num_chirps,
        num_tx=num_tx,
        num_rx=num_rx,
        sensor_pair_count=composed.sensor_pair_count,
        device=composed.sensor_pair_index.device,
    )
    return table.index_select(1, composed.sensor_pair_index)


def gather_slot_rows(frames, slot_of_row: torch.Tensor):
    """One composed batch per chirp, each row taken from its own slot.

    ``frames[t]`` is the composition of slot ``t``; ``slot_of_row[c, r]`` says
    which of them chirp ``c``'s row ``r`` belongs to. Under TDM two rows of the
    same chirp are a whole chirp period apart in slow time whenever they are
    driven by different transmitters, so a per-chirp cube cannot be built from
    a single slot.
    """

    rows = frames[0].path_count
    columns = torch.arange(rows, device=slot_of_row.device, dtype=torch.int64)
    delays = torch.stack([frame.total_delay_s for frame in frames])
    transfer = torch.stack([frame.complex_transfer_ref for frame in frames])
    validity = None if frames[0].row_valid is None else torch.stack([frame.row_valid for frame in frames])
    gathered = []
    for chirp in range(int(slot_of_row.shape[0])):
        index = slot_of_row[chirp]
        gathered.append(
            replace(
                frames[0],
                total_delay_s=delays[index, columns],
                complex_transfer_ref=transfer[index, columns],
                delay_rate=None,
                row_valid=None if validity is None else validity[index, columns],
            )
        )
    return gathered


def refreshed_cube(frames, spec, *, num_chirps: int):
    """The slow-time cube with the weight refreshed at every slot.

    One synthesis per chirp, each over a ONE-chirp spec whose carrier rate is
    zero because the weight already walked. Stacking them is the refreshed
    equivalent of the frozen mode's single launch; it is slower by exactly the
    factor this oracle is willing to pay to be independent.
    """

    from witwin.radar.synthesis import SlowTimeMode, SynthesisPathBatch, synthesize_fmcw

    if spec.carrier_hz != 0.0:
        raise ValueError(
            "the refreshed oracle expects the weight to own the carrier "
            "(carrier_hz == 0); a kernel carrier would be a different "
            "comparison"
        )
    one_chirp = replace(spec, num_chirps=1, carrier_rate_hz=0.0)
    slot_of_row = slot_of_each_row(frames[0], num_chirps=num_chirps, num_tx=spec.num_tx, num_rx=spec.num_rx)
    cubes = []
    for composed in gather_slot_rows(frames, slot_of_row):
        batch = SynthesisPathBatch.from_radar_paths(composed, slow_time_mode=SlowTimeMode.REFRESHED_WEIGHT_NO_RATE)
        cubes.append(synthesize_fmcw(batch, one_chirp))
    return torch.cat(cubes, dim=0)


__all__ = ["gather_slot_rows", "refreshed_cube", "slot_of_each_row"]
