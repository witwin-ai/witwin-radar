"""Float64 Torch DSP oracles for the Dirichlet spectrum family.

These were shipped inside the wheel, in ``witwin/radar/solvers/common.py``,
with zero production callers: only tests ever used them. A CPU/Torch reference
implementation belongs under ``tests/`` precisely so that no production module
can import or dispatch to one, so they moved here unchanged.

Unchanged is the point. They are the independent chirp and MIMO references the
native Dirichlet kernels are validated against, so editing them while moving
them would invalidate every comparison that uses them.

Independent is also the point, and it used to be untrue. Until Phase 6 this
module imported ``compute_path_amplitudes`` and ``compute_total_path_lengths``
from ``witwin.radar.solvers.common`` - the very module the Phase-6 native
migration rewrites. An oracle built out of the code under test cannot witness
that the migration preserved anything. Those two expressions now live in
:mod:`reference.path_math`, copied verbatim, and nothing under ``tests/``
imports ``witwin.radar.solvers`` to build a reference.
"""

from __future__ import annotations

import torch

from .path_math import compute_path_amplitudes, compute_total_path_lengths


def pytorch_chirp_reference(radar, distances, amplitudes):
    d_rt = (distances * 2).unsqueeze(-1)
    toa = d_rt / radar.c0
    rx = radar.waveform(radar.t_sample - toa)
    rx_weighted = rx * amplitudes.unsqueeze(-1)
    rx_combined = rx_weighted.sum(dim=0)
    return radar.tx_waveform * torch.conj(rx_combined)


def pytorch_mimo_from_samples(radar, samples):
    """Float64 time-domain MIMO reference over per-TDM-slot samples.

    Slot ``chirp_id * num_tx + tx_id`` contributes the (tx_id, :, chirp_id)
    rows of the frame from its own scene state and TX antenna.
    """
    cfg = radar.config
    frame = torch.zeros(
        (cfg.num_tx, cfg.num_rx, cfg.chirp_per_frame, cfg.adc_samples),
        dtype=torch.complex128,
        device=radar.device,
    )
    rx_pos = radar.rx_pos

    for slot, sample in enumerate(samples):
        chirp_id, tx_id = divmod(slot, cfg.num_tx)
        if sample.points.shape[0] == 0:
            continue
        tx_pos = radar.tx_pos[tx_id : tx_id + 1]
        distances = compute_total_path_lengths(sample, tx_pos, rx_pos).unsqueeze(-1)
        toa = distances / radar.c0
        rx = radar.waveform(radar.t_sample - toa)
        amplitudes = compute_path_amplitudes(
            radar,
            sample,
            distances.squeeze(-1),
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            tx_index=tx_id,
        )
        rx_combined = torch.sum(rx * amplitudes.unsqueeze(-1), dim=-2)
        frame[tx_id, :, chirp_id] = radar.tx_waveform * torch.conj(rx_combined.squeeze(0))

    return frame

__all__ = ["pytorch_chirp_reference", "pytorch_mimo_from_samples"]
