"""Deprecated: the legacy Range-Doppler and point-cloud entries.

Every name here is re-exported from
:mod:`witwin.radar.processing.adapters`, which builds a ``ProcessingAxes`` and
an ``ArrayGeometry`` from a legacy ``Radar`` and calls the Phase-8 processing
owners. This module contains no DSP: after the cutover, every production
``torch.fft``, every detector, every angle estimator and every beamformer in the
radar tree lives under ``witwin/radar/processing/``.

Deleted here at the cutover, per the repository no-legacy rule:

* ``FrameConfig``'s seven raw ``radar.config.*`` reads;
* the duplicate range and Doppler transform bodies inside ``process_rd_tensor``;
* ``frame_reshape`` - ``synthesis/assembly.py::assemble_frame_cube`` already
  performs the correct sink-major to tx-major transpose, and two packers for one
  layout is how a TX/RX swap ships silently on a square array;
* ``_process_pc_cfar_tensor``, a near-duplicate of ``frame2pointcloud``
  differing only in the detector, which is now an argument;
* ``reg_data``'s ``numpy`` and ``np.random`` internals;
* the magic range gates ``[:, :25]`` and ``[:, 125:]``, now expressed in metres;
* ``_compensate_tdm_phase``'s Python transmitter loop with an in-place ``*=`` on
  a clone, now one broadcast multiply in
  :func:`witwin.radar.processing.aoa.tdm_compensate`.
"""

from __future__ import annotations

from ..processing.adapters import (
    FrameConfig,
    PointCloudProcessConfig,
    clutter_removal,
    doppler_fft,
    frame2pointcloud,
    naive_xyz,
    process_pc,
    process_pc_tensor,
    process_rd,
    process_rd_tensor,
    range_fft,
    reg_data,
)

__all__ = [
    "FrameConfig",
    "PointCloudProcessConfig",
    "clutter_removal",
    "doppler_fft",
    "frame2pointcloud",
    "naive_xyz",
    "process_pc",
    "process_pc_tensor",
    "process_rd",
    "process_rd_tensor",
    "range_fft",
    "reg_data",
]
