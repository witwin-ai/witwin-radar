"""The full processing pipeline, as ONE callable, with ONE fixture.

``tools/benchmark_processing.py`` measures it and
``tests/test_phase8_pipeline_budget.py`` pins the measurement. Both import from
here so that the thing measured and the thing budgeted are the same call
sequence rather than two sequences that agree today.

The front end is 3 TX x 4 RX. Twelve virtual elements is the smallest array on
which the angle-of-arrival stage and the point cloud are real calls rather than
refusals - ``phase_comparison_aoa`` needs three transmitter rows and the
two-dimensional route needs four.

The Channel endpoints are the multi-endpoint fixture's own physical positions,
spread along ``x``; the DECLARED array is a nominal half-wavelength MIMO array.
The two do not describe the same geometry, so the ANGLES this pipeline produces
are meaningless. That is deliberate and it is the whole scope of this module: it
is a COST fixture. Angular correctness is asserted in ``tests/processing/``
against analytic targets on exact bins.

It is NOT retargeted onto ``Radar.simulate`` by the Phase-11 cutover, and the
reason is the same one that makes it a cost fixture: what is budgeted here
starts at waveform SYNTHESIS, with freezing, discovery and composition already
paid for outside the timed region. ``Radar.simulate`` does all four in one call
and cannot hand back the composed batch alone. The production entry's own
per-frame cost is budgeted separately, in
``tests/test_phase8_pipeline_budget.py::
test_the_simulation_frame_cost_has_not_regressed``.
"""

from __future__ import annotations

from dataclasses import replace

PIPELINE_NUM_TX = 3
PIPELINE_NUM_RX = 4


def array_spec(num_tx: int = PIPELINE_NUM_TX, num_rx: int = PIPELINE_NUM_RX):
    """A ``SensorArraySpec`` for a nominal half-wavelength linear MIMO array."""

    from witwin.radar import RadarConfig
    from witwin.radar.sensors import SensorArraySpec

    from . import multi_endpoint_geometry as geo

    config = dict(geo.FIXTURE_RADAR_CONFIG)
    config["num_tx"] = num_tx
    config["num_rx"] = num_rx
    config["tx_loc"] = [[float(index), 0.0, 0.0] for index in range(num_tx)]
    config["rx_loc"] = [[float(index), 0.0, 0.0] for index in range(num_rx)]
    return SensorArraySpec.from_radar_config(RadarConfig.from_dict(config))


def pipeline_inputs(*, num_chirps: int = 8):
    """One real Channel frame at 3 TX x 4 RX, frozen OUTSIDE the timed region.

    Returns ``(batch, spec, array_spec)``. Freezing, discovery and composition
    are the simulation half and are budgeted separately; what the pipeline
    budget covers starts at waveform synthesis.
    """

    from . import exact_bin_grid as grid
    from . import multi_endpoint_driver as drv
    from . import multi_endpoint_geometry as geo

    base_tx = geo.TX_A_POSITION_M
    base_rx = geo.RX_A_POSITION_M
    transmitters = tuple(
        (900 + index, (base_tx[0] + 0.002 * index, base_tx[1], base_tx[2])) for index in range(PIPELINE_NUM_TX)
    )
    receivers = tuple(
        (950 + index, (base_rx[0] + 0.002 * index, base_rx[1], base_rx[2])) for index in range(PIPELINE_NUM_RX)
    )
    spike = drv.MultiEndpointSpike(transmitters=transmitters, receivers=receivers)
    composed, _, _ = spike.frame()
    batch = drv.to_synthesis(composed)
    spec = replace(grid.fmcw_spec(num_chirps), num_tx=PIPELINE_NUM_TX, num_rx=PIPELINE_NUM_RX, output_domain="spectrum")
    return batch, spec, array_spec()


def run_pipeline(batch, spec, spec_array, *, detector: str = "ca_cfar_fast"):
    """synthesize -> cube -> range -> RD -> CFAR -> AoA -> point cloud.

    One call sequence, no legacy name in it, and exactly one host observation:
    the ``torch.argwhere`` inside the point cloud, which IS the stage because a
    point cloud has a data-dependent length.
    """

    from witwin.radar.processing import (
        ArrayGeometry,
        ProcessingAxes,
        ProcessingCube,
        ca_cfar,
        ca_cfar_fast,
        os_cfar,
        point_cloud,
        range_doppler_map,
        range_profile,
    )
    from witwin.radar.synthesis import synthesize_fmcw
    from witwin.radar.synthesis.assembly import SynthesisResult

    detectors = {"ca_cfar": ca_cfar, "ca_cfar_fast": ca_cfar_fast, "os_cfar": os_cfar}
    cube = synthesize_fmcw(batch, spec)
    result = SynthesisResult.from_fmcw(cube, spec)
    axes = ProcessingAxes.from_synthesis(result, spec, spec_array)
    array = ArrayGeometry.from_axes(axes)
    processing = ProcessingCube.from_synthesis(result, axes)
    profile = range_profile(processing)
    rd = range_doppler_map(profile, window="hann")
    combined = rd.data.reshape(array.sensor_pair_count, *rd.data.shape[-2:]).sum(dim=0)
    cells = detectors[detector](combined.abs(), guard_cells=(1, 2), training_cells=(2, 3), pfa=1e-2)
    return point_cloud(cells, rd, axes, array, max_points=64)


__all__ = ["PIPELINE_NUM_RX", "PIPELINE_NUM_TX", "array_spec", "pipeline_inputs", "run_pipeline"]
