"""End-to-end validation on the scene-driven route: single-target scenarios.

The full production chain, once per test:

    Radar.simulate -> RadarSimulationResult.cube -> processing/

These are the only end-to-end range, velocity and angle ACCURACY tests in the
repository - everything else pins a contract, a launch count or a derivative -
so they were migrated rather than deleted when the previous simulation and DSP
routes were removed. Nothing here calls a compatibility layer; every assertion
runs through the canonical processing owners.

Three things changed with the route and are stated here because they change what
the numbers mean, not merely how they are spelled:

* **The world is a Core ``Scene``.** A target is a ``ScatterSitePolicy`` site,
  the run asks for ``components={"los"}`` at ``max_depth=0``, and the scene is
  empty, so what is measured is exactly the free-space round trip.
* **The radar looks along world +x.** See ``conftest`` for why: on the old
  ``-z`` boresight the endpoint polarization is parallel to the look direction
  and Channel correctly publishes an exactly zero transport.
* **Intra-frame Doppler is opened by the caller.** ``Radar.simulate`` has no
  velocity keyword, so ``conftest.simulate_point_targets`` dualises the site
  tensor through ``propagation.two_way_duals``. A moving-target test
  here is therefore also the statement that the kinematics seam composes with
  the production entry.

Coverage that did NOT survive the route, recorded rather than quietly dropped:
``sigproc``'s ``topk`` detector has no owner in ``witwin.radar.processing``, so
the two ``topk`` SNR tests became one comparison between the two detectors that
DO exist, ``ca_cfar_fast`` and ``os_cfar``.
"""

import numpy as np
import pytest
import torch

from conftest import (
    STANDARD_CONFIG, FAST_CONFIG,
    make_scene_radar_or_skip, simulate_point_targets,
)

pytestmark = pytest.mark.gpu

# Validation config: adc_start_time=0 for clean signal, enough chirps for Doppler
_VFAST = {
    **FAST_CONFIG,
    "adc_start_time": 0,
    "chirp_per_frame": 32,
    "num_doppler_bins": 32,
}
_VFULL = {**STANDARD_CONFIG, "adc_start_time": 0}


def _local(distance, *, x=0.0, y=0.0):
    """A target ``distance`` metres straight ahead, in the radar's local frame."""

    return (x, y, -float(distance))


def _strongest(cloud):
    """The highest-energy detection, as ``(x, y, z, velocity, energy, range)``."""

    assert int(cloud.xyz.shape[0]) > 0, "no points detected"
    index = int(cloud.energy.argmax())
    return (
        float(cloud.xyz[index, 0]),
        float(cloud.xyz[index, 1]),
        float(cloud.xyz[index, 2]),
        float(cloud.velocity_mps[index]),
        float(cloud.energy[index]),
        float(cloud.range_m[index]),
    )


class TestStaticTarget:
    """Single static target -> range + angle accuracy."""

    @pytest.mark.parametrize("distance", [1.5, 3.0, 5.0, 8.0])
    def test_range_accuracy(self, distance):
        """Detected range should match target distance within +/-2 range bins."""

        radar = make_scene_radar_or_skip(_VFAST)
        frame = simulate_point_targets(radar, [_local(distance)])
        cloud = frame.point_cloud(positive_velocity_only=False)

        ranges = cloud.range_m.cpu().numpy()
        best = ranges[np.argmin(np.abs(ranges - distance))]
        tol = frame.axes.range_bin_m * 2
        assert abs(best - distance) < tol, (
            f"distance={distance}m: detected {best:.3f}m, tolerance={tol:.3f}m"
        )

    def test_the_axes_record_describes_the_published_cube(self):
        """The processing metadata and the simulated cube are one product.

        ``ProcessingAxes`` is built from a re-synthesis of the same composed
        rows, because ``RadarSimulationResult`` publishes the stacked cube and
        not the rank-3 ``SynthesisResult`` the axes constructor reads. That
        workaround is only sound while the two agree BITWISE, so it is pinned
        here rather than assumed by every test above.
        """

        radar = make_scene_radar_or_skip(_VFAST)
        frame = simulate_point_targets(radar, [_local(3.0)])
        frame.assert_axes_describe_the_cube()

    def test_broadside_target_angle(self):
        """A target dead ahead lands near the boresight axis in the point cloud."""

        radar = make_scene_radar_or_skip(_VFAST)
        frame = simulate_point_targets(radar, [_local(3.0)])
        cloud = frame.point_cloud(positive_velocity_only=False)

        x, _, z, _, _, detected = _strongest(cloud)
        assert abs(detected - 3.0) < frame.axes.range_bin_m * 3
        assert abs(x) < 1.0, f"x={x:.3f} m too large for a broadside target"
        assert abs(z) < 1.0, f"z={z:.3f} m too large for a broadside target"

    def test_range_doppler_map_peak(self):
        """The RD map peaks at the target's range bin."""

        radar = make_scene_radar_or_skip(_VFAST)
        distance = 3.0
        frame = simulate_point_targets(radar, [_local(distance)])

        profile = frame.range_profile_db()
        peak = int(profile.argmax())
        detected = float(frame.axes.range_m[peak])
        assert abs(detected - distance) < frame.axes.range_bin_m * 2, (
            f"RD peak at range={detected:.3f}m, expected {distance}m"
        )

    def test_static_target_zero_doppler(self):
        """A still world is not "near" zero Doppler, it is EXACTLY zero.

        The old route measured a static target through a per-chirp scene sample
        and could only claim a small Doppler. On this route a frame with no
        velocity dual composes one frozen weight and the waveform kernel's
        slow-time carrier has nothing to walk, so every chirp of the cube is
        bitwise identical and the zero-Doppler bin is the peak by construction.
        Asserting the bitwise statement is what makes the moving cases below
        measurable at all.
        """

        radar = make_scene_radar_or_skip(_VFAST)
        frame = simulate_point_targets(radar, [_local(3.0)])

        chirps = frame.cube
        assert torch.equal(chirps[:, :, 0, :], chirps[:, :, -1, :])

        combined = frame.combined_map().abs()
        flat = int(combined.argmax())
        doppler_bin = flat // combined.shape[1]
        assert float(frame.axes.velocity_mps[doppler_bin]) == 0.0


class TestMovingTarget:
    """Single moving target -> velocity accuracy."""

    def test_velocity_accuracy(self):
        """An approaching target produces the Doppler its closing speed implies."""

        radar = make_scene_radar_or_skip(_VFULL)
        speed = 1.5  # m/s, closing: local +z shortens the boresight range
        frame = simulate_point_targets(
            radar, [(_local(3.0), (0.0, 0.0, speed))]
        )
        cloud = frame.point_cloud(positive_velocity_only=False)

        _, _, _, velocity, _, detected = _strongest(cloud)
        assert abs(detected - 3.0) < frame.axes.range_bin_m * 5
        tol = frame.axes.velocity_bin_mps * 3
        assert abs(abs(velocity) - speed) < tol, (
            f"expected |v| ~= {speed} m/s, detected {velocity:.3f} m/s"
        )

    def test_the_doppler_sign_follows_the_direction_of_travel(self):
        """Closing and receding at the same speed give opposite signs.

        The magnitude alone cannot say the sign convention survived the chain:
        ``PROCESSING_DOPPLER_CONVENTION`` is closing-positive and a chain that
        inverted it would report the same |v| for both directions.
        """

        radar = make_scene_radar_or_skip(_VFULL)
        speed = 1.5
        closing = simulate_point_targets(
            radar, [(_local(3.0), (0.0, 0.0, speed))]
        )
        receding = simulate_point_targets(
            radar, [(_local(3.0), (0.0, 0.0, -speed))]
        )
        forward = _strongest(closing.point_cloud(positive_velocity_only=False))[3]
        backward = _strongest(receding.point_cloud(positive_velocity_only=False))[3]

        assert forward > 0.0, forward
        assert backward < 0.0, backward
        assert abs(forward + backward) < closing.axes.velocity_bin_mps

    def test_rd_map_shows_doppler_shift(self):
        """A moving target moves the RD peak off the zero-Doppler bin."""

        radar = make_scene_radar_or_skip(_VFULL)
        speed = 2.0
        frame = simulate_point_targets(
            radar, [(_local(3.0), (0.0, 0.0, speed))]
        )

        combined = frame.combined_map().abs()
        flat = int(combined.argmax())
        doppler_bin = flat // combined.shape[1]
        velocity = float(frame.axes.velocity_mps[doppler_bin])
        assert abs(velocity) > frame.axes.velocity_bin_mps, (
            f"moving target ({speed} m/s) shows zero Doppler: {velocity:.4f}"
        )


class TestMovingTargetAngle:
    """Regression: the TDM slot phase must be simulated, not only compensated.

    ``processing.point_cloud`` calls ``processing.aoa.tdm_compensate``, which
    removes the per-transmitter phase a moving target writes across the TDM
    slots. If the simulated frame never carried that phase, the compensation
    would ADD one, and a broadside moving target would show a spurious
    elevation - historically about 0.8 m at 3 m range.
    """

    @pytest.mark.parametrize("closing_speed", [-1.5, 1.5])
    def test_moving_broadside_target_stays_broadside(self, closing_speed):
        radar = make_scene_radar_or_skip(_VFULL)
        frame = simulate_point_targets(
            radar, [(_local(3.0), (0.0, 0.0, closing_speed))]
        )
        cloud = frame.point_cloud(positive_velocity_only=False)

        x, _, z, _, _, detected = _strongest(cloud)
        assert abs(detected - 3.0) < frame.axes.range_bin_m * 4
        assert abs(x) < 0.2, f"azimuth x={x:.3f} m for a broadside moving target"
        assert abs(z) < 0.2, f"elevation z={z:.3f} m for a broadside moving target"


class TestEnergyScale:
    """The published detection energy is in dB relative to the map amplitude.

    ``PointCloud.energy`` is ``20 log10(|map| + energy_floor)`` and
    ``ScalarRcsResponse.from_rcs`` makes the transported amplitude proportional
    to ``sqrt(sigma)``. A hundredfold cross section is therefore exactly 20 dB,
    and that factor is what distinguishes a dB scale from a bare ``log10`` one.

    The cross sections here are large ON PURPOSE. The stage's ``energy_floor``
    is ADDITIVE and defaults to ``1e-6``, while the free-space round trip from a
    1 m^2 target at 3 m and 77 GHz lands at a map amplitude of ``3.6e-6`` - the
    same order as the floor, which biases the low reading by 1.9 dB and is a
    correct reading of a floored quantity rather than a scaling defect. Measured
    at ``1e4`` and ``1e6`` m^2, where the floor is four orders down, the ratio is
    the exact 20 dB.
    """

    #: Two cross sections a hundredfold apart, both far above the energy floor.
    SMALL_RCS_M2 = 1.0e4
    LARGE_RCS_M2 = 1.0e6

    @staticmethod
    def _peak_energy(radar, *, sigma_m2):
        frame = simulate_point_targets(
            radar, [_local(3.0)], sigma_m2=sigma_m2
        )
        cloud = frame.point_cloud(positive_velocity_only=False)
        return float(cloud.energy.max())

    def test_a_hundredfold_cross_section_is_twenty_decibels(self):
        radar = make_scene_radar_or_skip(_VFAST)
        low = self._peak_energy(radar, sigma_m2=self.SMALL_RCS_M2)
        high = self._peak_energy(radar, sigma_m2=self.LARGE_RCS_M2)
        assert high - low == pytest.approx(20.0, abs=0.1), (
            f"sigma x100 changed the peak by {high - low:.2f} dB"
        )

    def test_the_additive_energy_floor_is_what_biases_a_weak_target(self):
        """The falsifier for the paragraph above, measured rather than asserted.

        Without this a reader would have to take on trust that the 1.9 dB
        deviation at 1 m^2 is the floor and not a broken scale. The combined
        MAP - which carries no floor - scales by exactly 20 dB across the same
        pair of cross sections.
        """

        radar = make_scene_radar_or_skip(_VFAST)
        levels = []
        for sigma in (1.0, 100.0):
            frame = simulate_point_targets(radar, [_local(3.0)], sigma_m2=sigma)
            levels.append(float(frame.combined_map().abs().max()))
        assert 20.0 * np.log10(levels[1] / levels[0]) == pytest.approx(
            20.0, abs=1e-3
        )

    def test_the_two_detectors_report_the_same_scale_and_the_same_angle(self):
        """``ca_cfar_fast`` and ``os_cfar`` select different cells, not different
        physics.

        This replaces ``sigproc``'s CFAR-versus-topk pair: ``topk`` has no owner
        in ``witwin.radar.processing``, and the claim those two tests were making
        - that the detector choice does not change the level or the angle - is
        exactly what these two detectors can say.
        """

        from witwin.radar.processing import ca_cfar_fast, os_cfar, point_cloud

        radar = make_scene_radar_or_skip(_VFAST)
        frame = simulate_point_targets(radar, [_local(3.0)])
        rd = frame.range_doppler()
        combined = rd.data.reshape(
            frame.array.sensor_pair_count, *rd.data.shape[-2:]
        ).sum(dim=0)

        clouds = {}
        for name, detector in (("ca", ca_cfar_fast), ("os", os_cfar)):
            cells = detector(
                combined.abs(),
                guard_cells=(1, 2),
                training_cells=(2, 3),
                pfa=1e-2,
            )
            clouds[name] = point_cloud(
                cells,
                rd,
                frame.axes,
                frame.array,
                max_points=64,
                positive_velocity_only=False,
            )

        peaks = {name: _strongest(cloud) for name, cloud in clouds.items()}
        assert abs(peaks["ca"][4] - peaks["os"][4]) < 30.0, peaks
        assert abs(peaks["ca"][0] - peaks["os"][0]) < 1.0, peaks
        assert abs(peaks["ca"][5] - peaks["os"][5]) < frame.axes.range_bin_m * 3
