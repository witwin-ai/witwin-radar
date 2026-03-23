"""
Timeline — Multi-frame dynamic scene manager.

Stores discrete keyframes (point clouds + intensities at a fixed source frame
rate) and provides continuous-time interpolation for radar frame generation.

Usage:
    from radar.core import Timeline, Radar

    # From pre-computed point cloud sequence
    timeline = Timeline(frame_rate=30)
    timeline.add_pointcloud_sequence(pointclouds)  # list of (N_i, 3)

    # Generate radar MIMO frames for the full duration
    radar = Radar(config)
    frames = timeline.generate(radar)  # (num_radar_frames, TX, RX, chirps, ADC)

    # Or get the interpolator for single-frame use
    interp = timeline.get_interpolator()
    frame = radar.mimo(interp, t0=0.5)
"""

import torch
import numpy as np
from tqdm import tqdm


class Timeline:
    """Multi-frame dynamic scene manager.

    Stores discrete keyframes and provides continuous-time interpolation
    for radar frame generation.
    """

    def __init__(self, frame_rate=30):
        """
        Args:
            frame_rate: source keyframe rate in Hz (e.g. 30 for 30 FPS motion data)
        """
        self.frame_rate = frame_rate
        self._positions = []    # list of (N_i, 3) tensors
        self._intensities = []  # list of (N_i,) tensors
        self._num_frames = 0

    @property
    def duration(self):
        """Total duration in seconds."""
        if self._num_frames == 0:
            return 0.0
        return (self._num_frames - 1) / self.frame_rate

    @property
    def num_frames(self):
        """Number of keyframes."""
        return self._num_frames

    # ── Input methods ─────────────────────────────────────────────

    def add_keyframe(self, positions, intensities=None):
        """Append one keyframe.

        Args:
            positions: (N, 3) point positions — numpy array or torch tensor
            intensities: (N,) per-point reflectance, or None (defaults to 1.0)
        """
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(positions, dtype=torch.float32, device='cuda')
        elif positions.device.type != 'cuda':
            positions = positions.to(device='cuda', dtype=torch.float32)

        n = positions.shape[0]

        if intensities is None:
            intensities = torch.ones(n, dtype=torch.float32, device='cuda')
        elif isinstance(intensities, np.ndarray):
            intensities = torch.tensor(intensities, dtype=torch.float32, device='cuda')
        elif intensities.device.type != 'cuda':
            intensities = intensities.to(device='cuda', dtype=torch.float32)

        self._positions.append(positions)
        self._intensities.append(intensities)
        self._num_frames += 1

    def add_pointcloud_sequence(self, pointclouds, intensities=None):
        """Bulk-add from a sequence of point clouds.

        Args:
            pointclouds: list of (N_i, 3) arrays, or (F, N, 3) array/tensor
            intensities: list of (N_i,) arrays, (F, N) array/tensor, or None
        """
        if isinstance(pointclouds, (np.ndarray, torch.Tensor)) and pointclouds.ndim == 3:
            # (F, N, 3) uniform array
            for i in range(len(pointclouds)):
                ints_i = intensities[i] if intensities is not None else None
                self.add_keyframe(pointclouds[i], ints_i)
        else:
            # list of variable-length arrays
            for i, pc in enumerate(pointclouds):
                ints_i = intensities[i] if intensities is not None else None
                self.add_keyframe(pc, ints_i)

    def from_motion(self, scene, renderer, motion_data):
        """Render SMPL motion sequence into keyframes.

        Args:
            scene: Scene with an SMPL body named 'human'
            renderer: Renderer bound to the scene
            motion_data: dict or .npz path with keys:
                'pose': (F, 72) SMPL pose parameters
                'shape': (10,) or (F, 10) shape parameters
                'root_translation': (F, 3) root translation per frame
        """
        if isinstance(motion_data, str):
            motion_data = dict(np.load(motion_data))

        poses = motion_data['pose']
        shapes = motion_data['shape']
        translations = motion_data.get('root_translation', None)

        num_motion_frames = len(poses)

        if shapes.ndim == 1:
            shapes = np.tile(shapes, (num_motion_frames, 1))

        for i in tqdm(range(num_motion_frames), desc="Rendering motion frames"):
            trans = translations[i] if translations is not None else None
            scene.update_structure('human', pose=poses[i], shape=shapes[i], position=trans)
            points, intensities = renderer.trace()
            self.add_keyframe(points, intensities)

    # ── Interpolation ─────────────────────────────────────────────

    def get_interpolator(self):
        """Return an interpolator function: interpolator(t) -> (intensities, positions).

        Linear interpolation between adjacent keyframes.
        Points with intensity <= 0.01 are filtered out.
        """
        if self._num_frames == 0:
            raise ValueError("No keyframes added to timeline.")

        positions = self._positions
        intensities = self._intensities
        frame_rate = self.frame_rate
        num_frames = self._num_frames
        total_time = self.duration

        def interpolator(t):
            t = max(0.0, min(t, total_time))
            idx = int(t * frame_rate)
            if idx >= num_frames - 1:
                ints = intensities[-1]
                pos = positions[-1]
                mask = ints > 0.01
                return ints[mask], pos[mask]

            frac = (t * frame_rate) - idx

            pos1, pos2 = positions[idx], positions[idx + 1]
            int1, int2 = intensities[idx], intensities[idx + 1]

            # Handle variable point counts: interpolate matching points,
            # keep extras from the frame with more points
            n1, n2 = pos1.shape[0], pos2.shape[0]
            n_min = min(n1, n2)

            if n1 == n2:
                pos = pos1 * (1 - frac) + pos2 * frac
                ints = int1 * (1 - frac) + int2 * frac
            elif n1 < n2:
                pos_interp = pos1 * (1 - frac) + pos2[:n_min] * frac
                pos = torch.cat([pos_interp, pos2[n_min:]], dim=0)
                ints_interp = int1 * (1 - frac) + int2[:n_min] * frac
                ints = torch.cat([ints_interp, int2[n_min:] * frac], dim=0)
            else:
                pos_interp = pos1[:n_min] * (1 - frac) + pos2 * frac
                pos = torch.cat([pos_interp, pos1[n_min:]], dim=0)
                ints_interp = int1[:n_min] * (1 - frac) + int2 * frac
                ints = torch.cat([ints_interp, int1[n_min:] * (1 - frac)], dim=0)

            mask = ints > 0.01
            return ints[mask], pos[mask]

        return interpolator

    def get_frame_interpolator(self, radar, frame_idx):
        """Return an interpolator for a single radar frame with velocity correction.

        When source keyframe spacing differs from the radar frame duration,
        naive linear interpolation inflates/deflates velocities. This method
        scales the displacement so that the interpolated velocity matches the
        real physical velocity.

        Args:
            radar: Radar instance (provides frame timing)
            frame_idx: which radar frame (0-indexed) to generate

        Returns:
            interpolator: function(t) -> (intensities, positions)
            t0: start time for this frame
        """
        if self._num_frames < 2:
            raise ValueError("Need at least 2 keyframes for frame interpolator.")

        dt_source = 1.0 / self.frame_rate
        T_chirp = (radar.idle_time + radar.ramp_end_time) * 1e-6
        T_frame = T_chirp * radar.num_tx * radar.chirp_per_frame
        vel_scale = T_frame / dt_source

        t0_real = frame_idx / radar.frame_per_second
        kf_idx = int(t0_real * self.frame_rate)
        kf_idx = min(kf_idx, self._num_frames - 2)

        p0 = self._positions[kf_idx]
        p1 = self._positions[kf_idx + 1]
        int0 = self._intensities[kf_idx]

        n = min(p0.shape[0], p1.shape[0])
        if n == 0:
            def empty_interp(t):
                return (torch.zeros(0, device='cuda'),
                        torch.zeros(0, 3, device='cuda'))
            return empty_interp, 0.0

        p0_n, p1_n = p0[:n], p1[:n]
        inten = int0[:n]
        p1_scaled = p0_n + (p1_n - p0_n) * vel_scale

        def interpolator(t):
            frac = t / T_frame if T_frame > 0 else 0.0
            pos = p0_n + (p1_scaled - p0_n) * frac
            mask = inten > 0.01
            return inten[mask], pos[mask]

        return interpolator, 0.0

    # ── Generation ────────────────────────────────────────────────

    def generate(self, radar, progress=True, velocity_corrected=True):
        """Generate all radar MIMO frames for the full timeline.

        Args:
            radar: Radar instance
            progress: show tqdm progress bar
            velocity_corrected: if True (default), scale displacement per frame
                so interpolated velocity matches real physical velocity. Set to
                False for legacy behavior (may inflate velocities when source
                frame rate differs from radar frame rate).

        Returns:
            numpy array of shape (num_radar_frames, TX, RX, chirps, ADC) complex
        """
        if self._num_frames < 1:
            raise ValueError("Need at least 1 keyframe to generate frames.")

        num_radar_frames = max(1, int(self.duration * radar.frame_per_second))

        frames = []
        it = range(num_radar_frames)
        if progress:
            it = tqdm(it, desc="Generating radar frames")

        if velocity_corrected and self._num_frames >= 2:
            for i in it:
                interp, t0 = self.get_frame_interpolator(radar, i)
                frame = radar.mimo(interp, t0)
                frames.append(frame.cpu().numpy())
        else:
            interpolator = self.get_interpolator()
            for i in it:
                t0 = i / radar.frame_per_second
                frame = radar.mimo(interpolator, t0)
                frames.append(frame.cpu().numpy())

        return np.array(frames)

    def generate_rd(self, radar, tx=0, rx=0, progress=True):
        """Generate Range-Doppler maps for all frames.

        Args:
            radar: Radar instance
            tx: TX antenna index
            rx: RX antenna index
            progress: show tqdm progress bar

        Returns:
            rd_mags: (num_frames, chirps, ADC) float32 array — magnitude in dB
            ranges: (num_range_bins//2,) float64 array — range axis in meters
            velocities: (num_doppler_bins,) float64 array — velocity axis in m/s
        """
        frames = self.generate(radar, progress=progress)
        from ..sigproc import process_rd

        rd_mags = []
        for f in frames:
            rd_mag, _, _, _ = process_rd(radar, f, tx=tx, rx=rx)
            rd_mags.append(rd_mag)

        ranges = radar.ranges.cpu().numpy()
        velocities = radar.velocities.cpu().numpy()
        return np.array(rd_mags), ranges, velocities
