"""
Radar signal processing: Range-Doppler FFT and point cloud extraction.

Pipeline: radar frame -> range FFT -> clutter removal -> Doppler FFT -> peak
detection -> AoA -> 3D point cloud.
"""

from __future__ import annotations

import numpy as np
import torch

from ..types import DetectorType, normalize_detector_type
from ..utils.tensor import (
    is_torch_tensor as _is_torch_tensor,
    real_dtype as _real_dtype,
    to_numpy as _to_numpy,
)


def _hamming_window(length: int, *, reference):
    if _is_torch_tensor(reference):
        return torch.hamming_window(
            length,
            periodic=False,
            dtype=_real_dtype(reference),
            device=reference.device,
        )
    return np.hamming(length)


def _empty_pointcloud(reference):
    if _is_torch_tensor(reference):
        return torch.zeros((6, 0), dtype=torch.float64, device=reference.device)
    return np.zeros((6, 0), dtype=np.float64)


class FrameConfig:
    """Derived frame parameters from a Radar config."""

    def __init__(self, radar):
        self.numTxAntennas = radar.num_tx
        self.numRxAntennas = radar.num_rx
        self.numLoopsPerFrame = radar.chirp_per_frame
        self.numADCSamples = radar.adc_samples
        self.numAngleBins = radar.num_angle_bins

        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
        self.numRangeBins = self.numADCSamples
        self.numDopplerBins = self.numLoopsPerFrame

        self.chirpSize = self.numRxAntennas * self.numADCSamples
        self.chirpLoopSize = self.chirpSize * self.numTxAntennas
        self.frameSize = self.chirpLoopSize * self.numLoopsPerFrame

        self.range_resolution = radar.range_resolution
        self.doppler_resolution = radar.doppler_resolution

        # TX positions in half-wavelength units (for AoA compensation)
        self.tx_loc_hw = radar.tx_loc / (radar._lambda / 2)


class PointCloudProcessConfig:
    """Configuration for point cloud extraction from radar frames."""

    def __init__(
        self,
        radar,
        static_clutter_removal: bool = False,
        energy_top_k: int = 128,
        range_cut: bool = False,
        output_velocity: bool = True,
        output_snr: bool = True,
        output_range: bool = True,
        output_in_meter: bool = True,
    ):
        self.frame_config = FrameConfig(radar)
        self.enable_static_clutter_removal = static_clutter_removal
        self.use_energy_top_k = energy_top_k > 0
        self.energy_top_k = energy_top_k
        self.range_cut = range_cut
        self.output_velocity = output_velocity
        self.output_snr = output_snr
        self.output_range = output_range
        self.output_in_meter = output_in_meter

        dim = 3  # x, y, z
        if self.output_velocity:
            self.velocity_dim = dim
            dim += 1
        if self.output_snr:
            self.snr_dim = dim
            dim += 1
        if self.output_range:
            self.range_dim = dim
            dim += 1

        fc = self.frame_config
        self.coupling_signature_bin_front_idx = 5
        self.coupling_signature_bin_rear_idx = 4
        self.sum_coupling_signature_array = np.zeros(
            (
                fc.numTxAntennas,
                fc.numRxAntennas,
                self.coupling_signature_bin_front_idx + self.coupling_signature_bin_rear_idx,
            ),
            dtype=complex,
        )


# ---------------------------------------------------------------------------
# Core DSP functions
# ---------------------------------------------------------------------------

def frame_reshape(frame, frame_config):
    """Reshape a flat IQ frame into (TX, RX, chirps, ADC_samples)."""
    shape = (
        frame_config.numLoopsPerFrame,
        frame_config.numTxAntennas,
        frame_config.numRxAntennas,
        -1,
    )
    if _is_torch_tensor(frame):
        return frame.reshape(shape).permute(1, 2, 0, 3)
    return np.reshape(frame, shape).transpose(1, 2, 0, 3)


def range_fft(reshaped_frame, frame_config):
    """Apply a Hamming-windowed FFT along the fast-time axis."""
    window = _hamming_window(frame_config.numADCSamples, reference=reshaped_frame)
    if _is_torch_tensor(reshaped_frame):
        return torch.fft.fft(reshaped_frame * window.view(1, 1, 1, -1), dim=-1)
    return np.fft.fft(reshaped_frame * window, axis=-1)


def clutter_removal(input_val, axis=0):
    """Static clutter removal by subtracting the mean along one axis."""
    if _is_torch_tensor(input_val):
        return input_val - input_val.mean(dim=axis, keepdim=True)

    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    transposed = input_val.transpose(reordering)
    return (transposed - transposed.mean(0)).transpose(reordering)


def doppler_fft(range_result, frame_config):
    """Apply a Hamming-windowed FFT along slow time with fftshift."""
    window = _hamming_window(frame_config.numLoopsPerFrame, reference=range_result)
    if _is_torch_tensor(range_result):
        windowed = range_result * window.view(1, 1, -1, 1)
        return torch.fft.fftshift(torch.fft.fft(windowed, dim=2), dim=2)

    windowed = range_result * window.reshape(1, 1, -1, 1)
    return np.fft.fftshift(np.fft.fft(windowed, axis=2), axes=2)


def naive_xyz(virtual_ant, num_tx=3, num_rx=4, fft_size=64, tx_loc_hw=None):
    """Estimate direction cosines from virtual antenna data."""
    assert num_tx > 2, "Need > 2 TX antennas for 3D AoA estimation"

    if num_tx > 4:
        return _aoa_2d_fft(virtual_ant, num_tx, num_rx, fft_size)
    return _aoa_phase_comparison(virtual_ant, num_tx, num_rx, fft_size, tx_loc_hw=tx_loc_hw)


def _aoa_phase_comparison(virtual_ant, num_tx, num_rx, fft_size, tx_loc_hw=None):
    """Phase-comparison AoA for small arrays (for example 3TX 4RX)."""
    num_detected = virtual_ant.shape[1]
    n_az = min(2 * num_rx, fft_size)
    azimuth_ant = virtual_ant[:n_az, :]

    if _is_torch_tensor(virtual_ant):
        device = virtual_ant.device
        real_dtype = _real_dtype(virtual_ant)
        azimuth_padded = torch.zeros((fft_size, num_detected), dtype=virtual_ant.dtype, device=device)
        azimuth_padded[:n_az, :] = azimuth_ant
        azimuth_fft = torch.fft.fft(azimuth_padded, dim=0)
        k_max = torch.argmax(torch.abs(azimuth_fft), dim=0).to(torch.int64)
        peak_1 = azimuth_fft[k_max, torch.arange(num_detected, device=device)]
        signed_k_max = torch.where(k_max > (fft_size // 2) - 1, k_max - fft_size, k_max)
        wx = (2 * torch.pi / fft_size) * signed_k_max.to(real_dtype)
        x_vector = wx / torch.pi
    else:
        azimuth_padded = np.zeros((fft_size, num_detected), dtype=np.complex64)
        azimuth_padded[:n_az, :] = azimuth_ant
        azimuth_fft = np.fft.fft(azimuth_padded, axis=0)
        k_max = np.argmax(np.abs(azimuth_fft), axis=0).astype(np.int64, copy=False)
        peak_1 = azimuth_fft[k_max, np.arange(num_detected)]
        k_max[k_max > (fft_size // 2) - 1] -= fft_size
        wx = 2 * np.pi / fft_size * k_max
        x_vector = wx / np.pi

    el_start = 2 * num_rx
    n_el = min(num_rx, virtual_ant.shape[0] - el_start)
    elevation_ant = virtual_ant[el_start:el_start + n_el, :]

    if _is_torch_tensor(virtual_ant):
        device = virtual_ant.device
        real_dtype = _real_dtype(virtual_ant)
        elevation_padded = torch.zeros((fft_size, num_detected), dtype=virtual_ant.dtype, device=device)
        elevation_padded[:n_el, :] = elevation_ant
        elevation_fft = torch.fft.fft(elevation_padded, dim=0)
        elevation_power = torch.log2(torch.clamp(torch.abs(elevation_fft), min=1e-12))
        elevation_max = torch.argmax(elevation_power, dim=0).to(torch.int64)
        peak_2 = elevation_fft[elevation_max, torch.arange(num_detected, device=device)]
        el_tx_dx = float(tx_loc_hw[2][0] - tx_loc_hw[0][0]) if tx_loc_hw is not None else 2.0
        phase_adjust = torch.exp(1j * torch.tensor(el_tx_dx, dtype=real_dtype, device=device) * wx)
        wz = torch.angle(peak_1 * torch.conj(peak_2) * phase_adjust)
        z_vector = wz / torch.pi
        y_possible = 1 - x_vector.square() - z_vector.square()
        valid = y_possible >= 0
        x_vector = torch.where(valid, x_vector, torch.zeros_like(x_vector))
        z_vector = torch.where(valid, z_vector, torch.zeros_like(z_vector))
        y_vector = torch.sqrt(torch.clamp(y_possible, min=0.0))
        return x_vector, y_vector, z_vector

    elevation_padded = np.zeros((fft_size, num_detected), dtype=np.complex64)
    elevation_padded[:n_el, :] = elevation_ant
    elevation_fft = np.fft.fft(elevation_padded, axis=0)
    elevation_power = np.log2(np.maximum(np.abs(elevation_fft), 1e-12))
    elevation_max = np.argmax(elevation_power, axis=0)
    peak_2 = elevation_fft[elevation_max, np.arange(num_detected)]

    el_tx_dx = float(tx_loc_hw[2][0] - tx_loc_hw[0][0]) if tx_loc_hw is not None else 2.0
    wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * el_tx_dx * wx))
    z_vector = wz / np.pi

    y_possible = 1 - x_vector ** 2 - z_vector ** 2
    invalid = y_possible < 0
    x_vector = x_vector.copy()
    z_vector = z_vector.copy()
    y_vector = y_possible.copy()
    x_vector[invalid] = 0
    z_vector[invalid] = 0
    y_vector[invalid] = 0
    y_vector = np.sqrt(y_vector)
    return x_vector, y_vector, z_vector


def _aoa_2d_fft(virtual_ant, num_tx, num_rx, fft_size):
    """2D FFT AoA for larger virtual UPAs."""
    num_detected = virtual_ant.shape[1]
    n_az_per_row = 2 * num_rx
    n_el_rows = num_tx // 2
    reshaped = virtual_ant.reshape(num_tx, num_rx, num_detected)

    if _is_torch_tensor(virtual_ant):
        device = virtual_ant.device
        real_dtype = _real_dtype(virtual_ant)
        grid = torch.zeros((n_el_rows, n_az_per_row, num_detected), dtype=virtual_ant.dtype, device=device)
        grid[:, :num_rx, :] = reshaped[0::2]
        grid[:, num_rx:, :] = reshaped[1::2]

        padded = torch.zeros((fft_size, fft_size, num_detected), dtype=virtual_ant.dtype, device=device)
        padded[:n_el_rows, :n_az_per_row, :] = grid
        spectrum_2d = torch.fft.fft2(padded, dim=(0, 1))
        power_flat = torch.abs(spectrum_2d).reshape(-1, num_detected)
        peak_idx = torch.argmax(power_flat, dim=0).to(torch.int64)
        k_el = peak_idx // fft_size
        k_az = peak_idx % fft_size
        k_az = torch.where(k_az > fft_size // 2 - 1, k_az - fft_size, k_az)
        k_el = torch.where(k_el > fft_size // 2 - 1, k_el - fft_size, k_el)
        x_vector = (2 * torch.pi / fft_size) * k_az.to(real_dtype) / torch.pi
        z_vector = (2 * torch.pi / fft_size) * k_el.to(real_dtype) / torch.pi
        y_possible = 1 - x_vector.square() - z_vector.square()
        valid = y_possible >= 0
        x_vector = torch.where(valid, x_vector, torch.zeros_like(x_vector))
        z_vector = torch.where(valid, z_vector, torch.zeros_like(z_vector))
        y_vector = torch.sqrt(torch.clamp(y_possible, min=0.0))
        return x_vector, y_vector, z_vector

    grid = np.zeros((n_el_rows, n_az_per_row, num_detected), dtype=np.complex64)
    grid[:, :num_rx, :] = reshaped[0::2]
    grid[:, num_rx:, :] = reshaped[1::2]

    padded = np.zeros((fft_size, fft_size, num_detected), dtype=np.complex64)
    padded[:n_el_rows, :n_az_per_row, :] = grid
    spectrum_2d = np.fft.fft2(padded, axes=(0, 1))
    power_flat = np.abs(spectrum_2d).reshape(-1, num_detected)
    peak_idx = np.argmax(power_flat, axis=0)
    k_el = peak_idx // fft_size
    k_az = peak_idx % fft_size
    k_az[k_az > fft_size // 2 - 1] -= fft_size
    k_el[k_el > fft_size // 2 - 1] -= fft_size
    x_vector = (2 * np.pi / fft_size) * k_az / np.pi
    z_vector = (2 * np.pi / fft_size) * k_el / np.pi

    y_possible = 1 - x_vector ** 2 - z_vector ** 2
    invalid = y_possible < 0
    x_vector = x_vector.copy()
    z_vector = z_vector.copy()
    y_vector = y_possible.copy()
    x_vector[invalid] = 0
    z_vector[invalid] = 0
    y_vector[invalid] = 0
    y_vector = np.sqrt(y_vector)
    return x_vector, y_vector, z_vector


# ---------------------------------------------------------------------------
# TDM-MIMO phase compensation
# ---------------------------------------------------------------------------

def _compensate_tdm_phase(aoa_input, velocities, radar, fc):
    """Remove the velocity-dependent TDM-MIMO phase offset."""
    lam = radar._lambda
    t_chirp = (radar.idle_time + radar.ramp_end_time) * 1e-6
    num_tx = fc.numTxAntennas
    num_rx = fc.numRxAntennas

    compensated = aoa_input.clone() if _is_torch_tensor(aoa_input) else aoa_input.copy()
    for tx_i in range(1, num_tx):
        va_start = tx_i * num_rx
        va_end = va_start + num_rx
        if _is_torch_tensor(aoa_input):
            phase_offset = 4 * torch.pi * velocities * tx_i * t_chirp / lam
            compensation = torch.exp(-1j * phase_offset).unsqueeze(0)
            compensated[va_start:va_end, :] *= compensation
        else:
            phase_offset = 4 * np.pi * velocities * tx_i * t_chirp / lam
            compensation = np.exp(-1j * phase_offset)[np.newaxis, :]
            compensated[va_start:va_end, :] *= compensation

    return compensated


def _energy_topk_mask(values, top_k: int):
    total_bins = values.numel() if _is_torch_tensor(values) else values.size
    top_k = min(top_k, total_bins)
    if top_k <= 0:
        if _is_torch_tensor(values):
            return torch.zeros_like(values, dtype=torch.bool)
        return np.zeros_like(values, dtype=bool)
    if top_k >= total_bins:
        if _is_torch_tensor(values):
            return torch.ones_like(values, dtype=torch.bool)
        return np.ones_like(values, dtype=bool)

    if _is_torch_tensor(values):
        threshold = torch.topk(values.reshape(-1), top_k).values.min()
        return values >= threshold

    threshold = np.partition(values.ravel(), total_bins - top_k)[total_bins - top_k]
    return values >= threshold


def _transpose_point_cloud(point_cloud):
    if _is_torch_tensor(point_cloud):
        return point_cloud.transpose(0, 1)
    return np.transpose(point_cloud, (1, 0))


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def frame2pointcloud(frame, cfg, radar=None):
    """Convert a radar frame tensor to a point cloud."""
    if radar is None:
        raise ValueError("frame2pointcloud requires a radar instance so TDM-MIMO compensation is always applied.")

    fc = cfg.frame_config
    range_result = range_fft(frame, fc)
    if cfg.enable_static_clutter_removal:
        range_result = clutter_removal(range_result, axis=2)
    doppler_result = doppler_fft(range_result, fc)

    if _is_torch_tensor(doppler_result):
        doppler_sum = doppler_result.sum(dim=(0, 1))
        doppler_db = 20 * torch.log10(torch.abs(doppler_sum) + 1e-6)
    else:
        doppler_sum = np.sum(doppler_result, axis=(0, 1))
        doppler_db = 20 * np.log10(np.abs(doppler_sum) + 1e-6)

    if cfg.range_cut:
        doppler_db[:, :25] = -100
        doppler_db[:, 125:] = -100

    cfar_result = _energy_topk_mask(doppler_db, cfg.energy_top_k) if cfg.use_energy_top_k else (
        torch.zeros_like(doppler_db, dtype=torch.bool) if _is_torch_tensor(doppler_db) else np.zeros(doppler_db.shape, dtype=bool)
    )
    det_peaks = torch.argwhere(cfar_result) if _is_torch_tensor(cfar_result) else np.argwhere(cfar_result)
    if det_peaks.shape[0] == 0:
        return _empty_pointcloud(frame)

    if _is_torch_tensor(det_peaks):
        r = det_peaks[:, 1].to(torch.float64)
        v = (det_peaks[:, 0] - fc.numDopplerBins // 2).to(torch.float64)
        if cfg.output_in_meter:
            r = r * fc.range_resolution
            v = v * fc.doppler_resolution
    else:
        r = det_peaks[:, 1].astype(np.float64)
        v = (det_peaks[:, 0] - fc.numDopplerBins // 2).astype(np.float64)
        if cfg.output_in_meter:
            r *= fc.range_resolution
            v *= fc.doppler_resolution

    energy = doppler_db[cfar_result]
    aoa_input = doppler_result[:, :, cfar_result].reshape(fc.numTxAntennas * fc.numRxAntennas, -1)
    aoa_input = _compensate_tdm_phase(aoa_input, v, radar, fc)
    x_vec, y_vec, z_vec = naive_xyz(
        aoa_input,
        num_tx=fc.numTxAntennas,
        num_rx=fc.numRxAntennas,
        tx_loc_hw=fc.tx_loc_hw,
    )

    x, y, z = x_vec * r, y_vec * r, z_vec * r
    if _is_torch_tensor(x):
        point_cloud = torch.stack([x, y, z, v, energy.to(torch.float64), r], dim=0)
    else:
        point_cloud = np.stack([x, y, z, v, energy, r], axis=0)
    return point_cloud[:, y_vec != 0]


def process_pc(
    radar,
    frame,
    static_clutter_removal=True,
    positive_velocity_only=True,
    detector: DetectorType = "cfar",
    guard_cells=(2, 4),
    training_cells=(4, 8),
    pfa=1e-3,
    max_points=512,
    energy_top_k=128,
):
    """Radar frame -> filtered point cloud."""
    detector_kind = normalize_detector_type(detector)

    if detector_kind == "topk":
        cfg = PointCloudProcessConfig(
            radar,
            static_clutter_removal=static_clutter_removal,
            energy_top_k=energy_top_k,
        )
        pc = _transpose_point_cloud(frame2pointcloud(frame, cfg, radar=radar))
        if _is_torch_tensor(pc):
            if positive_velocity_only and pc.shape[0] > 0:
                pc = pc[pc[:, 3] > 0]
            return _to_numpy(pc)
        if positive_velocity_only and pc.shape[0] > 0:
            pc = pc[pc[:, 3] > 0]
        return pc

    return _process_pc_cfar(
        radar,
        frame,
        static_clutter_removal=static_clutter_removal,
        positive_velocity_only=positive_velocity_only,
        guard_cells=guard_cells,
        training_cells=training_cells,
        pfa=pfa,
        max_points=max_points,
    )


def _process_pc_cfar(
    radar,
    frame,
    static_clutter_removal=True,
    positive_velocity_only=True,
    guard_cells=(2, 4),
    training_cells=(4, 8),
    pfa=1e-3,
    max_points=512,
):
    """Internal: point cloud extraction via CA-CFAR detection."""
    from .cfar import ca_cfar_2d_fast

    fc = FrameConfig(radar)
    range_result = range_fft(frame, fc)
    if static_clutter_removal:
        range_result = clutter_removal(range_result, axis=2)
    doppler_result = doppler_fft(range_result, fc)

    if _is_torch_tensor(doppler_result):
        doppler_sum = doppler_result.sum(dim=(0, 1))
        doppler_mag = torch.abs(doppler_sum)
    else:
        doppler_sum = np.sum(doppler_result, axis=(0, 1))
        doppler_mag = np.abs(doppler_sum)

    cfar_mask, _ = ca_cfar_2d_fast(
        doppler_mag,
        guard_cells=guard_cells,
        training_cells=training_cells,
        pfa=pfa,
    )

    det_peaks = torch.argwhere(cfar_mask) if _is_torch_tensor(cfar_mask) else np.argwhere(cfar_mask)
    if det_peaks.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float64)

    if det_peaks.shape[0] > max_points:
        if _is_torch_tensor(det_peaks):
            energies = doppler_mag[det_peaks[:, 0], det_peaks[:, 1]]
            top_idx = torch.topk(energies, max_points).indices
            det_peaks = det_peaks[top_idx]
            cfar_mask = torch.zeros_like(cfar_mask, dtype=torch.bool)
            cfar_mask[det_peaks[:, 0], det_peaks[:, 1]] = True
        else:
            energies = doppler_mag[det_peaks[:, 0], det_peaks[:, 1]]
            top_idx = np.argsort(energies)[-max_points:]
            det_peaks = det_peaks[top_idx]
            cfar_mask = np.zeros_like(cfar_mask, dtype=bool)
            cfar_mask[det_peaks[:, 0], det_peaks[:, 1]] = True

    if _is_torch_tensor(det_peaks):
        r = det_peaks[:, 1].to(torch.float64) * fc.range_resolution
        v = (det_peaks[:, 0] - fc.numDopplerBins // 2).to(torch.float64) * fc.doppler_resolution
        energy = 20 * torch.log10(doppler_mag[cfar_mask] + 1e-6)
    else:
        r = det_peaks[:, 1].astype(np.float64) * fc.range_resolution
        v = (det_peaks[:, 0] - fc.numDopplerBins // 2).astype(np.float64) * fc.doppler_resolution
        energy = 20 * np.log10(doppler_mag[cfar_mask] + 1e-6)

    aoa_input = doppler_result[:, :, cfar_mask].reshape(fc.numTxAntennas * fc.numRxAntennas, -1)
    aoa_input = _compensate_tdm_phase(aoa_input, v, radar, fc)
    x_vec, y_vec, z_vec = naive_xyz(
        aoa_input,
        num_tx=fc.numTxAntennas,
        num_rx=fc.numRxAntennas,
        tx_loc_hw=fc.tx_loc_hw,
    )

    x, y, z = x_vec * r, y_vec * r, z_vec * r
    if _is_torch_tensor(x):
        pc = torch.stack([x, y, z, v, energy.to(torch.float64), r], dim=0)
        pc = pc[:, y_vec != 0].transpose(0, 1)
        if positive_velocity_only and pc.shape[0] > 0:
            pc = pc[pc[:, 3] > 0]
        return _to_numpy(pc)

    pc = np.stack([x, y, z, v, energy, r], axis=0)
    pc = pc[:, y_vec != 0].T
    if positive_velocity_only and pc.shape[0] > 0:
        pc = pc[pc[:, 3] > 0]
    return pc


def process_rd(radar, frame, tx=0, rx=0, *, static_clutter_removal=False):
    """Compute a Range-Doppler map from a MIMO frame."""
    if _is_torch_tensor(frame):
        data = frame[tx, rx].clone()
        data = data - data.mean(dim=-1, keepdim=True)
        if static_clutter_removal:
            data = data - data.mean(dim=-2, keepdim=True)

        n_adc = data.shape[-1]
        n_chirps = data.shape[-2]
        data = data * _hamming_window(n_adc, reference=data).view(1, -1)
        data = data * _hamming_window(n_chirps, reference=data).view(-1, 1)
        range_result = torch.fft.fft(data, dim=-1)
        rd_map = torch.fft.fftshift(torch.fft.fft(range_result, dim=-2), dim=-2)
        rd_mag = 20 * torch.log10(torch.abs(rd_map) + 1e-6)
        return (
            _to_numpy(rd_mag),
            _to_numpy(rd_map),
            radar.ranges.detach().cpu().numpy(),
            radar.velocities.detach().cpu().numpy(),
        )

    data = np.copy(frame[tx, rx])
    data = data - np.mean(data, axis=-1, keepdims=True)
    if static_clutter_removal:
        data = data - np.mean(data, axis=-2, keepdims=True)

    n_adc = data.shape[-1]
    n_chirps = data.shape[-2]
    data = data * np.hamming(n_adc)
    data = data * np.hamming(n_chirps)[:, None]
    range_result = np.fft.fft(data, axis=-1)
    rd_map = np.fft.fftshift(np.fft.fft(range_result, axis=-2), axes=-2)
    rd_mag = 20 * np.log10(np.abs(rd_map) + 1e-6)
    return rd_mag, rd_map, radar.ranges.cpu().numpy(), radar.velocities.cpu().numpy()


def reg_data(data, pc_size):
    """Regularize a point cloud to a fixed size by sampling or duplication."""
    pc_tmp = np.zeros((pc_size, data.shape[1]), dtype=np.float32)
    pc_no = data.shape[0]
    if pc_no == 0:
        return pc_tmp
    if pc_no < pc_size:
        fill_list = np.random.choice(pc_size, size=pc_no, replace=False)
        fill_set = set(fill_list)
        pc_tmp[fill_list] = data
        dupl_list = [x for x in range(pc_size) if x not in fill_set]
        dupl_pc = np.random.choice(pc_no, size=len(dupl_list), replace=True)
        pc_tmp[dupl_list] = data[dupl_pc]
    else:
        pc_tmp = data[np.random.choice(pc_no, size=pc_size, replace=False)]
    return pc_tmp
