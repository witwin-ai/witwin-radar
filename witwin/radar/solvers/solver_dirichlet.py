"""
Dirichlet kernel solver backend.

Computes the range spectrum directly in the frequency domain and keeps all
backend-specific state on the solver instance.

This module is orchestration only. The ``dirichlet_spectrum`` native family -
its operator names, its autograd bridge, its output buffers, and its
configuration scalars - is owned by
:mod:`witwin.radar.synthesis.dirichlet_spectrum`, because a waveform synthesis
backend belongs in the synthesis domain. What is left here is the
``pad_factor`` / ``N_fft`` / ``num_bins`` / ``n`` / ``k0_per_meter`` derivation,
the slot-group chunking, the ``torch.fft.ifft``, and the public method surface.
"""

from __future__ import annotations

import torch

from ..path_cache import MimoPathCache
from ..sensors.legacy_paths import (
    LegacySensorContext,
    evaluate_pair_rows,
    evaluate_slot_rows,
)
from ..synthesis.dirichlet_spectrum import (
    DirichletSpectrumSpec,
    MimoLinearFramePlan,
    chunked_spectra,
    mimo_linear_spectra,
    spectrum_vjp,
    spectrum_vjp_per_bin,
)
from . import Solver
from .common import (
    _stack_slot_samples,
    normalize_interpolated_sample,
    samples_require_grad,
)


def _load_module():
    from ..cuda import build

    return build.build_extension()


def collect_interpolated_samples(radar, interpolator, t0=0.0):
    """Evaluate the scene interpolator once per TDM chirp slot.

    This is a HOST LOOP and it stays one: an interpolator call is scene
    authoring, not path physics, and the loop runs over
    ``chirp_per_frame * num_tx`` slots rather than over paths. It lives here
    rather than in ``solvers/common.py`` because this solver is its only caller
    and the shared module is no longer allowed to grow a per-frame routine.

    TDM-MIMO fires TX antennas sequentially: slot ``chirp_id * num_tx + tx_id``
    starts ``slot * chirp_period`` into the frame. The returned list holds
    ``chirp_per_frame * num_tx`` samples in slot order, so per-TX motion phase
    (the phase ``_compensate_tdm_phase`` removes downstream) is simulated.
    """

    cfg = radar.config
    chirp_period = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
    samples = []
    for slot in range(cfg.chirp_per_frame * cfg.num_tx):
        sample = interpolator(t0 + slot * chirp_period)
        samples.append(normalize_interpolated_sample(sample, device=radar.device))
    return samples


def _to_f32(solver: "DirichletSolver", value: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=torch.float32, device=solver.device).contiguous()


def _same_tensor_storage(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.shape == b.shape and a.data_ptr() == b.data_ptr()


def _samples_share_storage(samples) -> bool:
    if not samples:
        return False
    first = samples[0]
    return all(
        _same_tensor_storage(first.points, sample.points)
        and _same_tensor_storage(first.intensities, sample.intensities)
        and _same_tensor_storage(first.entry_points, sample.entry_points)
        and _same_tensor_storage(first.fixed_path_lengths, sample.fixed_path_lengths)
        and _same_tensor_storage(first.depths, sample.depths)
        and _same_tensor_storage(first.normals, sample.normals)
        for sample in samples[1:]
    )


def _reshape_pair_targets(value: torch.Tensor, num_pairs: int, n_targets: int) -> torch.Tensor:
    return value.reshape(num_pairs, n_targets).contiguous()


def _spectrum_spec(
    solver: "DirichletSolver",
    *,
    num_bins: int,
    n_fft: int,
    k0_per_meter: float,
    tau_is_seconds: int = 0,
) -> DirichletSpectrumSpec:
    """Snapshot the configuration scalars the kernels read, by value.

    Read once, here, at the call site. The autograd context this replaced kept
    a live solver reference and reached through it to ``radar.config`` inside
    ``backward``, so a config mutated between the forward and the backward
    produced the gradient of a different function than the one evaluated.

    ``tau_is_seconds`` and ``k0_per_meter`` co-vary and must be taken from the
    same pair. ``0`` is the legacy public surface - ``chirp`` and ``chirp_mimo``
    take a ONE-WAY distance in metres. ``1`` is what every internal route uses,
    because the sensor-weight owner publishes a ROUND-TRIP delay and turning one
    back into a distance so the kernel can halve it again is how a path becomes
    self-consistently 2x wrong.
    """

    cfg = solver.radar.config
    return DirichletSpectrumSpec(
        n=solver.n,
        k0_per_meter=k0_per_meter,
        num_bins=num_bins,
        n_fft=n_fft,
        fc=cfg.fc,
        slope_hz_per_s=cfg.slope * 1e12,
        t_start_s=cfg.adc_start_time * 1e-6,
        tau_is_seconds=tau_is_seconds,
    )


def _delay_spectra(
    solver: "DirichletSolver",
    delays: torch.Tensor,
    weights: torch.Tensor,
    *,
    targets_per_spectrum: int,
    num_bins: int,
    n_fft: int,
    k0_per_second: float,
    shared_gradient: bool = False,
) -> torch.Tensor:
    """Chunked spectra of a ROUND-TRIP delay row set with a COMPLEX weight.

    Every internal route goes through here. The delay comes from the
    ``sensor_weight`` kernel and the weight is complex because a Channel
    coefficient is; a legacy real amplitude is the special case with a zero
    imaginary part, which the family evaluates as a separate accumulation of
    exactly zero.
    """

    tau = delays.reshape(-1)
    weight = weights.reshape(-1)
    if tau.shape != weight.shape:
        raise ValueError("delays and weights must have the same shape.")
    return chunked_spectra(
        tau,
        weight.real.contiguous(),
        weight.imag.contiguous(),
        spec=_spectrum_spec(
            solver,
            num_bins=num_bins,
            n_fft=n_fft,
            k0_per_meter=k0_per_second,
            tau_is_seconds=1,
        ),
        targets_per_spectrum=targets_per_spectrum,
        shared_gradient=shared_gradient,
    )


def _native_chunked_spectra(
    solver: "DirichletSolver",
    distances: torch.Tensor,
    amplitudes: torch.Tensor,
    *,
    targets_per_spectrum: int,
    num_bins: int,
    n_fft: int,
    k0_per_meter: float,
    shared_gradient: bool = False,
) -> torch.Tensor:
    """Normalize to flat float32 device tensors, then hand off to the owner.

    There is no ``requires_grad`` shortcut. The one that used to live here
    bypassed autograd whenever no input required grad, and an ADR-038
    forward-only dual has ``requires_grad == False`` - so the shortcut swallowed
    its tangent and returned a plain tensor that looked correct. The
    normalization is outside ``Function.apply`` on purpose: ``to`` and
    ``reshape`` are differentiable, so a caller that passes float64 or a
    non-flat tensor gets its gradient back in its own dtype and shape without
    the autograd bridge having to remember either.
    """

    d = _to_f32(solver, distances.reshape(-1))
    a = _to_f32(solver, amplitudes.reshape(-1))
    if d.shape != a.shape:
        raise ValueError("distances and amplitudes must have the same shape.")
    return chunked_spectra(
        d,
        a,
        spec=_spectrum_spec(
            solver, num_bins=num_bins, n_fft=n_fft, k0_per_meter=k0_per_meter
        ),
        targets_per_spectrum=targets_per_spectrum,
        shared_gradient=shared_gradient,
    )


def spectrum(
    solver: "DirichletSolver",
    distances,
    amplitudes,
    *,
    targets_per_chunk: int = 256,
    num_bins: int | None = None,
    n_fft: int | None = None,
    k0_per_meter: float | None = None,
):
    """Dirichlet spectrum with optional overrides for MIMO-mode calls."""
    num_bins = solver.num_bins if num_bins is None else num_bins
    n_fft = solver.N_fft if n_fft is None else n_fft
    k0_per_meter = solver.k0_per_meter if k0_per_meter is None else k0_per_meter

    num_targets = distances.shape[0]
    d = _to_f32(solver, distances)
    a = _to_f32(solver, amplitudes)

    if num_targets == 0:
        return torch.zeros(num_bins, dtype=torch.complex64, device=solver.device)
    chunks = _native_chunked_spectra(
        solver,
        d,
        a,
        targets_per_spectrum=targets_per_chunk,
        num_bins=num_bins,
        n_fft=n_fft,
        k0_per_meter=k0_per_meter,
        shared_gradient=True,
    )
    return chunks.sum(dim=0)


def backward(solver: "DirichletSolver", distances, amplitudes, grad_output_re, grad_output_im):
    """Backward pass with one block per target and parallel bin reduction.

    The real-amplitude surface returns two gradients. The owner returns three,
    because the weight is complex; the imaginary-part gradient of a weight that
    was real is not discarded information, it is identically the part of the
    cotangent that a real amplitude cannot receive.
    """

    d = _to_f32(solver, distances)
    a = _to_f32(solver, amplitudes)
    g_re = grad_output_re.to(dtype=torch.float32, device=solver.device).contiguous()
    g_im = grad_output_im.to(dtype=torch.float32, device=solver.device).contiguous()

    grad_d, grad_a, _ = spectrum_vjp(
        d,
        a,
        g_re,
        g_im,
        spec=_spectrum_spec(
            solver,
            num_bins=solver.num_bins,
            n_fft=solver.N_fft,
            k0_per_meter=solver.k0_per_meter,
        ),
    )
    return grad_d, grad_a


def backward_per_bin(
    solver: "DirichletSolver",
    distances,
    amplitudes,
    grad_output_re,
    grad_output_im,
    *,
    bins_per_chunk: int = 256,
):
    """Backward pass with one thread per spectrum bin."""
    d = _to_f32(solver, distances)
    a = _to_f32(solver, amplitudes)
    g_re = grad_output_re.to(dtype=torch.float32, device=solver.device).contiguous()
    g_im = grad_output_im.to(dtype=torch.float32, device=solver.device).contiguous()

    grad_d, grad_a, _ = spectrum_vjp_per_bin(
        d,
        a,
        g_re,
        g_im,
        spec=_spectrum_spec(
            solver,
            num_bins=solver.num_bins,
            n_fft=solver.N_fft,
            k0_per_meter=solver.k0_per_meter,
        ),
        bins_per_chunk=bins_per_chunk,
    )
    return grad_d, grad_a


class DirichletSolver(Solver):
    """Direct frequency-domain spectrum solver."""

    def __init__(self, radar, pad_factor: int = 16):
        super().__init__(radar)
        self._cuda_module = None

        cfg = radar.config
        fs = cfg.sample_rate * 1e3
        slope_hz = cfg.slope * 1e12

        self.pad_factor = int(pad_factor)
        self.N_fft = cfg.adc_samples * self.pad_factor
        self.num_bins = self.N_fft // 2
        self.n = (cfg.adc_samples - 1) / 2
        self.k0_per_meter = (slope_hz * 2 / radar.c0) * self.N_fft / fs

        self.mimo_N_fft = cfg.adc_samples
        self.mimo_num_bins = cfg.adc_samples
        self.mimo_k0_per_meter = (slope_hz * 2 / radar.c0) * self.mimo_N_fft / fs

        # The same beat-bin scale expressed against a ROUND-TRIP DELAY rather
        # than a one-way distance: k0 = f_beat * n_fft / fs with
        # f_beat = slope * tau. It is the identical number
        # (tau = 2 d / c0), written in the units the sensor-weight owner
        # publishes so that no caller has to reconstruct a distance.
        self.k0_per_second = slope_hz * self.N_fft / fs
        self.mimo_k0_per_second = slope_hz * self.mimo_N_fft / fs

    @property
    def _module(self):
        if self.device.type != "cuda":
            raise RuntimeError("The Dirichlet solver requires CUDA tensors; construct Radar with device='cuda'.")
        if self._cuda_module is None:
            self._cuda_module = _load_module()
        return self._cuda_module

    def chirp(self, distances, amplitudes):
        """High-resolution Dirichlet spectrum (pad_factor bins)."""
        return spectrum(self, distances, amplitudes)

    def chirp_mimo(self, distances, amplitudes):
        """Dirichlet spectrum at adc_samples resolution for MIMO output."""
        return spectrum(
            self,
            distances,
            amplitudes,
            num_bins=self.mimo_num_bins,
            n_fft=self.mimo_N_fft,
            k0_per_meter=self.mimo_k0_per_meter,
        )

    def frame(self, interpolator, t0=0):
        r = self.radar
        cfg = r.config
        chirp_period = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
        context = LegacySensorContext.from_radar(r, tx_count=1, rx_count=1)

        result = []
        for chirp_id in range(cfg.chirp_per_frame):
            time_in_frame = chirp_id * chirp_period * cfg.num_tx
            sample = normalize_interpolated_sample(interpolator(t0 + time_in_frame), device=self.device)
            if sample.points.shape[0] == 0:
                result.append(
                    torch.zeros(self.num_bins, dtype=torch.complex64, device=self.device)
                )
                continue
            weights = evaluate_pair_rows(context, sample)
            chunks = _delay_spectra(
                self,
                weights.total_delay_s,
                weights.weight,
                targets_per_spectrum=256,
                num_bins=self.num_bins,
                n_fft=self.N_fft,
                k0_per_second=self.k0_per_second,
                shared_gradient=True,
            )
            result.append(chunks.sum(dim=0))

        return torch.stack(result)

    def mimo(self, interpolator, t0=0, **options):
        """Generate a MIMO frame; each TX samples the scene at its TDM slot time."""
        freq_domain = self._pop_bool_option(options, "freq_domain", False)
        fast = self._pop_bool_option(options, "fast", False)
        self._ensure_no_options(options)

        r = self.radar
        samples = collect_interpolated_samples(r, interpolator, t0)
        if fast and not samples_require_grad(samples) and _samples_share_storage(samples):
            return self._mimo_from_sample_static(samples[0], freq_domain=freq_domain)

        frame = self._mimo_from_slot_samples(samples, freq_domain=freq_domain)
        return frame

    # Slot-group size cap: bounds the (slots, RX, N, 3) row transients handed to
    # the sensor-weight kernel to roughly 256 MB of float32.
    _SLOT_GROUP_ELEMENT_BUDGET = 1 << 26

    def _mimo_from_slot_samples(self, samples, *, freq_domain: bool):
        """One sensor-weight launch and one spectrum launch per slot group.

        The row set is ``(slot, rx, path)`` in that order, so the spectrum's
        chunk axis is ``slot * num_rx + rx`` - the same partition the previous
        Torch expression produced, reshaped rather than permuted.
        """

        r = self.radar
        cfg = r.config
        num_rx = cfg.num_rx
        num_slots = len(samples)
        n_max = max(int(sample.points.shape[0]) for sample in samples)
        if n_max == 0:
            return torch.zeros(
                (cfg.num_tx, cfg.num_rx, cfg.chirp_per_frame, cfg.adc_samples),
                dtype=torch.complex64,
                device=self.device,
            )

        context = LegacySensorContext.from_radar(r)
        spectrum_groups = []
        group = max(1, self._SLOT_GROUP_ELEMENT_BUDGET // max(1, n_max * num_rx * 3))
        for start in range(0, num_slots, group):
            stop = min(start + group, num_slots)
            packed = _stack_slot_samples(
                samples[start:stop], with_normals=context.uses_polarization
            )
            if packed is None:
                spectrum_groups.append(
                    torch.zeros(
                        ((stop - start) * num_rx, self.mimo_num_bins),
                        dtype=torch.complex64,
                        device=self.device,
                    )
                )
                continue
            weights = evaluate_slot_rows(context, packed, first_slot=start)
            group_n = int(packed[0].shape[1])
            spectrum_groups.append(
                _delay_spectra(
                    self,
                    weights.total_delay_s,
                    weights.weight,
                    targets_per_spectrum=group_n,
                    num_bins=self.mimo_num_bins,
                    n_fft=self.mimo_N_fft,
                    k0_per_second=self.mimo_k0_per_second,
                )
            )

        spectra = torch.cat(spectrum_groups, dim=0)
        if not freq_domain:
            spectra = torch.fft.ifft(spectra, dim=-1)
        return (
            spectra.view(cfg.chirp_per_frame, cfg.num_tx, num_rx, cfg.adc_samples)
            .permute(1, 2, 0, 3)
            .contiguous()
        )

    def mimo_from_trace(self, trace, *, velocities=None, t0=0.0, **options):
        """Generate a MIMO frame from one pre-traced frame.

        ``velocities`` is optional per-path world velocity with shape ``(N, 3)``.
        When omitted, the trace is treated as static across all chirps.
        """
        freq_domain = self._pop_bool_option(options, "freq_domain", False)
        amplitude_update = options.pop("amplitude_update", "range_loss")
        self._ensure_no_options(options)
        sample = normalize_interpolated_sample(trace, device=self.device)
        if velocities is None:
            cache = self._path_cache_from_sample(sample)
            return self.mimo_from_paths(cache, freq_domain=freq_domain, amplitude_update=amplitude_update)

        velocity_t = torch.as_tensor(velocities, dtype=torch.float32, device=self.device)
        if velocity_t.shape != sample.points.shape:
            raise ValueError(
                "velocities must have shape (N, 3), matching trace.points; "
                f"got {tuple(velocity_t.shape)} for {tuple(sample.points.shape)}."
            )
        cache = self._path_cache_from_sample(sample, velocity_t.contiguous())
        return self.mimo_from_paths(cache, freq_domain=freq_domain, amplitude_update=amplitude_update)

    def path_cache_from_trace(self, trace, *, velocities=None) -> MimoPathCache:
        """Precompute one frame's per-pair distances, amplitudes, and optional rates."""
        sample = normalize_interpolated_sample(trace, device=self.device)
        velocity_t = None
        if velocities is not None:
            velocity_t = torch.as_tensor(velocities, dtype=torch.float32, device=self.device).contiguous()
            if velocity_t.shape != sample.points.shape:
                raise ValueError(
                    "velocities must have shape (N, 3), matching trace.points; "
                    f"got {tuple(velocity_t.shape)} for {tuple(sample.points.shape)}."
                )
        return self._path_cache_from_sample(sample, velocity_t)

    def _path_cache_from_sample(self, sample, velocities: torch.Tensor | None = None) -> MimoPathCache:
        """One sensor-weight launch produces the whole cache.

        Delay, delay rate, and the complex weight all come out of the same
        kernel evaluation, so they cannot describe three slightly different
        geometries the way three independent Torch expressions could.
        """

        r = self.radar
        cfg = r.config
        shape = (cfg.num_tx, cfg.num_rx, int(sample.points.shape[0]))
        context = LegacySensorContext.from_radar(r)
        weights = evaluate_pair_rows(context, sample, velocities=velocities)
        return MimoPathCache(
            total_delay_s=weights.total_delay_s.view(shape).contiguous(),
            amplitudes=weights.weight.view(shape).contiguous(),
            delay_rate=(
                None if velocities is None else weights.delay_rate.view(shape).contiguous()
            ),
        )

    def mimo_from_paths(
        self,
        cache: MimoPathCache,
        *,
        freq_domain: bool = False,
        amplitude_update: str = "range_loss",
    ):
        if amplitude_update not in {"constant", "range_loss"}:
            raise ValueError("amplitude_update must be 'constant' or 'range_loss'.")
        delays = cache.total_delay_s.to(dtype=torch.float32, device=self.device).contiguous()
        amplitudes = cache.amplitudes.to(dtype=torch.complex64, device=self.device).contiguous()
        if delays.shape != amplitudes.shape or delays.ndim != 3:
            raise ValueError(
                "MimoPathCache total_delay_s and amplitudes must both have shape "
                "(num_tx, num_rx, num_paths)."
            )
        cfg = self.radar.config
        expected_shape = (cfg.num_tx, cfg.num_rx)
        if delays.shape[:2] != expected_shape:
            raise ValueError(
                "MimoPathCache antenna dimensions must match radar config; "
                f"got {tuple(delays.shape[:2])}, expected {expected_shape}."
            )
        n_targets = delays.shape[2]
        if n_targets == 0:
            return torch.zeros(
                (cfg.num_tx, cfg.num_rx, cfg.chirp_per_frame, cfg.adc_samples),
                dtype=torch.complex64,
                device=self.device,
            )
        num_pairs = cfg.num_tx * cfg.num_rx
        if cache.delay_rate is None:
            spectra = self._mimo_spectra_static(
                delays.reshape(-1).contiguous(),
                amplitudes.reshape(-1).contiguous(),
                num_pairs,
                n_targets,
            )
            if not freq_domain:
                spectra = torch.fft.ifft(spectra, dim=-1)
            chirp = spectra.view(cfg.num_tx, cfg.num_rx, cfg.adc_samples)
            return chirp.unsqueeze(2).expand(-1, -1, cfg.chirp_per_frame, -1).clone()

        rates = cache.delay_rate.to(dtype=torch.float32, device=self.device).contiguous()
        if rates.shape != delays.shape:
            raise ValueError("MimoPathCache delay_rate must match total_delay_s shape.")
        return self._mimo_from_path_tensors_linear(
            delays,
            amplitudes,
            rates,
            freq_domain=freq_domain,
            amplitude_update=amplitude_update,
        )

    def _mimo_from_sample_static(self, sample, *, freq_domain: bool):
        r = self.radar
        cfg = r.config
        num_pairs = cfg.num_tx * cfg.num_rx
        n_targets = sample.points.shape[0]
        if n_targets == 0:
            return torch.zeros(
                (cfg.num_tx, cfg.num_rx, cfg.chirp_per_frame, cfg.adc_samples),
                dtype=torch.complex64,
                device=self.device,
            )

        context = LegacySensorContext.from_radar(r)
        weights = evaluate_pair_rows(context, sample)
        spectra = self._mimo_spectra_static(
            weights.total_delay_s, weights.weight, num_pairs, n_targets
        )
        if not freq_domain:
            spectra = torch.fft.ifft(spectra, dim=-1)
        chirp = spectra.view(cfg.num_tx, cfg.num_rx, cfg.adc_samples)
        return chirp.unsqueeze(2).expand(-1, -1, cfg.chirp_per_frame, -1).clone()

    def _mimo_spectra_static(
        self,
        all_tau: torch.Tensor,
        all_weight: torch.Tensor,
        num_pairs: int,
        n_targets: int,
        *,
        targets_per_chunk: int = 256,
    ) -> torch.Tensor:
        chunks_per_pair = (n_targets + targets_per_chunk - 1) // targets_per_chunk
        n_padded = chunks_per_pair * targets_per_chunk
        if n_padded != n_targets:
            tau = torch.zeros((num_pairs, n_padded), dtype=torch.float32, device=self.device)
            weight = torch.zeros(
                (num_pairs, n_padded), dtype=torch.complex64, device=self.device
            )
            tau[:, :n_targets] = all_tau.view(num_pairs, n_targets)
            weight[:, :n_targets] = all_weight.view(num_pairs, n_targets)
            all_tau = tau.reshape(-1)
            all_weight = weight.reshape(-1)

        spectra = _delay_spectra(
            self,
            all_tau,
            all_weight,
            targets_per_spectrum=targets_per_chunk,
            num_bins=self.mimo_num_bins,
            n_fft=self.mimo_N_fft,
            k0_per_second=self.mimo_k0_per_second,
        )
        return spectra.view(num_pairs, chunks_per_pair, self.mimo_num_bins).sum(dim=1)

    def _mimo_from_path_tensors_linear(
        self,
        delays: torch.Tensor,
        amplitudes: torch.Tensor,
        rates: torch.Tensor,
        *,
        freq_domain: bool,
        amplitude_update: str,
    ):
        """One fused native launch over the whole TDM frame, in every AD mode.

        There is no ``requires_grad`` branch. The Torch expression that used to
        sit behind one - ``dist = d0 + rate * t`` with the ``d0 / dist``
        range-loss update, expanded to a ``chirps x pairs x targets``
        intermediate and pushed through the chunked forward - was a second
        implementation of ``forward_mimo_linear_chunked``'s physics, so a
        gradient came from different code than the value it was the gradient of.
        ``mimo_linear_backward`` and ``mimo_linear_jvp`` replace it.
        """

        cfg = self.radar.config
        n_targets = delays.shape[2]
        chirp_period = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
        weight = amplitudes.reshape(-1)
        spectra = mimo_linear_spectra(
            delays.reshape(-1).contiguous(),
            rates.reshape(-1).contiguous(),
            weight.real.contiguous(),
            weight.imag.contiguous(),
            spec=_spectrum_spec(
                self,
                num_bins=self.mimo_num_bins,
                n_fft=self.mimo_N_fft,
                k0_per_meter=self.mimo_k0_per_second,
                tau_is_seconds=1,
            ),
            plan=MimoLinearFramePlan(
                targets_per_pair=n_targets,
                num_pairs=cfg.num_tx * cfg.num_rx,
                chirp_per_frame=cfg.chirp_per_frame,
                chirp_period_s=chirp_period,
                num_tx=cfg.num_tx,
                range_loss_update=amplitude_update == "range_loss",
            ),
        )
        if not freq_domain:
            spectra = torch.fft.ifft(spectra, dim=-1)
        return spectra.view(cfg.chirp_per_frame, cfg.num_tx, cfg.num_rx, cfg.adc_samples).permute(1, 2, 0, 3).contiguous()

    def backward(self, distances, amplitudes, grad_output_re, grad_output_im):
        return backward(self, distances, amplitudes, grad_output_re, grad_output_im)

    def backward_per_bin(self, distances, amplitudes, grad_output_re, grad_output_im, bins_per_chunk: int = 256):
        return backward_per_bin(
            self,
            distances,
            amplitudes,
            grad_output_re,
            grad_output_im,
            bins_per_chunk=bins_per_chunk,
        )
