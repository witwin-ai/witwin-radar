# R-ADR-054: Back-projection imaging (SAR/ISAR) and calibration against a known scene

Status: Accepted (0.5.0). Amends R-ADR-017 sections 1, 4 and 6 (the processing
surface and the native-DSP gate) after R-ADR-027 and R-ADR-028.

## Context

The processing package (R-ADR-017, R-ADR-027, R-ADR-028) stopped at the angle
estimate and the point cloud. Two of the common radar products it lacked were
imaging from a moving radar or a turning target (SAR/ISAR), and the
calibrations a recorded array needs before its angles or ranges mean what the
axes say: per-channel gain and phase, a round-trip delay bias, and the static
background (transmit leakage, static clutter). A per-pixel image is not a few
library calls: the Torch formulation materialises the whole `[elements, slow,
pixels]` geometry and gathers from it.

## Decision

### 1. One imaging algorithm: time-domain back-projection

`witwin/radar/processing/imaging.py` owns it. Symbols and units:

| symbol | meaning | unit |
|---|---|---|
| `p` | pixel, in the aperture's frame (`world`, or a turning target's `body`) | m |
| `tx_cv`, `rx_cv` | phase centres of virtual element `v` (TX major) at slow-time sample `c` | m |
| `L = |p - tx| + |p - rx|`, `tau = L / c0` | two-way path, round-trip delay | m, s |
| `k = (L / 2 - r_0) / dr` | fractional bin on the profile's published range axis | - |
| `X_cv(k)` | range-compressed sample, linear interpolation between `floor(k)` and `floor(k) + 1`; zero off the axis | sqrt(W) |
| `(s, f, a)` | phase law: sign from the cube's phasor, carrier and curvature | -, Hz, Hz/s |

    I(p) = 1/(C V) sum_{c, v} X_cv(k) exp(-j s 2 pi (f tau - a tau^2))

Phase law, from each waveform's own synthesis convention:

* FMCW (beat phasor, `s = +1`): the de-chirped tone has phase `(f_ref + f_0)
  tau + S tau (t_adc - tau / 2)` at the first ADC sample (`cuda/fmcw_phase.cuh`),
  and a 1/N DFT evaluated at the tone's own frequency returns the first
  sample's phase, so `f = f_ref + f_0 + S t_adc`, `a = S / 2` (the residual
  video phase, 0.15 cycles at 100 ns and 60 MHz/us - not negligible).
* OFDM (Channel phasor, `s = -1`): `f = f_ref`, `a = 0`.
* Pulsed (Channel phasor): `f = f_ref`, `a = 0`; a stepped carrier is refused.
* Refused by name: chirp sequences (no single law), DDM/BPM before
  `demultiplex`, stepped pulses.

Normalisation follows `PROCESSING_AMPLITUDE_CONVENTION`: a point target images
at its compressed amplitude (times the range window's coherent gain, less the
linear-interpolation loss). The profile is the interpolation grid, so the
caller zero pads (`range_profile(range_bins=8 N)`): on the transform's own
grid neighbouring bins of an FMCW tone are close to antiphase and linear
interpolation cancels them; at 8x the loss is at most about 2.5 % and the
phase error about 1 mrad.

The `Aperture` record holds `tx_m`, `rx_m` (`[C, V, 3]` float64), `times_s`
and the phase law. `Aperture.from_radar(radar, time_s=, velocity_mps= | pose=)`
derives the sample instants from the waveform (FMCW TDM slot `chirp * num_tx +
tx` at its ramp start plus its ADC start; an OFDM symbol; a PRI) and moves the
authored `Radar.tx_pos` / `rx_pos` along a straight track or with the verbs'
pose, `p(t) = c + T(t) + R(t) (p - c)` about the authored position (the same
statement as the session's `_posed`). The start-stop model is used: motion
inside one chirp is not modelled. `isar_aperture` maps every element into a
turning target's body frame, `c + R(t)^T (x - c)`, for a constant spin vector
or a per-sample rotation table, so ISAR is the same sum.

### 2. Native, differentiable, with a Torch route on the CPU

The `backprojection` family (`cuda/processing_imaging.cu`): one thread per
`(batch, group, pixel)`, path, delay and phase in float64 (77 GHz is thousands
of cycles at a few metres; a float32 path at 10 m is a 1 um ulp), the phase
wrapped to its fractional cycle before `sincospi`, samples interpolated and
accumulated in their own precision. The map is linear in the samples: the JVP
is the forward map of the tangent (the same kernel under the `_jvp` symbol)
and the backward is its adjoint, `dX[i0] += scale (1 - f) conj(w) G`,
`dX[i0 + 1] += scale f conj(w) G`, accumulated with atomics. The geometry is a
host declaration: a differentiable aperture or pixel grid is refused at entry
(R-ADR-018). `groups` is 1 (one image) or `V` (one response per element, which
`element_responses` and the array calibration read).

A CPU profile runs the same equation in Torch primitives (vectorised over
elements, samples and pixels, one gather per tap); it is the oracle the GPU
tests compare against and is differentiable by autograd.

Counts (same process, GPU 1 contended; `[1, 8, 64, 2048]` complex128 samples,
2x4 array, 128 x 128 pixels; minimum of six runs):

| route | kernel launches | peak extra memory | time |
|---|---|---|---|
| native forward | 1 | 0.2 MB | 4.29 ms |
| Torch route on the same CUDA tensors | 38 | 1160 MB | 15.30 ms |
| native VJP | 1 (+ zero fill) | the gradient | 4.67 ms |

No host synchronisation and no H2D/D2H copy inside an image when the aperture
and pixels are on the device (`Aperture.to`; pinned by
`test_a_device_image_is_one_launch_with_no_host_round_trip`).

### 3. Calibration (`witwin/radar/processing/calibration.py`)

* **Array.** `m_v` = the element response at a known reflector (the
  per-element back-projection, slow-time mean). `g_v = m_v / (mean |m| exp(j
  arg m_ref))`: unit mean magnitude, zero phase on the reference channel.
  `ArrayCalibration.apply` divides a `ProcessingCube`, `RangeProfile` or
  `RangeDopplerMap`'s `[TX, RX]` axes by `g`. Whatever differs between channels
  toward the reflector (port gain and phase, line lengths, the element pattern
  in that direction) is absorbed.
* **Range bias.** Per element and sample, the compressed peak within
  `search_m` of the reflector's geometric range is refined by a parabola
  through `|X|^2`; the mean offset is `c tau_b / 2`. `RangeCalibration.apply`
  moves the axes' range origin by `-c tau_b / 2` and leaves the samples alone.
  The carrier phase `f tau_b` is common and absorbed by the array calibration.
  The estimate is an argmax and refuses a derivative.
* **Background.** `subtract_background(product, reference)` subtracts a
  target-free reference (or `average_background` of several) sample for
  sample; leakage and static clutter cancel exactly, noise adds.

These are not per-sample hot paths: the estimates read `V x C` samples once,
and the applications are one broadcast multiply, an axis replacement and one
subtraction, so they stay Torch on both devices (the array estimate's
per-element responses use the native family on a CUDA profile).

### 4. Not done

A frequency-domain route (range-Doppler or omega-k for a straight stripmap
aperture) was not built. Its Stolt mapping is a per-sample resampling that
would need its own native kernel with VJP/JVP, and back-projection is exact for
the geometries this package produces (bistatic TDM-MIMO, curved tracks, turning
targets) at one launch per image.

## Evidence

`tests/processing/test_imaging.py`, `tests/processing/test_calibration.py`.
Oracle cubes are written from the waveforms' closed forms with element
positions placed by the test, not by the module.

* Two point targets, a 1x1 radar on a 12 m/s track, 64 chirps, 256 samples
  (CPU): both image at their positions (within 1/20 of a resolution cell), at
  0.974 and 0.980 of their amplitudes; -3 dB widths 1.005 x `0.886 c / 2B`
  (4.29 cm) in range and 0.980 x `0.886 lambda R / 2L` (3.90 cm) across.
  Imaging the same cube as if the radar stood still gives 0.22.
* TDM 2x4 on a moving platform: every element's response agrees at the target
  (each transmitter at its own slot instant).
* ISAR, a target turning 0.2 rad about a vertical axis: points image in the
  body frame; cross-range width within 6 % of `0.886 lambda / (2 dtheta)`. The
  range width is narrower than `c / 2B`: a turn widens the range extent of the
  k-space arc to `B + f_min (1 - cos(dtheta / 2))` (11 % here), and the oracle
  uses that band (within 4 %).
* OFDM with the Channel law; the beat law on the same samples is refused by
  the record check and, forced, defocuses.
* GPU: native image equals the CPU route (complex128 to 1e-12, complex64 to
  2e-5 relative); native VJP equals the CPU route's autograd VJP; native JVP
  equals a central difference; the adjoint identity ties the two; a second
  order request fails in `first_order_only`.
* End to end through `Radar.simulate` (Channel 0.6, GPU): a moving radar past
  two point targets - once as a common target velocity, once with `pose=` -
  focuses both within 1/10 of a cell at >= 0.95 of the compressed-peak ceiling,
  with both widths within 6 % of theory; a simulated turning target images in
  its body frame the same way.
* Calibration: random per-channel errors (gain 0.6-1.4, any phase) on a 2x4
  array are estimated to 5 mrad and 2 %; after calibration the scan route
  reads exactly the ideal array's angle at four directions (it does not
  before). A 0.25 ns delay bias is measured to 1 % of a native bin and moves
  the image back onto the reflector. Leakage plus static clutter cancel to
  1e-10. End to end: a simulated 2x4 array with diagonal `Coupling` port errors
  and `Leakage` is background-subtracted and calibrated back to the ideal
  radar's simulated angle.

## Consequences

* New public names in `witwin.radar.processing`: `Aperture`, `SarImage`,
  `IMAGE_FRAMES`, `backproject`, `element_responses`, `isar_aperture`,
  `ArrayCalibration`, `RangeCalibration`, `estimate_array_calibration`,
  `estimate_range_calibration`, `subtract_background`, `average_background`.
* R-ADR-017 surface record: two new processing owners (imaging, calibration);
  a second declared native processing unit (`processing_imaging.cu`, R-ADR-027
  section 6 had one); no new frozen vendor primitive (`gather`, `polar`,
  `vector_norm`, `floor`, `einsum` are gather/arithmetic/contraction, outside
  the frozen cells).
* `Frame` has no imaging facade yet; `Aperture.from_radar(radar,
  time_s=frame.time_s, ...)` and `backproject(range_profile(frame.processing_cube(),
  range_bins=...), ...)` are the route.
