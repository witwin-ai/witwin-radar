# R-ADR-027: Square-law CFAR with solved constants, the processing additions, and native selections

Status: Accepted (0.5.0). Amends R-ADR-017 sections 5 and 6.

## Context

The detectors published a `pfa` they did not achieve.

* `ca_cfar` scaled the ring MEAN with `alpha = N (pfa^(-1/N) - 1)`, which is the
  exact inverse of the false-alarm law of an average of `N` exponential POWER
  cells. `Frame.points` fed it `|sum_p x_p|`, a magnitude. On 4.2 million
  complex Gaussian noise cells at the point-cloud default windows
  (`guard=(2, 4)`, `training=(4, 8)`) the measured rate was 2.4e-7 at a nominal
  1e-2 and zero at 1e-3 and 1e-4.
* `os_cfar` scaled an ORDERED statistic with that same cell-averaging constant.
  It achieved 0.18, 0.08 and 0.05 of its nominal rate at 1e-2, 1e-3, 1e-4.
* Both replicate-padded both axes. The Doppler axis of a slow-time DFT is
  circular; replication correlates a border cell's ring with itself, and the
  range-edge cells ran 3-7 % hot on power input.
* Plan F-7 asked for zero padding, slow-time clutter removal, GO/SO-CFAR, peak
  grouping, clustering, a range-angle map, a typed micro-Doppler product and a
  Kalman tracker.

## Decision

### 1. Every detector is square law and solves its constant from its own exact law

A complex map is detected on `|x|^2` (`re^2 + im^2`); a real map is a power map.
`combine_coherent` returns `|sum_p x_p|^2`, so `Frame.points`'s
`ca_cfar(combine_coherent(rd.data))` is square law without an edit to `Frame`.

The noise model is exponential power per cell (complex Gaussian IQ). The laws,
published as `cfar_false_alarm_probability`:

| kind | statistic | false-alarm probability |
|---|---|---|
| `ca` | mean of `N` ring cells | `(1 + a/N)^-N` |
| `so` | `min(S1/n1, S2/n2)` | `1 - a sum_{i<n1, j<n2} n1^i n2^j (i+j)! / (i! j! (a+n1+n2)^(i+j+1))` |
| `go` | `max(S1/n1, S2/n2)` | `(1+a/n1)^-n1 + (1+a/n2)^-n2 - P_so` |
| `os` | `k`-th smallest of `N`, `k - 1 = min(int(rank_fraction N), N - 1)` | `prod_{i<k} (N - i)/(N - i + a)` (Rohling) |

`a` is solved per distinct training count by bisection on the host (float64,
cached per configuration and per device). The SO series follows from
`P(S_i/n_i > t) = exp(-n_i t) sum_{m<n_i} (n_i t)^m / m!`; the GO law from
`exp(-a min) + exp(-a max) = exp(-a A1) + exp(-a A2)`. A half truncated to no
cells reduces GO and SO to CA over the other half.

The ring's two halves are split along range (lagging: negative range offset, or
zero range offset and negative Doppler offset; leading: the mirror), each a
union of disjoint rectangles, so a half sum is a sum of non-negative block sums
with no `outer - guard` cancellation: a strong return in the guard band can no
longer drown the noise estimate in float32 rounding.

### 2. Edges follow the axes' topology

Doppler wraps (circular padding; the window must fit in the axis, else
refused). Range is truncated: an edge cell keeps the training cells on the map
and its constant is solved for that count, so the edge holds `pfa` too.

### 3. Measured rates (CPU, complex Gaussian noise, 128 maps of [128, 256] = 4,194,304 cells, windows (2, 4)/(4, 8), whole map)

| nominal | before: CA, magnitude (Frame.points) | before: CA, power | before: OS, power | after: CA | after: OS | after: GO | after: SO |
|---|---|---|---|---|---|---|---|
| 1e-2 | 2.38e-7 (0.00) | 1.009e-2 (1.009) | 1.790e-3 (0.179) | 1.002e-2 (1.002) | 9.956e-3 (0.996) | 1.000e-2 (1.000) | 1.001e-2 (1.001) |
| 1e-3 | 0 (0.00) | 1.028e-3 (1.028) | 8.178e-5 (0.082) | 1.008e-3 (1.008) | 1.009e-3 (1.009) | 1.011e-3 (1.011) | 9.999e-4 (1.000) |
| 1e-4 | 0 (0.00) | 1.070e-4 (1.070) | 5.007e-6 (0.050) | 1.042e-4 (1.042) | 1.020e-4 (1.020) | 1.059e-4 (1.059) | 1.018e-4 (1.018) |

`tests/processing/test_cfar.py::test_m1_4_*` pins all four detectors within
±20 % at all three rates on 2.1 million cells, and every threshold against an
independent oracle (`tests/support/reference_cfar.py`: shifted sums with
`torch.roll`, constants from `scipy.optimize.brentq` on laws written from their
definitions, the GO/SO law by `scipy.integrate.quad` over gamma densities).

### 4. Additions, each with its owner

| addition | owner | notes |
|---|---|---|
| `go_cfar`, `so_cfar`, `cfar_false_alarm_probability`, `CFAR_KINDS` | `detection.py` | |
| `group_peaks` | `detection.py` | one `max_pool2d` with indices; a cell survives when the pool centred on it points back at it; ties by position. `point_cloud(peak_neighbourhood=(3, 3))` applies it by default |
| `cluster_points`, `PointClusters` | `detection.py` | DBSCAN; component root = smallest core index, border = smallest root among core neighbours |
| `range_profile(range_bins=)`, `range_doppler_map(doppler_bins=)`, `ProcessingAxes.with_bin_counts` | `range_doppler.py`, `signal.py` | zero padding with the `1 / N` normalisation preserved |
| `remove_static_clutter` (`mean`, `mti2`, `mti3`), `CLUTTER_FILTERS` | `range_doppler.py` | cancellers shorten the slow axis and re-derive the velocity axis |
| `range_angle_map`, `RangeAngleMap`, `azimuth_directions` | `angle.py` | conventional weights through `beam_cube`, power summed over Doppler |
| `MicroDopplerSpectrogram` | `range_doppler.py` | `microdoppler_spectrogram` returns it; `1 / window_slots` normalisation; optional velocity axis |
| `KalmanTracker`, `TrackState`, `greedy_assignment` | `tracking.py` | constant velocity, Mahalanobis gate, Joseph update; the reference associator now uses the same greedy rounds |

### 5. Amendment to the frozen vendor list (R-ADR-017 section 5)

* pooling: `torch.nn.functional.max_pool2d` (peak grouping);
* `F.pad(mode="circular")` is the same frozen `pad` entry.

No inverse is formed anywhere: the Kalman gain and the Mahalanobis distance use
`torch.linalg.solve`.

### 6. Amendment to the native-DSP gate (R-ADR-017 section 6)

R-ADR-017 measured dispatch-bound stages and shipped no native DSP. Two
selections are not dispatch bound and not a few library calls:

* the ordered statistic: the Torch route materialises `[cells, N]` patches and
  sorts them (`N = 212` at the defaults: 27 MB of float32 patches plus a
  54 MB index tensor per `[128, 256]` map). `cfar_order_statistic_forward` is
  one launch, one thread per cell selecting in a thread-local buffer (up to
  1024 training cells, refused by name beyond);
* DBSCAN: the Torch route is min-label propagation with one host read per
  round. `dbscan_labels_forward` is four launches (counts and core flags,
  lock-free union-find, compression, border) and no host read.

Both are used for CUDA tensors only. The CPU route is the same algorithm in
Torch primitives and is the oracle the GPU tests compare against bit for bit
(the statistic is a selection, the labels are integers). Neither is
differentiable and neither has a companion: the facades refuse a derivative
before any device work (R-ADR-018). `RADAR_ABI_VERSION` is 11.

## Consequences

* Thresholds and detection counts change for every caller (golden change). The
  previous OS-CFAR test pinned Rohling's law at the old, mis-solved constant;
  it now pins `pfa` itself.
* A caller who fed a magnitude or a decibel map must feed power or complex IQ.
* `point_cloud` groups peaks by default; `peak_neighbourhood=None` restores
  per-cell points.
