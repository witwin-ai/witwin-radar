# R-ADR-028: Geometry-driven angle estimation, the processing frame, and recorded cubes

Status: Accepted (0.5.0). Amends R-ADR-017 section 2.

## Context

* `point_cloud` and `Frame.points` estimated angles with two routes written for
  the TI three-transmitter half-wavelength layout. A 1x8 or 2x4 array was
  refused with "needs at least three transmitter rows"; a one-wavelength line
  returned `+0.344` for a true `sin(az) = +0.174`, because the azimuth was read
  as `wx / pi`, which is only `u_x` at half-wavelength pitch. An overlapping
  3x4 MIMO layout was steered as if its first eight virtual elements were a
  line, which they are not, and returned a wrong angle without complaint.
* Two frames were in use and neither was named. `ArrayGeometry.from_axes`
  copied the radar's LOCAL offsets (`right, up, -boresight`, the frame of
  `Radar.tx`) into a record whose steering directions and point clouds are
  `right, boresight, up`. An elevation offset declared UP was therefore
  steered as a BORESIGHT offset by `conventional_steering`, `beam_cube` and
  MUSIC; only the phase-comparison route was immune, because it never formed a
  steering vector. There was no transform from a point cloud to the world.
* `ProcessingAxes` could only be built from a `SynthesisResult`, a waveform
  spec and a `SensorArraySpec`, three internal records; a recorded cube could
  not be processed.

## Decision

### 1. One processing frame

`x` right, `y` boresight, `z` up, origin at the radar. Element positions,
steering directions, beam grids, direction cosines and point clouds are in it.
`ArrayGeometry.from_axes` maps a radar-local offset `(x, y, z)` to `(x, -z, y)`
once; `ArrayGeometry.from_offsets` takes processing-frame offsets.
`ProcessingAxes.tx_loc_half_wavelength` stays in the radar-local frame, since it
is the array spec's.

`PointCloud` carries `frame` (`"radar"` or `"world"`). `PointCloud.to_world(radar)`
maps `(x, y, z) -> (x, z, -y)` into the radar-local frame and then through
`Radar._world_from_local_matrix`, the pose owner that places the elements, so
there is one statement of the pose. Radial velocity, energy and range are
scalars and are copied.

### 2. Three named angle routes, each checking its layout

`AOA_ROUTES = {"phase_comparison", "fft2", "scan"}`, explicit selection kept.

* `ArrayLattice` is derived on the host from the declared offsets at
  construction: per-axis pitch (float GCD to 1e-6 spacing units), cell indices
  and extent, or `None` when the elements are off a planar `x`-`z` grid (a
  boresight offset, or incommensurate offsets).
* `phase_comparison`: the first `2 num_rx` virtual elements must be a uniform
  line along `x`, the next `num_rx` a line of the same pitch displaced along `z`
  and sitting directly above whole elements of the first. Azimuth is the FFT
  bin divided by the pitch in wavelengths; elevation is the phase of
  `sum_i b_i conj(a_{m+i})` over the vertically aligned pairs, which carries no
  azimuth term and needs no walk correction. Anything else is refused by name.
* `fft2` (the ULA route is its one-row case): the snapshot is placed on its
  lattice (shared cells averaged, holes zero), zero padded along each axis the
  array extends in, and the peak read as `u = s k / (fft_size pitch d / lambda)`.
  Unambiguous for `|u| < 1 / (2 pitch d / lambda)`. A non-lattice array is
  refused by name.
* `scan`: `|w(u)^H a|^2` over `fft_size` direction cosines per extended axis,
  `w` from `conventional_steering` of the declared positions; any layout. The
  grid's half width is the lattice's unambiguous half width when smaller than
  one, so a grating-lobe pair is never two equal peaks. On a half-wavelength
  lattice the grid is the FFT grid and the two routes agree bin for bin.
* A line array measures one cosine; the other is published as zero. A tilted
  line is refused.

The tests build snapshots from `Radar.tx_pos`/`rx_pos` and a far world target
(exact round-trip phases, beat phasor), run every route on 1x8, 2x4,
one-wavelength and TI 3x4 layouts under a non-default pose, and compare
`to_world` directions to the analytic ones on exact bins.

`point_cloud` defaults to `route="scan"`. `Frame.points` keeps passing `route="phase_comparison"` by default (Frame is
outside this package); on a layout that route cannot read it now refuses by
name and points to `route="scan"`.

### 3. Recorded and synthetic cubes

* `ProcessingAxes.from_radar(radar, *, range_oversample=1, device=None)` builds
  the record from `radar.waveform_spec()` and the radar's array, with the cube
  conventions from the `SynthesisResult` factory evaluated on a shape-only
  `meta` tensor, so it equals the echo's `Result.axes` field for field.
* `ProcessingCube.from_tensor(cube, radar)` accepts `[slow, pair, fast]` with
  SINK-MAJOR pairs (`pair = rx * num_tx + tx`, the synthesis layout, packed by
  `assemble_frame_cube`) or `[tx, rx, slow, fast]`. An FMCW cube must be in the
  beat phasor (positive beat frequency for positive range); ADC samples need
  `Fmcw(output="beat")`.

## Consequences

* Beam cubes, MUSIC and range-angle maps of arrays with elevation offsets are
  steered correctly; self-consistent synthetic tests that declared elevation
  along radar-local `z` (`[0, 0, 1]`, which is a boresight offset) were
  corrected to `[0, 1, 0]`.
* `fft2_aoa` no longer requires four transmitters and no longer assumes TI
  interleaving; it reads the lattice.
