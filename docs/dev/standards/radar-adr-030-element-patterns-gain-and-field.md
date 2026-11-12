# R-ADR-030: Element patterns carry an absolute gain, transmit and receive are separate, and a pattern may be a complex field

Status: Accepted (0.5.0)

## Context

The sensor stage weighted every composed row by `sqrt(G_t G_r)` from ONE
power table shared by every transmit and receive element. Three gaps followed:
there was no absolute antenna gain (the table was documented as normalised but
only non-negativity was checked, so a peak gain could only be smuggled in as a
table above one); a radar with different transmit and receive elements could
not be described; and a power table has no phase, so an element's phase
response - which steers a phased array and moves an angle estimate - could not
be represented.

## Decision

**Record.** `witwin.radar.Pattern` gains two fields and two constructors:

* `quantity`: `"power"` (a real normalised power gain in `[0, 1]`, as before)
  or `"field"` (a complex normalised field amplitude, `|F| <= 1`, whose phase
  is the element's phase response). `Pattern.field_separable` and
  `Pattern.field_table` build field patterns.
* `gain`: the element's absolute peak gain in dBi. The table field of the map
  kind is renamed `gain_table` so that `gain` names the dBi value. The
  normalisation bound is now ENFORCED (with 1e-6 float slack), so a peak can
  only enter through `gain`.

`Radar.pattern` is the transmit element and, unless `Radar.rx_pattern` is
given, the receive element. `Pattern.from_antenna` samples a
`witwin.core.AntennaPattern` custom callable onto a field table, normalises by
the largest magnitude and puts that magnitude in `gain`. That is the authoring
route for Channel's `presets.tr38901_pattern` and `presets.dipole_pattern`,
which return Core custom patterns: Radar imports only `witwin.core`, never
`witwin.channel`, for it. Core's built-in `vertical`/`horizontal` kinds are
polarisation shapes that Channel's transport applies itself and are refused by
name.

**Kernel.** Each element pattern crosses the ABI as one packed float32 buffer
(`[kind, quantity, num_x, num_y, peak_amplitude, 0, 0, 0, x_axis, y_axis,
values]`, real block then imaginary block), the transmit one and the receive
one. The sensor-weight operators take the two buffers instead of five shared
tables and an integer kind. The absolute gain is applied in this operator and
nowhere else, as `10^(G_t/20) 10^(G_r/20)` on the row weight. When both
elements are power patterns the operator runs the original real expression
verbatim, so every existing radar is bit-identical; when either is a field
pattern the scale is the complex product `A_t A_r` (a power element enters as
`sqrt(G)`), and the forward, the JVP and the VJP carry the complex scale and its
complex direction derivative.

**Identity skip.** A transmit and a receive element that are both the isotropic
0 dBi pattern make the stage the identity, so `RoundTripPatternStage.apply`
skips the launch and passes the weight tensor through (P-8). It sets
`weight_includes_antenna_pattern` exactly as the launch would.

## Polarimetric extension

The polarisation ownership change (Channel returning the unprojected
`complex3_transport`, Radar applying element pattern and polarisation, owner
decision 2) is a later task. The record and the ABI are shaped so that it adds
a variant without breaking this one:

* a new `quantity` value, `"jones"`, whose table holds two complex components
  (theta and phi) per knot. In the packed buffer they follow the scalar
  field's real and imaginary blocks, and the quantity code (2) is new. The
  header does not move and existing buffers keep their meaning.
* the kernel's `evaluate_element` returns a two-component value for that code,
  and the row scale becomes `F_rx^T T F_tx` once Channel publishes the
  transport `T`; a scalar pattern stays the special case it is now.
* `Pattern.from_antenna` already samples the callable's full output; a Core
  callable that returns two components would populate a Jones table.

## Evidence

`tests/test_element_patterns.py` (GPU, end to end through `Radar.simulate`):
16.000 dB for a 10 dBi transmit and 6 dBi receive element on boresight; a
receive-only dipole scales an off-boresight echo by `sqrt(G_dipole(30 deg))`;
a unit-magnitude linear-phase field pattern rotates the echo by its phase at the
target angle once per side; the field-pattern JVP equals central differences and
satisfies the VJP/JVP adjoint identity; power patterns are bit-identical through
the packed tables; `from_antenna` recovers a TR 38.901 element's 8 dBi peak and
its -3 dB beam edge; unnormalised tables are refused; the isotropic default
launches no `sensor_weight_forward`.

## Consequences

`Pattern.gain` (the map table) is renamed `gain_table`; a power table above one
is refused with a message that names `gain`. `SensorWeightPlan.tables` is the
pair of packed buffers.
