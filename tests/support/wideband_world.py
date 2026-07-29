"""Wideband material fixtures with closed-form references.

Four scenes, each isolating exactly one thing:

* :func:`compile_half_space` - a slab thick and lossy enough that the back face
  is invisible, so the reflection degenerates to the single-interface Fresnel
  coefficient. Smooth, monotone frequency dependence with NO fringes. This is
  the fixture that isolates the ``eps_c(f) = eps_r - j sigma/(omega eps0)``
  term from everything else.
* :func:`compile_slab` - a thin slab whose Airy fringe period
  ``c / (2 Re(sqrt(eps_r)) d cos(theta_t))`` falls inside a band a radar can
  actually transmit. This is the fixture that falsifies a narrowband
  implementation: the reflectivity is NOT flat across the band, and it is not
  monotone either.
* :func:`compile_dispersive` - a ``PowerLawDispersion`` material. It exists to
  be REFUSED. Channel evaluates a ``DispersionSpec`` once, at compile, so a
  band on such a scene would silently reuse ``eps_r(f_ref)`` at every column;
  :func:`dispersive_eps_r_drift` computes how wrong that would be, which is how
  a refused capability is explained with a number instead of a shrug.
* :func:`compile_rough` - a surface-roughness material, refused for the same
  class of reason: the Kirchhoff table is built per material cache token and
  that token hashes the frequency.

The geometry is deliberately the SIMPLEST that produces a reflection: one flat
plate at ``x = PLATE_PLANE_X_M``, one source, one sink, mirrored about it, so
the incidence angle is closed form and the specular point is analytic. The
multi-endpoint fixture exists for row-identity questions; this one exists for
material questions, and mixing the two would make a fringe test depend on a
pair partition.

``witwin.core.Mesh`` defaults ``recenter=True`` and silently rewrites authored
world coordinates, so every mesh here passes ``recenter=False``.
"""

from __future__ import annotations

import cmath
import math

import torch

C0_M_PER_S = 299792458.0
EPS0_F_PER_M = 8.8541878128e-12

#: 3 GHz rather than 77 GHz, on purpose. The Airy fringe period is
#: ``c / (2 n d)`` and does not depend on the carrier at all, so a 0.05 m slab
#: fringes every 655 MHz whatever the reference frequency is - but a 655 MHz
#: EXCURSION is 0.85% of 77 GHz and 22% of 3 GHz. At 77 GHz the float32 launch
#: grid resolves 8192 Hz; at 3 GHz it resolves 256 Hz. The lower carrier is what
#: lets a two-fringe sweep stay well inside the frequency-resolution budget
#: while still being a band a real system transmits.
REFERENCE_FREQUENCY_HZ = 3.0e9

PLATE_PLANE_X_M = 4.0
PLATE_HALF_M = 12.0
PLATE_VERTICES_M = (
    (PLATE_PLANE_X_M, -PLATE_HALF_M, -PLATE_HALF_M),
    (PLATE_PLANE_X_M, PLATE_HALF_M, -PLATE_HALF_M),
    (PLATE_PLANE_X_M, PLATE_HALF_M, PLATE_HALF_M),
    (PLATE_PLANE_X_M, -PLATE_HALF_M, PLATE_HALF_M),
)
PLATE_FACES = ((0, 1, 2), (0, 2, 3))

#: Source and sink, both in front of the plate and both off axis, so the
#: reflection is at a real oblique incidence rather than at normal incidence
#: where the TE and TM coefficients coincide and a polarization bug hides.
SOURCE_POSITION_M = (0.0, 0.0, 0.0)
SINK_POSITION_M = (0.6, 3.0, 0.0)
POLARIZATION = (0.0, 0.0, 1.0)
SOURCE_POWER_W = 0.25

SOURCE_STABLE_ID = 700
SINK_STABLE_ID = 701

#: The thin slab. ``eps_r = 4`` and ``d = 0.05 m`` give an analytic
#: normal-incidence fringe period of ``c / (2 * 2 * 0.05) = 1.5 GHz``; at the
#: fixture's oblique incidence the transmitted angle shortens the electrical
#: path and the period widens, which the test computes rather than assumes.
SLAB_EPS_R = 4.0
SLAB_SIGMA_E = 0.005
SLAB_THICKNESS_M = 0.05

#: The half space. Same permittivity, far more loss, and thick enough that
#: ``|exp(-j k_z d)|^2`` is below 1e-8 - the test asserts that degeneracy
#: rather than trusting it.
HALF_SPACE_EPS_R = 6.0
HALF_SPACE_SIGMA_E = 3.0
HALF_SPACE_THICKNESS_M = 2.0

#: The dispersive material. Same nominal permittivity as the slab so that the
#: only difference between the refused scene and the accepted one is the
#: ``DispersionSpec`` itself.
DISPERSION_EPS_R_EXPONENT = -0.3


def _mesh():
    from witwin.core import Mesh

    return Mesh(
        vertices=torch.tensor(PLATE_VERTICES_M, dtype=torch.float32),
        faces=torch.tensor(PLATE_FACES, dtype=torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )


def _scene(material):
    from witwin.core import AntennaState, Scene, Structure
    from witwin.core.identity import reserve_antenna_id

    mesh = _mesh()
    plate = Structure(geometry=mesh, material=material, structure_id=1, material_id=1, assignment_id=1, surface_id=1)
    scene = Scene(
        structures=(plate,),
        endpoints=[AntennaState(reserve_antenna_id(77801), "tx", torch.tensor(SOURCE_POSITION_M, dtype=torch.float32))],
    )
    authored = torch.tensor(PLATE_VERTICES_M, dtype=torch.float64)
    survived = mesh.vertices.detach().to(dtype=torch.float64).cpu()
    if not torch.allclose(survived, authored, atol=1.0e-9):
        raise AssertionError("the plate was recentred; every Mesh here must pass recenter=False")
    return scene


def _compile(material, *, reference_frequency_hz=REFERENCE_FREQUENCY_HZ):
    from witwin.channel.scene import compile as compile_scene

    return compile_scene(_scene(material), reference_frequency_hz=reference_frequency_hz)


def slab_material():
    from witwin.core import PhysicalMaterial

    return PhysicalMaterial(name="slab", eps_r=SLAB_EPS_R, sigma_e=SLAB_SIGMA_E, thickness_m=SLAB_THICKNESS_M)


def half_space_material():
    from witwin.core import PhysicalMaterial

    return PhysicalMaterial(
        name="half_space", eps_r=HALF_SPACE_EPS_R, sigma_e=HALF_SPACE_SIGMA_E, thickness_m=HALF_SPACE_THICKNESS_M
    )


def dispersive_material():
    from witwin.core import PhysicalMaterial
    from witwin.core.material import PowerLawDispersion

    return PhysicalMaterial(
        name="dispersive",
        eps_r=SLAB_EPS_R,
        sigma_e=SLAB_SIGMA_E,
        thickness_m=SLAB_THICKNESS_M,
        dispersion=PowerLawDispersion(
            reference_frequency_hz=REFERENCE_FREQUENCY_HZ, eps_r_exponent=DISPERSION_EPS_R_EXPONENT
        ),
    )


def rough_material():
    from witwin.core import PhysicalMaterial
    from witwin.core.material import SurfaceRoughness

    return PhysicalMaterial(
        name="rough",
        eps_r=SLAB_EPS_R,
        sigma_e=SLAB_SIGMA_E,
        thickness_m=SLAB_THICKNESS_M,
        roughness_front=SurfaceRoughness(
            rms_height_m=1.0e-3, correlation_length_x_m=5.0e-3, correlation_length_y_m=5.0e-3
        ),
    )


def compile_slab():
    return _compile(slab_material())


def compile_half_space():
    return _compile(half_space_material())


def compile_dispersive():
    return _compile(dispersive_material())


def compile_rough():
    return _compile(rough_material())


def endpoint_spec(position, stable_id, *, power_w=None, device="cuda"):
    from witwin.radar.propagation import RadarEndpointSpec

    return RadarEndpointSpec(
        stable_ids=torch.tensor([stable_id], dtype=torch.int64, device=device),
        positions_m=torch.tensor([position], dtype=torch.float32, device=device),
        polarizations=torch.tensor([POLARIZATION], dtype=torch.float32, device=device),
        powers_w=(None if power_w is None else torch.tensor([power_w], dtype=torch.float32, device=device)),
    )


def source_spec(device: str = "cuda"):
    return endpoint_spec(SOURCE_POSITION_M, SOURCE_STABLE_ID, power_w=SOURCE_POWER_W, device=device)


def sink_spec(device: str = "cuda"):
    return endpoint_spec(SINK_POSITION_M, SINK_STABLE_ID, device=device)


# ---------------------------------------------------------------------------
# The closed forms
# ---------------------------------------------------------------------------


def incidence_cosine() -> float:
    """``cos(theta_i)`` at the specular point, from the image source.

    The plate is the plane ``x = PLATE_PLANE_X_M`` with normal ``+x``, so the
    image of the source is its mirror and the reflected ray is the straight
    line from the image to the sink. The cosine is that line's ``x`` component
    over its length - exact, with no search and no tolerance.
    """

    image = (2.0 * PLATE_PLANE_X_M - SOURCE_POSITION_M[0], SOURCE_POSITION_M[1], SOURCE_POSITION_M[2])
    delta = tuple(image[axis] - SINK_POSITION_M[axis] for axis in range(3))
    length = math.sqrt(sum(value * value for value in delta))
    return abs(delta[0]) / length


def reflection_length_m() -> float:
    """The image-source path length: source -> plate -> sink."""

    image = (2.0 * PLATE_PLANE_X_M - SOURCE_POSITION_M[0], SOURCE_POSITION_M[1], SOURCE_POSITION_M[2])
    return math.sqrt(sum((image[axis] - SINK_POSITION_M[axis]) ** 2 for axis in range(3)))


def line_of_sight_length_m() -> float:
    return math.sqrt(sum((SOURCE_POSITION_M[axis] - SINK_POSITION_M[axis]) ** 2 for axis in range(3)))


def fringe_period_hz(*, eps_r: float = SLAB_EPS_R, thickness_m: float = SLAB_THICKNESS_M) -> float:
    """``c / (2 * Re(sqrt(eps_r)) * d * cos(theta_t))``, the Airy period.

    ``theta_t`` comes from Snell's law at the fixture's incidence, so this is
    the period of the slab AS THE FIXTURE SEES IT, not the normal-incidence
    one. Getting that distinction wrong is a 10% error in the period and shows
    up as a fringe test that drifts out of phase across a wide sweep.
    """

    n = math.sqrt(eps_r)
    sin_i = math.sqrt(max(0.0, 1.0 - incidence_cosine() ** 2))
    sin_t = sin_i / n
    cos_t = math.sqrt(max(0.0, 1.0 - sin_t * sin_t))
    return C0_M_PER_S / (2.0 * n * thickness_m * cos_t)


def dispersive_eps_r_drift(offsets_hz) -> float:
    """How far ``eps_r(f)`` moves across a band, as a fraction of ``eps_r``.

    This is the number that explains the dispersive REFUSAL. Channel evaluates
    a ``DispersionSpec`` once at compile, so a band on a dispersive scene would
    reuse ``eps_r(f_ref)`` at every column. The drift below is exactly the error
    that would introduce, and comparing it to the accepted narrowband error law
    is how "this capability is refused" becomes a statement with a number in it.
    """

    values = [
        SLAB_EPS_R * ((REFERENCE_FREQUENCY_HZ + offset) / REFERENCE_FREQUENCY_HZ) ** DISPERSION_EPS_R_EXPONENT
        for offset in offsets_hz
    ]
    return max(abs(value - SLAB_EPS_R) / SLAB_EPS_R for value in values)


def free_space_coefficient(frequency_hz: float, distance_m: float) -> complex:
    """``sqrt(P) * lambda/(4 pi d) * exp(-j k d)`` - Channel's excited LoS field.

    Written out rather than imported because it is the reference the wideband
    spreading tilt is measured against, and a reference that shares an
    implementation with what it checks proves nothing.
    """

    wavelength = C0_M_PER_S / frequency_hz
    amplitude = math.sqrt(SOURCE_POWER_W) * wavelength / (4.0 * math.pi * distance_m)
    phase = -2.0 * math.pi * frequency_hz * distance_m / C0_M_PER_S
    return amplitude * cmath.exp(1j * phase)


def slab_reflection_te(frequency_hz: float, *, material: str = "slab") -> complex:
    """The analytic TE reflection coefficient of the fixture's layer stack.

    Delegates to ``tests.reference`` is not possible - the radar package has no
    such oracle - so the transfer-matrix recursion is written here, in
    complex128, from the same definitions the Channel-side oracle uses:

        eps_c = eps_r - j * sigma / (omega * eps0)
        k_z   = (omega/c) * sqrt(eps_c - sin^2(theta_i))
        r_01  = (k_z0 - k_z1) / (k_z0 + k_z1)          [TE]
        r     = (r_01 + r_12 * p^2) / (1 + r_01 * r_12 * p^2),  p = exp(-j k_z1 d)

    with vacuum on both sides, which is the single-slab Airy expression. For the
    half-space fixture ``p`` is small enough that the second term vanishes and
    this collapses to ``r_01``; the test asserts that collapse numerically
    before using it.
    """

    if material == "slab":
        eps_r, sigma_e, thickness_m = SLAB_EPS_R, SLAB_SIGMA_E, SLAB_THICKNESS_M
    elif material == "half_space":
        eps_r, sigma_e, thickness_m = (HALF_SPACE_EPS_R, HALF_SPACE_SIGMA_E, HALF_SPACE_THICKNESS_M)
    else:
        raise ValueError(f"unknown material {material!r}")

    omega = 2.0 * math.pi * frequency_hz
    eps_c = complex(eps_r, -sigma_e / (omega * EPS0_F_PER_M))
    k0 = omega / C0_M_PER_S
    cos_i = incidence_cosine()
    sin2_i = 1.0 - cos_i * cos_i

    k_z0 = k0 * cos_i
    k_z1 = k0 * cmath.sqrt(eps_c - sin2_i)
    if k_z1.imag > 0.0:
        k_z1 = -k_z1

    r01 = (k_z0 - k_z1) / (k_z0 + k_z1)
    r12 = (k_z1 - k_z0) / (k_z1 + k_z0)
    p2 = cmath.exp(-2.0j * k_z1 * thickness_m)
    return (r01 + r12 * p2) / (1.0 + r01 * r12 * p2)


def bare_interface_te(frequency_hz: float, *, material: str = "half_space") -> complex:
    """``r_01`` alone: the single-interface Fresnel coefficient, no stack."""

    if material == "half_space":
        eps_r, sigma_e = HALF_SPACE_EPS_R, HALF_SPACE_SIGMA_E
    else:
        eps_r, sigma_e = SLAB_EPS_R, SLAB_SIGMA_E
    omega = 2.0 * math.pi * frequency_hz
    eps_c = complex(eps_r, -sigma_e / (omega * EPS0_F_PER_M))
    k0 = omega / C0_M_PER_S
    cos_i = incidence_cosine()
    k_z0 = k0 * cos_i
    k_z1 = k0 * cmath.sqrt(eps_c - (1.0 - cos_i * cos_i))
    if k_z1.imag > 0.0:
        k_z1 = -k_z1
    return (k_z0 - k_z1) / (k_z0 + k_z1)


__all__ = [
    "C0_M_PER_S",
    "DISPERSION_EPS_R_EXPONENT",
    "HALF_SPACE_EPS_R",
    "HALF_SPACE_SIGMA_E",
    "HALF_SPACE_THICKNESS_M",
    "PLATE_PLANE_X_M",
    "POLARIZATION",
    "REFERENCE_FREQUENCY_HZ",
    "SINK_POSITION_M",
    "SINK_STABLE_ID",
    "SLAB_EPS_R",
    "SLAB_SIGMA_E",
    "SLAB_THICKNESS_M",
    "SOURCE_POSITION_M",
    "SOURCE_POWER_W",
    "SOURCE_STABLE_ID",
    "bare_interface_te",
    "compile_dispersive",
    "compile_half_space",
    "compile_rough",
    "compile_slab",
    "dispersive_eps_r_drift",
    "dispersive_material",
    "endpoint_spec",
    "free_space_coefficient",
    "fringe_period_hz",
    "half_space_material",
    "incidence_cosine",
    "line_of_sight_length_m",
    "reflection_length_m",
    "rough_material",
    "sink_spec",
    "slab_material",
    "slab_reflection_te",
    "source_spec",
]
