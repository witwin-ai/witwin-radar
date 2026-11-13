"""The native facet operators against independent float64 references.

``facet_response`` is checked against a brute-force vector physical-optics
integral over each triangle (midpoint quadrature on a fine barycentric grid,
written here in NumPy, not the kernel's closed-form divided difference),
projected on the join's site bases, with the independent Airy slab
coefficients for the material law, and in both AD modes
against central finite differences; the reverse pass is also held to the exact
adjoint identity ``<u, J v> = <J^T u, v>``. ``facet_geometry`` is linear and is
checked the same way. Everything is driven through the autograd bridges the
production response uses, with synthetic leg tables, so no Channel run is
involved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from reference import radar_oracles as ref

pytestmark = pytest.mark.gpu

DEVICE = "cuda"


def _csr(keys: np.ndarray, owners: int) -> tuple[torch.Tensor, torch.Tensor]:
    order = np.argsort(keys, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(np.bincount(keys, minlength=owners))))
    return (
        torch.as_tensor(offsets, dtype=torch.int64, device=DEVICE),
        torch.as_tensor(order, dtype=torch.int64, device=DEVICE),
    )


def _unit(rng, rows: int, mean) -> np.ndarray:
    vectors = np.asarray(mean, dtype=np.float64)[None, :] + 0.35 * rng.standard_normal((rows, 3))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class Fixture:
    """Two sites: site 0 is one facet, site 1 a cluster of two. Four composed rows."""

    def __init__(self, *, material="pec", rms=0.0, wavenumber=60.0, seed=7):
        from witwin.radar.facets import MODEL_LAYERED, MODEL_PEC

        rng = np.random.default_rng(seed)
        # Facets facing -x, near the origin, a few centimetres across.
        base = np.array([[0.0, -0.02, -0.015], [0.0, 0.025, -0.02], [0.004, 0.0, 0.03]])
        corners = []
        for shift in (np.zeros(3), np.array([0.01, 0.05, 0.0]), np.array([-0.01, 0.03, 0.06])):
            tri = base + shift + 0.004 * rng.standard_normal((3, 3))
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if normal[0] > 0:
                tri = tri[[0, 2, 1]]
            corners.append(tri.reshape(-1))
        self.corners = np.asarray(corners)
        self.site_facets = np.array([0, 1, 2])
        self.site_offsets = np.array([0, 1, 3])
        self.facet_site = np.array([0, 1, 1])
        centroid = self.corners.reshape(3, 3, 3).mean(axis=1)
        self.sites = np.stack((centroid[0], centroid[1:].mean(axis=0)))
        # Both legs' final segments propagate INTO the sites, toward +x (R-ADR-039).
        self.k_in = _unit(rng, 3, (1.0, 0.1, -0.1))
        self.k_out = _unit(rng, 2, (1.0, -0.15, -0.1))
        self.idx_in = np.array([0, 1, 2, 0])
        self.idx_out = np.array([0, 1, 0, 1])
        self.idx_site = np.array([0, 0, 1, 1])
        self.k = float(wavenumber)
        if material == "pec":
            params, layers, offsets = [[MODEL_PEC, rms]], [[0.0, 1.0, 0.0, 1.0]], [0, 0]
        else:
            eps = complex(*material)
            params, layers, offsets = [[MODEL_LAYERED, rms]], [[0.02, eps.real, eps.imag, 1.0]], [0, 1]
        self.material_params = np.asarray(params, dtype=np.float64)
        self.layers = np.asarray(layers, dtype=np.float64)
        self.layer_offsets = np.asarray(offsets)
        self.seed = 11

    def law(self):
        from witwin.radar.scattering import _FacetLaw

        def long(values):
            return torch.as_tensor(values, dtype=torch.int64, device=DEVICE)

        def single(values):
            return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32, device=DEVICE)

        return _FacetLaw(
            site_offsets=long(self.site_offsets),
            site_facets=long(self.site_facets),
            facet_site=long(self.facet_site),
            facet_material=long(np.zeros(len(self.corners), dtype=np.int64)),
            material_params=single(self.material_params),
            layer_offsets=long(self.layer_offsets),
            layers=single(self.layers),
            wavenumber=self.k,
            seed=self.seed,
            # The inbound legs' last points are far up-range, never at a site.
            origin_in=single(-100.0 * self.k_in),
        )

    def tables(self):
        from witwin.radar.scattering import _BackwardTables

        by_in = _csr(self.idx_in, 3)
        by_out = _csr(self.idx_out, 2)
        by_site = _csr(self.idx_site, 2)
        return _BackwardTables(*by_in, *by_out, *by_site)

    def inputs(self):
        def single(values):
            return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32, device=DEVICE)

        sites = torch.as_tensor(self.sites, dtype=torch.float64, device=DEVICE)
        return [single(self.k_in), single(self.k_out), single(self.corners), sites]

    def evaluate(self, k_in, k_out, corners, sites, valid=None):
        """``complex128[4, 4]`` rows of [HH, HV, VH, VV] (as float pairs from the kernel), and the up axes."""

        from witwin.radar.scattering import _FacetResponse

        def long(values):
            return torch.as_tensor(values, dtype=torch.int64, device=DEVICE)

        flags = torch.ones(4, dtype=torch.int32, device=DEVICE) if valid is None else valid
        return _FacetResponse.apply(
            k_in,
            k_out,
            corners,
            sites,
            long(self.idx_in),
            long(self.idx_out),
            long(self.idx_site),
            flags,
            self.law(),
            self.tables(),
        )


def _site_basis(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The join's basis for up axis z: h = (z x k)/|z x k|, v = h x k, with its fallback."""

    up = np.array([0.0, 0.0, 1.0])
    w = np.cross(up, k)
    if not np.linalg.norm(w) > 1e-6:
        up = np.array([1.0, 0.0, 0.0]) if abs(k[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        w = np.cross(up, k)
    h = w / np.linalg.norm(w)
    return h, np.cross(h, k)


def _reflection_reference(fixture: Fixture, cosine: float) -> tuple[complex, complex]:
    """(r_TE, r_TM) at ``cosine`` in the tangential convention, from the independent Airy formula."""

    if fixture.material_params[0, 0] == 1:
        return -1.0 + 0.0j, -1.0 + 0.0j
    _, eps_re, eps_im, _ = fixture.layers[0]
    frequency = fixture.k * ref.C0_M_PER_S / (2.0 * math.pi)
    sigma = -eps_im * 2.0 * math.pi * frequency * ref.EPS0_F_PER_M
    common = {
        "eps_r": eps_re,
        "sigma_e": sigma,
        "thickness_m": fixture.layers[0, 0],
        "frequency_hz": frequency,
        "cos_incidence": cosine,
    }
    # The reference TM coefficient is the magnetic-field one; its negative is the tangential convention.
    return (
        ref.slab_reflection_coefficient(polarization="te", **common),
        -ref.slab_reflection_coefficient(polarization="tm", **common),
    )


def _reference_rows(fixture: Fixture, grid: int = 160) -> np.ndarray:
    """Brute-force vector PO: the PEC current 2 n x H radiated, Fresnel-weighted, projected on the site bases.

    The facet integral is a midpoint sum of exp(j k q.(r - x_s)) over each lit
    triangle, independent of the kernel's closed-form divided difference; the
    dyadic is built from its definition in plain float64 linear algebra.
    """

    k = fixture.k
    cells = []
    for i in range(grid):
        for j in range(grid - i):
            cells.append(((i + 1.0 / 3.0) / grid, (j + 1.0 / 3.0) / grid))
            if j < grid - i - 1:
                cells.append(((i + 2.0 / 3.0) / grid, (j + 2.0 / 3.0) / grid))
    cells = np.asarray(cells)
    rows = []
    for row in range(4):
        k_i = fixture.k_in[fixture.idx_in[row]]
        k_r = fixture.k_out[fixture.idx_out[row]]
        site = fixture.idx_site[row]
        dyadic = np.zeros((3, 3), dtype=complex)
        for entry in range(fixture.site_offsets[site], fixture.site_offsets[site + 1]):
            v0, v1, v2 = fixture.corners[fixture.site_facets[entry]].reshape(3, 3)
            cross = np.cross(v1 - v0, v2 - v0)
            area = 0.5 * np.linalg.norm(cross)
            normal = cross / (2.0 * area)
            ci, co = -normal @ k_i, -normal @ k_r
            if ci <= 0 or co <= 0:
                continue
            q = -(k_i + k_r)
            points = v0[None, :] + cells[:, :1] * (v1 - v0)[None, :] + cells[:, 1:] * (v2 - v0)[None, :]
            integral = np.exp(1j * k * (points - fixture.sites[site]) @ q).mean() * area
            r_te, r_tm = _reflection_reference(fixture, ci)
            te = np.cross(normal, k_i)
            te /= np.linalg.norm(te)
            tm = np.cross(te, k_i)
            # R x = (n.k_i) x - k_i (n.x), the radiated PO current of a field x.
            radiate = (normal @ k_i) * np.eye(3) - np.outer(k_i, normal)
            law = -(r_te * radiate @ np.outer(te, te) + r_tm * radiate @ np.outer(tm, tm))
            dyadic += 1j * (k * k / math.pi) * integral * law
        h_i, v_i = _site_basis(k_i)
        h_r, v_r = _site_basis(k_r)
        rows.append([b_r @ dyadic @ b_i for b_r in (h_r, v_r) for b_i in (h_i, v_i)])
    return np.asarray(rows)


def _complex(matrix: torch.Tensor) -> np.ndarray:
    return torch.view_as_complex(matrix.double().contiguous()).cpu().numpy()


@pytest.mark.parametrize("material", ["pec", (5.24, -0.18), (2.5, -0.01)], ids=["pec", "concrete", "board"])
def test_the_response_is_the_vector_physical_optics_integral(material):
    fixture = Fixture(material=material)
    matrix, up = fixture.evaluate(*fixture.inputs())
    expected = _reference_rows(fixture)
    assert np.all(np.abs(expected[:, 0]) > 0) and np.all(np.abs(expected[:, 3]) > 0)
    scale = np.abs(expected).max(axis=1, keepdims=True)
    np.testing.assert_allclose(_complex(matrix) / scale, expected / scale, rtol=0, atol=2e-4)
    assert torch.equal(up.cpu(), torch.tensor([[0.0, 0.0, 1.0]] * 4))


def test_a_broadside_plate_triangle_reads_four_pi_area_over_lambda_squared():
    """No quadrature: the closed form at the one point where it is trivial. PEC: D = -j (4 pi A / lambda^2) I."""

    fixture = Fixture()
    fixture.corners = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.04, 0.0]])
    fixture.site_facets, fixture.site_offsets, fixture.facet_site = np.array([0]), np.array([0, 1, 1]), np.array([0])
    fixture.sites = np.stack((fixture.corners.reshape(3, 3).mean(axis=0), np.zeros(3)))
    fixture.k_in = np.array([[1.0, 0.0, 0.0]] * 3)
    fixture.k_out = np.array([[1.0, 0.0, 0.0]] * 2)
    fixture.idx_site = np.array([0, 0, 0, 0])
    matrix, _ = fixture.evaluate(*fixture.inputs())
    wavelength = 2.0 * math.pi / fixture.k
    expected = 4.0 * math.pi * (0.5 * 0.03 * 0.04) / wavelength**2
    values = _complex(matrix)[0]
    np.testing.assert_allclose(values, [-1j * expected, 0.0, 0.0, -1j * expected], rtol=0, atol=1e-6 * expected)


def test_dead_rows_back_faces_and_self_illumination_publish_exactly_zero():
    fixture = Fixture()
    valid = torch.tensor([1, 0, 1, 1], dtype=torch.int32, device=DEVICE)
    matrix, _ = fixture.evaluate(*fixture.inputs(), valid=valid)
    assert torch.count_nonzero(matrix[1]).item() == 0
    flipped = Fixture()
    flipped.corners = flipped.corners.reshape(3, 3, 3)[:, [0, 2, 1]].reshape(3, 9)
    matrix, _ = flipped.evaluate(*flipped.inputs())
    assert torch.count_nonzero(matrix).item() == 0
    # An inbound row whose last point IS the site: the facet's own surface reflecting onto itself.
    selfish = Fixture()
    law = selfish.law
    selfish.law = lambda: _with_origin(law(), selfish)
    matrix, _ = selfish.evaluate(*selfish.inputs())
    # Row 0 reaches site 0 through that inbound row; row 3 reaches site 1 through it, from 5 cm away.
    assert torch.count_nonzero(matrix[0]).item() == 0
    assert torch.count_nonzero(matrix[3]).item() > 0


def _with_origin(law, fixture: Fixture):
    from dataclasses import replace

    origin = law.origin_in.clone()
    origin[0] = torch.as_tensor(fixture.sites[0], dtype=torch.float32)
    return replace(law, origin_in=origin)


def _loss_directions(fixture: Fixture, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tangents = [
        torch.randn(value.shape, generator=generator, dtype=value.dtype).to(DEVICE) for value in fixture.inputs()
    ]
    weights = torch.randn((4, 4, 2), generator=generator).to(DEVICE)
    return tangents, weights


@pytest.mark.parametrize("rms", [0.0, 0.004], ids=["smooth", "rough"])
@pytest.mark.parametrize("material", ["pec", (5.24, -0.18)], ids=["pec", "concrete"])
def test_both_ad_modes_match_finite_differences_and_each_other(material, rms):
    import torch.autograd.forward_ad as fwad

    fixture = Fixture(material=material, rms=rms)
    tangents, weights = _loss_directions(fixture, seed=3)
    inputs = [value.requires_grad_(True) for value in fixture.inputs()]
    matrix, _ = fixture.evaluate(*inputs)
    grads = torch.autograd.grad((weights * matrix).sum(), inputs)
    reverse = sum(float((g.double() * t.double()).sum()) for g, t in zip(grads, tangents, strict=True))

    with fwad.dual_level():
        duals = [fwad.make_dual(value.detach(), tangent) for value, tangent in zip(inputs, tangents, strict=True)]
        tangent_matrix, _ = fixture.evaluate(*duals)
        forward = float((weights * fwad.unpack_dual(tangent_matrix).tangent).sum())

    def value(step: float) -> float:
        shifted = [(v.detach().double() + step * t.double()).to(v.dtype) for v, t in zip(inputs, tangents, strict=True)]
        result, _ = fixture.evaluate(*shifted)
        return float((weights.double() * result.double()).sum())

    step = 2e-4
    central = (value(step) - value(-step)) / (2.0 * step)
    assert forward == pytest.approx(reverse, rel=1e-4, abs=0)
    assert forward == pytest.approx(central, rel=2e-2, abs=0)


def test_facet_geometry_and_its_two_companions():
    import torch.autograd.forward_ad as fwad

    from witwin.radar.facets import FacetModel, _FacetGeometry

    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.5], [2.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=DEVICE,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3]])
    site_of = np.array([0, 1, 1])
    incidence = faces.reshape(-1)

    def long(values):
        return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.int64, device=DEVICE)

    empty = torch.zeros(0, dtype=torch.float32, device=DEVICE)
    model = FacetModel(
        structure_ids=(1,),
        vertex_counts=(5,),
        faces=long(faces),
        site_offsets=long([0, 1, 3]),
        site_facets=long([0, 1, 2]),
        facet_site=long(site_of),
        vertex_offsets=long(np.concatenate(([0], np.cumsum(np.bincount(incidence, minlength=5))))),
        vertex_entries=long(np.argsort(incidence, kind="stable")),
        facet_material=long([0, 0, 0]),
        material_params=empty,
        layer_offsets=long([0]),
        layers=empty,
        key=(),
    )
    corners, sites = _FacetGeometry.apply(vertices, model)
    host = vertices.cpu().double().numpy()
    np.testing.assert_array_equal(corners.cpu().numpy(), host[faces].reshape(3, 9))
    centroid = host[faces].mean(axis=1)
    # Each site sits at its anchor, the first of its members.
    np.testing.assert_allclose(sites.cpu().numpy(), [centroid[0], centroid[1]], rtol=1e-12)
    assert sites.dtype == torch.float64

    leaf = vertices.clone().requires_grad_(True)
    tangent = torch.randn_like(vertices)
    u_corners, u_sites = torch.randn_like(corners), torch.randn_like(sites)
    c, s = _FacetGeometry.apply(leaf, model)
    (grad,) = torch.autograd.grad((c * u_corners).sum() + (s * u_sites).sum(), leaf)
    with fwad.dual_level():
        tc, ts = _FacetGeometry.apply(fwad.make_dual(vertices, tangent), model)
        tc, ts = fwad.unpack_dual(tc).tangent, fwad.unpack_dual(ts).tangent
    left = float((grad.double() * tangent.double()).sum())
    right = float((u_corners.double() * tc.double()).sum() + (u_sites.double() * ts.double()).sum())
    assert left == pytest.approx(right, rel=1e-5, abs=0)
