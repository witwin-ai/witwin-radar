"""The native facet operators against independent float64 references.

``facet_response`` is checked against a brute-force physical-optics integral
over each triangle (midpoint quadrature on a fine barycentric grid, written
here in NumPy, not the kernel's closed-form divided difference), against the
independent Airy slab coefficient for the material law, and in both AD modes
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
        # Inbound waves propagate toward +x (into the facets), outbound leave toward -x.
        self.dir_in = _unit(rng, 3, (1.0, 0.1, -0.1))
        self.dir_out = _unit(rng, 2, (-1.0, 0.15, 0.1))
        self.idx_in = np.array([0, 1, 2, 0])
        self.idx_out = np.array([0, 1, 0, 1])
        self.idx_site = np.array([0, 0, 1, 1])
        self.k = float(wavenumber)
        self.polarization = (0.0, 0.8, 0.6)
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
            return torch.as_tensor(values, dtype=torch.float32, device=DEVICE)

        return _FacetLaw(
            site_offsets=long(self.site_offsets),
            site_facets=long(self.site_facets),
            facet_site=long(self.facet_site),
            facet_material=long(np.zeros(len(self.corners), dtype=np.int64)),
            material_params=single(self.material_params),
            layer_offsets=long(self.layer_offsets),
            layers=single(self.layers),
            wavenumber=self.k,
            polarization=self.polarization,
            seed=self.seed,
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

        return [single(self.dir_in), single(self.dir_out), single(self.corners), single(self.sites)]

    def evaluate(self, dir_in, dir_out, corners, sites, valid=None):
        from witwin.radar.scattering import _FacetResponse

        def long(values):
            return torch.as_tensor(values, dtype=torch.int64, device=DEVICE)

        flags = torch.ones(4, dtype=torch.int32, device=DEVICE) if valid is None else valid
        return _FacetResponse.apply(
            dir_in,
            dir_out,
            corners,
            sites,
            long(self.idx_in),
            long(self.idx_out),
            long(self.idx_site),
            flags,
            self.law(),
            self.tables(),
        )


def _co_polar_reference(fixture: Fixture, normal, u_in, cosine) -> complex:
    """The co-polar coefficient at ``cosine`` with the polarization weight, in float64."""

    if fixture.material_params[0, 0] == 1:
        return -1.0 + 0.0j
    _, eps_re, eps_im, _ = fixture.layers[0]
    plane = np.cross(normal, u_in)
    p = np.asarray(fixture.polarization)
    p_perp2 = p @ p - (p @ u_in) ** 2
    weight = (p @ plane) ** 2 / ((plane @ plane) * p_perp2)
    frequency = fixture.k * ref.C0_M_PER_S / (2.0 * math.pi)
    sigma = -eps_im * 2.0 * math.pi * frequency * ref.EPS0_F_PER_M
    common = {
        "eps_r": eps_re,
        "sigma_e": sigma,
        "thickness_m": fixture.layers[0, 0],
        "frequency_hz": frequency,
        "cos_incidence": cosine,
    }
    te = ref.slab_reflection_coefficient(polarization="te", **common)
    tm = ref.slab_reflection_coefficient(polarization="tm", **common)
    # The reference TM coefficient is the magnetic-field one; its negative is the tangential convention.
    return weight * te - (1.0 - weight) * tm


def _reference_rows(fixture: Fixture, grid: int = 160) -> np.ndarray:
    """Brute-force PO: a midpoint sum of exp(j k q.(r - x_s)) over each lit triangle."""

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
        u_in = -fixture.dir_in[fixture.idx_in[row]]
        u_out = fixture.dir_out[fixture.idx_out[row]]
        site = fixture.idx_site[row]
        total = 0.0j
        for entry in range(fixture.site_offsets[site], fixture.site_offsets[site + 1]):
            v0, v1, v2 = fixture.corners[fixture.site_facets[entry]].reshape(3, 3)
            cross = np.cross(v1 - v0, v2 - v0)
            area = 0.5 * np.linalg.norm(cross)
            normal = cross / (2.0 * area)
            ci, co = normal @ u_in, normal @ u_out
            if ci <= 0 or co <= 0:
                continue
            q = u_in + u_out
            points = v0[None, :] + cells[:, :1] * (v1 - v0)[None, :] + cells[:, 1:] * (v2 - v0)[None, :]
            integral = np.exp(1j * k * (points - fixture.sites[site]) @ q).mean() * area
            gamma = _co_polar_reference(fixture, normal, u_in, ci)
            total += 1j * (k * k / math.pi) * 0.5 * (ci + co) * gamma * integral
        rows.append(total)
    return np.asarray(rows)


@pytest.mark.parametrize("material", ["pec", (5.24, -0.18), (2.5, -0.01)], ids=["pec", "concrete", "board"])
def test_the_response_is_the_physical_optics_integral(material):
    fixture = Fixture(material=material)
    s_re, s_im = fixture.evaluate(*fixture.inputs())
    measured = (s_re.double() + 1j * s_im.double()).cpu().numpy()
    expected = _reference_rows(fixture)
    assert np.all(np.abs(expected) > 0)
    np.testing.assert_allclose(measured, expected, rtol=2e-4, atol=0)


def test_a_broadside_plate_triangle_reads_four_pi_area_over_lambda_squared():
    """No quadrature: the closed form at the one point where it is trivial."""

    fixture = Fixture()
    corners = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.04, 0.0]], device=DEVICE)
    fixture.corners = corners.cpu().numpy().astype(np.float64)
    fixture.site_facets, fixture.site_offsets, fixture.facet_site = np.array([0]), np.array([0, 1, 1]), np.array([0])
    fixture.sites = np.stack((fixture.corners.reshape(3, 3).mean(axis=0), np.zeros(3)))
    fixture.dir_in = np.array([[1.0, 0.0, 0.0]] * 3)
    fixture.dir_out = np.array([[-1.0, 0.0, 0.0]] * 2)
    fixture.idx_site = np.array([0, 0, 0, 0])
    inputs = fixture.inputs()
    s_re, s_im = fixture.evaluate(*inputs)
    wavelength = 2.0 * math.pi / fixture.k
    expected = 4.0 * math.pi * (0.5 * 0.03 * 0.04) / wavelength**2
    # PEC, Gamma = -1, times j: S = -j 4 pi A / lambda^2.
    assert float(s_re[0]) == pytest.approx(0.0, abs=1e-6 * expected)
    assert float(s_im[0]) == pytest.approx(-expected, rel=1e-6, abs=0)


def test_a_dead_row_publishes_exactly_zero_and_a_back_face_contributes_nothing():
    fixture = Fixture()
    valid = torch.tensor([1, 0, 1, 1], dtype=torch.int32, device=DEVICE)
    s_re, s_im = fixture.evaluate(*fixture.inputs(), valid=valid)
    assert float(s_re[1]) == 0.0 and float(s_im[1]) == 0.0
    flipped = Fixture()
    flipped.corners = flipped.corners.reshape(3, 3, 3)[:, [0, 2, 1]].reshape(3, 9)
    s_re, s_im = flipped.evaluate(*flipped.inputs())
    assert torch.count_nonzero(s_re).item() == 0 and torch.count_nonzero(s_im).item() == 0


def _loss_directions(fixture: Fixture, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tangents = [torch.randn(value.shape, generator=generator).to(DEVICE) for value in fixture.inputs()]
    weights = torch.randn((2, 4), generator=generator).to(DEVICE)
    return tangents, weights


@pytest.mark.parametrize("rms", [0.0, 0.004], ids=["smooth", "rough"])
@pytest.mark.parametrize("material", ["pec", (5.24, -0.18)], ids=["pec", "concrete"])
def test_both_ad_modes_match_finite_differences_and_each_other(material, rms):
    import torch.autograd.forward_ad as fwad

    fixture = Fixture(material=material, rms=rms)
    tangents, weights = _loss_directions(fixture, seed=3)
    inputs = [value.requires_grad_(True) for value in fixture.inputs()]
    s_re, s_im = fixture.evaluate(*inputs)
    loss = (weights[0] * s_re + weights[1] * s_im).sum()
    grads = torch.autograd.grad(loss, inputs)
    reverse = sum(float((g.double() * t.double()).sum()) for g, t in zip(grads, tangents, strict=True))

    with fwad.dual_level():
        duals = [fwad.make_dual(value.detach(), tangent) for value, tangent in zip(inputs, tangents, strict=True)]
        t_re, t_im = fixture.evaluate(*duals)
        forward = float(
            (weights[0] * fwad.unpack_dual(t_re).tangent + weights[1] * fwad.unpack_dual(t_im).tangent).sum()
        )

    def value(step: float) -> float:
        shifted = [(v.detach().double() + step * t.double()).float() for v, t in zip(inputs, tangents, strict=True)]
        a, b = fixture.evaluate(*shifted)
        return float((weights[0].double() * a.double() + weights[1].double() * b.double()).sum())

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
    host = vertices.cpu().numpy()
    np.testing.assert_array_equal(corners.cpu().numpy(), host[faces].reshape(3, 9))
    centroid = host[faces].mean(axis=1)
    np.testing.assert_allclose(sites.cpu().numpy(), [centroid[0], centroid[1:].mean(axis=0)], rtol=1e-6)

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
