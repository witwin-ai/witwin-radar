"""Fresnel reflection model for radar surface interaction."""

import drjit as dr


def fresnel(cos_i, epsilon_r=5.0):
    """Compute Fresnel reflectance for an unpolarized electromagnetic wave.

    Args:
        cos_i: Dr.Jit float array, cosine of incidence angle (|dot(-d, n)|).
        epsilon_r: Relative permittivity of the reflecting material.

    Returns:
        Dr.Jit float array with Fresnel reflectance in [0, 1].
    """
    n2 = epsilon_r ** 0.5

    cos_i = dr.maximum(dr.minimum(cos_i, 1.0), 0.0)

    sin_i_sq = 1.0 - cos_i * cos_i
    n2_sq = n2 * n2
    cos_t_sq = 1.0 - sin_i_sq / n2_sq
    cos_t = dr.sqrt(dr.maximum(cos_t_sq, 0.0))

    rs = (cos_i - n2 * cos_t) / (cos_i + n2 * cos_t)
    rp = (n2 * cos_i - cos_t) / (n2 * cos_i + cos_t)
    reflectance = 0.5 * (rs * rs + rp * rp)

    return dr.select(cos_t_sq < 0.0, 1.0, reflectance)
