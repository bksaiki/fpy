"""
Examples: Level-of-detail (LOD) algorithm, anisotropic case.

Intended for testing basic language features.
"""

import fpy2 as fp

# TODO: vectors should be tensors
@fp.fpy(meta={
    'name': 'Level-of-detail (LOD) algorithm, anisotropic case',
    'cite': ['DirectX 11.3 specification, Microsoft-2015']
})
def lod_anisotropic(
    dx_u: fp.Real,
    dx_v: fp.Real,
    dy_u: fp.Real,
    dy_v: fp.Real,
    max_aniso: fp.Real,
):
    dx2 = dx_u ** 2 + dx_v ** 2
    dy2 = dy_u ** 2 + dy_v ** 2
    det = abs(dx_u * dy_v - dx_v * dy_u)
    x_major = dx2 > dy2
    major2 = dx2 if x_major else dy2
    major = fp.sqrt(major2)
    norm_major = fp.round(1.0) / major

    aniso_dir_u = (dx_u if x_major else dy_u) * norm_major
    aniso_dir_v = (dx_v if x_major else dy_v) * norm_major
    aniso_ratio = major2 / det

    # clamp anisotropy ratio and compute LOD
    if aniso_ratio > max_aniso:
        aniso_ratio = max_aniso
        minor = major / aniso_ratio
    else:
        minor = det / major

    # clamp LOD
    if minor < fp.round(1):
        aniso_ratio = max(fp.round(1), aniso_ratio * minor)

    lod = fp.log2(minor)
    return lod, aniso_ratio, aniso_dir_u, aniso_dir_v
