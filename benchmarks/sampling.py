from fpy2 import fpy
from fpy2.typing import *

#
#   Sampling based on linear interpolation
#

@fpy(
    name='linear interpolation',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda t, a, b: 0 <= t <= 1
)
def lerp(t: Real, a: Real, b: Real):
    """Linear interpolation between two points `a` and `b`."""
    return (1 - t) * a + t * b

@fpy(
    name='PDF of linear sampling',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda t, a, b: 0 <= t <= 1
)
def linear_pdf(t: Real, a: Real, b: Real):
    """PDF of sampling linearly between two points `a` and `b`."""
    if t < 0 or t > 1:
        pdf = 0
    else:
        pdf = 2 * lerp(t, a, b) / (a + b)
    return pdf

@fpy(
    name='linear sampling',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda u, a, b: 0 <= u <= 1
)
def linear_sample(u: Real, a: Real, b: Real):
    """
    Generates a linear sample on `[a, b]` from a uniform sample `u`.
    """
    if u == 0 and a == 0:
        # TODO: why is this here?
        cdf: Real = 0
    else:
        cdf = u * (a + b) / (a + sqrt(lerp(u, a * a, b * b)))
        cdf = min(cdf, hexfloat('0x1.fffffep-1'))
    return cdf

@fpy(
    name='inversion of linear sampling',
    precision='binary32',
    pre=lambda x, a, b: a <= x <= b
)
def invert_linear_sample(x: Real, a: Real, b: Real):
    """
    Transforms a linear sample on [a, b] to a uniform sample.
    """
    return x * (a * (2 - x) + b * x) / (a + b)

#
#   Uniform sampling of a triangle
#

@fpy(
    name='uniform sampling of a triangle',
    precision='binary32',
    cite='pbrt-v4',
    pre=lambda u: 0 <= u[0] <= 1 and 0 <= u[1] <= 1
)
def sample_uniform_triangle(u: tuple[Real, Real]):
    """
    Given a uniform sample on the unit square, generates
    a uniform sample on a triangle in barycentric coordinates.
    """
    if u[0] < u[1]:
        b0 = u[0] / 2
        b1 = u[1] - b0
    else:
        b1 = u[1] / 2
        b0 = u[0] - b1
    return b0, b1, 1 - b0 - b1
