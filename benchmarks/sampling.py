from fpy2 import fpy
from fpy2.typing import *

@fpy(name='clamp')
def clamp(x: Real, a: Real, b: Real):
    """
    Clamps a value `x` to the range [a, b].
    """
    return min(max(x, a), b)

#
# Sampling a linear function
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
def sample_linear(u: Real, a: Real, b: Real):
    """
    Generates a linear sample on `[a, b]` from a uniform sample `u`.
    """
    if u == 0 and a == 0:
        # TODO: why is this here?
        x: Real = 0
    else:
        x = u * (a + b) / (a + sqrt(lerp(u, a * a, b * b)))
        x = min(x, hexfloat('0x1.fffffep-1'))
    return x

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
# Uniform sampling of a triangle
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

@fpy(
    name='inversion of uniform sampling of a triangle',
    precision='binary32',
    cite='pbrt-v4',
    pre=lambda b: 0 <= b[0] <= 1 and 0 <= b[1] <= 1 and 0 <= b[2] <= 1
)
def invert_uniform_triangle_sample(b: tuple[Real, Real, Real]):
    """
    Transforms a uniform sample on a triangle in barycentric coordinates
    to a uniform sample on the unit square.
    """
    if b[0] > b[1]:
        # b0 = u[0] - u[1] / 2, b1 = u[1] / 2
        u0 = b[0] + b[1]
        u1 = 2 * b[1]
    else:
        # b1 = u[1] - u[0] / 2, b0 = u[0] / 2
        u0 = 2 * b[0]
        u1 = b[1] + b[0]
    return u0, u1

#
# Sampling the "tent" function of radius r
#

@fpy(
    name='sample discrete',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda u: 0 <= u <= 1
)
def sample_discrete(u: Real):
    """
    Splits a uniform sample into [0, 0.5) and [0.5, 1).

    Returns which half it is in and the remapped sample.
    """
    if u < 0.5:
        lower = True
        remapped = 2 * u  # maps [0, 0.5) to [0, 1)
    else:
        lower = False
        remapped = 2 * u - 1  # maps [0.5, 1) to [0, 1)
    return lower, remapped

@fpy(
    name='tent sampling',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda u, r: 0 <= u <= 1 and r > 0
)
def sample_tent(u: Real, r: Real):
    """
    Generates a sample from the "tent" function of radius `r`.

    The tent function is defined to be `f(x) = r - |x|`
    when `|x| < r` and 0 otherwise.
    """
    lower, u2 = sample_discrete(u)
    if lower:
        x = -r + r * sample_linear(u2, 0, 1)
    else:
        x = r * sample_linear(u2, 1, 0)
    return x

@fpy(
    name='PDF of tent function',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, r: r > 0
)
def tent_pdf(x : Real, r: Real):
    """
    PDF of sampling the "tent" function of radius `r`.
    """
    if abs(x) >= r:
        pdf: Real = 0
    else:
        pdf = 1 / r - abs(x) / (r * r)
    return pdf

@fpy(
    name='inversion of tent sampling',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, r: -r <= x <= r and r > 0
)
def invert_tent_sample(x: Real, r: Real):
    """
    Transforms a sample from the "tent" function of radius `r`
    to a uniform sample.
    """
    if x <= 0:
        u = 1 - invert_linear_sample(-x / r, 1, 0) / 2
    else:
        u = invert_linear_sample(x / r, 0, 1) / 2
    return u

#
# Sampling the exponential function
#

@fpy(
    name='PDF of the exponential function',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, a: a > 0
)
def exponential_pdf(x: Real, a: Real):
    """
    PDF of the exponential function.
    """
    return a * exp(-a * x)

@fpy(
    name='Sample the exponential function',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda u, a: a > 0
)
def sample_exponential(u: Real, a: Real):
    """
    Generates a sample from the exponential function.
    """
    return -log(1 - u) / a

@fpy(
    name='inversion of exponential sampling',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, a: a > 0
)
def invert_exponential_sample(x: Real, a: Real):
    """
    Transforms a sample from the exponential function to a uniform sample.
    """
    return 1 - exp(-a * x)

#
# Sampling the Gaussian distribution
#

@fpy(
    name='PDF of Gaussian sampling',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, a: a > 0
)
def normal_pdf(x: Real, mu: Real, sigma: Real):
    """
    PDF of sampling from a Gaussian distribution.
    """
    mu = x - mu
    scale = 1 / (abs(sigma) * sqrt(2 * PI))
    unscaled = exp(-(mu * mu) / (2 * sigma * sigma))
    return scale * unscaled

@fpy(
    name='Sample Gaussian distribution',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda u: 0 <= u <= 1
)
def sample_normal(u: Real, mu: Real, sigma: Real):
    """
    Generates a sample from a Gaussian distribution.
    """
    return mu + SQRT2 * sigma * erfc(2 * u - 1)

@fpy(
    name='inversion of Gaussian sampling',
    cite='pbrt-v4',
    precision='binary32',
)
def invert_normal_sample(x: Real, mu: Real, sigma: Real):
    """
    Transforms a sample from a Gaussian distribution to a uniform sample.
    """
    return 0.5 * (1 + erf((x - mu) / (sigma * SQRT2)))

#
# Sampling the logistic function
#

@fpy(
    name='PDF of logistic sampling',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, s: s > 0
)
def logistic_pdf(x: Real, s: Real):
    """
    PDF of sampling from a logistic distribution.
    """
    x = abs(x)
    t = 1 + exp(-x / s)
    return exp(-x / s) / (s * t * t)

@fpy(
    name='Sample logistic distribution',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda u: 0 <= u <= 1
)
def sample_logistic(u: Real, s: Real):
    """
    Generates a sample from a logistic distribution.
    """
    return -s * log(1 / u - 1)

@fpy(
    name='inversion of logistic sampling',
    cite='pbrt-v4',
    precision='binary32',
)
def invert_logistic_sample(x: Real, s: Real):
    """
    Transforms a sample from a logistic distribution to a uniform sample.
    """
    return 1 / (1 + exp(-x / s))

#
# Sampling the "trimmed" logistic function
#

@fpy(
    name='PDF of trimmed logistic sampling',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, a, b, s: s > 0 and a <= b
)
def trimmed_logistic_pdf(x: Real, a: Real, b: Real, s: Real):
    """
    PDF of sampling from a "trimmed" logistic distribution.
    """

    if x < a or x > b:
        pdf = 0
    else:
        t = logistic_pdf(x, x)
        inv_a = invert_logistic_sample(a, s)
        inv_b = invert_logistic_sample(b, s)
        pdf = t / (inv_b - inv_a)

    return pdf

@fpy(
    name='Sample trimmed logistic distribution',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda u, a, b, s: a <= b and s > 0
)
def sample_trimmed_logistic(u: Real, a: Real, b: Real, s: Real):
    """
    Generates a sample from a "trimmed" logistic distribution.
    """
    inv_a = invert_logistic_sample(a, s)
    inv_b = invert_logistic_sample(b, s)
    u = lerp(u, inv_a, inv_b)

    x = sample_logistic(u, s)
    return clamp(x, a, b)

@fpy(
    name='inversion of trimmed logistic sampling',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, a, b, s: a < b and s > 0
)
def invert_trimmed_logistic_sample(x: Real, a: Real, b: Real, s: Real):
    """
    Transforms a sample from a "trimmed" logistic distribution to a uniform sample.
    """
    inv_a = invert_logistic_sample(a, s)
    inv_b = invert_logistic_sample(b, s)
    inv_x = invert_logistic_sample(x, s)
    return (inv_x - inv_a) / (inv_b - inv_a)

#
# Sampling a cubic interpolant
#

@fpy(
    name='cubic interpolation',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, a, b: a <= b
)
def smooth_step(x: Real, a: Real, b: Real):
    """
    Cubic interpolation between two value `a` and `b`.

    Given a range [a, b] and a value x, the function returns
    0 when `x <= a`, 1 when `x >= b`, and a interpolates between
    them usig a cubic polynomial that ensures that the first
    derivative is continuous.
    """
    if a == b:
        if x < a:
            y = 0
        else:
            y = 1
    else:
        t = clamp((x - a) / (b - a), 0, 1)
        y = t * t * (3 - 2 * t)

    return y


@fpy(
    name='PDF of cubic interpolation',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, a, b: a <= b
)
def smooth_step_pdf(x: Real, a: Real, b: Real):
    """
    PDF of sampling a cubic interpolant between two values.
    """
    if x < a or x > b:
        pdf = 0
    else:
        pdf = 2 / (b - a) * smooth_step(x, a, b)
    return pdf


@fpy(name='single step of Newton iteration')
def _sample_smooth_step_newton_once(x: Real, u: Real, a: Real, b: Real):
    """
    Single step of Newton iteration for the cubic interpolant.
    """
    t = (x - a) / (b - a)
    p = 2 * (t * t * t) - (t * t * t * t)
    pderiv = smooth_step_pdf(x, a, b)
    return p - u, pderiv


@fpy(
    name='Sample cubic interpolation',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda u, a, b: a <= b
)
def sample_smooth_step(u: Real, a: Real, b: Real):
    """
    Generates a sample from a cubic interpolant between two values.
    """
    eps = 1e-6

    # initial endpoints
    x0 = a
    x1 = b

    # compute outputs
    fx0, _ = _sample_smooth_step_newton_once(x0, u, a, b)
    fx1, _ = _sample_smooth_step_newton_once(x1, u, a, b)

    if abs(fx0) < eps:
        x = x0
    elif abs(fx1) < eps:
        x = x1
    else:
        start_negative = fx0 < 0

        # initial midpoint
        xmid = x0 + (x1 - x0) * -fx0 / (fx1 - fx0)

        # fall back to bisection is _xmid_ is out of bounds
        if xmid <= x0 or xmid >= x1:
            xmid = (x0 + x1) / 2

        # evaluate the function at the midpoint
        fmid, fmid_deriv = _sample_smooth_step_newton_once(xmid, u, a, b)
        if start_negative == (fmid < 0):
            x0 = xmid
        else:
            x1 = xmid

        # Iterate
        while x1 - x0 >= eps and abs(fmid) >= eps:
            # recompute the mid point
            xmid -= fmid / fmid_deriv

            # fall back to bisection is _xmid_ is out of bounds
            if xmid <= x0 or xmid >= x1:
                xmid = (x0 + x1) / 2

            # evaluate the function at the midpoint
            fmid, fmid_deriv = _sample_smooth_step_newton_once(xmid, u, a, b)
            if start_negative == (fmid < 0):
                x0 = xmid
            else:
                x1 = xmid

        x = xmid

    return x


@fpy(
    name='inversion of cubic interpolation',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, a, b: a <= x <= b
)
def invert_smooth_step_sample(x: Real, a: Real, b: Real):
    """
    Transforms a sample from a cubic interpolant between two values to a uniform sample.
    """
    t = (x - a) / (b - a)
    inv_a = 2 * (a * a * a) - (a * a * a * a)
    inv_b = 2 * (b * b * b) - (b * b * b * b)
    inv_x = 2 * (x * x * x) - (x * x * x * x)
    return (inv_x - inv_a) / (inv_b - inv_a)
