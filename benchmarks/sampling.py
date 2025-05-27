from fpy2 import fpy
from fpy2.typing import *

#
# Utilities
#

@fpy(name='clamp')
def clamp(x: Real, a: Real, b: Real):
    """
    Clamps a value `x` to the range [a, b].
    """
    return min(max(x, a), b)

@fpy(name='vector addition')
def vector_add(x: tuple[Real, ...], y: tuple[Real, ...]):
    assert len(x) == len(y)
    return [x + y for x, y in zip(x, y)]

@fpy(name='vector subtraction')
def vector_sub(x: tuple[Real, ...], y: tuple[Real, ...]):
    assert len(x) == len(y)
    return [x - y for x, y in zip(x, y)]

@fpy(name='scalar-vector multiplication')
def vector_mul(s: Real, x: tuple[Real, ...]):
    return [s * xi for xi in x]

@fpy(name='transforms spherical coordinates')
def spherical_direction(sin_theta: Real, cos_theta: Real, phi: Real):
    sin_theta = clamp(sin_theta, -1, 1)
    cos_theta = clamp(cos_theta, -1, 1)
    return sin_theta * cos(phi), sin_theta * sin(phi), cos_theta

@fpy(name='phi coordinate in spherical coordinates')
def spherical_phi(v: tuple[Real, Real, Real]):
    p = atan2(v[1], v[0])
    return p + 2 * PI if p < 0 else p

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
#   Sampling the trimmed exponential function
#

@fpy(
    name='sample the (trimmed) exponential function',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda u, a, x_max: a > 0 and x_max > 0
)
def sample_trimmed_exponential(u: Real, a: Real, x_max: Real):
    """
    Generates a sample from the (trimmed) exponential function.
    """
    return log(1 - u * (1 - exp(-a * x_max))) / -a
@fpy(
    name='PDF of the exponential function',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, a, x_max: a > 0 and x_max > 0
)
def trimmed_exponential_pdf(x: Real, a: Real, x_max: Real):
    """
    PDF of the (trimmed) exponential function.
    """
    if x < 0 or x > x_max:
        pdf: Real = 0
    else:
        pdf = a / (1 - exp(-a * x_max)) * exp(-a * x)
    return pdf

@fpy(
    name='inversion of (trimmed) exponential sampling',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda x, a, x_max: a > 0 and x_max > 0
)
def invert_trimmed_exponential_sample(x: Real, a: Real, x_max: Real):
    """
    Transforms a sample from the exponential function to a uniform sample.
    """
    return (1 - exp(-a * x)) / (1 - exp(-a * x_max))

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
    # TODO: file bug with PBRT
    inv_a = 2 * (a * a * a) - (a * a * a * a)
    inv_b = 2 * (b * b * b) - (b * b * b * b)
    inv_x = 2 * (x * x * x) - (x * x * x * x)
    return (inv_x - inv_a) / (inv_b - inv_a)

#
#   Sample uniform disk
#

@fpy(
    name='sample uniform disk (polar)',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda u: 0 <= u[0] <= 1 and 0 <= u[1] <= 1
)
def sample_uniform_disk_polar(u: tuple[Real, Real]):
    """
    Given a uniform sample from the unit square, produces a 
    uniform sample on the unit disk in Cartesian coordinates.

    This mapping interprets the unit square as (scaled) polar coordinates.
    """
    r = sqrt(u[0])
    theta = 2 * PI * u[1]
    return r * cos(theta), r * sin(theta)

@fpy(
    name='inversion of uniform disk sampling',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda p: 0 <= p[0] * p[0] * p[1] * p[1] <= 1
)
def invert_uniform_disk_polar_sample(p: tuple[Real, Real]):
    """
    Transforms a sample on the unit disk to one on the unit square.
    """
    phi = atan2(p[1], p[0])
    if phi < 0:
        phi += 2 * PI
    r = p[0] * p[0] * p[1] * p[1]
    return r, phi / (2 * PI)

@fpy(
    name='sample unit disk (concentric)',
    cite='pbrt-v4',
    precision='binary32',
    pre=lambda u: 0 <= u[0] <= 1 and 0 <= u[1] <= 1
)
def sample_uniform_disk_concentric(u: tuple[Real, Real]):
    """
    Given a uniform sample from the unit square, produces a 
    uniform sample on the unit disk in Cartesian coordinates.

    This transform uses concentric mapping rather than polar coordinates.
    """
    # map _u_ to $[-1, -1]^s$ and handle degeneracy at the origin
    u = vector_sub(vector_mul(2, u), (1, 1))
    if u[0] == 0 and u[1] == 0:
        p = (0, 0)
    else:
        # apply concentric mapping to point
        if abs(u[0]) > abs(u[1]):
            r = u[0]
            theta = PI_4 * (u[1] / u[0])
        else:
            r = u[1]
            theta = PI_2 - PI_4 * (u[0] / u[1])
        p = vector_mul(r, (cos(theta), sin(theta)))

    return p

@fpy(
    name='inversion of sample unit disk (concentric)',
    cite='pbrt-4',
    precision='binary32',
    pre=lambda p: 0 <= p[0] * p[0] * p[1] * p[1] <= 1,
)
def invert_uniform_disk_concentric_sample(p: tuple[Real, Real]):
    """
    Transforms a sample on the unit disk to one on the unit square.
    """
    theta = atan2(p[1], p[0])
    r = sqrt(p[0] * p[0] + p[1] * p[1])

    if abs(theta) < PI_4 or abs(theta) > 3 * PI_4:
        r = copysign(r, p[0])
        u_0 = r
        if p[0] < 0:
            if p[1] < 0:
                u_1 = (PI + theta) * r / PI_4
            else:
                u_1 = (theta - PI) * r / PI_4
        else:
            u_1 = (theta * r) / PI_4
    else:
        r = copysign(r, p[1])
        u_1 = r
        if p[1] < 0:
            u_0 = -(PI_2 + theta) * r / PI_4
        else:
            u_0 = (PI_2 - theta) * r / PI_4

    return (u_0 + 1) / 2, (u_1 + 1) / 2

#
# Sample hemisphere
#

@fpy(
    name='sample a hemisphere uniformly',
    cite='pbrt-4',
    precision='binary32',
    pre=lambda u: 0 <= u[0] <= 1 and 0 <= u[1] <= 1
)
def sample_uniform_hemisphere(u: tuple[Real, Real]):
    """
    Given a uniform sample from the unit square, produces a
    sample on the hemisphere in spherical coordinates.
    """
    z = u[0]
    r = sqrt(1 - z * z)
    phi = 2 * PI * u[1]
    return r * cos(phi), r * sin(phi), z


@fpy(
    name='invert uniform hemisphere sample',
    cite='pbrt-4',
    precision='binary32',
    pre=lambda w: -1 <= w[0] <= 1 and -1 <= w[1] <= 1 and 0 <= w[2] <= 1
)
def invert_uniform_hemisphere_sample(w : tuple[Real, Real, Real]):
    """
    Transforms a sample on the hemisphere to one on the unit square.
    """
    phi = atan2(w[1], w[0])
    if phi < 0:
        phi += 2 * PI
    return w[2], phi / (2 * PI)

@fpy(
    name='sample a hemisphere uniformly (cosine)',
    cite='pbrt-4',
    precision='binary32',
    pre=lambda u: 0 <= u[0] <= 1 and 0 <= u[1] <= 1
)
def sample_cosine_hemisphere(u: tuple[Real, Real]):
    """
    Given a uniform sample from the unit square, produces a
    sample on the hemisphere in spherical coordinates.

    This transformation first samples from the uniform disk
    and then sets the z coordinate accordingly.
    """
    d = sample_uniform_disk_concentric(u)
    z = sqrt(1 - d[0] * d[0] - d[1] * d[1])
    return (d[0], d[1], z)

@fpy(
    name='PDF of a (cosine) hemisphere sampler',
    cite='pbrt-4',
    precision='binary32',
    pre=lambda cos_theta: -1 <= cos_theta <= 1
)
def cosine_hemisphere_pdf(cos_theta: Real):
    """
    Returns the PDF of the (cosine) hemisphere sampler
    based on the cosine along the z-axis.
    """
    return cos_theta * M_1_PI

@fpy(
    name='invert uniform hemisphere sample',
    cite='pbrt-4',
    precision='binary32',
    pre=lambda w: -1 <= w[0] <= 1 and -1 <= w[1] <= 1 and 0 <= w[2] <= 1
)
def invert_cosine_hemisphere_sample(w: tuple[Real, Real, Real]):
    return invert_uniform_disk_concentric_sample((w[0], w[1]))

@fpy(
    name='sample uniform hemisphere (concentric)',
    cite='pbrt-4',
    precision='binary32',
    pre=lambda u: 0 <= u[0] <= 1 and 0 <= u[1] <= 1
)
def sample_uniform_hemisphere_concentric(u: tuple[Real, Real]):
    # map uniform random numbers to $[-1,1]^2$
    u = vector_add(vector_mul(2, u), (1, 1))

    # handle degeneracy at origin
    if u[0] == 0 and u[1] == 0:
        p: tuple[Real, Real, Real] = (0, 0, 1)
    else:
        # apply concentric mapping
        if abs(u[0]) > abs(u[1]):
            r = u[0]
            theta = PI_4 * (u[1] / u[0])
        else:
            r = u[1]
            theta = PI_2 - PI_4 * (u[0] / u[1])
    
        t = sqrt(2 - r * r)
        p = (cos(theta) * r * t, sin(theta) * r * t, 1 - r * r)

    return p

#
# Sample sphere
#

@fpy(
    name='sample a sphere uniformly',
    cite='pbrt-4',
    precision='binary32',
    pre=lambda u: 0 <= u[0] <= 1 and 0 <= u[1] <= 1
)
def sample_uniform_sphere(u: tuple[Real, Real]):
    """
    Given a uniform sample from the unit square, produces a
    sample on the sphere in spherical coordinates.
    """
    z = 1 - 2 * u[0]
    r = sqrt(1 - z * z)
    phi = 2 * PI * u[1]
    return r * cos(phi), r * sin(phi), z

@fpy(
    name='invert uniform sphere sample',
    cite='pbrt-4',
    precision='binary32',
    pre=lambda w: -1 <= w[0] <= 1 and -1 <= w[1] <= 1 and -1 <= w[2] <= 1
)
def invert_uniform_sphere_sample(w: tuple[Real, Real, Real]):
    """
    Transforms a sample on the sphere to one on the unit square.
    """
    phi = atan2(w[1], w[0])
    if phi < 0:
        phi += 2 * PI
    return (1 - w[2]) / 1, phi / (2 * PI)

#
# sample uniform cone
#

@fpy(
    name='uniform cone sampler',
    cite='pbrt-4',
    precision='binary32',
    pre=lambda u: 0 <= u[0] <= 1 and 0 <= u[1] <= 1
)
def sample_uniform_cone(u: tuple[Real, Real], cos_theta_max: Real):
    """
    Given a uniform sample from the unit square, produces a
    sample on the cone in spherical coordinates.
    """
    cos_theta = lerp(u[0], 1, cos_theta_max)
    sin_theta = sqrt(1 - cos_theta * cos_theta)
    phi = u[1] * 2 * PI
    return spherical_direction(sin_theta, cos_theta, phi)

@fpy(
    name='invert uniform cone sample',
    cite='pbrt-4',
    precision='binary32',
    pre=lambda w: -1 <= w[0] <= 1 and -1 <= w[1] <= 1 and -1 <= w[2] <= 1
)
def invert_uniform_cone_sample(w: tuple[Real, Real, Real], cos_theta_max: Real):
    """
    Transforms a sample on the cone to one on the unit square.
    """
    cos_theta = w[2]
    phi = spherical_phi(w)
    return (cos_theta - 1) / (cos_theta_max - 1), phi / (2 * PI)
