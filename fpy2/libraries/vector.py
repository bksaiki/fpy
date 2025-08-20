"""
Operations on vectors.
"""

from . import base as fp

@fp.fpy
def dot(x: list[fp.Real], y: list[fp.Real]) -> fp.Real:
    """
    Compute the dot product of two vectors.

    :param x: First vector.
    :param y: Second vector.
    :return: Dot product of x and y.
    """
    assert len(x) == len(y)
    return sum([a * b for a, b in zip(x, y)])

@fp.fpy
def axpy(a: fp.Real, x: list[fp.Real], y: list[fp.Real]):
    """
    Compute a*x + y (AXPY operation).

    :param a: Scalar multiplier.
    :param x: First vector.
    :param y: Second vector.
    :return: Result vector a*x + y.
    """
    assert len(x) == len(y)
    return [a * xi + yi for xi, yi in zip(x, y)]


@fp.fpy
def scale(a: fp.Real, x: list[fp.Real]):
    """
    Scale a vector by a scalar.

    :param a: Scalar multiplier.
    :param x: Input vector.
    :return: Result vector a*x.
    """
    return [a * xi for xi in x]

@fp.fpy
def dot_add(x: list[fp.Real], y: list[fp.Real], c: fp.Real):
    """
    Compute `xy + c`, dot product with addition.

    :param x: First vector.
    :param y: Second vector.
    :param c: Scalar to add.
    :return: Result vector x*y + c.
    """
    return dot(x, y) + c

@fp.fpy
def norm1(x: list[fp.Real]) -> fp.Real:
    """
    Compute the L1 norm (Manhattan norm) of a vector.

    :param x: Input vector.
    :return: L1 norm of x.
    """
    return sum([abs(xi) for xi in x])

@fp.fpy
def norm2(x: list[fp.Real]) -> fp.Real:
    """
    Compute the L2 norm (Euclidean norm) of a vector.

    :param x: Input vector.
    :return: L2 norm of x.
    """
    return fp.sqrt(sum([xi * xi for xi in x]))

@fp.fpy
def norm_inf(x: list[fp.Real]) -> fp.Real:
    """
    Compute the infinity norm (maximum norm) of a vector.

    :param x: Input vector.
    :return: Infinity norm of x.
    """

    assert len(x) > 0

    # TODO: should there be `max(<iterable>)`?
    t = abs(x[0])
    for xi in x[1:]:
        t = max(t, abs(xi))
    return t

@fp.fpy
def norm_p(x: list[fp.Real], p: fp.Real) -> fp.Real:
    """
    Compute the p-norm of a vector.

    :param x: Input vector.
    :param p: Norm parameter (p >= 1).
    :return: p-norm of x.
    """
    return fp.pow(sum([fp.pow(abs(xi), p) for xi in x]), 1.0 / p)
