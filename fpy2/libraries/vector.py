"""
Operations on vectors.
"""

from . import base as fp

@fp.fpy
def zeros(n: int) -> list[fp.Real]:
    """
    Create a zero vector of length n.

    :param n: Vector length.
    :return: Zero vector.
    """
    return [0.0 for _ in range(n)]

@fp.fpy
def ones(n: int) -> list[fp.Real]:
    """
    Create a vector of ones of length n.

    :param n: Vector length.
    :return: Vector of ones.
    """
    return [1.0 for _ in range(n)]

@fp.fpy
def add(x: list[fp.Real], y: list[fp.Real]) -> list[fp.Real]:
    """
    Element-wise addition of two vectors.

    :param x: First vector.
    :param y: Second vector.
    :return: Result vector x + y.
    """
    assert len(x) == len(y)
    return [xi + yi for xi, yi in zip(x, y)]

@fp.fpy
def sub(x: list[fp.Real], y: list[fp.Real]) -> list[fp.Real]:
    """
    Element-wise subtraction of two vectors.

    :param x: First vector.
    :param y: Second vector.
    :return: Result vector x - y.
    """
    assert len(x) == len(y)
    return [xi - yi for xi, yi in zip(x, y)]

@fp.fpy
def hadamard(x: list[fp.Real], y: list[fp.Real]) -> list[fp.Real]:
    """
    Element-wise multiplication (Hadamard product) of two vectors.

    :param x: First vector.
    :param y: Second vector.
    :return: Result vector x ⊙ y.
    """
    assert len(x) == len(y)
    return [xi * yi for xi, yi in zip(x, y)]


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
    return fp.pow(sum([fp.pow(abs(xi), p) for xi in x]), fp.R(1) / p)

@fp.fpy
def cosine_similarity(x: list[fp.Real], y: list[fp.Real]) -> fp.Real:
    """
    Compute cosine similarity between two vectors.

    :param x: First vector.
    :param y: Second vector.
    :return: Cosine similarity x·y / (||x|| ||y||).
    """
    dot_xy = dot(x, y)
    norm_x = norm2(x)
    norm_y = norm2(y)
    return dot_xy / (norm_x * norm_y)

@fp.fpy
def normalize(x: list[fp.Real]) -> list[fp.Real]:
    """
    Normalize a vector to unit length (L2 norm).

    :param x: Input vector.
    :return: Unit vector in direction of x.
    """
    norm = norm2(x)
    return [xi / norm for xi in x]

@fp.fpy
def normalize_p(x: list[fp.Real], p: fp.Real) -> list[fp.Real]:
    """
    Normalize a vector using p-norm.

    :param x: Input vector.
    :param p: Norm parameter.
    :return: Vector normalized by p-norm.
    """
    norm = norm_p(x, p)
    return [xi / norm for xi in x]

@fp.fpy
def cross(x: list[fp.Real], y: list[fp.Real]) -> list[fp.Real]:
    """
    Compute cross product of two 3D vectors.

    :param x: First 3D vector.
    :param y: Second 3D vector.
    :return: Cross product x × y.
    """
    assert len(x) == 3 and len(y) == 3
    return [
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0]
    ]

@fp.fpy
def mean(x: list[fp.Real]) -> fp.Real:
    """
    Compute the mean of vector elements.

    :param x: Input vector.
    :return: Mean of elements.
    """
    return sum(x) / fp.R(len(x))

@fp.fpy
def min_element(x: list[fp.Real]) -> fp.Real:
    """
    Find minimum element in vector.
    
    :param x: Input vector.
    :return: Minimum element.
    """
    # assert message: "Vector must not be empty"
    assert len(x) > 0
    result = x[0]
    for xi in x[1:]:
        if xi < result:
            result = xi
    return result

@fp.fpy
def max_element(x: list[fp.Real]) -> fp.Real:
    """
    Find maximum element in vector.
    
    :param x: Input vector.
    :return: Maximum element.
    """
    # assert message: "Vector must not be empty"
    assert len(x) > 0
    result = x[0]
    for xi in x[1:]:
        if xi > result:
            result = xi
    return result
