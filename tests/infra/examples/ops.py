"""
Example functions for each operation.
"""

import fpy2 as fp

@fp.fpy
def test_logb(x: fp.Real) -> fp.Real:
    """Example function for `logb`."""
    return fp.logb(x)

@fp.fpy
def test_logb_guarded(x: fp.Real) -> fp.Real:
    """`logb` of a finite non-zero value, which is a small integer.

    The one shape that reaches an integer storage through a value class: the
    guards rule out the NaN and the ``+inf``, ``x == 0`` rules out the ``-inf``
    that ``logb(0)`` gives, and the ``REAL`` block keeps the exact integer
    format -- under a concrete context the class is the one the context can
    represent, which is every class.
    """
    if fp.isnan(x) or fp.isinf(x) or x == 0:
        return 0
    else:
        with fp.REAL:
            return fp.logb(x)

@fp.fpy
def test_pow(x: fp.Real, y: fp.Real) -> fp.Real:
    """Example function for `pow`."""
    return x ** y

@fp.fpy
def test_empty1(n: fp.Real):
    """Example function for `empty`."""
    arr = fp.empty(n)
    for i in range(n):
        arr[i] = i
    return arr

@fp.fpy
def test_empty2(m: fp.Real, n: fp.Real):
    """Example function for `empty`."""
    arr = fp.empty(m, n)
    for i in range(m):
        for j in range(n):
            arr[i][j] = i * n + j
    return arr

@fp.fpy
def test_empty3(k: fp.Real, m: fp.Real, n: fp.Real):
    """Example function for `empty`."""
    arr = fp.empty(k, m, n)
    for i in range(k):
        for j in range(m):
            for l in range(n):
                arr[i][j][l] = (i * m + j) * n + l
    return arr
