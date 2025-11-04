"""
Examples hosted on readthedocs.io
"""

import fpy2 as fp

@fp.fpy
def dot_prod(a: list[fp.Real], b: list[fp.Real]) -> fp.Real:
    """
    Computes the dot product of two vectors.

    Parameters:
        a: First vector.
        b: Second vector.

    Returns:
        The dot product of the two vectors, correctly rounded under the current context.
    """
    assert len(a) == len(b)
    sum: fp.Real = 0
    with fp.REAL:
        for ai, bi in zip(a, b):
            sum += ai * bi
    return fp.round(sum)
