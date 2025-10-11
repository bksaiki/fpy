"""
fused-dp/model.py: FPy model of fused N-term dot product

This model is based on "Optimized Fused Floating-Point Many-Term Dot-Product
Hardware for Machine Learning Accelerators" (Kaul et al., 2019).
"""

import fpy2 as fp

@fp.fpy
def dot_prod_impl(xs: list[fp.Real], ys: list[fp.Real], p: int):
    """
    Computes the dot product of two vectors.

    This model is based on
    "Optimized Fused Floating-Point Many-Term Dot-Product
    Hardware for Machine Learning Accelerators" (Kaul et al., 2019).

    Args:
        xs (list[fp.Real]): first vector of length N
        ys (list[fp.Real]): second vector of length N
        p (int): aligned mantissa size (1.P)

    Returns:
        fp.Real: The dot product of the two vectors.
    """

    assert len(xs) == len(ys)

    with fp.REAL:
        # find maximum exponent
        max_e = -fp.inf()
        for x, y in zip(xs, ys):
            xe = -fp.inf() if x == 0 else fp.libraries.core.logb(x)
            ye = -fp.inf() if y == 0 else fp.libraries.core.logb(y)
            e = xe + ye
            if fp.isfinite(e) and e > max_e:
                max_e = e

        # compute alignment context
        if fp.isfinite(max_e):
            # compute alignment point
            n = max_e - p

            # compute context
            ctx: fp.Context = fp.MPFixedContext(n, fp.RM.RTZ)
        else:
            # no product is finite, so alignment is not needed
            ctx = fp.REAL

    # multiplication, alignment, and accumulation
    with fp.REAL:
        acc = 0
        for x, y in zip(xs, ys):
            prod = x * y
            with ctx:
                aligned = fp.round(prod)
            acc += aligned

    return fp.round(acc)
