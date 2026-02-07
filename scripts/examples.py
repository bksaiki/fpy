import fpy2 as fp

@fp.fpy(ctx=fp.REAL)
def pre_round_mul(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.MX_E5M2:
        xq = fp.round(x)
        yq = fp.round(y)
    with fp.FP32:
        t = xq * yq
    return t

@fp.fpy(ctx=fp.REAL)
def dot_prod_1(xs: list[fp.Real], ys: list[fp.Real], c: fp.Real) -> fp.Real:
    QUANT_CTX = fp.MX_E5M2
    MUL_CTX = fp.IEEEContext(6, 12)
    ACCUM_CTX = fp.IEEEContext(8, 24)
    FINAL_CTX = fp.FP32

    # initialize accumulator
    with ACCUM_CTX:
        z = fp.round(0)

    # perform the dot product
    for x, y in zip(xs, ys):
        # quantize inputs
        with QUANT_CTX:
            xq = fp.round(x)
            yq = fp.round(y)

        # multiply
        with MUL_CTX:
            t = xq * yq

        # accumulate
        with ACCUM_CTX:
            z += t

    # add constant
    with FINAL_CTX:
        return z + c

@fp.fpy(ctx=fp.REAL)
def foo(
    a_scale: fp.Real,
    b_scale: fp.Real,
    a_elts: list[fp.Real],
    b_elts: list[fp.Real]
):
    scale = a_scale * b_scale
    with fp.FP32:
        dot_prod = fp.round(0)
        for a_elt, b_elt in zip(a_elts, b_elts):
            with fp.REAL:
                prod = a_elt * b_elt
            dot_prod += prod

    with fp.FP32:
        return dot_prod * scale

@fp.fpy(ctx=fp.REAL)
def dot_prod_blocked(xs: list[fp.Real], ys: list[fp.Real], c: fp.Real) -> fp.Real:
    """
    Dot product with blocking with size K.

    Args:
        xs: list of Real values
        ys: list of Real values
        c: Real constant to add to the final result
    """
    QUANT_CTX = fp.MX_E5M2
    MUL_CTX = fp.IEEEContext(5, 13)
    ACCUM_CTX = fp.IEEEContext(8, 24)
    FINAL_CTX = fp.FP32
    K = 8

    # initialize accumulator
    with FINAL_CTX:
        acc = fp.round(c)

    # perform the dot product
    n = len(xs)
    for start in range(0, n, K):
        # end index for the block
        end = start + K

        # extract block of size K
        x_block = fp.empty(K)
        y_block = fp.empty(K)
        for i in range(start, min(end, n)):
            with QUANT_CTX:
                x = fp.round(xs[i])
                y = fp.round(ys[i])
            x_block[i - start] = x
            y_block[i - start] = y
        for i in range(min(end, n), end):
            with QUANT_CTX:
                x = fp.round(0)
                y = fp.round(0)
            x_block[i - start] = x
            y_block[i - start] = y

        # initialize internal accumulator
        with ACCUM_CTX:
            z = fp.round(0)

        # process dot product
        for x, y in zip(x_block, y_block):
            # multiply
            with MUL_CTX:
                t = x * y

            # accumulate
            with ACCUM_CTX:
                z += t

        # add to final accumulator
        with FINAL_CTX:
            acc += z

    return acc

@fp.fpy(ctx=fp.REAL)
def dot_prod_arm(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    """
    Dot product with blocking with size K=4.
    From "Fused FP8 4-Way Dot Product With Scaling and FP32 Accumulation"

    Args:
        xs: list of Real values
        ys: list of Real values
    """
    QUANT_X_CTX = fp.MX_E4M3
    QUANT_Y_CTX = fp.MX_E4M3
    FINAL_CTX = fp.FP32
    K = 4

    assert len(xs) == len(ys), "Input lists must have the same length"

    # initialize accumulator
    with FINAL_CTX:
        acc = fp.round(0)

    # perform the dot product over blocks of size K
    n = len(xs)
    for start in range(0, n, K):
        # clamped end index for the block
        end = min(start + K, n)

        # extract block of size K, padding with zeros if necessary
        x_block = fp.empty(K)
        y_block = fp.empty(K)
        for i in range(start, end):
            with QUANT_X_CTX:
                x = fp.round(xs[i])
            with QUANT_Y_CTX:
                y = fp.round(ys[i])
            x_block[i-start] = x
            y_block[i-start] = y
        for i in range(end, start + K):
            with QUANT_X_CTX:
                x = fp.round(0)
            with QUANT_Y_CTX:
                y = fp.round(0)
            x_block[i-start] = x
            y_block[i-start] = y

        # process dot product
        z = sum([x * y for x, y in zip(x_block, y_block)])

        # add to final accumulator
        with fp.FP32:
            acc += z

    return acc

@fp.fpy(ctx=fp.REAL)
def mx_dot_prod(
    a_scale: fp.Real,
    b_scale: fp.Real,
    a_elts: list[fp.Real],
    b_elts: list[fp.Real]
):
    scale = a_scale * b_scale
    with fp.FP32:
        dot_prod = fp.round(0)
        for a_elt, b_elt in zip(a_elts, b_elts):
            with fp.REAL:
                prod = a_elt * b_elt
            dot_prod += prod

    with fp.FP32:
        return dot_prod * scale
