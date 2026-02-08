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
def extract_block(xs: list[fp.Real], n: int, start: int, K: int) -> list[fp.Real]:
    # end index for a full block
    end = start + K

    # extract block of size K and pad with zeros if needed
    block = fp.empty(K)
    for i in range(start, min(end, n)):
        block[i - start] = xs[i]
    for i in range(min(end, n), end):
        with fp.declcontext(block[0]):
            x = fp.round(0)
        block[i - start] = x

    return block

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
        # extract blocks of size K, padding with zeros if necessary
        xblock = extract_block(xs, n, start, K)
        yblock = extract_block(ys, n, start, K)

        # quantize the elements of the blocks
        with QUANT_CTX:
            xblock_q = [fp.round(x) for x in xblock]
            yblock_q = [fp.round(y) for y in yblock]

        # initialize internal accumulator
        with ACCUM_CTX:
            z = fp.round(0)

        # process dot product
        for x, y in zip(xblock_q, yblock_q):
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
        # extract blocks of size K, padding with zeros if necessary
        xblock = extract_block(xs, n, start, K)
        yblock = extract_block(ys, n, start, K)

        # quantize the elements of the blocks
        with QUANT_X_CTX:
            xblock_q = [fp.round(x) for x in xblock]
        with QUANT_Y_CTX:
            yblock_q = [fp.round(y) for y in yblock]

        # process dot product
        z = sum([x * y for x, y in zip(xblock_q, yblock_q)])

        # add to final accumulator
        with fp.FP32:
            acc += z

    return acc

@fp.fpy(ctx=fp.REAL)
def mx_block_round(xs: list[fp.Real]):
    SCALE_CTX = fp.MX_E8M0
    ELT_CTX = fp.MX_E4M3

    # compute the maximum exponent
    max_e: fp.Real = 0
    max_e_valid = False
    for x in xs:
        if not fp.isnan(x) and not fp.isinf(x) and x != 0:
            if max_e_valid:
                max_e = max(max_e, fp.logb(x))
            else:
                max_e = fp.logb(x)
                max_e_valid = True

    if max_e_valid:
        # we found at least one valid exponent
        scale_e = max_e - 8
        with SCALE_CTX:
            scale = fp.libraries.core.ldexp(1, scale_e)
        with ELT_CTX:
            elts = [elt / scale for elt in xs]
    else:
        # all inputs were invalid, return dummy values
        with SCALE_CTX:
            scale = fp.round(1)
        with ELT_CTX:
            elts = [fp.round(x) for x in xs]

    return scale, elts


@fp.fpy(ctx=fp.REAL)
def mx_dot_prod(xs: list[fp.Real], ys: list[fp.Real], c: fp.Real) -> fp.Real:
    """
    Dot product with blocking with size K.

    Args:
        xs: list of Real values
        ys: list of Real values
        c: Real constant to add to the final result
    """
    FINAL_CTX = fp.FP32
    K = 32

    # initialize accumulator
    with FINAL_CTX:
        acc = fp.round(c)

    # perform the dot product
    n = len(xs)
    for start in range(0, n, K):
        # extract block of size K
        x_block = extract_block(xs, n, start, K)
        y_block = extract_block(ys, n, start, K)

        # apply quantization to the blocks
        xscale, x_elts = mx_block_round(x_block)
        yscale, y_elts = mx_block_round(y_block)

        # process dot product
        z = sum([x * y for x, y in zip(x_elts, y_elts)])

        # apply the scales
        z *= xscale * yscale

        # add to final accumulator
        with FINAL_CTX:
            acc += z

    return acc


@fp.fpy(ctx=fp.REAL)
def mx_matmul(
    A: list[list[fp.Real]],  # matrix: MxK
    Bt: list[list[fp.Real]], # matrix transpose: NxK
) -> list[list[fp.Real]]:
    """Matrix multiplication using MX-style dot products."""

    # extract dimensions
    Am = len(A)
    Ak = len(A[0])
    Bn = len(Bt)
    Bk = len(Bt[0])
    assert Ak == Bk, "Inner dimensions of A and Bt must match"

    C: list[list[fp.Real]] = fp.empty(Am, Bn)
    for i in range(Am):
        for j in range(Bn):
            C[i][j] = mx_dot_prod(A[i], Bt[j], 0.0)

    return C
