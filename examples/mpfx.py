import fpy2 as fp

@fp.fpy(ctx=fp.REAL)
def dot_prod(xs: list[fp.Real], ys: list[fp.Real], c: fp.Real) -> fp.Real:
    with fp.REAL:
        return sum([x * y for x, y in zip(xs, ys)]) + c

@fp.fpy(ctx=fp.REAL)
def dot_prod_mixed(xs: list[fp.Real], ys: list[fp.Real], c: fp.Real) -> fp.Real:
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
def dot_prod_block(xs: list[fp.Real], ys: list[fp.Real], c: fp.Real) -> fp.Real:
    QUANT_CTX = fp.MX_E5M2
    MUL_CTX = fp.IEEEContext(5, 13)
    ACCUM_CTX = fp.IEEEContext(8, 24)
    FINAL_CTX = fp.FP32
    K = 16

    # initialize accumulator
    with FINAL_CTX:
        acc = fp.round(c)

    # perform the dot product
    n = len(xs)
    for start in range(0, n, K):
        # extract block of size K
        end = start + K
        if end <= n:
            x_block = xs[start:end]
            y_block = ys[start:end]
        else:
            x_block = fp.empty(K)
            y_block = fp.empty(K)
            for j in range(start, n):
                x_block[j - start] = xs[j]
                y_block[j - start] = ys[j]
            for j in range(n, end):
                x_block[j - start] = 0
                y_block[j - start] = 0

        # initialize internal accumulator
        with ACCUM_CTX:
            z = fp.round(0)

        # process dot product
        for x, y in zip(x_block, y_block):
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

        # add to final accumulator
        with FINAL_CTX:
            acc += z

    return acc
