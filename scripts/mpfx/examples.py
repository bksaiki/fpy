import fpy2 as fp

@fp.fpy(ctx=fp.REAL)
def vec_add_fp8(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
    xN = len(xs)
    yN = len(ys)
    assert xN == yN, "Input lists must have the same length"

    res = fp.empty(xN)
    for i in range(xN):
        with fp.MX_E5M2:
            xq = fp.round(xs[i])
            yq = fp.round(ys[i])
        with fp.FP32:
            res[i] = xq + yq

    return res

@fp.fpy(ctx=fp.REAL)
def vec_mul_fp8(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
    xN = len(xs)
    yN = len(ys)
    assert xN == yN, "Input lists must have the same length"

    res = fp.empty(xN)
    for i in range(xN):
        with fp.MX_E5M2:
            xq = fp.round(xs[i])
            yq = fp.round(ys[i])
        with fp.FP32:
            res[i] = xq * yq

    return res

@fp.fpy(ctx=fp.REAL)
def dot_prod_1(xs: list[fp.Real], ys: list[fp.Real], c: fp.Real) -> fp.Real:
    QUANT_CTX = fp.MX_E5M2
    MUL_CTX = fp.IEEEContext(6, 12)
    ACCUM_CTX = fp.TF32
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
    MUL_CTX = fp.IEEEContext(5, 12)
    ACCUM_CTX = fp.TF32
    FINAL_CTX = fp.FP32
    K = 32

    # initialize accumulator
    with FINAL_CTX:
        acc = fp.round(c)

    # perform the dot product
    n = len(xs)
    for start in range(0, n, K):
        # extract blocks of size K, padding with zeros if necessary
        xblock = extract_block(xs, n, start, K)
        yblock = extract_block(ys, n, start, K)

        # initialize internal accumulator
        with ACCUM_CTX:
            z = fp.round(0)

        # process dot product
        for x, y in zip(xblock, yblock):
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

@fp.fpy
def vec_scale(A: tuple[fp.Real, fp.Real, fp.Real], scale: fp.Real):
    x, y, z = A
    return (x * scale, y * scale, z * scale)

@fp.fpy
def vec_add(
    A: tuple[fp.Real, fp.Real, fp.Real],
    B: tuple[fp.Real, fp.Real, fp.Real]
) -> tuple[fp.Real, fp.Real, fp.Real]:
    x1, y1, z1 = A
    x2, y2, z2 = B
    return (x1 + x2, y1 + y2, z1 + z2)

@fp.fpy(ctx=fp.REAL)
def lorenz_3d(xyz: tuple[fp.Real, fp.Real, fp.Real]):
    x, y, z = xyz
    with fp.FP16:
        sigma = fp.round(10)
        beta = fp.round(8) / fp.round(3)
        rho = fp.round(28)

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return dx, dy, dz

@fp.fpy(ctx=fp.REAL)
def rk4_lorenz_3d(xyz: tuple[fp.Real, fp.Real, fp.Real], dt: fp.Real):
    """
    4th-order Runge-Kutta method for lorenz_3d.

    ```
    k1 = f(xyz)
    k2 = f(xyz + 0.5 * dt * k1)
    k3 = f(xyz + 0.5 * dt * k2)
    k4 = f(xyz + dt * k3)
    new_xyz = xyz + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    ```
    """
    with fp.IEEEContext(5, 14):
        with fp.IEEEContext(5, 13):
            k1 = vec_scale(lorenz_3d(xyz), dt)
        with fp.IEEEContext(5, 10):
            k2 = vec_scale(lorenz_3d(vec_add(xyz, vec_scale(k1, fp.round(0.5)))), dt)
        with fp.IEEEContext(5, 12):
            k3 = vec_scale(lorenz_3d(vec_add(xyz, vec_scale(k2, fp.round(0.5)))), dt)
        with fp.IEEEContext(5, 9):
            k4 = vec_scale(lorenz_3d(vec_add(xyz, k3)), dt)

        return vec_add(
            xyz,
            vec_scale(
                vec_add(
                    k1,
                    vec_add(vec_scale(k2, fp.round(2)),
                    vec_add(vec_scale(k3, fp.round(2)),
                    k4))),
                dt / fp.round(6)))

@fp.fpy(ctx=fp.REAL)
def run_rk4_lorenz_3d(init: tuple[fp.Real, fp.Real, fp.Real], dt: fp.Real):
    N = 8192

    # quantize the initial state
    x, y, z = init
    with fp.IEEEContext(5, 14):
        xq = fp.round(x)
        yq = fp.round(y)
        zq = fp.round(z)
        xyz = (xq, yq, zq)

    for _ in range(N):
        xyz = rk4_lorenz_3d(xyz, dt)

    return xyz


@fp.fpy(ctx=fp.REAL)
def fastblur_mask_3x3(img: list[list[list[fp.Real]]], mask: list[list[fp.Real]]) -> list[list[list[fp.Real]]]:
    # rounding contexts
    CTX1 = fp.IEEEContext(5, 10)
    CTX2 = fp.IEEEContext(5, 12)
    CTX3 = fp.IEEEContext(5, 9)

    # dimensions of image
    W = len(img)
    H = len(img[0])
    C = len(img[0][0])

    # mask dimensions
    MH = len(mask)
    MW = len(mask[0])
    assert MH == 3 and MW == 3, "Mask must be 3x3"

    # bounds
    ymax = W - 1
    xmax = H - 1

    # output image
    out = fp.empty(W, H, C)

    # apply 3x3 mask to each pixel
    for y in range(W):
        for x in range(H):
            in_bounds = False
            xx = 0
            yy = 0

            with CTX1:
                mw = fp.round(0)

            w = fp.empty(C)
            for my in range(MH):
                for mx in range(MW):
                    # update input coordinates
                    yy = y + (my - 1)
                    xx = x + (mx - 1)
                    in_bounds = (0 <= yy < ymax) and (0 <= xx < xmax)

                    # update sum of mask weights
                    if in_bounds:
                        with CTX1:
                            mw += mask[my][mx]
                    else:
                        mw = mw

                    # update weights
                    if in_bounds:
                        for c in range(C):
                            with CTX3:
                                m = mask[my][mx] * img[yy][xx][c]
                            with CTX2:
                                w[c] = w[c] + m
                    else:
                        w = w

            with CTX1:
                for c in range(C):
                    out[y][x][c] = w[c] / mw

    return out


@fp.fpy(ctx=fp.REAL)
def fastblur_example(
    r: list[list[fp.Real]],
    g: list[list[fp.Real]],
    b: list[list[fp.Real]],
    a: list[list[fp.Real]]
):
    rW, rH = len(r), len(r[0])
    gW, gH = len(g), len(g[0])
    bW, bH = len(b), len(b[0])
    aW, aH = len(a), len(a[0])

    assert rW == gW == bW == aW and rH == gH == bH == aH, "Input channels must have same dimensions"

    with fp.FP16:
        mask = [
            [fp.round(fp.rational(1, 3)), fp.round(fp.rational(1, 2)), fp.round(fp.rational(1, 3))],
            [fp.round(fp.rational(1, 2)), fp.round(fp.rational(3, 2)), fp.round(fp.rational(2, 3))],
            [fp.round(fp.rational(1, 3)), fp.round(fp.rational(1, 2)), fp.round(fp.rational(1, 3))]
        ]

    # stack channels into image
    img = fp.empty(rW, rH, 4)
    for y in range(rW):
        for x in range(rH):
            img[y][x][0] = r[y][x]
            img[y][x][1] = g[y][x]
            img[y][x][2] = b[y][x]
            img[y][x][3] = a[y][x]

    return fastblur_mask_3x3(img, mask)
