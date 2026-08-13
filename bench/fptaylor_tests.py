import fpy2 as fp


@fp.fpy(
    meta={
        "name": "intro-example",
        "cite": ["solovyev-et-al-2015"],
        "pre": lambda t: fp.round(0) <= t <= fp.round(999),
    }
)
def intro_example(t: fp.Real) -> fp.Real:
    return t / (t + fp.round(1))


@fp.fpy(
    ctx=fp.FP64,
    meta={
        "name": "sec4-example",
        "cite": ["solovyev-et-al-2015"],
        "pre": lambda x, y: fp.round(1.001) <= x <= fp.round(2)
        and fp.round(1.001) <= y <= fp.round(2),
    },
)
def sec4_example(x: fp.Real, y: fp.Real) -> fp.Real:
    t = x * y
    return (t - fp.round(1)) / (t * t - fp.round(1))


@fp.fpy(
    ctx=fp.FP32,
    meta={
        "name": "test01_sum3",
        "pre": lambda x0, x1, x2: fp.round(1) < x0 < fp.round(2)
        and fp.round(1) < x1 < fp.round(2)
        and fp.round(1) < x2 < fp.round(2),
    },
)
def test01_sum3(x0: fp.Real, x1: fp.Real, x2: fp.Real) -> fp.Real:
    p0 = (x0 + x1) - x2
    p1 = (x1 + x2) - x0
    p2 = (x2 + x0) - x1
    return (p0 + p1) + p2


@fp.fpy(
    ctx=fp.FP64,
    meta={
        "name": "test02_sum8",
        "pre": lambda x0, x1, x2, x3, x4, x5, x6, x7: (
            fp.round(1) < x0 < fp.round(2)
            and fp.round(1) < x1 < fp.round(2)
            and fp.round(1) < x2 < fp.round(2)
            and fp.round(1) < x3 < fp.round(2)
            and fp.round(1) < x4 < fp.round(2)
            and fp.round(1) < x5 < fp.round(2)
            and fp.round(1) < x6 < fp.round(2)
            and fp.round(1) < x7 < fp.round(2)
        ),
    },
)
def test02_sum8(
    x0: fp.Real,
    x1: fp.Real,
    x2: fp.Real,
    x3: fp.Real,
    x4: fp.Real,
    x5: fp.Real,
    x6: fp.Real,
    x7: fp.Real,
) -> fp.Real:
    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7


@fp.fpy(
    ctx=fp.FP64,
    meta={
        "name": "test03_nonlin2",
        "pre": lambda x, y: fp.round(0) < x < fp.round(1)
        and fp.round(-1) < y < fp.round(-0.1),
    },
)
def test03_nonlin2(x: fp.Real, y: fp.Real) -> fp.Real:
    return (x + y) / (x - y)


@fp.fpy(
    ctx=fp.FP64,
    meta={
        "name": "test04_dqmom9",
        "pre": lambda m0, m1, m2, w0, w1, w2, a0, a1, a2: (
            fp.round(-1) < m0 < fp.round(1)
            and fp.round(-1) < m1 < fp.round(1)
            and fp.round(-1) < m2 < fp.round(1)
            and fp.round(0.00001) < w0 < fp.round(1)
            and fp.round(0.00001) < w1 < fp.round(1)
            and fp.round(0.00001) < w2 < fp.round(1)
            and fp.round(0.00001) < a0 < fp.round(1)
            and fp.round(0.00001) < a1 < fp.round(1)
            and fp.round(0.00001) < a2 < fp.round(1)
        ),
    },
)
def test04_dqmom9(
    m0: fp.Real,
    m1: fp.Real,
    m2: fp.Real,
    w0: fp.Real,
    w1: fp.Real,
    w2: fp.Real,
    a0: fp.Real,
    a1: fp.Real,
    a2: fp.Real,
) -> fp.Real:
    v2 = (w2 * (fp.round(0) - m2)) * (
        fp.round(-3) * (fp.round(1) * (a2 / w2) * (a2 / w2))
    )
    v1 = (w1 * (fp.round(0) - m1)) * (
        fp.round(-3) * (fp.round(1) * (a1 / w1) * (a1 / w1))
    )
    v0 = (w0 * (fp.round(0) - m0)) * (
        fp.round(-3) * (fp.round(1) * (a0 / w0) * (a0 / w0))
    )
    return fp.round(0.0) + (
        v0 * fp.round(1) + (v1 * fp.round(1) + (v2 * fp.round(1) + fp.round(0.0)))
    )


@fp.fpy(
    ctx=fp.FP64,
    meta={
        "name": "test05_nonlin1, r4",
        "pre": lambda x: fp.round(1.00001) < x < fp.round(2),
    },
)
def test05_nonlin1_r4(x: fp.Real) -> fp.Real:
    r1 = x - fp.round(1)
    r2 = x * x
    return r1 / (r2 - fp.round(1))


@fp.fpy(
    ctx=fp.FP64,
    meta={
        "name": "test05_nonlin1, test2",
        "pre": lambda x: fp.round(1.00001) < x < fp.round(2),
    },
)
def test05_nonlin1_test2(x: fp.Real) -> fp.Real:
    return fp.round(1) / (x + fp.round(1))


@fp.fpy(
    ctx=fp.FP32,
    meta={
        "name": "test06_sums4, sum1",
        "pre": lambda x0, x1, x2, x3: (
            fp.round(-1e-5) < x0 < fp.round(1.00001)
            and fp.round(0) < x1 < fp.round(1)
            and fp.round(0) < x2 < fp.round(1)
            and fp.round(0) < x3 < fp.round(1)
        ),
    },
)
def test06_sums4_sum1(x0: fp.Real, x1: fp.Real, x2: fp.Real, x3: fp.Real) -> fp.Real:
    return x0 + x1 + x2 + x3


@fp.fpy(
    ctx=fp.FP32,
    meta={
        "name": "test06_sums4, sum2",
        "pre": lambda x0, x1, x2, x3: (
            fp.round(-1e-5) < x0 < fp.round(1.00001)
            and fp.round(0) < x1 < fp.round(1)
            and fp.round(0) < x2 < fp.round(1)
            and fp.round(0) < x3 < fp.round(1)
        ),
    },
)
def test06_sums4_sum2(x0: fp.Real, x1: fp.Real, x2: fp.Real, x3: fp.Real) -> fp.Real:
    return (x0 + x1) + (x2 + x3)
