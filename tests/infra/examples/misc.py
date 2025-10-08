"""
Examples: miscellaneous tests

Intended for testing basic language features.
"""

import fpy2 as fp

@fp.fpy
def fma_ctx(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
    with fp.RealContext():
        prod = x * y
    return prod + z

@fp.fpy
def dpN(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    assert len(xs) == len(ys)
    sum = 0.0
    with fp.REAL:
        for x, y in zip(xs, ys):
            sum += x * y
    return fp.round(sum)

@fp.fpy
def nmse3_1(x: fp.Real):
    return fp.sqrt(x + fp.R(1)) - fp.sqrt(x)

# TODO: precondition
@fp.fpy(meta={
    'name': 'Daisy example instantaneousCurrent',
    'cite': 'daisy-2018'
})
def instCurrent(
    t: fp.Real,
    resistance: fp.Real,
    frequency: fp.Real,
    inductance: fp.Real,
    maxVoltage: fp.Real
):
    pi = fp.round(3.14159265359)
    impedance_re = resistance
    impedance_im = fp.R(2) * pi * frequency * inductance
    denom = impedance_re ** 2+ impedance_im ** 2
    current_re = (maxVoltage - impedance_re) / denom
    current_im = (maxVoltage - impedance_im) / denom
    maxCurrent = fp.sqrt(current_re ** 2 + current_im ** 2)
    theta = fp.atan(current_im / current_re)
    return maxCurrent * fp.cos(fp.R(2) * pi * frequency * t + theta)

@fp.fpy(meta={
    'name': 'azimuth',
    'cite': ['solovyev-2015']
})
def azimuth(lat1: fp.Real, lat2: fp.Real, lon1: fp.Real, lon2: fp.Real):
    dLon = lon2 - lon1
    s_lat1 = fp.sin(lat1)
    c_lat1 = fp.cos(lat1)
    s_lat2 = fp.sin(lat2)
    c_lat2 = fp.cos(lat2)
    s_dLon = fp.sin(dLon)
    c_dLon = fp.cos(dLon)
    return fp.atan((c_lat2 * s_dLon) / ((c_lat1 * s_lat2) - (s_lat1 * c_lat2 * c_dLon)))


@fp.fpy(meta={
    'name': 'Whetstone Loop 1',
    'cite': ['Curnow-and-Wichmann-1976'],
})
def whetsone1(n: int):
    t = fp.R(0.499975)
    x1 = fp.R(1.0)
    x2 = fp.R(-1.0)
    x3 = fp.R(-1.0)
    x4 = fp.R(-1.0)
    for _ in range(n):
        x1 = (x1 + x2 + x3 - x4) * t
        x2 = (x1 + x2 - x3 - x4) * t
        x3 = (x1 - x2 + x3 + x4) * t
        x4 = (-x1 + x2 + x3 + x4) * t
    return x1, x2, x3, x4

@fp.fpy
def example_sum(n: fp.Real):
    x = fp.round(0)
    for i in range(n):
        x += fp.round(i)
    return x

@fp.fpy
def example_set(y: fp.Real):
    x = [fp.round(0), fp.round(1)]
    x[0] = y
    return x

@fp.fpy
def example_static_context1():
    ES = 2
    NB = 8
    with fp.IEEEContext(ES, NB):
        return fp.round(1)

@fp.fpy
def example_static_context2():
    ES = 2
    NB = 8
    with fp.IEEEContext(ES + 2, NB + 2):
        return fp.round(1)

@fp.fpy
def example_fold_op1():
    with fp.FP64:
        t = fp.const_pi()
        return t

@fp.fpy
def example_fold_op2():
    with fp.FP64:
        t = fp.sin(1)
        return t

@fp.fpy
def example_fold_op3():
    with fp.FP64:
        t = 1 + 2
        return t

@fp.fpy
def example_fold_op4():
    with fp.FP64:
        t = fp.fma(3, 2, 1)
        return t

@fp.fpy(ctx=fp.REAL)
def _select_ctx(x: fp.Real):
    e = fp.libraries.core.logb(x)
    n = e - 1
    return fp.MPFixedContext(n)

@fp.fpy(ctx=fp.REAL)
def keep_p_1(x: fp.Real):
    with _select_ctx(x):
        return fp.round(x)
