from fpy2 import *
import fpy2 as fp

### Simple tests

@fpy
def test_bool1():
    return True

@fpy
def test_bool2():
    return False

@fpy
def test_integer1():
    return 0

@fpy
def test_integer2():
    return 1

@fpy
def test_integer3():
    return fp.round(1)

@fpy
def test_decnum1():
    return 0.0

@fpy
def test_decnum2():
    return 1.5

@fpy
def test_decnum3():
    return fp.round(2)

@fpy
def test_hexnum1():
    return fp.hexfloat('0x1.921fb54442d18p+1')

@fpy
def test_hexnum2():
    return fp.round(fp.hexfloat('0x1.921fb54442d18p+1'))

@fpy
def test_rational1():
    return fp.rational(1, 3)

@fpy
def test_rational2():
    return fp.round(fp.rational(1, 3))

@fpy
def test_digits1():
    return digits(0, 0, 2)

@fpy
def test_digits2():
    return digits(1, 0, 2)

@fpy
def test_digits3():
    return digits(-1, 0, 2)

@fpy
def test_digits4():
    return digits(3, -1, 2)

@fpy
def test_digits5():
    return fp.round(digits(3, -1, 2))

@fpy
def test_let1():
    a = True
    return a

@fpy
def test_let2():
    a = 1.0
    return a

@fpy
def test_let3():
    a = 1.0
    b = 1.0
    return a + b

@fpy
def test_augassign1():
    x = 1.0
    x += 1.0
    return x

@fpy
def test_augassign2():
    x = 1.0
    x -= 1.0
    return x

@fpy
def test_augassign3():
    x = 1.0
    x *= 2.0
    return x

@fpy
def test_augassign4():
    x = 1.0
    x /= 2.0
    return x

@fpy
def test_ife1():
    return False if True else False

@fpy
def test_ife2():
  return 1.0 if 1.0 > 0.0 else 0.0

@fpy
def test_ife3():
  return 1.0 if 0.0 < 1.0 < 2.0 else 0.0

@fpy
def test_ife4():
  x = 1.0
  y = 2.0
  z = 3.0
  t = 4.0
  return 1.0 if (x + 1.0) < (y + 2.0) < (z + 3.0) < (t + 4.0) else 0.0

@fpy
def test_ife5():
  x = 1.0
  y = 2.0
  z = 3.0
  t = 4.0
  return 1.0 if (x + 1.0) < (y + 2.0) <= (z + 3.0) < (t + 4.0) else 0.0

@fpy
def test_context_expr1():
    ctx = IEEEContext(8, 32, RM.RNE)
    return 1

@fpy
def test_context_expr2():
    x = 8
    ctx = MPFixedContext(x, RM.RTZ)
    return 1

@fpy
def test_tuple1():
    return False, True

@fpy
def test_tuple2():
    return 1.0, 2.0, 3.0

@fpy
def test_tuple3():
    return False, 1.0

@fpy
def test_tuple4():
    x, y = 1.0, 2.0
    return x + y

@fpy
def test_tuple5():
    x, y = (1.0, 2.0), (3.0, 4.0)
    x0, x1 = x
    y0, y1 = y
    return x0 * y0 + x1 * y1

@fpy
def test_tuple6():
    (x, y), (z, _) = (1.0, 2.0), (3.0, 4.0)
    return x + y + z

@fpy
def test_list1():
    return []

@fpy
def test_list2():
    return [1]

@fpy
def test_list3():
    return [1.0, 2.0, 3.0]

@fpy
def test_list4():
    return [False, True]

@fpy
def test_list_len1():
    x = []
    return len(x)

@fpy
def test_list_len2():
    x = [1.0, 2.0, 3.0]
    return len(x)

@fpy
def test_list_len3():
    x = [[1.0, 2.0, 3.0]]
    return len(x)

@fpy
def test_list_len4():
    x = [False, True]
    return len(x)

@fpy
def test_list_dim1():
    x = []
    return dim(x)

@fpy
def test_list_dim2():
    x = [1.0, 2.0, 3.0]
    return dim(x)

@fpy
def test_list_dim3():
    x = [[1.0, 2.0], [3.0, 4.0]]
    return dim(x)

@fpy
def test_list_dim4():
    x = [False, True]
    return dim(x)

@fpy
def test_list_size1():
    x = []
    return size(x, 0)

@fpy
def test_list_size2():
    x = [1.0, 2.0, 3.0]
    return size(x, 0)

@fpy
def test_list_size3():
    x = [[1.0, 2.0], [3.0, 4.0]]
    return size(x, 1)

@fpy
def test_list_size4():
    x = [False, True]
    return size(x, 0)

@fpy
def test_enumerate1():
    xs = []
    return enumerate(xs)

@fpy
def test_enumerate2():
    xs = [1.0, 2.0, 3.0]
    return enumerate(xs)

@fpy
def test_enumerate3():
    xs = [False, True]
    return enumerate(xs)

# @fpy(name='Test zip (1/4)')
# def test_list_zip1():
#     return zip()

@fpy
def test_list_zip2():
    xs = [1.0, 2.0, 3.0]
    return zip(xs)

@fpy
def test_list_zip3():
    xs = [1.0, 2.0, 3.0]
    ys = [4.0, 5.0, 6.0]
    return zip(xs, ys)

@fpy
def test_list_zip4():
    xs = [1.0, 2.0, 3.0]
    ys = [4.0, 5.0, 6.0]
    zs = [7.0, 8.0, 9.0]
    return zip(xs, ys, zs)

@fpy
def test_list_zip5():
    xs = [False, True]
    ys = [True, False]
    return zip(xs, ys)

@fpy
def test_list_sum(x: fp.Real, y: fp.Real, z: fp.Real) -> Real:
    t1 = sum([x])
    t2 = sum([x, y])
    t3 = sum([x, y, z])
    return t1 + t2 + t3

@fpy
def test_min(x: fp.Real, y: fp.Real, z: fp.Real) -> Real:
    t2 = min(x, y)
    t3 = min(x, y, z)
    return t2 + t3

@fpy
def test_max(x: fp.Real, y: fp.Real, z: fp.Real) -> Real:
    t2 = max(x, y)
    t3 = max(x, y, z)
    return t2 + t3

@fpy
def test_list_comp1():
    return [x + 1 for x in range(5)]

@fpy
def test_list_comp2():
    return [x + y for x in range(4) for y in range(5)]

@fpy
def test_list_comp3():
    return [x + y for x, y in zip([0, 1, 2], [3, 4, 5])]

@fpy
def test_list_comp4():
    return [x and y for x, y in zip([True, False], [False, True])]

@fpy
def test_list_ref1():
    x = [1.0, 2.0, 3.0]
    return x[0]

@fpy
def test_list_ref2():
    x = [[1.0, 2.0, 3.0]]
    return x[0][0]

@fpy
def test_list_ref3():
    x = [1.0, 2.0, 3.0]
    return x[:]

@fpy
def test_list_ref4():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    return x[1:]

@fpy
def test_list_ref5():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    return x[:3]

@fpy
def test_list_ref6():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    return x[1:3]

@fpy
def test_list_set1():
    x = [1.0, 2.0, 3.0]
    x[0] = 0.0
    return x

@fpy
def test_list_set2():
    x = [[1.0, 2.0, 3.0]]
    x[0][0] = 0.0
    return x

@fpy
def test_list_set3():
    x = [[[1.0, 2.0, 3.0]]]
    x[0][0][0] = 0.0
    return x

@fpy
def test_if1():
    t = False
    if True:
        t = True
    return t

@fpy
def test_if2():
    t = 0
    if 0 < 1:
        t = 1
    return t

@fpy
def test_if3():
    t = 0
    a = 1
    if 0 < 1:
        t = 1
    return t + a

@fpy
def test_if4():
    if 0 < 1:
        t = 1
    else:
        t = 0
    return t

@fpy
def test_if5():
    if 0 < 1:
        if 1 < 2:
            t = 0
        else:
            t = 1
    else:
        if 2 > 1:
            t = 2
        else:
            t = 3
    return t

@fpy
def test_if6():
    if 0 < 1:
        t = 0
    elif 1 < 2:
        t = 1
    else:
        t = 2
    return t

@fpy
def test_if7():
    t = 0
    a = 1
    if t < 0:
        t = 1
    else:
        a = 0
    return t + a

@fpy
def test_while1():
    t = False
    while False:
        t = True
    return t

@fpy
def test_while2():
    while False:
        x = 1
    return 0

@fpy
def test_while3():
    x = 0
    while x < 1:
        x = 1
    return x

@fpy
def test_while4():
    x = 0
    t = 1
    while x < 1:
        x = 1
    return x + t

@fpy
def test_while5():
    x = 0
    y = 0
    while x < 5:
        x += 1
        y += x
    return x, y

@fpy
def test_while6():
    x = 0
    y = 0
    while x < 5:
        while y < 25:
            y += 1
            x += y
    return x, y

@fpy
def test_while7():
    a = 0
    b = 0
    while a <= 3:
        a = a + 1
        i = 0
        x = 0
        while i <= a:
            i = i + 1
            x = x + i
        b = x
    return b

@fpy
def test_for1():
    t = False
    for s in [False, True]:
        t = s
    return t

@fpy
def test_for2():
    j = 0
    for i in range(5):
        j += i
    return j

@fpy
def test_for3():
    accum = 0
    for i in range(5):
        for j in range(5):
            accum += i * j
    return accum

@fpy
def test_for4():
    x = 0
    y = 0
    for i in range(5):
        x += i
        y += 2 * i
    return x, y

@fpy
def test_for5():
    xs = [1, 2, 3]
    ys = [3, 5, 7]
    sum = 0.0
    for x, y in zip(xs, ys):
        sum += x * y
    return sum

@fp.fpy
def test_context1():
    with IEEEContext(8, 32, RM.RNE):
        return fp.round(0)

@fp.fpy
def test_context2():
    x = fp.round(1)
    with IEEEContext(8, 32, RM.RNE):
        return x + fp.round(1)

@fp.fpy
def test_context3(x: fp.Real):
    with fp.INTEGER:
        return x + fp.round(1)

@fp.fpy
def test_context4():
    x = 3.1415
    with MPFixedContext(-2, rm=fp.RM.RNE):
        return fp.round(x)

@fp.fpy
def test_context5(s: fp.Real): # s : real @ b
    t: fp.Real = fp.round(0) # t : real @ a
    if s < fp.round(0): # < : real @ b -> real @ a -> bool @ a
        t += fp.round(1)  # + : real @ a -> real @ a -> real @ a
    else:
        with fp.FP32:
            tmp = t + s # + : real @ a -> real @ b -> real @ FP32
        t = fp.round(tmp) # round : real @ FP32 -> real @ a
    return t

@fp.fpy
def test_context6():
    with fp.UINT64:
        z = fp.round(0) # z : real U64
        for i in range(fp.round(10)): # i : real R
            j = fp.round(i)
            z += j * j # + : real U64 -> real U64 -> real U64
        return z

@fp.fpy
def test_context7():
    with fp.MPFixedContext(-4) as ctx:
        a = const_pi()
    return a, ctx

@fp.fpy
def test_context8():
    with fp.MPFixedContext(-4) as ctx:
        a = const_pi()
        with fp.IEEEContext(5, 12):
            b = const_pi()
            with ctx:
                return a / b

@fpy
def test_assert1():
    assert True
    return False

@fpy
def test_assert2():
    assert 0 == 0, "assert message"
    return 0

@fpy
def test_assert3():
    assert 0 == 0, 1 + 1
    return 0

@fpy
def test_pass1():
    pass
    return True

### Examples

@fpy
def fma_ctx(x: Real, y: Real, z: Real) -> Real:
    with RealContext():
        prod = x * y
    return prod + z

@fpy
def dpN(xs: list[Real], ys: list[Real]) -> Real:
    assert len(xs) == len(ys)
    sum = 0.0
    with RealContext():
        for x, y in zip(xs, ys):
            sum += x * y
    return round(sum)

@fpy
def nmse3_1(x: Real):
    return sqrt(x + fp.R(1)) - sqrt(x)

# TODO: precondition
@fpy(meta={
    'name': 'Daisy example instantaneousCurrent',
    'cite': 'daisy-2018'
})
def instCurrent(
    t: Real,
    resistance: Real,
    frequency: Real,
    inductance: Real,
    maxVoltage: Real
):
    pi = fp.round(3.14159265359)
    impedance_re = resistance
    impedance_im = fp.R(2) * pi * frequency * inductance
    denom = impedance_re ** 2+ impedance_im ** 2
    current_re = (maxVoltage - impedance_re) / denom
    current_im = (maxVoltage - impedance_im) / denom
    maxCurrent = sqrt(current_re ** 2 + current_im ** 2)
    theta = atan(current_im / current_re)
    return maxCurrent * cos(fp.R(2) * pi * frequency * t + theta)

@fpy(meta={
    'name': 'azimuth',
    'cite': ['solovyev-2015']
})
def azimuth(lat1: Real, lat2: Real, lon1: Real, lon2: Real):
    dLon = lon2 - lon1
    s_lat1 = sin(lat1)
    c_lat1 = cos(lat1)
    s_lat2 = sin(lat2)
    c_lat2 = cos(lat2)
    s_dLon = sin(dLon)
    c_dLon = cos(dLon)
    return atan((c_lat2 * s_dLon) / ((c_lat1 * s_lat2) - (s_lat1 * c_lat2 * c_dLon)))

# TODO: vectors should be tensors
@fpy(meta={
    'name': 'Level-of-detail (LOD) algorithm, anisotropic case',
    'cite': ['DirectX 11.3 specification, Microsoft-2015']
})
def lod_anisotropic(
    dx_u: Real,
    dx_v: Real,
    dy_u: Real,
    dy_v: Real,
    max_aniso: Real,
):
    dx2 = dx_u ** 2 + dx_v ** 2
    dy2 = dy_u ** 2 + dy_v ** 2
    det = fabs(dx_u * dy_v - dx_v * dy_u)
    x_major = dx2 > dy2
    major2 = dx2 if x_major else dy2
    major = sqrt(major2)
    norm_major = fp.R(1.0) / major

    aniso_dir_u = (dx_u if x_major else dy_u) * norm_major
    aniso_dir_v = (dx_v if x_major else dy_v) * norm_major
    aniso_ratio = major2 / det

    # clamp anisotropy ratio and compute LOD
    if aniso_ratio > max_aniso:
        aniso_ratio = max_aniso
        minor = major / aniso_ratio
    else:
        minor = det / major

    # clamp LOD
    if minor < fp.R(1):
        aniso_ratio = max(fp.R(1), aniso_ratio * minor)

    lod = log2(minor)
    return lod, aniso_ratio, aniso_dir_u, aniso_dir_v

@fpy(meta={
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

@fp.fpy(ctx=fp.REAL)
def _select_ctx(x: fp.Real):
    e = fp.libraries.core.logb(x)
    n = e - 1
    return fp.MPFixedContext(n)

@fp.fpy(ctx=fp.REAL)
def keep_p_1(x: fp.Real):
    with _select_ctx(x):
        return fp.round(x)


tests: list[Function] = [
    # Tests
    test_bool1,
    test_bool2,
    test_integer1,
    test_integer2,
    test_integer3,
    test_decnum1,
    test_decnum2,
    test_decnum3,
    test_hexnum1,
    test_hexnum2,
    test_rational1,
    test_rational2,
    test_digits1,
    test_digits2,
    test_digits3,
    test_digits4,
    test_digits5,
    test_let1,
    test_let2,
    test_augassign1,
    test_augassign2,
    test_augassign3,
    test_augassign4,
    test_ife1,
    test_ife2,
    test_ife3,
    test_ife4,
    test_ife5,
    test_context_expr1,
    test_context_expr2,
    test_tuple1,
    test_tuple2,
    test_tuple3,
    test_tuple4,
    test_tuple5,
    test_tuple6,
    test_list1,
    test_list2,
    test_list3,
    test_list4,
    test_list_len1,
    test_list_len2,
    test_list_len3,
    test_list_len4,
    test_list_dim1,
    test_list_dim2,
    test_list_dim3,
    test_list_dim4,
    test_list_size1,
    test_list_size2,
    test_list_size3,
    test_list_size4,
    test_enumerate1,
    test_enumerate2,
    test_enumerate3,
    # test_list_zip1,
    test_list_zip2,
    test_list_zip3,
    test_list_zip4,
    test_list_zip5,
    test_list_sum,
    test_min,
    test_max,
    test_list_comp1,
    test_list_comp2,
    test_list_comp3,
    test_list_comp4,
    test_list_ref1,
    test_list_ref2,
    test_list_ref3,
    test_list_ref4,
    test_list_ref5,
    test_list_ref6,
    test_list_set1,
    test_list_set2,
    test_list_set3,
    test_if1,
    test_if2,
    test_if3,
    test_if4,
    test_if5,
    test_if6,
    test_if7,
    test_while1,
    test_while2,
    test_while3,
    test_while4,
    test_while5,
    test_while6,
    test_while7,
    test_for1,
    test_for2,
    test_for3,
    test_for4,
    test_for5,
    test_context1,
    test_context2,
    test_context3,
    test_context4,
    test_context5,
    test_context6,
    test_context7,
    test_context8,
    test_assert1,
    test_assert2,
    test_assert3,
    test_pass1,
]

# Examples
examples: list[Function] = [
    fma_ctx,
    # dpN,
    nmse3_1,
    instCurrent,
    azimuth,
    lod_anisotropic,
    whetsone1,
    keep_p_1
]
