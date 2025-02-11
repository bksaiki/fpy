from fpy2 import fpy
from fpy2.typing import *

### Simple tests

@fpy
def test_simple1():
    return 0

@fpy(
    name='Test annotation',
    spec='0.0'
)
def test_simple2():
    return 0

@fpy(
    name='Test decnum (1/1)',
    spec='0.0'
)
def test_decnum():
    return 0.0

@fpy(
    name='Test digits (1/4)',
    spec='0.0'
)
def test_digits1():
    return digits(0, 0, 2)

@fpy(
    name='Test digits (2/4)',
    spec='1.0'
)
def test_digits2():
    return digits(1, 0, 2)

@fpy(
    name='Test digits (3/4)',
    spec='-2.0'
)
def test_digits3():
    return digits(-1, 0, 2)

@fpy(
    name='Test digits (4/4)',
    spec='1.5'
)
def test_digits4():
    return digits(3, -1, 2)

@fpy(
    name='Test let (1/2)',
    spec='1.0'
)
def test_let1():
    a = 1.0
    return a

@fpy(
    name='Test let (2/2)',
    spec='2.0'
)
def test_let2():
    a = 1.0
    b = 1.0
    return a + b

@fpy(name='Test augassign (1/5)')
def test_augassign1():
    x = 1.0
    x += 1.0
    return x

@fpy(name='Test augassign (2/5)')
def test_augassign2():
    x = 1.0
    x -= 1.0
    return x

@fpy(name='Test augassign (3/5)')
def test_augassign3():
    x = 1.0
    x *= 2.0
    return x

@fpy(name='Test augassign (4/5)')
def test_augassign4():
    x = 1.0
    x /= 2.0
    return x

@fpy(name='Test augassign (5/5)')
def test_augassign5():
    x = 1.0
    x %= 2.0
    return x

@fpy(
  name='Test if expression (1/6)',
  spec='1.0'
)
def test_ife1():
  return 1.0 if 1.0 > 0.0 else 0.0

@fpy(name='Test if expression (2/6)')
def test_ife2():
  return 1.0 if 0.0 < 1.0 < 2.0 else 0.0

@fpy(name='Test if expression (3/6)')
def test_ife3():
  x = 1.0
  y = 2.0
  z = 3.0
  t = 4.0
  return 1.0 if (x + 1.0) < (y + 2.0) < (z + 3.0) < (t + 4.0) else 0.0

@fpy(name='Test if expression (4/6)')
def test_ife4():
  x = 1.0
  y = 2.0
  z = 3.0
  t = 4.0
  return 1.0 if (x + 1.0) < (y + 2.0) <= (z + 3.0) < (t + 4.0) else 0.0

@fpy
def test_tuple1():
    return (1.0, 2.0, 3.0)

@fpy
def test_tuple2():
    return 1.0, 2.0, 3.0

@fpy
def test_tuple3():
    x, y = 1.0, 2.0
    return x + y

@fpy
def test_tuple4():
    x, y = (1.0, 2.0), (3.0, 4.0)
    x0, x1 = x
    y0, y1 = y
    return x0 * y0 + x1 * y1

@fpy
def test_list1():
    return [1.0, 2.0, 3.0]

@fpy
def test_list2():
    x, y = [1.0, 2.0], [3.0, 4.0]
    x0, x1 = x
    y0, y1 = y
    return x0 * y0 + x1 * y1

@fpy
def test_list_comp1():
    return [x + 1 for x in range(5)]

@fpy
def test_list_comp2():
    return [x + y for x in range(4) for y in range(5)]

@fpy(name='Test list ref (1/3)')
def test_list_ref1():
    x = [1.0, 2.0, 3.0]
    return x[0]

@fpy(name='Test list ref (2/3)')
def test_list_ref2():
    x = [[1.0, 2.0, 3.0]]
    return x[0][0]

@fpy(name='Test list ref (3/3)')
def test_list_ref3():
    x = [[1.0, 2.0], [3.0, 4.0]]
    return x[0, 1]

@fpy(name='Test list set (1/3)')
def test_list_set1():
    x = [1.0, 2.0, 3.0]
    x[0] = 0.0
    return x

@fpy(name='Test list set (2/3)')
def test_list_set2():
    x = [[1.0, 2.0, 3.0]]
    x[0][0] = 0.0
    return x

@fpy(name='Test list set (3/3)')
def test_list_set3():
    x = [[[1.0, 2.0, 3.0]]]
    x[0][0][0] = 0.0
    return x

@fpy(name='Test if statement (1/6)')
def test_if1():
    t = 0
    if 0 < 1:
        t = 1
    return t

@fpy(name='Test if statement (2/6)')
def test_if2():
    t = 0
    a = 1
    if 0 < 1:
        t = 1
    return t + a

@fpy(name='Test if statement (3/6)')
def test_if3():
    if 0 < 1:
        t = 1
    else:
        t = 0
    return t

@fpy(name='Test if statement (4/6)')
def test_if4():
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

@fpy(name='Test if statement (5/6)')
def test_if5():
    if 0 < 1:
        t = 0
    elif 1 < 2:
        t = 1
    else:
        t = 2
    return t

@fpy(name='Test if statement (6/6)')
def test_if6():
    t = 0
    a = 1
    if t < 0:
        t = 1
    else:
        a = 0
    return t + a

@fpy
def test_while1():
    x = 0
    while x < 1:
        x = 1
    return x

@fpy
def test_while2():
    x = 0
    t = 1
    while x < 1:
        x = 1
    return x + t

@fpy
def test_while3():
    x = 0
    y = 0
    while x < 5:
        x += 1
        y += x
    return x, y

@fpy
def test_while4():
    x = 0
    y = 0
    while x < 5:
        while y < 25:
            y += 1
            x += y
    return x, y

@fpy
def test_while5():
    x = 0
    y = 0
    while x < 5:
        x += 1
        y += x
        while y < 25:
            y += 1
            x += y
    return x, y

@fpy
def test_while6():
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
    j = 0
    for i in range(5):
        j += i
    return j

@fpy
def test_for2():
    accum = 0
    for i in range(5):
        for j in range(5):
            accum += i * j
    return accum

@fpy
def test_for3():
    x = 0
    y = 0
    for i in range(5):
        x += i
        y += 2 * i
    return x, y

@fpy(name='Test context statement (1/3)')
def test_context1():
    with Context():
        return 0

@fpy(name='Test context statement (2/3)')
def test_context2():
    x = 1
    with Context(precision='binary32'):
        return x + 1

@fpy(name='Test context statement (3/3)')
def test_context3(x: Real, y: Real):
    with Context(precision='binary32'):
        with Context(round='toPositive'):
            t0 = x + y
        with Context(round='toNegative'):
            t1 = x - y
        return t0 - t1

@fpy(name='Test assertion (1/1)')
def test_assert():
    assert 0 == 0
    return 0

### Examples

@fpy(
    name='NMSE example 3.1',
    cite=['hamming-1987', 'herbie-2015'],
    fpbench_domain='textbook',
)
def nmse3_1(x: Real):
    return sqrt(x + 1) - sqrt(x)

# TODO: precondition
@fpy(
    name='Daisy example instantaneousCurrent',
    cite=['daisy-2018']
)
def instCurrent(
    t: Real,
    resistance: Real,
    frequency: Real,
    inductance: Real,
    maxVoltage: Real
):
    pi = 3.14159265359
    impedance_re = resistance
    impedance_im = 2 * pi * frequency * inductance
    denom = impedance_re ** 2 + impedance_im ** 2
    current_re = (maxVoltage - impedance_re) / denom
    current_im = (maxVoltage - impedance_im) / denom
    maxCurrent = sqrt(current_re ** 2 + current_im ** 2)
    theta = atan(current_im / current_re)
    return maxCurrent * cos(2 * pi * frequency * t + theta)

@fpy(
    name='azimuth',
    cite=['solovyev-2015']
)
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
@fpy(
    name='Level-of-detail (LOD) algorithm, anisotropic case',
    cite=['DirectX 11.3 specification, Microsoft-2015']
)
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
    norm_major = 1.0 / major

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
    if minor < 1.0:
        aniso_ratio = fmax(1.0, aniso_ratio * minor)

    lod = log2(minor)
    return lod, aniso_ratio, aniso_dir_u, aniso_dir_v

# @fpcore(name='Ray-sphere intersection')
# def ray_sphere(
#     origin: tuple[Real, Real, Real],
#     direction: tuple[Real, Real, Real],
#     center: tuple[Real, Real, Real],
#     radius: Real
# ):
#     pass

@fpy(
    name='Whetstone Loop 1',
    cite=['Curnow-and-Wichmann-1976'],
)
def whetsone1(n: int):
    t = 0.499975
    x1 = 1.0
    x2 = -1.0
    x3 = -1.0
    x4 = -1.0
    for _ in range(n):
        x1 = (x1 + x2 + x3 - x4) * t
        x2 = (x1 + x2 - x3 - x4) * t
        x3 = (x1 - x2 + x3 + x4) * t
        x4 = (-x1 + x2 + x3 + x4) * t
    return x1, x2, x3, x4


tests = [
    # Tests
    test_simple1,
    test_simple2,
    test_decnum,
    test_digits1,
    test_digits2,
    test_digits3,
    test_digits4,
    test_let1,
    test_let2,
    test_augassign1,
    test_augassign2,
    test_augassign3,
    test_augassign4,
    test_augassign5,
    test_ife1,
    test_ife2,
    test_ife3,
    test_ife4,
    test_tuple1,
    test_tuple2,
    test_tuple3,
    test_tuple4,
    test_list1,
    test_list2,
    test_list_comp1,
    test_list_comp2,
    test_list_ref1,
    test_list_ref2,
    test_list_ref3,
    test_list_set1,
    test_list_set2,
    test_list_set3,
    test_if1,
    test_if2,
    test_if3,
    test_if4,
    test_if5,
    test_if6,
    test_while1,
    test_while2,
    test_while3,
    test_while4,
    test_while5,
    test_for1,
    test_for2,
    test_for3,
    test_context1,
    test_context2,
    test_context3,
    test_assert,
]

# Examples
examples = [
    nmse3_1,
    instCurrent,
    azimuth,
    lod_anisotropic,
    whetsone1
]
