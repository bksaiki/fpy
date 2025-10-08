"""
Examples: unit tests

Intended for testing basic language features.
"""

import fpy2 as fp

@fp.fpy
def test_bool1():
    return True

@fp.fpy
def test_bool2():
    return False

@fp.fpy
def test_integer1():
    return 0

@fp.fpy
def test_integer2():
    return 1

@fp.fpy
def test_integer3():
    return fp.round(1)

@fp.fpy
def test_decnum1():
    return 0.0

@fp.fpy
def test_decnum2():
    return 1.5

@fp.fpy
def test_decnum3():
    return fp.round(2)

@fp.fpy
def test_hexnum1():
    return fp.hexfloat('0x1.921fb54442d18p+1')

@fp.fpy
def test_hexnum2():
    return fp.round(fp.hexfloat('0x1.921fb54442d18p+1'))

@fp.fpy
def test_rational1():
    return fp.rational(1, 3)

@fp.fpy
def test_rational2():
    return fp.round(fp.rational(1, 3))

@fp.fpy
def test_digits1():
    return fp.digits(0, 0, 2)

@fp.fpy
def test_digits2():
    return fp.digits(1, 0, 2)

@fp.fpy
def test_digits3():
    return fp.digits(-1, 0, 2)

@fp.fpy
def test_digits4():
    return fp.digits(3, -1, 2)

@fp.fpy
def test_digits5():
    return fp.round(fp.digits(3, -1, 2))

@fp.fpy
def test_let1():
    a = True
    return a

@fp.fpy
def test_let2():
    a = 1.0
    return a

@fp.fpy
def test_let3():
    a = 1.0
    b = 1.0
    return a + b

@fp.fpy
def test_augassign1():
    x = 1.0
    x += 1.0
    return x

@fp.fpy
def test_augassign2():
    x = 1.0
    x -= 1.0
    return x

@fp.fpy
def test_augassign3():
    x = 1.0
    x *= 2.0
    return x

@fp.fpy
def test_augassign4():
    x = 1.0
    x /= 2.0
    return x

@fp.fpy
def test_ife1():
    return False if True else False

@fp.fpy
def test_ife2():
  return 1.0 if 1.0 > 0.0 else 0.0

@fp.fpy
def test_ife3():
  return 1.0 if 0.0 < 1.0 < 2.0 else 0.0

@fp.fpy
def test_ife4():
  x = 1.0
  y = 2.0
  z = 3.0
  t = 4.0
  return 1.0 if (x + 1.0) < (y + 2.0) < (z + 3.0) < (t + 4.0) else 0.0

@fp.fpy
def test_ife5():
  x = 1.0
  y = 2.0
  z = 3.0
  t = 4.0
  return 1.0 if (x + 1.0) < (y + 2.0) <= (z + 3.0) < (t + 4.0) else 0.0

@fp.fpy
def test_context_expr1():
    ctx = fp.IEEEContext(8, 32, fp.RM.RNE)
    return 1

@fp.fpy
def test_context_expr2():
    x = 8
    ctx = fp.MPFixedContext(x, fp.RM.RTZ)
    return 1

@fp.fpy
def test_tuple1():
    return False, True

@fp.fpy
def test_tuple2():
    return 1.0, 2.0, 3.0

@fp.fpy
def test_tuple3():
    return False, 1.0

@fp.fpy
def test_tuple4():
    x, y = 1.0, 2.0
    return x + y

@fp.fpy
def test_tuple5():
    x, y = (1.0, 2.0), (3.0, 4.0)
    x0, x1 = x
    y0, y1 = y
    return x0 * y0 + x1 * y1

@fp.fpy
def test_tuple6():
    (x, y), (z, _) = (1.0, 2.0), (3.0, 4.0)
    return x + y + z

@fp.fpy
def test_list1():
    return []

@fp.fpy
def test_list2():
    return [1]

@fp.fpy
def test_list3():
    return [1.0, 2.0, 3.0]

@fp.fpy
def test_list4():
    return [False, True]

@fp.fpy
def test_list_len1():
    x = []
    return len(x)

@fp.fpy
def test_list_len2():
    x = [1.0, 2.0, 3.0]
    return len(x)

@fp.fpy
def test_list_len3():
    x = [[1.0, 2.0, 3.0]]
    return len(x)

@fp.fpy
def test_list_len4():
    x = [False, True]
    return len(x)

@fp.fpy
def test_list_dim1():
    x = []
    return fp.dim(x)

@fp.fpy
def test_list_dim2():
    x = [1.0, 2.0, 3.0]
    return fp.dim(x)

@fp.fpy
def test_list_dim3():
    x = [[1.0, 2.0], [3.0, 4.0]]
    return fp.dim(x)

@fp.fpy
def test_list_dim4():
    x = [False, True]
    return fp.dim(x)

@fp.fpy
def test_list_size1():
    x = []
    return fp.size(x, 0)

@fp.fpy
def test_list_size2():
    x = [1.0, 2.0, 3.0]
    return fp.size(x, 0)

@fp.fpy
def test_list_size3():
    x = [[1.0, 2.0], [3.0, 4.0]]
    return fp.size(x, 1)

@fp.fpy
def test_range1():
    return range(5)

@fp.fpy
def test_range2():
    return range(1, 5)

@fp.fpy
def test_range3():
    return range(1, 5, 2)

@fp.fpy
def test_list_size4():
    x = [False, True]
    return fp.size(x, 0)

@fp.fpy
def test_enumerate1():
    xs = []
    return enumerate(xs)

@fp.fpy
def test_enumerate2():
    xs = [1.0, 2.0, 3.0]
    return enumerate(xs)

@fp.fpy
def test_enumerate3():
    xs = [False, True]
    return enumerate(xs)

# @fp.fpy(name='Test zip (1/4)')
# def test_list_zip1():
#     return zip()

@fp.fpy
def test_list_zip2():
    xs = [1.0, 2.0, 3.0]
    return zip(xs)

@fp.fpy
def test_list_zip3():
    xs = [1.0, 2.0, 3.0]
    ys = [4.0, 5.0, 6.0]
    return zip(xs, ys)

@fp.fpy
def test_list_zip4():
    xs = [1.0, 2.0, 3.0]
    ys = [4.0, 5.0, 6.0]
    zs = [7.0, 8.0, 9.0]
    return zip(xs, ys, zs)

@fp.fpy
def test_list_zip5():
    xs = [False, True]
    ys = [True, False]
    return zip(xs, ys)

@fp.fpy
def test_list_sum(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
    t1 = sum([x])
    t2 = sum([x, y])
    t3 = sum([x, y, z])
    return t1 + t2 + t3

@fp.fpy
def test_min(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
    t2 = min(x, y)
    t3 = min(x, y, z)
    return t2 + t3

@fp.fpy
def test_max(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
    t2 = max(x, y)
    t3 = max(x, y, z)
    return t2 + t3

@fp.fpy
def test_list_comp1():
    return [x + 1 for x in range(5)]

@fp.fpy
def test_list_comp2():
    return [x + y for x in range(4) for y in range(5)]

@fp.fpy
def test_list_comp3():
    return [x + y for x, y in zip([0, 1, 2], [3, 4, 5])]

@fp.fpy
def test_list_comp4():
    return [x and y for x, y in zip([True, False], [False, True])]

@fp.fpy
def test_list_ref1():
    x = [1.0, 2.0, 3.0]
    return x[0]

@fp.fpy
def test_list_ref2():
    x = [[1.0, 2.0, 3.0]]
    return x[0][0]

@fp.fpy
def test_list_ref3():
    x = [1.0, 2.0, 3.0]
    return x[:]

@fp.fpy
def test_list_ref4():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    return x[1:]

@fp.fpy
def test_list_ref5():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    return x[:3]

@fp.fpy
def test_list_ref6():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    return x[1:3]

@fp.fpy
def test_list_set1():
    x = [1.0, 2.0, 3.0]
    x[0] = 0.0
    return x

@fp.fpy
def test_list_set2():
    x = [[1.0, 2.0, 3.0]]
    x[0][0] = 0.0
    return x

@fp.fpy
def test_list_set3():
    x = [[[1.0, 2.0, 3.0]]]
    x[0][0][0] = 0.0
    return x

@fp.fpy
def test_if1():
    t = False
    if True:
        t = True
    return t

@fp.fpy
def test_if2():
    t = 0
    if 0 < 1:
        t = 1
    return t

@fp.fpy
def test_if3():
    t = 0
    a = 1
    if 0 < 1:
        t = 1
    return t + a

@fp.fpy
def test_if4():
    if 0 < 1:
        t = 1
    else:
        t = 0
    return t

@fp.fpy
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

@fp.fpy
def test_if6():
    if 0 < 1:
        t = 0
    elif 1 < 2:
        t = 1
    else:
        t = 2
    return t

@fp.fpy
def test_if7():
    t = 0
    a = 1
    if t < 0:
        t = 1
    else:
        a = 0
    return t + a

@fp.fpy
def test_while1():
    t = False
    while False:
        t = True
    return t

@fp.fpy
def test_while2():
    while False:
        x = 1
    return 0

@fp.fpy
def test_while3():
    x = 0
    while x < 1:
        x = 1
    return x

@fp.fpy
def test_while4():
    x = 0
    t = 1
    while x < 1:
        x = 1
    return x + t

@fp.fpy
def test_while5():
    x = 0
    y = 0
    while x < 5:
        x += 1
        y += x
    return x, y

@fp.fpy
def test_while6():
    x = 0
    y = 0
    while x < 5:
        while y < 25:
            y += 1
            x += y
    return x, y

@fp.fpy
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

@fp.fpy
def test_for1():
    t = False
    for s in [False, True]:
        t = s
    return t

@fp.fpy
def test_for2():
    j = 0
    for i in range(5):
        j += i
    return j

@fp.fpy
def test_for3():
    accum = 0
    for i in range(5):
        for j in range(5):
            accum += i * j
    return accum

@fp.fpy
def test_for4():
    x = 0
    y = 0
    for i in range(5):
        x += i
        y += 2 * i
    return x, y

@fp.fpy
def test_for5():
    xs = [1, 2, 3]
    ys = [3, 5, 7]
    sum = 0.0
    for x, y in zip(xs, ys):
        sum += x * y
    return sum

@fp.fpy
def test_context1():
    with fp.IEEEContext(8, 32, fp.RM.RNE):
        return fp.round(0)

@fp.fpy
def test_context2():
    x = fp.round(1)
    with fp.IEEEContext(8, 32, fp.RM.RNE):
        return x + fp.round(1)

@fp.fpy
def test_context3(x: fp.Real):
    with fp.INTEGER:
        return x + fp.round(1)

@fp.fpy
def test_context4():
    x = 3.1415
    with fp.MPFixedContext(-2, rm=fp.RM.RNE):
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
        a = fp.const_pi()
    return a, ctx

@fp.fpy
def test_context8():
    with fp.MPFixedContext(-4) as ctx:
        a = fp.const_pi()
        with fp.IEEEContext(5, 12):
            b = fp.const_pi()
            with ctx:
                return a / b

@fp.fpy
def test_assert1():
    assert True
    return False

@fp.fpy
def test_assert2():
    assert 0 == 0, "assert message"
    return 0

@fp.fpy
def test_assert3():
    assert 0 == 0, 1 + 1
    return 0

@fp.fpy
def test_pass1():
    pass
    return True

def test_meta(n):
    @fp.fpy
    def bar(x):
       return x + n
    return bar

test_meta0 = test_meta(0)
test_meta1 = test_meta(1)
