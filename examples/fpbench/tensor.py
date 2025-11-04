from fpy2 import *

@fpy(
)
def I_2x2():
    A = [[round(1), round(0)], [round(0), round(1)]]
    return A

@fpy(
)
def dim_2x3():
    A = [[round(1), round(0), round(1)], [round(0), round(1), round(1)]]
    return dim(A)

@fpy(
)
def size_2x3():
    A = [[round(1), round(0), round(1)], [round(0), round(1), round(1)]]
    rows = size(A, round(0))
    cols = size(A, round(1))
    return [rows, cols]

@fpy(
)
def alternating_sum(A):
    n = size(A, 0)
    t = n
    sign = round(1)
    total = round(0)
    for i in range(t):
        sign = -sign
        total = (total + (sign * A[i]))
    return total

@fpy(
)
def main():
    return alternating_sum([round(1), round(2), round(3), round(4)])

@fpy(
)
def dim4(A, B, C, D):
    bm = size(B, 0)
    bn = size(B, 1)
    cm = size(C, 0)
    cn = size(C, 1)
    dm = size(D, 0)
    dn = size(D, 1)
    D0 = [A, B, C, D]
    return dim(D0)

@fpy(
)
def main():
    A = [[round(1), round(0)], [round(0), round(1)]]
    B = [[True, False], [False, True]]
    return dim4(A, A, A, A)

@fpy(
)
def dim4(A, B, C, D):
    am = size(A, 0)
    an = size(A, 1)
    bm = size(B, 0)
    bn = size(B, 1)
    cm = size(C, 0)
    cn = size(C, 1)
    dm = size(D, 0)
    dn = size(D, 1)
    D0 = [A, B, C, D]
    return dim(D0)

@fpy(
)
def main():
    A = [[round(1), round(0), round(0)], [round(0), round(1), round(0)]]
    return dim4(A, A, A, A)

@fpy(
)
def expand(A):
    n = size(A, 0)
    t = n
    t0 = empty(t)
    for i in range(t):
        t0[i] = A
    return t0

@fpy(
)
def main():
    A = [round(1), round(2), round(3), round(4)]
    return expand(A)

@fpy(
)
def fibonacci(n):
    t = n
    f_n_2 = round(0)
    f_n_1 = round(0)
    f_n = round(0)
    t0 = empty(t)
    for i in range(t):
        f_n_2 = f_n_1
        f_n_1 = f_n
        if i == round(1):
            t1 = round(1)
        else:
            t1 = (f_n_1 + f_n_2)
        f_n = t1
        t0[i] = f_n
    return t0

@fpy(
    meta={
        'pre': lambda A, B: an == bm,
    }
)
def matmul(A, B):
    am = size(A, 0)
    an = size(A, 1)
    bm = size(B, 0)
    bn = size(B, 1)
    t = am
    t0 = bn
    t1 = [empty(t0) for _ in range(t)]
    for m in range(t):
        for n in range(t0):
            t2 = bm
            prod = round(0)
            for i in range(t2):
                prod3 = (prod + (A[m][i] * B[i][n]))
                prod = prod3
            t1[m][n] = prod
    return t1

@fpy(
)
def fib_Q():
    return [[round(1), round(1)], [round(1), round(0)]]

@fpy(
)
def fib_iterative(n):
    t = n
    A = fib_Q()
    for i in range(t):
        A0 = matmul(A, fib_Q())
        A = A0
    return A[round(1)][round(1)]

@fpy(
)
def f18(n):
    with MPFixedContext(nmin=-1, rm=RoundingMode.RNE, num_randbits=0, enable_nan=False, enable_inf=False, nan_value=None, inf_value=None):
        t = round(1)
    return t

@fpy(
    meta={
        'pre': lambda A, B: am == bm and an == bn,
    }
)
def mat_add(A, B):
    am = size(A, 0)
    an = size(A, 1)
    bm = size(B, 0)
    bn = size(B, 1)
    t = am
    t0 = an
    t1 = [empty(t0) for _ in range(t)]
    for row in range(t):
        for col in range(t0):
            t1[row][col] = (A[row][col] + B[row][col])
    return t1

@fpy(
)
def mat_add4(A, B, C, D):
    am = size(A, 0)
    an = size(A, 1)
    bm = size(B, 0)
    bn = size(B, 1)
    cm = size(C, 0)
    cn = size(C, 1)
    dm = size(D, 0)
    dn = size(D, 1)
    A_B = mat_add(A, B)
    C_D = mat_add(C, D)
    return mat_add(A_B, C_D)

@fpy(
)
def main():
    A = [[round(1), round(0)], [round(0), round(1)]]
    return mat_add4(A, A, A, A)

@fpy(
    meta={
        'pre': lambda A, B: an == bm,
    }
)
def matmul(A, B):
    am = size(A, 0)
    an = size(A, 1)
    bm = size(B, 0)
    bn = size(B, 1)
    t = am
    t0 = bn
    t1 = [empty(t0) for _ in range(t)]
    for m in range(t):
        for n in range(t0):
            t2 = bm
            prod = round(0)
            for i in range(t2):
                prod3 = (prod + (A[m][i] * B[i][n]))
                prod = prod3
            t1[m][n] = prod
    return t1

@fpy(
)
def main():
    A = [[round(0), round(1)], [round(1), round(0)]]
    B = [[round(1), round(2)], [round(3), round(4)]]
    return matmul(A, B)

@fpy(
)
def f(x):
    return ((x * x) - round(612))

@fpy(
)
def fprime(x):
    return (round(2) * x)

@fpy(
    ctx=IEEEContext(es=15, nbits=256, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
)
def newton_raphson(x0, tolerance):
    x1 = x0
    x2 = (x0 - (f(x0) / fprime(x0)))
    while abs((x2 - x1)) > tolerance:
        t = x2
        t3 = (x2 - (f(x2) / fprime(x2)))
        x1 = t
        x2 = t3
    return x2

@fpy(
)
def sqrt_newton(a):
    iters = round(10)
    x0 = round(10)
    t = iters
    x1 = x0
    x2 = (x0 - (((x0 * x0) - a) / (round(2) * x0)))
    t3 = empty(t)
    for i in range(t):
        x1 = x2
        x2 = (x1 - (((x1 * x1) - a) / (round(2) * x1)))
        t3[i] = sqrt(a)
    return t3

@fpy(
    meta={
        'pre': lambda a: a >= round(0),
    }
)
def sqrt_residual(a):
    x0 = a
    x1 = (x0 - (((x0 * x0) - a) / (round(2) * x0)))
    old_residual = inf()
    residual = abs(((x1 * x1) - a))
    while old_residual > residual > round(0):
        x0 = x1
        x1 = (x0 - (((x0 * x0) - a) / (round(2) * x0)))
        old_residual = residual
        residual = abs(((x1 * x1) - a))
    if residual == round(0):
        t = x1
    else:
        t = x0
    return t

@fpy(
    meta={
        'pre': lambda a: round(0) <= a,
    }
)
def sqrt_epsilon(a):
    x_n = round(0)
    e = inf()
    x = a
    while e > round(rational(1, 2000)):
        x_n = (x - (((x * x) - a) / (round(2) * x)))
        e = abs((x_n - x))
        x = x_n
    return x

@fpy(
    meta={
        'pre': lambda a: round(0) <= a <= round(1e100),
    }
)
def sqrt_residual_2(a):
    x = a
    prev_residual = inf()
    residual = inf()
    while prev_residual > residual > round(0) or residual == inf():
        x = (x - (((x * x) - a) / (round(2) * x)))
        prev_residual = residual
        residual = abs(((x * x) - a))
    return x

@fpy(
    meta={
        'pre': lambda a: round(0) <= a,
        'spec': lambda a: sqrt(a),
    }
)
def babylonian_residual(a):
    prev_x = a
    x = a
    prev_residual = inf()
    residual = inf()
    while prev_residual > residual > round(0) or residual == inf():
        prev_x = x
        x = (round(rational(1, 2)) * (x + (a / x)))
        prev_residual = residual
        residual = abs(((x * x) - a))
    if residual == round(0):
        t = x
    else:
        t = prev_x
    return t

@fpy(
)
def main(a):
    result = babylonian_residual(a)
    return [result, (result - sqrt(a))]

@fpy(
)
def f(x):
    return ((x * x) - round(612))

@fpy(
)
def fprime(x):
    return (round(2) * x)

@fpy(
)
def newton_raphson(x0, tolerance):
    x1 = x0
    x2 = (x0 - (f(x0) / fprime(x0)))
    while abs((x2 - x1)) > tolerance:
        t = x2
        t3 = (x2 - (f(x2) / fprime(x2)))
        x1 = t
        x2 = t3
    return x2

@fpy(
)
def sqrt_newton(a):
    iters = round(10)
    x0 = round(10)
    t = iters
    x1 = x0
    x2 = (x0 - (((x0 * x0) - a) / (round(2) * x0)))
    t3 = empty(t)
    for i in range(t):
        x1 = x2
        x2 = (x1 - (((x1 * x1) - a) / (round(2) * x1)))
        t3[i] = sqrt(a)
    return t3

@fpy(
    meta={
        'pre': lambda a: a >= round(0),
    }
)
def sqrt_residual(a):
    x0 = a
    x1 = (x0 - (((x0 * x0) - a) / (round(2) * x0)))
    old_residual = inf()
    residual = abs(((x1 * x1) - a))
    while old_residual > residual > round(0):
        x0 = x1
        x1 = (x0 - (((x0 * x0) - a) / (round(2) * x0)))
        old_residual = residual
        residual = abs(((x1 * x1) - a))
    if residual == round(0):
        t = x1
    else:
        t = x0
    return t

@fpy(
    meta={
        'pre': lambda a: round(0) <= a,
    }
)
def sqrt_epsilon(a):
    x_n = round(0)
    e = inf()
    x = a
    while e > round(rational(1, 2000)):
        x_n = (x - (((x * x) - a) / (round(2) * x)))
        e = abs((x_n - x))
        x = x_n
    return x

@fpy(
    meta={
        'pre': lambda a: round(0) <= a <= round(1e100),
    }
)
def sqrt_residual_2(a):
    x = a
    prev_residual = inf()
    residual = inf()
    while prev_residual > residual > round(0) or residual == inf():
        x = (x - (((x * x) - a) / (round(2) * x)))
        prev_residual = residual
        residual = abs(((x * x) - a))
    return x

@fpy(
    meta={
        'pre': lambda a: round(0) <= a,
        'spec': lambda a: sqrt(a),
    }
)
def babylonian_residual(a):
    prev_x = a
    x = a
    prev_residual = inf()
    residual = inf()
    while prev_residual > residual > round(0) or residual == inf():
        prev_x = x
        x = (round(rational(1, 2)) * (x + (a / x)))
        prev_residual = residual
        residual = abs(((x * x) - a))
    if residual == round(0):
        t = x
    else:
        t = prev_x
    return t

@fpy(
)
def sqrt_newton(a, residual_bound):
    x = a
    residual = residual_bound
    while residual >= residual_bound:
        x = (x - (((x * x) - a) / (round(2) * x)))
        residual = abs(((x * x) - a))
    return x

@fpy(
)
def sqrt_bfloat_limit(a, residual_bound):
    x = a
    with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
        t = ((x * x) - a)
    residual = t
    steps = round(0)
    with MPFixedContext(nmin=-1, rm=RoundingMode.RNE, num_randbits=0, enable_nan=False, enable_inf=False, nan_value=None, inf_value=None):
        t0 = round(2)
    while steps < t0 and abs(residual) >= residual_bound:
        x = (x - (residual / (round(2) * x)))
        with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
            t1 = ((x * x) - a)
        residual = t1
        with MPFixedContext(nmin=-1, rm=RoundingMode.RNE, num_randbits=0, enable_nan=False, enable_inf=False, nan_value=None, inf_value=None):
            t2 = (round(1) + steps)
        steps = t2
    return x

@fpy(
    meta={
        'pre': lambda a, residual_bound: round(0) <= a,
        'spec': lambda a, residual_bound: sqrt(a),
    }
)
def bab_bfloat_limit(a, residual_bound):
    x = a
    residual = abs(((x * x) - a))
    steps = round(0)
    with MPFixedContext(nmin=-1, rm=RoundingMode.RNE, num_randbits=0, enable_nan=False, enable_inf=False, nan_value=None, inf_value=None):
        t = round(20)
    while steps < t and abs(residual) >= residual_bound:
        x = (round(rational(1, 2)) * (x + (a / x)))
        residual = abs(((x * x) - a))
        with MPFixedContext(nmin=-1, rm=RoundingMode.RNE, num_randbits=0, enable_nan=False, enable_inf=False, nan_value=None, inf_value=None):
            t0 = (round(1) + steps)
        steps = t0
    return x

@fpy(
)
def sqrt_bfloat(a, residual_bound):
    x = a
    with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
        t = ((x * x) - a)
    residual = t
    while abs(residual) >= residual_bound:
        x = (x - (residual / (round(2) * x)))
        with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
            t0 = ((x * x) - a)
        residual = t0
    return x

@fpy(
)
def main(a):
    result = sqrt_bfloat_limit(a, round(rational(1, 100)))
    return [result, (result - sqrt(a))]

@fpy(
)
def bottom_left():
    A = [[round(1), round(2)], [round(3), round(4)]]
    return A[round(1)][round(0)]

@fpy(
)
def top_right():
    A = [[round(1), round(2)], [round(3), round(4)]]
    return A[round(0)][round(1)]

@fpy(
)
def main():
    return [bottom_left(), top_right()]

@fpy(
)
def vec_scale(A, x):
    n = size(A, 0)
    t = n
    t0 = empty(t)
    for i in range(t):
        t0[i] = (A[i] * x)
    return t0

@fpy(
    meta={
        'pre': lambda A, B: n == m,
    }
)
def vec_add(A, B):
    n = size(A, 0)
    m = size(B, 0)
    t = n
    t0 = empty(t)
    for i in range(t):
        t0[i] = (A[i] + B[i])
    return t0

@fpy(
)
def sum_1d(A):
    n = size(A, 0)
    t = n
    total = round(0)
    for i in range(t):
        total0 = (total + A[i])
        total = total0
    return total

@fpy(
)
def main(n):
    t = n
    t0 = empty(t)
    for i in range(t):
        t0[i] = i
    A = t0
    return sum_1d(A)

