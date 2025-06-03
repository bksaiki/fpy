from fpy2 import *
from fpy2.typing import *

@fpy
def I_2x2():
    A = ((1, 0), (0, 1))
    return A

@fpy
def dim_2x3():
    A = ((1, 0, 1), (0, 1, 1))
    return dim(A)

@fpy
def size_2x3():
    A = ((1, 0, 1), (0, 1, 1))
    rows = size(A, 0)
    cols = size(A, 1)
    return (rows, cols)

@fpy
def alternating_sum(A):
    (n,) = shape(A)
    t = n
    sign = 1
    total = 0
    for i in range(t):
        sign = -sign
        total = (total + (sign * A[i]))
    return total

@fpy
def main():
    return alternating_sum((1, 2, 3, 4))

@fpy
def dim4(A, B, C, D):
    bm, bn = shape(B)
    cm, cn = shape(C)
    dm, dn = shape(D)
    D0 = (A, B, C, D)
    return dim(D0)

@fpy
def main():
    A = ((1, 0), (0, 1))
    B = ((True, False), (False, True))
    return dim4(A, A, A, A)

@fpy
def dim4(A, B, C, D):
    am, an = shape(A)
    bm, bn = shape(B)
    cm, cn = shape(C)
    dm, dn = shape(D)
    D0 = (A, B, C, D)
    return dim(D0)

@fpy
def main():
    A = ((1, 0, 0), (0, 1, 0))
    return dim4(A, A, A, A)

@fpy
def expand(A):
    (n,) = shape(A)
    t = n
    t0 = [0 for _ in range(t)]
    for i in range(t):
        t0[i] = A
    return t0

@fpy
def main():
    A = (1, 2, 3, 4)
    return expand(A)

@fpy
def fibonacci(n):
    t = n
    f_n_2 = 0
    f_n_1 = 0
    f_n = 0
    t0 = [0 for _ in range(t)]
    for i in range(t):
        t0[i] = f_n
        f_n_2 = f_n_1
        f_n_1 = f_n
        if i == 1:
            t1 = 1
        else:
            t1 = (f_n_1 + f_n_2)
        f_n = t1
    return t0

@fpy(pre=lambda A, B: an == bm)
def matmul(A, B):
    am, an = shape(A)
    bm, bn = shape(B)
    t = am
    t0 = bn
    t1 = [0 for _ in range(t) for _ in range(t0)]
    for m in range(t):
        for n in range(t0):
            t2 = bm
            prod = 0
            for i in range(t2):
                prod3 = (prod + (A[m][i] * B[i][n]))
                prod = prod3
            t1[m][n] = prod
    return t1

@fpy
def fib_Q():
    return ((1, 1), (1, 0))

# @fpy
# def fib_rec(A, n):
#     if n == 2:
#         t1 = A[0][0]
#     else:
#         if n == 1:
#             t0 = A[0][1]
#         else:
#             if n <= 0:
#                 t = A[1][1]
#             else:
#                 t = fib_rec(matmul(A, fib_Q()), (n - 1))
#             t0 = t
#         t1 = t0
#     return t1

# @fpy
# def fib(n):
#     return fib_rec(fib_Q(), n)

@fpy
def fib_iterative(n):
    t = n
    A = fib_Q()
    for i in range(t):
        A0 = matmul(A, fib_Q())
        A = A0
    return A[1][1]

# @fpy
# def main(n):
#     t = n
#     t0 = 2
#     t1 = [0 for _ in range(t) for _ in range(t0)]
#     for i in range(t):
#         for method in range(t0):
#             if method == 0:
#                 t2 = fib(i)
#             else:
#                 t2 = fib_iterative(i)
#             t1[i][method] = t2
#     return t1

@fpy
def f18(n):
    with FPCoreContext(precision='integer') as _:
        t = 1
    return t

@fpy(pre=lambda A, B: am == bm and an == bn)
def mat_add(A, B):
    am, an = shape(A)
    bm, bn = shape(B)
    t = am
    t0 = an
    t1 = [0 for _ in range(t) for _ in range(t0)]
    for row in range(t):
        for col in range(t0):
            t1[row][col] = (A[row][col] + B[row][col])
    return t1

@fpy
def mat_add4(A, B, C, D):
    am, an = shape(A)
    bm, bn = shape(B)
    cm, cn = shape(C)
    dm, dn = shape(D)
    A_B = mat_add(A, B)
    C_D = mat_add(C, D)
    return mat_add(A_B, C_D)

@fpy
def main():
    A = ((1, 0), (0, 1))
    return mat_add4(A, A, A, A)

@fpy(pre=lambda A, B: an == bm)
def matmul(A, B):
    am, an = shape(A)
    bm, bn = shape(B)
    t = am
    t0 = bn
    t1 = [0 for _ in range(t) for _ in range(t0)]
    for m in range(t):
        for n in range(t0):
            t2 = bm
            prod = 0
            for i in range(t2):
                prod3 = (prod + (A[m][i] * B[i][n]))
                prod = prod3
            t1[m][n] = prod
    return t1

@fpy
def main():
    A = ((0, 1), (1, 0))
    B = ((1, 2), (3, 4))
    return matmul(A, B)

# @fpy
# def even_int(n):
#     if n <= 0:
#         t = True
#     else:
#         t = odd_int((n - 1))
#     return t

# @fpy
# def odd_int(n):
#     if n <= 0:
#         t = False
#     else:
#         t = even_int((n - 1))
#     return t

# @fpy
# def even_odd_tensor(A):
#     (n,) = shape(A)
#     t = n
#     t0 = [0 for _ in range(t)]
#     for i in range(t):
#         t0[i] = even_int(A[i])
#     return t0

# @fpy
# def f27(n):
#     t = n
#     t0 = [0 for _ in range(t)]
#     for i in range(t):
#         t0[i] = i
#     A = t0
#     return even_odd_tensor(A)

@fpy
def f(x):
    return ((x * x) - 612)

@fpy
def fprime(x):
    return (2 * x)

@fpy(context=IEEEContext(es=15, nbits=256, rm=RoundingMode.RNE))
def newton_raphson(x0, tolerance):
    x00 = x0
    x1 = (x0 - (f(x0) / fprime(x0)))
    while fabs((x1 - x00)) > tolerance:
        t = x1
        t1 = (x1 - (f(x1) / fprime(x1)))
        x00 = t
        x1 = t1
    return x1

@fpy
def sqrt_newton(a):
    iters = 10
    x0 = 10
    t = iters
    x00 = x0
    x1 = (x0 - (((x0 * x0) - a) / (2 * x0)))
    t1 = [0 for _ in range(t)]
    for i in range(t):
        t1[i] = sqrt(a)
        x00 = x1
        x1 = (x00 - (((x00 * x00) - a) / (2 * x00)))
    return t1

@fpy(pre=lambda a: a >= 0)
def sqrt_residual(a):
    x0 = a
    x1 = (x0 - (((x0 * x0) - a) / (2 * x0)))
    old_residual = INFINITY
    residual = fabs(((x1 * x1) - a))
    while old_residual > residual > 0:
        x0 = x1
        x1 = (x0 - (((x0 * x0) - a) / (2 * x0)))
        old_residual = residual
        residual = fabs(((x1 * x1) - a))
    if residual == 0:
        t = x1
    else:
        t = x0
    return t

@fpy(pre=lambda a: 0 <= a)
def sqrt_epsilon(a):
    x_n = 0
    e = INFINITY
    x = a
    while e > rational(1, 2000):
        x_n = (x - (((x * x) - a) / (2 * x)))
        e = fabs((x_n - x))
        x = x_n
    return x

@fpy(pre=lambda a: 0 <= a <= 1e100)
def sqrt_residual_2(a):
    x = a
    prev_residual = INFINITY
    residual = INFINITY
    while prev_residual > residual > 0 or residual == INFINITY:
        x = (x - (((x * x) - a) / (2 * x)))
        prev_residual = residual
        residual = fabs(((x * x) - a))
    return x

@fpy(
    pre=lambda a: 0 <= a,
    spec=lambda a: sqrt(a),
)
def babylonian_residual(a):
    prev_x = a
    x = a
    prev_residual = INFINITY
    residual = INFINITY
    while prev_residual > residual > 0 or residual == INFINITY:
        prev_x = x
        x = (rational(1, 2) * (x + (a / x)))
        prev_residual = residual
        residual = fabs(((x * x) - a))
    if residual == 0:
        t = x
    else:
        t = prev_x
    return t

@fpy
def main(a):
    result = babylonian_residual(a)
    return (result, (result - sqrt(a)))

@fpy
def f(x):
    return ((x * x) - 612)

@fpy
def fprime(x):
    return (2 * x)

@fpy
def newton_raphson(x0, tolerance):
    x00 = x0
    x1 = (x0 - (f(x0) / fprime(x0)))
    while fabs((x1 - x00)) > tolerance:
        t = x1
        t1 = (x1 - (f(x1) / fprime(x1)))
        x00 = t
        x1 = t1
    return x1

@fpy
def sqrt_newton(a):
    iters = 10
    x0 = 10
    t = iters
    x00 = x0
    x1 = (x0 - (((x0 * x0) - a) / (2 * x0)))
    t1 = [0 for _ in range(t)]
    for i in range(t):
        t1[i] = sqrt(a)
        x00 = x1
        x1 = (x00 - (((x00 * x00) - a) / (2 * x00)))
    return t1

@fpy(pre=lambda a: a >= 0)
def sqrt_residual(a):
    x0 = a
    x1 = (x0 - (((x0 * x0) - a) / (2 * x0)))
    old_residual = INFINITY
    residual = fabs(((x1 * x1) - a))
    while old_residual > residual > 0:
        x0 = x1
        x1 = (x0 - (((x0 * x0) - a) / (2 * x0)))
        old_residual = residual
        residual = fabs(((x1 * x1) - a))
    if residual == 0:
        t = x1
    else:
        t = x0
    return t

@fpy(pre=lambda a: 0 <= a)
def sqrt_epsilon(a):
    x_n = 0
    e = INFINITY
    x = a
    while e > rational(1, 2000):
        x_n = (x - (((x * x) - a) / (2 * x)))
        e = fabs((x_n - x))
        x = x_n
    return x

@fpy(pre=lambda a: 0 <= a <= 1e100)
def sqrt_residual_2(a):
    x = a
    prev_residual = INFINITY
    residual = INFINITY
    while prev_residual > residual > 0 or residual == INFINITY:
        x = (x - (((x * x) - a) / (2 * x)))
        prev_residual = residual
        residual = fabs(((x * x) - a))
    return x

@fpy(
    pre=lambda a: 0 <= a,
    spec=lambda a: sqrt(a),
)
def babylonian_residual(a):
    prev_x = a
    x = a
    prev_residual = INFINITY
    residual = INFINITY
    while prev_residual > residual > 0 or residual == INFINITY:
        prev_x = x
        x = (rational(1, 2) * (x + (a / x)))
        prev_residual = residual
        residual = fabs(((x * x) - a))
    if residual == 0:
        t = x
    else:
        t = prev_x
    return t

@fpy
def sqrt_newton(a, residual_bound):
    x = a
    residual = residual_bound
    while residual >= residual_bound:
        x = (x - (((x * x) - a) / (2 * x)))
        residual = fabs(((x * x) - a))
    return x

@fpy
def sqrt_bfloat_limit(a, residual_bound):
    x = a
    with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE) as _:
        t = ((x * x) - a)
    residual = t
    steps = 0
    with FPCoreContext(precision='integer') as _:
        t0 = 2
    while steps < t0 and fabs(residual) >= residual_bound:
        x = (x - (residual / (2 * x)))
        with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE) as _:
            t1 = ((x * x) - a)
        residual = t1
        with FPCoreContext(precision='integer') as _:
            t2 = (1 + steps)
        steps = t2
    return x

@fpy(
    pre=lambda a, residual_bound: 0 <= a,
    spec=lambda a, residual_bound: sqrt(a),
)
def bab_bfloat_limit(a, residual_bound):
    x = a
    residual = fabs(((x * x) - a))
    steps = 0
    with FPCoreContext(precision='integer') as _:
        t = 20
    while steps < t and fabs(residual) >= residual_bound:
        x = (rational(1, 2) * (x + (a / x)))
        residual = fabs(((x * x) - a))
        with FPCoreContext(precision='integer') as _:
            t0 = (1 + steps)
        steps = t0
    return x

@fpy
def sqrt_bfloat(a, residual_bound):
    x = a
    with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE) as _:
        t = ((x * x) - a)
    residual = t
    while fabs(residual) >= residual_bound:
        x = (x - (residual / (2 * x)))
        with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE) as _:
            t0 = ((x * x) - a)
        residual = t0
    return x

@fpy
def main(a):
    result = sqrt_bfloat_limit(a, rational(1, 100))
    return (result, (result - sqrt(a)))

@fpy
def bottom_left():
    A = ((1, 2), (3, 4))
    return A[1][0]

@fpy
def top_right():
    A = ((1, 2), (3, 4))
    return A[0][1]

@fpy
def main():
    return (bottom_left(), top_right())

@fpy
def vec_scale(A, x):
    (n,) = shape(A)
    t = n
    t0 = [0 for _ in range(t)]
    for i in range(t):
        t0[i] = (A[i] * x)
    return t0

@fpy(pre=lambda A, B: n == m)
def vec_add(A, B):
    (n,) = shape(A)
    (m,) = shape(B)
    t = n
    t0 = [0 for _ in range(t)]
    for i in range(t):
        t0[i] = (A[i] + B[i])
    return t0

@fpy
def lorenz_3d(xyz):
    sigma = 10
    beta = rational(8, 3)
    rho = 28
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    return ((sigma * (y - x)), ((x * (rho - z)) - y), ((x * y) - (beta * z)))

@fpy
def forward_euler_3d(xyz, h):
    k1 = vec_scale(target_3d(xyz), h)
    return vec_add(xyz, k1)

@fpy
def midpoint_3d(xyz, h):
    k1 = vec_scale(target_3d(xyz), h)
    k2 = vec_scale(target_3d(vec_add(xyz, vec_scale(k1, rational(1, 2)))), h)
    return vec_add(xyz, k2)

@fpy
def ralston_3d(xyz, h):
    k1 = vec_scale(target_3d(xyz), h)
    k2 = vec_scale(target_3d(vec_add(xyz, vec_scale(k1, rational(2, 3)))), h)
    t = 3
    t0 = [0 for _ in range(t)]
    for i in range(t):
        t0[i] = (xyz[i] + (rational(1, 4) * (k1[i] + (k2[i] * 3))))
    return t0

@fpy
def rk4_step_3d(xyz, h):
    k1 = vec_scale(target_3d(xyz), h)
    k2 = vec_scale(target_3d(vec_add(xyz, vec_scale(k1, rational(1, 2)))), h)
    k3 = vec_scale(target_3d(vec_add(xyz, vec_scale(k2, rational(1, 2)))), h)
    k4 = vec_scale(target_3d(vec_add(xyz, k3)), h)
    t = 3
    t0 = [0 for _ in range(t)]
    for i in range(t):
        t0[i] = (xyz[i] + (rational(1, 6) * (((k1[i] + (k2[i] * 2)) + (k3[i] * 2)) + k4[i])))
    return t0

@fpy
def target_3d(xyz):
    return lorenz_3d(xyz)

@fpy
def step_3d(xyz, h):
    return rk4_step_3d(xyz, h)

@fpy
def rk4_3d_run(initial_conditions, h, steps):
    t = steps
    xyz = initial_conditions
    t0 = [0 for _ in range(t)]
    for step in range(t):
        t0[step] = xyz
        xyz = step_3d(xyz, h)
    return t0

@fpy
def main():
    return rk4_3d_run((1, 1, 1), .02, 685)

@fpy
def sum_1d(A):
    (n,) = shape(A)
    t = n
    total = 0
    for i in range(t):
        total0 = (total + A[i])
        total = total0
    return total

@fpy
def main(n):
    t = n
    t0 = [0 for _ in range(t)]
    for i in range(t):
        t0[i] = i
    A = t0
    return sum_1d(A)

