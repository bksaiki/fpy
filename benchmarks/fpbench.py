from fpy2 import *
from fpy2.typing import *

@fpy
def Arrow_Hurwicz(x, y, u, v):
    a = 1
    b = -1
    c = -1
    r = rational(1, 2)
    u0 = u
    v1 = v
    x2 = x
    y3 = y
    while fmax(fabs((x2 - u0)), fabs((y3 - v1))) > 1e-9:
        u0 = x2
        v1 = y3
        x2 = (u0 - (r * ((a * u0) + (b * v1))))
        y3 = fmax((v1 + ((r / 2) * ((b * u0) - c))), 0)
    return (x2, y3)

@fpy
def Euler_Oscillator(x, v):
    h = 0.01
    v0 = v
    x1 = x
    while True:
        t = ((v0 * (1 - h)) - (h * x1))
        t2 = (x1 + (h * v0))
        v0 = t
        x1 = t2
    return (x1, v0)

@fpy
def Filter(x, y):
    x0 = x
    y1 = y
    while True:
        x0 = ((rational(3, 4) * x0) - (rational(1, 8) * y1))
        y1 = x0
    return y1

@fpy
def Symplectic_Oscillator(x, v):
    tau = 0.1
    x0 = x
    v1 = v
    while v1 >= rational(1, 2):
        x0 = (((1 - (tau / 2)) * x0) + ((tau - (pow(tau, 3) / 4)) * v1))
        v1 = ((-tau * x0) + ((1 - (tau / 2)) * v1))
    return (x0, v1)

@fpy
def Circle(x, y):
    d = 0
    x0 = x
    y1 = y
    while True:
        d = (((0.1 + (x0 * x0)) + (y1 * y1)) / 2)
        x0 = (x0 * d)
        y1 = (y1 * d)
    return (x0, y1)

@fpy
def Flower(x, y):
    x0 = x
    y1 = y
    while (((0.15 + (x0 * x0)) + (y1 * y1)) / 2) > 0.1:
        t = (x0 / (((0.15 + (x0 * x0)) + (y1 * y1)) / 2))
        t2 = (y1 * (((0.15 + (x0 * x0)) + (y1 * y1)) / 2))
        x0 = t
        y1 = t2
    return (x0, y1)

@fpy
def carthesianToPolar_u44__radius(x, y):
    return sqrt(((x * x) + (y * y)))

@fpy
def carthesianToPolar_u44__theta(x, y):
    pi = 3.14159265359
    radiant = atan((y / x))
    return (radiant * (180.0 / pi))

@fpy
def polarToCarthesian_u44__x(radius, theta):
    pi = 3.14159265359
    radiant = (theta * (pi / 180.0))
    return (radius * cos(radiant))

@fpy
def polarToCarthesian_u44__y(radius, theta):
    pi = 3.14159265359
    radiant = (theta * (pi / 180.0))
    return (radius * sin(radiant))

@fpy
def instantaneousCurrent(t, resistance, frequency, inductance, maxVoltage):
    pi = 3.14159265359
    impedance_re = resistance
    impedance_im = (((2 * pi) * frequency) * inductance)
    denom = ((impedance_re * impedance_re) + (impedance_im * impedance_im))
    current_re = ((maxVoltage * impedance_re) / denom)
    current_im = (-(maxVoltage * impedance_im) / denom)
    maxCurrent = sqrt(((current_re * current_re) + (current_im * current_im)))
    theta = atan((current_im / current_re))
    return (maxCurrent * cos(((((2 * pi) * frequency) * t) + theta)))

@fpy
def matrixDeterminant(a, b, c, d, e, f, g, h, i):
    return (((((a * e) * i) + ((b * f) * g)) + ((c * d) * h)) - ((((c * e) * g) + ((b * d) * i)) + ((a * f) * h)))

@fpy
def matrixDeterminant2(a, b, c, d, e, f, g, h, i):
    return (((a * (e * i)) + ((g * (b * f)) + (c * (d * h)))) - ((e * (c * g)) + ((i * (b * d)) + (a * (f * h)))))

@fpy
def intro_example_mixed(t):
    with IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE) as _:
        with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE) as _:
            t0 = (t + 1)
        t1 = (t / t0)
    return cast(t1)

@fpy
def delta4(x1, x2, x3, x4, x5, x6):
    return ((((((-x2 * x3) - (x1 * x4)) + (x2 * x5)) + (x3 * x6)) - (x5 * x6)) + (x1 * (((((-x1 + x2) + x3) - x4) + x5) + x6)))

@fpy
def delta(x1, x2, x3, x4, x5, x6):
    return ((((((((x1 * x4) * (((((-x1 + x2) + x3) - x4) + x5) + x6)) + ((x2 * x5) * (((((x1 - x2) + x3) + x4) - x5) + x6))) + ((x3 * x6) * (((((x1 + x2) - x3) + x4) + x5) - x6))) + ((-x2 * x3) * x4)) + ((-x1 * x3) * x5)) + ((-x1 * x2) * x6)) + ((-x4 * x5) * x6))

@fpy
def sqrt_add(x):
    return (1 / (sqrt((x + 1)) + sqrt(x)))

@fpy
def exp1x(x):
    return ((exp(x) - 1) / x)

@fpy
def exp1x_32(x):
    return ((exp(x) - 1) / x)

@fpy
def floudas(x1, x2):
    return (x1 + x2)

@fpy
def exp1x_log(x):
    return ((exp(x) - 1) / log(exp(x)))

@fpy
def x_by_xy(x, y):
    return (x / (x + y))

@fpy
def hypot(x1, x2):
    return sqrt(((x1 * x1) + (x2 * x2)))

@fpy
def hypot32(x1, x2):
    return sqrt(((x1 * x1) + (x2 * x2)))

@fpy
def logexp(x):
    return log((1 + exp(x)))

@fpy
def sum(x0, x1, x2):
    p0 = ((x0 + x1) - x2)
    p1 = ((x1 + x2) - x0)
    p2 = ((x2 + x0) - x1)
    return ((p0 + p1) + p2)

@fpy
def nonlin1(z):
    return (z / (z + 1))

@fpy
def nonlin2(x, y):
    t = (x * y)
    return ((t - 1) / ((t * t) - 1))

@fpy
def i4(x, y):
    return sqrt((x + (y * y)))

@fpy
def i6(x, y):
    return sin((x * y))

@fpy
def himmilbeau(x1, x2):
    a = (((x1 * x1) + x2) - 11)
    b = ((x1 + (x2 * x2)) - 7)
    return ((a * a) + (b * b))

@fpy
def logexp(x):
    e = exp(x)
    return log((1 + e))

@fpy
def sphere(x, r, lat, lon):
    sinLat = sin(lat)
    cosLon = cos(lon)
    return (x + ((r * sinLat) * cosLon))

@fpy
def azimuth(lat1, lat2, lon1, lon2):
    dLon = (lon2 - lon1)
    s_lat1 = sin(lat1)
    c_lat1 = cos(lat1)
    s_lat2 = sin(lat2)
    c_lat2 = cos(lat2)
    s_dLon = sin(dLon)
    c_dLon = cos(dLon)
    return atan(((c_lat2 * s_dLon) / ((c_lat1 * s_lat2) - ((s_lat1 * c_lat2) * c_dLon))))

@fpy
def floudas1(x1, x2, x3, x4, x5, x6):
    return ((((((-25 * ((x1 - 2) * (x1 - 2))) - ((x2 - 2) * (x2 - 2))) - ((x3 - 1) * (x3 - 1))) - ((x4 - 4) * (x4 - 4))) - ((x5 - 1) * (x5 - 1))) - ((x6 - 4) * (x6 - 4)))

@fpy
def floudas2(x1, x2):
    return (-x1 - x2)

@fpy
def floudas3(x1, x2):
    return (((-12 * x1) - (7 * x2)) + (x2 * x2))

@fpy
def hartman3(x1, x2, x3):
    e1 = (((3.0 * ((x1 - 0.3689) * (x1 - 0.3689))) + (10.0 * ((x2 - 0.117) * (x2 - 0.117)))) + (30.0 * ((x3 - 0.2673) * (x3 - 0.2673))))
    e2 = (((0.1 * ((x1 - 0.4699) * (x1 - 0.4699))) + (10.0 * ((x2 - 0.4387) * (x2 - 0.4387)))) + (35.0 * ((x3 - 0.747) * (x3 - 0.747))))
    e3 = (((3.0 * ((x1 - 0.1091) * (x1 - 0.1091))) + (10.0 * ((x2 - 0.8732) * (x2 - 0.8732)))) + (30.0 * ((x3 - 0.5547) * (x3 - 0.5547))))
    e4 = (((0.1 * ((x1 - 0.03815) * (x1 - 0.03815))) + (10.0 * ((x2 - 0.5743) * (x2 - 0.5743)))) + (35.0 * ((x3 - 0.8828) * (x3 - 0.8828))))
    exp1 = exp(-e1)
    exp2 = exp(-e2)
    exp3 = exp(-e3)
    exp4 = exp(-e4)
    return -((((1.0 * exp1) + (1.2 * exp2)) + (3.0 * exp3)) + (3.2 * exp4))

@fpy
def hartman6(x1, x2, x3, x4, x5, x6):
    e1 = ((((((10.0 * ((x1 - 0.1312) * (x1 - 0.1312))) + (3.0 * ((x2 - 0.1696) * (x2 - 0.1696)))) + (17.0 * ((x3 - 0.5569) * (x3 - 0.5569)))) + (3.5 * ((x4 - 0.0124) * (x4 - 0.0124)))) + (1.7 * ((x5 - 0.8283) * (x5 - 0.8283)))) + (8.0 * ((x6 - 0.5886) * (x6 - 0.5886))))
    e2 = ((((((0.05 * ((x1 - 0.2329) * (x1 - 0.2329))) + (10.0 * ((x2 - 0.4135) * (x2 - 0.4135)))) + (17.0 * ((x3 - 0.8307) * (x3 - 0.8307)))) + (0.1 * ((x4 - 0.3736) * (x4 - 0.3736)))) + (8.0 * ((x5 - 0.1004) * (x5 - 0.1004)))) + (14.0 * ((x6 - 0.9991) * (x6 - 0.9991))))
    e3 = ((((((3.0 * ((x1 - 0.2348) * (x1 - 0.2348))) + (3.5 * ((x2 - 0.1451) * (x2 - 0.1451)))) + (1.7 * ((x3 - 0.3522) * (x3 - 0.3522)))) + (10.0 * ((x4 - 0.2883) * (x4 - 0.2883)))) + (17.0 * ((x5 - 0.3047) * (x5 - 0.3047)))) + (8.0 * ((x6 - 0.665) * (x6 - 0.665))))
    e4 = ((((((17.0 * ((x1 - 0.4047) * (x1 - 0.4047))) + (8.0 * ((x2 - 0.8828) * (x2 - 0.8828)))) + (0.05 * ((x3 - 0.8732) * (x3 - 0.8732)))) + (10.0 * ((x4 - 0.5743) * (x4 - 0.5743)))) + (0.1 * ((x5 - 0.1091) * (x5 - 0.1091)))) + (14.0 * ((x6 - 0.0381) * (x6 - 0.0381))))
    exp1 = exp(-e1)
    exp2 = exp(-e2)
    exp3 = exp(-e3)
    exp4 = exp(-e4)
    return -((((1.0 * exp1) + (1.2 * exp2)) + (3.0 * exp3)) + (3.2 * exp4))

@fpy
def kepler0(x1, x2, x3, x4, x5, x6):
    return (((((x2 * x5) + (x3 * x6)) - (x2 * x3)) - (x5 * x6)) + (x1 * (((((-x1 + x2) + x3) - x4) + x5) + x6)))

@fpy
def kepler1(x1, x2, x3, x4):
    return ((((((((x1 * x4) * (((-x1 + x2) + x3) - x4)) + (x2 * (((x1 - x2) + x3) + x4))) + (x3 * (((x1 + x2) - x3) + x4))) - ((x2 * x3) * x4)) - (x1 * x3)) - (x1 * x2)) - x4)

@fpy
def kepler2(x1, x2, x3, x4, x5, x6):
    return ((((((((x1 * x4) * (((((-x1 + x2) + x3) - x4) + x5) + x6)) + ((x2 * x5) * (((((x1 - x2) + x3) + x4) - x5) + x6))) + ((x3 * x6) * (((((x1 + x2) - x3) + x4) + x5) - x6))) - ((x2 * x3) * x4)) - ((x1 * x3) * x5)) - ((x1 * x2) * x6)) - ((x4 * x5) * x6))

@fpy
def intro_example(t):
    return (t / (t + 1))

@fpy
def sec4_example(x, y):
    t = (x * y)
    return ((t - 1) / ((t * t) - 1))

@fpy
def test01_sum3(x0, x1, x2):
    p0 = ((x0 + x1) - x2)
    p1 = ((x1 + x2) - x0)
    p2 = ((x2 + x0) - x1)
    return ((p0 + p1) + p2)

@fpy
def test02_sum8(x0, x1, x2, x3, x4, x5, x6, x7):
    return (((((((x0 + x1) + x2) + x3) + x4) + x5) + x6) + x7)

@fpy
def test03_nonlin2(x, y):
    return ((x + y) / (x - y))

@fpy
def test04_dqmom9(m0, m1, m2, w0, w1, w2, a0, a1, a2):
    v2 = ((w2 * (0 - m2)) * (-3 * ((1 * (a2 / w2)) * (a2 / w2))))
    v1 = ((w1 * (0 - m1)) * (-3 * ((1 * (a1 / w1)) * (a1 / w1))))
    v0 = ((w0 * (0 - m0)) * (-3 * ((1 * (a0 / w0)) * (a0 / w0))))
    return (0.0 + ((v0 * 1) + ((v1 * 1) + ((v2 * 1) + 0.0))))

@fpy
def test05_nonlin1_u44__r4(x):
    r1 = (x - 1)
    r2 = (x * x)
    return (r1 / (r2 - 1))

@fpy
def test05_nonlin1_u44__test2(x):
    return (1 / (x + 1))

@fpy
def test06_sums4_u44__sum1(x0, x1, x2, x3):
    return (((x0 + x1) + x2) + x3)

@fpy
def test06_sums4_u44__sum2(x0, x1, x2, x3):
    return ((x0 + x1) + (x2 + x3))

@fpy
def NMSE_example_3_u46_1(x):
    return (sqrt((x + 1)) - sqrt(x))

@fpy
def NMSE_example_3_u46_3(x, eps):
    return (sin((x + eps)) - sin(x))

@fpy
def NMSE_example_3_u46_4(x):
    return ((1 - cos(x)) / sin(x))

@fpy
def NMSE_example_3_u46_5(N):
    return (atan((N + 1)) - atan(N))

@fpy
def NMSE_example_3_u46_6(x):
    return ((1 / sqrt(x)) - (1 / sqrt((x + 1))))

@fpy
def NMSE_problem_3_u46_3_u46_1(x):
    return ((1 / (x + 1)) - (1 / x))

@fpy
def NMSE_problem_3_u46_3_u46_2(x, eps):
    return (tan((x + eps)) - tan(x))

@fpy
def NMSE_problem_3_u46_3_u46_3(x):
    return (((1 / (x + 1)) - (2 / x)) + (1 / (x - 1)))

@fpy
def NMSE_problem_3_u46_3_u46_4(x):
    return (pow((x + 1), (1 / 3)) - pow(x, (1 / 3)))

@fpy
def NMSE_problem_3_u46_3_u46_5(x, eps):
    return (cos((x + eps)) - cos(x))

@fpy
def NMSE_problem_3_u46_3_u46_6(N):
    return (log((N + 1)) - log(N))

@fpy
def NMSE_problem_3_u46_3_u46_7(x):
    return ((exp(x) - 2) + exp(-x))

@fpy
def NMSE_p42_u44__positive(a, b, c):
    return ((-b + sqrt(((b * b) - (4 * (a * c))))) / (2 * a))

@fpy
def NMSE_p42_u44__negative(a, b, c):
    return ((-b - sqrt(((b * b) - (4 * (a * c))))) / (2 * a))

@fpy
def NMSE_problem_3_u46_2_u46_1_u44__positive(a, b2, c):
    return ((-b2 + sqrt(((b2 * b2) - (a * c)))) / a)

@fpy
def NMSE_problem_3_u46_2_u46_1_u44__negative(a, b2, c):
    return ((-b2 - sqrt(((b2 * b2) - (a * c)))) / a)

@fpy
def NMSE_example_3_u46_7(x):
    return (exp(x) - 1)

@fpy
def NMSE_example_3_u46_8(N):
    return ((((N + 1) * log((N + 1))) - (N * log(N))) - 1)

@fpy
def NMSE_example_3_u46_9(x):
    return ((1 / x) - (1 / tan(x)))

@fpy
def NMSE_example_3_u46_10(x):
    return (log((1 - x)) / log((1 + x)))

@fpy
def NMSE_problem_3_u46_4_u46_1(x):
    return ((1 - cos(x)) / (x * x))

@fpy
def NMSE_problem_3_u46_4_u46_2(a, b, eps):
    return ((eps * (exp(((a + b) * eps)) - 1)) / ((exp((a * eps)) - 1) * (exp((b * eps)) - 1)))

@fpy
def NMSE_problem_3_u46_4_u46_3(eps):
    return log(((1 - eps) / (1 + eps)))

@fpy
def NMSE_problem_3_u46_4_u46_4(x):
    return sqrt(((exp((2 * x)) - 1) / (exp(x) - 1)))

@fpy
def NMSE_problem_3_u46_4_u46_5(x):
    return ((x - sin(x)) / (x - tan(x)))

@fpy
def NMSE_problem_3_u46_4_u46_6(x, n):
    return (pow((x + 1), (1 / n)) - pow(x, (1 / n)))

@fpy
def NMSE_section_3_u46_5(a, x):
    return (exp((a * x)) - 1)

@fpy
def NMSE_section_3_u46_11(x):
    return (exp(x) / (exp(x) - 1))

@fpy
def Complex_square_root(re, im):
    return (0.5 * sqrt((2.0 * (sqrt(((re * re) + (im * im))) + re))))

@fpy
def Complex_sine_and_cosine(re, im):
    return ((0.5 * sin(re)) * (exp(-im) - exp(im)))

@fpy
def Probabilities_in_a_clustering_algorithm(cp, cn, t, s):
    return ((pow((1 / (1 + exp(-s))), cp) * pow((1 - (1 / (1 + exp(-s)))), cn)) / (pow((1 / (1 + exp(-t))), cp) * pow((1 - (1 / (1 + exp(-t)))), cn)))

@fpy
def arclength_of_a_wiggly_function(n):
    dppi = PI
    h = (dppi / n)
    t1 = 0
    t2 = 0
    s1 = 0
    t10 = t1
    with FPCoreContext(precision='integer') as _:
        t = 1
    i = t
    while i <= n:
        x = (i * h)
        with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE) as _:
            t1 = 1
        d1 = t1
        t14 = x
        with FPCoreContext(precision='integer') as _:
            t3 = 1
        k = t3
        while k <= 5:
            with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE) as _:
                t4 = (d1 * 2)
            d1 = t4
            t14 = (t14 + (sin((d1 * x)) / d1))
            with FPCoreContext(precision='integer') as _:
                t5 = (k + 1)
            k = t5
        t2 = t14
        s0 = sqrt(((h * h) + ((t2 - t10) * (t2 - t10))))
        with FPCoreContext(precision='binary80') as _:
            t6 = (s1 + s0)
        s1 = t6
        t10 = t2
        with FPCoreContext(precision='integer') as _:
            t7 = (i + 1)
        i = t7
    return s1

@fpy
def arclength_of_a_wiggly_function__u40_old_version_u41_(n):
    dppi = acos(-1.0)
    h = (dppi / n)
    s1 = 0.0
    t1 = 0.0
    i = 1
    while i <= n:
        x = (i * h)
        with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE) as _:
            t = 2.0
        d0 = t
        t0 = x
        k = 1
        while k <= 5:
            with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE) as _:
                t0 = (2.0 * d0)
            t1 = t0
            t2 = (t0 + (sin((d0 * x)) / d0))
            t3 = (k + 1)
            d0 = t1
            t0 = t2
            k = t3
        t2 = t0
        s0 = sqrt(((h * h) + ((t2 - t1) * (t2 - t1))))
        with FPCoreContext(precision='binary80') as _:
            t4 = (s1 + s0)
        t5 = t4
        x6 = (i * h)
        with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE) as _:
            t7 = 2.0
        d08 = t7
        t09 = x6
        k10 = 1
        while k10 <= 5:
            with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE) as _:
                t11 = (2.0 * d08)
            t12 = t11
            t13 = (t09 + (sin((d08 * x6)) / d08))
            t14 = (k10 + 1)
            d08 = t12
            t09 = t13
            k10 = t14
        t215 = t09
        t16 = t215
        t17 = (i + 1)
        s1 = t5
        t1 = t16
        i = t17
    return s1

@fpy
def doppler1(u, v, T):
    t1 = (331.4 + (0.6 * T))
    return ((-t1 * v) / ((t1 + u) * (t1 + u)))

@fpy
def doppler2(u, v, T):
    t1 = (331.4 + (0.6 * T))
    return ((-t1 * v) / ((t1 + u) * (t1 + u)))

@fpy
def doppler3(u, v, T):
    t1 = (331.4 + (0.6 * T))
    return ((-t1 * v) / ((t1 + u) * (t1 + u)))

@fpy
def rigidBody1(x1, x2, x3):
    return (((-(x1 * x2) - ((2 * x2) * x3)) - x1) - x3)

@fpy
def rigidBody2(x1, x2, x3):
    return (((((((2 * x1) * x2) * x3) + ((3 * x3) * x3)) - (((x2 * x1) * x2) * x3)) + ((3 * x3) * x3)) - x2)

@fpy
def jetEngine(x1, x2):
    t = ((((3 * x1) * x1) + (2 * x2)) - x1)
    t_u42_ = ((((3 * x1) * x1) - (2 * x2)) - x1)
    d = ((x1 * x1) + 1)
    s = (t / d)
    s_u42_ = (t_u42_ / d)
    return (x1 + (((((((((2 * x1) * s) * (s - 3)) + ((x1 * x1) * ((4 * s) - 6))) * d) + (((3 * x1) * x1) * s)) + ((x1 * x1) * x1)) + x1) + (3 * s_u42_)))

@fpy
def turbine1(v, w, r):
    return (((3 + (2 / (r * r))) - (((0.125 * (3 - (2 * v))) * (((w * w) * r) * r)) / (1 - v))) - 4.5)

@fpy
def turbine2(v, w, r):
    return (((6 * v) - (((0.5 * v) * (((w * w) * r) * r)) / (1 - v))) - 2.5)

@fpy
def turbine3(v, w, r):
    return (((3 - (2 / (r * r))) - (((0.125 * (1 + (2 * v))) * (((w * w) * r) * r)) / (1 - v))) - 0.5)

@fpy
def verhulst(x):
    r = 4.0
    K = 1.11
    return ((r * x) / (1 + (x / K)))

@fpy
def predatorPrey(x):
    r = 4.0
    K = 1.11
    return (((r * x) * x) / (1 + ((x / K) * (x / K))))

@fpy
def carbonGas(v):
    p = 3.5e7
    a = 0.401
    b = 42.7e-6
    t = 300
    n = 1000
    k = 1.3806503e-23
    return (((p + ((a * (n / v)) * (n / v))) * (v - (n * b))) - ((k * n) * t))

@fpy
def sine(x):
    return (((x - (((x * x) * x) / 6.0)) + (((((x * x) * x) * x) * x) / 120)) - (((((((x * x) * x) * x) * x) * x) * x) / 5040))

@fpy
def sqroot(x):
    return ((((1.0 + (0.5 * x)) - ((0.125 * x) * x)) + (((0.0625 * x) * x) * x)) - ((((0.0390625 * x) * x) * x) * x))

@fpy
def sineOrder3(x):
    return ((0.954929658551372 * x) - (0.12900613773279798 * ((x * x) * x)))

@fpy
def smartRoot(c):
    a0 = 3
    b1 = 3.5
    discr = ((b1 * b1) - ((a0 * c) * 4.0))
    if ((b1 * b1) - (a0 * c)) > 10:
        if b1 > 0:
            t2 = ((c * 2) / (-b1 - sqrt(discr)))
        else:
            if b1 < 0:
                t = ((-b1 + sqrt(discr)) / (a0 * 2))
            else:
                t = ((-b1 + sqrt(discr)) / (a0 * 2))
            t2 = t
        t5 = t2
    else:
        t5 = ((-b1 + sqrt(discr)) / (a0 * 2))
    return t5

@fpy
def cav10(x):
    if ((x * x) - x) >= 0:
        t = (x / 10)
    else:
        t = ((x * x) + 2)
    return t

@fpy
def squareRoot3(x):
    if x < 1e-5:
        t = (1 + (0.5 * x))
    else:
        t = sqrt((1 + x))
    return t

@fpy
def squareRoot3Invalid(x):
    if x < 1e-4:
        t = (1 + (0.5 * x))
    else:
        t = sqrt((1 + x))
    return t

@fpy
def triangle(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def triangle1(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def triangle2(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def triangle3(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def triangle4(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def triangle5(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def triangle6(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def triangle7(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def triangle8(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def triangle9(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def triangle10(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def triangle11(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def triangle12(a, b, c):
    s = (((a + b) + c) / 2)
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy
def bspline3(u):
    return (-((u * u) * u) / 6)

@fpy
def triangleSorted(a, b, c):
    if a < b:
        t = (sqrt(((((c + (b + a)) * (a - (c - b))) * (a + (c - b))) * (c + (b - a)))) / 4.0)
    else:
        t = (sqrt(((((c + (a + b)) * (b - (c - a))) * (b + (c - a))) * (c + (a - b)))) / 4.0)
    return t

@fpy
def N_Body_Simulation(x0, y0, z0, vx0, vy0, vz0):
    dt = 0.1
    solarMass = 39.47841760435743
    x = x0
    y = y0
    z = z0
    vx = vx0
    vy = vy0
    vz = vz0
    i = 0
    while i < 100:
        distance = sqrt((((x * x) + (y * y)) + (z * z)))
        mag = (dt / ((distance * distance) * distance))
        vxNew = (vx - ((x * solarMass) * mag))
        t = (x + (dt * vxNew))
        distance0 = sqrt((((x * x) + (y * y)) + (z * z)))
        mag1 = (dt / ((distance0 * distance0) * distance0))
        vyNew = (vy - ((y * solarMass) * mag1))
        t2 = (y + (dt * vyNew))
        distance3 = sqrt((((x * x) + (y * y)) + (z * z)))
        mag4 = (dt / ((distance3 * distance3) * distance3))
        vzNew = (vz - ((z * solarMass) * mag4))
        t5 = (z + (dt * vzNew))
        distance6 = sqrt((((x * x) + (y * y)) + (z * z)))
        mag7 = (dt / ((distance6 * distance6) * distance6))
        t8 = (vx - ((x * solarMass) * mag7))
        distance9 = sqrt((((x * x) + (y * y)) + (z * z)))
        mag10 = (dt / ((distance9 * distance9) * distance9))
        t11 = (vy - ((y * solarMass) * mag10))
        distance12 = sqrt((((x * x) + (y * y)) + (z * z)))
        mag13 = (dt / ((distance12 * distance12) * distance12))
        t14 = (vz - ((z * solarMass) * mag13))
        t15 = (i + 1)
        x = t
        y = t2
        z = t5
        vx = t8
        vy = t11
        vz = t14
        i = t15
    return x

@fpy
def Pendulum(t0, w0, N):
    h = 0.01
    L = 2.0
    m = 1.5
    g = 9.80665
    t = t0
    w = w0
    n = 0
    while n < N:
        k1w = ((-g / L) * sin(t))
        k2t = (w + ((h / 2) * k1w))
        t3 = (t + (h * k2t))
        k2w = ((-g / L) * sin((t + ((h / 2) * w))))
        t1 = (w + (h * k2w))
        t2 = (n + 1)
        t = t3
        w = t1
        n = t2
    return t

@fpy
def Sine_Newton(x0):
    x = x0
    i = 0
    while i < 10:
        t = (x - ((((x - (pow(x, 3) / 6.0)) + (pow(x, 5) / 120.0)) + (pow(x, 7) / 5040.0)) / (((1.0 - ((x * x) / 2.0)) + (pow(x, 4) / 24.0)) + (pow(x, 6) / 720.0))))
        t0 = (i + 1)
        x = t
        i = t0
    return x

@fpy
def Rump_u39_s_example_u44__with_pow(a, b):
    return ((((333.75 * pow(b, 6)) + (pow(a, 2) * (((((11 * pow(a, 2)) * pow(b, 2)) - pow(b, 6)) - (121 * pow(b, 4))) - 2))) + (5.5 * pow(b, 8))) + (a / (2 * b)))

@fpy
def Rump_u39_s_example_u44__from_C_program(a, b):
    b2 = (b * b)
    b4 = (b2 * b2)
    b6 = (b4 * b2)
    b8 = (b4 * b4)
    a2 = (a * a)
    firstexpr = (((((11 * a2) * b2) - b6) - (121 * b4)) - 2)
    return ((((333.75 * b6) + (a2 * firstexpr)) + (5.5 * b8)) + (a / (2 * b)))

@fpy
def Rump_u39_s_example_revisited_for_floating_point(a, b):
    b2 = (b * b)
    b4 = (b2 * b2)
    b6 = (b4 * b2)
    b8 = (b4 * b4)
    a2 = (a * a)
    firstexpr = ((((11 * a2) * b2) - (121 * b4)) - 2)
    return (((((333.75 - a2) * b6) + (a2 * firstexpr)) + (5.5 * b8)) + (a / (2 * b)))

@fpy
def Odometry(sr_u42_, sl_u42_):
    inv_l = 0.1
    c = 12.34
    delta_dl = 0.0
    delta_dr = 0.0
    delta_d = 0.0
    delta_theta = 0.0
    arg = 0.0
    cosi = 0.0
    x = 0.0
    sini = 0.0
    y = 0.0
    theta = -.985
    t = 0
    tmp = sl_u42_
    sl = sl_u42_
    sr = sr_u42_
    j = 0
    while t < 1000:
        delta_dl = (c * sl)
        delta_dr = (c * sr)
        delta_d = ((delta_dl + delta_dr) * 0.5)
        delta_theta = ((delta_dr - delta_dl) * inv_l)
        arg = (theta + (delta_theta * 0.5))
        cosi = ((1 - ((arg * arg) * 0.5)) + ((((arg * arg) * arg) * arg) * 0.0416666666))
        x = (x + (delta_d * cosi))
        sini = ((arg - (((arg * arg) * arg) * 0.1666666666)) + (((((arg * arg) * arg) * arg) * arg) * 0.008333333))
        y = (y + (delta_d * sini))
        theta = (theta + delta_theta)
        t = (t + 1)
        tmp = sl
        if j == 50:
            t0 = sr
        else:
            t0 = sl
        sl = t0
        if j == 50:
            t1 = tmp
        else:
            t1 = sr
        sr = t1
        if j == 50:
            t2 = 0
        else:
            t2 = (j + 1)
        j = t2
    return x

@fpy
def PID(m, kp, ki, kd, c):
    dt = 0.5
    invdt = (1 / dt)
    e = 0.0
    p = 0.0
    i = 0.0
    d = 0.0
    r = 0.0
    m0 = m
    eold = 0.0
    t = 0.0
    while t < 100.0:
        e = (c - m0)
        p = (kp * e)
        i = (i + ((ki * dt) * e))
        d = ((kd * invdt) * (e - eold))
        r = ((p + i) + d)
        m0 = (m0 + (0.01 * r))
        eold = e
        t = (t + dt)
    return m0

@fpy
def Runge_Kutta_4(h, y_n_u42_, c):
    sixieme = (1 / 6)
    eps = 0.005
    k = 1.2
    y_n = y_n_u42_
    i = 0.0
    e = 1.0
    while e > eps:
        v = (c - y_n)
        k1 = ((k * v) * v)
        v0 = (c - (y_n + ((0.5 * h) * k1)))
        k2 = ((k * v0) * v0)
        v1 = (c - (y_n + ((0.5 * h) * k2)))
        k3 = ((k * v1) * v1)
        v2 = (c - (y_n + (h * k3)))
        k4 = ((k * v2) * v2)
        t = (y_n + ((sixieme * h) * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4)))
        t3 = (i + 1.0)
        t4 = (e - eps)
        y_n = t
        i = t3
        e = t4
    return fabs(e)

@fpy
def Lead_lag_System(y, yd):
    eps = 0.01
    Dc = -1280.0
    Ac00 = .499
    Ac01 = -.05
    Ac10 = .010
    Ac11 = 1.00
    Bc0 = 1.0
    Bc1 = 0.0
    Cc0 = 564.48
    Cc1 = 0.0
    yc = 0.0
    u = 0.0
    xc0 = 0.0
    xc1 = 0.0
    i = 0.0
    e = 1.0
    while e > eps:
        v = (y - yd)
        if v < -1.0:
            t0 = -1.0
        else:
            if 1.0 < v:
                t = 1.0
            else:
                t = v
            t0 = t
        yc = t0
        u = ((Cc0 * xc0) + ((Cc1 * xc1) + (Dc * yc)))
        xc0 = ((Ac00 * xc0) + ((Ac01 * xc1) + (Bc0 * yc)))
        xc1 = ((Ac10 * xc0) + ((Ac11 * xc1) + (Bc1 * yc)))
        i = (i + 1.0)
        e = fabs((yc - xc1))
    return xc1

@fpy
def Trapeze(u):
    a = 0.25
    b = 5000
    n = 25
    h = ((b - a) / n)
    xb = 0
    r = 0
    xa = 0.25
    while xa < 5000:
        v = (xa + h)
        if v > 5000:
            t = 5000
        else:
            t = v
        xb = t
        gxa = (u / ((((((0.7 * xa) * xa) * xa) - ((0.6 * xa) * xa)) + (0.9 * xa)) - 0.2))
        gxb = (u / ((((((0.7 * xb) * xb) * xb) - ((0.6 * xb) * xb)) + (0.9 * xb)) - 0.2))
        r = (r + (((gxa + gxb) * 0.5) * h))
        xa = (xa + h)
    return r

@fpy
def Rocket_Trajectory(Mf, A):
    R = 6400.0e3
    G = 6.67428e-11
    Mt = 5.9736e24
    dt = 0.1
    T = (24.0 * 3600.0)
    nombrepas = (T / dt)
    r0 = ((400.0 * 10e3) + R)
    vr0 = 0.0
    teta0 = 0.0
    viss = sqrt(((G * Mt) / r0))
    vteta0 = (viss / r0)
    rf = R
    vrf = 0.0
    tetaf = 0.0
    vl = sqrt(((G * Mt) / R))
    vlrad = (vl / r0)
    vtetaf = (1.1 * vlrad)
    t_i = 0.0
    mf_i = 0
    u1_i = 0
    u3_i = 0
    w1_i = 0
    w3_i = 0
    u2_i = 0
    u4_i = 0
    w2_i = 0
    w4_i = 0
    x = 0
    y = 0
    i = 1.0
    u1_im1 = r0
    u2_im1 = vr0
    u3_im1 = teta0
    u4_im1 = vteta0
    w1_im1 = rf
    w2_im1 = vrf
    w3_im1 = tetaf
    w4_im1 = vtetaf
    t_im1 = 0
    mf_im1 = Mf
    while i < 2000000.0:
        t_i = (t_im1 + dt)
        mf_i = (mf_im1 - (A * t_im1))
        u1_i = ((u2_im1 * dt) + u1_im1)
        u3_i = ((u4_im1 * dt) + u3_im1)
        w1_i = ((w2_im1 * dt) + w1_im1)
        w3_i = ((w4_im1 * dt) + w3_im1)
        u2_i = (((-G * (Mt / (u1_im1 * u1_im1))) * dt) + ((u1_im1 * u4_im1) * (u4_im1 * dt)))
        u4_i = (((-2.0 * (u2_im1 * (u4_im1 / u1_im1))) * dt) + u4_im1)
        if mf_im1 > 0.0:
            t = (((A * w2_im1) / (Mf - (A * t_im1))) * dt)
        else:
            t = 0.0
        w2_i = ((((-G * (Mt / (w1_im1 * w1_im1))) * dt) + ((w1_im1 * w4_im1) * (w4_im1 * dt))) + (t + w2_im1))
        if mf_im1 > 0.0:
            t0 = (A * ((w4_im1 / (Mf - (A * t_im1))) * dt))
        else:
            t0 = 0.0
        w4_i = (((-2.0 * (w2_im1 * (w4_im1 / w1_im1))) * dt) + (t0 + w4_im1))
        x = (u1_i * cos(u3_i))
        y = (u1_i * sin(u3_i))
        i = (i + 1.0)
        u1_im1 = u1_i
        u2_im1 = u2_i
        u3_im1 = u3_i
        u4_im1 = u4_i
        w1_im1 = w1_i
        w2_im1 = w2_i
        w3_im1 = w3_i
        w4_im1 = w4_i
        t_im1 = t_i
        mf_im1 = mf_i
    return x

@fpy
def Jacobi_u39_s_Method(a11, a22, a33, a44, b1, b2, b3, b4):
    eps = 0.00000000000000001
    x_n1 = 0.0
    x_n2 = 0.0
    x_n3 = 0.0
    x_n4 = 0.0
    i = 0.0
    e = 1.0
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    x4 = 0.0
    while e > eps:
        x_n1 = ((((b1 / a11) - ((0.1 / a11) * x2)) - ((0.2 / a11) * x3)) + ((0.3 / a11) * x4))
        x_n2 = ((((b2 / a22) - ((0.3 / a22) * x1)) + ((0.1 / a22) * x3)) - ((0.2 / a22) * x4))
        x_n3 = ((((b3 / a33) - ((0.2 / a33) * x1)) + ((0.3 / a33) * x2)) - ((0.1 / a33) * x4))
        x_n4 = ((((b4 / a44) + ((0.1 / a44) * x1)) - ((0.2 / a44) * x2)) - ((0.3 / a44) * x3))
        i = (i + 1.0)
        e = fabs((x_n4 - x4))
        x1 = x_n1
        x2 = x_n2
        x3 = x_n3
        x4 = x_n4
    return x2

@fpy
def Newton_Raphson_u39_s_Method(x0):
    eps = 0.0005
    x_n = 0.0
    e = 1.0
    x = x0
    i = 0.0
    while e > eps and i < 100000:
        f = (((((((x * x) * ((x * x) * x)) - ((10.0 * x) * ((x * x) * x))) + ((40.0 * x) * (x * x))) - ((80.0 * x) * x)) + (80.0 * x)) - 32.0)
        ff = ((((((5.0 * x) * ((x * x) * x)) - ((40.0 * x) * (x * x))) + ((120.0 * x) * x)) - (160.0 * x)) + 80.0)
        x_n = (x - (f / ff))
        e = fabs((x - x_n))
        x = x_n
        i = (i + 1.0)
    return x

@fpy
def Eigenvalue_Computation(a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44, v1, v2, v3, v4):
    eps = 0.0005
    vx = 0
    vy = 0
    vz = 0
    vw = 0
    i = 0.0
    v10 = v1
    v21 = v2
    v32 = v3
    v43 = v4
    e = 1.0
    while e > eps:
        vx = (((a11 * v10) + (a12 * v21)) + ((a13 * v32) + (a14 * v43)))
        vy = (((a21 * v10) + (a22 * v21)) + ((a23 * v32) + (a24 * v43)))
        vz = (((a31 * v10) + (a32 * v21)) + ((a33 * v32) + (a34 * v43)))
        vw = (((a41 * v10) + (a42 * v21)) + ((a43 * v32) + (a44 * v43)))
        i = (i + 1.0)
        v10 = (vx / vw)
        v21 = (vy / vw)
        v32 = (vz / vw)
        v43 = 1.0
        e = fabs((1.0 - v10))
    return v10

@fpy
def Iterative_Gram_Schmidt_Method(Q11, Q12, Q13, Q21, Q22, Q23, Q31, Q32, Q33):
    eps = .000005
    h1 = 0
    h2 = 0
    h3 = 0
    qj1 = Q31
    qj2 = Q32
    qj3 = Q33
    r1 = 0.0
    r2 = 0.0
    r3 = 0.0
    r = (((qj1 * qj1) + (qj2 * qj2)) + (qj3 * qj3))
    rjj = 0
    e = 10.0
    i = 1.0
    rold = sqrt(r)
    while e > eps:
        h1 = (((Q11 * qj1) + (Q21 * qj2)) + (Q31 * qj3))
        h2 = (((Q12 * qj1) + (Q22 * qj2)) + (Q32 * qj3))
        h3 = (((Q13 * qj1) + (Q23 * qj2)) + (Q33 * qj3))
        qj1 = (qj1 - (((Q11 * h1) + (Q12 * h2)) + (Q13 * h3)))
        qj2 = (qj2 - (((Q21 * h1) + (Q22 * h2)) + (Q23 * h3)))
        qj3 = (qj3 - (((Q31 * h1) + (Q32 * h2)) + (Q33 * h3)))
        r1 = (r1 + h1)
        r2 = (r2 + h2)
        r3 = (r3 + h3)
        r = (((qj1 * qj1) + (qj2 * qj2)) + (qj3 * qj3))
        rjj = sqrt(r)
        e = fabs((1.0 - (rjj / rold)))
        i = (i + 1.0)
        rold = rjj
    return qj1

