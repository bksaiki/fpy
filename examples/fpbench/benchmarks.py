from fpy2 import *

@fpy(
    meta={
        'name': 'Arrow-Hurwicz',
        'cite': ['adje-2010', 'bibek-2020'],
        'pre': lambda x, y, u, v: round(0) <= u <= round(1) and round(0) <= v <= round(1) and round(0) <= x <= round(rational(3, 2)) and round(rational(3, 8)) <= y <= round(rational(11, 8)),
    }
)
def Arrow_Hurwicz(x, y, u, v):
    a = round(1)
    b = round(-1)
    c = round(-1)
    r = round(rational(1, 2))
    u0 = u
    v1 = v
    x2 = x
    y3 = y
    while max(abs((x2 - u0)), abs((y3 - v1))) > round(1e-9):
        u0 = x2
        v1 = y3
        x2 = (u0 - (r * ((a * u0) + (b * v1))))
        y3 = max((v1 + ((r / round(2)) * ((b * u0) - c))), round(0))
    return [x2, y3]

@fpy(
    meta={
        'name': 'Euler Oscillator',
        'cite': ['adje-2010', 'bibek-2020'],
        'pre': lambda x, v: round(0) <= x <= round(1) and round(0) <= v <= round(1),
    }
)
def Euler_Oscillator(x, v):
    h = round(0.01)
    v0 = v
    x1 = x
    while True:
        t = ((v0 * (round(1) - h)) - (h * x1))
        t2 = (x1 + (h * v0))
        v0 = t
        x1 = t2
    return [x1, v0]

@fpy(
    meta={
        'name': 'Filter',
        'cite': ['adje-2010', 'bibek-2020'],
        'pre': lambda x, y: round(0) <= x <= round(1) and round(0) <= y <= round(1),
    }
)
def Filter(x, y):
    x0 = x
    y1 = y
    while True:
        x0 = ((round(rational(3, 4)) * x0) - (round(rational(1, 8)) * y1))
        y1 = x0
    return y1

@fpy(
    meta={
        'name': 'Symplectic Oscillator',
        'cite': ['adje-2010', 'bibek-2020'],
        'pre': lambda x, v: round(0) <= x <= round(1) and v <= round(0) <= round(1),
    }
)
def Symplectic_Oscillator(x, v):
    tau = round(0.1)
    x0 = x
    v1 = v
    while v1 >= round(rational(1, 2)):
        x0 = (((round(1) - (tau / round(2))) * x0) + ((tau - (pow(tau, round(3)) / round(4))) * v1))
        v1 = ((-tau * x0) + ((round(1) - (tau / round(2))) * v1))
    return [x0, v1]

@fpy(
    meta={
        'name': 'Circle',
        'cite': ['bibek-2020'],
        'pre': lambda x, y: round(rational(-1, 2)) <= x <= round(rational(1, 2)) and round(rational(-1, 2)) <= y <= round(rational(1, 2)),
    }
)
def Circle(x, y):
    d = round(0)
    x0 = x
    y1 = y
    while True:
        d = (((round(0.1) + (x0 * x0)) + (y1 * y1)) / round(2))
        x0 = (x0 * d)
        y1 = (y1 * d)
    return [x0, y1]

@fpy(
    meta={
        'name': 'Flower',
        'cite': ['bibek-2020'],
        'pre': lambda x, y: round(-0.1) <= x <= round(0.1) and round(-0.1) <= y <= round(0.1),
    }
)
def Flower(x, y):
    x0 = x
    y1 = y
    while (((round(0.15) + (x0 * x0)) + (y1 * y1)) / round(2)) > round(0.1):
        t = (x0 / (((round(0.15) + (x0 * x0)) + (y1 * y1)) / round(2)))
        t2 = (y1 * (((round(0.15) + (x0 * x0)) + (y1 * y1)) / round(2)))
        x0 = t
        y1 = t2
    return [x0, y1]

@fpy(
    meta={
        'name': 'carthesianToPolar, radius',
        'pre': lambda x, y: round(1) <= x <= round(100) and round(1) <= y <= round(100),
        'spec': lambda x, y: hypot(x, y),
    }
)
def carthesianToPolar_u44__radius(x, y):
    return sqrt(((x * x) + (y * y)))

@fpy(
    meta={
        'name': 'carthesianToPolar, theta',
        'pre': lambda x, y: round(1) <= x <= round(100) and round(1) <= y <= round(100),
        'spec': lambda x, y: (atan2(y, x) * (round(180) / const_pi())),
    }
)
def carthesianToPolar_u44__theta(x, y):
    pi = round(3.14159265359)
    radiant = atan((y / x))
    return (radiant * (round(180.0) / pi))

@fpy(
    meta={
        'name': 'polarToCarthesian, x',
        'pre': lambda radius, theta: round(1) <= radius <= round(10) and round(0) <= theta <= round(360),
        'spec': lambda radius, theta: (radius * cos((theta * (round(180) / const_pi())))),
    }
)
def polarToCarthesian_u44__x(radius, theta):
    pi = round(3.14159265359)
    radiant = (theta * (pi / round(180.0)))
    return (radius * cos(radiant))

@fpy(
    meta={
        'name': 'polarToCarthesian, y',
        'pre': lambda radius, theta: round(1) <= radius <= round(10) and round(0) <= theta <= round(360),
        'spec': lambda radius, theta: (radius * sin((theta * (round(180) / const_pi())))),
    }
)
def polarToCarthesian_u44__y(radius, theta):
    pi = round(3.14159265359)
    radiant = (theta * (pi / round(180.0)))
    return (radius * sin(radiant))

@fpy(
    meta={
        'name': 'instantaneousCurrent',
        'pre': lambda t, resistance, frequency, inductance, maxVoltage: round(0) <= t <= round(300.0) and round(1) <= resistance <= round(50) and round(1) <= frequency <= round(100) and round(0.001) <= inductance <= round(0.004) and round(1) <= maxVoltage <= round(12),
    }
)
def instantaneousCurrent(t, resistance, frequency, inductance, maxVoltage):
    pi = round(3.14159265359)
    impedance_re = resistance
    impedance_im = (((round(2) * pi) * frequency) * inductance)
    denom = ((impedance_re * impedance_re) + (impedance_im * impedance_im))
    current_re = ((maxVoltage * impedance_re) / denom)
    current_im = (-(maxVoltage * impedance_im) / denom)
    maxCurrent = sqrt(((current_re * current_re) + (current_im * current_im)))
    theta = atan((current_im / current_re))
    return (maxCurrent * cos(((((round(2) * pi) * frequency) * t) + theta)))

@fpy(
    meta={
        'name': 'matrixDeterminant',
        'pre': lambda a, b, c, d, e, f, g, h, i: round(-10) <= a <= round(10) and round(-10) <= b <= round(10) and round(-10) <= c <= round(10) and round(-10) <= d <= round(10) and round(-10) <= e <= round(10) and round(-10) <= f <= round(10) and round(-10) <= g <= round(10) and round(-10) <= h <= round(10) and round(-10) <= i <= round(10),
    }
)
def matrixDeterminant(a, b, c, d, e, f, g, h, i):
    return (((((a * e) * i) + ((b * f) * g)) + ((c * d) * h)) - ((((c * e) * g) + ((b * d) * i)) + ((a * f) * h)))

@fpy(
    meta={
        'name': 'matrixDeterminant2',
        'pre': lambda a, b, c, d, e, f, g, h, i: round(-10) <= a <= round(10) and round(-10) <= b <= round(10) and round(-10) <= c <= round(10) and round(-10) <= d <= round(10) and round(-10) <= e <= round(10) and round(-10) <= f <= round(10) and round(-10) <= g <= round(10) and round(-10) <= h <= round(10) and round(-10) <= i <= round(10),
    }
)
def matrixDeterminant2(a, b, c, d, e, f, g, h, i):
    return (((a * (e * i)) + ((g * (b * f)) + (c * (d * h)))) - ((e * (c * g)) + ((i * (b * d)) + (a * (f * h)))))

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'intro-example-mixed',
        'description': 'Generated by FPTaylor',
        'pre': lambda t: round(1) <= t <= round(999),
    }
)
def intro_example_mixed(t):
    with IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
        with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
            t0 = (t + round(1))
        t1 = (t / t0)
    return round(t1)

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'delta4',
        'description': 'Generated by FPTaylor',
        'pre': lambda x1, x2, x3, x4, x5, x6: round(4) <= x1 <= round(rational(3969, 625)) and round(4) <= x2 <= round(rational(3969, 625)) and round(4) <= x3 <= round(rational(3969, 625)) and round(4) <= x4 <= round(rational(3969, 625)) and round(4) <= x5 <= round(rational(3969, 625)) and round(4) <= x6 <= round(rational(3969, 625)),
    }
)
def delta4(x1, x2, x3, x4, x5, x6):
    return ((((((-x2 * x3) - (x1 * x4)) + (x2 * x5)) + (x3 * x6)) - (x5 * x6)) + (x1 * (((((-x1 + x2) + x3) - x4) + x5) + x6)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'delta',
        'description': 'Generated by FPTaylor',
        'pre': lambda x1, x2, x3, x4, x5, x6: round(4) <= x1 <= round(rational(3969, 625)) and round(4) <= x2 <= round(rational(3969, 625)) and round(4) <= x3 <= round(rational(3969, 625)) and round(4) <= x4 <= round(rational(3969, 625)) and round(4) <= x5 <= round(rational(3969, 625)) and round(4) <= x6 <= round(rational(3969, 625)),
    }
)
def delta(x1, x2, x3, x4, x5, x6):
    return ((((((((x1 * x4) * (((((-x1 + x2) + x3) - x4) + x5) + x6)) + ((x2 * x5) * (((((x1 - x2) + x3) + x4) - x5) + x6))) + ((x3 * x6) * (((((x1 + x2) - x3) + x4) + x5) - x6))) + ((-x2 * x3) * x4)) + ((-x1 * x3) * x5)) + ((-x1 * x2) * x6)) + ((-x4 * x5) * x6))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'sqrt_add',
        'description': 'Generated by FPTaylor',
        'pre': lambda x: round(1) <= x <= round(1000),
    }
)
def sqrt_add(x):
    return (round(1) / (sqrt((x + round(1))) + sqrt(x)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'exp1x',
        'description': 'Generated by FPTaylor',
        'pre': lambda x: round(rational(1, 100)) <= x <= round(rational(1, 2)),
    }
)
def exp1x(x):
    return ((exp(x) - round(1)) / x)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'exp1x_32',
        'description': 'Generated by FPTaylor',
        'pre': lambda x: round(rational(1, 100)) <= x <= round(rational(1, 2)),
    }
)
def exp1x_32(x):
    return ((exp(x) - round(1)) / x)

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'floudas',
        'description': 'Generated by FPTaylor',
        'pre': lambda x1, x2: round(0) <= x1 <= round(2) and round(0) <= x2 <= round(3) and (x1 + x2) <= round(2),
    }
)
def floudas(x1, x2):
    return (x1 + x2)

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'exp1x_log',
        'description': 'Generated by FPTaylor',
        'pre': lambda x: round(rational(1, 100)) <= x <= round(rational(1, 2)),
    }
)
def exp1x_log(x):
    return ((exp(x) - round(1)) / log(exp(x)))

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'x_by_xy',
        'description': 'Generated by FPTaylor',
        'pre': lambda x, y: round(1) <= x <= round(4) and round(1) <= y <= round(4),
    }
)
def x_by_xy(x, y):
    return (x / (x + y))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'hypot',
        'description': 'Generated by FPTaylor',
        'pre': lambda x1, x2: round(1) <= x1 <= round(100) and round(1) <= x2 <= round(100),
    }
)
def hypot(x1, x2):
    return sqrt(((x1 * x1) + (x2 * x2)))

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'hypot32',
        'description': 'Generated by FPTaylor',
        'pre': lambda x1, x2: round(1) <= x1 <= round(100) and round(1) <= x2 <= round(100),
    }
)
def hypot32(x1, x2):
    return sqrt(((x1 * x1) + (x2 * x2)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'logexp',
        'description': 'Generated by FPTaylor',
        'pre': lambda x: round(-8) <= x <= round(8),
    }
)
def logexp(x):
    return log((round(1) + exp(x)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'sum',
        'description': 'Generated by FPTaylor',
        'pre': lambda x0, x1, x2: round(1) <= x0 <= round(2) and round(1) <= x1 <= round(2) and round(1) <= x2 <= round(2),
    }
)
def sum(x0, x1, x2):
    p0 = ((x0 + x1) - x2)
    p1 = ((x1 + x2) - x0)
    p2 = ((x2 + x0) - x1)
    return ((p0 + p1) + p2)

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'nonlin1',
        'description': 'Generated by FPTaylor',
        'pre': lambda z: round(0) <= z <= round(999),
    }
)
def nonlin1(z):
    return (z / (z + round(1)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'nonlin2',
        'description': 'Generated by FPTaylor',
        'pre': lambda x, y: round(rational(1001, 1000)) <= x <= round(2) and round(rational(1001, 1000)) <= y <= round(2),
    }
)
def nonlin2(x, y):
    t = (x * y)
    return ((t - round(1)) / ((t * t) - round(1)))

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'i4',
        'description': 'Generated by FPTaylor',
        'pre': lambda x, y: round(rational(1, 10)) <= x <= round(10) and round(-5) <= y <= round(5),
    }
)
def i4(x, y):
    return sqrt((x + (y * y)))

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'i6',
        'description': 'Generated by FPTaylor',
        'pre': lambda x, y: round(rational(1, 10)) <= x <= round(10) and round(-5) <= y <= round(5),
    }
)
def i6(x, y):
    return sin((x * y))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'himmilbeau',
        'description': 'Generated by FPTaylor',
        'pre': lambda x1, x2: round(-5) <= x1 <= round(5) and round(-5) <= x2 <= round(5),
    }
)
def himmilbeau(x1, x2):
    a = (((x1 * x1) + x2) - round(11))
    b = ((x1 + (x2 * x2)) - round(7))
    return ((a * a) + (b * b))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'logexp',
        'cite': ['solovyev-et-al-2015'],
        'pre': lambda x: round(-8) <= x <= round(8),
    }
)
def logexp(x):
    e = exp(x)
    return log((round(1) + e))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'sphere',
        'cite': ['solovyev-et-al-2015'],
        'pre': lambda x, r, lat, lon: round(-10) <= x <= round(10) and round(0) <= r <= round(10) and round(-1.570796) <= lat <= round(1.570796) and round(-3.14159265) <= lon <= round(3.14159265),
    }
)
def sphere(x, r, lat, lon):
    sinLat = sin(lat)
    cosLon = cos(lon)
    return (x + ((r * sinLat) * cosLon))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'azimuth',
        'cite': ['solovyev-et-al-2015'],
        'pre': lambda lat1, lat2, lon1, lon2: round(0) <= lat1 <= round(0.4) and round(0.5) <= lat2 <= round(1) and round(0) <= lon1 <= round(3.14159265) and round(-3.14159265) <= lon2 <= round(-0.5),
    }
)
def azimuth(lat1, lat2, lon1, lon2):
    dLon = (lon2 - lon1)
    s_lat1 = sin(lat1)
    c_lat1 = cos(lat1)
    s_lat2 = sin(lat2)
    c_lat2 = cos(lat2)
    s_dLon = sin(dLon)
    c_dLon = cos(dLon)
    return atan(((c_lat2 * s_dLon) / ((c_lat1 * s_lat2) - ((s_lat1 * c_lat2) * c_dLon))))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'floudas1',
        'pre': lambda x1, x2, x3, x4, x5, x6: round(0) <= x1 <= round(6) and round(0) <= x2 <= round(6) and round(1) <= x3 <= round(5) and round(0) <= x4 <= round(6) and round(0) <= x5 <= round(6) and round(0) <= x6 <= round(10) and ((((x3 - round(3)) * (x3 - round(3))) + x4) - round(4)) >= round(0) and ((((x5 - round(3)) * (x5 - round(3))) + x6) - round(4)) >= round(0) and ((round(2) - x1) + (round(3) * x2)) >= round(0) and ((round(2) + x1) - x2) >= round(0) and ((round(6) - x1) - x2) >= round(0) and ((x1 + x2) - round(2)) >= round(0),
    }
)
def floudas1(x1, x2, x3, x4, x5, x6):
    return ((((((round(-25) * ((x1 - round(2)) * (x1 - round(2)))) - ((x2 - round(2)) * (x2 - round(2)))) - ((x3 - round(1)) * (x3 - round(1)))) - ((x4 - round(4)) * (x4 - round(4)))) - ((x5 - round(1)) * (x5 - round(1)))) - ((x6 - round(4)) * (x6 - round(4))))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'floudas2',
        'pre': lambda x1, x2: round(0) <= x1 <= round(3) and round(0) <= x2 <= round(4) and ((((round(2) * ((x1 * x1) * (x1 * x1))) - ((round(8) * (x1 * x1)) * x1)) + ((round(8) * x1) * x1)) - x2) >= round(0) and ((((((round(4) * ((x1 * x1) * (x1 * x1))) - ((round(32) * (x1 * x1)) * x1)) + ((round(88) * x1) * x1)) - (round(96) * x1)) + round(36)) - x2) >= round(0),
    }
)
def floudas2(x1, x2):
    return (-x1 - x2)

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'floudas3',
        'pre': lambda x1, x2: round(0) <= x1 <= round(2) and round(0) <= x2 <= round(3) and ((round(-2) * ((x1 * x1) * (x1 * x1))) + round(2)) >= x2,
    }
)
def floudas3(x1, x2):
    return (((round(-12) * x1) - (round(7) * x2)) + (x2 * x2))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'hartman3',
        'pre': lambda x1, x2, x3: round(0) <= x1 <= round(1) and round(0) <= x2 <= round(1) and round(0) <= x3 <= round(1),
    }
)
def hartman3(x1, x2, x3):
    e1 = (((round(3.0) * ((x1 - round(0.3689)) * (x1 - round(0.3689)))) + (round(10.0) * ((x2 - round(0.117)) * (x2 - round(0.117))))) + (round(30.0) * ((x3 - round(0.2673)) * (x3 - round(0.2673)))))
    e2 = (((round(0.1) * ((x1 - round(0.4699)) * (x1 - round(0.4699)))) + (round(10.0) * ((x2 - round(0.4387)) * (x2 - round(0.4387))))) + (round(35.0) * ((x3 - round(0.747)) * (x3 - round(0.747)))))
    e3 = (((round(3.0) * ((x1 - round(0.1091)) * (x1 - round(0.1091)))) + (round(10.0) * ((x2 - round(0.8732)) * (x2 - round(0.8732))))) + (round(30.0) * ((x3 - round(0.5547)) * (x3 - round(0.5547)))))
    e4 = (((round(0.1) * ((x1 - round(0.03815)) * (x1 - round(0.03815)))) + (round(10.0) * ((x2 - round(0.5743)) * (x2 - round(0.5743))))) + (round(35.0) * ((x3 - round(0.8828)) * (x3 - round(0.8828)))))
    exp1 = exp(-e1)
    exp2 = exp(-e2)
    exp3 = exp(-e3)
    exp4 = exp(-e4)
    return -((((round(1.0) * exp1) + (round(1.2) * exp2)) + (round(3.0) * exp3)) + (round(3.2) * exp4))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'hartman6',
        'pre': lambda x1, x2, x3, x4, x5, x6: round(0) <= x1 <= round(1) and round(0) <= x2 <= round(1) and round(0) <= x3 <= round(1) and round(0) <= x4 <= round(1) and round(0) <= x5 <= round(1) and round(0) <= x6 <= round(1),
    }
)
def hartman6(x1, x2, x3, x4, x5, x6):
    e1 = ((((((round(10.0) * ((x1 - round(0.1312)) * (x1 - round(0.1312)))) + (round(3.0) * ((x2 - round(0.1696)) * (x2 - round(0.1696))))) + (round(17.0) * ((x3 - round(0.5569)) * (x3 - round(0.5569))))) + (round(3.5) * ((x4 - round(0.0124)) * (x4 - round(0.0124))))) + (round(1.7) * ((x5 - round(0.8283)) * (x5 - round(0.8283))))) + (round(8.0) * ((x6 - round(0.5886)) * (x6 - round(0.5886)))))
    e2 = ((((((round(0.05) * ((x1 - round(0.2329)) * (x1 - round(0.2329)))) + (round(10.0) * ((x2 - round(0.4135)) * (x2 - round(0.4135))))) + (round(17.0) * ((x3 - round(0.8307)) * (x3 - round(0.8307))))) + (round(0.1) * ((x4 - round(0.3736)) * (x4 - round(0.3736))))) + (round(8.0) * ((x5 - round(0.1004)) * (x5 - round(0.1004))))) + (round(14.0) * ((x6 - round(0.9991)) * (x6 - round(0.9991)))))
    e3 = ((((((round(3.0) * ((x1 - round(0.2348)) * (x1 - round(0.2348)))) + (round(3.5) * ((x2 - round(0.1451)) * (x2 - round(0.1451))))) + (round(1.7) * ((x3 - round(0.3522)) * (x3 - round(0.3522))))) + (round(10.0) * ((x4 - round(0.2883)) * (x4 - round(0.2883))))) + (round(17.0) * ((x5 - round(0.3047)) * (x5 - round(0.3047))))) + (round(8.0) * ((x6 - round(0.665)) * (x6 - round(0.665)))))
    e4 = ((((((round(17.0) * ((x1 - round(0.4047)) * (x1 - round(0.4047)))) + (round(8.0) * ((x2 - round(0.8828)) * (x2 - round(0.8828))))) + (round(0.05) * ((x3 - round(0.8732)) * (x3 - round(0.8732))))) + (round(10.0) * ((x4 - round(0.5743)) * (x4 - round(0.5743))))) + (round(0.1) * ((x5 - round(0.1091)) * (x5 - round(0.1091))))) + (round(14.0) * ((x6 - round(0.0381)) * (x6 - round(0.0381)))))
    exp1 = exp(-e1)
    exp2 = exp(-e2)
    exp3 = exp(-e3)
    exp4 = exp(-e4)
    return -((((round(1.0) * exp1) + (round(1.2) * exp2)) + (round(3.0) * exp3)) + (round(3.2) * exp4))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'kepler0',
        'pre': lambda x1, x2, x3, x4, x5, x6: round(4) <= x1 <= round(6.36) and round(4) <= x2 <= round(6.36) and round(4) <= x3 <= round(6.36) and round(4) <= x4 <= round(6.36) and round(4) <= x5 <= round(6.36) and round(4) <= x6 <= round(6.36),
    }
)
def kepler0(x1, x2, x3, x4, x5, x6):
    return (((((x2 * x5) + (x3 * x6)) - (x2 * x3)) - (x5 * x6)) + (x1 * (((((-x1 + x2) + x3) - x4) + x5) + x6)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'kepler1',
        'pre': lambda x1, x2, x3, x4: round(4) <= x1 <= round(6.36) and round(4) <= x2 <= round(6.36) and round(4) <= x3 <= round(6.36) and round(4) <= x4 <= round(6.36),
    }
)
def kepler1(x1, x2, x3, x4):
    return ((((((((x1 * x4) * (((-x1 + x2) + x3) - x4)) + (x2 * (((x1 - x2) + x3) + x4))) + (x3 * (((x1 + x2) - x3) + x4))) - ((x2 * x3) * x4)) - (x1 * x3)) - (x1 * x2)) - x4)

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'kepler2',
        'pre': lambda x1, x2, x3, x4, x5, x6: round(4) <= x1 <= round(6.36) and round(4) <= x2 <= round(6.36) and round(4) <= x3 <= round(6.36) and round(4) <= x4 <= round(6.36) and round(4) <= x5 <= round(6.36) and round(4) <= x6 <= round(6.36),
    }
)
def kepler2(x1, x2, x3, x4, x5, x6):
    return ((((((((x1 * x4) * (((((-x1 + x2) + x3) - x4) + x5) + x6)) + ((x2 * x5) * (((((x1 - x2) + x3) + x4) - x5) + x6))) + ((x3 * x6) * (((((x1 + x2) - x3) + x4) + x5) - x6))) - ((x2 * x3) * x4)) - ((x1 * x3) * x5)) - ((x1 * x2) * x6)) - ((x4 * x5) * x6))

@fpy(
    meta={
        'name': 'intro-example',
        'cite': ['solovyev-et-al-2015'],
        'pre': lambda t: round(0) <= t <= round(999),
    }
)
def intro_example(t):
    return (t / (t + round(1)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'sec4-example',
        'cite': ['solovyev-et-al-2015'],
        'pre': lambda x, y: round(1.001) <= x <= round(2) and round(1.001) <= y <= round(2),
    }
)
def sec4_example(x, y):
    t = (x * y)
    return ((t - round(1)) / ((t * t) - round(1)))

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'test01_sum3',
        'pre': lambda x0, x1, x2: round(1) < x0 < round(2) and round(1) < x1 < round(2) and round(1) < x2 < round(2),
    }
)
def test01_sum3(x0, x1, x2):
    p0 = ((x0 + x1) - x2)
    p1 = ((x1 + x2) - x0)
    p2 = ((x2 + x0) - x1)
    return ((p0 + p1) + p2)

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'test02_sum8',
        'pre': lambda x0, x1, x2, x3, x4, x5, x6, x7: round(1) < x0 < round(2) and round(1) < x1 < round(2) and round(1) < x2 < round(2) and round(1) < x3 < round(2) and round(1) < x4 < round(2) and round(1) < x5 < round(2) and round(1) < x6 < round(2) and round(1) < x7 < round(2),
    }
)
def test02_sum8(x0, x1, x2, x3, x4, x5, x6, x7):
    return (((((((x0 + x1) + x2) + x3) + x4) + x5) + x6) + x7)

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'test03_nonlin2',
        'pre': lambda x, y: round(0) < x < round(1) and round(-1) < y < round(-0.1),
    }
)
def test03_nonlin2(x, y):
    return ((x + y) / (x - y))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'test04_dqmom9',
        'pre': lambda m0, m1, m2, w0, w1, w2, a0, a1, a2: round(-1) < m0 < round(1) and round(-1) < m1 < round(1) and round(-1) < m2 < round(1) and round(0.00001) < w0 < round(1) and round(0.00001) < w1 < round(1) and round(0.00001) < w2 < round(1) and round(0.00001) < a0 < round(1) and round(0.00001) < a1 < round(1) and round(0.00001) < a2 < round(1),
    }
)
def test04_dqmom9(m0, m1, m2, w0, w1, w2, a0, a1, a2):
    v2 = ((w2 * (round(0) - m2)) * (round(-3) * ((round(1) * (a2 / w2)) * (a2 / w2))))
    v1 = ((w1 * (round(0) - m1)) * (round(-3) * ((round(1) * (a1 / w1)) * (a1 / w1))))
    v0 = ((w0 * (round(0) - m0)) * (round(-3) * ((round(1) * (a0 / w0)) * (a0 / w0))))
    return (round(0.0) + ((v0 * round(1)) + ((v1 * round(1)) + ((v2 * round(1)) + round(0.0)))))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'test05_nonlin1, r4',
        'pre': lambda x: round(1.00001) < x < round(2),
    }
)
def test05_nonlin1_u44__r4(x):
    r1 = (x - round(1))
    r2 = (x * x)
    return (r1 / (r2 - round(1)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'test05_nonlin1, test2',
        'pre': lambda x: round(1.00001) < x < round(2),
    }
)
def test05_nonlin1_u44__test2(x):
    return (round(1) / (x + round(1)))

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'test06_sums4, sum1',
        'pre': lambda x0, x1, x2, x3: round(-1e-5) < x0 < round(1.00001) and round(0) < x1 < round(1) and round(0) < x2 < round(1) and round(0) < x3 < round(1),
    }
)
def test06_sums4_u44__sum1(x0, x1, x2, x3):
    return (((x0 + x1) + x2) + x3)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'test06_sums4, sum2',
        'pre': lambda x0, x1, x2, x3: round(-1e-5) < x0 < round(1.00001) and round(0) < x1 < round(1) and round(0) < x2 < round(1) and round(0) < x3 < round(1),
    }
)
def test06_sums4_u44__sum2(x0, x1, x2, x3):
    return ((x0 + x1) + (x2 + x3))

@fpy(
    meta={
        'name': 'NMSE example 3.1',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda x: x >= round(0),
    }
)
def NMSE_example_3_u46_1(x):
    return (sqrt((x + round(1))) - sqrt(x))

@fpy(
    meta={
        'name': 'NMSE example 3.3',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
    }
)
def NMSE_example_3_u46_3(x, eps):
    return (sin((x + eps)) - sin(x))

@fpy(
    meta={
        'name': 'NMSE example 3.4',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda x: x != round(0),
    }
)
def NMSE_example_3_u46_4(x):
    return ((round(1) - cos(x)) / sin(x))

@fpy(
    meta={
        'name': 'NMSE example 3.5',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
    }
)
def NMSE_example_3_u46_5(N):
    return (atan((N + round(1))) - atan(N))

@fpy(
    meta={
        'name': 'NMSE example 3.6',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda x: x >= round(0),
    }
)
def NMSE_example_3_u46_6(x):
    return ((round(1) / sqrt(x)) - (round(1) / sqrt((x + round(1)))))

@fpy(
    meta={
        'name': 'NMSE problem 3.3.1',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda x: x != round(0),
    }
)
def NMSE_problem_3_u46_3_u46_1(x):
    return ((round(1) / (x + round(1))) - (round(1) / x))

@fpy(
    meta={
        'name': 'NMSE problem 3.3.2',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
    }
)
def NMSE_problem_3_u46_3_u46_2(x, eps):
    return (tan((x + eps)) - tan(x))

@fpy(
    meta={
        'name': 'NMSE problem 3.3.3',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda x: x != round(0) != round(1) != round(-1),
    }
)
def NMSE_problem_3_u46_3_u46_3(x):
    return (((round(1) / (x + round(1))) - (round(2) / x)) + (round(1) / (x - round(1))))

@fpy(
    meta={
        'name': 'NMSE problem 3.3.4',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda x: x >= round(0),
    }
)
def NMSE_problem_3_u46_3_u46_4(x):
    return (pow((x + round(1)), (round(1) / round(3))) - pow(x, (round(1) / round(3))))

@fpy(
    meta={
        'name': 'NMSE problem 3.3.5',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
    }
)
def NMSE_problem_3_u46_3_u46_5(x, eps):
    return (cos((x + eps)) - cos(x))

@fpy(
    meta={
        'name': 'NMSE problem 3.3.6',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda N: N > round(0),
    }
)
def NMSE_problem_3_u46_3_u46_6(N):
    return (log((N + round(1))) - log(N))

@fpy(
    meta={
        'name': 'NMSE problem 3.3.7',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
    }
)
def NMSE_problem_3_u46_3_u46_7(x):
    return ((exp(x) - round(2)) + exp(-x))

@fpy(
    meta={
        'name': 'NMSE p42, positive',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda a, b, c: (b * b) >= (round(4) * (a * c)) and a != round(0),
    }
)
def NMSE_p42_u44__positive(a, b, c):
    return ((-b + sqrt(((b * b) - (round(4) * (a * c))))) / (round(2) * a))

@fpy(
    meta={
        'name': 'NMSE p42, negative',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda a, b, c: (b * b) >= (round(4) * (a * c)) and a != round(0),
    }
)
def NMSE_p42_u44__negative(a, b, c):
    return ((-b - sqrt(((b * b) - (round(4) * (a * c))))) / (round(2) * a))

@fpy(
    meta={
        'name': 'NMSE problem 3.2.1, positive',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda a, b2, c: (b2 * b2) >= (a * c) and a != round(0),
    }
)
def NMSE_problem_3_u46_2_u46_1_u44__positive(a, b2, c):
    return ((-b2 + sqrt(((b2 * b2) - (a * c)))) / a)

@fpy(
    meta={
        'name': 'NMSE problem 3.2.1, negative',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda a, b2, c: (b2 * b2) >= (a * c) and a != round(0),
    }
)
def NMSE_problem_3_u46_2_u46_1_u44__negative(a, b2, c):
    return ((-b2 - sqrt(((b2 * b2) - (a * c)))) / a)

@fpy(
    meta={
        'name': 'NMSE example 3.7',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
    }
)
def NMSE_example_3_u46_7(x):
    return (exp(x) - round(1))

@fpy(
    meta={
        'name': 'NMSE example 3.8',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda N: N > round(0),
    }
)
def NMSE_example_3_u46_8(N):
    return ((((N + round(1)) * log((N + round(1)))) - (N * log(N))) - round(1))

@fpy(
    meta={
        'name': 'NMSE example 3.9',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda x: x != round(0),
    }
)
def NMSE_example_3_u46_9(x):
    return ((round(1) / x) - (round(1) / tan(x)))

@fpy(
    meta={
        'name': 'NMSE example 3.10',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'daisy_pre': ['<', '-0.99', 'x', '0.99'],
        'pre': lambda x: round(-1) < x < round(1),
    }
)
def NMSE_example_3_u46_10(x):
    return (log((round(1) - x)) / log((round(1) + x)))

@fpy(
    meta={
        'name': 'NMSE problem 3.4.1',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda x: x != round(0),
    }
)
def NMSE_problem_3_u46_4_u46_1(x):
    return ((round(1) - cos(x)) / (x * x))

@fpy(
    meta={
        'name': 'NMSE problem 3.4.2',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda a, b, eps: eps != round(0),
    }
)
def NMSE_problem_3_u46_4_u46_2(a, b, eps):
    return ((eps * (exp(((a + b) * eps)) - round(1))) / ((exp((a * eps)) - round(1)) * (exp((b * eps)) - round(1))))

@fpy(
    meta={
        'name': 'NMSE problem 3.4.3',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda eps: round(-1) < eps < round(1),
    }
)
def NMSE_problem_3_u46_4_u46_3(eps):
    return log(((round(1) - eps) / (round(1) + eps)))

@fpy(
    meta={
        'name': 'NMSE problem 3.4.4',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda x: x != round(0),
    }
)
def NMSE_problem_3_u46_4_u46_4(x):
    return sqrt(((exp((round(2) * x)) - round(1)) / (exp(x) - round(1))))

@fpy(
    meta={
        'name': 'NMSE problem 3.4.5',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda x: x != round(0),
    }
)
def NMSE_problem_3_u46_4_u46_5(x):
    return ((x - sin(x)) / (x - tan(x)))

@fpy(
    meta={
        'name': 'NMSE problem 3.4.6',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda x, n: x >= round(0),
    }
)
def NMSE_problem_3_u46_4_u46_6(x, n):
    return (pow((x + round(1)), (round(1) / n)) - pow(x, (round(1) / n)))

@fpy(
    meta={
        'name': 'NMSE section 3.5',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
    }
)
def NMSE_section_3_u46_5(a, x):
    return (exp((a * x)) - round(1))

@fpy(
    meta={
        'name': 'NMSE section 3.11',
        'cite': ['hamming-1987', 'herbie-2015'],
        'fpbench_domain': 'textbook',
        'pre': lambda x: x != round(0),
    }
)
def NMSE_section_3_u46_11(x):
    return (exp(x) / (exp(x) - round(1)))

@fpy(
    meta={
        'name': 'Complex square root',
        'cite': ['herbie-2015'],
    }
)
def Complex_square_root(re, im):
    return (round(0.5) * sqrt((round(2.0) * (sqrt(((re * re) + (im * im))) + re))))

@fpy(
    meta={
        'name': 'Complex sine and cosine',
        'cite': ['herbie-2015'],
    }
)
def Complex_sine_and_cosine(re, im):
    return ((round(0.5) * sin(re)) * (exp(-im) - exp(im)))

@fpy(
    meta={
        'name': 'Probabilities in a clustering algorithm',
        'cite': ['herbie-2015'],
        'pre': lambda cp, cn, t, s: round(0) < cp and round(0) < cn,
    }
)
def Probabilities_in_a_clustering_algorithm(cp, cn, t, s):
    return ((pow((round(1) / (round(1) + exp(-s))), cp) * pow((round(1) - (round(1) / (round(1) + exp(-s)))), cn)) / (pow((round(1) / (round(1) + exp(-t))), cp) * pow((round(1) - (round(1) / (round(1) + exp(-t)))), cn)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'arclength of a wiggly function',
        'cite': ['precimonious-2013'],
        'pre': lambda n: n >= round(0),
        'fpbench_allowed_ulps': '10',
        'fpbench_pre_override': ['<=', '10', 'x', '200'],
    }
)
def arclength_of_a_wiggly_function(n):
    dppi = const_pi()
    h = (dppi / n)
    t1 = round(0)
    t2 = round(0)
    s1 = round(0)
    t0 = t1
    with MPFixedContext(nmin=-1, rm=RoundingMode.RNE, num_randbits=0, enable_nan=False, enable_inf=False, nan_value=None, inf_value=None):
        t = round(1)
    i = t
    while i <= n:
        x = (i * h)
        with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
            t3 = round(1)
        d1 = t3
        t4 = x
        with MPFixedContext(nmin=-1, rm=RoundingMode.RNE, num_randbits=0, enable_nan=False, enable_inf=False, nan_value=None, inf_value=None):
            t5 = round(1)
        k = t5
        while k <= round(5):
            with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
                t6 = (d1 * round(2))
            d1 = t6
            t4 = (t4 + (sin((d1 * x)) / d1))
            with MPFixedContext(nmin=-1, rm=RoundingMode.RNE, num_randbits=0, enable_nan=False, enable_inf=False, nan_value=None, inf_value=None):
                t7 = (k + round(1))
            k = t7
        t2 = t4
        s0 = sqrt(((h * h) + ((t2 - t0) * (t2 - t0))))
        with IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
            t8 = (s1 + s0)
        s1 = t8
        t0 = t2
        with MPFixedContext(nmin=-1, rm=RoundingMode.RNE, num_randbits=0, enable_nan=False, enable_inf=False, nan_value=None, inf_value=None):
            t9 = (i + round(1))
        i = t9
    return s1

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'arclength of a wiggly function (old version)',
        'cite': ['precimonious-2013'],
        'pre': lambda n: n >= round(0),
        'fpbench_pre_override': ['<=', '10', 'x', '50'],
    }
)
def arclength_of_a_wiggly_function__u40_old_version_u41_(n):
    dppi = acos(round(-1.0))
    h = (dppi / n)
    s1 = round(0.0)
    t1 = round(0.0)
    i = round(1)
    while i <= n:
        x = (i * h)
        with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
            t = round(2.0)
        d0 = t
        t0 = x
        k = round(1)
        while k <= round(5):
            with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
                t2 = (round(2.0) * d0)
            t3 = t2
            t4 = (t0 + (sin((d0 * x)) / d0))
            t5 = (k + round(1))
            d0 = t3
            t0 = t4
            k = t5
        t6 = t0
        s0 = sqrt(((h * h) + ((t6 - t1) * (t6 - t1))))
        with IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
            t7 = (s1 + s0)
        t8 = t7
        x9 = (i * h)
        with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
            t10 = round(2.0)
        d11 = t10
        t12 = x9
        k13 = round(1)
        while k13 <= round(5):
            with IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
                t14 = (round(2.0) * d11)
            t15 = t14
            t16 = (t12 + (sin((d11 * x9)) / d11))
            t17 = (k13 + round(1))
            d11 = t15
            t12 = t16
            k13 = t17
        t18 = t12
        t19 = t18
        t20 = (i + round(1))
        s1 = t8
        t1 = t19
        i = t20
    return s1

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'doppler1',
        'cite': ['darulova-kuncak-2014'],
        'fpbench_domain': 'science',
        'pre': lambda u, v, T: round(-100) <= u <= round(100) and round(20) <= v <= round(20000) and round(-30) <= T <= round(50),
        'rosa_ensuring': '1e-12',
    }
)
def doppler1(u, v, T):
    t1 = (round(331.4) + (round(0.6) * T))
    return ((-t1 * v) / ((t1 + u) * (t1 + u)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'doppler2',
        'cite': ['darulova-kuncak-2014'],
        'fpbench_domain': 'science',
        'pre': lambda u, v, T: round(-125) <= u <= round(125) and round(15) <= v <= round(25000) and round(-40) <= T <= round(60),
    }
)
def doppler2(u, v, T):
    t1 = (round(331.4) + (round(0.6) * T))
    return ((-t1 * v) / ((t1 + u) * (t1 + u)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'doppler3',
        'cite': ['darulova-kuncak-2014'],
        'fpbench_domain': 'science',
        'pre': lambda u, v, T: round(-30) <= u <= round(120) and round(320) <= v <= round(20300) and round(-50) <= T <= round(30),
    }
)
def doppler3(u, v, T):
    t1 = (round(331.4) + (round(0.6) * T))
    return ((-t1 * v) / ((t1 + u) * (t1 + u)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'rigidBody1',
        'cite': ['darulova-kuncak-2014', 'solovyev-et-al-2015'],
        'fpbench_domain': 'science',
        'pre': lambda x1, x2, x3: round(-15) <= x1 <= round(15) and round(-15) <= x2 <= round(15) and round(-15) <= x3 <= round(15),
    }
)
def rigidBody1(x1, x2, x3):
    return (((-(x1 * x2) - ((round(2) * x2) * x3)) - x1) - x3)

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'rigidBody2',
        'cite': ['darulova-kuncak-2014', 'solovyev-et-al-2015'],
        'fpbench_domain': 'science',
        'pre': lambda x1, x2, x3: round(-15) <= x1 <= round(15) and round(-15) <= x2 <= round(15) and round(-15) <= x3 <= round(15),
    }
)
def rigidBody2(x1, x2, x3):
    return (((((((round(2) * x1) * x2) * x3) + ((round(3) * x3) * x3)) - (((x2 * x1) * x2) * x3)) + ((round(3) * x3) * x3)) - x2)

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'jetEngine',
        'cite': ['darulova-kuncak-2014', 'solovyev-et-al-2015'],
        'fpbench_domain': 'controls',
        'pre': lambda x1, x2: round(-5) <= x1 <= round(5) and round(-20) <= x2 <= round(5),
    }
)
def jetEngine(x1, x2):
    t = ((((round(3) * x1) * x1) + (round(2) * x2)) - x1)
    t_u42_ = ((((round(3) * x1) * x1) - (round(2) * x2)) - x1)
    d = ((x1 * x1) + round(1))
    s = (t / d)
    s_u42_ = (t_u42_ / d)
    return (x1 + (((((((((round(2) * x1) * s) * (s - round(3))) + ((x1 * x1) * ((round(4) * s) - round(6)))) * d) + (((round(3) * x1) * x1) * s)) + ((x1 * x1) * x1)) + x1) + (round(3) * s_u42_)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'turbine1',
        'cite': ['darulova-kuncak-2014', 'solovyev-et-al-2015'],
        'fpbench_domain': 'controls',
        'pre': lambda v, w, r: round(-4.5) <= v <= round(-0.3) and round(0.4) <= w <= round(0.9) and round(3.8) <= r <= round(7.8),
    }
)
def turbine1(v, w, r):
    return (((round(3) + (round(2) / (r * r))) - (((round(0.125) * (round(3) - (round(2) * v))) * (((w * w) * r) * r)) / (round(1) - v))) - round(4.5))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'turbine2',
        'cite': ['darulova-kuncak-2014', 'solovyev-et-al-2015'],
        'fpbench_domain': 'controls',
        'pre': lambda v, w, r: round(-4.5) <= v <= round(-0.3) and round(0.4) <= w <= round(0.9) and round(3.8) <= r <= round(7.8),
    }
)
def turbine2(v, w, r):
    return (((round(6) * v) - (((round(0.5) * v) * (((w * w) * r) * r)) / (round(1) - v))) - round(2.5))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'turbine3',
        'cite': ['darulova-kuncak-2014', 'solovyev-et-al-2015'],
        'fpbench_domain': 'controls',
        'pre': lambda v, w, r: round(-4.5) <= v <= round(-0.3) and round(0.4) <= w <= round(0.9) and round(3.8) <= r <= round(7.8),
    }
)
def turbine3(v, w, r):
    return (((round(3) - (round(2) / (r * r))) - (((round(0.125) * (round(1) + (round(2) * v))) * (((w * w) * r) * r)) / (round(1) - v))) - round(0.5))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'verhulst',
        'cite': ['darulova-kuncak-2014', 'solovyev-et-al-2015'],
        'fpbench_domain': 'science',
        'pre': lambda x: round(0.1) <= x <= round(0.3),
    }
)
def verhulst(x):
    r = round(4.0)
    K = round(1.11)
    return ((r * x) / (round(1) + (x / K)))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'predatorPrey',
        'cite': ['darulova-kuncak-2014', 'solovyev-et-al-2015'],
        'fpbench_domain': 'science',
        'pre': lambda x: round(0.1) <= x <= round(0.3),
    }
)
def predatorPrey(x):
    r = round(4.0)
    K = round(1.11)
    return (((r * x) * x) / (round(1) + ((x / K) * (x / K))))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'carbonGas',
        'cite': ['darulova-kuncak-2014', 'solovyev-et-al-2015'],
        'fpbench_domain': 'science',
        'pre': lambda v: round(0.1) <= v <= round(0.5),
    }
)
def carbonGas(v):
    p = round(3.5e7)
    a = round(0.401)
    b = round(42.7e-6)
    t = round(300)
    n = round(1000)
    k = round(1.3806503e-23)
    return (((p + ((a * (n / v)) * (n / v))) * (v - (n * b))) - ((k * n) * t))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'sine',
        'cite': ['darulova-kuncak-2014', 'solovyev-et-al-2015'],
        'fpbench_domain': 'mathematics',
        'rosa_post': ['=>', 'res', ['<', '-1', 'res', '1']],
        'rosa_ensuring': '1e-14',
        'pre': lambda x: round(-1.57079632679) < x < round(1.57079632679),
    }
)
def sine(x):
    return (((x - (((x * x) * x) / round(6.0))) + (((((x * x) * x) * x) * x) / round(120))) - (((((((x * x) * x) * x) * x) * x) * x) / round(5040)))

@fpy(
    meta={
        'name': 'sqroot',
        'cite': ['darulova-kuncak-2014', 'solovyev-et-al-2015'],
        'fpbench_domain': 'mathematics',
        'pre': lambda x: round(0) <= x <= round(1),
    }
)
def sqroot(x):
    return ((((round(1.0) + (round(0.5) * x)) - ((round(0.125) * x) * x)) + (((round(0.0625) * x) * x) * x)) - ((((round(0.0390625) * x) * x) * x) * x))

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'sineOrder3',
        'cite': ['darulova-kuncak-2014', 'solovyev-et-al-2015'],
        'fpbench_domain': 'mathematics',
        'pre': lambda x: round(-2) < x < round(2),
        'rosa_post': ['=>', 'res', ['<', '-1', 'res', '1']],
        'rosa_ensuring': '1e-14',
    }
)
def sineOrder3(x):
    return ((round(0.954929658551372) * x) - (round(0.12900613773279798) * ((x * x) * x)))

@fpy(
    meta={
        'name': 'smartRoot',
        'cite': ['darulova-kuncak-2014'],
        'fpbench_domain': 'mathematics',
        'pre': lambda c: round(-2) <= c <= round(2) and ((b * b) - ((a * c) * round(4.0))) > round(0.1),
        'rosa_ensuring': '6e-15',
    }
)
def smartRoot(c):
    a0 = round(3)
    b1 = round(3.5)
    discr = ((b1 * b1) - ((a0 * c) * round(4.0)))
    if ((b1 * b1) - (a0 * c)) > round(10):
        if b1 > round(0):
            t2 = ((c * round(2)) / (-b1 - sqrt(discr)))
        else:
            if b1 < round(0):
                t = ((-b1 + sqrt(discr)) / (a0 * round(2)))
            else:
                t = ((-b1 + sqrt(discr)) / (a0 * round(2)))
            t2 = t
        t3 = t2
    else:
        t3 = ((-b1 + sqrt(discr)) / (a0 * round(2)))
    return t3

@fpy(
    meta={
        'name': 'cav10',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda x: round(0) < x < round(10),
        'rosa_post': ['=>', 'res', ['<=', '0', 'res', '3.0']],
        'rosa_ensuring': '3.0',
    }
)
def cav10(x):
    if ((x * x) - x) >= round(0):
        t = (x / round(10))
    else:
        t = ((x * x) + round(2))
    return t

@fpy(
    meta={
        'name': 'squareRoot3',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda x: round(0) < x < round(10),
        'rosa_ensuring': '1e-10',
    }
)
def squareRoot3(x):
    if x < round(1e-5):
        t = (round(1) + (round(0.5) * x))
    else:
        t = sqrt((round(1) + x))
    return t

@fpy(
    meta={
        'name': 'squareRoot3Invalid',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda x: round(0) < x < round(10),
        'rosa_ensuring': '1e-10',
    }
)
def squareRoot3Invalid(x):
    if x < round(1e-4):
        t = (round(1) + (round(0.5) * x))
    else:
        t = sqrt((round(1) + x))
    return t

@fpy(
    meta={
        'name': 'triangle',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(9.0) <= a <= round(9.0) and round(4.71) <= b <= round(4.89) and round(4.71) <= c <= round(4.89),
    }
)
def triangle(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'triangle1',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(0.1)) and (a + c) > (b + round(0.1)) and (b + c) > (a + round(0.1)),
        'rosa_post': ['=>', 'res', ['<=', '0.29', 'res', '35.1']],
        'rosa_ensuring': '2.7e-11',
    }
)
def triangle1(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'triangle2',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(1e-2)) and (a + c) > (b + round(1e-2)) and (b + c) > (a + round(1e-2)),
    }
)
def triangle2(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'triangle3',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(1e-3)) and (a + c) > (b + round(1e-3)) and (b + c) > (a + round(1e-3)),
    }
)
def triangle3(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'triangle4',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(1e-4)) and (a + c) > (b + round(1e-4)) and (b + c) > (a + round(1e-4)),
    }
)
def triangle4(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'triangle5',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(1e-5)) and (a + c) > (b + round(1e-5)) and (b + c) > (a + round(1e-5)),
    }
)
def triangle5(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'triangle6',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(1e-6)) and (a + c) > (b + round(1e-6)) and (b + c) > (a + round(1e-6)),
    }
)
def triangle6(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'triangle7',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(1e-7)) and (a + c) > (b + round(1e-7)) and (b + c) > (a + round(1e-7)),
    }
)
def triangle7(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'triangle8',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(1e-8)) and (a + c) > (b + round(1e-8)) and (b + c) > (a + round(1e-8)),
    }
)
def triangle8(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'triangle9',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(1e-9)) and (a + c) > (b + round(1e-9)) and (b + c) > (a + round(1e-9)),
    }
)
def triangle9(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'triangle10',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(1e-10)) and (a + c) > (b + round(1e-10)) and (b + c) > (a + round(1e-10)),
    }
)
def triangle10(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'triangle11',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(1e-11)) and (a + c) > (b + round(1e-11)) and (b + c) > (a + round(1e-11)),
    }
)
def triangle11(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'triangle12',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(1e-12)) and (a + c) > (b + round(1e-12)) and (b + c) > (a + round(1e-12)),
    }
)
def triangle12(a, b, c):
    s = (((a + b) + c) / round(2))
    return sqrt((((s * (s - a)) * (s - b)) * (s - c)))

@fpy(
    meta={
        'name': 'bspline3',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda u: round(0) <= u <= round(1),
        'rosa_post': ['=>', 'res', ['<=', '-0.17', 'res', '0.05']],
        'rosa_ensuring': '1e-11',
    }
)
def bspline3(u):
    return (-((u * u) * u) / round(6))

@fpy(
    meta={
        'name': 'triangleSorted',
        'cite': ['darulova-kuncak-2014'],
        'pre': lambda a, b, c: round(1) <= a <= round(9) and round(1) <= b <= round(9) and round(1) <= c <= round(9) and (a + b) > (c + round(1e-6)) and (a + c) > (b + round(1e-6)) and (b + c) > (a + round(1e-6)) and a < c and b < c,
        'rosa_post': ['=>', 'res', ['>=', 'res', '0']],
        'rosa_ensuring': '2e-9',
        'example': [['b', '4.0'], ['c', '8.5']],
    }
)
def triangleSorted(a, b, c):
    if a < b:
        t = (sqrt(((((c + (b + a)) * (a - (c - b))) * (a + (c - b))) * (c + (b - a)))) / round(4.0))
    else:
        t = (sqrt(((((c + (a + b)) * (b - (c - a))) * (b + (c - a))) * (c + (a - b)))) / round(4.0))
    return t

@fpy(
    meta={
        'name': 'N Body Simulation',
        'fpbench_domain': 'science',
        'pre': lambda x0, y0, z0, vx0, vy0, vz0: round(-6) < x0 < round(6) and round(-6) < y0 < round(6) and round(-0.2) < z0 < round(0.2) and round(-3) < vx0 < round(3) and round(-3) < vy0 < round(3) and round(-0.1) < vz0 < round(0.1),
    }
)
def N_Body_Simulation(x0, y0, z0, vx0, vy0, vz0):
    dt = round(0.1)
    solarMass = round(39.47841760435743)
    x = x0
    y = y0
    z = z0
    vx = vx0
    vy = vy0
    vz = vz0
    i = round(0)
    while i < round(100):
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
        t15 = (i + round(1))
        x = t
        y = t2
        z = t5
        vx = t8
        vy = t11
        vz = t14
        i = t15
    return x

@fpy(
    meta={
        'name': 'Pendulum',
        'fpbench_domain': 'science',
        'pre': lambda t0, w0, N: round(-2) < t0 < round(2) and round(-5) < w0 < round(5),
        'example': [['N', '1000']],
    }
)
def Pendulum(t0, w0, N):
    h = round(0.01)
    L = round(2.0)
    m = round(1.5)
    g = round(9.80665)
    t = t0
    w = w0
    n = round(0)
    while n < N:
        k1w = ((-g / L) * sin(t))
        k2t = (w + ((h / round(2)) * k1w))
        t1 = (t + (h * k2t))
        k2w = ((-g / L) * sin((t + ((h / round(2)) * w))))
        t2 = (w + (h * k2w))
        t3 = (n + round(1))
        t = t1
        w = t2
        n = t3
    return t

@fpy(
    meta={
        'name': 'Sine Newton',
        'fpbench_domain': 'mathematics',
        'pre': lambda x0: round(-1) < x0 < round(1),
    }
)
def Sine_Newton(x0):
    x = x0
    i = round(0)
    while i < round(10):
        t = (x - ((((x - (pow(x, round(3)) / round(6.0))) + (pow(x, round(5)) / round(120.0))) + (pow(x, round(7)) / round(5040.0))) / (((round(1.0) - ((x * x) / round(2.0))) + (pow(x, round(4)) / round(24.0))) + (pow(x, round(6)) / round(720.0)))))
        t0 = (i + round(1))
        x = t
        i = t0
    return x

@fpy(
    meta={
        'name': "Rump's example, with pow",
        'example': [['a', '77617'], ['b', '33096']],
    }
)
def Rump_u39_s_example_u44__with_pow(a, b):
    return ((((round(333.75) * pow(b, round(6))) + (pow(a, round(2)) * (((((round(11) * pow(a, round(2))) * pow(b, round(2))) - pow(b, round(6))) - (round(121) * pow(b, round(4)))) - round(2)))) + (round(5.5) * pow(b, round(8)))) + (a / (round(2) * b)))

@fpy(
    meta={
        'name': "Rump's example, from C program",
        'example': [['a', '77617'], ['b', '33096']],
    }
)
def Rump_u39_s_example_u44__from_C_program(a, b):
    b2 = (b * b)
    b4 = (b2 * b2)
    b6 = (b4 * b2)
    b8 = (b4 * b4)
    a2 = (a * a)
    firstexpr = (((((round(11) * a2) * b2) - b6) - (round(121) * b4)) - round(2))
    return ((((round(333.75) * b6) + (a2 * firstexpr)) + (round(5.5) * b8)) + (a / (round(2) * b)))

@fpy(
    meta={
        'name': "Rump's example revisited for floating point",
        'example': [['a', '77617'], ['b', '33096']],
        'cite': ['rump-revisited-2002'],
    }
)
def Rump_u39_s_example_revisited_for_floating_point(a, b):
    b2 = (b * b)
    b4 = (b2 * b2)
    b6 = (b4 * b2)
    b8 = (b4 * b4)
    a2 = (a * a)
    firstexpr = ((((round(11) * a2) * b2) - (round(121) * b4)) - round(2))
    return (((((round(333.75) - a2) * b6) + (a2 * firstexpr)) + (round(5.5) * b8)) + (a / (round(2) * b)))

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'Odometry',
        'description': ('Compute the position of a robot from the speed of the wheels.\n'
 'Inputs: Speed `sl`, `sr` of the left and right wheel, in rad/s.'),
        'cite': ['damouche-martel-chapoutot-fmics15'],
        'fpbench_domain': 'controls',
        'pre': lambda sr_u42_, sl_u42_: round(0.05) < sl_u42_ < (round(2) * const_pi()) and round(0.05) < sr_u42_ < (round(2) * const_pi()),
        'example': [['sr*', '0.0785398163397'], ['sl*', '0.0525398163397']],
    }
)
def Odometry(sr_u42_, sl_u42_):
    inv_l = round(0.1)
    c = round(12.34)
    delta_dl = round(0.0)
    delta_dr = round(0.0)
    delta_d = round(0.0)
    delta_theta = round(0.0)
    arg = round(0.0)
    cosi = round(0.0)
    x = round(0.0)
    sini = round(0.0)
    y = round(0.0)
    theta = round(-.985)
    t = round(0)
    tmp = sl_u42_
    sl = sl_u42_
    sr = sr_u42_
    j = round(0)
    while t < round(1000):
        delta_dl = (c * sl)
        delta_dr = (c * sr)
        delta_d = ((delta_dl + delta_dr) * round(0.5))
        delta_theta = ((delta_dr - delta_dl) * inv_l)
        arg = (theta + (delta_theta * round(0.5)))
        cosi = ((round(1) - ((arg * arg) * round(0.5))) + ((((arg * arg) * arg) * arg) * round(0.0416666666)))
        x = (x + (delta_d * cosi))
        sini = ((arg - (((arg * arg) * arg) * round(0.1666666666))) + (((((arg * arg) * arg) * arg) * arg) * round(0.008333333)))
        y = (y + (delta_d * sini))
        theta = (theta + delta_theta)
        t = (t + round(1))
        tmp = sl
        if j == round(50):
            t0 = sr
        else:
            t0 = sl
        sl = t0
        if j == round(50):
            t1 = tmp
        else:
            t1 = sr
        sr = t1
        if j == round(50):
            t2 = round(0)
        else:
            t2 = (j + round(1))
        j = t2
    return x

@fpy(
    ctx=IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'PID',
        'description': ('Keep a measure at its setpoint using a PID controller.\n'
 'Inputs: Measure `m`; gains `kp`, `ki`, `kd`; setpoint `c`'),
        'cite': ['damouche-martel-chapoutot-nsv14', 'damouche-martel-chapoutot-fmics15'],
        'fpbench_domain': 'controls',
        'pre': lambda m, kp, ki, kd, c: round(-10.0) < m < round(10.0) and round(-10.0) < c < round(10.0),
        'example': [['m', '-5.0'], ['kp', '9.4514'], ['ki', '0.69006'], ['kd', '2.8454']],
    }
)
def PID(m, kp, ki, kd, c):
    dt = round(0.5)
    invdt = (round(1) / dt)
    e = round(0.0)
    p = round(0.0)
    i = round(0.0)
    d = round(0.0)
    r = round(0.0)
    m0 = m
    eold = round(0.0)
    t = round(0.0)
    while t < round(100.0):
        e = (c - m0)
        p = (kp * e)
        i = (i + ((ki * dt) * e))
        d = ((kd * invdt) * (e - eold))
        r = ((p + i) + d)
        m0 = (m0 + (round(0.01) * r))
        eold = e
        t = (t + dt)
    return m0

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'Runge-Kutta 4',
        'description': ("Solve the differential equation `y' = (c - y)^2\n"
 'Inputs: Step size `h`; initial condition `y_n*`; paramter `c`'),
        'cite': ['damouche-martel-chapoutot-fmics15'],
        'fpbench_domain': 'mathematics',
        'pre': lambda h, y_n_u42_, c: round(0) < y_n_u42_ < round(100) and round(10e-6) < h < round(0.1) and round(50) < c < round(200),
        'example': [['h', '0.1'], ['y_n*', '10.1'], ['c', '100.1']],
    }
)
def Runge_Kutta_4(h, y_n_u42_, c):
    sixieme = (round(1) / round(6))
    eps = round(0.005)
    k = round(1.2)
    y_n = y_n_u42_
    i = round(0.0)
    e = round(1.0)
    while e > eps:
        v = (c - y_n)
        k1 = ((k * v) * v)
        v0 = (c - (y_n + ((round(0.5) * h) * k1)))
        k2 = ((k * v0) * v0)
        v1 = (c - (y_n + ((round(0.5) * h) * k2)))
        k3 = ((k * v1) * v1)
        v2 = (c - (y_n + (h * k3)))
        k4 = ((k * v2) * v2)
        t = (y_n + ((sixieme * h) * (((k1 + (round(2.0) * k2)) + (round(2.0) * k3)) + k4)))
        t3 = (i + round(1.0))
        t4 = (e - eps)
        y_n = t
        i = t3
        e = t4
    return abs(e)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'Lead-lag System',
        'description': ('Move a mass from an initial position to a desired position.\n'
 'Inputs: Initial position `y`; desired position `yd`'),
        'cite': ['feron-ieee10', 'damouche-martel-chapoutot-fmics15'],
        'fpbench_domain': 'controls',
        'pre': lambda y, yd: round(0) < yd < round(50) and round(0) < y < round(50),
        'example': [['y', '2.5'], ['yd', '5.0']],
    }
)
def Lead_lag_System(y, yd):
    eps = round(0.01)
    Dc = round(-1280.0)
    Ac0 = round(.499)
    Ac1 = round(-.05)
    Ac10 = round(.010)
    Ac11 = round(1.00)
    Bc0 = round(1.0)
    Bc1 = round(0.0)
    Cc0 = round(564.48)
    Cc1 = round(0.0)
    yc = round(0.0)
    u = round(0.0)
    xc0 = round(0.0)
    xc1 = round(0.0)
    i = round(0.0)
    e = round(1.0)
    while e > eps:
        v = (y - yd)
        if v < round(-1.0):
            t0 = round(-1.0)
        else:
            if round(1.0) < v:
                t = round(1.0)
            else:
                t = v
            t0 = t
        yc = t0
        u = ((Cc0 * xc0) + ((Cc1 * xc1) + (Dc * yc)))
        xc0 = ((Ac0 * xc0) + ((Ac1 * xc1) + (Bc0 * yc)))
        xc1 = ((Ac10 * xc0) + ((Ac11 * xc1) + (Bc1 * yc)))
        i = (i + round(1.0))
        e = abs((yc - xc1))
    return xc1

@fpy(
    meta={
        'name': 'Trapeze',
        'cite': ['damouche-martel-chapoutot-fmics15'],
        'fpbench_domain': 'mathematics',
        'pre': lambda u: round(1.11) <= u <= round(2.22),
    }
)
def Trapeze(u):
    a = round(0.25)
    b = round(5000)
    n = round(25)
    h = ((b - a) / n)
    xb = round(0)
    r = round(0)
    xa = round(0.25)
    while xa < round(5000):
        v = (xa + h)
        if v > round(5000):
            t = round(5000)
        else:
            t = v
        xb = t
        gxa = (u / ((((((round(0.7) * xa) * xa) * xa) - ((round(0.6) * xa) * xa)) + (round(0.9) * xa)) - round(0.2)))
        gxb = (u / ((((((round(0.7) * xb) * xb) * xb) - ((round(0.6) * xb) * xb)) + (round(0.9) * xb)) - round(0.2)))
        r = (r + (((gxa + gxb) * round(0.5)) * h))
        xa = (xa + h)
    return r

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'Rocket Trajectory',
        'description': ('Compute the trajectory of a rocket around the earth.\n'
 'Inputs: Mass `Mf`; acceleration `A`'),
        'cite': ['damouche-martel-chapoutot-cf15'],
        'fpbench_domain': 'controls',
        'example': [['Mf', '150000.0'], ['A', '140.0']],
    }
)
def Rocket_Trajectory(Mf, A):
    R = round(6400.0e3)
    G = round(6.67428e-11)
    Mt = round(5.9736e24)
    dt = round(0.1)
    T = (round(24.0) * round(3600.0))
    nombrepas = (T / dt)
    r0 = ((round(400.0) * round(10e3)) + R)
    vr0 = round(0.0)
    teta0 = round(0.0)
    viss = sqrt(((G * Mt) / r0))
    vteta0 = (viss / r0)
    rf = R
    vrf = round(0.0)
    tetaf = round(0.0)
    vl = sqrt(((G * Mt) / R))
    vlrad = (vl / r0)
    vtetaf = (round(1.1) * vlrad)
    t_i = round(0.0)
    mf_i = round(0)
    u1_i = round(0)
    u3_i = round(0)
    w1_i = round(0)
    w3_i = round(0)
    u2_i = round(0)
    u4_i = round(0)
    w2_i = round(0)
    w4_i = round(0)
    x = round(0)
    y = round(0)
    i = round(1.0)
    u1_im1 = r0
    u2_im1 = vr0
    u3_im1 = teta0
    u4_im1 = vteta0
    w1_im1 = rf
    w2_im1 = vrf
    w3_im1 = tetaf
    w4_im1 = vtetaf
    t_im1 = round(0)
    mf_im1 = Mf
    while i < round(2000000.0):
        t_i = (t_im1 + dt)
        mf_i = (mf_im1 - (A * t_im1))
        u1_i = ((u2_im1 * dt) + u1_im1)
        u3_i = ((u4_im1 * dt) + u3_im1)
        w1_i = ((w2_im1 * dt) + w1_im1)
        w3_i = ((w4_im1 * dt) + w3_im1)
        u2_i = (((-G * (Mt / (u1_im1 * u1_im1))) * dt) + ((u1_im1 * u4_im1) * (u4_im1 * dt)))
        u4_i = (((round(-2.0) * (u2_im1 * (u4_im1 / u1_im1))) * dt) + u4_im1)
        if mf_im1 > round(0.0):
            t = (((A * w2_im1) / (Mf - (A * t_im1))) * dt)
        else:
            t = round(0.0)
        w2_i = ((((-G * (Mt / (w1_im1 * w1_im1))) * dt) + ((w1_im1 * w4_im1) * (w4_im1 * dt))) + (t + w2_im1))
        if mf_im1 > round(0.0):
            t0 = (A * ((w4_im1 / (Mf - (A * t_im1))) * dt))
        else:
            t0 = round(0.0)
        w4_i = (((round(-2.0) * (w2_im1 * (w4_im1 / w1_im1))) * dt) + (t0 + w4_im1))
        x = (u1_i * cos(u3_i))
        y = (u1_i * sin(u3_i))
        i = (i + round(1.0))
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

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': "Jacobi's Method",
        'description': ('Solve a linear system `Ax = b`.\n'
 'Inputs: Array entries `aij`; vector entries `bi`'),
        'cite': ['atkinson-1989'],
        'fpbench_domain': 'mathematics',
        'example': [['a11', '0.61'],
 ['a22', '0.62'],
 ['a33', '0.6006'],
 ['a44', '0.601'],
 ['b1', '0.5'],
 ['b2', ['/', '1.0', '3.0']],
 ['b3', '0.25'],
 ['b4', ['/', '1.0', '5.0']]],
    }
)
def Jacobi_u39_s_Method(a11, a22, a33, a44, b1, b2, b3, b4):
    eps = round(0.00000000000000001)
    x_n1 = round(0.0)
    x_n2 = round(0.0)
    x_n3 = round(0.0)
    x_n4 = round(0.0)
    i = round(0.0)
    e = round(1.0)
    x1 = round(0.0)
    x2 = round(0.0)
    x3 = round(0.0)
    x4 = round(0.0)
    while e > eps:
        x_n1 = ((((b1 / a11) - ((round(0.1) / a11) * x2)) - ((round(0.2) / a11) * x3)) + ((round(0.3) / a11) * x4))
        x_n2 = ((((b2 / a22) - ((round(0.3) / a22) * x1)) + ((round(0.1) / a22) * x3)) - ((round(0.2) / a22) * x4))
        x_n3 = ((((b3 / a33) - ((round(0.2) / a33) * x1)) + ((round(0.3) / a33) * x2)) - ((round(0.1) / a33) * x4))
        x_n4 = ((((b4 / a44) + ((round(0.1) / a44) * x1)) - ((round(0.2) / a44) * x2)) - ((round(0.3) / a44) * x3))
        i = (i + round(1.0))
        e = abs((x_n4 - x4))
        x1 = x_n1
        x2 = x_n2
        x3 = x_n3
        x4 = x_n4
    return x2

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': "Newton-Raphson's Method",
        'description': 'Find the zeros of a function `f = (x - 2)**5`.\nInputs: Initial guess `x0`',
        'cite': ['atkinson-1989'],
        'fpbench_domain': 'mathematics',
        'pre': lambda x0: round(0) < x0 < round(3),
        'example': [['x0', '0.0']],
    }
)
def Newton_Raphson_u39_s_Method(x0):
    eps = round(0.0005)
    x_n = round(0.0)
    e = round(1.0)
    x = x0
    i = round(0.0)
    while e > eps and i < round(100000):
        f = (((((((x * x) * ((x * x) * x)) - ((round(10.0) * x) * ((x * x) * x))) + ((round(40.0) * x) * (x * x))) - ((round(80.0) * x) * x)) + (round(80.0) * x)) - round(32.0))
        ff = ((((((round(5.0) * x) * ((x * x) * x)) - ((round(40.0) * x) * (x * x))) + ((round(120.0) * x) * x)) - (round(160.0) * x)) + round(80.0))
        x_n = (x - (f / ff))
        e = abs((x - x_n))
        x = x_n
        i = (i + round(1.0))
    return x

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'Eigenvalue Computation',
        'description': ('Compute the largest eigenvalue of a matrix and return its vector.\n'
 'Inputs: Matrix `aij`; initial guess `vi` with one nonzero element'),
        'cite': ['golub-vanloan-1996'],
        'fpbench_domain': 'mathematics',
        'pre': lambda a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44, v1, v2, v3, v4: round(150) < (((((((((((((((a11 * a22) * a33) * a44) + (((a12 * a23) * a34) * a41)) + (((a13 * a24) * a31) * a42)) + (((a14 * a21) * a32) * a43)) + (((a11 * a23) * a34) * a42)) + (((a12 * a21) * a33) * a44)) + (((a13 * a21) * a32) * a44)) + (((a14 * a22) * a33) * a41)) + (((a11 * a24) * a32) * a43)) + (((a12 * a24) * a31) * a43)) + (((a13 * a22) * a34) * a41)) + (((a14 * a23) * a31) * a42)) - ((((((((((((((a11 * a22) * a34) * a43) + (((a12 * a23) * a31) * a44)) + (((a13 * a24) * a32) * a41)) + (((a14 * a21) * a33) * a42)) + (((a11 * a23) * a32) * a44)) + (((a12 * a21) * a34) * a43)) + (((a13 * a21) * a34) * a42)) + (((a14 * a22) * a31) * a43)) + (((a11 * a24) * a33) * a42)) + (((a12 * a24) * a33) * a41)) + (((a13 * a22) * a31) * a44)) + (((a14 * a23) * a32) * a41))) < round(200),
        'example': [['a11', '150.0'],
 ['a12', '0.01'],
 ['a13', '0.01'],
 ['a14', '0.01'],
 ['a21', '0.01'],
 ['a22', '150.0'],
 ['a23', '0.01'],
 ['a24', '0.01'],
 ['a31', '0.01'],
 ['a32', '0.01'],
 ['a33', '150.0'],
 ['a34', '0.01'],
 ['a41', '0.01'],
 ['a42', '0.01'],
 ['a43', '0.01'],
 ['a44', '150.0'],
 ['v1', '0.0'],
 ['v2', '0.0'],
 ['v3', '0.0'],
 ['v4', '1.0']],
    }
)
def Eigenvalue_Computation(a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44, v1, v2, v3, v4):
    eps = round(0.0005)
    vx = round(0)
    vy = round(0)
    vz = round(0)
    vw = round(0)
    i = round(0.0)
    v0 = v1
    v5 = v2
    v6 = v3
    v7 = v4
    e = round(1.0)
    while e > eps:
        vx = (((a11 * v0) + (a12 * v5)) + ((a13 * v6) + (a14 * v7)))
        vy = (((a21 * v0) + (a22 * v5)) + ((a23 * v6) + (a24 * v7)))
        vz = (((a31 * v0) + (a32 * v5)) + ((a33 * v6) + (a34 * v7)))
        vw = (((a41 * v0) + (a42 * v5)) + ((a43 * v6) + (a44 * v7)))
        i = (i + round(1.0))
        v0 = (vx / vw)
        v5 = (vy / vw)
        v6 = (vz / vw)
        v7 = round(1.0)
        e = abs((round(1.0) - v0))
    return v0

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
        'name': 'Iterative Gram-Schmidt Method',
        'description': ('Orthogonalize a set of non-zero vectors in a Euclidian or Hermitian space.\n'
 'Inputs: Vectors `Qij`'),
        'cite': ['abdelmalek-bit71', 'golub-vanloan-1996', 'hernandez-roman-tomas-vidal-tr07'],
        'fpbench_domain': 'mathematics',
        'example': [['Q11', ['/', '1', '63']],
 ['Q12', '0'],
 ['Q13', '0'],
 ['Q21', '0'],
 ['Q22', ['/', '1', '225']],
 ['Q23', '0'],
 ['Q31', ['/', '1', '2592']],
 ['Q32', ['/', '1', '2601']],
 ['Q33', ['/', '1', '2583']]],
    }
)
def Iterative_Gram_Schmidt_Method(Q11, Q12, Q13, Q21, Q22, Q23, Q31, Q32, Q33):
    eps = round(.000005)
    h1 = round(0)
    h2 = round(0)
    h3 = round(0)
    qj1 = Q31
    qj2 = Q32
    qj3 = Q33
    r1 = round(0.0)
    r2 = round(0.0)
    r3 = round(0.0)
    r = (((qj1 * qj1) + (qj2 * qj2)) + (qj3 * qj3))
    rjj = round(0)
    e = round(10.0)
    i = round(1.0)
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
        e = abs((round(1.0) - (rjj / rold)))
        i = (i + round(1.0))
        rold = rjj
    return qj1

