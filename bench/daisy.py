import fpy2 as fp


@fp.fpy(
    meta={
        "name": "carthesianToPolar, radius",
        "pre": lambda x, y: fp.round(1) <= x <= fp.round(100)
        and fp.round(1) <= y <= fp.round(100),
        "spec": lambda x, y: fp.hypot(x, y),
    }
)
def carthesianToPolar_radius(x: fp.Real, y: fp.Real) -> fp.Real:
    return fp.sqrt(x * x + y * y)


@fp.fpy(
    meta={
        "name": "carthesianToPolar, theta",
        "pre": lambda x, y: fp.round(1) <= x <= fp.round(100)
        and fp.round(1) <= y <= fp.round(100),
        "spec": lambda x, y: fp.atan2(y, x) * (fp.round(180) / fp.const_pi()),
    }
)
def carthesianToPolar_theta(x: fp.Real, y: fp.Real) -> fp.Real:
    pi = fp.round(3.14159265359)
    radiant = fp.atan(y / x)
    return radiant * (fp.round(180.0) / pi)


@fp.fpy(
    meta={
        "name": "polarToCarthesian, x",
        "pre": lambda radius, theta: fp.round(1) <= radius <= fp.round(10)
        and fp.round(0) <= theta <= fp.round(360),
        "spec": lambda radius, theta: radius
        * fp.cos(theta * (fp.round(180) / fp.const_pi())),
    }
)
def polarToCarthesian_x(radius: fp.Real, theta: fp.Real) -> fp.Real:
    pi = fp.round(3.14159265359)
    radiant = theta * (pi / fp.round(180.0))
    return radius * fp.cos(radiant)


@fp.fpy(
    meta={
        "name": "polarToCarthesian, y",
        "pre": lambda radius, theta: fp.round(1) <= radius <= fp.round(10)
        and fp.round(0) <= theta <= fp.round(360),
        "spec": lambda radius, theta: radius
        * fp.sin(theta * (fp.round(180) / fp.const_pi())),
    }
)
def polarToCarthesian_y(radius: fp.Real, theta: fp.Real) -> fp.Real:
    pi = fp.round(3.14159265359)
    radiant = theta * (pi / fp.round(180.0))
    return radius * fp.sin(radiant)


@fp.fpy(
    meta={
        "name": "instantaneousCurrent",
        "pre": lambda t, resistance, frequency, inductance, maxVoltage: (
            fp.round(0) <= t <= fp.round(300.0)
            and fp.round(1) <= resistance <= fp.round(50)
            and fp.round(1) <= frequency <= fp.round(100)
            and fp.round(0.001) <= inductance <= fp.round(0.004)
            and fp.round(1) <= maxVoltage <= fp.round(12)
        ),
    }
)
def instantaneousCurrent(
    t: fp.Real,
    resistance: fp.Real,
    frequency: fp.Real,
    inductance: fp.Real,
    maxVoltage: fp.Real,
) -> fp.Real:
    pi = fp.round(3.14159265359)
    impedance_re = resistance
    impedance_im = fp.round(2) * pi * frequency * inductance
    denom = impedance_re * impedance_re + impedance_im * impedance_im
    current_re = (maxVoltage * impedance_re) / denom
    current_im = -(maxVoltage * impedance_im) / denom
    maxCurrent = fp.sqrt(current_re * current_re + current_im * current_im)
    theta = fp.atan(current_im / current_re)
    return maxCurrent * fp.cos(fp.round(2) * pi * frequency * t + theta)


@fp.fpy(
    meta={
        "name": "matrixDeterminant",
        "pre": lambda a, b, c, d, e, f, g, h, i: (
            fp.round(-10) <= a <= fp.round(10)
            and fp.round(-10) <= b <= fp.round(10)
            and fp.round(-10) <= c <= fp.round(10)
            and fp.round(-10) <= d <= fp.round(10)
            and fp.round(-10) <= e <= fp.round(10)
            and fp.round(-10) <= f <= fp.round(10)
            and fp.round(-10) <= g <= fp.round(10)
            and fp.round(-10) <= h <= fp.round(10)
            and fp.round(-10) <= i <= fp.round(10)
        ),
    }
)
def matrixDeterminant(
    a: fp.Real,
    b: fp.Real,
    c: fp.Real,
    d: fp.Real,
    e: fp.Real,
    f: fp.Real,
    g: fp.Real,
    h: fp.Real,
    i: fp.Real,
) -> fp.Real:
    return (a * e * i + b * f * g + c * d * h) - (c * e * g + b * d * i + a * f * h)


@fp.fpy(
    meta={
        "name": "matrixDeterminant2",
        "pre": lambda a, b, c, d, e, f, g, h, i: (
            fp.round(-10) <= a <= fp.round(10)
            and fp.round(-10) <= b <= fp.round(10)
            and fp.round(-10) <= c <= fp.round(10)
            and fp.round(-10) <= d <= fp.round(10)
            and fp.round(-10) <= e <= fp.round(10)
            and fp.round(-10) <= f <= fp.round(10)
            and fp.round(-10) <= g <= fp.round(10)
            and fp.round(-10) <= h <= fp.round(10)
            and fp.round(-10) <= i <= fp.round(10)
        ),
    }
)
def matrixDeterminant2(
    a: fp.Real,
    b: fp.Real,
    c: fp.Real,
    d: fp.Real,
    e: fp.Real,
    f: fp.Real,
    g: fp.Real,
    h: fp.Real,
    i: fp.Real,
) -> fp.Real:
    return (a * (e * i) + (g * (b * f) + c * (d * h))) - (
        e * (c * g) + (i * (b * d) + a * (f * h))
    )
