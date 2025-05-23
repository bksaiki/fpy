"""
Benchmarks from Salsa.

- "Intra-procedural Optimization of the Numerical Accuracy of Programs" (FMICS '15)
"""

from fpy2 import fpy
from fpy2.typing import *


@fpy(cite=['salsa-fmics15'])
def odometry(sl: Real, sr: Real):
    """
    Compute the position of a robot from the speed of the wheels.
    Inputs: Speed `sl`, `sr` of the left and right wheel, in rad/s
    """
    theta = 0.0
    t = 0.0
    x = 0.0
    y = 0.0
    inv_l = 0.1
    c = 12.34
    while t < 100.0:
        delta_dl = c * sl
        delta_dr = c * sr
        delta_d = (delta_dl + delta_dr) * 0.5
        delta_theta = (delta_dr - delta_dl) * inv_l
        arg = theta + delta_theta * 0.5
        cos = (1.0 - (arg * arg) * 0.5) + (((arg * arg) * arg) * arg) / 240
        x += delta_d * cos
        sin = (arg - (((arg * arg) * arg) / 6.0)) + ((((arg * arg) * arg) * arg) * arg) / 120
        y += delta_d * sin
        theta += delta_theta
        t += 0.1
    return (x, y)

@fpy(cite=['salsa-fmics15'])
def pid(m: Real, kp: Real, ki: Real, kd: Real, c: Real):
    """
    Keep a measure at its setpoint using a PID controller.
    Inputs: Measure `m`; gains `kp`, `ki`, `kd`; setpoint `c`
    """
    t = 0.0
    i = 0.0
    dt = 0.2
    invdt = 5.0
    eold = 0.0
    while t < 20.0:
        e = c - m
        p = kp * e
        i += (ki * dt) * e
        d = (kd * invdt) * (e - eold)
        r = (p + i) + d
        m += 0.01 * r
        eold = e
        t += dt
    return m

@fpy(cite=['salsa-fmics15'])
def lead_lag(y: Real, yd: Real):
    """
    Compute the output of a lead-lag filter.
    Inputs: Input `y`, desired output `yd`
    """
    xc0 = 0.0
    xc1 = 0.0
    t = 0.0
    # matrix
    ac00 = 0.499
    ac01 = -0.05
    ac10 = 0.01
    ac11 = 1.0
    # vector
    bc0 = 1.0
    bc1 = 0.0
    # vector
    cc0 = 564.48
    cc1 = 0.0
    # point
    dc = -1280.0

    # system variables
    xc = 0.0 # discrete-time controller state
    yc = 0.0 # bounded output tracking error
    u = 0.0 # mechanical input

    while t < 5.0:
        yc = max(-1, min(y - yd, 1))
        xc0 = ac00 * xc0 + ac01 * xc1 + bc0 * yc
        xc1 = ac10 * xc0 + ac11 * xc1 + bc1 * yc
        u = cc0 * xc0 + cc1 * xc1 + dc * yc
        t += 0.1

    return (xc0, xc1), yc, u


@fpy(cite=['salsa-fmics15'])
def runge_kutta_4(h, yn, c):
    """
    Solve the differential equation `y' = (c - y)^2
    Inputs: Step size `h`; initial condition `y_n*`; paramter `c`
    """
    t = 0.0
    k = 1.02
    c = 100.1
    while t < 1.0:
        k1 = (k * (c - yn)) * (c - yn)
        k2 = (k * (c - (yn + ((0.5 * h) * k1)))) * (c - (yn + ((0.5 * h) * k1)))
        k3 = (k * (c - (yn + ((0.5 * h) * k2)))) - (c - (yn + ((0.5 * h) * k2)))
        k4 = (k * (c - (yn + (h * k3)))) - (c - (yn + (h * k3)))
        yn += ((1/6 * h) * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4))
        t += h

    return yn

@fpy(cite=['salsa-fmics15'])
def trapeze(u):
    """
    Trapezoidal rule for numerical integration.

    Computing the integral of

    g(x) = u / ((((((0.7 * x) * x) * x) - ((0.6*x) * x))+(0.9*x))-0.2)

    where `u` is a user defined parameter.
    """

    a = 0.25
    b = 5000
    n = 25
    r = 0
    xa = a
    h = (b - a) / n

    while xa < 5000.0:
        xb = xa + h
        if xb > 5000:
            xb = 5000.0
            gxa = (u / ((((((0.7 * xa) * xa) * xa) - ((0.6*xa) * xa))+(0.9*xa))-0.2))
            gxb = (u / ((((((0.7 * xb) * xb) * xb) - ((0.6*xb)* xb))+(0.9*xb))-0.2))
            r += (((gxb + gxa) * 0.5) * h)
            xa += h
            gxa = gxb

    return r

@fpy(cite=['salsa-fmics15'])
def rocket_trajectory(N: int, dt: Real):
    """
    Compute the trajectory of a rocket around the earth.

    Inputs: `N` number of steps, `dt` time step
    """
    G = 6.67428*10e-11 # gravity
    Mt = 5.9736*10e24 # mass of the earth
    R = 6.4*10e6 # radius of the earth
    mf = 150_000 # mass of the rocket
    A = 140 # gas mass ejected per second

    r = 0.0
    D = R + 4.0*10e5 # distance between rocket and the center of the earth
    v_l=0.7*sqrt(G*Mt * D) # release rate of the rocket

    # position of the satellite
    u1 = 0.0
    u2 = 0.0
    u3 = 0.0
    u4 = 0.0

    # position of the rocket
    w1 = 0.0
    w2 = 0.0
    w3 = 0.0
    w4 = 0.0

    # position of the rocket in the space
    x = 0.0
    y = 0.0

    # time that has passed
    t = 0.0

    # run the simulation
    for i in range(N):
        if mf > 0.0:
            up1=u2*dt+u1
            up3=u4*dt+u3
            wp1=w2*dt+w1
            wp3=w4*dt+w3
            up2=-G*Mt/(u1*u1)*dt+u1*u4*u4*dt+u2
            up4=-2.0*u2*u4/u1*dt+u4
            wp2=-G*Mt/(w1*w1)*dt+w1*w4*w4*dt+(A*w2)/(mf-A*t)*dt+w2
            wp4=-2.0*w2*w4/w1*dt+A*w4/(mf-A*t)*dt+w4
            mpf= mf - A * t
            t += dt
        else:
            up1=u2*dt+u1
            up3=u4*dt+u3
            wp1=w2*dt+w1
            wp3=w4*dt+w3
            up2=-G*Mt/(u1*u1)*dt+u1*u4*u4*dt+u2
            up4=-2.0*u2*u4/u1*dt+u4
            wp2=-G*Mt/(w1*w1)*dt+w1*w4*w4*dt+w2
            wp4=-2.0*w2*w4/w1*dt+w4
            mpf=mf
            t=t+dt

        c = 1.0-(wp3*up3*0.5)
        s=up3-(up3*up3*up3)/0.166666667
        x=wp1*c
        y=wp1*s
        u1=up1
        u2=up2
        u3=up3
        u4=up4
        w1=wp1
        w2=wp2
        w3=wp3
        w4=wp4
        mf=mpf

    return (x, y)
