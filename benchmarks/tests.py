from fpy2 import *

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f0():
    return -1.0

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f1(arg1):
    return -arg1

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f2():
    return (1.0 + 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f3(arg1):
    return (arg1 + 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f4(arg1):
    return (1.0 + arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f5(arg1):
    return (arg1 + arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f6(arg1, arg2):
    return (arg1 + arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f7(arg1, arg2):
    return (arg2 + arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f8():
    return (1.0 - 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f9(arg1):
    return (arg1 - 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f10(arg1):
    return (1.0 - arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f11(arg1):
    return (arg1 - arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f12(arg1, arg2):
    return (arg1 - arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f13(arg1, arg2):
    return (arg2 - arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f14():
    return (1.0 * 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f15(arg1):
    return (arg1 * 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f16(arg1):
    return (1.0 * arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f17(arg1):
    return (arg1 * arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f18(arg1, arg2):
    return (arg1 * arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f19(arg1, arg2):
    return (arg2 * arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f20():
    return (1.0 / 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f21(arg1):
    return (arg1 / 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f22(arg1):
    return (1.0 / arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f23(arg1):
    return (arg1 / arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f24(arg1, arg2):
    return (arg1 / arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f25(arg1, arg2):
    return (arg2 / arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f26():
    return fabs(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f27(arg1):
    return fabs(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f28():
    return fma(1.0, 1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f29(arg1):
    return fma(arg1, 1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f30(arg1):
    return fma(1.0, arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f31(arg1):
    return fma(1.0, 1.0, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f32(arg1):
    return fma(arg1, arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f33(arg1):
    return fma(arg1, 1.0, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f34(arg1):
    return fma(1.0, arg1, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f35(arg1, arg2):
    return fma(arg1, arg2, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f36(arg1, arg2):
    return fma(arg2, arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f37(arg1, arg2):
    return fma(arg1, 1.0, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f38(arg1, arg2):
    return fma(1.0, arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f39(arg1, arg2):
    return fma(arg2, 1.0, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f40(arg1, arg2):
    return fma(1.0, arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f41(arg1):
    return fma(arg1, arg1, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f42(arg1, arg2):
    return fma(arg1, arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f43(arg1, arg2):
    return fma(arg1, arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f44(arg1, arg2):
    return fma(arg2, arg1, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f45(arg1, arg2):
    return fma(arg1, arg2, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f46(arg1, arg2):
    return fma(arg2, arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f47(arg1, arg2):
    return fma(arg2, arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f48(arg1, arg2, arg3):
    return fma(arg1, arg2, arg3)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f49(arg1, arg2, arg3):
    return fma(arg2, arg1, arg3)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f50(arg1, arg2, arg3):
    return fma(arg1, arg3, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f51(arg1, arg2, arg3):
    return fma(arg3, arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f52(arg1, arg2, arg3):
    return fma(arg2, arg3, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f53(arg1, arg2, arg3):
    return fma(arg3, arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f54():
    return exp(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f55(arg1):
    return exp(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f56():
    return exp2(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f57(arg1):
    return exp2(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f58():
    return expm1(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f59(arg1):
    return expm1(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f60():
    return log(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f61(arg1):
    return log(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f62():
    return log10(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f63(arg1):
    return log10(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f64():
    return log2(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f65(arg1):
    return log2(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f66():
    return log1p(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f67(arg1):
    return log1p(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f68():
    return pow(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f69(arg1):
    return pow(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f70(arg1):
    return pow(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f71(arg1):
    return pow(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f72(arg1, arg2):
    return pow(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f73(arg1, arg2):
    return pow(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f74():
    return sqrt(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f75(arg1):
    return sqrt(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f76():
    return cbrt(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f77(arg1):
    return cbrt(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f78():
    return hypot(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f79(arg1):
    return hypot(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f80(arg1):
    return hypot(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f81(arg1):
    return hypot(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f82(arg1, arg2):
    return hypot(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f83(arg1, arg2):
    return hypot(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f84():
    return sin(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f85(arg1):
    return sin(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f86():
    return cos(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f87(arg1):
    return cos(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f88():
    return tan(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f89(arg1):
    return tan(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f90():
    return asin(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f91(arg1):
    return asin(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f92():
    return acos(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f93(arg1):
    return acos(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f94():
    return atan(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f95(arg1):
    return atan(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f96():
    return atan2(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f97(arg1):
    return atan2(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f98(arg1):
    return atan2(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f99(arg1):
    return atan2(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f100(arg1, arg2):
    return atan2(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f101(arg1, arg2):
    return atan2(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f102():
    return sinh(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f103(arg1):
    return sinh(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f104():
    return cosh(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f105(arg1):
    return cosh(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f106():
    return tanh(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f107(arg1):
    return tanh(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f108():
    return asinh(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f109(arg1):
    return asinh(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f110():
    return acosh(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f111(arg1):
    return acosh(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f112():
    return atanh(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f113(arg1):
    return atanh(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f114():
    return erf(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f115(arg1):
    return erf(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f116():
    return erfc(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f117(arg1):
    return erfc(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f118():
    return tgamma(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f119(arg1):
    return tgamma(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f120():
    return lgamma(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f121(arg1):
    return lgamma(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f122():
    return ceil(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f123(arg1):
    return ceil(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f124():
    return floor(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f125(arg1):
    return floor(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f126():
    return fmod(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f127(arg1):
    return fmod(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f128(arg1):
    return fmod(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f129(arg1):
    return fmod(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f130(arg1, arg2):
    return fmod(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f131(arg1, arg2):
    return fmod(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f132():
    return remainder(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f133(arg1):
    return remainder(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f134(arg1):
    return remainder(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f135(arg1):
    return remainder(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f136(arg1, arg2):
    return remainder(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f137(arg1, arg2):
    return remainder(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f138():
    return max(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f139(arg1):
    return max(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f140(arg1):
    return max(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f141(arg1):
    return max(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f142(arg1, arg2):
    return max(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f143(arg1, arg2):
    return max(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f144():
    return min(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f145(arg1):
    return min(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f146(arg1):
    return min(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f147(arg1):
    return min(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f148(arg1, arg2):
    return min(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f149(arg1, arg2):
    return min(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f150():
    return fdim(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f151(arg1):
    return fdim(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f152(arg1):
    return fdim(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f153(arg1):
    return fdim(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f154(arg1, arg2):
    return fdim(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f155(arg1, arg2):
    return fdim(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f156():
    return copysign(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f157(arg1):
    return copysign(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f158(arg1):
    return copysign(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f159(arg1):
    return copysign(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f160(arg1, arg2):
    return copysign(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f161(arg1, arg2):
    return copysign(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f162():
    return trunc(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f163(arg1):
    return trunc(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f164():
    return round(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f165(arg1):
    return round(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f166():
    return nearbyint(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f167(arg1):
    return nearbyint(arg1)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f168():
    return round(1.0)

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f169(arg1):
    return round(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f170():
    return (1.0 + 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f171(arg1):
    return (arg1 + 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f172(arg1):
    return (1.0 + arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f173(arg1):
    return (arg1 + arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f174(arg1, arg2):
    return (arg1 + arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f175(arg1, arg2):
    return (arg2 + arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f176():
    return (1.0 - 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f177(arg1):
    return (arg1 - 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f178(arg1):
    return (1.0 - arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f179(arg1):
    return (arg1 - arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f180(arg1, arg2):
    return (arg1 - arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f181(arg1, arg2):
    return (arg2 - arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f182():
    return (1.0 * 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f183(arg1):
    return (arg1 * 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f184(arg1):
    return (1.0 * arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f185(arg1):
    return (arg1 * arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f186(arg1, arg2):
    return (arg1 * arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f187(arg1, arg2):
    return (arg2 * arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f188():
    return (1.0 / 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f189(arg1):
    return (arg1 / 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f190(arg1):
    return (1.0 / arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f191(arg1):
    return (arg1 / arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f192(arg1, arg2):
    return (arg1 / arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f193(arg1, arg2):
    return (arg2 / arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f194():
    return fabs(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f195(arg1):
    return fabs(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f196():
    return fma(1.0, 1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f197(arg1):
    return fma(arg1, 1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f198(arg1):
    return fma(1.0, arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f199(arg1):
    return fma(1.0, 1.0, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f200(arg1):
    return fma(arg1, arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f201(arg1):
    return fma(arg1, 1.0, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f202(arg1):
    return fma(1.0, arg1, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f203(arg1, arg2):
    return fma(arg1, arg2, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f204(arg1, arg2):
    return fma(arg2, arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f205(arg1, arg2):
    return fma(arg1, 1.0, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f206(arg1, arg2):
    return fma(1.0, arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f207(arg1, arg2):
    return fma(arg2, 1.0, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f208(arg1, arg2):
    return fma(1.0, arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f209(arg1):
    return fma(arg1, arg1, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f210(arg1, arg2):
    return fma(arg1, arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f211(arg1, arg2):
    return fma(arg1, arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f212(arg1, arg2):
    return fma(arg2, arg1, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f213(arg1, arg2):
    return fma(arg1, arg2, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f214(arg1, arg2):
    return fma(arg2, arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f215(arg1, arg2):
    return fma(arg2, arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f216(arg1, arg2, arg3):
    return fma(arg1, arg2, arg3)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f217(arg1, arg2, arg3):
    return fma(arg2, arg1, arg3)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f218(arg1, arg2, arg3):
    return fma(arg1, arg3, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f219(arg1, arg2, arg3):
    return fma(arg3, arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f220(arg1, arg2, arg3):
    return fma(arg2, arg3, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f221(arg1, arg2, arg3):
    return fma(arg3, arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f222():
    return exp(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f223(arg1):
    return exp(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f224():
    return exp2(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f225(arg1):
    return exp2(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f226():
    return expm1(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f227(arg1):
    return expm1(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f228():
    return log(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f229(arg1):
    return log(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f230():
    return log10(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f231(arg1):
    return log10(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f232():
    return log2(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f233(arg1):
    return log2(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f234():
    return log1p(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f235(arg1):
    return log1p(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f236():
    return pow(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f237(arg1):
    return pow(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f238(arg1):
    return pow(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f239(arg1):
    return pow(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f240(arg1, arg2):
    return pow(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f241(arg1, arg2):
    return pow(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f242():
    return sqrt(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f243(arg1):
    return sqrt(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f244():
    return cbrt(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f245(arg1):
    return cbrt(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f246():
    return hypot(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f247(arg1):
    return hypot(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f248(arg1):
    return hypot(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f249(arg1):
    return hypot(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f250(arg1, arg2):
    return hypot(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f251(arg1, arg2):
    return hypot(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f252():
    return sin(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f253(arg1):
    return sin(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f254():
    return cos(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f255(arg1):
    return cos(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f256():
    return tan(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f257(arg1):
    return tan(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f258():
    return asin(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f259(arg1):
    return asin(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f260():
    return acos(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f261(arg1):
    return acos(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f262():
    return atan(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f263(arg1):
    return atan(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f264():
    return atan2(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f265(arg1):
    return atan2(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f266(arg1):
    return atan2(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f267(arg1):
    return atan2(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f268(arg1, arg2):
    return atan2(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f269(arg1, arg2):
    return atan2(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f270():
    return sinh(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f271(arg1):
    return sinh(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f272():
    return cosh(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f273(arg1):
    return cosh(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f274():
    return tanh(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f275(arg1):
    return tanh(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f276():
    return asinh(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f277(arg1):
    return asinh(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f278():
    return acosh(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f279(arg1):
    return acosh(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f280():
    return atanh(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f281(arg1):
    return atanh(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f282():
    return erf(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f283(arg1):
    return erf(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f284():
    return erfc(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f285(arg1):
    return erfc(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f286():
    return tgamma(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f287(arg1):
    return tgamma(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f288():
    return lgamma(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f289(arg1):
    return lgamma(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f290():
    return ceil(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f291(arg1):
    return ceil(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f292():
    return floor(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f293(arg1):
    return floor(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f294():
    return fmod(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f295(arg1):
    return fmod(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f296(arg1):
    return fmod(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f297(arg1):
    return fmod(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f298(arg1, arg2):
    return fmod(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f299(arg1, arg2):
    return fmod(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f300():
    return remainder(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f301(arg1):
    return remainder(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f302(arg1):
    return remainder(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f303(arg1):
    return remainder(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f304(arg1, arg2):
    return remainder(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f305(arg1, arg2):
    return remainder(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f306():
    return max(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f307(arg1):
    return max(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f308(arg1):
    return max(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f309(arg1):
    return max(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f310(arg1, arg2):
    return max(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f311(arg1, arg2):
    return max(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f312():
    return min(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f313(arg1):
    return min(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f314(arg1):
    return min(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f315(arg1):
    return min(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f316(arg1, arg2):
    return min(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f317(arg1, arg2):
    return min(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f318():
    return fdim(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f319(arg1):
    return fdim(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f320(arg1):
    return fdim(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f321(arg1):
    return fdim(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f322(arg1, arg2):
    return fdim(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f323(arg1, arg2):
    return fdim(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f324():
    return copysign(1.0, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f325(arg1):
    return copysign(arg1, 1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f326(arg1):
    return copysign(1.0, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f327(arg1):
    return copysign(arg1, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f328(arg1, arg2):
    return copysign(arg1, arg2)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f329(arg1, arg2):
    return copysign(arg2, arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f330():
    return trunc(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f331(arg1):
    return trunc(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f332():
    return round(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f333(arg1):
    return round(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f334():
    return nearbyint(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f335(arg1):
    return nearbyint(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f336():
    return round(1.0)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f337(arg1):
    return round(arg1)

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f338():
    return -1.0

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f339(arg1):
    return -arg1

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f340():
    if 1.0 < 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f341(arg1):
    if arg1 < 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f342(arg1):
    if 1.0 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f343(arg1):
    if arg1 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f344(arg1, arg2):
    if arg1 < arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f345(arg1, arg2):
    if arg2 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f346():
    if 1.0 > 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f347(arg1):
    if arg1 > 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f348(arg1):
    if 1.0 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f349(arg1):
    if arg1 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f350(arg1, arg2):
    if arg1 > arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f351(arg1, arg2):
    if arg2 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f352():
    if 1.0 <= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f353(arg1):
    if arg1 <= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f354(arg1):
    if 1.0 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f355(arg1):
    if arg1 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f356(arg1, arg2):
    if arg1 <= arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f357(arg1, arg2):
    if arg2 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f358():
    if 1.0 >= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f359(arg1):
    if arg1 >= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f360(arg1):
    if 1.0 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f361(arg1):
    if arg1 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f362(arg1, arg2):
    if arg1 >= arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f363(arg1, arg2):
    if arg2 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f364():
    if 1.0 == 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f365(arg1):
    if arg1 == 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f366(arg1):
    if 1.0 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f367(arg1):
    if arg1 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f368(arg1, arg2):
    if arg1 == arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f369(arg1, arg2):
    if arg2 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f370():
    if 1.0 != 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f371(arg1):
    if arg1 != 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f372(arg1):
    if 1.0 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f373(arg1):
    if arg1 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f374(arg1, arg2):
    if arg1 != arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f375(arg1, arg2):
    if arg2 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f376():
    if isfinite(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f377(arg1):
    if isfinite(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f378():
    if isinf(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f379(arg1):
    if isinf(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f380():
    if isnan(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f381(arg1):
    if isnan(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f382():
    if isnormal(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f383(arg1):
    if isnormal(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f384():
    if signbit(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f385(arg1):
    if signbit(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f386():
    if 1.0 < 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f387(arg1):
    if arg1 < 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f388(arg1):
    if 1.0 < arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f389(arg1):
    if arg1 < arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f390(arg1, arg2):
    if arg1 < arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f391(arg1, arg2):
    if arg2 < arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f392():
    if 1.0 > 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f393(arg1):
    if arg1 > 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f394(arg1):
    if 1.0 > arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f395(arg1):
    if arg1 > arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f396(arg1, arg2):
    if arg1 > arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f397(arg1, arg2):
    if arg2 > arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f398():
    if 1.0 <= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f399(arg1):
    if arg1 <= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f400(arg1):
    if 1.0 <= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f401(arg1):
    if arg1 <= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f402(arg1, arg2):
    if arg1 <= arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f403(arg1, arg2):
    if arg2 <= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f404():
    if 1.0 >= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f405(arg1):
    if arg1 >= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f406(arg1):
    if 1.0 >= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f407(arg1):
    if arg1 >= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f408(arg1, arg2):
    if arg1 >= arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f409(arg1, arg2):
    if arg2 >= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f410():
    if 1.0 == 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f411(arg1):
    if arg1 == 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f412(arg1):
    if 1.0 == arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f413(arg1):
    if arg1 == arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f414(arg1, arg2):
    if arg1 == arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f415(arg1, arg2):
    if arg2 == arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f416():
    if 1.0 != 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f417(arg1):
    if arg1 != 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f418(arg1):
    if 1.0 != arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f419(arg1):
    if arg1 != arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f420(arg1, arg2):
    if arg1 != arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f421(arg1, arg2):
    if arg2 != arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f422():
    if isfinite(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f423(arg1):
    if isfinite(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f424():
    if isinf(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f425(arg1):
    if isinf(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f426():
    if isnan(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f427(arg1):
    if isnan(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f428():
    if isnormal(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f429(arg1):
    if isnormal(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f430():
    if signbit(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    ctx=IEEEContext(es=15, nbits=79, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0),
    meta={
    }
)
def f431(arg1):
    if signbit(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(
    meta={
    }
)
def f432():
    if 1.0 < 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f433(arg1):
    if arg1 < 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f434(arg1):
    if 1.0 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f435(arg1):
    if arg1 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f436(arg1, arg2):
    if arg1 < arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f437(arg1, arg2):
    if arg2 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f438():
    if 1.0 > 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f439(arg1):
    if arg1 > 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f440(arg1):
    if 1.0 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f441(arg1):
    if arg1 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f442(arg1, arg2):
    if arg1 > arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f443(arg1, arg2):
    if arg2 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f444():
    if 1.0 <= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f445(arg1):
    if arg1 <= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f446(arg1):
    if 1.0 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f447(arg1):
    if arg1 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f448(arg1, arg2):
    if arg1 <= arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f449(arg1, arg2):
    if arg2 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f450():
    if 1.0 >= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f451(arg1):
    if arg1 >= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f452(arg1):
    if 1.0 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f453(arg1):
    if arg1 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f454(arg1, arg2):
    if arg1 >= arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f455(arg1, arg2):
    if arg2 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f456():
    if 1.0 == 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f457(arg1):
    if arg1 == 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f458(arg1):
    if 1.0 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f459(arg1):
    if arg1 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f460(arg1, arg2):
    if arg1 == arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f461(arg1, arg2):
    if arg2 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f462():
    if 1.0 != 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f463(arg1):
    if arg1 != 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f464(arg1):
    if 1.0 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f465(arg1):
    if arg1 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f466(arg1, arg2):
    if arg1 != arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f467(arg1, arg2):
    if arg2 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f468():
    if isfinite(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f469(arg1):
    if isfinite(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f470():
    if isinf(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f471(arg1):
    if isinf(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f472():
    if isnan(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f473(arg1):
    if isnan(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f474():
    if isnormal(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f475(arg1):
    if isnormal(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f476():
    if signbit(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f477(arg1):
    if signbit(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    meta={
    }
)
def f478():
    return -1.0

@fpy(
    meta={
    }
)
def f479(arg1):
    return -arg1

@fpy(
    meta={
    }
)
def f480():
    return (1.0 + 1.0)

@fpy(
    meta={
    }
)
def f481(arg1):
    return (arg1 + 1.0)

@fpy(
    meta={
    }
)
def f482(arg1):
    return (1.0 + arg1)

@fpy(
    meta={
    }
)
def f483(arg1):
    return (arg1 + arg1)

@fpy(
    meta={
    }
)
def f484(arg1, arg2):
    return (arg1 + arg2)

@fpy(
    meta={
    }
)
def f485(arg1, arg2):
    return (arg2 + arg1)

@fpy(
    meta={
    }
)
def f486():
    return (1.0 - 1.0)

@fpy(
    meta={
    }
)
def f487(arg1):
    return (arg1 - 1.0)

@fpy(
    meta={
    }
)
def f488(arg1):
    return (1.0 - arg1)

@fpy(
    meta={
    }
)
def f489(arg1):
    return (arg1 - arg1)

@fpy(
    meta={
    }
)
def f490(arg1, arg2):
    return (arg1 - arg2)

@fpy(
    meta={
    }
)
def f491(arg1, arg2):
    return (arg2 - arg1)

@fpy(
    meta={
    }
)
def f492():
    return (1.0 * 1.0)

@fpy(
    meta={
    }
)
def f493(arg1):
    return (arg1 * 1.0)

@fpy(
    meta={
    }
)
def f494(arg1):
    return (1.0 * arg1)

@fpy(
    meta={
    }
)
def f495(arg1):
    return (arg1 * arg1)

@fpy(
    meta={
    }
)
def f496(arg1, arg2):
    return (arg1 * arg2)

@fpy(
    meta={
    }
)
def f497(arg1, arg2):
    return (arg2 * arg1)

@fpy(
    meta={
    }
)
def f498():
    return (1.0 / 1.0)

@fpy(
    meta={
    }
)
def f499(arg1):
    return (arg1 / 1.0)

@fpy(
    meta={
    }
)
def f500(arg1):
    return (1.0 / arg1)

@fpy(
    meta={
    }
)
def f501(arg1):
    return (arg1 / arg1)

@fpy(
    meta={
    }
)
def f502(arg1, arg2):
    return (arg1 / arg2)

@fpy(
    meta={
    }
)
def f503(arg1, arg2):
    return (arg2 / arg1)

@fpy(
    meta={
    }
)
def f504():
    return fabs(1.0)

@fpy(
    meta={
    }
)
def f505(arg1):
    return fabs(arg1)

@fpy(
    meta={
    }
)
def f506():
    return fma(1.0, 1.0, 1.0)

@fpy(
    meta={
    }
)
def f507(arg1):
    return fma(arg1, 1.0, 1.0)

@fpy(
    meta={
    }
)
def f508(arg1):
    return fma(1.0, arg1, 1.0)

@fpy(
    meta={
    }
)
def f509(arg1):
    return fma(1.0, 1.0, arg1)

@fpy(
    meta={
    }
)
def f510(arg1):
    return fma(arg1, arg1, 1.0)

@fpy(
    meta={
    }
)
def f511(arg1):
    return fma(arg1, 1.0, arg1)

@fpy(
    meta={
    }
)
def f512(arg1):
    return fma(1.0, arg1, arg1)

@fpy(
    meta={
    }
)
def f513(arg1, arg2):
    return fma(arg1, arg2, 1.0)

@fpy(
    meta={
    }
)
def f514(arg1, arg2):
    return fma(arg2, arg1, 1.0)

@fpy(
    meta={
    }
)
def f515(arg1, arg2):
    return fma(arg1, 1.0, arg2)

@fpy(
    meta={
    }
)
def f516(arg1, arg2):
    return fma(1.0, arg1, arg2)

@fpy(
    meta={
    }
)
def f517(arg1, arg2):
    return fma(arg2, 1.0, arg1)

@fpy(
    meta={
    }
)
def f518(arg1, arg2):
    return fma(1.0, arg2, arg1)

@fpy(
    meta={
    }
)
def f519(arg1):
    return fma(arg1, arg1, arg1)

@fpy(
    meta={
    }
)
def f520(arg1, arg2):
    return fma(arg1, arg1, arg2)

@fpy(
    meta={
    }
)
def f521(arg1, arg2):
    return fma(arg1, arg2, arg1)

@fpy(
    meta={
    }
)
def f522(arg1, arg2):
    return fma(arg2, arg1, arg1)

@fpy(
    meta={
    }
)
def f523(arg1, arg2):
    return fma(arg1, arg2, arg2)

@fpy(
    meta={
    }
)
def f524(arg1, arg2):
    return fma(arg2, arg1, arg2)

@fpy(
    meta={
    }
)
def f525(arg1, arg2):
    return fma(arg2, arg2, arg1)

@fpy(
    meta={
    }
)
def f526(arg1, arg2, arg3):
    return fma(arg1, arg2, arg3)

@fpy(
    meta={
    }
)
def f527(arg1, arg2, arg3):
    return fma(arg2, arg1, arg3)

@fpy(
    meta={
    }
)
def f528(arg1, arg2, arg3):
    return fma(arg1, arg3, arg2)

@fpy(
    meta={
    }
)
def f529(arg1, arg2, arg3):
    return fma(arg3, arg1, arg2)

@fpy(
    meta={
    }
)
def f530(arg1, arg2, arg3):
    return fma(arg2, arg3, arg1)

@fpy(
    meta={
    }
)
def f531(arg1, arg2, arg3):
    return fma(arg3, arg2, arg1)

@fpy(
    meta={
    }
)
def f532():
    return exp(1.0)

@fpy(
    meta={
    }
)
def f533(arg1):
    return exp(arg1)

@fpy(
    meta={
    }
)
def f534():
    return exp2(1.0)

@fpy(
    meta={
    }
)
def f535(arg1):
    return exp2(arg1)

@fpy(
    meta={
    }
)
def f536():
    return expm1(1.0)

@fpy(
    meta={
    }
)
def f537(arg1):
    return expm1(arg1)

@fpy(
    meta={
    }
)
def f538():
    return log(1.0)

@fpy(
    meta={
    }
)
def f539(arg1):
    return log(arg1)

@fpy(
    meta={
    }
)
def f540():
    return log10(1.0)

@fpy(
    meta={
    }
)
def f541(arg1):
    return log10(arg1)

@fpy(
    meta={
    }
)
def f542():
    return log2(1.0)

@fpy(
    meta={
    }
)
def f543(arg1):
    return log2(arg1)

@fpy(
    meta={
    }
)
def f544():
    return log1p(1.0)

@fpy(
    meta={
    }
)
def f545(arg1):
    return log1p(arg1)

@fpy(
    meta={
    }
)
def f546():
    return pow(1.0, 1.0)

@fpy(
    meta={
    }
)
def f547(arg1):
    return pow(arg1, 1.0)

@fpy(
    meta={
    }
)
def f548(arg1):
    return pow(1.0, arg1)

@fpy(
    meta={
    }
)
def f549(arg1):
    return pow(arg1, arg1)

@fpy(
    meta={
    }
)
def f550(arg1, arg2):
    return pow(arg1, arg2)

@fpy(
    meta={
    }
)
def f551(arg1, arg2):
    return pow(arg2, arg1)

@fpy(
    meta={
    }
)
def f552():
    return sqrt(1.0)

@fpy(
    meta={
    }
)
def f553(arg1):
    return sqrt(arg1)

@fpy(
    meta={
    }
)
def f554():
    return cbrt(1.0)

@fpy(
    meta={
    }
)
def f555(arg1):
    return cbrt(arg1)

@fpy(
    meta={
    }
)
def f556():
    return hypot(1.0, 1.0)

@fpy(
    meta={
    }
)
def f557(arg1):
    return hypot(arg1, 1.0)

@fpy(
    meta={
    }
)
def f558(arg1):
    return hypot(1.0, arg1)

@fpy(
    meta={
    }
)
def f559(arg1):
    return hypot(arg1, arg1)

@fpy(
    meta={
    }
)
def f560(arg1, arg2):
    return hypot(arg1, arg2)

@fpy(
    meta={
    }
)
def f561(arg1, arg2):
    return hypot(arg2, arg1)

@fpy(
    meta={
    }
)
def f562():
    return sin(1.0)

@fpy(
    meta={
    }
)
def f563(arg1):
    return sin(arg1)

@fpy(
    meta={
    }
)
def f564():
    return cos(1.0)

@fpy(
    meta={
    }
)
def f565(arg1):
    return cos(arg1)

@fpy(
    meta={
    }
)
def f566():
    return tan(1.0)

@fpy(
    meta={
    }
)
def f567(arg1):
    return tan(arg1)

@fpy(
    meta={
    }
)
def f568():
    return asin(1.0)

@fpy(
    meta={
    }
)
def f569(arg1):
    return asin(arg1)

@fpy(
    meta={
    }
)
def f570():
    return acos(1.0)

@fpy(
    meta={
    }
)
def f571(arg1):
    return acos(arg1)

@fpy(
    meta={
    }
)
def f572():
    return atan(1.0)

@fpy(
    meta={
    }
)
def f573(arg1):
    return atan(arg1)

@fpy(
    meta={
    }
)
def f574():
    return atan2(1.0, 1.0)

@fpy(
    meta={
    }
)
def f575(arg1):
    return atan2(arg1, 1.0)

@fpy(
    meta={
    }
)
def f576(arg1):
    return atan2(1.0, arg1)

@fpy(
    meta={
    }
)
def f577(arg1):
    return atan2(arg1, arg1)

@fpy(
    meta={
    }
)
def f578(arg1, arg2):
    return atan2(arg1, arg2)

@fpy(
    meta={
    }
)
def f579(arg1, arg2):
    return atan2(arg2, arg1)

@fpy(
    meta={
    }
)
def f580():
    return sinh(1.0)

@fpy(
    meta={
    }
)
def f581(arg1):
    return sinh(arg1)

@fpy(
    meta={
    }
)
def f582():
    return cosh(1.0)

@fpy(
    meta={
    }
)
def f583(arg1):
    return cosh(arg1)

@fpy(
    meta={
    }
)
def f584():
    return tanh(1.0)

@fpy(
    meta={
    }
)
def f585(arg1):
    return tanh(arg1)

@fpy(
    meta={
    }
)
def f586():
    return asinh(1.0)

@fpy(
    meta={
    }
)
def f587(arg1):
    return asinh(arg1)

@fpy(
    meta={
    }
)
def f588():
    return acosh(1.0)

@fpy(
    meta={
    }
)
def f589(arg1):
    return acosh(arg1)

@fpy(
    meta={
    }
)
def f590():
    return atanh(1.0)

@fpy(
    meta={
    }
)
def f591(arg1):
    return atanh(arg1)

@fpy(
    meta={
    }
)
def f592():
    return erf(1.0)

@fpy(
    meta={
    }
)
def f593(arg1):
    return erf(arg1)

@fpy(
    meta={
    }
)
def f594():
    return erfc(1.0)

@fpy(
    meta={
    }
)
def f595(arg1):
    return erfc(arg1)

@fpy(
    meta={
    }
)
def f596():
    return tgamma(1.0)

@fpy(
    meta={
    }
)
def f597(arg1):
    return tgamma(arg1)

@fpy(
    meta={
    }
)
def f598():
    return lgamma(1.0)

@fpy(
    meta={
    }
)
def f599(arg1):
    return lgamma(arg1)

@fpy(
    meta={
    }
)
def f600():
    return ceil(1.0)

@fpy(
    meta={
    }
)
def f601(arg1):
    return ceil(arg1)

@fpy(
    meta={
    }
)
def f602():
    return floor(1.0)

@fpy(
    meta={
    }
)
def f603(arg1):
    return floor(arg1)

@fpy(
    meta={
    }
)
def f604():
    return fmod(1.0, 1.0)

@fpy(
    meta={
    }
)
def f605(arg1):
    return fmod(arg1, 1.0)

@fpy(
    meta={
    }
)
def f606(arg1):
    return fmod(1.0, arg1)

@fpy(
    meta={
    }
)
def f607(arg1):
    return fmod(arg1, arg1)

@fpy(
    meta={
    }
)
def f608(arg1, arg2):
    return fmod(arg1, arg2)

@fpy(
    meta={
    }
)
def f609(arg1, arg2):
    return fmod(arg2, arg1)

@fpy(
    meta={
    }
)
def f610():
    return remainder(1.0, 1.0)

@fpy(
    meta={
    }
)
def f611(arg1):
    return remainder(arg1, 1.0)

@fpy(
    meta={
    }
)
def f612(arg1):
    return remainder(1.0, arg1)

@fpy(
    meta={
    }
)
def f613(arg1):
    return remainder(arg1, arg1)

@fpy(
    meta={
    }
)
def f614(arg1, arg2):
    return remainder(arg1, arg2)

@fpy(
    meta={
    }
)
def f615(arg1, arg2):
    return remainder(arg2, arg1)

@fpy(
    meta={
    }
)
def f616():
    return max(1.0, 1.0)

@fpy(
    meta={
    }
)
def f617(arg1):
    return max(arg1, 1.0)

@fpy(
    meta={
    }
)
def f618(arg1):
    return max(1.0, arg1)

@fpy(
    meta={
    }
)
def f619(arg1):
    return max(arg1, arg1)

@fpy(
    meta={
    }
)
def f620(arg1, arg2):
    return max(arg1, arg2)

@fpy(
    meta={
    }
)
def f621(arg1, arg2):
    return max(arg2, arg1)

@fpy(
    meta={
    }
)
def f622():
    return min(1.0, 1.0)

@fpy(
    meta={
    }
)
def f623(arg1):
    return min(arg1, 1.0)

@fpy(
    meta={
    }
)
def f624(arg1):
    return min(1.0, arg1)

@fpy(
    meta={
    }
)
def f625(arg1):
    return min(arg1, arg1)

@fpy(
    meta={
    }
)
def f626(arg1, arg2):
    return min(arg1, arg2)

@fpy(
    meta={
    }
)
def f627(arg1, arg2):
    return min(arg2, arg1)

@fpy(
    meta={
    }
)
def f628():
    return fdim(1.0, 1.0)

@fpy(
    meta={
    }
)
def f629(arg1):
    return fdim(arg1, 1.0)

@fpy(
    meta={
    }
)
def f630(arg1):
    return fdim(1.0, arg1)

@fpy(
    meta={
    }
)
def f631(arg1):
    return fdim(arg1, arg1)

@fpy(
    meta={
    }
)
def f632(arg1, arg2):
    return fdim(arg1, arg2)

@fpy(
    meta={
    }
)
def f633(arg1, arg2):
    return fdim(arg2, arg1)

@fpy(
    meta={
    }
)
def f634():
    return copysign(1.0, 1.0)

@fpy(
    meta={
    }
)
def f635(arg1):
    return copysign(arg1, 1.0)

@fpy(
    meta={
    }
)
def f636(arg1):
    return copysign(1.0, arg1)

@fpy(
    meta={
    }
)
def f637(arg1):
    return copysign(arg1, arg1)

@fpy(
    meta={
    }
)
def f638(arg1, arg2):
    return copysign(arg1, arg2)

@fpy(
    meta={
    }
)
def f639(arg1, arg2):
    return copysign(arg2, arg1)

@fpy(
    meta={
    }
)
def f640():
    return trunc(1.0)

@fpy(
    meta={
    }
)
def f641(arg1):
    return trunc(arg1)

@fpy(
    meta={
    }
)
def f642():
    return round(1.0)

@fpy(
    meta={
    }
)
def f643(arg1):
    return round(arg1)

@fpy(
    meta={
    }
)
def f644():
    return nearbyint(1.0)

@fpy(
    meta={
    }
)
def f645(arg1):
    return nearbyint(arg1)

@fpy(
    meta={
    }
)
def f646():
    return round(1.0)

@fpy(
    meta={
    }
)
def f647(arg1):
    return round(arg1)

@fpy(
    meta={
        'spec': lambda a, b: (b - a),
    }
)
def f648(a, b):
    a0 = b
    b1 = a
    return (a0 - b1)

@fpy(
    meta={
        'spec': lambda a, b: (b - b),
    }
)
def f649(a, b):
    a0 = b
    b1 = a0
    return (a0 - b1)

@fpy(
    meta={
        'spec': lambda a, b: 1,
    }
)
def f650(a, b):
    a0 = 1
    b1 = a0
    return b1

@fpy(
    meta={
        'pre': lambda a: 1 < a < 1000,
    }
)
def f651(a):
    c = 0
    d = 0
    while c < a:
        t = (1 + c)
        t0 = (d + c)
        c = t
        d = t0
    return d

@fpy(
    meta={
        'pre': lambda a: 1 < a < 1000,
    }
)
def f652(a):
    c = 0
    d = 0
    while c < a:
        c = (1 + c)
        d = (d + c)
    return d

