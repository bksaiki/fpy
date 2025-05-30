from fpy2 import *
from fpy2.typing import *

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f0():
    return -1.0

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f1(arg1):
    return -arg1

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f2():
    return (1.0 + 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f3(arg1):
    return (arg1 + 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f4(arg1):
    return (1.0 + arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f5(arg1):
    return (arg1 + arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f6(arg1, arg2):
    return (arg1 + arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f7(arg1, arg2):
    return (arg2 + arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f8():
    return (1.0 - 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f9(arg1):
    return (arg1 - 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f10(arg1):
    return (1.0 - arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f11(arg1):
    return (arg1 - arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f12(arg1, arg2):
    return (arg1 - arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f13(arg1, arg2):
    return (arg2 - arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f14():
    return (1.0 * 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f15(arg1):
    return (arg1 * 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f16(arg1):
    return (1.0 * arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f17(arg1):
    return (arg1 * arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f18(arg1, arg2):
    return (arg1 * arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f19(arg1, arg2):
    return (arg2 * arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f20():
    return (1.0 / 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f21(arg1):
    return (arg1 / 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f22(arg1):
    return (1.0 / arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f23(arg1):
    return (arg1 / arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f24(arg1, arg2):
    return (arg1 / arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f25(arg1, arg2):
    return (arg2 / arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f26():
    return fabs(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f27(arg1):
    return fabs(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f28():
    return fma(1.0, 1.0, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f29(arg1):
    return fma(arg1, 1.0, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f30(arg1):
    return fma(1.0, arg1, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f31(arg1):
    return fma(1.0, 1.0, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f32(arg1):
    return fma(arg1, arg1, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f33(arg1):
    return fma(arg1, 1.0, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f34(arg1):
    return fma(1.0, arg1, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f35(arg1, arg2):
    return fma(arg1, arg2, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f36(arg1, arg2):
    return fma(arg2, arg1, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f37(arg1, arg2):
    return fma(arg1, 1.0, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f38(arg1, arg2):
    return fma(1.0, arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f39(arg1, arg2):
    return fma(arg2, 1.0, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f40(arg1, arg2):
    return fma(1.0, arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f41(arg1):
    return fma(arg1, arg1, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f42(arg1, arg2):
    return fma(arg1, arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f43(arg1, arg2):
    return fma(arg1, arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f44(arg1, arg2):
    return fma(arg2, arg1, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f45(arg1, arg2):
    return fma(arg1, arg2, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f46(arg1, arg2):
    return fma(arg2, arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f47(arg1, arg2):
    return fma(arg2, arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f48(arg1, arg2, arg3):
    return fma(arg1, arg2, arg3)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f49(arg1, arg2, arg3):
    return fma(arg2, arg1, arg3)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f50(arg1, arg2, arg3):
    return fma(arg1, arg3, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f51(arg1, arg2, arg3):
    return fma(arg3, arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f52(arg1, arg2, arg3):
    return fma(arg2, arg3, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f53(arg1, arg2, arg3):
    return fma(arg3, arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f54():
    return exp(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f55(arg1):
    return exp(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f56():
    return exp2(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f57(arg1):
    return exp2(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f58():
    return expm1(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f59(arg1):
    return expm1(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f60():
    return log(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f61(arg1):
    return log(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f62():
    return log10(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f63(arg1):
    return log10(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f64():
    return log2(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f65(arg1):
    return log2(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f66():
    return log1p(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f67(arg1):
    return log1p(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f68():
    return pow(1.0, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f69(arg1):
    return pow(arg1, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f70(arg1):
    return pow(1.0, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f71(arg1):
    return pow(arg1, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f72(arg1, arg2):
    return pow(arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f73(arg1, arg2):
    return pow(arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f74():
    return sqrt(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f75(arg1):
    return sqrt(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f76():
    return cbrt(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f77(arg1):
    return cbrt(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f78():
    return hypot(1.0, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f79(arg1):
    return hypot(arg1, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f80(arg1):
    return hypot(1.0, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f81(arg1):
    return hypot(arg1, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f82(arg1, arg2):
    return hypot(arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f83(arg1, arg2):
    return hypot(arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f84():
    return sin(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f85(arg1):
    return sin(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f86():
    return cos(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f87(arg1):
    return cos(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f88():
    return tan(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f89(arg1):
    return tan(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f90():
    return asin(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f91(arg1):
    return asin(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f92():
    return acos(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f93(arg1):
    return acos(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f94():
    return atan(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f95(arg1):
    return atan(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f96():
    return atan2(1.0, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f97(arg1):
    return atan2(arg1, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f98(arg1):
    return atan2(1.0, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f99(arg1):
    return atan2(arg1, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f100(arg1, arg2):
    return atan2(arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f101(arg1, arg2):
    return atan2(arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f102():
    return sinh(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f103(arg1):
    return sinh(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f104():
    return cosh(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f105(arg1):
    return cosh(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f106():
    return tanh(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f107(arg1):
    return tanh(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f108():
    return asinh(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f109(arg1):
    return asinh(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f110():
    return acosh(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f111(arg1):
    return acosh(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f112():
    return atanh(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f113(arg1):
    return atanh(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f114():
    return erf(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f115(arg1):
    return erf(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f116():
    return erfc(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f117(arg1):
    return erfc(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f118():
    return tgamma(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f119(arg1):
    return tgamma(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f120():
    return lgamma(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f121(arg1):
    return lgamma(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f122():
    return ceil(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f123(arg1):
    return ceil(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f124():
    return floor(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f125(arg1):
    return floor(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f126():
    return fmod(1.0, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f127(arg1):
    return fmod(arg1, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f128(arg1):
    return fmod(1.0, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f129(arg1):
    return fmod(arg1, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f130(arg1, arg2):
    return fmod(arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f131(arg1, arg2):
    return fmod(arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f132():
    return remainder(1.0, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f133(arg1):
    return remainder(arg1, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f134(arg1):
    return remainder(1.0, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f135(arg1):
    return remainder(arg1, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f136(arg1, arg2):
    return remainder(arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f137(arg1, arg2):
    return remainder(arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f138():
    return fmax(1.0, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f139(arg1):
    return fmax(arg1, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f140(arg1):
    return fmax(1.0, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f141(arg1):
    return fmax(arg1, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f142(arg1, arg2):
    return fmax(arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f143(arg1, arg2):
    return fmax(arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f144():
    return fmin(1.0, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f145(arg1):
    return fmin(arg1, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f146(arg1):
    return fmin(1.0, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f147(arg1):
    return fmin(arg1, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f148(arg1, arg2):
    return fmin(arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f149(arg1, arg2):
    return fmin(arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f150():
    return fdim(1.0, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f151(arg1):
    return fdim(arg1, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f152(arg1):
    return fdim(1.0, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f153(arg1):
    return fdim(arg1, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f154(arg1, arg2):
    return fdim(arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f155(arg1, arg2):
    return fdim(arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f156():
    return copysign(1.0, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f157(arg1):
    return copysign(arg1, 1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f158(arg1):
    return copysign(1.0, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f159(arg1):
    return copysign(arg1, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f160(arg1, arg2):
    return copysign(arg1, arg2)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f161(arg1, arg2):
    return copysign(arg2, arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f162():
    return trunc(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f163(arg1):
    return trunc(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f164():
    return round(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f165(arg1):
    return round(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f166():
    return nearbyint(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f167(arg1):
    return nearbyint(arg1)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f168():
    return cast(1.0)

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f169(arg1):
    return cast(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f170():
    return (1.0 + 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f171(arg1):
    return (arg1 + 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f172(arg1):
    return (1.0 + arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f173(arg1):
    return (arg1 + arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f174(arg1, arg2):
    return (arg1 + arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f175(arg1, arg2):
    return (arg2 + arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f176():
    return (1.0 - 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f177(arg1):
    return (arg1 - 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f178(arg1):
    return (1.0 - arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f179(arg1):
    return (arg1 - arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f180(arg1, arg2):
    return (arg1 - arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f181(arg1, arg2):
    return (arg2 - arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f182():
    return (1.0 * 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f183(arg1):
    return (arg1 * 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f184(arg1):
    return (1.0 * arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f185(arg1):
    return (arg1 * arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f186(arg1, arg2):
    return (arg1 * arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f187(arg1, arg2):
    return (arg2 * arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f188():
    return (1.0 / 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f189(arg1):
    return (arg1 / 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f190(arg1):
    return (1.0 / arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f191(arg1):
    return (arg1 / arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f192(arg1, arg2):
    return (arg1 / arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f193(arg1, arg2):
    return (arg2 / arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f194():
    return fabs(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f195(arg1):
    return fabs(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f196():
    return fma(1.0, 1.0, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f197(arg1):
    return fma(arg1, 1.0, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f198(arg1):
    return fma(1.0, arg1, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f199(arg1):
    return fma(1.0, 1.0, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f200(arg1):
    return fma(arg1, arg1, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f201(arg1):
    return fma(arg1, 1.0, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f202(arg1):
    return fma(1.0, arg1, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f203(arg1, arg2):
    return fma(arg1, arg2, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f204(arg1, arg2):
    return fma(arg2, arg1, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f205(arg1, arg2):
    return fma(arg1, 1.0, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f206(arg1, arg2):
    return fma(1.0, arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f207(arg1, arg2):
    return fma(arg2, 1.0, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f208(arg1, arg2):
    return fma(1.0, arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f209(arg1):
    return fma(arg1, arg1, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f210(arg1, arg2):
    return fma(arg1, arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f211(arg1, arg2):
    return fma(arg1, arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f212(arg1, arg2):
    return fma(arg2, arg1, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f213(arg1, arg2):
    return fma(arg1, arg2, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f214(arg1, arg2):
    return fma(arg2, arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f215(arg1, arg2):
    return fma(arg2, arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f216(arg1, arg2, arg3):
    return fma(arg1, arg2, arg3)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f217(arg1, arg2, arg3):
    return fma(arg2, arg1, arg3)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f218(arg1, arg2, arg3):
    return fma(arg1, arg3, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f219(arg1, arg2, arg3):
    return fma(arg3, arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f220(arg1, arg2, arg3):
    return fma(arg2, arg3, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f221(arg1, arg2, arg3):
    return fma(arg3, arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f222():
    return exp(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f223(arg1):
    return exp(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f224():
    return exp2(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f225(arg1):
    return exp2(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f226():
    return expm1(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f227(arg1):
    return expm1(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f228():
    return log(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f229(arg1):
    return log(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f230():
    return log10(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f231(arg1):
    return log10(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f232():
    return log2(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f233(arg1):
    return log2(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f234():
    return log1p(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f235(arg1):
    return log1p(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f236():
    return pow(1.0, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f237(arg1):
    return pow(arg1, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f238(arg1):
    return pow(1.0, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f239(arg1):
    return pow(arg1, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f240(arg1, arg2):
    return pow(arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f241(arg1, arg2):
    return pow(arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f242():
    return sqrt(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f243(arg1):
    return sqrt(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f244():
    return cbrt(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f245(arg1):
    return cbrt(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f246():
    return hypot(1.0, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f247(arg1):
    return hypot(arg1, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f248(arg1):
    return hypot(1.0, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f249(arg1):
    return hypot(arg1, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f250(arg1, arg2):
    return hypot(arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f251(arg1, arg2):
    return hypot(arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f252():
    return sin(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f253(arg1):
    return sin(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f254():
    return cos(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f255(arg1):
    return cos(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f256():
    return tan(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f257(arg1):
    return tan(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f258():
    return asin(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f259(arg1):
    return asin(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f260():
    return acos(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f261(arg1):
    return acos(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f262():
    return atan(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f263(arg1):
    return atan(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f264():
    return atan2(1.0, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f265(arg1):
    return atan2(arg1, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f266(arg1):
    return atan2(1.0, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f267(arg1):
    return atan2(arg1, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f268(arg1, arg2):
    return atan2(arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f269(arg1, arg2):
    return atan2(arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f270():
    return sinh(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f271(arg1):
    return sinh(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f272():
    return cosh(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f273(arg1):
    return cosh(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f274():
    return tanh(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f275(arg1):
    return tanh(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f276():
    return asinh(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f277(arg1):
    return asinh(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f278():
    return acosh(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f279(arg1):
    return acosh(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f280():
    return atanh(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f281(arg1):
    return atanh(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f282():
    return erf(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f283(arg1):
    return erf(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f284():
    return erfc(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f285(arg1):
    return erfc(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f286():
    return tgamma(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f287(arg1):
    return tgamma(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f288():
    return lgamma(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f289(arg1):
    return lgamma(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f290():
    return ceil(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f291(arg1):
    return ceil(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f292():
    return floor(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f293(arg1):
    return floor(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f294():
    return fmod(1.0, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f295(arg1):
    return fmod(arg1, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f296(arg1):
    return fmod(1.0, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f297(arg1):
    return fmod(arg1, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f298(arg1, arg2):
    return fmod(arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f299(arg1, arg2):
    return fmod(arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f300():
    return remainder(1.0, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f301(arg1):
    return remainder(arg1, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f302(arg1):
    return remainder(1.0, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f303(arg1):
    return remainder(arg1, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f304(arg1, arg2):
    return remainder(arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f305(arg1, arg2):
    return remainder(arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f306():
    return fmax(1.0, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f307(arg1):
    return fmax(arg1, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f308(arg1):
    return fmax(1.0, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f309(arg1):
    return fmax(arg1, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f310(arg1, arg2):
    return fmax(arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f311(arg1, arg2):
    return fmax(arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f312():
    return fmin(1.0, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f313(arg1):
    return fmin(arg1, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f314(arg1):
    return fmin(1.0, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f315(arg1):
    return fmin(arg1, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f316(arg1, arg2):
    return fmin(arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f317(arg1, arg2):
    return fmin(arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f318():
    return fdim(1.0, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f319(arg1):
    return fdim(arg1, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f320(arg1):
    return fdim(1.0, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f321(arg1):
    return fdim(arg1, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f322(arg1, arg2):
    return fdim(arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f323(arg1, arg2):
    return fdim(arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f324():
    return copysign(1.0, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f325(arg1):
    return copysign(arg1, 1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f326(arg1):
    return copysign(1.0, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f327(arg1):
    return copysign(arg1, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f328(arg1, arg2):
    return copysign(arg1, arg2)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f329(arg1, arg2):
    return copysign(arg2, arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f330():
    return trunc(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f331(arg1):
    return trunc(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f332():
    return round(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f333(arg1):
    return round(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f334():
    return nearbyint(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f335(arg1):
    return nearbyint(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f336():
    return cast(1.0)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f337(arg1):
    return cast(arg1)

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f338():
    return -1.0

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f339(arg1):
    return -arg1

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f340():
    if 1.0 < 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f341(arg1):
    if arg1 < 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f342(arg1):
    if 1.0 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f343(arg1):
    if arg1 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f344(arg1, arg2):
    if arg1 < arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f345(arg1, arg2):
    if arg2 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f346():
    if 1.0 > 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f347(arg1):
    if arg1 > 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f348(arg1):
    if 1.0 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f349(arg1):
    if arg1 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f350(arg1, arg2):
    if arg1 > arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f351(arg1, arg2):
    if arg2 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f352():
    if 1.0 <= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f353(arg1):
    if arg1 <= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f354(arg1):
    if 1.0 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f355(arg1):
    if arg1 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f356(arg1, arg2):
    if arg1 <= arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f357(arg1, arg2):
    if arg2 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f358():
    if 1.0 >= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f359(arg1):
    if arg1 >= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f360(arg1):
    if 1.0 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f361(arg1):
    if arg1 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f362(arg1, arg2):
    if arg1 >= arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f363(arg1, arg2):
    if arg2 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f364():
    if 1.0 == 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f365(arg1):
    if arg1 == 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f366(arg1):
    if 1.0 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f367(arg1):
    if arg1 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f368(arg1, arg2):
    if arg1 == arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f369(arg1, arg2):
    if arg2 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f370():
    if 1.0 != 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f371(arg1):
    if arg1 != 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f372(arg1):
    if 1.0 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f373(arg1):
    if arg1 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f374(arg1, arg2):
    if arg1 != arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f375(arg1, arg2):
    if arg2 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f376():
    if isfinite(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f377(arg1):
    if isfinite(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f378():
    if isinf(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f379(arg1):
    if isinf(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f380():
    if isnan(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f381(arg1):
    if isnan(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f382():
    if isnormal(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f383(arg1):
    if isnormal(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f384():
    if signbit(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=8, nbits=32, rm=RoundingMode.RNE))
def f385(arg1):
    if signbit(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f386():
    if 1.0 < 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f387(arg1):
    if arg1 < 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f388(arg1):
    if 1.0 < arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f389(arg1):
    if arg1 < arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f390(arg1, arg2):
    if arg1 < arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f391(arg1, arg2):
    if arg2 < arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f392():
    if 1.0 > 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f393(arg1):
    if arg1 > 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f394(arg1):
    if 1.0 > arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f395(arg1):
    if arg1 > arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f396(arg1, arg2):
    if arg1 > arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f397(arg1, arg2):
    if arg2 > arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f398():
    if 1.0 <= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f399(arg1):
    if arg1 <= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f400(arg1):
    if 1.0 <= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f401(arg1):
    if arg1 <= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f402(arg1, arg2):
    if arg1 <= arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f403(arg1, arg2):
    if arg2 <= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f404():
    if 1.0 >= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f405(arg1):
    if arg1 >= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f406(arg1):
    if 1.0 >= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f407(arg1):
    if arg1 >= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f408(arg1, arg2):
    if arg1 >= arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f409(arg1, arg2):
    if arg2 >= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f410():
    if 1.0 == 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f411(arg1):
    if arg1 == 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f412(arg1):
    if 1.0 == arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f413(arg1):
    if arg1 == arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f414(arg1, arg2):
    if arg1 == arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f415(arg1, arg2):
    if arg2 == arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f416():
    if 1.0 != 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f417(arg1):
    if arg1 != 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f418(arg1):
    if 1.0 != arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f419(arg1):
    if arg1 != arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f420(arg1, arg2):
    if arg1 != arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f421(arg1, arg2):
    if arg2 != arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f422():
    if isfinite(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f423(arg1):
    if isfinite(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f424():
    if isinf(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f425(arg1):
    if isinf(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f426():
    if isnan(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f427(arg1):
    if isnan(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f428():
    if isnormal(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f429(arg1):
    if isnormal(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f430():
    if signbit(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(context=IEEEContext(es=15, nbits=80, rm=RoundingMode.RNE))
def f431(arg1):
    if signbit(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy
def f432():
    if 1.0 < 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f433(arg1):
    if arg1 < 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f434(arg1):
    if 1.0 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f435(arg1):
    if arg1 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f436(arg1, arg2):
    if arg1 < arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f437(arg1, arg2):
    if arg2 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f438():
    if 1.0 > 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f439(arg1):
    if arg1 > 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f440(arg1):
    if 1.0 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f441(arg1):
    if arg1 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f442(arg1, arg2):
    if arg1 > arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f443(arg1, arg2):
    if arg2 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f444():
    if 1.0 <= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f445(arg1):
    if arg1 <= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f446(arg1):
    if 1.0 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f447(arg1):
    if arg1 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f448(arg1, arg2):
    if arg1 <= arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f449(arg1, arg2):
    if arg2 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f450():
    if 1.0 >= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f451(arg1):
    if arg1 >= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f452(arg1):
    if 1.0 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f453(arg1):
    if arg1 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f454(arg1, arg2):
    if arg1 >= arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f455(arg1, arg2):
    if arg2 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f456():
    if 1.0 == 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f457(arg1):
    if arg1 == 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f458(arg1):
    if 1.0 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f459(arg1):
    if arg1 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f460(arg1, arg2):
    if arg1 == arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f461(arg1, arg2):
    if arg2 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f462():
    if 1.0 != 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f463(arg1):
    if arg1 != 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f464(arg1):
    if 1.0 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f465(arg1):
    if arg1 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f466(arg1, arg2):
    if arg1 != arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f467(arg1, arg2):
    if arg2 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f468():
    if isfinite(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f469(arg1):
    if isfinite(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f470():
    if isinf(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f471(arg1):
    if isinf(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f472():
    if isnan(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f473(arg1):
    if isnan(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f474():
    if isnormal(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f475(arg1):
    if isnormal(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f476():
    if signbit(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f477(arg1):
    if signbit(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy
def f478():
    return -1.0

@fpy
def f479(arg1):
    return -arg1

@fpy
def f480():
    return (1.0 + 1.0)

@fpy
def f481(arg1):
    return (arg1 + 1.0)

@fpy
def f482(arg1):
    return (1.0 + arg1)

@fpy
def f483(arg1):
    return (arg1 + arg1)

@fpy
def f484(arg1, arg2):
    return (arg1 + arg2)

@fpy
def f485(arg1, arg2):
    return (arg2 + arg1)

@fpy
def f486():
    return (1.0 - 1.0)

@fpy
def f487(arg1):
    return (arg1 - 1.0)

@fpy
def f488(arg1):
    return (1.0 - arg1)

@fpy
def f489(arg1):
    return (arg1 - arg1)

@fpy
def f490(arg1, arg2):
    return (arg1 - arg2)

@fpy
def f491(arg1, arg2):
    return (arg2 - arg1)

@fpy
def f492():
    return (1.0 * 1.0)

@fpy
def f493(arg1):
    return (arg1 * 1.0)

@fpy
def f494(arg1):
    return (1.0 * arg1)

@fpy
def f495(arg1):
    return (arg1 * arg1)

@fpy
def f496(arg1, arg2):
    return (arg1 * arg2)

@fpy
def f497(arg1, arg2):
    return (arg2 * arg1)

@fpy
def f498():
    return (1.0 / 1.0)

@fpy
def f499(arg1):
    return (arg1 / 1.0)

@fpy
def f500(arg1):
    return (1.0 / arg1)

@fpy
def f501(arg1):
    return (arg1 / arg1)

@fpy
def f502(arg1, arg2):
    return (arg1 / arg2)

@fpy
def f503(arg1, arg2):
    return (arg2 / arg1)

@fpy
def f504():
    return fabs(1.0)

@fpy
def f505(arg1):
    return fabs(arg1)

@fpy
def f506():
    return fma(1.0, 1.0, 1.0)

@fpy
def f507(arg1):
    return fma(arg1, 1.0, 1.0)

@fpy
def f508(arg1):
    return fma(1.0, arg1, 1.0)

@fpy
def f509(arg1):
    return fma(1.0, 1.0, arg1)

@fpy
def f510(arg1):
    return fma(arg1, arg1, 1.0)

@fpy
def f511(arg1):
    return fma(arg1, 1.0, arg1)

@fpy
def f512(arg1):
    return fma(1.0, arg1, arg1)

@fpy
def f513(arg1, arg2):
    return fma(arg1, arg2, 1.0)

@fpy
def f514(arg1, arg2):
    return fma(arg2, arg1, 1.0)

@fpy
def f515(arg1, arg2):
    return fma(arg1, 1.0, arg2)

@fpy
def f516(arg1, arg2):
    return fma(1.0, arg1, arg2)

@fpy
def f517(arg1, arg2):
    return fma(arg2, 1.0, arg1)

@fpy
def f518(arg1, arg2):
    return fma(1.0, arg2, arg1)

@fpy
def f519(arg1):
    return fma(arg1, arg1, arg1)

@fpy
def f520(arg1, arg2):
    return fma(arg1, arg1, arg2)

@fpy
def f521(arg1, arg2):
    return fma(arg1, arg2, arg1)

@fpy
def f522(arg1, arg2):
    return fma(arg2, arg1, arg1)

@fpy
def f523(arg1, arg2):
    return fma(arg1, arg2, arg2)

@fpy
def f524(arg1, arg2):
    return fma(arg2, arg1, arg2)

@fpy
def f525(arg1, arg2):
    return fma(arg2, arg2, arg1)

@fpy
def f526(arg1, arg2, arg3):
    return fma(arg1, arg2, arg3)

@fpy
def f527(arg1, arg2, arg3):
    return fma(arg2, arg1, arg3)

@fpy
def f528(arg1, arg2, arg3):
    return fma(arg1, arg3, arg2)

@fpy
def f529(arg1, arg2, arg3):
    return fma(arg3, arg1, arg2)

@fpy
def f530(arg1, arg2, arg3):
    return fma(arg2, arg3, arg1)

@fpy
def f531(arg1, arg2, arg3):
    return fma(arg3, arg2, arg1)

@fpy
def f532():
    return exp(1.0)

@fpy
def f533(arg1):
    return exp(arg1)

@fpy
def f534():
    return exp2(1.0)

@fpy
def f535(arg1):
    return exp2(arg1)

@fpy
def f536():
    return expm1(1.0)

@fpy
def f537(arg1):
    return expm1(arg1)

@fpy
def f538():
    return log(1.0)

@fpy
def f539(arg1):
    return log(arg1)

@fpy
def f540():
    return log10(1.0)

@fpy
def f541(arg1):
    return log10(arg1)

@fpy
def f542():
    return log2(1.0)

@fpy
def f543(arg1):
    return log2(arg1)

@fpy
def f544():
    return log1p(1.0)

@fpy
def f545(arg1):
    return log1p(arg1)

@fpy
def f546():
    return pow(1.0, 1.0)

@fpy
def f547(arg1):
    return pow(arg1, 1.0)

@fpy
def f548(arg1):
    return pow(1.0, arg1)

@fpy
def f549(arg1):
    return pow(arg1, arg1)

@fpy
def f550(arg1, arg2):
    return pow(arg1, arg2)

@fpy
def f551(arg1, arg2):
    return pow(arg2, arg1)

@fpy
def f552():
    return sqrt(1.0)

@fpy
def f553(arg1):
    return sqrt(arg1)

@fpy
def f554():
    return cbrt(1.0)

@fpy
def f555(arg1):
    return cbrt(arg1)

@fpy
def f556():
    return hypot(1.0, 1.0)

@fpy
def f557(arg1):
    return hypot(arg1, 1.0)

@fpy
def f558(arg1):
    return hypot(1.0, arg1)

@fpy
def f559(arg1):
    return hypot(arg1, arg1)

@fpy
def f560(arg1, arg2):
    return hypot(arg1, arg2)

@fpy
def f561(arg1, arg2):
    return hypot(arg2, arg1)

@fpy
def f562():
    return sin(1.0)

@fpy
def f563(arg1):
    return sin(arg1)

@fpy
def f564():
    return cos(1.0)

@fpy
def f565(arg1):
    return cos(arg1)

@fpy
def f566():
    return tan(1.0)

@fpy
def f567(arg1):
    return tan(arg1)

@fpy
def f568():
    return asin(1.0)

@fpy
def f569(arg1):
    return asin(arg1)

@fpy
def f570():
    return acos(1.0)

@fpy
def f571(arg1):
    return acos(arg1)

@fpy
def f572():
    return atan(1.0)

@fpy
def f573(arg1):
    return atan(arg1)

@fpy
def f574():
    return atan2(1.0, 1.0)

@fpy
def f575(arg1):
    return atan2(arg1, 1.0)

@fpy
def f576(arg1):
    return atan2(1.0, arg1)

@fpy
def f577(arg1):
    return atan2(arg1, arg1)

@fpy
def f578(arg1, arg2):
    return atan2(arg1, arg2)

@fpy
def f579(arg1, arg2):
    return atan2(arg2, arg1)

@fpy
def f580():
    return sinh(1.0)

@fpy
def f581(arg1):
    return sinh(arg1)

@fpy
def f582():
    return cosh(1.0)

@fpy
def f583(arg1):
    return cosh(arg1)

@fpy
def f584():
    return tanh(1.0)

@fpy
def f585(arg1):
    return tanh(arg1)

@fpy
def f586():
    return asinh(1.0)

@fpy
def f587(arg1):
    return asinh(arg1)

@fpy
def f588():
    return acosh(1.0)

@fpy
def f589(arg1):
    return acosh(arg1)

@fpy
def f590():
    return atanh(1.0)

@fpy
def f591(arg1):
    return atanh(arg1)

@fpy
def f592():
    return erf(1.0)

@fpy
def f593(arg1):
    return erf(arg1)

@fpy
def f594():
    return erfc(1.0)

@fpy
def f595(arg1):
    return erfc(arg1)

@fpy
def f596():
    return tgamma(1.0)

@fpy
def f597(arg1):
    return tgamma(arg1)

@fpy
def f598():
    return lgamma(1.0)

@fpy
def f599(arg1):
    return lgamma(arg1)

@fpy
def f600():
    return ceil(1.0)

@fpy
def f601(arg1):
    return ceil(arg1)

@fpy
def f602():
    return floor(1.0)

@fpy
def f603(arg1):
    return floor(arg1)

@fpy
def f604():
    return fmod(1.0, 1.0)

@fpy
def f605(arg1):
    return fmod(arg1, 1.0)

@fpy
def f606(arg1):
    return fmod(1.0, arg1)

@fpy
def f607(arg1):
    return fmod(arg1, arg1)

@fpy
def f608(arg1, arg2):
    return fmod(arg1, arg2)

@fpy
def f609(arg1, arg2):
    return fmod(arg2, arg1)

@fpy
def f610():
    return remainder(1.0, 1.0)

@fpy
def f611(arg1):
    return remainder(arg1, 1.0)

@fpy
def f612(arg1):
    return remainder(1.0, arg1)

@fpy
def f613(arg1):
    return remainder(arg1, arg1)

@fpy
def f614(arg1, arg2):
    return remainder(arg1, arg2)

@fpy
def f615(arg1, arg2):
    return remainder(arg2, arg1)

@fpy
def f616():
    return fmax(1.0, 1.0)

@fpy
def f617(arg1):
    return fmax(arg1, 1.0)

@fpy
def f618(arg1):
    return fmax(1.0, arg1)

@fpy
def f619(arg1):
    return fmax(arg1, arg1)

@fpy
def f620(arg1, arg2):
    return fmax(arg1, arg2)

@fpy
def f621(arg1, arg2):
    return fmax(arg2, arg1)

@fpy
def f622():
    return fmin(1.0, 1.0)

@fpy
def f623(arg1):
    return fmin(arg1, 1.0)

@fpy
def f624(arg1):
    return fmin(1.0, arg1)

@fpy
def f625(arg1):
    return fmin(arg1, arg1)

@fpy
def f626(arg1, arg2):
    return fmin(arg1, arg2)

@fpy
def f627(arg1, arg2):
    return fmin(arg2, arg1)

@fpy
def f628():
    return fdim(1.0, 1.0)

@fpy
def f629(arg1):
    return fdim(arg1, 1.0)

@fpy
def f630(arg1):
    return fdim(1.0, arg1)

@fpy
def f631(arg1):
    return fdim(arg1, arg1)

@fpy
def f632(arg1, arg2):
    return fdim(arg1, arg2)

@fpy
def f633(arg1, arg2):
    return fdim(arg2, arg1)

@fpy
def f634():
    return copysign(1.0, 1.0)

@fpy
def f635(arg1):
    return copysign(arg1, 1.0)

@fpy
def f636(arg1):
    return copysign(1.0, arg1)

@fpy
def f637(arg1):
    return copysign(arg1, arg1)

@fpy
def f638(arg1, arg2):
    return copysign(arg1, arg2)

@fpy
def f639(arg1, arg2):
    return copysign(arg2, arg1)

@fpy
def f640():
    return trunc(1.0)

@fpy
def f641(arg1):
    return trunc(arg1)

@fpy
def f642():
    return round(1.0)

@fpy
def f643(arg1):
    return round(arg1)

@fpy
def f644():
    return nearbyint(1.0)

@fpy
def f645(arg1):
    return nearbyint(arg1)

@fpy
def f646():
    return cast(1.0)

@fpy
def f647(arg1):
    return cast(arg1)

@fpy(spec=lambda a, b: (b - a))
def f648(a, b):
    a0 = b
    b1 = a
    return (a0 - b1)

@fpy(spec=lambda a, b: (b - b))
def f649(a, b):
    a0 = b
    b1 = a0
    return (a0 - b1)

@fpy(spec=lambda a, b: 1)
def f650(a, b):
    a0 = 1
    b1 = a0
    return b1

@fpy(pre=lambda a: 1 < a < 1000)
def f651(a):
    c = 0
    d = 0
    while c < a:
        t = (1 + c)
        t0 = (d + c)
        c = t
        d = t0
    return d

@fpy(pre=lambda a: 1 < a < 1000)
def f652(a):
    c = 0
    d = 0
    while c < a:
        c = (1 + c)
        d = (d + c)
    return d

