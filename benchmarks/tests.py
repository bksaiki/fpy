from fpy2 import *
from fpy2.typing import *

@fpy(precision='binary32')
def f0():
    return -1.0

@fpy(precision='binary32')
def f1(arg1):
    return -arg1

@fpy(precision='binary32')
def f2():
    return (1.0 + 1.0)

@fpy(precision='binary32')
def f3(arg1):
    return (arg1 + 1.0)

@fpy(precision='binary32')
def f4(arg1):
    return (1.0 + arg1)

@fpy(precision='binary32')
def f5(arg1):
    return (arg1 + arg1)

@fpy(precision='binary32')
def f6(arg1, arg2):
    return (arg1 + arg2)

@fpy(precision='binary32')
def f7(arg1, arg2):
    return (arg2 + arg1)

@fpy(precision='binary32')
def f8():
    return (1.0 - 1.0)

@fpy(precision='binary32')
def f9(arg1):
    return (arg1 - 1.0)

@fpy(precision='binary32')
def f10(arg1):
    return (1.0 - arg1)

@fpy(precision='binary32')
def f11(arg1):
    return (arg1 - arg1)

@fpy(precision='binary32')
def f12(arg1, arg2):
    return (arg1 - arg2)

@fpy(precision='binary32')
def f13(arg1, arg2):
    return (arg2 - arg1)

@fpy(precision='binary32')
def f14():
    return (1.0 * 1.0)

@fpy(precision='binary32')
def f15(arg1):
    return (arg1 * 1.0)

@fpy(precision='binary32')
def f16(arg1):
    return (1.0 * arg1)

@fpy(precision='binary32')
def f17(arg1):
    return (arg1 * arg1)

@fpy(precision='binary32')
def f18(arg1, arg2):
    return (arg1 * arg2)

@fpy(precision='binary32')
def f19(arg1, arg2):
    return (arg2 * arg1)

@fpy(precision='binary32')
def f20():
    return (1.0 / 1.0)

@fpy(precision='binary32')
def f21(arg1):
    return (arg1 / 1.0)

@fpy(precision='binary32')
def f22(arg1):
    return (1.0 / arg1)

@fpy(precision='binary32')
def f23(arg1):
    return (arg1 / arg1)

@fpy(precision='binary32')
def f24(arg1, arg2):
    return (arg1 / arg2)

@fpy(precision='binary32')
def f25(arg1, arg2):
    return (arg2 / arg1)

@fpy(precision='binary32')
def f26():
    return fabs(1.0)

@fpy(precision='binary32')
def f27(arg1):
    return fabs(arg1)

@fpy(precision='binary32')
def f28():
    return fma(1.0, 1.0, 1.0)

@fpy(precision='binary32')
def f29(arg1):
    return fma(arg1, 1.0, 1.0)

@fpy(precision='binary32')
def f30(arg1):
    return fma(1.0, arg1, 1.0)

@fpy(precision='binary32')
def f31(arg1):
    return fma(1.0, 1.0, arg1)

@fpy(precision='binary32')
def f32(arg1):
    return fma(arg1, arg1, 1.0)

@fpy(precision='binary32')
def f33(arg1):
    return fma(arg1, 1.0, arg1)

@fpy(precision='binary32')
def f34(arg1):
    return fma(1.0, arg1, arg1)

@fpy(precision='binary32')
def f35(arg1, arg2):
    return fma(arg1, arg2, 1.0)

@fpy(precision='binary32')
def f36(arg1, arg2):
    return fma(arg2, arg1, 1.0)

@fpy(precision='binary32')
def f37(arg1, arg2):
    return fma(arg1, 1.0, arg2)

@fpy(precision='binary32')
def f38(arg1, arg2):
    return fma(1.0, arg1, arg2)

@fpy(precision='binary32')
def f39(arg1, arg2):
    return fma(arg2, 1.0, arg1)

@fpy(precision='binary32')
def f40(arg1, arg2):
    return fma(1.0, arg2, arg1)

@fpy(precision='binary32')
def f41(arg1):
    return fma(arg1, arg1, arg1)

@fpy(precision='binary32')
def f42(arg1, arg2):
    return fma(arg1, arg1, arg2)

@fpy(precision='binary32')
def f43(arg1, arg2):
    return fma(arg1, arg2, arg1)

@fpy(precision='binary32')
def f44(arg1, arg2):
    return fma(arg2, arg1, arg1)

@fpy(precision='binary32')
def f45(arg1, arg2):
    return fma(arg1, arg2, arg2)

@fpy(precision='binary32')
def f46(arg1, arg2):
    return fma(arg2, arg1, arg2)

@fpy(precision='binary32')
def f47(arg1, arg2):
    return fma(arg2, arg2, arg1)

@fpy(precision='binary32')
def f48(arg1, arg2, arg3):
    return fma(arg1, arg2, arg3)

@fpy(precision='binary32')
def f49(arg1, arg2, arg3):
    return fma(arg2, arg1, arg3)

@fpy(precision='binary32')
def f50(arg1, arg2, arg3):
    return fma(arg1, arg3, arg2)

@fpy(precision='binary32')
def f51(arg1, arg2, arg3):
    return fma(arg3, arg1, arg2)

@fpy(precision='binary32')
def f52(arg1, arg2, arg3):
    return fma(arg2, arg3, arg1)

@fpy(precision='binary32')
def f53(arg1, arg2, arg3):
    return fma(arg3, arg2, arg1)

@fpy(precision='binary32')
def f54():
    return exp(1.0)

@fpy(precision='binary32')
def f55(arg1):
    return exp(arg1)

@fpy(precision='binary32')
def f56():
    return exp2(1.0)

@fpy(precision='binary32')
def f57(arg1):
    return exp2(arg1)

@fpy(precision='binary32')
def f58():
    return expm1(1.0)

@fpy(precision='binary32')
def f59(arg1):
    return expm1(arg1)

@fpy(precision='binary32')
def f60():
    return log(1.0)

@fpy(precision='binary32')
def f61(arg1):
    return log(arg1)

@fpy(precision='binary32')
def f62():
    return log10(1.0)

@fpy(precision='binary32')
def f63(arg1):
    return log10(arg1)

@fpy(precision='binary32')
def f64():
    return log2(1.0)

@fpy(precision='binary32')
def f65(arg1):
    return log2(arg1)

@fpy(precision='binary32')
def f66():
    return log1p(1.0)

@fpy(precision='binary32')
def f67(arg1):
    return log1p(arg1)

@fpy(precision='binary32')
def f68():
    return pow(1.0, 1.0)

@fpy(precision='binary32')
def f69(arg1):
    return pow(arg1, 1.0)

@fpy(precision='binary32')
def f70(arg1):
    return pow(1.0, arg1)

@fpy(precision='binary32')
def f71(arg1):
    return pow(arg1, arg1)

@fpy(precision='binary32')
def f72(arg1, arg2):
    return pow(arg1, arg2)

@fpy(precision='binary32')
def f73(arg1, arg2):
    return pow(arg2, arg1)

@fpy(precision='binary32')
def f74():
    return sqrt(1.0)

@fpy(precision='binary32')
def f75(arg1):
    return sqrt(arg1)

@fpy(precision='binary32')
def f76():
    return cbrt(1.0)

@fpy(precision='binary32')
def f77(arg1):
    return cbrt(arg1)

@fpy(precision='binary32')
def f78():
    return hypot(1.0, 1.0)

@fpy(precision='binary32')
def f79(arg1):
    return hypot(arg1, 1.0)

@fpy(precision='binary32')
def f80(arg1):
    return hypot(1.0, arg1)

@fpy(precision='binary32')
def f81(arg1):
    return hypot(arg1, arg1)

@fpy(precision='binary32')
def f82(arg1, arg2):
    return hypot(arg1, arg2)

@fpy(precision='binary32')
def f83(arg1, arg2):
    return hypot(arg2, arg1)

@fpy(precision='binary32')
def f84():
    return sin(1.0)

@fpy(precision='binary32')
def f85(arg1):
    return sin(arg1)

@fpy(precision='binary32')
def f86():
    return cos(1.0)

@fpy(precision='binary32')
def f87(arg1):
    return cos(arg1)

@fpy(precision='binary32')
def f88():
    return tan(1.0)

@fpy(precision='binary32')
def f89(arg1):
    return tan(arg1)

@fpy(precision='binary32')
def f90():
    return asin(1.0)

@fpy(precision='binary32')
def f91(arg1):
    return asin(arg1)

@fpy(precision='binary32')
def f92():
    return acos(1.0)

@fpy(precision='binary32')
def f93(arg1):
    return acos(arg1)

@fpy(precision='binary32')
def f94():
    return atan(1.0)

@fpy(precision='binary32')
def f95(arg1):
    return atan(arg1)

@fpy(precision='binary32')
def f96():
    return atan2(1.0, 1.0)

@fpy(precision='binary32')
def f97(arg1):
    return atan2(arg1, 1.0)

@fpy(precision='binary32')
def f98(arg1):
    return atan2(1.0, arg1)

@fpy(precision='binary32')
def f99(arg1):
    return atan2(arg1, arg1)

@fpy(precision='binary32')
def f100(arg1, arg2):
    return atan2(arg1, arg2)

@fpy(precision='binary32')
def f101(arg1, arg2):
    return atan2(arg2, arg1)

@fpy(precision='binary32')
def f102():
    return sinh(1.0)

@fpy(precision='binary32')
def f103(arg1):
    return sinh(arg1)

@fpy(precision='binary32')
def f104():
    return cosh(1.0)

@fpy(precision='binary32')
def f105(arg1):
    return cosh(arg1)

@fpy(precision='binary32')
def f106():
    return tanh(1.0)

@fpy(precision='binary32')
def f107(arg1):
    return tanh(arg1)

@fpy(precision='binary32')
def f108():
    return asinh(1.0)

@fpy(precision='binary32')
def f109(arg1):
    return asinh(arg1)

@fpy(precision='binary32')
def f110():
    return acosh(1.0)

@fpy(precision='binary32')
def f111(arg1):
    return acosh(arg1)

@fpy(precision='binary32')
def f112():
    return atanh(1.0)

@fpy(precision='binary32')
def f113(arg1):
    return atanh(arg1)

@fpy(precision='binary32')
def f114():
    return erf(1.0)

@fpy(precision='binary32')
def f115(arg1):
    return erf(arg1)

@fpy(precision='binary32')
def f116():
    return erfc(1.0)

@fpy(precision='binary32')
def f117(arg1):
    return erfc(arg1)

@fpy(precision='binary32')
def f118():
    return tgamma(1.0)

@fpy(precision='binary32')
def f119(arg1):
    return tgamma(arg1)

@fpy(precision='binary32')
def f120():
    return lgamma(1.0)

@fpy(precision='binary32')
def f121(arg1):
    return lgamma(arg1)

@fpy(precision='binary32')
def f122():
    return ceil(1.0)

@fpy(precision='binary32')
def f123(arg1):
    return ceil(arg1)

@fpy(precision='binary32')
def f124():
    return floor(1.0)

@fpy(precision='binary32')
def f125(arg1):
    return floor(arg1)

@fpy(precision='binary32')
def f126():
    return fmod(1.0, 1.0)

@fpy(precision='binary32')
def f127(arg1):
    return fmod(arg1, 1.0)

@fpy(precision='binary32')
def f128(arg1):
    return fmod(1.0, arg1)

@fpy(precision='binary32')
def f129(arg1):
    return fmod(arg1, arg1)

@fpy(precision='binary32')
def f130(arg1, arg2):
    return fmod(arg1, arg2)

@fpy(precision='binary32')
def f131(arg1, arg2):
    return fmod(arg2, arg1)

@fpy(precision='binary32')
def f132():
    return remainder(1.0, 1.0)

@fpy(precision='binary32')
def f133(arg1):
    return remainder(arg1, 1.0)

@fpy(precision='binary32')
def f134(arg1):
    return remainder(1.0, arg1)

@fpy(precision='binary32')
def f135(arg1):
    return remainder(arg1, arg1)

@fpy(precision='binary32')
def f136(arg1, arg2):
    return remainder(arg1, arg2)

@fpy(precision='binary32')
def f137(arg1, arg2):
    return remainder(arg2, arg1)

@fpy(precision='binary32')
def f138():
    return fmax(1.0, 1.0)

@fpy(precision='binary32')
def f139(arg1):
    return fmax(arg1, 1.0)

@fpy(precision='binary32')
def f140(arg1):
    return fmax(1.0, arg1)

@fpy(precision='binary32')
def f141(arg1):
    return fmax(arg1, arg1)

@fpy(precision='binary32')
def f142(arg1, arg2):
    return fmax(arg1, arg2)

@fpy(precision='binary32')
def f143(arg1, arg2):
    return fmax(arg2, arg1)

@fpy(precision='binary32')
def f144():
    return fmin(1.0, 1.0)

@fpy(precision='binary32')
def f145(arg1):
    return fmin(arg1, 1.0)

@fpy(precision='binary32')
def f146(arg1):
    return fmin(1.0, arg1)

@fpy(precision='binary32')
def f147(arg1):
    return fmin(arg1, arg1)

@fpy(precision='binary32')
def f148(arg1, arg2):
    return fmin(arg1, arg2)

@fpy(precision='binary32')
def f149(arg1, arg2):
    return fmin(arg2, arg1)

@fpy(precision='binary32')
def f150():
    return fdim(1.0, 1.0)

@fpy(precision='binary32')
def f151(arg1):
    return fdim(arg1, 1.0)

@fpy(precision='binary32')
def f152(arg1):
    return fdim(1.0, arg1)

@fpy(precision='binary32')
def f153(arg1):
    return fdim(arg1, arg1)

@fpy(precision='binary32')
def f154(arg1, arg2):
    return fdim(arg1, arg2)

@fpy(precision='binary32')
def f155(arg1, arg2):
    return fdim(arg2, arg1)

@fpy(precision='binary32')
def f156():
    return copysign(1.0, 1.0)

@fpy(precision='binary32')
def f157(arg1):
    return copysign(arg1, 1.0)

@fpy(precision='binary32')
def f158(arg1):
    return copysign(1.0, arg1)

@fpy(precision='binary32')
def f159(arg1):
    return copysign(arg1, arg1)

@fpy(precision='binary32')
def f160(arg1, arg2):
    return copysign(arg1, arg2)

@fpy(precision='binary32')
def f161(arg1, arg2):
    return copysign(arg2, arg1)

@fpy(precision='binary32')
def f162():
    return trunc(1.0)

@fpy(precision='binary32')
def f163(arg1):
    return trunc(arg1)

@fpy(precision='binary32')
def f164():
    return round(1.0)

@fpy(precision='binary32')
def f165(arg1):
    return round(arg1)

@fpy(precision='binary32')
def f166():
    return nearbyint(1.0)

@fpy(precision='binary32')
def f167(arg1):
    return nearbyint(arg1)

@fpy(precision='binary32')
def f168():
    return cast(1.0)

@fpy(precision='binary32')
def f169(arg1):
    return cast(arg1)

@fpy(precision='binary80')
def f170():
    return (1.0 + 1.0)

@fpy(precision='binary80')
def f171(arg1):
    return (arg1 + 1.0)

@fpy(precision='binary80')
def f172(arg1):
    return (1.0 + arg1)

@fpy(precision='binary80')
def f173(arg1):
    return (arg1 + arg1)

@fpy(precision='binary80')
def f174(arg1, arg2):
    return (arg1 + arg2)

@fpy(precision='binary80')
def f175(arg1, arg2):
    return (arg2 + arg1)

@fpy(precision='binary80')
def f176():
    return (1.0 - 1.0)

@fpy(precision='binary80')
def f177(arg1):
    return (arg1 - 1.0)

@fpy(precision='binary80')
def f178(arg1):
    return (1.0 - arg1)

@fpy(precision='binary80')
def f179(arg1):
    return (arg1 - arg1)

@fpy(precision='binary80')
def f180(arg1, arg2):
    return (arg1 - arg2)

@fpy(precision='binary80')
def f181(arg1, arg2):
    return (arg2 - arg1)

@fpy(precision='binary80')
def f182():
    return (1.0 * 1.0)

@fpy(precision='binary80')
def f183(arg1):
    return (arg1 * 1.0)

@fpy(precision='binary80')
def f184(arg1):
    return (1.0 * arg1)

@fpy(precision='binary80')
def f185(arg1):
    return (arg1 * arg1)

@fpy(precision='binary80')
def f186(arg1, arg2):
    return (arg1 * arg2)

@fpy(precision='binary80')
def f187(arg1, arg2):
    return (arg2 * arg1)

@fpy(precision='binary80')
def f188():
    return (1.0 / 1.0)

@fpy(precision='binary80')
def f189(arg1):
    return (arg1 / 1.0)

@fpy(precision='binary80')
def f190(arg1):
    return (1.0 / arg1)

@fpy(precision='binary80')
def f191(arg1):
    return (arg1 / arg1)

@fpy(precision='binary80')
def f192(arg1, arg2):
    return (arg1 / arg2)

@fpy(precision='binary80')
def f193(arg1, arg2):
    return (arg2 / arg1)

@fpy(precision='binary80')
def f194():
    return fabs(1.0)

@fpy(precision='binary80')
def f195(arg1):
    return fabs(arg1)

@fpy(precision='binary80')
def f196():
    return fma(1.0, 1.0, 1.0)

@fpy(precision='binary80')
def f197(arg1):
    return fma(arg1, 1.0, 1.0)

@fpy(precision='binary80')
def f198(arg1):
    return fma(1.0, arg1, 1.0)

@fpy(precision='binary80')
def f199(arg1):
    return fma(1.0, 1.0, arg1)

@fpy(precision='binary80')
def f200(arg1):
    return fma(arg1, arg1, 1.0)

@fpy(precision='binary80')
def f201(arg1):
    return fma(arg1, 1.0, arg1)

@fpy(precision='binary80')
def f202(arg1):
    return fma(1.0, arg1, arg1)

@fpy(precision='binary80')
def f203(arg1, arg2):
    return fma(arg1, arg2, 1.0)

@fpy(precision='binary80')
def f204(arg1, arg2):
    return fma(arg2, arg1, 1.0)

@fpy(precision='binary80')
def f205(arg1, arg2):
    return fma(arg1, 1.0, arg2)

@fpy(precision='binary80')
def f206(arg1, arg2):
    return fma(1.0, arg1, arg2)

@fpy(precision='binary80')
def f207(arg1, arg2):
    return fma(arg2, 1.0, arg1)

@fpy(precision='binary80')
def f208(arg1, arg2):
    return fma(1.0, arg2, arg1)

@fpy(precision='binary80')
def f209(arg1):
    return fma(arg1, arg1, arg1)

@fpy(precision='binary80')
def f210(arg1, arg2):
    return fma(arg1, arg1, arg2)

@fpy(precision='binary80')
def f211(arg1, arg2):
    return fma(arg1, arg2, arg1)

@fpy(precision='binary80')
def f212(arg1, arg2):
    return fma(arg2, arg1, arg1)

@fpy(precision='binary80')
def f213(arg1, arg2):
    return fma(arg1, arg2, arg2)

@fpy(precision='binary80')
def f214(arg1, arg2):
    return fma(arg2, arg1, arg2)

@fpy(precision='binary80')
def f215(arg1, arg2):
    return fma(arg2, arg2, arg1)

@fpy(precision='binary80')
def f216(arg1, arg2, arg3):
    return fma(arg1, arg2, arg3)

@fpy(precision='binary80')
def f217(arg1, arg2, arg3):
    return fma(arg2, arg1, arg3)

@fpy(precision='binary80')
def f218(arg1, arg2, arg3):
    return fma(arg1, arg3, arg2)

@fpy(precision='binary80')
def f219(arg1, arg2, arg3):
    return fma(arg3, arg1, arg2)

@fpy(precision='binary80')
def f220(arg1, arg2, arg3):
    return fma(arg2, arg3, arg1)

@fpy(precision='binary80')
def f221(arg1, arg2, arg3):
    return fma(arg3, arg2, arg1)

@fpy(precision='binary80')
def f222():
    return exp(1.0)

@fpy(precision='binary80')
def f223(arg1):
    return exp(arg1)

@fpy(precision='binary80')
def f224():
    return exp2(1.0)

@fpy(precision='binary80')
def f225(arg1):
    return exp2(arg1)

@fpy(precision='binary80')
def f226():
    return expm1(1.0)

@fpy(precision='binary80')
def f227(arg1):
    return expm1(arg1)

@fpy(precision='binary80')
def f228():
    return log(1.0)

@fpy(precision='binary80')
def f229(arg1):
    return log(arg1)

@fpy(precision='binary80')
def f230():
    return log10(1.0)

@fpy(precision='binary80')
def f231(arg1):
    return log10(arg1)

@fpy(precision='binary80')
def f232():
    return log2(1.0)

@fpy(precision='binary80')
def f233(arg1):
    return log2(arg1)

@fpy(precision='binary80')
def f234():
    return log1p(1.0)

@fpy(precision='binary80')
def f235(arg1):
    return log1p(arg1)

@fpy(precision='binary80')
def f236():
    return pow(1.0, 1.0)

@fpy(precision='binary80')
def f237(arg1):
    return pow(arg1, 1.0)

@fpy(precision='binary80')
def f238(arg1):
    return pow(1.0, arg1)

@fpy(precision='binary80')
def f239(arg1):
    return pow(arg1, arg1)

@fpy(precision='binary80')
def f240(arg1, arg2):
    return pow(arg1, arg2)

@fpy(precision='binary80')
def f241(arg1, arg2):
    return pow(arg2, arg1)

@fpy(precision='binary80')
def f242():
    return sqrt(1.0)

@fpy(precision='binary80')
def f243(arg1):
    return sqrt(arg1)

@fpy(precision='binary80')
def f244():
    return cbrt(1.0)

@fpy(precision='binary80')
def f245(arg1):
    return cbrt(arg1)

@fpy(precision='binary80')
def f246():
    return hypot(1.0, 1.0)

@fpy(precision='binary80')
def f247(arg1):
    return hypot(arg1, 1.0)

@fpy(precision='binary80')
def f248(arg1):
    return hypot(1.0, arg1)

@fpy(precision='binary80')
def f249(arg1):
    return hypot(arg1, arg1)

@fpy(precision='binary80')
def f250(arg1, arg2):
    return hypot(arg1, arg2)

@fpy(precision='binary80')
def f251(arg1, arg2):
    return hypot(arg2, arg1)

@fpy(precision='binary80')
def f252():
    return sin(1.0)

@fpy(precision='binary80')
def f253(arg1):
    return sin(arg1)

@fpy(precision='binary80')
def f254():
    return cos(1.0)

@fpy(precision='binary80')
def f255(arg1):
    return cos(arg1)

@fpy(precision='binary80')
def f256():
    return tan(1.0)

@fpy(precision='binary80')
def f257(arg1):
    return tan(arg1)

@fpy(precision='binary80')
def f258():
    return asin(1.0)

@fpy(precision='binary80')
def f259(arg1):
    return asin(arg1)

@fpy(precision='binary80')
def f260():
    return acos(1.0)

@fpy(precision='binary80')
def f261(arg1):
    return acos(arg1)

@fpy(precision='binary80')
def f262():
    return atan(1.0)

@fpy(precision='binary80')
def f263(arg1):
    return atan(arg1)

@fpy(precision='binary80')
def f264():
    return atan2(1.0, 1.0)

@fpy(precision='binary80')
def f265(arg1):
    return atan2(arg1, 1.0)

@fpy(precision='binary80')
def f266(arg1):
    return atan2(1.0, arg1)

@fpy(precision='binary80')
def f267(arg1):
    return atan2(arg1, arg1)

@fpy(precision='binary80')
def f268(arg1, arg2):
    return atan2(arg1, arg2)

@fpy(precision='binary80')
def f269(arg1, arg2):
    return atan2(arg2, arg1)

@fpy(precision='binary80')
def f270():
    return sinh(1.0)

@fpy(precision='binary80')
def f271(arg1):
    return sinh(arg1)

@fpy(precision='binary80')
def f272():
    return cosh(1.0)

@fpy(precision='binary80')
def f273(arg1):
    return cosh(arg1)

@fpy(precision='binary80')
def f274():
    return tanh(1.0)

@fpy(precision='binary80')
def f275(arg1):
    return tanh(arg1)

@fpy(precision='binary80')
def f276():
    return asinh(1.0)

@fpy(precision='binary80')
def f277(arg1):
    return asinh(arg1)

@fpy(precision='binary80')
def f278():
    return acosh(1.0)

@fpy(precision='binary80')
def f279(arg1):
    return acosh(arg1)

@fpy(precision='binary80')
def f280():
    return atanh(1.0)

@fpy(precision='binary80')
def f281(arg1):
    return atanh(arg1)

@fpy(precision='binary80')
def f282():
    return erf(1.0)

@fpy(precision='binary80')
def f283(arg1):
    return erf(arg1)

@fpy(precision='binary80')
def f284():
    return erfc(1.0)

@fpy(precision='binary80')
def f285(arg1):
    return erfc(arg1)

@fpy(precision='binary80')
def f286():
    return tgamma(1.0)

@fpy(precision='binary80')
def f287(arg1):
    return tgamma(arg1)

@fpy(precision='binary80')
def f288():
    return lgamma(1.0)

@fpy(precision='binary80')
def f289(arg1):
    return lgamma(arg1)

@fpy(precision='binary80')
def f290():
    return ceil(1.0)

@fpy(precision='binary80')
def f291(arg1):
    return ceil(arg1)

@fpy(precision='binary80')
def f292():
    return floor(1.0)

@fpy(precision='binary80')
def f293(arg1):
    return floor(arg1)

@fpy(precision='binary80')
def f294():
    return fmod(1.0, 1.0)

@fpy(precision='binary80')
def f295(arg1):
    return fmod(arg1, 1.0)

@fpy(precision='binary80')
def f296(arg1):
    return fmod(1.0, arg1)

@fpy(precision='binary80')
def f297(arg1):
    return fmod(arg1, arg1)

@fpy(precision='binary80')
def f298(arg1, arg2):
    return fmod(arg1, arg2)

@fpy(precision='binary80')
def f299(arg1, arg2):
    return fmod(arg2, arg1)

@fpy(precision='binary80')
def f300():
    return remainder(1.0, 1.0)

@fpy(precision='binary80')
def f301(arg1):
    return remainder(arg1, 1.0)

@fpy(precision='binary80')
def f302(arg1):
    return remainder(1.0, arg1)

@fpy(precision='binary80')
def f303(arg1):
    return remainder(arg1, arg1)

@fpy(precision='binary80')
def f304(arg1, arg2):
    return remainder(arg1, arg2)

@fpy(precision='binary80')
def f305(arg1, arg2):
    return remainder(arg2, arg1)

@fpy(precision='binary80')
def f306():
    return fmax(1.0, 1.0)

@fpy(precision='binary80')
def f307(arg1):
    return fmax(arg1, 1.0)

@fpy(precision='binary80')
def f308(arg1):
    return fmax(1.0, arg1)

@fpy(precision='binary80')
def f309(arg1):
    return fmax(arg1, arg1)

@fpy(precision='binary80')
def f310(arg1, arg2):
    return fmax(arg1, arg2)

@fpy(precision='binary80')
def f311(arg1, arg2):
    return fmax(arg2, arg1)

@fpy(precision='binary80')
def f312():
    return fmin(1.0, 1.0)

@fpy(precision='binary80')
def f313(arg1):
    return fmin(arg1, 1.0)

@fpy(precision='binary80')
def f314(arg1):
    return fmin(1.0, arg1)

@fpy(precision='binary80')
def f315(arg1):
    return fmin(arg1, arg1)

@fpy(precision='binary80')
def f316(arg1, arg2):
    return fmin(arg1, arg2)

@fpy(precision='binary80')
def f317(arg1, arg2):
    return fmin(arg2, arg1)

@fpy(precision='binary80')
def f318():
    return fdim(1.0, 1.0)

@fpy(precision='binary80')
def f319(arg1):
    return fdim(arg1, 1.0)

@fpy(precision='binary80')
def f320(arg1):
    return fdim(1.0, arg1)

@fpy(precision='binary80')
def f321(arg1):
    return fdim(arg1, arg1)

@fpy(precision='binary80')
def f322(arg1, arg2):
    return fdim(arg1, arg2)

@fpy(precision='binary80')
def f323(arg1, arg2):
    return fdim(arg2, arg1)

@fpy(precision='binary80')
def f324():
    return copysign(1.0, 1.0)

@fpy(precision='binary80')
def f325(arg1):
    return copysign(arg1, 1.0)

@fpy(precision='binary80')
def f326(arg1):
    return copysign(1.0, arg1)

@fpy(precision='binary80')
def f327(arg1):
    return copysign(arg1, arg1)

@fpy(precision='binary80')
def f328(arg1, arg2):
    return copysign(arg1, arg2)

@fpy(precision='binary80')
def f329(arg1, arg2):
    return copysign(arg2, arg1)

@fpy(precision='binary80')
def f330():
    return trunc(1.0)

@fpy(precision='binary80')
def f331(arg1):
    return trunc(arg1)

@fpy(precision='binary80')
def f332():
    return round(1.0)

@fpy(precision='binary80')
def f333(arg1):
    return round(arg1)

@fpy(precision='binary80')
def f334():
    return nearbyint(1.0)

@fpy(precision='binary80')
def f335(arg1):
    return nearbyint(arg1)

@fpy(precision='binary80')
def f336():
    return cast(1.0)

@fpy(precision='binary80')
def f337(arg1):
    return cast(arg1)

@fpy(precision='binary80')
def f338():
    return -1.0

@fpy(precision='binary80')
def f339(arg1):
    return -arg1

@fpy(precision='binary32')
def f340():
    if 1.0 < 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f341(arg1):
    if arg1 < 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f342(arg1):
    if 1.0 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f343(arg1):
    if arg1 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f344(arg1, arg2):
    if arg1 < arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f345(arg1, arg2):
    if arg2 < arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f346():
    if 1.0 > 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f347(arg1):
    if arg1 > 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f348(arg1):
    if 1.0 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f349(arg1):
    if arg1 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f350(arg1, arg2):
    if arg1 > arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f351(arg1, arg2):
    if arg2 > arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f352():
    if 1.0 <= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f353(arg1):
    if arg1 <= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f354(arg1):
    if 1.0 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f355(arg1):
    if arg1 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f356(arg1, arg2):
    if arg1 <= arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f357(arg1, arg2):
    if arg2 <= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f358():
    if 1.0 >= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f359(arg1):
    if arg1 >= 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f360(arg1):
    if 1.0 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f361(arg1):
    if arg1 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f362(arg1, arg2):
    if arg1 >= arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f363(arg1, arg2):
    if arg2 >= arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f364():
    if 1.0 == 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f365(arg1):
    if arg1 == 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f366(arg1):
    if 1.0 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f367(arg1):
    if arg1 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f368(arg1, arg2):
    if arg1 == arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f369(arg1, arg2):
    if arg2 == arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f370():
    if 1.0 != 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f371(arg1):
    if arg1 != 1.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f372(arg1):
    if 1.0 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f373(arg1):
    if arg1 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f374(arg1, arg2):
    if arg1 != arg2:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f375(arg1, arg2):
    if arg2 != arg1:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f376():
    if isfinite(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f377(arg1):
    if isfinite(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f378():
    if isinf(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f379(arg1):
    if isinf(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f380():
    if isnan(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f381(arg1):
    if isnan(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f382():
    if isnormal(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f383(arg1):
    if isnormal(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f384():
    if signbit(1.0):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary32')
def f385(arg1):
    if signbit(arg1):
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(precision='binary80')
def f386():
    if 1.0 < 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f387(arg1):
    if arg1 < 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f388(arg1):
    if 1.0 < arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f389(arg1):
    if arg1 < arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f390(arg1, arg2):
    if arg1 < arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f391(arg1, arg2):
    if arg2 < arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f392():
    if 1.0 > 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f393(arg1):
    if arg1 > 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f394(arg1):
    if 1.0 > arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f395(arg1):
    if arg1 > arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f396(arg1, arg2):
    if arg1 > arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f397(arg1, arg2):
    if arg2 > arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f398():
    if 1.0 <= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f399(arg1):
    if arg1 <= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f400(arg1):
    if 1.0 <= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f401(arg1):
    if arg1 <= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f402(arg1, arg2):
    if arg1 <= arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f403(arg1, arg2):
    if arg2 <= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f404():
    if 1.0 >= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f405(arg1):
    if arg1 >= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f406(arg1):
    if 1.0 >= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f407(arg1):
    if arg1 >= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f408(arg1, arg2):
    if arg1 >= arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f409(arg1, arg2):
    if arg2 >= arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f410():
    if 1.0 == 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f411(arg1):
    if arg1 == 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f412(arg1):
    if 1.0 == arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f413(arg1):
    if arg1 == arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f414(arg1, arg2):
    if arg1 == arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f415(arg1, arg2):
    if arg2 == arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f416():
    if 1.0 != 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f417(arg1):
    if arg1 != 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f418(arg1):
    if 1.0 != arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f419(arg1):
    if arg1 != arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f420(arg1, arg2):
    if arg1 != arg2:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f421(arg1, arg2):
    if arg2 != arg1:
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f422():
    if isfinite(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f423(arg1):
    if isfinite(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f424():
    if isinf(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f425(arg1):
    if isinf(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f426():
    if isnan(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f427(arg1):
    if isnan(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f428():
    if isnormal(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f429(arg1):
    if isnormal(arg1):
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
def f430():
    if signbit(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(precision='binary80')
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

