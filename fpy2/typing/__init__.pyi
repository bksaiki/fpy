"""Typing hints for FPy programs"""

from typing import TypeAlias

Real: TypeAlias = int | float

def __add__(self: Real, other: Real) -> Real: ...
def __sub__(self: Real, other: Real) -> Real: ...
def __mul__(self: Real, other: Real) -> Real: ...
def __truediv__(self: Real, other: Real) -> Real: ...

def __radd__(self: Real, other: Real) -> Real: ...
def __rsub__(self: Real, other: Real) -> Real: ...
def __rmul__(self: Real, other: Real) -> Real: ...
def __rtruediv__(self: Real, other: Real) -> Real: ...

def __pow__(self: Real, other: int) -> Real: ...

def __eq__(self: Real, other) -> bool: ...
def __ne__(self: Real, other) -> bool: ...
def __lt__(self: Real, other: Real) -> bool: ...
def __le__(self: Real, other: Real) -> bool: ...
def __gt__(self: Real, other: Real) -> bool: ...
def __ge__(self: Real, other: Real) -> bool: ...

def fabs(x: Real) -> Real: ...
def sqrt(x: Real) -> Real: ...
def cbrt(x: Real) -> Real: ...
def ceil(x: Real) -> Real: ...
def floor(x: Real) -> Real: ...
def nearbyint(x: Real) -> Real: ...
def round(x: Real) -> Real: ...
def trunc(x: Real) -> Real: ...
def acos(x: Real) -> Real: ...
def asin(x: Real) -> Real: ...
def atan(x: Real) -> Real: ...
def cos(x: Real) -> Real: ...
def sin(x: Real) -> Real: ...
def tan(x: Real) -> Real: ...
def acosh(x: Real) -> Real: ...
def asinh(x: Real) -> Real: ...
def atanh(x: Real) -> Real: ...
def exp(x: Real) -> Real: ...
def exp2(x: Real) -> Real: ...
def expm1(x: Real) -> Real: ...
def log(x: Real) -> Real: ...
def log10(x: Real) -> Real: ...
def log1p(x: Real) -> Real: ...
def log2(x: Real) -> Real: ...
def erf(x: Real) -> Real: ...
def erfc(x: Real) -> Real: ...
def lgamma(x: Real) -> Real: ...
def tgamma(x: Real) -> Real: ...
def isfinite(x: Real) -> bool: ...
def isinf(x: Real) -> bool: ...
def isnan(x: Real) -> bool: ...
def isnormal(x: Real) -> bool: ...
def signbit(x: Real) -> bool: ...

def copysign(x: Real, y: Real) -> Real: ...
def fdim(x: Real, y: Real) -> Real: ...
def fmax(x: Real, y: Real) -> Real: ...
def fmin(x: Real, y: Real) -> Real: ...
def fmod(x: Real, y: Real) -> Real: ...
def remainder(x: Real, y: Real) -> Real: ...
def hypot(x: Real, y: Real) -> Real: ...
def atan2(x: Real, y: Real) -> Real: ...
def pow(x: Real, y: Real) -> Real: ...

def fma(x: Real, y: Real, z: Real) -> Real: ...

def digits(m: int, e: int, b: int) -> Real: ...

class Context():
    def __init__(self, **kwargs): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    ...
