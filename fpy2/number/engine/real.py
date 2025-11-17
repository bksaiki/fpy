"""
Real arithmetic engine for exact computation.

This engine handles exact arithmetic on Float and Fraction types,
computing results without any rounding. It only works when the
context requests exact computation (prec=None, n=None).
"""

from fractions import Fraction

from ..context import Context, REAL
from ..gmp import mpfr_value
from ..number import Float
from ..round import RoundingMode

from .engine import Engine, EngineArg, EngineRes


def _is_nan(x: EngineArg) -> bool:
    return isinstance(x, Float) and x.isnan

def _is_inf(x: EngineArg) -> bool:
    return isinstance(x, Float) and x.isinf

def _is_zero(x: EngineArg) -> bool:
    return x.is_zero() if isinstance(x, Float) else x == 0

def _signbit(x: EngineArg) -> bool:
    return x.s if isinstance(x, Float) else (x < 0)


_real_engine_inst = None
"""single instance of Real engine"""


class RealEngine(Engine):
    """
    Engine that performs exact real arithmetic.

    This engine only handles operations when the context requests
    exact computation (both prec and n are None). It works with
    both Float and Fraction types.
    """

    @staticmethod
    def instance() -> 'RealEngine':
        """Returns the singleton instance of the Real engine."""
        global _real_engine_inst
        if _real_engine_inst is None:
            _real_engine_inst = RealEngine()
        return _real_engine_inst

    def _real_rint(self, x: Float | Fraction, rm: RoundingMode) -> Float:
        """
        Round a real number to the nearest integer under a specific rounding mode.
        """
        match x:
            case Float():
                if x.is_nar():
                    # special value
                    return Float(x=x, ctx=REAL)
                else:
                    # finite value
                    r = x.as_real().round(None, -1, rm)
            case Fraction():
                y = mpfr_value(x, n=-1)
                r = y.as_real().round(None, -1, rm)
            case _:
                raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')

        return Float(x=r, ctx=REAL)

    # Unary operations

    def acos(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def acosh(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def asin(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def asinh(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def atan(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def atanh(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def cbrt(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def ceil(self, x: EngineArg, ctx: Context) -> EngineRes:
        return self._real_rint(x, RoundingMode.RTP)

    def cos(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def cosh(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def erf(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def erfc(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def exp(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def exp2(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def exp10(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def expm1(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def fabs(self, x: EngineArg, ctx: Context) -> EngineRes:
        match x:
            case Float():
                return Float(s=False, x=x, ctx=REAL)
            case Fraction():
                return abs(x)
            case _:
                raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(x)}\' for x={x}')

    def floor(self, x: EngineArg, ctx: Context) -> EngineRes:
        return self._real_rint(x, RoundingMode.RTP)

    def lgamma(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def log(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def log10(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def log1p(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def log2(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def neg(self, x: EngineArg, ctx: Context) -> EngineRes:
        match x:
            case Float():
                return Float(s=not x.s, x=x, ctx=REAL)
            case Fraction():
                return -x
            case _:
                raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(x)}\' for x={x}')

    def roundint(self, x: EngineArg, ctx: Context) -> EngineRes:
        return self._real_rint(x, RoundingMode.RNA)

    def sin(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def sinh(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def sqrt(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def tan(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def tanh(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def tgamma(self, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def trunc(self, x: EngineArg, ctx: Context) -> EngineRes:
        return self._real_rint(x, RoundingMode.RTZ)

    # Binary operations

    def add(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if _is_nan(x) or _is_nan(y):
            # either is NaN
            return Float(isnan=True, ctx=REAL)
        elif _is_inf(x):
            # x is Inf
            if _is_inf(y):
                # y is also Inf
                if _signbit(x) == _signbit(y):
                    # Inf + Inf = Inf
                    return Float(s=_signbit(x), isinf=True, ctx=REAL)
                else:
                    # Inf + -Inf = NaN
                    return Float(isnan=True, ctx=REAL)
            else:
                # y is finite: Inf + y = Inf
                return Float(s=_signbit(x), isinf=True, ctx=REAL)
        elif _is_inf(y):
            # y is Inf, x is finite: x + Inf = Inf
            return Float(s=_signbit(y), isinf=True, ctx=REAL)
        else:
            # both are finite
            match x, y:
                case Float(), Float():
                    r = x.as_real() + y.as_real()
                    return Float(x=r, ctx=REAL)
                case Fraction(), Fraction():
                    return x + y
                case Fraction(), Float():
                    return x + y.as_rational()
                case Float(), Fraction():
                    return x.as_rational() + y
                case _:
                    raise RuntimeError("unreachable case")

    def atan2(self, y: EngineArg, x: EngineArg, ctx: Context) -> EngineRes:
        return None

    def copysign(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return None

    def div(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return None

    def fdim(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return None

    def fmod(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return None

    def fmax(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return max(x, y)

    def fmin(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return min(x, y)

    def hypot(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return None

    def mod(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return None

    def mul(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        if _is_nan(x) or _is_nan(y):
            # either is NaN
            return Float(isnan=True, ctx=REAL)
        elif _is_inf(x):
            # x is Inf
            if _is_zero(y):
                # Inf * 0 = NaN
                return Float(isnan=True, ctx=REAL)
            else:
                # Inf * y = Inf
                s = _signbit(x) != _signbit(y)
                return Float(s=s, isinf=True, ctx=REAL)
        elif _is_inf(y):
            # y is Inf
            if _is_zero(x):
                # 0 * Inf = NaN
                return Float(isnan=True, ctx=REAL)
            else:
                # x * Inf = Inf
                s = _signbit(x) != _signbit(y)
                return Float(s=s, isinf=True, ctx=REAL)
        else:
            # both are finite
            match x, y:
                case Float(), Float():
                    r = x.as_real() * y.as_real()
                    return Float(x=r, ctx=REAL)
                case Fraction(), Fraction():
                    return x * y
                case Fraction(), Float():
                    return x * y.as_rational()
                case Float(), Fraction():
                    return x.as_rational() * y
                case _:
                    raise RuntimeError("unreachable case")

    def pow(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return None

    def remainder(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        return None

    def sub(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        # Implement as add(x, neg(y))
        neg_y = self.neg(y, ctx)
        if neg_y is None:
            return None
        return self.add(x, neg_y, ctx)

    # Ternary operations

    def fma(self, x: EngineArg, y: EngineArg, z: EngineArg, ctx: Context) -> EngineRes:
        # Implement as add(mul(x, y), z)
        mul_result = self.mul(x, y, ctx)
        if mul_result is None:
            return None
        return self.add(mul_result, z, ctx)

    # Mathematical constants

    def const_e(self, ctx: Context) -> EngineRes:
        return None

    def const_log2e(self, ctx: Context) -> EngineRes:
        return None

    def const_log10e(self, ctx: Context) -> EngineRes:
        return None

    def const_ln2(self, ctx: Context) -> EngineRes:
        return None

    def const_ln10(self, ctx: Context) -> EngineRes:
        return None

    def const_pi(self, ctx: Context) -> EngineRes:
        return None

    def const_pi_2(self, ctx: Context) -> EngineRes:
        return None

    def const_pi_4(self, ctx: Context) -> EngineRes:
        return None

    def const_1_pi(self, ctx: Context) -> EngineRes:
        return None

    def const_2_pi(self, ctx: Context) -> EngineRes:
        return None

    def const_2_sqrtpi(self, ctx: Context) -> EngineRes:
        return None

    def const_sqrt2(self, ctx: Context) -> EngineRes:
        return None

    def const_sqrt1_2(self, ctx: Context) -> EngineRes:
        return None
