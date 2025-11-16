"""
Abstract base class for round-to-odd arithmetic engines.

An Engine provides methods for computing arithmetic operations with
round-to-odd rounding mode. Each method takes Float arguments and a rounding
context, returning a Float result or None if the engine cannot handle
the computation.

The context provides precision parameters through ctx.round_params():
- (prec, None): floating-point style with prec bits of precision
- (None, n): fixed-point style with first unrepresentable digit n
- (prec, n): floating-point with subnormalization
- (None, None): real computation (exact arithmetic)

Methods should return None to signal that the engine cannot handle
the operation with the given inputs/context. This allows for
efficient dispatch to alternative engines without exception overhead.
"""

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Iterator

from ..context import Context
from ..number import Float, Real
from ...utils import is_dyadic

__all__ = [
    'Engine',
    'EngineList',
]


class Engine(ABC):
    """
    Abstract base class for round-to-odd arithmetic engines.

    Engines provide computational methods that implement various arithmetic
    operations with round-to-odd rounding mode. Each method returns a Float
    result computed with sufficient precision for safe re-rounding, or None
    if the engine cannot handle the operation.

    All computational methods follow the same signature pattern:
        method(self, *args: Float | Fraction, ctx: Context) -> Float | None

    where:
        - args are the Float operands
        - ctx is the rounding context (provides precision via ctx.round_params())
        - Returns Float with round-to-odd result, or None if can't handle
    """

    @staticmethod
    def cvt_to_real(x: Real) -> Float | Fraction:
        """Converts `x` to `Float` or `Fraction`."""
        match x:
            case Float() | Fraction():
                return x
            case int():
                return Float.from_int(x)
            case float():
                return Float.from_float(x)
            case _:
                raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(x)}\' for x={x}')

    @staticmethod
    def cvt_to_float(x: Float | Fraction) -> Float:
        """Converts `x` to Float."""
        match x:
            case Float():
                return x
            case Fraction():
                if not is_dyadic(x):
                    raise ValueError(f'Cannot convert non-dyadic rational to Float: {x}')
                return Float.from_rational(x)
            case _:
                raise TypeError(f'Expected \'Float\' or \'Fraction\', got \'{type(x)}\' for x={x}')

    # Unary operations

    @abstractmethod
    def acos(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute acos(x) with round-to-odd."""
        ...

    @abstractmethod
    def acosh(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute acosh(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def asin(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute asin(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def asinh(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute asinh(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def atan(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute atan(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def atanh(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute atanh(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def cbrt(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute cbrt(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def cos(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute cos(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def cosh(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute cosh(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def erf(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute erf(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def erfc(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute erfc(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def exp(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute exp(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def exp2(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute 2**x with round-to-odd."""
        ...
    
    @abstractmethod
    def exp10(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute 10**x with round-to-odd."""
        ...
    
    @abstractmethod
    def expm1(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute exp(x) - 1 with round-to-odd."""
        ...
    
    @abstractmethod
    def fabs(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute abs(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def lgamma(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute lgamma(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def log(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute ln(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def log10(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute log10(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def log1p(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute log(1 + x) with round-to-odd."""
        ...
    
    @abstractmethod
    def log2(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute log2(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def neg(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute -x with round-to-odd."""
        ...
    
    @abstractmethod
    def sin(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute sin(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def sinh(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute sinh(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def sqrt(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute sqrt(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def tan(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute tan(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def tanh(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute tanh(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def tgamma(self, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute tgamma(x) with round-to-odd."""
        ...
    
    # Binary operations
    
    @abstractmethod
    def add(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Compute x + y with round-to-odd."""
        ...
    
    @abstractmethod
    def atan2(self, y: Float | Fraction, x: Float | Fraction, ctx: Context) -> Float | None:
        """Compute atan2(y, x) with round-to-odd."""
        ...
    
    @abstractmethod
    def copysign(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Return x with the sign of y."""
        ...
    
    @abstractmethod
    def div(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Compute x / y with round-to-odd."""
        ...
    
    @abstractmethod
    def fdim(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Compute max(x - y, 0) with round-to-odd."""
        ...
    
    @abstractmethod
    def fmod(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Compute x % y (C-style remainder) with round-to-odd."""
        ...
    
    @abstractmethod
    def fmax(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Compute max(x, y) with round-to-odd."""
        ...
    
    @abstractmethod
    def fmin(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Compute min(x, y) with round-to-odd."""
        ...
    
    @abstractmethod
    def hypot(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Compute sqrt(x*x + y*y) with round-to-odd."""
        ...
    
    @abstractmethod
    def mod(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Compute x % y (Python-style modulus) with round-to-odd."""
        ...
    
    @abstractmethod
    def mul(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Compute x * y with round-to-odd."""
        ...
    
    @abstractmethod
    def pow(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Compute x ** y with round-to-odd."""
        ...
    
    @abstractmethod
    def remainder(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Compute IEEE remainder of x / y with round-to-odd."""
        ...
    
    @abstractmethod
    def sub(self, x: Float | Fraction, y: Float | Fraction, ctx: Context) -> Float | None:
        """Compute x - y with round-to-odd."""
        ...
    
    # Ternary operations
    
    @abstractmethod
    def fma(self, x: Float | Fraction, y: Float | Fraction, z: Float | Fraction, ctx: Context) -> Float | None:
        """Compute x * y + z with round-to-odd."""
        ...
    
    # Mathematical constants
    
    @abstractmethod
    def const_e(self, ctx: Context) -> Float | None:
        """Compute Euler's number (e) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_log2e(self, ctx: Context) -> Float | None:
        """Compute log2(e) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_log10e(self, ctx: Context) -> Float | None:
        """Compute log10(e) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_ln2(self, ctx: Context) -> Float | None:
        """Compute ln(2) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_ln10(self, ctx: Context) -> Float | None:
        """Compute ln(10) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_pi(self, ctx: Context) -> Float | None:
        """Compute pi with round-to-odd."""
        ...
    
    @abstractmethod
    def const_pi_2(self, ctx: Context) -> Float | None:
        """Compute pi/2 with round-to-odd."""
        ...
    
    @abstractmethod
    def const_pi_4(self, ctx: Context) -> Float | None:
        """Compute pi/4 with round-to-odd."""
        ...
    
    @abstractmethod
    def const_1_pi(self, ctx: Context) -> Float | None:
        """Compute 1/pi with round-to-odd."""
        ...
    
    @abstractmethod
    def const_2_pi(self, ctx: Context) -> Float | None:
        """Compute 2/pi with round-to-odd."""
        ...
    
    @abstractmethod
    def const_2_sqrtpi(self, ctx: Context) -> Float | None:
        """Compute 2/sqrt(pi) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_sqrt2(self, ctx: Context) -> Float | None:
        """Compute sqrt(2) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_sqrt1_2(self, ctx: Context) -> Float | None:
        """Compute sqrt(1/2) with round-to-odd."""
        ...


class EngineList:
    def __init__(self):
        self._engines: list[tuple[int, Engine]] = []

    def register(self, engine: Engine, priority: int = 0):
        """
        Adds a new engine to the list with the given priority.
        Engines with higher priority are considered first during dispatch.
        """
        self._engines.append((priority, engine))
        self._engines.sort(key=lambda x: x[0], reverse=True)

    def __iter__(self) -> Iterator[Engine]:
        return (e for _, e in self._engines)


ENGINES = EngineList()
"""list of all engines"""

register_engine = ENGINES.register
