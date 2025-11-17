"""
Abstract base class for arithmetic engines.

An Engine provides methods for computing arithmetic operations such
that the result can be safely re-rounded under the rounding context provided.

Each method takes Float arguments and a rounding context, returning
a `Float` result or None if the engine cannot handle the computation.
Each method must guarantee that the returned Float is computed with sufficient
precision to allow safe re-rounding to the target precision specified in the context.

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
from typing import Iterator, TypeAlias

from ..context import Context
from ..number import Float

__all__ = [
    'Engine',
    'EngineList',
    'EngineArg',
    'EngineRes',
]


EngineArg: TypeAlias = Float | Fraction
EngineRes: TypeAlias = Float | Fraction | None


class Engine(ABC):
    """
    Abstract base class for arithmetic engines.

    Engines provide computational methods that implement various arithmetic
    operations. Each method returns a Float result computed with sufficient
    precision for safe re-rounding, or None if the engine cannot handle the
    operation.

    All computational methods follow the same signature pattern:
        method(self, *args: ArgType, ctx: Context) -> RetType

    where:
        - args are the Float operands
        - ctx is the rounding context (provides precision via ctx.round_params())
        - Returns Float with round-to-odd result, or None if can't handle
    """

    # Unary operations

    @abstractmethod
    def acos(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute acos(x) with round-to-odd."""
        ...

    @abstractmethod
    def acosh(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute acosh(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def asin(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute asin(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def asinh(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute asinh(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def atan(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute atan(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def atanh(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute atanh(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def cbrt(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute cbrt(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def ceil(self, x: EngineArg, ctx: Context) -> EngineRes:
        """
        Round a real number up to the nearest integer.
        """
        ...

    @abstractmethod
    def cos(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute cos(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def cosh(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute cosh(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def erf(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute erf(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def erfc(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute erfc(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def exp(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute exp(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def exp2(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute 2**x with round-to-odd."""
        ...
    
    @abstractmethod
    def exp10(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute 10**x with round-to-odd."""
        ...
    
    @abstractmethod
    def expm1(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute exp(x) - 1 with round-to-odd."""
        ...
    
    @abstractmethod
    def fabs(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute abs(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def floor(self, x: EngineArg, ctx: Context) -> EngineRes:
        """
        Round a real number down to the nearest integer.
        """
        ...

    @abstractmethod
    def lgamma(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute lgamma(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def log(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute ln(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def log10(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute log10(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def log1p(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute log(1 + x) with round-to-odd."""
        ...
    
    @abstractmethod
    def log2(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute log2(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def neg(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute -x with round-to-odd."""
        ...
    
    @abstractmethod
    def roundint(self, x: EngineArg, ctx: Context) -> EngineRes:
        """
        Round a real number to the nearest integer,
        rounding ties away from zero in halfway cases.
        """
        ...

    @abstractmethod
    def sin(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute sin(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def sinh(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute sinh(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def sqrt(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute sqrt(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def tan(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute tan(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def tanh(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute tanh(x) with round-to-odd."""
        ...
    
    @abstractmethod
    def tgamma(self, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute tgamma(x) with round-to-odd."""
        ...

    @abstractmethod
    def trunc(self, x: EngineArg, ctx: Context) -> EngineRes:
        """
        Rounds a real number towards the nearest integer
        with smaller or equal magnitude to `x`.
        """
        ...

    # Binary operations
    
    @abstractmethod
    def add(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Compute x + y with round-to-odd."""
        ...
    
    @abstractmethod
    def atan2(self, y: EngineArg, x: EngineArg, ctx: Context) -> EngineRes:
        """Compute atan2(y, x) with round-to-odd."""
        ...

    @abstractmethod
    def copysign(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Return x with the sign of y."""
        ...
    
    @abstractmethod
    def div(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Compute x / y with round-to-odd."""
        ...
    
    @abstractmethod
    def fdim(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Compute max(x - y, 0) with round-to-odd."""
        ...
    
    @abstractmethod
    def fmod(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Compute x % y (C-style remainder) with round-to-odd."""
        ...
    
    @abstractmethod
    def fmax(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Compute max(x, y) with round-to-odd."""
        ...
    
    @abstractmethod
    def fmin(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Compute min(x, y) with round-to-odd."""
        ...
    
    @abstractmethod
    def hypot(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Compute sqrt(x*x + y*y) with round-to-odd."""
        ...
    
    @abstractmethod
    def mod(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Compute x % y (Python-style modulus) with round-to-odd."""
        ...
    
    @abstractmethod
    def mul(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Compute x * y with round-to-odd."""
        ...
    
    @abstractmethod
    def pow(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Compute x ** y with round-to-odd."""
        ...
    
    @abstractmethod
    def remainder(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Compute IEEE remainder of x / y with round-to-odd."""
        ...
    
    @abstractmethod
    def sub(self, x: EngineArg, y: EngineArg, ctx: Context) -> EngineRes:
        """Compute x - y with round-to-odd."""
        ...
    
    # Ternary operations
    
    @abstractmethod
    def fma(self, x: EngineArg, y: EngineArg, z: EngineArg, ctx: Context) -> EngineRes:
        """Compute x * y + z with round-to-odd."""
        ...
    
    # Mathematical constants
    
    @abstractmethod
    def const_e(self, ctx: Context) -> EngineRes:
        """Compute Euler's number (e) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_log2e(self, ctx: Context) -> EngineRes:
        """Compute log2(e) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_log10e(self, ctx: Context) -> EngineRes:
        """Compute log10(e) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_ln2(self, ctx: Context) -> EngineRes:
        """Compute ln(2) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_ln10(self, ctx: Context) -> EngineRes:
        """Compute ln(10) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_pi(self, ctx: Context) -> EngineRes:
        """Compute pi with round-to-odd."""
        ...
    
    @abstractmethod
    def const_pi_2(self, ctx: Context) -> EngineRes:
        """Compute pi/2 with round-to-odd."""
        ...
    
    @abstractmethod
    def const_pi_4(self, ctx: Context) -> EngineRes:
        """Compute pi/4 with round-to-odd."""
        ...
    
    @abstractmethod
    def const_1_pi(self, ctx: Context) -> EngineRes:
        """Compute 1/pi with round-to-odd."""
        ...
    
    @abstractmethod
    def const_2_pi(self, ctx: Context) -> EngineRes:
        """Compute 2/pi with round-to-odd."""
        ...
    
    @abstractmethod
    def const_2_sqrtpi(self, ctx: Context) -> EngineRes:
        """Compute 2/sqrt(pi) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_sqrt2(self, ctx: Context) -> EngineRes:
        """Compute sqrt(2) with round-to-odd."""
        ...
    
    @abstractmethod
    def const_sqrt1_2(self, ctx: Context) -> EngineRes:
        """Compute sqrt(1/2) with round-to-odd."""
        ...


class EngineList:
    """
    Engine registry and dispatcher.
    """

    def __init__(self):
        self._items: list[tuple[int, Engine]] = []
        self._cached_engines: list[Engine] = []

    def register(self, engine: Engine, priority: int = 0):
        """
        Adds a new engine to the list with the given priority.
        Engines with higher priority are considered first during dispatch.
        """
        self._items.append((priority, engine))
        self._items.sort(key=lambda x: x[0], reverse=True)
        self._cached_engines = [e for _, e in self._items]

    def __iter__(self) -> Iterator[Engine]:
        return iter(self._cached_engines)


ENGINES = EngineList()
"""list of all engines"""

register_engine = ENGINES.register
