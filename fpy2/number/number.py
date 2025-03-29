"""
This module defines the floating-point number type `Float`.
"""

from typing import Callable, Optional, Self, TypeAlias

from .real import RealFloat
from .context import Context

from ..utils import default_repr, Ordering, rcomparable


_FloatCvt: TypeAlias = Callable[['Float'], float]

_default_float_converter: Optional[_FloatCvt] = None

def get_default_float_converter() -> _FloatCvt:
    """Gets the default `__float__` implementation for `Float`."""
    global _default_float_converter
    if _default_float_converter is None:
        raise RuntimeError('default float converter not set')
    return _default_float_converter

def set_default_float_converter(cvt: _FloatCvt):
    """Sets the default `__float__` implementation for `Float`."""
    global _default_float_converter
    _default_float_converter = cvt


@rcomparable(RealFloat)
class Float:
    """
    The basic floating-point number extended with infinities and NaN.

    This type encodes a base-2 number in unnormalized scientific notation:
    `(-1)^s * 2^exp * c` where:
     - `s` is the sign;
     - `exp` is the absolute position of the least-significant bit (LSB),
       also called the unnormalized exponent; and
     - `c` is the integer significand.

    There are no constraints on the values of `exp` and `c`.
    Unlike `RealFloat`, this number can encode infinity and NaN.

    This type can also encode uncertainty introduced by rounding.
    The uncertaintly is represented by an interval, also called
    a rounding envelope. The interval includes this value and
    extends either below or above it (`interval_down`).
    The interval always contains this value and may contain
    the other endpoint as well (`interval_closed`).
    The size of the interval is `2**(exp + interval_size)`.
    It must be the case that `interval_size <= 0`.

    Instances of `Float` are usually constructed under
    some rounding context, i.e., the result of an operation with rounding.
    The attribute `ctx` stores that rounding context if one exists.
    In general, `Float` objects should not be manually constructed,
    but rather through context-based constructors.
    """

    isinf: bool = False
    """is this number is infinite?"""

    isnan: bool = False
    """is this number is NaN?"""

    ctx: Optional[Context] = None
    """rounding context during construction"""

    _real: RealFloat
    """the real number (if it is real)"""

    def __init__(
        self,
        s: Optional[bool] = None,
        exp: Optional[int] = None,
        c: Optional[int] = None,
        *,
        x: Optional[RealFloat | Self] = None,
        e: Optional[int] = None,
        m: Optional[int] = None,
        isinf: Optional[bool] = None,
        isnan: Optional[bool] = None,
        interval_size: Optional[int] = None,
        interval_down: Optional[bool] = None,
        interval_closed: Optional[bool] = None,
        ctx: Optional[Context] = None
    ):
        if x is not None and not isinstance(x, RealFloat | Float):
            raise TypeError(f'expected Float, got {type(x)}')

        if isinf is not None:
            self.isinf = isinf
        elif isinstance(x, Float):
            self.isinf = x.isinf
        else:
            self.isinf = type(self).isinf

        if isnan is not None:
            self.isnan = isnan
        elif isinstance(x, Float):
            self.isnan = x.isnan
        else:
            self.isnan = type(self).isnan

        if self.isinf and self.isnan:
            raise ValueError('cannot be both infinite and NaN')

        if ctx is not None:
            self.ctx = ctx
        elif isinstance(x, Float):
            self.ctx = x.ctx
        else:
            self.ctx = type(self).ctx

        if isinstance(x, RealFloat):
            real = x
        elif isinstance(x, Float):
            real = x._real
        else:
            real = None

        self._real = RealFloat(
            s=s,
            exp=exp,
            c=c,
            x=real,
            e=e,
            m=m,
            interval_size=interval_size,
            interval_down=interval_down,
            interval_closed=interval_closed
        )

    def __repr__(self):
        return (f'{self.__class__.__name__}('
            + 's=' + repr(self._real.s)
            + ', exp=' + repr(self._real.exp)
            + ', c=' + repr(self._real.c)
            + ', isinf=' + repr(self.isinf)
            + ', isnan=' + repr(self.isnan)
            + ', interval_size=' + repr(self._real.interval_size)
            + ', interval_down=' + repr(self._real.interval_size)
            + ', interval_closed=' + repr(self._real.interval_closed)
            + ', ctx=' + repr(self.ctx)
            + ')'
        )

    def __eq__(self, other):
        ord = self.compare(other)
        return ord is not None and ord == Ordering.EQUAL

    def __lt__(self, other):
        ord = self.compare(other)
        return ord is not None and ord == Ordering.LESS

    def __le__(self, other):
        ord = self.compare(other)
        return ord is not None and ord != Ordering.GREATER

    def __gt__(self, other):
        ord = self.compare(other)
        return ord is not None and ord == Ordering.GREATER

    def __ge__(self, other):
        ord = self.compare(other)
        return ord is not None and ord != Ordering.LESS

    def __float__(self):
        cvt = get_default_float_converter()
        return cvt(self)

    @property
    def base(self):
        """Integer base of this number. Always 2."""
        return 2

    @property
    def s(self) -> bool:
        """Is the sign negative?"""
        return self._real.s

    @property
    def exp(self) -> int:
        """Absolute position of the LSB."""
        return self._real.exp

    @property
    def c(self) -> int:
        """Integer significand."""
        return self._real.c

    @property
    def p(self):
        """Minimum number of binary digits required to represent this number."""
        if self.is_nar():
            raise ValueError('cannot compute precision of infinity or NaN')
        return self._real.p

    @property
    def e(self) -> int:
        """
        Normalized exponent of this number.

        When `self.c == 0` (i.e. the number is zero), this method returns
        `self.exp - 1`. In other words, `self.c != 0` iff `self.e >= self.exp`.

        The interval `[self.exp, self.e]` represents the absolute positions
        of digits in the significand.
        """
        if self.is_nar():
            raise ValueError('cannot compute exponent of infinity or NaN')
        return self._real.e

    @property
    def n(self) -> int:
        """
        Position of the first unrepresentable digit below the significant digits.
        This is exactly `self.exp - 1`.
        """
        if self.is_nar():
            raise ValueError('cannot compute exponent of infinity or NaN')
        return self._real.n

    @property
    def m(self) -> int:
        """Significand of this number."""
        if self.is_nar():
            raise ValueError('cannot compute significand of infinity or NaN')
        return self._real.m

    @property
    def interval_size(self) -> int | None:
        """Rounding envelope: size relative to `2**exp`."""
        return self._real.interval_size

    @property
    def interval_down(self) -> bool | None:
        """Rounding envelope: extends below the value."""
        return self._real.interval_down

    @property
    def inexact(self) -> bool:
        """Return whether this number is inexact."""
        return self._real.inexact

    def is_zero(self) -> bool:
        """Returns whether this value represents zero."""
        return not self.is_nar() and self._real.is_zero()

    def is_positive(self) -> bool:
        """Returns whether this value is positive."""
        return not self.is_nar() and self._real.is_positive()

    def is_negative(self) -> bool:
        """Returns whether this value is negative."""
        return not self.is_nar() and self._real.is_negative()

    def is_integer(self) -> bool:
        """Returns whether this value is an integer."""
        return not self.is_nar() and self._real.is_integer()

    def is_finite(self) -> bool:
        """Returns whether this value is finite."""
        return not self.is_nar()

    def is_nonzero(self) -> bool:
        """Returns whether this value is (finite) nonzero."""
        return self.is_finite() and not self.is_zero()

    def is_nar(self) -> bool:
        """Return whether this number is infinity or NaN."""
        return self.isinf or self.isnan

    def is_representable(self) -> bool:
        """
        Checks if this number is representable under
        the rounding context during its construction.
        Usually just a sanity check.
        """
        return self.ctx is None or self.ctx.is_representable(self)

    def is_canonical(self) -> bool:
        """
        Returns if `x` is canonical under this context.

        This function only considers relevant attributes to judge
        if a value is canonical. Thus, there may be more than
        one canonical value for a given number despite the function name.
        The result of `self.normalize()` is always canonical.

        Raises a `ValueError` when `self.ctx is None`.
        """
        if self.ctx is None:
            raise ValueError(f'Float values without a context cannot be normalized: self={self}')
        return self.ctx.is_canonical(self)

    def as_real(self) -> RealFloat:
        """Returns the real part of this number."""
        if self.is_nar():
            raise ValueError('cannot convert infinity or NaN to real')
        return RealFloat(x=self._real)

    def normalize(self) -> 'Float':
        """
        Returns the canonical reprsentation of this number.

        Raises a `ValueError` when `self.ctx is None`.
        """
        if self.ctx is None:
            raise ValueError(f'cannot normalize without a context: self={self}')
        return self.ctx.normalize(self)

    def compare(self, other: Self | RealFloat) -> Optional[Ordering]:
        """
        Compare `self` and `other` values returning an `Optional[Ordering]`.
        """
        if isinstance(other, RealFloat):
            if self.isnan:
                return None
            elif self.isnan:
                if self.s:
                    return Ordering.LESS
                else:
                    return Ordering.GREATER
            else:
                return self._real.compare(other)
        elif isinstance(other, Float):
            if self.isnan or other.isnan:
                return None
            elif self.isinf:
                if other.isinf and self.s == other.s:
                    return Ordering.EQUAL
                elif self.s:
                    return Ordering.LESS
                else:
                    return Ordering.GREATER
            elif other.isinf:
                if other.s:
                    return Ordering.GREATER
                else:
                    return Ordering.LESS
            else:
                return self._real.compare(other._real)
        else:
            raise TypeError(f'expected Float or RealFloat, got {type(other)}')
