"""
This module defines the `Float` number type,
an arbitrary-precision floating-point number.
"""

import math

from fractions import Fraction
from typing import Optional, Self, TYPE_CHECKING

from ..globals import get_current_float_converter, get_current_str_converter
from ...utils import DefaultOr, Ordering, rcomparable, DEFAULT
from .reals import RealFloat


if TYPE_CHECKING:
    from ..context import Context
    from . import Real
else:
    Context = None  # type: ignore
    Real = None  # type: ignore


@rcomparable(RealFloat)
class Float:
    """
    The basic floating-point number extended with infinities and NaN.

    This type encodes a base-2 number in unnormalized scientific
    notation `(-1)^s * 2^exp * c` where:

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

    __slots__ = ('_isinf', '_isnan', '_ctx', '_real')

    _isinf: bool
    """is this number is infinite?"""

    _isnan: bool
    """is this number is NaN?"""

    _ctx: Optional[Context]
    """rounding context during construction"""

    _real: RealFloat
    """the real number (if it is real)"""

    def __init__(
        self,
        s: bool | None = None,
        exp: int | None = None,
        c: int | None = None,
        *,
        x: RealFloat | Self | None = None,
        e: int | None = None,
        m: int | None = None,
        isinf: bool | None = None,
        isnan: bool | None = None,
        interval_size: int | None = None,
        interval_down: bool | None = None,
        interval_closed: bool | None = None,
        ctx: DefaultOr[Optional[Context]] = DEFAULT
    ):
        match x:
            case None:
                real = None
            case RealFloat():
                real = x
            case Float():
                real = x._real
                if isinf is None:
                    isinf = x._isinf
                if isnan is None:
                    isnan = x._isnan
                if ctx is DEFAULT:
                    ctx = x._ctx
            case _:
                raise TypeError(f'expected \'RealFloat\' or \'Float\', got {type(x)} for x={x}')

        if isinf is not None:
            self._isinf = isinf
        else:
            self._isinf = False

        if isnan is not None:
            self._isnan = isnan
        else:
            self._isnan = False

        if self._isinf and self._isnan:
            raise ValueError('cannot be both infinite and NaN')

        if ctx is DEFAULT:
            self._ctx = None
        else:
            self._ctx = ctx

        # create a new RealFloat instance if any field is overriden
        if (s is None
            and exp is None
            and c is None
            and e is None
            and m is None
            and interval_size is None
            and interval_down is None
            and interval_closed is None):
            # no fields are overriden
            if real is None:
                # use the default `RealFloat`
                self._real = RealFloat()
            else:
                # use the `real` value exactly
                self._real = real
        else:
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
            + 's=' + repr(self._real._s)
            + ', exp=' + repr(self._real._exp)
            + ', c=' + repr(self._real._c)
            + ', isinf=' + repr(self._isinf)
            + ', isnan=' + repr(self._isnan)
            + ', interval_size=' + repr(self._real._interval_size)
            + ', interval_down=' + repr(self._real._interval_size)
            + ', interval_closed=' + repr(self._real._interval_closed)
            + ', ctx=' + repr(self._ctx)
            + ')'
        )

    def __str__(self):
        fn = get_current_str_converter()
        return fn(self)

    def __hash__(self): # type: ignore
        # Complex has __hash__ = None, so mypy thinks there's a type mismatch.
        return hash((self._isinf, self._isnan, self._real))

    def __eq__(self, other):
        if not isinstance(other, Float | RealFloat | int | float | Fraction):
            return False
        ord = self.compare(other)
        return ord == Ordering.EQUAL

    def __lt__(self, other):
        ord = self.compare(other)
        return ord == Ordering.LESS

    def __le__(self, other):
        ord = self.compare(other)
        return ord == Ordering.LESS or ord == Ordering.EQUAL

    def __gt__(self, other):
        ord = self.compare(other)
        return ord == Ordering.GREATER

    def __ge__(self, other):
        ord = self.compare(other)
        return ord == Ordering.GREATER or ord == Ordering.EQUAL

    def __neg__(self):
        """
        Unary minus.

        Returns this `Float` with opposite sign (`self.s`)
        and no context (`self.ctx is None`).
        """
        return Float(s=not self._real._s, x=self, ctx=None)

    def __pos__(self):
        """
        Unary plus.

        Returns this `Float` with no context (`self.ctx is None`).
        """
        return Float(s=False, x=self, ctx=None)

    def __abs__(self):
        """
        Absolute value.

        Returns this `Float` without the sign (`self.s = False`)
        with no context (`self.ctx is None`).
        """
        return Float(s=False, x=self, ctx=None)

    def __add__(self, other):
        """
        Addition: `self + other`.

        Returns the exact sum of `self` and `other` as a new `Float`.
        The result has no context (`self.ctx is None`).
        """
        match other:
            case Float():
                pass
            case RealFloat():
                other = Float.from_real(other)
            case int():
                other = Float.from_int(other)
            case float():
                other = Float.from_float(other)
            case Fraction():
                other = Float.from_rational(other)
            case _:
                raise TypeError(f'unsupported operand type(s) for +: \'RealFloat\' and \'{type(other)}\'')

        if self.isnan or other.isnan:
            # either is NaN
            return Float(isnan=True)
        elif self.isinf:
            # self is Inf
            if other.isinf:
                # other is Inf
                if self.s == other.s:
                    # Inf + Inf
                    return Float(s=self.s, isinf=True)
                else:
                    # Inf - Inf
                    return Float(isnan=True)
            else:
                # other is finite, Inf + y = Inf
                return Float(s=self.s, isinf=True)
        elif other.isinf:
            # self is finite, x + Inf = Inf
            return Float(s=other.s, isinf=True)
        else:
            # both are finite
            return Float.from_real(self._real + other._real)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        """
        Multiplication: `self * other`.

        Returns the exact product of `self` and `other` as a new `Float`.
        The result has no context (`self.ctx is None`).
        """
        match other:
            case Float():
                pass
            case RealFloat():
                other = Float.from_real(other)
            case int():
                other = Float.from_int(other)
            case float():
                other = Float.from_float(other)
            case Fraction():
                other = Float.from_rational(other)
            case _:
                raise TypeError(f'unsupported operand type(s) for +: \'RealFloat\' and \'{type(other)}\'')

        if self.isnan or other.isnan:
            # either is NaN
            return Float(isnan=True)
        elif self.isinf:
            # self is Inf
            if other.is_zero():
                # Inf * 0 = NaN
                return Float(isnan=True)
            else:
                # Inf * y = Inf
                s = self._real._s != other._real._s
                return Float(s=s, isinf=True)
        elif other.isinf:
            # y is Inf
            if self.is_zero():
                # 0 * Inf = NaN
                return Float(isnan=True)
            else:
                # x * Inf = Inf
                s = self._real._s != other._real._s
                return Float(s=s, isinf=True)
        else:
            # both are finite
            return Float.from_real(self._real * other._real)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other: 'Real'):
        if TYPE_CHECKING:
            return Float()
        else:
            raise RuntimeError('FPy runtime: do not call directly')

    def __rtruediv__(self, other: 'Real'):
        if TYPE_CHECKING:
            return Float()
        else:
            raise RuntimeError('FPy runtime: do not call directly')

    def __pow__(self, exponent: 'Real'):
        """
        Raising `self` by `exponent` exactly.

        This operation is only valid for `exponent` of type `int` with `exponent >= 0`.
        """
        if not isinstance(exponent, int):
            raise TypeError(f'unsupported operand type(s) for **: \'RealFloat\' and \'{type(exponent)}\'')
        if exponent < 0:
            raise ValueError('negative exponent unsupported; cannot be implemented exactly')

        if self.is_nar():
            s = self._real._s and (exponent % 2 != 0)
            return Float(x=self, s=s, ctx=None)
        else:
            return Float.from_real(self._real ** exponent)

    def __rpow__(self, base: 'Real'):
        if TYPE_CHECKING:
            return Float()
        else:
            raise RuntimeError('FPy runtime: do not call directly')

    def __trunc__(self):
        if self.is_nar():
            raise ValueError('cannot round infinity or NaN')
        return self._real.__trunc__()

    def __floor__(self):
        if self.is_nar():
            raise ValueError('cannot round infinity or NaN')
        return self._real.__floor__()

    def __ceil__(self):
        if self.is_nar():
            raise ValueError('cannot round infinity or NaN')
        return self._real.__ceil__()

    def __round__(self, *args, **kwargs):
        if self.is_nar():
            raise ValueError('cannot round infinity or NaN')
        return self._real.__round__()

    def __floordiv__(self, other: 'Real'):
        if TYPE_CHECKING:
            return Float()
        else:
            raise RuntimeError('FPy runtime: do not call directly')

    def __rfloordiv__(self, other: 'Real'):
        if TYPE_CHECKING:
            return Float()
        else:
            raise RuntimeError('FPy runtime: do not call directly')

    def __mod__(self, other: 'Real'):
        if TYPE_CHECKING:
            return Float()
        else:
            raise RuntimeError('FPy runtime: do not call directly')

    def __rmod__(self, other: 'Real'):
        if TYPE_CHECKING:
            return Float()
        else:
            raise RuntimeError('FPy runtime: do not call directly')

    def __float__(self):
        """
        Casts this value exactly to a native Python float.

        If the value is not representable, a `ValueError` is raised.
        """
        fn = get_current_float_converter()
        return fn(self)

    def __int__(self):
        """
        Casts this value exactly to a native Python integer.

        If the value is not representable, a `ValueError` is raised.
        """
        if not self.is_integer():
            raise ValueError(f'{self} is not an integer')
        return int(self._real)

    @property
    def isinf(self) -> bool:
        """Is this number infinite?"""
        return self._isinf

    @property
    def isnan(self) -> bool:
        """Is this number NaN?"""
        return self._isnan

    @property
    def ctx(self) -> Optional[Context]:
        """
        Rounding context under which this number was constructed.

        If `None`, this number was constructed without a context.
        In that case, the number is always exact and representable.
        """
        return self._ctx

    @property
    def base(self):
        """Integer base of this number. Always 2."""
        return 2

    @property
    def s(self) -> bool:
        """Is the sign negative?"""
        return self._real._s

    @property
    def exp(self) -> int:
        """Absolute position of the LSB."""
        return self._real._exp

    @property
    def c(self) -> int:
        """Integer significand."""
        return self._real._c

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
        return self._real._interval_size

    @property
    def interval_down(self) -> bool | None:
        """Rounding envelope: extends below the value."""
        return self._real._interval_down

    @property
    def inexact(self) -> bool:
        """Return whether this number is inexact."""
        return self._real.inexact

    @property
    def numerator(self):
        if self.is_nar():
            raise ValueError('cannot compute numerator of infinity or NaN')
        return self._real.as_rational().numerator

    @property
    def denominator(self):
        if self.is_nar():
            raise ValueError('cannot compute denominator of infinity or NaN')
        return self._real.as_rational().denominator

    def is_zero(self) -> bool:
        """Returns whether this value represents zero."""
        return not self.is_nar() and self._real.is_zero()

    def is_positive(self) -> bool:
        """Returns whether this value is positive."""
        if self._isnan:
            return False
        elif self._isinf:
            return not self._real._s
        else:
            return self._real.is_positive()

    def is_negative(self) -> bool:
        """Returns whether this value is negative."""
        if self._isnan:
            return False
        elif self._isinf:
            return self._real._s
        else:
            return self._real.is_negative()

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
        return self._isinf or self._isnan

    def is_representable(self) -> bool:
        """
        Checks if this number is representable under
        the rounding context during its construction.
        Usually just a sanity check.
        """
        return self._ctx is None or self._ctx.representable_under(self)

    def is_canonical(self) -> bool:
        """
        Returns if `x` is canonical under this context.

        This function only considers relevant attributes to judge
        if a value is canonical. Thus, there may be more than
        one canonical value for a given number despite the function name.
        The result of `self.normalize()` is always canonical.

        Raises a `ValueError` when `self.ctx is None`.
        """
        if self._ctx is None:
            raise ValueError(f'Float values without a context cannot be normalized: self={self}')
        return self._ctx.canonical_under(self)

    def is_normal(self) -> bool:
        """
        Returns if this number is "normal".

        For IEEE-style contexts, this means that the number is finite, non-zero,
        and `x.normalize()` has full precision.
        """
        if self._ctx is None:
            raise ValueError(f'Float values without a context cannot be normalized: self={self}')
        return self._ctx.normal_under(self)

    def is_more_significant(self, n: int) -> bool:
        """
        Returns `True` iff this value only has significant digits above `n`,
        that is, every non-zero digit in the number is more significant than `n`.

        Raises a `ValueError` when `self.is_nar()`.

        This method is equivalent to::

            assert not self.is_nar()
            _, lo = self.split(n)
            return lo.is_zero()
        """
        if not isinstance(n, int):
            raise TypeError(f'expected \'int\' for n, got {n}')
        if self.is_nar():
            raise ValueError('cannot check significance of infinity or NaN')
        return self._real.is_more_significant(n)

    def as_rational(self) -> Fraction:
        """
        Converts this value to a `Fraction` representing the same value.

        If the value is not representable, a `ValueError` is raised.
        """
        if self.is_nar():
            raise ValueError(f'{self} is not representable as a rational number')
        return self._real.as_rational()

    @staticmethod
    def nan(s: bool = False, ctx: Optional[Context] = None):
        """
        Returns a `Float` representation of NaN.

        Optionally specify a rounding context under which to
        construct this value. If a rounding context is specified,
        `x` must be representable under `ctx`.
        """
        return Float(isnan=True, s=s, ctx=ctx)

    @staticmethod
    def inf(s: bool = False, ctx: Optional[Context] = None):
        """
        Returns a `Float` representation of infinity.

        Optionally specify a rounding context under which to
        construct this value. If a rounding context is specified,
        `x` must be representable under `ctx`.
        """
        return Float(isinf=True, s=s, ctx=ctx)

    @staticmethod
    def zero(s: bool = False, ctx: Optional[Context] = None):
        """
        Returns a `Float` representation of zero.

        Optionally specify a rounding context under which to
        construct this value. If a rounding context is specified,
        `x` must be representable under `ctx`.
        """
        return Float.from_real(RealFloat.zero(s), ctx)

    @staticmethod
    def from_real(x: RealFloat, ctx: Optional[Context] = None, checked: bool = True):
        """
        Converts a `RealFloat` number to a `Float` number.

        Optionally specify a rounding context under which to
        construct this value. If a rounding context is specified,
        `x` must be representable under `ctx` unless `checked=False`.
        """
        if not isinstance(x, RealFloat):
            raise TypeError(f'expected RealFloat, got {type(x)}')

        if ctx is None:
            # no context specified, so its rounded exactly
            return Float(x=x, ctx=ctx)
        elif checked and not ctx.representable_under(x):
            # context specified, but `x` is not representable under it
            raise ValueError(f'{x} is not representable under {ctx}')
        else:
            return Float(x=x, ctx=ctx)

    @staticmethod
    def from_int(x: int, ctx: Optional[Context] = None, checked: bool = True):
        """
        Converts an integer to a `Float` number.

        Optionally specify a rounding context under which to
        construct this value. If a rounding context is specified,
        `x` must be representable under `ctx` unless `checked=False`.
        """
        if not isinstance(x, int):
            raise TypeError(f'expected int, got {type(x)}')

        xr = RealFloat.from_int(x)
        return Float.from_real(xr, ctx, checked)

    @staticmethod
    def from_float(x: float, ctx: Optional[Context] = None, checked: bool = True):
        """
        Converts a native Python float to a `Float` number.

        Optionally specify a rounding context under which to
        construct this value. If a rounding context is specified,
        `x` must be representable under `ctx` unless `checked=False`.
        """
        if not isinstance(x, float):
            raise TypeError(f'expected int, got {type(x)}')

        if math.isnan(x):
            s = math.copysign(1, x) < 0
            return Float.nan(s=s, ctx=ctx)
        elif math.isinf(x):
            s = x < 0
            return Float.inf(s=s, ctx=ctx)
        else:
            xr = RealFloat.from_float(x)
            return Float.from_real(xr, ctx, checked)

    @staticmethod
    def from_rational(x: Fraction, ctx: Optional[Context] = None, checked: bool = True):
        """
        Converts a `Fraction` to a `Float` number.

        Optionally specify a rounding context under which to
        construct this value. If a rounding context is specified,
        `x` must be representable under `ctx` unless `checked=False`.
        """
        if not isinstance(x, Fraction):
            raise TypeError(f'expected Fraction, got {type(x)}')
        xr = RealFloat.from_rational(x)
        return Float.from_real(xr, ctx, checked)

    def as_real(self) -> RealFloat:
        """Returns the real part of this number."""
        if self.is_nar():
            raise ValueError('cannot convert infinity or NaN to real')
        return self._real

    def normalize(self, p: int | None = None, n: int | None = None):
        """
        Returns a value numerically equivalent to `self` based on
        precision `p` and position `n`:

        - `None, None`: the canonical representation of `self` under `self.ctx`.
        - `p, None`: a copy of `self` that has exactly `p` bits of precision.
        - `None, n`: a copy of `self` where `self.exp == n + 1`.
        - `p, n`: a copy of `self` such that `self.exp >= n + 1` and
            has maximal precision up to `p` bits.

        Raises a `ValueError` if no such value exists, i.e.,
        if the value cannot be represented with the given `p` and `n`.
        """
        if p is not None:
            if not isinstance(p, int):
                raise TypeError(f'expected \'int\' for p={p}, got {type(p)}')
            if p < 0:
                raise ValueError(f'expected non-negative integer for p={p}')
        if n is not None and not isinstance(n, int):
            raise TypeError(f'expected \'int\' for n={n}, got {type(n)}')

        if p is None and n is None:
            # normalize under the context
            if self._ctx is None:
                raise ValueError(f'cannot normalize without a context: self={self}')
            return self._ctx.normalize(self)
        else:
            # normalize with given parameters
            if self.isnan:
                return Float(isnan=True, s=self.s, ctx=self._ctx)
            elif self.isinf:
                return Float(isinf=True, s=self.s, ctx=self._ctx)
            else:
                xr = self._real.normalize(p, n)
                return Float(x=xr, ctx=self._ctx)

    def split(self, n: int):
        """
        Splits `self` into two `Float` values where the first value represents
        the digits above `n` and the second value represents the digits below
        and including `n`.
        """
        if not isinstance(n, int):
            raise TypeError(f'expected \'int\' for n={n}, got {type(n)}')

        if self.isnan:
            hi = Float(isnan=True, s=self.s, ctx=self._ctx)
            lo = Float(isnan=True, s=self.s, ctx=self._ctx)
        elif self.isinf:
            hi = Float(isinf=True, s=self.s, ctx=self._ctx)
            lo = Float(isinf=True, s=self.s, ctx=self._ctx)
        else:
            hr, lr = self._real.split(n)
            hi = Float(x=hr, ctx=self._ctx)
            lo = Float(x=lr, ctx=self._ctx)

        return hi, lo


    def round(self, ctx: Context):
        """
        Rounds this number under the given context.

        This method is equivalent to `ctx.round(self)`.
        """
        if not isinstance(ctx, Context):
            raise TypeError(f'expected Context, got {type(ctx)}')
        return ctx.round(self)

    def round_at(self, ctx: Context, n: int):
        """
        Rounds this number at the given position.

        This method is equivalent to `self.ctx.round_at(self, n)`.
        """
        if not isinstance(ctx, Context):
            raise TypeError(f'expected Context, got {type(ctx)}')
        return ctx.round_at(self, n)

    def round_integer(self, ctx: Context):
        """
        Rounds this number to the nearest integer.

        This method is equivalent to `self.ctx.round_integer(self)`.
        """
        if not isinstance(ctx, Context):
            raise TypeError(f'expected Context, got {type(ctx)}')
        return ctx.round_integer(self)

    def compare(self, other: Self | RealFloat | int | float | Fraction) -> Ordering | None:
        """
        Compare `self` and `other` values returning an `Optional[Ordering]`.
        """
        if self._isnan:
            return None
        else:
            match other:
                case RealFloat():
                    if self._isinf:
                        if self.s:
                            return Ordering.LESS
                        else:
                            return Ordering.GREATER
                    else:
                        return self._real.compare(other)
                case Float():
                    if other._isnan:
                        return None
                    elif self._isinf:
                        if other._isinf and self.s == other.s:
                            return Ordering.EQUAL
                        elif self.s:
                            return Ordering.LESS
                        else:
                            return Ordering.GREATER
                    elif other._isinf:
                        if other.s:
                            return Ordering.GREATER
                        else:
                            return Ordering.LESS
                    else:
                        return self._real.compare(other._real)
                case int():
                    return self.compare(RealFloat.from_int(other))
                case float():
                    return self.compare(Float.from_float(other))
                case Fraction():
                    if self._isinf:
                        if self.s:
                            return Ordering.LESS
                        else:
                            return Ordering.GREATER
                    else:
                        return self._real.compare(other)
                case _:
                    return False
