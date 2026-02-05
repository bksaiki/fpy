"""
Abstract number system.
"""

import math

from typing import TypeAlias

from ...number import (
    MPFixedContext, MPBFixedContext, FixedContext,
    MPFloatContext, MPSFloatContext, MPBFloatContext, EFloatContext,
    ExpContext,
    RealFloat
)
from ...utils import default_repr
from ...utils.ordering import Ordering

__all__ = [
    'AbstractFormat',
    'SupportedContext'
]

SupportedContext: TypeAlias = (
    MPFixedContext | MPBFixedContext | ExpContext |
    MPFloatContext | MPSFloatContext | MPBFloatContext | EFloatContext
)


def _maxval_precision(bound: RealFloat, exp: int) -> int:
    """
    Computes the precision of `bound` when represented
    with the exponent `exp`, i.e., the number of bits
    required to represent `c` where `bound = c * 2**exp`.
    """
    n = exp - 1
    bound = bound.normalize(n=n)
    return bound.c.bit_length()


@default_repr
class AbstractFormat:
    """
    Abstract number system.
    - `prec`: maximum precision (use float('inf') for unbounded)
    - `exp`: minimum unnormalized exponent (use float('-inf') for unbounded)
    - `pos_bound`: largest positive representable number (use float('inf') for unbounded)
    - `neg_bound`: largest negative representable number (use float('inf') for unbounded magnitude)
    """

    prec: int | float
    exp: int | float
    pos_bound: RealFloat | float
    neg_bound: RealFloat | float

    def __init__(
        self,
        prec: int | float,
        exp: int | float,
        bound: RealFloat | float,
        *,
        neg_bound: RealFloat | float | None = None,
    ):
        if prec <= 0:
            raise ValueError("`prec` must be positive.")

        self.prec = prec
        self.exp = exp
        self.pos_bound = bound
        self.neg_bound = -bound if neg_bound is None else neg_bound

    def __hash__(self):
        return hash((self.prec, self.exp, self.pos_bound, self.neg_bound))

    def __eq__(self, other):
        return (
            isinstance(other, AbstractFormat)
            and self.prec == other.prec
            and self.exp == other.exp
            and self.pos_bound == other.pos_bound
            and self.neg_bound == other.neg_bound
        )

    def __str__(self) -> str:
        return f'A({self.prec}, {self.exp}, +{str(self.pos_bound)}, {str(self.neg_bound)})'

    def __pos__(self) -> 'AbstractFormat':
        """Identity of the format."""
        return AbstractFormat(self.prec, self.exp, self.pos_bound, neg_bound=self.neg_bound)

    def __neg__(self) -> 'AbstractFormat':
        """Negation of the format (swaps positive and negative bounds)."""
        return AbstractFormat(self.prec, self.exp, -self.neg_bound, neg_bound=-self.pos_bound)

    def __abs__(self) -> 'AbstractFormat':
        """Absolute value of the format (makes negative bound equal to positive bound)."""
        return AbstractFormat(self.prec, self.exp, self.pos_bound, neg_bound=0)

    def __add__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """
        Addition of two formats.

        Produces a format that can represent the sum of any
        pair of representable numbers from the two formats.
        """
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        # exponent: min(e1, e2)
        # bounds: b1 + b2
        exp = min(self.exp, other.exp)
        pos_bound = self.pos_bound + other.pos_bound
        neg_bound = self.neg_bound + other.neg_bound

        # compute precision based on bounds and exponent
        if isinstance(pos_bound, float) or isinstance(neg_bound, float):
            # precision must be unbounded since we need to represent
            # any sum of the form `+HUGE - quantum`
            prec = float('inf')
        elif isinstance(exp, float):
            # no subnormalization point means we need to represent
            # any sum of the form `+x - SMALL`
            prec = float('inf')
        else:
            # compute the magnitude of the largest bound
            max_bound = max(pos_bound, abs(neg_bound))

            # normalize the largest bound with the desired quantum
            # its precision is the required precision
            max_bound = max_bound.normalize(n=exp - 1)
            prec = max_bound.p

        return AbstractFormat(prec, exp, pos_bound, neg_bound=neg_bound)

    def __sub__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """
        Subtraction of two formats.

        Produces a format that can represent the difference of any
        pair of representable numbers from the two formats.
        """
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        # exponent: min(e1, e2)
        # bounds: b1 + b2
        exp = min(self.exp, other.exp)
        pos_bound = self.pos_bound + abs(other.neg_bound)
        neg_bound = self.neg_bound + abs(other.pos_bound)

        # compute precision based on bounds and exponent
        if isinstance(pos_bound, float) or isinstance(neg_bound, float):
            # precision must be unbounded since we need to represent
            # any difference of the form `+HUGE - quantum`
            prec = float('inf')
        elif isinstance(exp, float):
            # no subnormalization point means we need to represent
            # any difference of the form `+x - SMALL`
            prec = float('inf')
        else:
            # compute the magnitude of the largest bound
            max_bound = max(pos_bound, abs(neg_bound))

            # normalize the largest bound with the desired quantum
            # its precision is the required precision
            max_bound = max_bound.normalize(n=exp - 1)
            prec = max_bound.p

        return AbstractFormat(prec, exp, pos_bound, neg_bound=neg_bound)


    def __mul__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """
        Multiplication of two formats.

        Produces a format that can represent the product of any
        pair of representable numbers from the two formats.
        """
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        # precision: p1 + p2
        # exponent: e1 + e2
        # bounds: b1 * b2
        prec = self.effective_prec() + other.effective_prec()
        exp = self.exp + other.exp
        pos_bound = max(self.pos_bound * other.pos_bound, self.neg_bound * other.neg_bound)
        neg_bound = max(self.pos_bound * other.neg_bound, self.neg_bound * other.pos_bound)
        return AbstractFormat(prec, exp, pos_bound, neg_bound=neg_bound)

    def __and__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """Intersection of two formats."""
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        prec = min(self.prec, other.prec)
        exp = max(self.exp, other.exp)
        pos_bound = min(self.pos_bound, other.pos_bound)
        neg_bound = max(self.neg_bound, other.neg_bound)
        return AbstractFormat(prec, exp, pos_bound, neg_bound=neg_bound)

    def __or__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """Union of two formats."""
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        prec = max(self.prec, other.prec)
        exp = min(self.exp, other.exp)
        pos_bound = max(self.pos_bound, other.pos_bound)
        neg_bound = min(self.neg_bound, other.neg_bound)
        return AbstractFormat(prec, exp, pos_bound, neg_bound=neg_bound)

    def __lt__(self, other) -> bool:
        if not isinstance(other, AbstractFormat):
            return NotImplemented
        return self.compare(other) == Ordering.LESS

    def __le__(self, other) -> bool:
        if not isinstance(other, AbstractFormat):
            return NotImplemented
        result = self.compare(other)
        return result == Ordering.LESS or result == Ordering.EQUAL

    def __gt__(self, other) -> bool:
        if not isinstance(other, AbstractFormat):
            return NotImplemented
        return self.compare(other) == Ordering.GREATER

    def __ge__(self, other) -> bool:
        if not isinstance(other, AbstractFormat):
            return NotImplemented
        result = self.compare(other)
        return result == Ordering.GREATER or result == Ordering.EQUAL

    @property
    def bound(self) -> RealFloat | float:
        """Maximum magnitude bound (pos or neg)."""
        return max(self.pos_bound, abs(self.neg_bound))

    @staticmethod
    def from_context(ctx: SupportedContext) -> 'AbstractFormat':
        match ctx:
            case FixedContext() if not ctx.signed:
                pos_maxval = ctx.maxval().as_real()
                neg_maxval = RealFloat.from_int(0)
                return AbstractFormat(float('inf'), ctx.expmin, pos_maxval, neg_bound=neg_maxval)
            case MPFloatContext():
                return AbstractFormat(ctx.pmax, float('-inf'), float('inf'))
            case MPSFloatContext():
                return AbstractFormat(ctx.pmax, ctx.expmin, float('inf'))
            case MPBFloatContext():
                pos_maxval = ctx.maxval().as_real()
                neg_maxval = ctx.maxval(True).as_real()
                return AbstractFormat(ctx.pmax, ctx.expmin, pos_maxval, neg_bound=neg_maxval)
            case EFloatContext():
                pos_maxval = ctx.maxval().as_real()
                neg_maxval = ctx.maxval(True).as_real()
                return AbstractFormat(ctx.pmax, ctx.expmin, pos_maxval, neg_bound=neg_maxval)
            case MPFixedContext():
                return AbstractFormat(float('inf'), ctx.expmin, float('inf'))
            case MPBFixedContext():
                pos_maxval = ctx.maxval().as_real()
                neg_maxval = ctx.maxval(True).as_real()
                return AbstractFormat(float('inf'), ctx.expmin, pos_maxval, neg_bound=neg_maxval)
            case ExpContext():
                pos_maxval = ctx.maxval().as_real()
                neg_maxval = RealFloat.from_int(0)
                expmin = ctx.minval().exp
                return AbstractFormat(1, expmin, pos_maxval, neg_bound=neg_maxval)
            case _:
                raise TypeError(f'Unsupported context type: {type(ctx)}')

    def effective_prec(self):
        """Effective maximum precision."""
        if isinstance(self.prec, float) and not isinstance(self.bound, float):
            # bounded fixed-point format
            assert not isinstance(self.exp, float)
            return _maxval_precision(self.bound, self.exp)

        if not isinstance(self.prec, float) and not isinstance(self.bound, float) and not isinstance(self.exp, float):
            # bounded floating-point format
            # check against the cutoff value
            cutoff = RealFloat(False, self.exp, 1 << self.prec)
            if self.bound <= cutoff:
                # format acts like a fixed-point format
                return _maxval_precision(self.bound, self.exp)

        # everything else
        return self.prec

    def compare(self, other: 'AbstractFormat') -> Ordering | None:
        """Compare this format with another using the containment partial order.

        Returns:
            Ordering.LESS if this format is contained in other,
            Ordering.EQUAL if both formats contain each other,
            Ordering.GREATER if other is contained in this format,
            None if formats are incomparable.
        """
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')

        prec = self.effective_prec()
        other_prec = other.effective_prec()

        polarity: bool | None = None # True if larger, False if smaller
        if prec != other_prec:
            polarity = prec > other_prec
        if self.exp != other.exp:
            exp_polarity = self.exp < other.exp
            if polarity is None:
                polarity = exp_polarity
            elif polarity != exp_polarity:
                return None
        if self.pos_bound != other.pos_bound:
            bound_polarity = self.pos_bound > other.pos_bound
            if polarity is None:
                polarity = bound_polarity
            elif polarity != bound_polarity:
                return None
        if self.neg_bound != other.neg_bound:
            bound_polarity = self.neg_bound < other.neg_bound
            if polarity is None:
                polarity = bound_polarity
            elif polarity != bound_polarity:
                return None

        if polarity is None:
            return Ordering.EQUAL
        elif polarity:
            return Ordering.GREATER
        else:
            return Ordering.LESS

    def contained_in(self, other: 'AbstractFormat') -> bool:
        """Check if this format is contained in another format."""
        result = self.compare(other)
        return result == Ordering.LESS or result == Ordering.EQUAL

    def with_prec_offset(self, delta: int) -> 'AbstractFormat':
        """
        Return a new format with precision adjusted by delta.

        Args:
            delta: Amount to add to precision (can be negative).
        Returns:
            New AbstractFormat with adjusted precision.
        """
        new_prec = self.prec + delta
        if new_prec < 1:
            raise ValueError("resulting precision must be at least 1")
        return AbstractFormat(new_prec, self.exp, self.pos_bound, neg_bound=self.neg_bound)

    def with_exp_offset(self, delta: int) -> 'AbstractFormat':
        """
        Return a new format with exponent adjusted by delta.

        Args:
            delta: Amount to add to exponent (can be negative).
        Returns:
            New AbstractFormat with adjusted exponent.
        """
        new_exp = self.exp + delta
        return AbstractFormat(self.prec, new_exp, self.pos_bound, neg_bound=self.neg_bound)

    def with_bounds_scale(self, factor: RealFloat) -> 'AbstractFormat':
        """
        Return a new format with bounds scaled by factor.

        Args:
            factor: Factor to multiply bounds by (must be positive).
        Returns:
            New AbstractFormat with scaled bounds.
        """
        if factor <= 0:
            raise ValueError("Factor must be positive")

        # inf * positive = inf, so no need to check
        new_pos_bound = self.pos_bound * factor
        new_neg_bound = self.neg_bound * factor
        return AbstractFormat(self.prec, self.exp, new_pos_bound, neg_bound=new_neg_bound)
