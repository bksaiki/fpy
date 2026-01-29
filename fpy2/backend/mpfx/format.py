"""
Abstract number system.
"""

import math

from typing import TypeAlias

from ...number import (
    MPFixedContext, MPBFixedContext, FixedContext,
    MPFloatContext, MPSFloatContext, MPBFloatContext, EFloatContext,
    RealFloat
)
from...utils import default_repr

__all__ = [
    'AbstractFormat',
    'SupportedContext'
]

SupportedContext: TypeAlias = (
    MPFixedContext | MPBFixedContext |
    MPFloatContext | MPSFloatContext | MPBFloatContext | EFloatContext
)


def _maxval_precision(bound: RealFloat, exp: int) -> int:
    """
    Computes the precision of `bound` when represented
    with the exponent `exp`, i.e., the number of bits
    required to represent `c` where `bound = c * 2**exp`.
    """
    assert not bound.is_zero()
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
        if prec <= 0 or math.isnan(prec):
            raise ValueError("`prec` must be positive.")
        if math.isnan(exp):
            raise ValueError("`exp` cannot be NaN")
        if math.isnan(bound):
            raise ValueError("`pos_bound` must not be NaN")
        if neg_bound is not None and math.isnan(neg_bound):
            raise ValueError("`neg_bound` must not be NaN")

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

    def __mul__(self, other: 'AbstractFormat') -> 'AbstractFormat':
        """Multiply two formats."""
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

    def contained_in(self, other: 'AbstractFormat') -> bool:
        """Check if this format is contained in another format."""
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        return (
            self.effective_prec() <= other.effective_prec()
            and self.exp >= other.exp
            and self.pos_bound <= other.pos_bound
            and self.neg_bound >= other.neg_bound
        )

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
