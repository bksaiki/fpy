"""
Abstract number system.
"""

from typing import TypeAlias

from ...number import (
    MPFixedContext, MPBFixedContext, FixedContext,
    MPFloatContext, MPSFloatContext, MPBFloatContext, EFloatContext,
    RealFloat
)
from...utils import default_repr

__all__ = [
    'AbstractFormat',
    'SupporedContext'
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
    - `prec`: maximum precision
    - `exp`: minimum unnormalized exponent
    - `pos_bound`: largest positive representable number
    - `neg_bound`: largest negative representable number
    """

    prec: int | None
    exp: int | None
    pos_bound: RealFloat | None
    neg_bound: RealFloat | None

    def __init__(
        self,
        prec: int | None,
        exp: int | None,
        pos_bound: RealFloat | None,
        *,
        neg_bound: RealFloat | None = None,
    ):
        if prec is None and exp is None:
            raise ValueError("At least one of `prec` or `exp` must be specified.")
        if prec is not None and prec <= 0:
            raise ValueError("`prec` must be positive.")

        self.prec = prec
        self.exp = exp
        self.pos_bound = pos_bound

        if neg_bound is None:
            if pos_bound is None:
                self.neg_bound = None
            else:
                self.neg_bound = -pos_bound
        else:
            self.neg_bound = neg_bound

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

    @property
    def bound(self) -> RealFloat | None:
        if self.pos_bound is None:
            if self.neg_bound is None:
                return None
            else:
                return abs(self.neg_bound)
        elif self.neg_bound is None:
            return self.pos_bound
        else:
            return max(self.pos_bound, abs(self.neg_bound))

    @staticmethod
    def from_context(ctx: SupportedContext) -> 'AbstractFormat':
        match ctx:
            case FixedContext() if not ctx.signed:
                pos_maxval = ctx.maxval().as_real()
                neg_maxval = RealFloat.from_int(0)
                return AbstractFormat(None, ctx.expmin, pos_maxval, neg_bound=neg_maxval)
            case MPFloatContext():
                return AbstractFormat(ctx.pmax, None, None)
            case MPSFloatContext():
                return AbstractFormat(ctx.pmax, ctx.expmin, None)
            case MPBFloatContext():
                pos_maxval = ctx.maxval().as_real()
                neg_maxval = abs(ctx.maxval(True)).as_real()
                return AbstractFormat(ctx.pmax, ctx.expmin, pos_maxval, neg_bound=neg_maxval)
            case EFloatContext():
                pos_maxval = ctx.maxval().as_real()
                neg_maxval = abs(ctx.maxval(True)).as_real()
                return AbstractFormat(ctx.pmax, ctx.expmin, pos_maxval, neg_bound=neg_maxval)
            case MPFixedContext():
                return AbstractFormat(None, ctx.expmin, None)
            case MPBFixedContext():
                pos_maxval = ctx.maxval().as_real()
                neg_maxval = abs(ctx.maxval(True)).as_real()
                return AbstractFormat(None, ctx.expmin, pos_maxval, neg_bound=neg_maxval)
            case _:
                raise TypeError(f'Unsupported context type: {type(ctx)}')

    def _effective_prec(self):
        """Effective maximum precision."""
        if self.prec is None and self.bound is not None:
            # bounded fixed-point format
            assert self.exp is not None
            return _maxval_precision(self.bound, self.exp)

        if self.prec is not None and self.exp is not None and self.bound is not None:
            # bounded floating-point format
            # check against the cutoff value
            cutoff = RealFloat(False, self.exp, 1 << self.prec)
            if self.bound <= cutoff:
                # format acts like a fixed-point format
                return _maxval_precision(self.bound, self.exp)

        # everything else
        return self.prec

    def _effective_params(self):
        prec = self._effective_prec()
        if prec is None:
            prec = float('inf')

        exp = -float('inf') if self.exp is None else self.exp
        pos_bound = float('inf') if self.bound is None else self.bound
        neg_bound = float('inf') if self.neg_bound is None else abs(self.neg_bound)

        return prec, exp, pos_bound, neg_bound

    def contained_in(self, other: 'AbstractFormat') -> bool:
        """Check if this format is contained in another format."""
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        prec1, exp1, pos_bound1, neg_bound1 = self._effective_params()
        prec2, exp2, pos_bound2, neg_bound2 = other._effective_params()
        return (
            prec1 <= prec2
            and exp1 >= exp2
            and pos_bound1 <= pos_bound2
            and neg_bound1 <= neg_bound2
        )

    def with_prec_offset(self, delta: int) -> 'AbstractFormat':
        """
        Return a new format with precision adjusted by delta.
        
        Args:
            delta: Amount to add to precision (can be negative).
        Returns:
            New AbstractFormat with adjusted precision.
        """
        if self.prec is None:
            raise ValueError("Cannot adjust precision when prec is None")
        new_prec = max(1, self.prec + delta)
        return AbstractFormat(new_prec, self.exp, self.pos_bound, neg_bound=self.neg_bound)

    def with_exp_offset(self, delta: int) -> 'AbstractFormat':
        """
        Return a new format with exponent adjusted by delta.
        
        Args:
            delta: Amount to add to exponent (can be negative).
        Returns:
            New AbstractFormat with adjusted exponent.
        """
        if self.exp is None:
            raise ValueError("Cannot adjust exponent when exp is None")
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
        
        new_pos_bound = None if self.pos_bound is None else self.pos_bound * factor
        new_neg_bound = None if self.neg_bound is None else self.neg_bound * factor
        return AbstractFormat(self.prec, self.exp, new_pos_bound, neg_bound=new_neg_bound)
