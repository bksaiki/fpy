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
        self.prec = prec
        self.exp = exp
        self.pos_bound = pos_bound
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
        # maximum (effective) precision
        prec = self._effective_prec()
        if prec is None:
            prec = float('inf')

        # minimum unnormalized exponent
        if self.exp is None:
            exp = float('-inf')
        else:
            exp = self.exp

        # maximum representable value
        if self.bound is None:
            bound: RealFloat | float = float('inf')
        else:
            bound = self.bound

        return prec, exp, bound

    def contains(self, other: 'AbstractFormat') -> bool:
        """Check if this format contains another format."""
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        prec1, exp1, bound1 = self._effective_params()
        prec2, exp2, bound2 = other._effective_params()
        return prec1 <= prec2 and exp1 >= exp2 and bound1 <= bound2
