"""
Abstract number system.
"""

from typing import TypeAlias

from ...number import (
    MPFixedContext, MPBFixedContext,
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
    - `bound`: maximum representable number
    """

    prec: int | None
    exp: int | None
    bound: RealFloat | None

    def __init__(self, prec: int | None, exp: int | None, bound: RealFloat | None):
        if prec is None and exp is None:
            raise ValueError("At least one of `prec` or `exp` must be specified.")
        if bound is not None and bound < 0:
            raise ValueError(f"`bound={bound}` must be non-negative.")
        self.prec = prec
        self.exp = exp
        self.bound = bound

    def __hash__(self):
        return hash((self.prec, self.exp, self.bound))

    def __eq__(self, other):
        return (
            isinstance(other, AbstractFormat)
            and self.prec == other.prec
            and self.exp == other.exp
            and self.bound == other.bound
        )

    @staticmethod
    def from_context(ctx: SupportedContext) -> 'AbstractFormat':
        match ctx:
            case MPFloatContext():
                return AbstractFormat(ctx.pmax, None, None)
            case MPSFloatContext():
                return AbstractFormat(ctx.pmax, ctx.expmin, None)
            case MPBFloatContext():
                maxval = max(ctx.pos_maxval, abs(ctx.neg_maxval))
                return AbstractFormat(ctx.pmax, ctx.expmin, maxval)
            case EFloatContext():
                maxval = max(ctx.maxval(), abs(ctx.maxval(True)))
                return AbstractFormat(ctx.pmax, ctx.expmin, maxval)
            case MPFixedContext():
                return AbstractFormat(None, ctx.expmin, None)
            case MPBFixedContext():
                maxval = max(ctx.pos_maxval, abs(ctx.neg_maxval))
                return AbstractFormat(None, ctx.expmin, maxval)
            case _:
                raise TypeError(f'Unsupported context type: {type(ctx)}')

    def _effective_params(self):
        # maximum precision
        if self.prec is None:
            prec = float('inf')
        else:
            prec = self.prec

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

    def effective_prec(self):
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


    def contains(self, other: 'AbstractFormat') -> bool:
        """Check if this format contains another format."""
        if not isinstance(other, AbstractFormat):
            raise TypeError(f'Expected \'AbstractFormat\', got {other}')
        prec1, exp1, bound1 = self._effective_params()
        prec2, exp2, bound2 = other._effective_params()
        return prec1 <= prec2 and exp1 >= exp2 and bound1 <= bound2
