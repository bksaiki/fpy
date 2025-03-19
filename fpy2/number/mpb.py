"""
This module defines floating-point numbers as implemented by MPFR
but with subnormalization and a maximum value, that is multi-precision
and bounded. Hence, "MP-B."
"""

from fractions import Fraction
from typing import Optional

from ..utils import default_repr, bitmask

from .context import SizedContext
from .float import Float
from .real import RealFloat
from .round import RoundingMode
from .utils import from_mpfr

@default_repr
class MPBContext(SizedContext):
    """
    Rounding context for multi-precision floating-point numbers with
    a minimum exponent (and subnormalization) and a maximum value.

    This context is parameterized by a fixed precision `pmax`,
    a minimum (normalized) exponent `emin`, a (positive) maximum value
    `maxval`, and a rounding mode `rm`. A separate negative maximum value
    may be specified as well, but by default it is set to the negative
    of `maxval`.

    Unlike `MPContext`, the `MPBContext` is inherits from `SizedContext`
    since the set of representable values may be encoded in
    a finite amount of space.
    """

    pmax: int
    """maximum precision"""

    emin: int
    """minimum (normalized exponent)"""

    pos_maxval: RealFloat
    """positive maximum value"""

    neg_maxval: RealFloat
    """negative maximum value"""

    rm: RoundingMode
    """rounding mode"""

    def __init__(
        self,
        pmax: int,
        emin: int,
        maxval: RealFloat, 
        rm: RoundingMode, *,
        neg_maxval: Optional[RealFloat] = None
    ):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise TypeError(f'Expected integer p < 1 for p={pmax}')
        if not isinstance(emin, int):
            raise TypeError(f'Expected \'int\' for emin={emin}, got {type(emin)}')
        if not isinstance(maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for maxval={maxval}, got {type(maxval)}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')

        if maxval.s:
            raise ValueError(f'Expected positive maxval={maxval}, got {maxval}')
        elif maxval.p > pmax:
            raise ValueError(f'Expected maxval={maxval} to be representable in pmax={pmax}')

        if neg_maxval is None:
            neg_maxval = RealFloat(s=True, x=maxval)
        elif not isinstance(neg_maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for neg_maxval={neg_maxval}, got {type(neg_maxval)}')
        elif not neg_maxval.s:
            raise ValueError(f'Expected negative neg_maxval={neg_maxval}, got {neg_maxval}')
        elif neg_maxval.p > pmax:
            raise ValueError(f'Expected neg_maxval={neg_maxval} to be representable in pmax={pmax}')

        self.pmax = pmax
        self.emin = emin
        self.pos_maxval = maxval
        self.neg_maxval = neg_maxval
        self.rm = rm


    def is_representable(self, x):
        raise NotImplementedError

    def is_canonical(self, x):
        raise NotImplementedError

    def normalize(self, x):
        raise NotImplementedError

    def round(self, x):
        raise NotImplementedError

    def to_ordinal(self, x, infval = False):
        raise NotImplementedError

    def from_ordinal(self, x, infval = False):
        raise NotImplementedError

    def minval(self, s = False):
        raise NotImplementedError

    def maxval(self, s = False):
        raise NotImplementedError

