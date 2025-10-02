"""
This module defines floating-point numbers as implemented by MPFR,
that is, multi-precision floating-point numbers. Hence, "MP."
"""

from fractions import Fraction

from ..utils import bitmask, default_repr, DefaultOr, DEFAULT

from .context import Context
from .number import RealFloat, Float
from .round import RoundingMode
from .gmp import mpfr_value

@default_repr
class MPFloatContext(Context):
    """
    Rounding context for multi-precision floating-point numbers.

    This context is parameterized by a fixed precision `pmax`
    and a rounding mode `rm`. It emulates floating-point numbers
    as implemented by MPFR.
    """

    pmax: int
    """maximum precision"""

    rm: RoundingMode
    """rounding mode"""

    num_randbits: int | None
    """number of random bits for stochastic rounding, if applicable"""

    def __init__(
        self,
        pmax: int,
        rm: RoundingMode = RoundingMode.RNE,
        num_randbits: int | None = 0
    ):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise TypeError(f'Expected integer p < 1 for p={pmax}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')
        if num_randbits is not None and not isinstance(num_randbits, int):
            raise TypeError(f'Expected \'int\' for num_randbits={num_randbits}, got {type(num_randbits)}')

        self.pmax = pmax
        self.rm = rm
        self.num_randbits = num_randbits

    def __eq__(self, other):
        return (
            isinstance(other, MPFloatContext)
            and self.pmax == other.pmax
            and self.rm == other.rm
            and self.num_randbits == other.num_randbits
        )

    def __hash__(self):
        return hash((self.__class__, self.pmax, self.rm, self.num_randbits))

    def with_params(
        self, *, 
        pmax: DefaultOr[int] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
        **kwargs
    ) -> 'MPFloatContext':
        if pmax is DEFAULT:
            pmax = self.pmax
        if rm is DEFAULT:
            rm = self.rm
        if num_randbits is DEFAULT:
            num_randbits = self.num_randbits
        if kwargs:
            raise TypeError(f'Unexpected keyword arguments: {kwargs}')
        return MPFloatContext(pmax, rm, num_randbits)

    def is_stochastic(self) -> bool:
        return self.num_randbits != 0

    def is_equiv(self, other):
        if not isinstance(other, Context):
            raise TypeError(f'Expected \'Context\', got \'{type(other)}\' for other={other}')
        return isinstance(other, MPFloatContext) and self.pmax == other.pmax

    def representable_under(self, x: RealFloat | Float) -> bool:
        match x:
            case Float():
                if x.ctx is not None and self.is_equiv(x.ctx):
                    # same context, so representable
                    return True
                if x.is_nar() or x.is_zero():
                    # special values or zeros are valid
                    return True
            case RealFloat():
                if x.is_zero():
                    # zeros are valid
                    return True
            case _:
                raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')

        # precision is possibly out of bounds
        # check if the value can be normalized with fewer digits
        p_over = x.p - self.pmax
        if p_over < 0:
            # precision is within bounds
            return True
        else:
            c_lost = x.c & bitmask(p_over) # bits that would be lost via normalization
            return c_lost == 0

    def canonical_under(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        # case split on class
        if x.is_nar():
            # NaN or Inf
            return True
        elif x.is_zero():
            # zero
            return x.exp == 0
        else:
            # non-zero value
            return x.p == self.pmax

    def normalize(self, x: Float) -> Float:
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        # case split by class
        if x.isnan:
            # NaN
            return Float(isnan=True, s=x.s, ctx=self)
        elif x.isinf:
            # Inf
            return Float(isinf=True, s=x.s, ctx=self)
        elif x.c == 0:
            # zero
            return Float(c=0, exp=0, s=x.s, ctx=self)
        else:
            # non-zero
            xr = x._real.normalize(self.pmax, None)
            return Float(x=x, exp=xr.exp, c=xr.c, ctx=self)

    def normal_under(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        return x.is_nonzero()

    def _round_at(self, x: RealFloat | Float, n: int | None, exact: bool) -> Float:
        """
        Like `self.round()` but for only `RealFloat` and `Float` inputs.

        Optionally specify `n` as the least absolute digit position.
        """
        # step 1. handle special values
        if isinstance(x, Float):
            if x.isnan:
                return Float(isnan=True, ctx=self)
            elif x.isinf:
                return Float(s=x.s, isinf=True, ctx=self)
            else:
                x = x._real

        # step 2. shortcut for exact zero values
        if x.is_zero():
            # exactly zero
            return Float(ctx=self)

        # step 3. round value based on rounding parameters
        xr = x.round(self.pmax, n, self.rm, self.num_randbits, exact=exact)

        # step 4. wrap the result in a Float
        return Float(x=xr, ctx=self)

    def round(self, x, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, None, exact)

    def round_at(self, x, n: int, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, n, exact)

    def round_params(self) -> tuple[int | None, int | None]:
        if self.num_randbits is None:
            return None, None
        else:
            pmax = self.pmax + self.num_randbits
            return pmax, None
