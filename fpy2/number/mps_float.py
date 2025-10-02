"""
This module defines floating-point numbers as implemented by MPFR
but with a subnormalization, that is multi-precision floating-point
numbers with subnormals. Hence, "MP-S."
"""

from fractions import Fraction

from ..utils import bitmask, default_repr, DefaultOr, DEFAULT

from .context import Context, OrdinalContext
from .number import RealFloat, Float
from .mp_float import MPFloatContext
from .round import RoundingMode
from .gmp import mpfr_value

@default_repr
class MPSFloatContext(OrdinalContext):
    """
    Rounding context for multi-precision floating-point numbers with
    a minimum exponent (and subnormalization).

    This context is parameterized by a fixed precision `pmax`,
    a minimum (normalized) exponent `emin`, and a rounding mode `rm`.
    It emulates floating-point numbers as implemented by MPFR
    with subnormalization.

    Unlike `MPFloatContext`, the `MPSFloatContext` inherits from `OrdinalContext`
    since each representable value can be mapped to the ordinals.
    """

    pmax: int
    """maximum precision"""

    emin: int
    """minimum (normalized exponent)"""

    rm: RoundingMode
    """rounding mode"""

    num_randbits: int | None
    """number of random bits for stochastic rounding, if applicable"""

    _mp_ctx: MPFloatContext
    """this context without subnormalization"""

    _pos_minval: Float
    """minimum positive value"""

    _neg_minval: Float
    """minimum negative value"""

    def __init__(
        self,
        pmax: int,
        emin: int,
        rm: RoundingMode = RoundingMode.RNE,
        num_randbits: int | None = 0
    ):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise TypeError(f'Expected integer p < 1 for p={pmax}')
        if not isinstance(emin, int):
            raise TypeError(f'Expected \'int\' for emin={emin}, got {type(emin)}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')
        if num_randbits is not None and not isinstance(num_randbits, int):
            raise TypeError(f'Expected \'int\' for num_randbits={num_randbits}, got {type(num_randbits)}')

        self.pmax = pmax
        self.emin = emin
        self.rm = rm
        self.num_randbits = num_randbits
        self._mp_ctx = MPFloatContext(pmax, rm, num_randbits)
        self._pos_minval = Float(s=False, c=1, exp=self.expmin, ctx=self)
        self._neg_minval = Float(s=True, c=1, exp=self.expmin, ctx=self)

    def __eq__(self, other):
        return (
            isinstance(other, MPSFloatContext)
            and self.pmax == other.pmax
            and self.emin == other.emin
            and self.rm == other.rm
            and self.num_randbits == other.num_randbits
        )

    def __hash__(self):
        return hash((self.pmax, self.emin, self.rm, self.num_randbits))

    @property
    def expmin(self):
        """Minimum unnormalized exponent."""
        return self.emin - self.pmax + 1

    @property
    def nmin(self):
        """
        First unrepresentable digit for every value in the representation.
        """
        return self.expmin - 1

    def with_params(
        self, *,
        pmax: DefaultOr[int] = DEFAULT,
        emin: DefaultOr[int] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
        **kwargs
    ) -> 'MPSFloatContext':
        if pmax is DEFAULT:
            pmax = self.pmax
        if emin is DEFAULT:
            emin = self.emin
        if rm is DEFAULT:
            rm = self.rm
        if num_randbits is DEFAULT:
            num_randbits = self.num_randbits
        if kwargs:
            raise TypeError(f'Unexpected keyword arguments: {kwargs}')
        return MPSFloatContext(pmax, emin, rm, num_randbits)

    def is_stochastic(self) -> bool:
        return self.num_randbits != 0

    def is_equiv(self, other):
        if not isinstance(other, Context):
            raise TypeError(f'Expected \'Context\', got \'{type(other)}\' for other={other}')
        return (
            isinstance(other, MPSFloatContext)
            and self.pmax == other.pmax
            and self.emin == other.emin
        )

    def representable_under(self, x: RealFloat | Float) -> bool:
        match x:
            case Float():
                if x.ctx is not None and self.is_equiv(x.ctx):
                    # same context, so representable
                    return True
            case RealFloat():
                pass
            case _:
                raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')

        if not self._mp_ctx.representable_under(x):
            # not representable even without subnormalization
            return False
        elif not x.is_nonzero():
            # NaN, Inf, 0
            return True
        elif x.s:
            # tight check (negative values)
            return x <= self._neg_minval
        else:
            # tight check (non-negative values)
            return self._pos_minval <= x

    def canonical_under(self, x):
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        # case split by class
        if x.is_nar():
            # NaN or Inf
            return True
        elif x.c == 0:
            # zero
            return x.exp == self.expmin
        elif x.e < self.emin:
            # subnormal
            return x.exp == self.expmin
        else:
            # normal
            return x.p == self.pmax

    def _normalize(self, x: Float) -> Float:
        # case split by class
        if x.isnan:
            # NaN
            return Float(isnan=True, s=x.s, ctx=self)
        elif x.isinf:
            # Inf
            return Float(isinf=True, s=x.s, ctx=self)
        elif x.c == 0:
            # zero
            return Float(c=0, exp=self.expmin, s=x.s, ctx=self)
        else:
            # non-zero
            xr = x._real.normalize(self.pmax, self.nmin)
            return Float(x=x, exp=xr.exp, c=xr.c, ctx=self)

    def normalize(self, x):
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._normalize(x)

    def normal_under(self, x: Float):
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return x.is_nonzero() and x.e >= self.emin

    def round_params(self):
        if self.num_randbits is None:
            return None, None
        else:
            pmax = self.pmax + self.num_randbits
            nmin = self.nmin - self.num_randbits
            return pmax, nmin

    def _round_at(self, x: RealFloat | Float, n: int | None, exact: bool) -> Float:
        """
        Like `self.round()` but for only `RealFloat` and `Float` inputs.

        Optionally specify `n` as the least absolute digit position.
        Only overrides rounding behavior when `n > self.nmin`.
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

        # step 3. select rounding parameter `n`
        if n is None or n < self.nmin:
            # no rounding parameter
            n = self.nmin

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

    def to_ordinal(self, x: Float, infval = False) -> int:
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        if infval:
            raise ValueError('infval=True is invalid for contexts without a maximum value')

        # case split by class
        if x.is_nar():
            # NaN or Inf
            raise TypeError(f'Expected a finite value for x={x}')
        elif x.is_zero():
            # zero
            return 0
        elif x.e <= self.emin:
            # subnormal: sgn(x) * [ 0 | m ]
            # need to ensure that exp=self.expmin
            offset = x.exp - self.expmin
            if offset > 0:
                # need to increase precision of `c`
                c = x.c << offset
            elif offset < 0:
                # need to decrease precision of `c`
                c = x.c >> -offset
            else:
                # no change
                c = x.c

            # ordinal components
            eord = 0
            mord = c
        else:
            # normal: sgn(x) * [ eord | m ]
            # normalize so that p=self.pmax
            offset = x.p - self.pmax
            if offset > 0:
                # too much precision
                c = x.c >> offset
            elif offset < 0:
                # too little precision
                c = x.c << -offset
            else:
                # no change
                c = x.c

            # ordinal components
            eord = x.e - self.emin + 1
            mord = c & bitmask(self.pmax - 1)

        uord = (eord << (self.pmax - 1)) + mord
        return (-1 if x.s else 1) * uord

    def from_ordinal(self, x: int, infval = False):
        if not isinstance(x, int):
            raise TypeError(f'Expected an \'int\', got \'{type(x)}\' for x={x}')
        if infval:
            raise ValueError('infval=True is invalid for contexts without a maximum value')

        s = x < 0
        uord = abs(x)

        if x == 0:
            # zero
            return Float(ctx=self)
        else:
            # finite values
            eord, mord = divmod(uord, 1 << (self.pmax - 1))
            if eord == 0:
                # subnormal
                return Float(s=s, c=mord, exp=self.expmin, ctx=self)
            else:
                # normal
                c = (1 << (self.pmax - 1)) | mord
                exp = self.expmin + (eord - 1)
                return Float(s=s, c=c, exp=exp, ctx=self)


    def zero(self, s: bool = False) -> Float:
        """Returns a signed 0 under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(s=s, c=0, exp=self.expmin, ctx=self)

    def minval(self, s: bool = False) -> Float:
        """Returns the smallest non-zero value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(x=self._neg_minval) if s else Float(x=self._pos_minval)

    def min_subnormal(self, s: bool = False) -> Float:
        """Returns the smallest subnormal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return self.minval(s)

    def max_subnormal(self, s: bool = False) -> Float:
        """Returns the largest subnormal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        c = bitmask(self.pmax - 1)
        exp = self.expmin
        return Float(s=s, c=c, exp=exp, ctx=self)

    def min_normal(self, s: bool = False) -> Float:
        """Returns the smallest normal value with sign `s` under this context."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        c = 1 << (self.pmax - 1)
        exp = self.emin
        return Float(s=s, c=c, exp=exp, ctx=self)

