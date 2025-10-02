"""
This module defines fixed-point numbers with a fixed least-significant digit
but no most-significand digit, that is, a fixed-point number with arbitrary precision.
Hence, "MP-F".
"""

from fractions import Fraction

from ..utils import default_repr, DEFAULT, DefaultOr

from .context import Context, OrdinalContext
from .number import Float
from .real import RealFloat
from .round import RoundingMode
from .gmp import mpfr_value

@default_repr
class MPFixedContext(OrdinalContext):
    """
    Rounding context for mulit-precision fixed-point numbers.

    This context is parameterized by the most significant digit
    that is not representable `nmin` and a rounding mode `rm`.
    It emulates fixed-point numbers with arbitrary precision.

    Optionally, specify the following keywords:

    - `enable_nan`: if `True`, then NaN is representable [default: `False`]
    - `enable_inf`: if `True`, then infinity is representable [default: `False`]
    - `nan_value`: if NaN is not enabled, what value should NaN round to? [default: `None`];
      if not set, then `round()` will raise a `ValueError` on NaN.
    - `inf_value`: if Inf is not enabled, what value should Inf round to? [default: `None`];
      if not set, then `round()` will raise a `ValueError` on infinity.

    `MPFixedContext` inherits from `OrdinalContext` since each representable
    value can be mapped to the ordinals.
    """

    nmin: int
    """the first unrepresentable digit"""

    rm: RoundingMode
    """rounding mode"""

    num_randbits: int | None
    """number of random bits for stochastic rounding, if applicable"""

    enable_nan: bool
    """is NaN representable?"""

    enable_inf: bool
    """is infinity representable?"""

    nan_value: Float | None
    """
    if NaN is not enabled, what value should NaN round to?
    if not set, then `round()` will raise a `ValueError`.
    """

    inf_value: Float | None
    """
    if Inf is not enabled, what value should Inf round to?
    if not set, then `round()` will raise a `ValueError`.
    """

    def __init__(
        self,
        nmin: int,
        rm: RoundingMode = RoundingMode.RNE,
        num_randbits: int | None = 0,
        *,
        enable_nan: bool = False,
        enable_inf: bool = False,
        nan_value: Float | None = None,
        inf_value: Float | None = None
    ):
        if not isinstance(nmin, int):
            raise TypeError(f'Expected \'int\' for nmin={nmin}, got {type(nmin)}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')
        if num_randbits is not None and not isinstance(num_randbits, int):
            raise TypeError(f'Expected \'int\' or None for num_randbits={num_randbits}, got {type(num_randbits)}')
        if not isinstance(enable_nan, bool):
            raise TypeError(f'Expected \'bool\' for enable_nan={enable_nan}, got {type(enable_nan)}')
        if not isinstance(enable_inf, bool):
            raise TypeError(f'Expected \'bool\' for enable_inf={enable_inf}, got {type(enable_inf)}')

        if nan_value is not None:
            if not isinstance(nan_value, Float):
                raise TypeError(f'Expected \'RealFloat\' for nan_value={nan_value}, got {type(nan_value)}')
            if not enable_nan:
                # this field matters
                if nan_value.isinf:
                    if not enable_inf:
                        raise ValueError('Rounding NaN to infinity, but infinity not enabled')
                elif nan_value.is_finite():
                    if not nan_value.as_real().is_more_significant(nmin):
                        raise ValueError('Rounding NaN to unrepresentable value')

        if inf_value is not None:
            if not isinstance(inf_value, Float):
                raise TypeError(f'Expected \'RealFloat\' for inf_value={inf_value}, got {type(inf_value)}')
            if not enable_inf:
                # this field matters
                if inf_value.isnan:
                    if not enable_nan:
                        raise ValueError('Rounding Inf to NaN, but NaN not enabled')
                elif inf_value.is_finite():
                    if not inf_value.as_real().is_more_significant(nmin):
                        raise ValueError('Rounding Inf to unrepresentable value')

        self.nmin = nmin
        self.rm = rm
        self.num_randbits = num_randbits
        self.enable_nan = enable_nan
        self.enable_inf = enable_inf
        self.nan_value = nan_value
        self.inf_value = inf_value

    def __eq__(self, other):
        return (
            isinstance(other, MPFixedContext)
            and self.nmin == other.nmin
            and self.rm == other.rm
            and self.num_randbits == other.num_randbits
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
            and self.nan_value == other.nan_value
            and self.inf_value == other.inf_value
        )

    def __hash__(self):
        return hash((
            self.nmin,
            self.rm,
            self.num_randbits,
            self.enable_nan,
            self.enable_inf,
            self.nan_value,
            self.inf_value
        ))


    @property
    def expmin(self) -> int:
        """
        The minimum exponent for this context.
        This is equal to `nmin + 1`.
        """
        return self.nmin + 1

    def with_params(
        self, *,
        nmin: DefaultOr[int] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        enable_nan: DefaultOr[bool] = DEFAULT,
        enable_inf: DefaultOr[bool] = DEFAULT,
        nan_value: DefaultOr[Float | None] = DEFAULT,
        inf_value: DefaultOr[Float | None] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
        **kwargs
    ) -> 'MPFixedContext':
        if nmin is DEFAULT:
            nmin = self.nmin
        if rm is DEFAULT:
            rm = self.rm
        if enable_nan is DEFAULT:
            enable_nan = self.enable_nan
        if enable_inf is DEFAULT:
            enable_inf = self.enable_inf
        if nan_value is DEFAULT:
            nan_value = self.nan_value
        if inf_value is DEFAULT:
            inf_value = self.inf_value
        if num_randbits is DEFAULT:
            num_randbits = self.num_randbits
        if kwargs:
            raise TypeError(f'Unexpected parameters {kwargs} for MPFixedContext')
        return MPFixedContext(
            nmin,
            rm,
            num_randbits,
            enable_nan=enable_nan,
            enable_inf=enable_inf,
            nan_value=nan_value,
            inf_value=inf_value
        )

    def is_stochastic(self) -> bool:
        return self.num_randbits != 0

    def is_equiv(self, other):
        if not isinstance(other, Context):
            raise TypeError(f'Expected \'Context\', got \'{type(other)}\' for other={other}')
        return (
            isinstance(other, MPFixedContext)
            and self.nmin == other.nmin
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
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

        match x:
            case Float():
                if x.isnan:
                    return self.enable_nan
                elif x.isinf:
                    return self.enable_inf
                else:
                    xr = x.as_real()
            case RealFloat():
                xr = x
            case _:
                raise RuntimeError(f'unreachable {x}')

        return xr.is_more_significant(self.nmin)

    def canonical_under(self, x: Float):
        if not isinstance(x, Float) and self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return x.exp == self.expmin

    def normalize(self, x: Float):
        if not isinstance(x, Float) and self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        offset = x.exp - self.expmin
        if offset > 0:
            # shift the significand to the right
            c = x.c >> offset
            exp = x.exp - offset
        elif offset < 0:
            # shift the significand to the left
            c = x.c << -offset
            exp = x.exp - offset
        else:
            c = x.c
            exp = x.exp

        return Float(exp=exp, c=c, x=x, ctx=self)

    def normal_under(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        return x.is_nonzero()

    def round_params(self):
        if self.num_randbits is None:
            return None, None
        else:
            nmin = self.nmin - self.num_randbits
            return None, nmin

    def _round_at(self, x: RealFloat | Float, n: int | None, exact: bool) -> Float:
        """
        Like `self.round_at()` but only for `RealFloat` or `Float` instances.

        Optionally, specify `n` to override the least absolute digit position.
        If `n < self.nmin`, it will be set to `self.nmin`.
        """
        if n is None:
            n = self.nmin
        else:
            n = max(n, self.nmin)

        # step 1. handle special values
        match x:
            case Float():
                if x.isnan:
                    if self.enable_nan:
                        return Float(isnan=True, ctx=self)
                    elif self.nan_value is None:
                        raise ValueError('Cannot round NaN under this context')
                    else:
                        return Float(x=self.nan_value, ctx=self)
                elif x.isinf:
                    if self.enable_inf:
                        return Float(isinf=True, ctx=self)
                    elif self.inf_value is None:
                        raise ValueError('Cannot round infinity under this context')
                    else:
                        return Float(x=self.inf_value, ctx=self)
                else:
                    xr = x._real
            case RealFloat():
                xr = x
            case _:
                raise RuntimeError(f'unreachable {x}')

        # step 2. shortcut for exact zero values
        if xr.is_zero():
            # exactly zero
            return Float(ctx=self)

        # step 3. round value based on rounding parameters
        xr = xr.round(min_n=n, rm=self.rm, num_randbits=self.num_randbits, exact=exact)

        # step 4. wrap the value in a Float
        return Float(x=xr, ctx=self)

    def round(self, x, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, None, exact)

    def round_at(self, x, n: int, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, n, exact)

    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_under(x):
            raise ValueError(f'Expected representable \'Float\', got x={x}')
        if infval:
            raise ValueError('infvalue=True is invalid for contexts without maximum value')

        # case split by class
        if x.is_nar():
            # NaN or Inf
            raise TypeError(f'Expected a finite value for={x}')
        elif x.is_zero():
            # zero -> 0
            return 0
        else:
            # finite, non-zero
            offset = x.exp - self.expmin
            if offset > 0:
                # need to increase precision of `c`
                c = x.c << offset
            elif offset < 0:
                # need to decrease precision of `c`
                c = x.c >> -offset
            else:
                c = x.c

            # apply sign
            if x.s:
                c *= -1

            return c

    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected an \'int\', got \'{type(x)}\' for x={x}')
        if infval:
            raise ValueError('infval=True is invalid for contexts without a maximum value')

        s = x < 0
        uord = abs(x)

        if x == 0:
            # 0 -> zero
            return Float(ctx=self)
        else:
            # finite, non-zero
            return Float(s=s, c=uord, exp=self.expmin, ctx=self)

    def minval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(s=s, c=1, exp=self.expmin, ctx=self)

