"""
This module defines fixed-point numbers with a fixed least-significant digit
but no most-significand digit, that is, a fixed-point number with arbitrary precision.
Hence, "MP-F".
"""

from fractions import Fraction
from typing import Optional

from ..utils import default_repr

from .context import OrdinalContext
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

    enable_nan: bool
    """is NaN representable?"""

    enable_inf: bool
    """is infinity representable?"""

    nan_value: Optional[Float]
    """
    if NaN is not enabled, what value should NaN round to?
    if not set, then `round()` will raise a `ValueError`.
    """

    inf_value: Optional[Float]
    """
    if Inf is not enabled, what value should Inf round to?
    if not set, then `round()` will raise a `ValueError`.
    """

    def __init__(
        self,
        nmin: int,
        rm: RoundingMode,
        *,
        enable_nan: bool = False,
        enable_inf: bool = False,
        nan_value: Optional[Float] = None,
        inf_value: Optional[Float] = None
    ):
        if not isinstance(nmin, int):
            raise TypeError(f'Expected \'int\' for nmin={nmin}, got {type(nmin)}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')
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
        self.enable_nan = enable_nan
        self.enable_inf = enable_inf
        self.nan_value = nan_value
        self.inf_value = inf_value

    @property
    def expmin(self) -> int:
        """
        The minimum exponent for this context.
        This is equal to `nmin + 1`.
        """
        return self.nmin + 1

    def with_rm(self, rm: RoundingMode):
        return MPFixedContext(
            self.nmin,
            rm,
            enable_nan=self.enable_nan,
            enable_inf=self.enable_inf,
            nan_value=self.nan_value,
            inf_value=self.inf_value
        )

    def is_representable(self, x: RealFloat | Float) -> bool:
        if not isinstance(x, RealFloat | Float):
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

    def is_canonical(self, x: Float):
        if not isinstance(x, Float) and self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return x.exp == self.expmin

    def normalize(self, x: Float):
        if not isinstance(x, Float) and self.is_representable(x):
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

    def is_normal(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        return x.is_nonzero()

    def round_params(self):
        return None, self.nmin

    def _round_float_at(self, x: RealFloat | Float, n: Optional[int]) -> Float:
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
        xr = xr.round(min_n=n, rm=self.rm)
        return Float(x=xr, ctx=self)

    def _round_at(self, x, n: Optional[int]) -> Float:
        match x:
            case Float() | RealFloat():
                xr = x
            case int():
                xr = RealFloat(c=x)
            case float() | str():
                xr = mpfr_value(x, n=self.nmin)
            case Fraction():
                if x.denominator == 1:
                    xr = RealFloat(c=int(x))
                else:
                    xr = mpfr_value(x, n=self.nmin)
            case _:
                raise TypeError(f'not valid argument x={x}')

        return self._round_float_at(xr, n)

    def round(self, x):
        return self._round_at(x, None)

    def round_at(self, x, n: int):
        return self._round_at(x, n)

    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        if not self.is_representable(x):
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

