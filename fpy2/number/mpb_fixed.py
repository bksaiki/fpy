"""
This module defines fixed-pont numbers with a maximum value,
that is, multiprecision and bounded. Hence "MP-B".
"""

from fractions import Fraction
from typing import Optional

from ..utils import default_repr

from .context import SizedContext
from .number import RealFloat, Float
from .mp_fixed import MPFixedContext
from .round import RoundingMode, RoundingDirection
from .gmp import mpfr_value


@default_repr
class MPBFixedContext(SizedContext):
    """
    Rounding context for multi-precision, fixed-point numbers with
    a maximum value.

    This context is parameterized by the most significant digit that
    is not representable `nmin`, a (positive) maximum value `maxval`,
    and a rounding mode `rm`. A separate negative maximum value may be
    specified as well, but by default it is the negative of `maxval`.

    Optionally, specify the following keywords:
    - `enable_nan`: if `True`, then NaN is representable [default: `False`]
    - `enable_inf`: if `True`, then infinity is representable [default: `False`]
    - `nan_value`: if NaN is not enabled, what value should NaN round to? [default: `None`];
    if not set, then `round()` will raise a `ValueError` on NaN.
    - `inf_value`: if Inf is not enabled, what value should Inf round to? [default: `None`];
    if not set, then `round()` will raise a `ValueError` on infinity.

    Unlike `MPFixedContext`, the `MPBFixedContext` inherits from
    `SizedContext`, since the set of representable numbers is finite.
    """

    nmin: int
    """the first unrepresentable digit"""

    pos_maxval: RealFloat
    """positive maximum value"""

    neg_maxval: RealFloat
    """negative maximum value"""

    rm: RoundingMode
    """rounding mode"""

    enbale_nan: bool
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

    _mp_ctx: MPFixedContext
    """this context without maximum values"""

    _pos_maxval_ord: int
    """precomputed ordinal of `self.pos_maxval`"""

    _neg_maxval_ord: int
    """precomputed ordinal of `self.neg_maxval`"""


    def __init__(
        self,
        nmin: int,
        maxval: RealFloat,
        rm: RoundingMode,
        *,
        neg_maxval: Optional[RealFloat] = None,
        enable_nan: bool = False,
        enable_inf: bool = False,
        nan_value: Optional[Float] = None,
        inf_value: Optional[Float] = None
    ):
        if not isinstance(nmin, int):
            raise TypeError(f'Expected \'int\' for nmin={nmin}, got {type(nmin)}')
        if not isinstance(maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for maxval={maxval}, got {type(maxval)}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')
        if not isinstance(enable_nan, bool):
            raise TypeError(f'Expected \'bool\' for enable_nan={enable_nan}, got {type(enable_nan)}')
        if not isinstance(enable_inf, bool):
            raise TypeError(f'Expected \'bool\' for enable_inf={enable_inf}, got {type(enable_inf)}')

        if maxval.s:
            raise ValueError(f'Expected positive maximum value, got {maxval}')
        if not maxval.is_more_significant(nmin):
            raise ValueError(f'maxval={maxval} is an unrepresentable value')

        if neg_maxval is None:
            neg_maxval = RealFloat(s=True, x=maxval)
        elif not isinstance(neg_maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for neg_maxval={neg_maxval}, got {type(neg_maxval)}')
        elif not neg_maxval.s:
            raise ValueError(f'Expected negative maximum value, got {neg_maxval}')
        elif not neg_maxval.is_more_significant(nmin):
            raise ValueError(f'neg_maxval={neg_maxval} is an unrepresentable value')

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
        self.pos_maxval = maxval
        self.neg_maxval = neg_maxval
        self.rm = rm
        self.enable_nan = enable_nan
        self.enable_inf = enable_inf
        self.nan_value = nan_value
        self.inf_value = inf_value

        self._mp_ctx = MPFixedContext(nmin, rm, enable_nan=enable_nan, enable_inf=enable_inf)
        pos_maxval_mp = Float(x=self.pos_maxval, ctx=self._mp_ctx)
        neg_maxval_mp = Float(x=self.neg_maxval, ctx=self._mp_ctx)
        self._pos_maxval_ord = self._mp_ctx.to_ordinal(pos_maxval_mp)
        self._neg_maxval_ord = self._mp_ctx.to_ordinal(neg_maxval_mp)


    def with_rm(self, rm: RoundingMode):
        return MPBFixedContext(
            nmin=self.nmin,
            maxval=self.pos_maxval,
            rm=rm,
            neg_maxval=self.neg_maxval,
            enable_nan=self.enable_nan,
            enable_inf=self.enable_inf,
            nan_value=self.nan_value,
            inf_value=self.inf_value
        )

    def is_representable(self, x: RealFloat | Float) -> bool:
        raise NotImplementedError

    def is_canonical(self, x: Float):
        raise NotImplementedError

    def normalize(self, x: Float) -> Float:
        raise NotImplementedError

    def is_normal(self, x: Float) -> bool:
        raise NotImplementedError

    def round_params(self):
        raise NotImplementedError

    def round(self, x):
        raise NotImplementedError

    def round_at(self, x, n: int):
        raise NotImplementedError

    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected \'Float\' for x={x}, got {type(x)}')

        # case split by class
        if x.isnan:
            # NaN
            raise ValueError('Cannot convert NaN to ordinal')
        elif x.isinf:
            # INf
            if not infval:
                raise ValueError(f'Expected a finite value for x={x} when infval=False')
            elif x.s:
                # -Inf mapped to 1 less than -MAX
                return self._neg_maxval_ord - 1
            else:
                # +Inf mapped to 1 more than +MAX
                return self._pos_maxval_ord + 1
        else:
            # finite, real
            return self._mp_ctx.to_ordinal(x)

    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected \'int\' for x={x}, got {type(x)}')

        if infval:
            pos_maxord = self._pos_maxval_ord + 1
            neg_maxord = self._neg_maxval_ord - 1
        else:
            pos_maxord = self._pos_maxval_ord
            neg_maxord = self._neg_maxval_ord

        if x > pos_maxord:
            raise ValueError(f'Expected an \'int\' between {neg_maxord} and {pos_maxord}, got x={x}')
        elif x < neg_maxord:
            raise ValueError(f'Expected an \'int\' between {neg_maxord} and {pos_maxord}, got x={x}')
        elif x > self._pos_maxval_ord:
            # +Inf
            return Float(isinf=True, ctx=self)
        elif x < self._neg_maxval_ord:
            # -Inf
            return Float(isinf=True, s=True, ctx=self)
        else:
            # finite, real
            v = self._mp_ctx.from_ordinal(x)
            v.ctx = self
            return v

    def minval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        x = self._mp_ctx.minval(s=s)
        x.ctx = self
        return x

    def maxval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        if s:
            return Float(x=self.pos_maxval, ctx=self)
        else:
            return Float(x=self.neg_maxval, ctx=self)
