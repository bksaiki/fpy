"""
This module defines fixed-pont numbers with a maximum value,
that is, multiprecision and bounded. Hence "MP-B".
"""

from fractions import Fraction

from ..utils import default_repr, DefaultOr, DEFAULT

from .context import Context, SizedContext
from .mp_fixed import MPFixedContext
from .number import RealFloat, Float
from .round import RoundingMode, OverflowMode
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

    overflow: OverflowMode
    """overflow behavior"""

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
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode = OverflowMode.WRAP,
        num_randbits: int | None = 0,
        *,
        neg_maxval: RealFloat | None = None,
        enable_nan: bool = False,
        enable_inf: bool = False,
        nan_value: Float | None = None,
        inf_value: Float | None = None
    ):
        if not isinstance(nmin, int):
            raise TypeError(f'Expected \'int\' for nmin={nmin}, got {type(nmin)}')
        if not isinstance(maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for maxval={maxval}, got {type(maxval)}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')
        if not isinstance(overflow, OverflowMode):
            raise TypeError(f'Expected \'FixedOverflowKind\' for overflow={overflow}, got {type(overflow)}')
        if num_randbits is not None and not isinstance(num_randbits, int):
            raise TypeError(f'Expected \'int\' for num_randbits={num_randbits}, got {type(num_randbits)}')
        if not isinstance(enable_nan, bool):
            raise TypeError(f'Expected \'bool\' for enable_nan={enable_nan}, got {type(enable_nan)}')
        if not isinstance(enable_inf, bool):
            raise TypeError(f'Expected \'bool\' for enable_inf={enable_inf}, got {type(enable_inf)}')

        if maxval.is_negative():
            raise ValueError(f'Expected non-negative maximum value, got {maxval}')
        if not maxval.is_more_significant(nmin):
            raise ValueError(f'maxval={maxval} is an unrepresentable value')

        if neg_maxval is None:
            neg_maxval = RealFloat(s=True, x=maxval)
        elif not isinstance(neg_maxval, RealFloat):
            raise TypeError(f'Expected \'RealFloat\' for neg_maxval={neg_maxval}, got {type(neg_maxval)}')
        elif neg_maxval.is_positive():
            raise ValueError(f'Expected a non-positive maximum value, got {neg_maxval}')
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
        self.overflow = overflow
        self.num_randbits = num_randbits
        self.enable_nan = enable_nan
        self.enable_inf = enable_inf
        self.nan_value = nan_value
        self.inf_value = inf_value

        self._mp_ctx = MPFixedContext(nmin, rm, num_randbits, enable_nan=enable_nan, enable_inf=enable_inf)
        pos_maxval_mp = Float(x=self.pos_maxval, ctx=self._mp_ctx)
        neg_maxval_mp = Float(x=self.neg_maxval, ctx=self._mp_ctx)
        self._pos_maxval_ord = self._mp_ctx.to_ordinal(pos_maxval_mp)
        self._neg_maxval_ord = self._mp_ctx.to_ordinal(neg_maxval_mp)

    def __eq__(self, other):
        return (
            isinstance(other, MPBFixedContext)
            and self.nmin == other.nmin
            and self.pos_maxval == other.pos_maxval
            and self.neg_maxval == other.neg_maxval
            and self.rm == other.rm
            and self.overflow == other.overflow
            and self.num_randbits == other.num_randbits
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
            and self.nan_value == other.nan_value
            and self.inf_value == other.inf_value
        )
    
    def __hash__(self):
        return hash((
            self.nmin,
            self.pos_maxval,
            self.neg_maxval,
            self.rm,
            self.overflow,
            self.num_randbits,
            self.enable_nan,
            self.enable_inf,
            self.nan_value,
            self.inf_value
        ))

    def with_params(
        self, *,
        nmin: DefaultOr[int] = DEFAULT,
        maxval: DefaultOr[RealFloat] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        overflow: DefaultOr[OverflowMode] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
        neg_maxval: DefaultOr[RealFloat] = DEFAULT,
        enable_nan: DefaultOr[bool] = DEFAULT,
        enable_inf: DefaultOr[bool] = DEFAULT,
        nan_value: DefaultOr[Float | None] = DEFAULT,
        inf_value: DefaultOr[Float | None] = DEFAULT,
        **kwargs
    ) -> 'MPBFixedContext':
        if nmin is DEFAULT:
            nmin = self.nmin
        if maxval is DEFAULT:
            maxval = self.pos_maxval
        if rm is DEFAULT:
            rm = self.rm
        if overflow is DEFAULT:
            overflow = self.overflow
        if num_randbits is DEFAULT:
            num_randbits = self.num_randbits
        if neg_maxval is DEFAULT:
            neg_maxval = self.neg_maxval
        if neg_maxval is DEFAULT:
            neg_maxval = self.neg_maxval
        if enable_nan is DEFAULT:
            enable_nan = self.enable_nan
        if enable_inf is DEFAULT:
            enable_inf = self.enable_inf
        if nan_value is DEFAULT:
            nan_value = self.nan_value
        if inf_value is DEFAULT:
            inf_value = self.inf_value
        if kwargs:
            raise TypeError(f'Unexpected keyword arguments: {kwargs}')
        return MPBFixedContext(
            nmin=nmin,
            maxval=maxval,
            rm=rm,
            num_randbits=num_randbits,
            overflow=overflow,
            neg_maxval=neg_maxval,
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
            isinstance(other, MPBFixedContext)
            and self.nmin == other.nmin
            and self.pos_maxval == other.pos_maxval
            and self.neg_maxval == other.neg_maxval
            and self.enable_inf == other.enable_inf
            and self.enable_nan == other.enable_nan
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
            # not representable even without a maximum value
            return False
        elif not x.is_nonzero():
            # NaN, Inf, 0
            return True
        elif x.s:
            # check bounded (negative values)
            return self.neg_maxval <= x
        else:
            # check bounded (positive values)
            return x <= self.pos_maxval

    def canonical_under(self, x: Float):
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mp_ctx.canonical_under(x)

    def normalize(self, x: Float) -> Float:
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return Float(x=self._mp_ctx.normalize(x), ctx=self)

    def normal_under(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.representable_under(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mp_ctx.normal_under(x)

    def round_params(self):
        return self._mp_ctx.round_params()

    def _is_overflowing(self, x: RealFloat) -> bool:
        """Checks if `x` is overflowing."""
        if x.s:
            return x < self.neg_maxval
        else:
            return x > self.pos_maxval

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

        # step 4. check for overflow
        if self._is_overflowing(xr):
            # check if overflow is allowed
            if exact:
                raise ValueError(f'Rounding {x} under self={self} with n={n} would overflow')

            # overflow
            match self.overflow:
                case OverflowMode.OVERFLOW:
                    raise OverflowError(f'Overflow when rounding {x} under context {self}')
                case OverflowMode.SATURATE:
                    return self.maxval(s=xr.s)
                case OverflowMode.WRAP:
                    # wrap around the ordinals
                    ord_abs = self._mp_ctx.to_ordinal(Float(x=xr, ctx=self)) - self._neg_maxval_ord
                    total_ord = self._pos_maxval_ord - self._neg_maxval_ord + 1
                    ord_mod = (ord_abs % total_ord) + self._neg_maxval_ord
                    return self.from_ordinal(ord_mod, infval=False)
                case _:
                    raise RuntimeError(f'unreachable overflow kind {self.overflow}')

        # step 5. return the rounded value
        return Float(x=xr, ctx=self)

    def round(self, x, *, exact: bool = False):
        x = self._round_prepare(x)
        return self._round_at(x, None, exact)

    def round_at(self, x, n: int, *, exact: bool = False):
        if not isinstance(n, int):
            raise TypeError(f'Expected \'int\' for n={n}, got {type(n)}')
        x = self._round_prepare(x)
        return self._round_at(x, n, exact)

    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        if not isinstance(x, Float) or not self.representable_under(x):
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
            return Float(x=self._mp_ctx.from_ordinal(x), ctx=self)

    def minval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(x=self._mp_ctx.minval(s=s), ctx=self)

    def maxval(self, s: bool = False) -> Float:
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        if s:
            return Float(x=self.neg_maxval, ctx=self)
        else:
            return Float(x=self.pos_maxval, ctx=self)
