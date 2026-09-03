"""
This module defines floating-point numbers as implemented by MPFR
but with a subnormalization, that is multi-precision floating-point
numbers with subnormals. Hence, "MP-S."
"""

from fractions import Fraction

from ...utils import DEFAULT, DefaultOr, bitmask, default_repr
from ..number import RNG, Float, RealFloat, same_value
from ..round import RoundingMode
from .context import OrdinalContext, zero_sign
from .format import OrdinalFormat


@default_repr
class MPSFloatFormat(OrdinalFormat):
    """
    Number format for multi-precision floating-point numbers with subnormalization.

    This format is parameterized by a fixed precision `pmax` and
    minimum normalized exponent `emin`.
    It describes the set of representable values for `MPSFloatContext`.

    `enable_neg_zero` says whether `-0.0` is one of them.
    """

    pmax: int
    """maximum precision"""

    emin: int
    """minimum (normalized) exponent"""

    enable_nan: bool
    """whether NaN is representable"""

    enable_inf: bool
    """whether (signed) infinity is representable"""

    enable_neg_zero: bool
    """whether `-0.0` is representable"""

    def __init__(
        self,
        pmax: int,
        emin: int,
        enable_nan: bool = True,
        enable_inf: bool = True,
        enable_neg_zero: bool = True,
    ):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise ValueError(f'Expected positive integer for pmax={pmax}')
        if not isinstance(emin, int):
            raise TypeError(f'Expected \'int\' for emin={emin}, got {type(emin)}')
        if not isinstance(enable_neg_zero, bool):
            raise TypeError(f'Expected \'bool\' for enable_neg_zero={enable_neg_zero}, got {type(enable_neg_zero)}')
        self.pmax = pmax
        self.emin = emin
        self.enable_nan = enable_nan
        self.enable_inf = enable_inf
        self.enable_neg_zero = enable_neg_zero

    def __eq__(self, other):
        return (
            isinstance(other, MPSFloatFormat)
            and self.pmax == other.pmax
            and self.emin == other.emin
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
            and self.enable_neg_zero == other.enable_neg_zero
        )

    def __hash__(self):
        return hash((
            self.__class__, self.pmax, self.emin, self.enable_nan,
            self.enable_inf, self.enable_neg_zero,
        ))

    @property
    def expmin(self) -> int:
        """Minimum unnormalized exponent."""
        return self.emin - self.pmax + 1

    @property
    def nmin(self) -> int:
        """First unrepresentable digit for every value in the format."""
        return self.expmin - 1

    def representable_in(self, x: RealFloat | Float) -> bool:
        match x:
            case Float():
                if x.isnan:
                    return self.enable_nan
                if x.isinf:
                    return self.enable_inf
                xr = x._real
            case RealFloat():
                xr = x
            case _:
                raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')

        if xr.is_zero():
            return self.enable_neg_zero or not xr.s

        # precision check (must fit in pmax digits)
        if xr.p > self.pmax:
            p_over = xr.p - self.pmax
            if xr.c & bitmask(p_over) != 0:
                return False

        # position check (subnormal range)
        return xr.is_more_significant(self.nmin)

    def canonical_under(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.representable_in(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        if x.is_nar():
            return True
        elif x.c == 0 or x.e < self.emin:
            return x.exp == self.expmin
        else:
            return x.p == self.pmax

    def normal_under(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.representable_in(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return x.is_nonzero() and x.e >= self.emin

    def normalize(self, x: Float) -> Float:
        if not isinstance(x, Float) or not self.representable_in(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        if x.isnan:
            return Float(isnan=True, s=x.s)
        elif x.isinf:
            return Float(isinf=True, s=x.s)
        elif x.c == 0:
            return Float(c=0, exp=self.expmin, s=x.s)
        else:
            xr = x._real.normalize(self.pmax, self.nmin)
            return Float(x=x, exp=xr.exp, c=xr.c, ctx=None)

    def _to_ordinal(self, x: RealFloat) -> int:
        if x.is_zero():
            return 0
        elif x.e <= self.emin:
            # subnormal: sgn(x) * [ 0 | m ]
            offset = x.exp - self.expmin
            if offset > 0:
                c = x.c << offset
            elif offset < 0:
                c = x.c >> -offset
            else:
                c = x.c
            eord = 0
            mord = c
        else:
            # normal: sgn(x) * [ eord | m ]
            offset = x.p - self.pmax
            if offset > 0:
                c = x.c >> offset
            elif offset < 0:
                c = x.c << -offset
            else:
                c = x.c
            eord = x.e - self.emin + 1
            mord = c & bitmask(self.pmax - 1)

        uord = (eord << (self.pmax - 1)) + mord
        return (-1 if x.s else 1) * uord

    def to_ordinal(self, x: Float, infval: bool = False) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected a \'Float\', got \'{type(x)}\' for x={x}')
        if not self.representable_in(x):
            raise ValueError(f'x={x} is not representable under this format')
        if infval:
            raise ValueError('infval=True is invalid for formats without a maximum value')
        if x.is_nar():
            raise ValueError(f'Expected a finite value for x={x}')
        return self._to_ordinal(x.as_real())

    def to_fractional_ordinal(self, x: Float) -> Fraction:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        if x.is_nar():
            raise ValueError(f'Expected a finite value for x={x}')

        if self.representable_in(x):
            return Fraction(self._to_ordinal(x.as_real()))

        xr = x.as_real()
        above = xr.round(self.pmax, self.nmin, rm=RoundingMode.RTP)
        below = xr.round(self.pmax, self.nmin, rm=RoundingMode.RTN)

        delta_x: RealFloat = xr - below
        delta: RealFloat = above - below
        t = delta_x.as_rational() / delta.as_rational()

        below_ord = self._to_ordinal(below)
        return Fraction(below_ord) + t

    def from_ordinal(self, x: int, infval: bool = False) -> Float:
        if not isinstance(x, int):
            raise TypeError(f'Expected an \'int\', got \'{type(x)}\' for x={x}')
        if infval:
            raise ValueError('infval=True is invalid for formats without a maximum value')

        s = x < 0
        uord = abs(x)

        if x == 0:
            return Float()
        eord, mord = divmod(uord, 1 << (self.pmax - 1))
        if eord == 0:
            return Float(s=s, c=mord, exp=self.expmin)
        c = (1 << (self.pmax - 1)) | mord
        exp = self.expmin + (eord - 1)
        return Float(s=s, c=c, exp=exp)

    def zero(self, s: bool = False) -> Float:
        """Returns a signed 0 under this format.

        Raises a `ValueError` for `s=True` where the format has no negative
        zero: there is no such value, and dropping the sign would answer a
        different question than the one asked.
        """
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        if s and not self.enable_neg_zero:
            raise ValueError('`-0.0` is not representable under this format')
        return Float(s=s, c=0, exp=self.expmin)

    def minval(self, s: bool = False) -> Float:
        """Returns the smallest non-zero value with sign `s` under this format."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(s=s, c=1, exp=self.expmin)

    def min_subnormal(self, s: bool = False) -> Float:
        """Returns the smallest subnormal value with sign `s` under this format."""
        return self.minval(s)

    def max_subnormal(self, s: bool = False) -> Float:
        """Returns the largest subnormal value with sign `s` under this format."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(s=s, c=bitmask(self.pmax - 1), exp=self.expmin)

    def min_normal(self, s: bool = False) -> Float:
        """Returns the smallest normal value with sign `s` under this format."""
        if not isinstance(s, bool):
            raise TypeError(f'Expected \'bool\' for s={s}, got {type(s)}')
        return Float(s=s, c=1 << (self.pmax - 1), exp=self.expmin)


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

    Optionally, specify the following keywords:

    - `enable_nan`: if `True`, then NaN is representable [default: `True`]
    - `enable_inf`: if `True`, then infinity is representable [default: `True`]
    - `enable_neg_zero`: if `True`, then `-0.0` is representable [default: `True`]
    - `nan_value`: if NaN is not enabled, what value should NaN round to? [default: `None`];
      if not set, then `round()` will raise a `ValueError` on NaN.
    - `inf_value`: if Inf is not enabled, what value should Inf round to? [default: `None`];
      if not set, then `round()` will raise a `ValueError` on infinity.
    """

    pmax: int
    """maximum precision"""

    emin: int
    """minimum (normalized exponent)"""

    rm: RoundingMode
    """rounding mode"""

    num_randbits: int | None
    """number of random bits for stochastic rounding, if applicable"""

    rng: RNG | None
    """random number generator for stochastic rounding, if applicable"""

    enable_nan: bool
    """is NaN representable?"""

    enable_inf: bool
    """is infinity representable?"""

    enable_neg_zero: bool
    """is `-0.0` representable?  When `False`, a negative value that rounds to
    zero gives `+0.0`, so a result is always in `format()`."""

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

    _fmt: MPSFloatFormat
    """precomputed format object"""

    def __init__(
        self,
        pmax: int,
        emin: int,
        rm: RoundingMode = RoundingMode.RNE,
        num_randbits: int | None = 0,
        *,
        rng: RNG | None = None,
        enable_nan: bool = True,
        enable_inf: bool = True,
        enable_neg_zero: bool = True,
        nan_value: Float | None = None,
        inf_value: Float | None = None
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
        if not isinstance(enable_nan, bool):
            raise TypeError(f'Expected \'bool\' for enable_nan={enable_nan}, got {type(enable_nan)}')
        if not isinstance(enable_inf, bool):
            raise TypeError(f'Expected \'bool\' for enable_inf={enable_inf}, got {type(enable_inf)}')
        if not isinstance(enable_neg_zero, bool):
            raise TypeError(f'Expected \'bool\' for enable_neg_zero={enable_neg_zero}, got {type(enable_neg_zero)}')

        fmt = MPSFloatFormat(pmax, emin, enable_nan, enable_inf, enable_neg_zero)

        if nan_value is not None:
            if not isinstance(nan_value, Float):
                raise TypeError(f'Expected \'Float\' for nan_value={nan_value}, got {type(nan_value)}')
            if not enable_nan and not fmt.representable_in(nan_value):
                raise ValueError(f'Rounding NaN to unrepresentable value {nan_value}')

        if inf_value is not None:
            if not isinstance(inf_value, Float):
                raise TypeError(f'Expected \'Float\' for inf_value={inf_value}, got {type(inf_value)}')
            # the substitute takes the operand's sign, so both are reachable
            for signed in (Float(s=False, x=inf_value), Float(s=True, x=inf_value)):
                if not enable_inf and not fmt.representable_in(signed):
                    raise ValueError(f'Rounding Inf to unrepresentable value {signed}')

        self.pmax = pmax
        self.emin = emin
        self.rm = rm
        self.num_randbits = num_randbits
        self.rng = rng
        self.enable_nan = enable_nan
        self.enable_inf = enable_inf
        self.enable_neg_zero = enable_neg_zero
        self.nan_value = nan_value
        self.inf_value = inf_value
        self._fmt = fmt

    def __eq__(self, other):
        return (
            isinstance(other, MPSFloatContext)
            and self.pmax == other.pmax
            and self.emin == other.emin
            and self.rm == other.rm
            and self.num_randbits == other.num_randbits
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
            and self.enable_neg_zero == other.enable_neg_zero
            and same_value(self.nan_value, other.nan_value)
            and same_value(self.inf_value, other.inf_value)
        )

    def __hash__(self):
        return hash((
            self.pmax,
            self.emin,
            self.rm,
            self.num_randbits,
            self.enable_nan,
            self.enable_inf,
            self.enable_neg_zero,
            self.nan_value,
            self.inf_value
        ))

    @property
    def expmin(self):
        """Minimum unnormalized exponent."""
        return self.emin - self.pmax + 1

    @property
    def nmin(self):
        """First unrepresentable digit for every value in the representation."""
        return self.expmin - 1

    def with_params(
        self, *,
        pmax: DefaultOr[int] = DEFAULT,
        emin: DefaultOr[int] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
        rng: DefaultOr[RNG | None] = DEFAULT,
        enable_nan: DefaultOr[bool] = DEFAULT,
        enable_inf: DefaultOr[bool] = DEFAULT,
        enable_neg_zero: DefaultOr[bool] = DEFAULT,
        nan_value: DefaultOr[Float | None] = DEFAULT,
        inf_value: DefaultOr[Float | None] = DEFAULT,
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
        if rng is DEFAULT:
            rng = self.rng
        if enable_nan is DEFAULT:
            enable_nan = self.enable_nan
        if enable_inf is DEFAULT:
            enable_inf = self.enable_inf
        if enable_neg_zero is DEFAULT:
            enable_neg_zero = self.enable_neg_zero
        if nan_value is DEFAULT:
            nan_value = self.nan_value
        if inf_value is DEFAULT:
            inf_value = self.inf_value
        if kwargs:
            raise TypeError(f'Unexpected keyword arguments: {kwargs}')
        return MPSFloatContext(
            pmax, emin, rm, num_randbits,
            rng=rng,
            enable_nan=enable_nan,
            enable_inf=enable_inf,
            enable_neg_zero=enable_neg_zero,
            nan_value=nan_value,
            inf_value=inf_value
        )

    def is_stochastic(self) -> bool:
        return self.num_randbits != 0

    def format(self) -> MPSFloatFormat:
        return self._fmt

    @classmethod
    def from_format(
        cls,
        fmt: MPSFloatFormat,
        *,
        rm: RoundingMode = RoundingMode.RNE,
        num_randbits: int | None = 0,
        rng: 'RNG | None' = None,
        nan_value: Float | None = None,
        inf_value: Float | None = None
    ) -> 'MPSFloatContext':
        """Creates a context from a `MPSFloatFormat` and rounding parameters."""
        if not isinstance(fmt, MPSFloatFormat):
            raise TypeError(f'Expected \'MPSFloatFormat\', got {type(fmt)}')
        return cls(
            fmt.pmax, fmt.emin, rm, num_randbits,
            rng=rng,
            enable_nan=fmt.enable_nan,
            enable_inf=fmt.enable_inf,
            enable_neg_zero=fmt.enable_neg_zero,
            nan_value=nan_value,
            inf_value=inf_value
        )

    def rounding_mode(self):
        return self.rm

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
                if self.enable_nan:
                    return Float(isnan=True, ctx=self)
                elif self.nan_value is None:
                    raise ValueError('Cannot round NaN under this context')
                else:
                    return Float(x=self.nan_value, ctx=self)
            elif x.isinf:
                if self.enable_inf:
                    return Float(s=x.s, isinf=True, ctx=self)
                elif self.inf_value is None:
                    raise ValueError('Cannot round infinity under this context')
                else:
                    return Float(s=x.s, x=self.inf_value, ctx=self)
            else:
                x = x._real

        # step 2. shortcut for exact zero values (preserve signed zero)
        if x.is_zero():
            return Float(s=zero_sign(x, self.enable_neg_zero), ctx=self)

        # step 3. select rounding parameter `n`
        if n is None or n < self.nmin:
            n = self.nmin

        # step 4. round value based on rounding parameters
        xr = x.round(self.pmax, n, self.rm, self.num_randbits, rng=self.rng, exact=exact)

        # step 5. wrap the result in a Float
        return Float(x=xr, s=zero_sign(xr, self.enable_neg_zero), ctx=self)

    def round(self, x, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, None, exact)

    def round_at(self, x, n: int, *, exact: bool = False) -> Float:
        x = self._round_prepare(x)
        return self._round_at(x, n, exact)

    def zero(self, s: bool = False) -> Float:
        """Returns a signed 0 under this context."""
        return Float(x=self.format().zero(s), ctx=self)

    def min_subnormal(self, s: bool = False) -> Float:
        """Returns the smallest subnormal value with sign `s` under this context."""
        return Float(x=self.format().min_subnormal(s), ctx=self)

    def max_subnormal(self, s: bool = False) -> Float:
        """Returns the largest subnormal value with sign `s` under this context."""
        return Float(x=self.format().max_subnormal(s), ctx=self)

    def min_normal(self, s: bool = False) -> Float:
        """Returns the smallest normal value with sign `s` under this context."""
        return Float(x=self.format().min_normal(s), ctx=self)
