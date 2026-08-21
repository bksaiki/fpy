"""
This module defines floating-point numbers as implemented by MPFR,
that is, multi-precision floating-point numbers. Hence, "MP."
"""

from ...utils import DEFAULT, DefaultOr, bitmask, default_repr
from ..number import RNG, Float, RealFloat, same_value
from ..round import RoundingMode
from .context import Context
from .format import Format


@default_repr
class MPFloatFormat(Format):
    """
    Number format for multi-precision floating-point numbers.

    This format is parameterized by a fixed precision `pmax`.
    It describes the set of representable values for `MPFloatContext`.
    """

    pmax: int
    """maximum precision"""

    enable_nan: bool
    """whether NaN is representable"""

    enable_inf: bool
    """whether (signed) infinity is representable"""

    def __init__(self, pmax: int, enable_nan: bool = True, enable_inf: bool = True):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise ValueError(f'Expected positive integer for pmax={pmax}')
        self.pmax = pmax
        self.enable_nan = enable_nan
        self.enable_inf = enable_inf

    def __eq__(self, other):
        return (
            isinstance(other, MPFloatFormat)
            and self.pmax == other.pmax
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
        )

    def __hash__(self):
        return hash((self.__class__, self.pmax, self.enable_nan, self.enable_inf))

    def representable_in(self, x: RealFloat | Float) -> bool:
        match x:
            case Float():
                if x.isnan:
                    return self.enable_nan
                if x.isinf:
                    return self.enable_inf
                if x.is_zero():
                    return True
            case RealFloat():
                if x.is_zero():
                    return True
            case _:
                raise TypeError(f'Expected \'RealFloat\' or \'Float\', got \'{type(x)}\' for x={x}')

        # value is finite and non-zero
        # check if the value can be normalized within pmax digits
        if x.p <= self.pmax:
            return True
        else:
            # excess bits must all be zero
            p_over = x.p - self.pmax
            c_lost = x.c & bitmask(p_over)
            return c_lost == 0

    def canonical_under(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.representable_in(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        if x.is_nar():
            return True
        elif x.is_zero():
            return x.exp == 0
        else:
            return x.p == self.pmax

    def normal_under(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'Float\', got \'{type(x)}\' for x={x}')
        return x.is_nonzero()

    def normalize(self, x: Float) -> Float:
        if not isinstance(x, Float) or not self.representable_in(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

        if x.isnan:
            return Float(isnan=True, s=x.s)
        elif x.isinf:
            return Float(isinf=True, s=x.s)
        elif x.c == 0:
            return Float(c=0, exp=0, s=x.s)
        else:
            xr = x._real.normalize(self.pmax, None)
            return Float(x=x, exp=xr.exp, c=xr.c, ctx=None)


@default_repr
class MPFloatContext(Context):
    """
    Rounding context for multi-precision floating-point numbers.

    This context is parameterized by a fixed precision `pmax`
    and a rounding mode `rm`. It emulates floating-point numbers
    as implemented by MPFR.

    Optionally, specify the following keywords:

    - `enable_nan`: if `True`, then NaN is representable [default: `True`]
    - `enable_inf`: if `True`, then infinity is representable [default: `True`]
    - `nan_value`: if NaN is not enabled, what value should NaN round to? [default: `None`];
      if not set, then `round()` will raise a `ValueError` on NaN.
    - `inf_value`: if Inf is not enabled, what value should Inf round to? [default: `None`];
      if not set, then `round()` will raise a `ValueError` on infinity.
    """

    pmax: int
    """maximum precision"""

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

    _fmt: MPFloatFormat
    """precomputed format object"""

    def __init__(
        self,
        pmax: int,
        rm: RoundingMode = RoundingMode.RNE,
        num_randbits: int | None = 0,
        *,
        rng: RNG | None = None,
        enable_nan: bool = True,
        enable_inf: bool = True,
        nan_value: Float | None = None,
        inf_value: Float | None = None
    ):
        if not isinstance(pmax, int):
            raise TypeError(f'Expected \'int\' for pmax={pmax}, got {type(pmax)}')
        if pmax < 1:
            raise TypeError(f'Expected integer p < 1 for p={pmax}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\' for rm={rm}, got {type(rm)}')
        if num_randbits is not None and not isinstance(num_randbits, int):
            raise TypeError(f'Expected \'int\' for num_randbits={num_randbits}, got {type(num_randbits)}')
        if not isinstance(enable_nan, bool):
            raise TypeError(f'Expected \'bool\' for enable_nan={enable_nan}, got {type(enable_nan)}')
        if not isinstance(enable_inf, bool):
            raise TypeError(f'Expected \'bool\' for enable_inf={enable_inf}, got {type(enable_inf)}')

        fmt = MPFloatFormat(pmax, enable_nan, enable_inf)

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
        self.rm = rm
        self.num_randbits = num_randbits
        self.rng = rng
        self.enable_nan = enable_nan
        self.enable_inf = enable_inf
        self.nan_value = nan_value
        self.inf_value = inf_value
        self._fmt = fmt

    def __eq__(self, other):
        return (
            isinstance(other, MPFloatContext)
            and self.pmax == other.pmax
            and self.rm == other.rm
            and self.num_randbits == other.num_randbits
            and self.enable_nan == other.enable_nan
            and self.enable_inf == other.enable_inf
            and same_value(self.nan_value, other.nan_value)
            and same_value(self.inf_value, other.inf_value)
        )

    def __hash__(self):
        return hash((
            self.__class__,
            self.pmax,
            self.rm,
            self.num_randbits,
            self.enable_nan,
            self.enable_inf,
            self.nan_value,
            self.inf_value
        ))

    def with_params(
        self, *,
        pmax: DefaultOr[int] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
        rng: DefaultOr[RNG | None] = DEFAULT,
        enable_nan: DefaultOr[bool] = DEFAULT,
        enable_inf: DefaultOr[bool] = DEFAULT,
        nan_value: DefaultOr[Float | None] = DEFAULT,
        inf_value: DefaultOr[Float | None] = DEFAULT,
        **kwargs
    ) -> 'MPFloatContext':
        if pmax is DEFAULT:
            pmax = self.pmax
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
        if nan_value is DEFAULT:
            nan_value = self.nan_value
        if inf_value is DEFAULT:
            inf_value = self.inf_value
        if kwargs:
            raise TypeError(f'Unexpected keyword arguments: {kwargs}')
        return MPFloatContext(
            pmax, rm, num_randbits,
            rng=rng,
            enable_nan=enable_nan,
            enable_inf=enable_inf,
            nan_value=nan_value,
            inf_value=inf_value
        )

    def is_stochastic(self) -> bool:
        return self.num_randbits != 0

    def format(self) -> MPFloatFormat:
        return self._fmt

    @classmethod
    def from_format(
        cls,
        fmt: MPFloatFormat,
        *,
        rm: RoundingMode = RoundingMode.RNE,
        num_randbits: int | None = 0,
        rng: 'RNG | None' = None,
        nan_value: Float | None = None,
        inf_value: Float | None = None
    ) -> 'MPFloatContext':
        """Creates a context from a `MPFloatFormat` and rounding parameters."""
        if not isinstance(fmt, MPFloatFormat):
            raise TypeError(f'Expected \'MPFloatFormat\', got {type(fmt)}')
        return cls(
            fmt.pmax, rm, num_randbits,
            rng=rng,
            enable_nan=fmt.enable_nan,
            enable_inf=fmt.enable_inf,
            nan_value=nan_value,
            inf_value=inf_value
        )

    def _round_at(self, x: RealFloat | Float, n: int | None, exact: bool) -> Float:
        """
        Like `self.round()` but for only `RealFloat` and `Float` inputs.

        Optionally specify `n` as the least absolute digit position.
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
            return Float(s=x.s, ctx=self)

        # step 3. round value based on rounding parameters
        xr = x.round(self.pmax, n, self.rm, self.num_randbits, rng=self.rng, exact=exact)

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
