"""
This module defines floating-point numbers as defined
by the IEEE 754 standard.
"""

from fractions import Fraction

from ..utils import default_repr, bitmask

from .context import SizedContext
from .float import Float
from .real import RealFloat
from .round import RoundingMode, RoundingDirection
from .utils import from_mpfr

@default_repr
class IEEEContext(SizedContext):
    """
    Rounding context for IEEE 754 floating-point values.
    """

    es: int
    """size of the exponent field"""

    nbits: int
    """size of the total representation"""

    rm: RoundingMode
    """rounding mode"""

    def __init__(self, es: int, nbits: int, rm: RoundingMode):
        if es < 2:
            raise ValueError(f'Invalid es={es}, must be at least 2')
        if nbits < es + 2:
            raise ValueError(f'Invalid nbits={nbits}, must be at least es+2={es + 2}')

        self.es = es
        self.nbits = nbits
        self.rm = rm

    @property
    def pmax(self):
        """Maximum allowable precision."""
        return self.nbits - self.es

    @property
    def m(self):
        """Size of the mantissa field."""
        return self.pmax - 1

    @property
    def emax(self):
        """Maximum normalized exponent."""
        return (1 << (self.es - 1)) - 1

    @property
    def emin(self):
        """Minimum normalized exponent."""
        return 1 - self.emax

    @property
    def expmax(self):
        """Maximum unnormalized exponent."""
        return self.emax - self.pmax + 1

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

    @property
    def ebias(self):
        """The exponent "bias" as defined by the IEEE 754 standard."""
        return self.emax


    def is_zero(self, x: Float):
        """Returns if `x` is a zero number."""
        return x.is_zero()

    def is_subnormal(self, x: Float):
        """Returns if `x` is a subnormal number."""
        return x.is_nonzero() and x.e < self.emin

    def is_normal(self, x: Float):
        """Returns if `x` is a normal number."""
        return x.is_nonzero() and x.e >= self.emin

    def is_infinite(self, x: Float):
        """Returns if `x` is an infinite number."""
        return x.isinf

    def is_nan(self, x: Float):
        """Returns if `x` is a NaN number."""
        return x.isnan

    def is_representable(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'RealFloat\', got \'{type(x)}\' for x={x}')

        if x.isnan or x.isinf:
            # special values are valid
            return True
        elif x.exp < self.expmin or x.e > self.emax:
            # rough check on out of range values
            return False
        elif x.p > self.pmax:
            # check on precision
            return False
        elif x.is_zero():
            # shortcut for exact zero
            return True
        elif x.s:
            # tight check (negative values)
            return self.maxval(True) <= x <= self.minval(True)
        else:
            # tight check (non-negative values)
            return self.minval(False) <= x <= self.maxval(False)

    def is_canonical(self, x: Float) -> bool:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'RealFloat\', got \'{type(x)}\' for x={x}')
        if not self.is_representable(x):
            raise TypeError(f'Expected representable value x={x}')

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

    def _overflow_to_infinity(self, x: RealFloat):
        """Should overflows round to infinity (rather than MAX_VAL)?"""
        _, direction = self.rm.to_direction(x.s)
        match direction:
            case RoundingDirection.RTZ:
                # always round towards zero
                return False
            case RoundingDirection.RAZ:
                # always round towards infinity
                return True
            case RoundingDirection.RTE:
                # infinity is considered even for rounding
                return True
            case RoundingDirection.RTO:
                # infinity is considered even for rounding
                return False
            case _:
                raise RuntimeError(f'unrechable {direction}')

    def _round_float(self, x: RealFloat | Float):
        """Like `self.round()` but for only `RealFloat` and `Float` inputs"""
        # step 1. handle special values
        if isinstance(x, Float):
            if x.isnan:
                return Float(isnan=True, ctx=self)
            elif x.isinf:
                return Float(s=x.s, isinf=True, ctx=self)
            else:
                x = x.as_real()

        # step 2. shortcut for exact zero values
        if x.is_zero():
            # exactly zero
            return Float(ctx=self)

        # step 3. round value based on rounding parameters
        rounded = x.round(self.pmax, self.nmin, self.rm)

        # step 4. check for overflow
        if rounded.e > self.emax:
            # overflowing => check which way to round
            if self._overflow_to_infinity(rounded):
                # overflow to infinity
                return Float(x=x, isinf=True, ctx=self)
            else:
                # overflow to MAX_VAL
                max_val = self.maxval(rounded.s)
                return Float(x=max_val, ctx=self)

        # step 5. return rounded result
        return Float(x=rounded, ctx=self)

    def round(self, x):
        match x:
            case Float() | RealFloat():
                x_ = x
            case int():
                x_ = RealFloat(c=x)
            case float() | str():
                x_ = from_mpfr(x, self.pmax)
            case Fraction():
                if x.is_integer():
                    x_ = RealFloat(c=int(x))
                else:
                    x_ = from_mpfr(x, self.pmax)
            case _:
                raise TypeError(f'not valid argument x={x}')

        return self._round_float(x_)

    def normalize(self, x: Float) -> Float:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'RealFloat\', got \'{type(x)}\' for x={x}')
        if not self.is_representable(x):
            raise TypeError(f'Expected representable value x={x}')
        
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
            return Float(x=x.as_real().normalize(self.pmax, self.nmin), ctx=self)

    def maxval(self, s: bool = False):
        c = 1 << self.pmax
        return Float(s=s, c=c, exp=self.expmax, ctx=self)

    def minval(self, s: bool = False):
        return Float(s=s, c=1, exp=self.expmin, ctx=self)

    def to_ordinal(self, x: Float):
        raise NotImplementedError

    def from_ordinal(self, x: int):
        raise NotImplementedError

    def encode(self, x: Float) -> int:
        if not isinstance(x, Float):
            raise TypeError(f'Expected \'RealFloat\', got \'{type(x)}\' for x={x}')
        if not self.is_representable(x):
            raise TypeError(f'Expected representable value x={x}')

        # sign bit
        sbit = 1 if x.s else 0

        # case split by class
        if x.isnan:
            # NaN => qNaN(0)
            ebits = bitmask(self.es)
            mbits = 1 << (self.m - 1)
        elif x.isinf:
            # infinite
            ebits = bitmask(self.es)
            mbits = 0
        elif x.is_zero():
            # zero
            ebits = 0
            mbits = 0
        else:
            # non-zero number
            # first, canonicalize number with maximum precision
            x = x.normalize()

            # case split by class
            if x.e <= self.emin:
                # subnormal number
                ebits = 0
                mbits = x.c
            else:
                # normal number
                ebits = x.e - self.emin + 1
                mbits = x.c & bitmask(self.pmax - 1)

        return (sbit << (self.nbits - 1)) | (ebits << self.m) | mbits


    def decode(self, x: int) -> Float:
        if not isinstance(x, int) and x >= 0 and x < 2 ** self.nbits:
            raise TypeError(f'Expected integer x={x} on [0, 2 ** {self.nbits})')

        # bitmasks
        emask = bitmask(self.es)
        mmask = bitmask(self.m)

        # extract bits
        sbit = x >> (self.nbits - 1)
        ebits = (x >> self.m) & emask
        mbits = x & mmask

        # sign bit
        s = sbit != 0

        # case split on ebits
        if ebits == 0:
            # subnormal / zero
            c = mbits
            return Float(s=s, c=c, exp=self.expmin, ctx=self)
        elif ebits == emask:
            # infinite / NaN
            if mbits == 0:
                return Float(s=s, isinf=True, ctx=self)
            else:
                return Float(s=s, isnan=True, ctx=self)
        else:
            # normal number
            c = (1 << self.m) | mbits
            exp = self.expmin + (ebits - 1)
            return Float(s=s, c=c, exp=exp, ctx=self)
