"""
This module defines floating-point numbers as defined
by the IEEE 754 standard.
"""

from fractions import Fraction

from ..utils import default_repr, bitmask

from .context import EncodableContext
from .float import Float
from .mpb import MPBContext
from .real import RealFloat
from .round import RoundingMode, RoundingDirection
from .utils import from_mpfr

def _ieee_to_mpb(es: int, nbits: int, rm: RoundingMode):
    """Converts IEEEContext parameters to MPBContext parameters"""
    # IEEE 754 derived parameters
    p = nbits - es
    emax = (1 << (es - 1)) - 1
    emin = 1 - emax
    expmax = emax - p + 1
    expmin = emin - p + 1
    # +MAX_VAL
    maxval = RealFloat(c=bitmask(p), exp=expmax)
    # MPBContext
    return MPBContext(p, emin, maxval, rm)


@default_repr(ignore=['_mpb_ctx'])
class IEEEContext(EncodableContext):
    """
    Rounding context for IEEE 754 floating-point values.
    """

    es: int
    """size of the exponent field"""

    nbits: int
    """size of the total representation"""

    rm: RoundingMode
    """rounding mode"""

    _mpb_ctx: MPBContext
    """this context as an `MPBContext`"""

    def __init__(self, es: int, nbits: int, rm: RoundingMode):
        if es < 2:
            raise ValueError(f'Invalid es={es}, must be at least 2')
        if nbits < es + 2:
            raise ValueError(f'Invalid nbits={nbits}, must be at least es+2={es + 2}')

        self.es = es
        self.nbits = nbits
        self.rm = rm
        self._mpb_ctx = _ieee_to_mpb(es, nbits, rm)

    @property
    def pmax(self):
        """Maximum allowable precision."""
        return self._mpb_ctx.pmax

    @property
    def emax(self):
        """Maximum normalized exponent."""
        return self._mpb_ctx.emax

    @property
    def emin(self):
        """Minimum normalized exponent."""
        return self._mpb_ctx.emin

    @property
    def expmax(self):
        """Maximum unnormalized exponent."""
        return self._mpb_ctx.expmax

    @property
    def expmin(self):
        """Minimum unnormalized exponent."""
        return self._mpb_ctx.expmin

    @property
    def nmin(self):
        """
        First unrepresentable digit for every value in the representation.
        """
        return self._mpb_ctx.nmin

    @property
    def m(self):
        """Size of the mantissa field."""
        return self.pmax - 1

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
        return self._mpb_ctx.is_representable(x)

    def is_canonical(self, x: Float) -> bool:
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mpb_ctx.is_canonical(x)

    def normalize(self, x: Float) -> Float:
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        x = self._mpb_ctx.normalize(x)
        x.ctx = self
        return x

    def round(self, x):
        return self._mpb_ctx.round(x)

    def to_ordinal(self, x: Float, infval = False) -> int:
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')
        return self._mpb_ctx.to_ordinal(x, infval=infval)

    def from_ordinal(self, x: int, infval = False):
        return self._mpb_ctx.from_ordinal(x, infval=infval)

    def minval(self, s: bool = False):
        minval = self._mpb_ctx.minval(s)
        minval.ctx = self
        return minval

    def maxval(self, s: bool = False):
        maxval = self._mpb_ctx.maxval(s)
        maxval.ctx = self
        return maxval

    def encode(self, x: Float) -> int:
        if not isinstance(x, Float) or not self.is_representable(x):
            raise TypeError(f'Expected a representable \'Float\', got \'{type(x)}\' for x={x}')

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

            # canonicalize number if necessary
            if not x.is_canonical():
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

