

from ..utils import default_repr, bitmask

from enum import IntEnum

from .context import EncodableContext
from .number import Float
from .mpb import MPBContext
from .real import RealFloat
from .round import RoundingMode


class ExtNanKind(IntEnum):
    """
    Describes how NaN values are encoded for `ExtContext` rounding contexts.
    """
    IEEE_754 = 0
    """IEEE 754 compliant: NaNs have the largest exponent"""
    MAX_VAL = 1
    """NaN has largest exponent and mantissa of all ones"""
    NEG_ZERO = 2
    """NaN replaces -0"""
    NONE = 3
    """No NaNs"""


@default_repr
class ExtContext(EncodableContext):
    """
    Rounding context for the "extended" floating-point format
    as described in Brett Saiki's blog post. These formats extend
    the usual IEEE 754 format with three addition parameters:
     - are infinities enabled?
     - how are NaNs encoded?
     - should the exponent be shifted?

    See https://uwplse.org/2025/02/17/Small-Floats.html for details.
    """

    es: int
    """size of the exponent field"""

    nbits: int
    """size of the total representation"""

    enable_inf: bool
    """whether to enable infinities"""

    nan_encoding: ExtNanKind
    """how NaNs are encoded"""

    eoffset: int
    """the exponent offset"""

    rm: RoundingMode
    """rounding mode"""

    _mpb_ctx: MPBContext
    """this context as an `MPBContext`"""

    def __init__(
        self,
        es: int,
        nbits: int,
        enable_inf: bool,
        nan_encoding: ExtNanKind,
        eoffset: int,
        rm: RoundingMode,
    ):
        if not isinstance(es, int):
            raise TypeError(f'Expected \'int\', got \'{type(es)}\' for es={es}')
        if not isinstance(nbits, int):
            raise TypeError(f'Expected \'int\', got \'{type(nbits)}\' for nbits={nbits}')
        if not isinstance(rm, RoundingMode):
            raise TypeError(f'Expected \'RoundingMode\', got \'{type(rm)}\' for rm={rm}')
        if not isinstance(enable_inf, bool):
            raise TypeError(f'Expected \'bool\', got \'{type(enable_inf)}\' for enable_inf={enable_inf}')
        if not isinstance(nan_encoding, ExtNanKind):
            raise TypeError(f'Expected \'NanEncoding\', got \'{type(nan_encoding)}\' for nan_encoding={nan_encoding}')
        if not isinstance(eoffset, int):
            raise TypeError(f'Expected \'int\', got \'{type(eoffset)}\' for eoffset={eoffset}')

        if es < 0:
            raise ValueError(f'Invalid es={es}, must be an integer >= 2')
        if nbits < 1:
            raise ValueError(f'Invalid nbits={nbits}, must be an integer >= 1')

        self.es = es
        self.nbits = nbits
        self.enable_inf = enable_inf
        self.nan_encoding = nan_encoding
        self.eoffset = eoffset
        self.rm = rm
        self._mpb_ctx = None

    def with_rm(self, rm):
        raise NotImplementedError

    def is_representable(self, x: RealFloat | Float) -> bool:
        raise NotImplementedError

    def is_canonical(self, x: Float) -> bool:
        raise NotImplementedError

    def normalize(self, x: Float) -> Float:
        raise NotImplementedError

    def round_params(self):
        raise NotImplementedError

    def round(self, x):
        raise NotImplementedError

    def round_at(self, x, n):
        raise NotImplementedError

    def to_ordinal(self, x: Float, infval = False) -> int:
        raise NotImplementedError

    def from_ordinal(self, x: int, infval = False) -> Float:
        raise NotImplementedError

    def minval(self, s = False) -> Float:
        raise NotImplementedError

    def maxval(self, s = False) -> Float:
        raise NotImplementedError

    def encode(self, x: Float) -> int:
        raise NotImplementedError

    def decode(self, x: int) -> Float:
        raise NotImplementedError

