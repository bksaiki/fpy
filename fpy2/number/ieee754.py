"""
This module defines floating-point numbers as defined
by the IEEE 754 standard.
"""

from ..utils import DEFAULT, DefaultOr
from .efloat import EFloatContext, EFloatNanKind
from .round import RoundingMode, OverflowMode

class IEEEContext(EFloatContext):
    """
    Rounding context for IEEE 754 floating-point values.

    This context is parameterized by the size of
    the exponent field `es`, the size of the total
    representation `nbits`, and the rounding mode `rm`.

    This context is implemented as a subclass of `ExtFloatContext` which is
    a more general definition of IEEE 754-like floating-point numbers.
    By inheritance, `IEEEContext` implements `EncodingContext`.
    """

    def __init__(
        self,
        es: int,
        nbits: int,
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode = OverflowMode.OVERFLOW,
        num_randbits: int | None = 0
    ):
        super().__init__(es, nbits, True, EFloatNanKind.IEEE_754, 0, rm, overflow, num_randbits)

    def __repr__(self):
        return self.__class__.__name__ + f'(es={self.es}, nbits={self.nbits}, rm={self.rm!r}, overflow={self.overflow!r}, num_randbits={self.num_randbits!r})'

    def with_params(
        self, *,
        es: DefaultOr[int] = DEFAULT,
        nbits: DefaultOr[int] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        overflow: DefaultOr[OverflowMode] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
        **kwargs
    ) -> 'IEEEContext':
        if es is DEFAULT:
            es = self.es
        if nbits is DEFAULT:
            nbits = self.nbits
        if rm is DEFAULT:
            rm = self.rm
        if overflow is DEFAULT:
            overflow = self.overflow
        if num_randbits is DEFAULT:
            num_randbits = self.num_randbits
        if kwargs:
            raise TypeError(f'Unexpected parameters {kwargs} for IEEEContext')
        return IEEEContext(es, nbits, rm, overflow, num_randbits)
