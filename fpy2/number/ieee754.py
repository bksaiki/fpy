"""
This module defines floating-point numbers as defined
by the IEEE 754 standard.
"""

from .ext_float import ExtFloatContext, ExtFloatNanKind
from .round import RoundingMode

class IEEEContext(ExtFloatContext):
    """
    Rounding context for IEEE 754 floating-point values.

    This context is parameterized by the size of
    the exponent field `es`, the size of the total
    representation `nbits`, and the rounding mode `rm`.

    This context is implemented as a subclass of `ExtFloatContext` which is
    a more general definition of IEEE 754-like floating-point numbers.
    By inheritance, `IEEEContext` implements `EncodingContext`.
    """

    def __init__(self, es: int, nbits: int, rm: RoundingMode):
        super().__init__(es, nbits, True, ExtFloatNanKind.IEEE_754, 0, rm)

    def __repr__(self):
        return self.__class__.__name__ + f'(es={self.es}, nbits={self.nbits}, rm={self.rm!r})'

    def with_rm(self, rm: RoundingMode):
        return IEEEContext(self.es, self.nbits, rm)
