"""
This module defines the usual fixed-width fixed-point numbers.
"""

from fractions import Fraction
from typing import Optional

from ..utils import default_repr

from .context import EncodableContext
from .mpb_fixed import MPBFixedContext, FixedOverflowKind
from .number import RealFloat, Float
from .round import RoundingMode

@default_repr
class FixedContext(MPBFixedContext, EncodableContext):
    """
    Rounding context for fixed-width fixed-point numbers.

    This context is parameterized by whether it is signed, `signed`,
    the scale factor `scale`, the total number of bits `nbits`,
    the rounding mode `rm`, and the overflow behavior `overflow`.

    Optionally, specify the following keywords:
    - `enable_nan`: if `True`, then NaN is representable [default: `False`]
    - `enable_inf`: if `True`, then infinity is representable [default: `False`]
    - `nan_value`: if NaN is not enabled, what value should NaN round to? [default: `None`];
    if not set, then `round()` will raise a `ValueError` on NaN.
    - `inf_value`: if Inf is not enabled, what value should Inf round to? [default: `None`];
    if not set, then `round()` will raise a `ValueError` on infinity.

    Unlike `MPBFixedContext`, the `FixedContext` inherits from
    `EncodableContext`, since the representation has a well-defined encoding.
    """

    def __init__(
        self,
        signed: bool,
        scale: int,
        nbits: int,
        rm: RoundingMode,
        overflow: FixedOverflowKind,
        *,
        enable_nan: bool = False,
        enable_inf: bool = False,
        nan_value: Optional[Float] = None,
        inf_value: Optional[Float] = None
    ):
        raise NotImplementedError

    def encode(self, x: Float) -> int:
        raise NotImplementedError

    def decode(self, x: int) -> Float:
        raise NotImplementedError

