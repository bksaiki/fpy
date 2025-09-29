"""
Basic functionality for FPy.
Import this module if you wish to use FPy without any additional libraries.

This module provides:
- runtime decorators
- rounding contexts
- builtin operations
- typing hints
"""

# rounding contexts
from ..number import (
    # number types
    Float,
    RealFloat,
    # abstract context types
    Context,
    OrdinalContext,
    SizedContext,
    EncodableContext,
    # concrete context types
    EFloatContext,
    ExpContext,
    FixedContext,
    MPFixedContext,
    MPFloatContext,
    MPBFixedContext,
    MPBFloatContext,
    MPSFloatContext,
    IEEEContext,
    RealContext,
    SMFixedContext,
    # rounding utilities
    RoundingMode,
    RoundingDirection, RM,
    # encoding utilities
    EFloatNanKind,
    OverflowMode, OV,
    # type aliases
    REAL,
    FP256, FP128, FP64, FP32, FP16,
    TF32, BF16,
    S1E5M2, S1E4M3,
    MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1,
    MX_E8M0, MX_INT8,
    FP8P1, FP8P2, FP8P3, FP8P4, FP8P5, FP8P6, FP8P7,
    INTEGER,
    SINT8, SINT16, SINT32, SINT64,
    UINT8, UINT16, UINT32, UINT64,
    Real,
)

# builtin operations
from ..ops import *

# runtime
from ..decorator import fpy, pattern, fpy_primitive
from ..env import ForeignEnv
from ..function import Function
from ..primitive import Primitive

# alias for `round`
R = round

# integer rounding
@fpy
def Z(x: Real):
    with INTEGER:
        return round(x)
