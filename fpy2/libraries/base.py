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
# runtime
from ..decorator import fpy, fpy_primitive, pattern
from ..env import ForeignEnv
from ..function import Function
from ..number import (
    BF16,
    FP8P1,
    FP8P2,
    FP8P3,
    FP8P4,
    FP8P5,
    FP8P6,
    FP8P7,
    FP16,
    FP32,
    FP64,
    FP128,
    FP256,
    INTEGER,
    MX_E2M1,
    MX_E2M3,
    MX_E3M2,
    MX_E4M3,
    MX_E5M2,
    MX_E8M0,
    MX_INT8,
    OV,
    # type aliases
    REAL,
    RM,
    S1E4M3,
    S1E5M2,
    SINT8,
    SINT16,
    SINT32,
    SINT64,
    TF32,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    # abstract context types
    Context,
    # concrete context types
    EFloatContext,
    # encoding utilities
    EFloatNanKind,
    EncodableContext,
    ExpContext,
    FixedContext,
    # number types
    Float,
    IEEEContext,
    MPBFixedContext,
    MPBFloatContext,
    MPFixedContext,
    MPFloatContext,
    MPSFloatContext,
    OrdinalContext,
    OverflowMode,
    Real,
    RealFloat,
    RoundingDirection,
    # rounding utilities
    RoundingMode,
    SizedContext,
    SMFixedContext,
)

# builtin operations
from ..ops import *
from ..primitive import Primitive
