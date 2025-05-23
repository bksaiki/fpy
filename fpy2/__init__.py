"""
FPy is a library for simulating numerical programs
with many different number systems.

It provides an embedded DSL for specifying programs via its `@fpy` decorator.
The language has a runtime that can simulate programs
under different number systems and compilers to other languages.

The numbers library supports many different number types including:

 - multiprecision floating point (`MPContext`)
 - multiprecision floatingpoint with subnormalization (`MPSContext`)
 - bounded, multiprecision floating point (`MPBContext`)
 - IEEE 754 floating point (`IEEEContext`)

These number systems guarantee correct rounding via MPFR.
"""

from .number import (
    # number types
    Float,
    RealFloat,
    # abstract context types
    Context,
    OrdinalContext,
    SizedContext,
    EncodableContext,
    # concrete context types
    ExtContext,
    MPContext,
    MPSContext,
    MPBContext,
    IEEEContext,
    # rounding utilities
    RoundingMode,
    RoundingDirection, RM,
    # encoding utilities
    ExtNanKind,
    # type aliases
    FP256, FP128, FP64, FP32, FP16,
    TF32, BF16,
    S1E5M2, S1E4M3,
    MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1,
    FP8P1, FP8P2, FP8P3, FP8P4, FP8P5, FP8P6, FP8P7
)

from .fpc_context import FPCoreContext, NoSuchContextError

from .decorator import fpy, pattern

from .backend import (
    Backend,
    FPCoreCompiler,
    FPYCompiler,
)

from .interpret import (
    Interpreter,
    PythonInterpreter,
    RealInterpreter,
    DefaultInterpreter,
    set_default_interpreter,
    get_default_interpreter,
)

from .function import Function

from .utils import (
    fraction,
    digits_to_fraction as digits,
    decnum_to_fraction as decnum,
    hexnum_to_fraction as hexnum,
)
