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
    MPContext,
    MPSContext,
    MPBContext,
    IEEEContext,
    # rounding utilities
    RoundingMode,
    RoundingDirection,
    RM,
)

from .frontend import fpy

from .backend import (
    Backend,
    FPCoreCompiler,
    FPYCompiler,
)

from .interpret import (
    PythonInterpreter,
    RealInterpreter,
    TitanicInterpreter,
)

from .runtime import (
    Function,
    Interpreter,
    set_default_interpreter,
    get_default_interpreter,
    FunctionProfiler,
    ExprProfiler,
)

from .utils import (
    fraction,
    digits_to_fraction as digits,
    decnum_to_fraction as decnum,
    hexnum_to_fraction as hexnum,
)
