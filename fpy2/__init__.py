"""
FPy is a library for simulating numerical programs
with many different number systems.

It provides an embedded DSL for specifying programs via its `@fpy` decorator.
The language has a runtime that can simulate programs
under different number systems and compilers to other languages.

The numbers library supports many different number types including:

 - multiprecision floating point (`MPFloatContext`)
 - multiprecision floatingpoint with subnormalization (`MPSFloatContext`)
 - bounded, multiprecision floating point (`MPBFloatContext`)
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
    ExtFloatContext,
    FixedContext,
    MPFixedContext,
    MPFloatContext,
    MPBFixedContext,
    MPBFloatContext,
    MPSFloatContext,
    IEEEContext,
    RealContext,
    # rounding utilities
    RoundingMode,
    RoundingDirection, RM,
    # encoding utilities
    ExtFloatNanKind,
    FixedOverflowKind, OF,
    # type aliases
    FP256, FP128, FP64, FP32, FP16,
    TF32, BF16,
    S1E5M2, S1E4M3,
    MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1,
    FP8P1, FP8P2, FP8P3, FP8P4, FP8P5, FP8P6, FP8P7,
    INTEGER,
    SINT8, SINT16, SINT32, SINT64,
    UINT8, UINT16, UINT32, UINT64
)

from .fpc_context import FPCoreContext, NoSuchContextError

from .decorator import fpy, pattern, fpy_prim

from .backend import (
    Backend,
    FPCoreCompiler
)

from .interpret import (
    Interpreter,
    PythonInterpreter,
    DefaultInterpreter,
    set_default_interpreter,
    get_default_interpreter,
)

from .function import Function
from .env import ForeignEnv

from .utils import (
    fraction,
    digits_to_fraction as digits,
    decnum_to_fraction as decnum,
    hexnum_to_fraction as hexnum,
)

###########################################################
# Re-exports

from titanfp.titanic.ndarray import NDArray

###########################################################
# typing hints

import typing

from typing import Literal as Dim

_Dims = typing.TypeVarTuple('_Dims')
_DType = typing.TypeVar('_DType')

class Tensor(tuple, typing.Generic[*_Dims, _DType]):
    """
    FPy type hint for a homogenous tensor object::

        from fpy2 import Tensor, Real
        from typing import TypeAlias

        MatrixN: TypeAlias = Tensor[Literal['N', 'N'], Real]
        Matrix3: TypeAlias = Tensor[Literal[3, 3], Real]

    Tensors have fixed or symbolic sizes and a uniform scalar data type.

    Values of this type should not be constructed directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
