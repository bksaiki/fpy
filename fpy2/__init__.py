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

# base library
from .libraries.base import *

# standard library
from . import libraries

# runtime support
from .fpc_context import FPCoreContext, NoSuchContextError
from .interpret import (
    Interpreter,
    DefaultInterpreter,
    set_default_interpreter,
    get_default_interpreter,
)

# compiler
from .backend import (
    Backend,
    CppBackend,
    FPCoreCompiler
)

# submodules
from . import ast
from . import analysis
from . import transform
from . import utils

###########################################################
# typing hints
# TODO: remove these hints

import typing as _typing

from typing import Literal as Dim

_Dims = _typing.TypeVarTuple('_Dims')
_DType = _typing.TypeVar('_DType')

class Tensor(tuple, _typing.Generic[*_Dims, _DType]):
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
