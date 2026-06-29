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

# submodules
from . import ast
from . import analysis
from . import number
from . import transform
from . import strategies
from . import types
from . import utils

# runtime support
from .fpc_context import FPCoreContext, NoSuchContextError
from .interpret import (
    Interpreter,
    BytecodeInterpreter,
    set_default_interpreter,
    get_default_interpreter,
)

# module
from .module import Module, ModuleCallGraph, ModuleEntry

# compiler
from .backend import (
    Backend,
    CppCompiler,
    FPCoreCompiler,
)

# runner
from .runner import Runner, RunnerWorkerTask
