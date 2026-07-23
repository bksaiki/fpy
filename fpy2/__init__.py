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
# standard library
# submodules
from . import analysis, ast, libraries, number, strategies, transform, types, utils

# compiler
from .backend import (
    Backend,
    CppCompiler,
    FPCoreCompiler,
)

# runtime support
from .fpc_context import FPCoreContext, NoSuchContextError
from .interpret import (
    BytecodeInterpreter,
    Interpreter,
    get_default_interpreter,
    set_default_interpreter,
)
from .libraries.base import *

# module
from .module import Module, ModuleCallGraph, ModuleEntry

# runner
from .runner import Runner, RunnerWorkerTask
