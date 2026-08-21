"""Interpreters for FPy."""

from .byte import BytecodeInterpreter
from .interpreter import Interpreter, get_default_interpreter, set_default_interpreter
from .value import Foreign, RealValue, ScalarValue, Value

set_default_interpreter(BytecodeInterpreter())
