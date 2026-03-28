"""Interpreters for FPy."""

from .byte import BytecodeInterpreter
from .interpreter import Interpreter, get_default_interpreter, set_default_interpreter

set_default_interpreter(BytecodeInterpreter())
