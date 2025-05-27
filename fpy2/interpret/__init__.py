"""Interpreters for FPy."""

from .interpreter import Interpreter, get_default_interpreter, set_default_interpreter

from .native import PythonInterpreter
from .default import DefaultInterpreter

set_default_interpreter(DefaultInterpreter())
