"""Interpreters for FPy."""

from .interpreter import Interpreter, get_default_interpreter, set_default_interpreter

from .native import PythonInterpreter
from .real import RealInterpreter
from .titanic import TitanicInterpreter

set_default_interpreter(TitanicInterpreter())
