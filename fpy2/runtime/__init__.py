from .env import ForeignEnv
from .function import (
    Function,
    Interpreter,
    get_default_interpreter,
    set_default_interpreter
)

from .native import PythonInterpreter
from .real import RealInterpreter
from .titanic import TitanicInterpreter

from .profile import ExprProfiler, FunctionProfiler

set_default_interpreter(TitanicInterpreter())
