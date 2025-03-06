from .env import PythonEnv
from .function import (
    Function,
    Interpreter,
    get_default_interpreter,
    set_default_interpreter
)

from .native import PythonInterpreter
from .real import RealInterpreter, FunctionProfiler
from .sampling import sample_function
from .titanic import TitanicInterpreter

set_default_interpreter(TitanicInterpreter())
