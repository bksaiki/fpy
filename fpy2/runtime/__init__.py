from .env import PythonEnv
from .function import (
    Function,
    BaseInterpreter,
    get_default_interpreter,
    set_default_interpreter
)
from .real import RealInterpreter
from .titanic import TitanicInterpreter

set_default_interpreter(TitanicInterpreter())
