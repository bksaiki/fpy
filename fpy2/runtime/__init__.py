from .env import PythonEnv
from .function import Function, BaseInterpreter, get_default_interpreter, set_default_interpreter
from .interpreter import Interpreter

set_default_interpreter(Interpreter())
