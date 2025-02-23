"""
Profiler for numerical accuracy.
"""

from typing import Any, Optional

from ..function import Function, Interpreter, get_default_interpreter
from .interpreter import RealInterpreter

class FunctionProfiler:
    """
    Per-expression profiler

    Profiles each expression in a function for its numerical accuracy
    on a set of inputs.
    """

    def profile(
        self,
        func: Function,
        inputs: list[Any],
        interpreter: Optional[Interpreter] = None,
    ):
        """
        Profile the function.

        If no interpreter is provided, the default interpreter is used.
        """
        # if interpreter is None:
        #     interpreter = get_default_interpreter()
        # ref_interpreter = RealInterpreter()

        # TODO: howww?
        raise NotImplementedError
