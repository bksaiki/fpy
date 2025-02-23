"""
Profiler for numerical accuracy.
"""

from typing import Any, Optional

from ..function import Function, Interpreter, get_default_interpreter
from .interpreter import RealInterpreter

class FunctionProfiler:
    """
    Function profiler.

    Profiles a function's numerical accuracy on a set of inputs.
    Compare the actual output against the real number result.
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
        if interpreter is None:
            interpreter = get_default_interpreter()
        ref_interpreter = RealInterpreter()

        # TODO: what happens when Rival fails
        fl_outputs: list[Any] = []
        ref_outputs: list[Any] = []
        for input in inputs:
            fl_output = interpreter.eval(func, input)
            ref_output = ref_interpreter.eval(func, input)
            fl_outputs.append(fl_output)
            ref_outputs.append(ref_output)

        # TODO: compute accuracy metrics

