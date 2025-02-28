"""
Profiler for numerical accuracy.
"""

from typing import Any, Optional

from ..function import Function, Interpreter, get_default_interpreter
from .interpreter import RealInterpreter
from .rival_manager import PrecisionLimitExceeded

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
            try:
                fl_output = interpreter.eval(func, input)
                ref_output = ref_interpreter.eval(func, input)
                fl_outputs.append(fl_output)
                ref_outputs.append(ref_output)
            except PrecisionLimitExceeded as e:
                pass

        # TODO: compute accuracy metrics
        assert len(fl_outputs) == len(ref_outputs)

        # TODO: Use the math library to compute the accuracy metrics
        # Report how many points are being skipped for precision errors
        errors = []
        for fl, ref in zip(fl_outputs, ref_outputs):
            errors.append(abs(float(fl) - float(ref)) / (abs(float(ref)) + 1e-8) * 100)

        accuracy = 1.0 - sum(errors) / len(errors)
        return accuracy

