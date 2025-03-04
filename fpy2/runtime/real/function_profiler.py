"""
Profiler for numerical accuracy.
"""

import math

from typing import Any, Optional
from titanfp.arithmetic.ieee754 import Float

from ..function import Function, Interpreter, get_default_interpreter
from .interpreter import RealInterpreter
from .rival_manager import PrecisionLimitExceeded

from .error import ordinal_error

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
        ref_interpreter = RealInterpreter(logging=True)

        skipped_inputs: list[Any] = []
        fl_outputs: list[Any] = []
        ref_outputs: list[Any] = []

        # evaluate for every input
        for input in inputs:
            try:
                # evaluate in both interpreters
                ref_output = ref_interpreter.eval(func, input)
                fl_output = interpreter.eval(func, input)
                # add to set of points
                ref_outputs.append(ref_output)
                fl_outputs.append(self._normalize(ref_output, fl_output))
            except PrecisionLimitExceeded:
                skipped_inputs.append(input)

        # TODO: Use the math library to compute the accuracy metrics
        # Report how many points are being skipped for precision errors
        errors = []
        for fl, ref in zip(fl_outputs, ref_outputs):
            ord_err = ordinal_error(fl, ref)
            errors.append(math.log2(ord_err + 1))

        # TODO: summarize better
        if errors == []:
            return (None, len(skipped_inputs))
        else:
            avg_error = sum(errors) / len(errors)
            return (avg_error, len(skipped_inputs))


    def _normalize(self, ref, fl):
        """Returns `fl` so that it is the same type as `ref`."""
        match ref:
            case Float():
                if math.isnan(fl):
                    return Float(isnan=True, ctx=ref.ctx)
                else:
                    return Float(x=fl, ctx=ref.ctx)
            case _:
                raise NotADirectoryError(f'unexpected type {ref}')
