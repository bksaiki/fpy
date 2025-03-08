"""
Profiler for numerical accuracy.
"""

from typing import Any, Optional

from ..function import Function, Interpreter, get_default_interpreter
from .interpreter import RealInterpreter
from ..titanic import TitanicInterpreter
from .rival_manager import PrecisionLimitExceeded
from .expr_trace import ExprTrace
from .error import ordinal_error
import math



class ExpressionProfiler:
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
        if interpreter is None:
            interpreter = TitanicInterpreter() # default interpreter is returning RealInterpreter
        ref_interpreter = RealInterpreter()

        skipped_inputs: list[Any] = []
        fl_outputs: list[Any] = []
        ref_outputs: list[Any] = []

        # evaluate for every input
        for input in inputs:
            try:
                # evaluate in both interpreters
                ref_output = ref_interpreter.trace(func, input)
                fl_output = interpreter.eval_expr(ref_output)

                # add to set of points
                ref_outputs.append(ref_output)
                fl_outputs.append(fl_output)

            except PrecisionLimitExceeded:
                skipped_inputs.append(input)

        # Aggreagte result by expression id
        result = self._group_expr_traces(ref_output, fl_output)

        # Compute average ordinal errors
        for k, v in result.items():
            errors = []
            for val in v["values"]:
                # TODO: what to do with booleans
                if isinstance(val[0], bool): continue
                ord_err = ordinal_error(val[1], val[0])
                errors.append(math.log2(ord_err + 1))

            # TODO
            if len(errors) == 0: continue
            result[k]["errors"] = sum(errors) / len(errors)
        
        return result

    def _group_expr_traces(self, traces1, traces2):
        result = {}
        assert len(traces1) == len(traces2)

        for i in range(len(traces1)):
            eid = id(traces1[i].expr)
            if eid not in result:
                result[eid] = {
                    "expression": traces1[i].expr,
                    "values": []
                }
            result[eid]["values"].append((traces1[i].value, traces2[i].value))
        return result