"""
Profiler for numerical accuracy.
"""

from typing import Any, Optional

from ..function import Function, Interpreter, get_default_interpreter
from .interpreter import RealInterpreter
from ..titanic import TitanicInterpreter
from .rival_manager import PrecisionLimitExceeded
from .expr_trace import ExprTrace



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

        # TODO: What format do we prefer? 
        # Currently it is 2 list of ExprTrace objects which feels redundant.
        # Perhaps ExprTrace should have 2 values field? 
        # Or we just save the value from fl_output without the trace
        for i in range(len(ref_outputs)):
            print(i, ref_outputs[0][i].value, fl_outputs[0][i].value)

        # TODO: Aggregate across location??

        # TODO: compute error
