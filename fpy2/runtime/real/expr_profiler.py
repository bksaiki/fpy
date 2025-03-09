"""
Profiler for numerical accuracy.
"""

import math
from typing import Any, Optional

from ..function import Function, Interpreter, get_default_interpreter
from .interpreter import RealInterpreter
from .rival_manager import PrecisionLimitExceeded
from .expr_trace import ExprTraceEntry
from .error import ordinal_error

from ...ir import Expr


class ExpressionProfiler:
    """
    Per-expression profiler

    Profiles each expression in a function for its numerical accuracy
    on a set of inputs.
    """

    logging: bool

    def __init__(self, logging: bool = False):
        self.logging = logging

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

        skipped_inputs: list[Any] = []
        traces: list[list[ExprTraceEntry]] = []

        # evaluate for every input
        for input in inputs:
            try:
                # evaluate in both interpreters
                trace = ref_interpreter.trace(func, input)
                traces.append(trace)
                # log
                if self.logging:
                    print('.', end='', flush=True)
            except PrecisionLimitExceeded:
                skipped_inputs.append(input)
                if self.logging:
                    print('X', end='', flush=True)

        errors_by_expr: dict[Expr, list[float]] = {}
        for trace in traces:
            for entry in trace:
                if not isinstance(entry.value, bool):
                    fl_output = interpreter.eval_expr(entry.expr, entry.env, entry.ctx)
                    ord_err = ordinal_error(fl_output, entry.value)
                    repr_err = math.log2(ord_err + 1)
                    if entry.expr not in errors_by_expr:
                        errors_by_expr[entry.expr] = [repr_err]
                    else:
                        errors_by_expr[entry.expr].append(repr_err)

        result: dict[str, tuple[float, int]] = {}
        for e, errors in errors_by_expr.items():
            s = e.format()
            avg = sum(errors) / len(errors)
            total = len(errors)
            result[s] = (avg, total)

        return (result, len(skipped_inputs))
