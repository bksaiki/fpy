"""
Profiler for numerical accuracy.
"""

import math
from typing import Any, Optional

from titanfp.arithmetic.ieee754 import Float, IEEECtx
from titanfp.arithmetic.mpmf import MPMF

from ..function import Function, Interpreter, get_default_interpreter
from .interpreter import RealInterpreter
from .rival_manager import PrecisionLimitExceeded
from .error import ordinal_error

from ..expr_trace import ExprTraceEntry
from ...ir import Expr


class ExpressionProfiler:
    """
    Per-expression profiler

    Profiles each expression in a function for its numerical accuracy
    on a set of inputs.
    """

    interpreter: Optional[Interpreter]
    """the interpreter to use"""

    reference: Interpreter
    """the reference interpreter to use"""

    logging: bool
    """is logging enabled?"""

    def __init__(
        self,
        *,
        interpreter: Optional[Interpreter] = None,
        reference: Optional[Interpreter] = None,
        logging: bool = False
    ):
        """
        If no interpreter is provided, the default interpreter is used.
        If no reference interpreter is provided, the `RealInterpreter` is used.
        """
        if reference is None:
            reference = RealInterpreter()

        self.interpreter = interpreter
        self.reference = reference
        self.logging = logging

    def profile(self, func: Function, inputs: list[Any]):
        """
        Profile the function.

        If no interpreter is provided, the default interpreter is used.
        """
        # select the interpreter
        if self.interpreter is None:
            interpreter = get_default_interpreter()
        else:
            interpreter = self.interpreter

        skipped_inputs: list[Any] = []
        traces: list[list[ExprTraceEntry]] = []

        # evaluate for every input
        for input in inputs:
            try:
                # evaluate in both interpreters
                _, trace = self.reference.eval_with_trace(func, input)
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
                    ref_output, fl_output = self._normalize(entry.value, fl_output, entry.ctx)
                    ord_err = ordinal_error(ref_output, fl_output)
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

    def _normalize(self, ref, fl, ctx):
        """Returns `ref` rounded to the same context as `fl`."""
        if not isinstance(fl, Float | MPMF):
            raise TypeError(f'Expected Float or MPMF for {fl}, got {type(fl)}')
        if not isinstance(fl.ctx, IEEECtx):
            raise TypeError(f'Expected IEEECtx for {fl}, got {type(fl.ctx)}')

        ref = Float._round_to_context(ref, ctx=ctx)
        fl = Float._round_to_context(fl, ctx=ctx)
        return ref, fl
