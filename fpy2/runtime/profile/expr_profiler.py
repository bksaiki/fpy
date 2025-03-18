"""
Profiler for numerical accuracy.
"""

import math
import numpy as np
from typing import Any, Literal, Optional

from titanfp.arithmetic.ieee754 import Float, IEEECtx
from titanfp.arithmetic.mpmf import MPMF

from ..common import ExprTraceEntry
from ..function import Function, Interpreter, get_default_interpreter
from ..metric import ordinal_error
from ..real.interpreter import RealInterpreter
from ..real.rival_manager import PrecisionLimitExceeded

from ...ir import Expr

Verbosity = Literal["minimal", "standard", "detailed"]

class ExprProfileResult:
    """
    A class to store and analyze the errors in a set of sampled data, 
    including the skipped samples and their associated error statistics.
    """

    samples: list[Any]

    skipped_samples: list[Any]

    errors: dict[Expr, list[float]]

    def __init__(self, samples, skipped_samples, errors: dict[Expr, list[float]]):
        self.samples = samples
        self.skipped_samples = skipped_samples
        self.errors = errors

    def _compute_statistics(self, stats: list[float], verbosity: Verbosity) -> dict[str, float]:
        """
        Computes statistics based on the provided list of error values with given verbosity
        """
        np_stats = np.array(stats)
        mean = np.mean(np_stats)
        statistics = {"Mean": mean}

        if verbosity in ["standard", "detailed"]:
            statistics.update({
                "Median": np.median(np_stats),
                "Min": np.min(np_stats),
                "Max": np.max(np_stats)
            })

        if verbosity == "detailed":
            statistics.update({
                "Q1": np.percentile(np_stats, 25),
                "Q3": np.percentile(np_stats, 75),
                "Std Dev": np.std(np_stats, ddof=1)
            })

        return statistics

    def print_summary(self, verbosity: Verbosity = "standard", decimal_places = 4) -> None:
        """
        Prints a summary of the error profile including statistics for each expression.
        """

        num_samples       = len(self.samples)
        num_skipped       = len(self.skipped_samples)
        percent_samples   = ((num_samples - num_skipped) / num_samples) * 100 if num_samples > 0 else 0
        total_expressions = len(self.errors)

        # Compute mean errors and categorize expressions
        error_expr = []
        no_error_exprs = []

        for expr, eval in self.errors.items():
            mean_error = np.mean(eval).item()
            if mean_error > 0:
                error_expr.append((expr, eval, mean_error))
            else:
                no_error_exprs.append((expr, len(eval)))

        # Sort expressions by descending mean error
        error_expr.sort(key=lambda x: x[2], reverse=True)

        # Compute percentage of expressions with non-zero errors
        num_error_expr = len(error_expr)
        percent_error_expr = (num_error_expr / total_expressions) * 100 if total_expressions > 0 else 0

        print("=" * 40)
        print(" Expression Profiler Summary".center(40))
        print("=" * 40)
        print(f"Evaluated Sampled points: {num_samples - num_skipped} / {num_samples} ({percent_samples:.2f}%)")
        print(f"Expressions with errors : {num_error_expr} / {total_expressions} ({percent_error_expr:.2f}%)\n")

        if error_expr:
            print("Expressions with errors (sorted by mean error in descending order):")
            print("=" * 40)

        for idx, (expr, eval, _) in enumerate(error_expr, start=1):
            print(f"{idx}. {expr.format()}")
            print(f"  Number of evaluations: {len(eval)}")

            if eval:
                statistics = self._compute_statistics(eval, verbosity)

                print("  Error Stats:")

                # Enforce ordering of error keys
                ordered_keys = ["Min", "Q1", "Median", "Mean", "Q3", "Max", "Std Dev"]
                if verbosity == "standard":
                    ordered_keys = ["Min", "Median", "Mean", "Max"]
                elif verbosity == "minimal":
                    ordered_keys = ["Mean"]

                max_key_length = max(len(key) for key in ordered_keys if key in statistics)
                for key in ordered_keys:
                    if key in statistics:
                        print(f"    {key.ljust(max_key_length)} : {statistics[key]:.{decimal_places}f}") 
            else:
                print("  No evaluations available.")

            print()

        # Print expressions with zero errors at the end
        if no_error_exprs:
            print("Expressions with no errors:")
            print("=" * 40)
            for idx, (expr, eval_count) in enumerate(no_error_exprs, start = 1):
                print(f"{idx}.  {expr.format()}")
                print(f"  Number of evaluations: {eval_count}")
            print()

    def __repr__(self) -> str:
        num_samples = len(self.samples)
        num_skipped = len(self.skipped_samples)
        
        exprs_summary = [
            {"expr": expr.format(), "mean_error": round(np.mean(evals).item(), 4) if evals else None} # TODO: 4 decimal places?
            for expr, evals in self.errors.items()
        ]
        
        return str({
            "sampled": num_samples,
            "skipped": num_skipped,
            "exprs": exprs_summary
        })



class ExprProfiler:
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

        return ExprProfileResult(inputs, skipped_inputs, errors_by_expr)

    def _normalize(self, ref, fl, ctx):
        """Returns `ref` rounded to the same context as `fl`."""
        if not isinstance(fl, Float | MPMF):
            raise TypeError(f'Expected Float or MPMF for {fl}, got {type(fl)}')
        if not isinstance(fl.ctx, IEEECtx):
            raise TypeError(f'Expected IEEECtx for {fl}, got {type(fl.ctx)}')

        ref = Float._round_to_context(ref, ctx=ctx)
        fl = Float._round_to_context(fl, ctx=ctx)
        return ref, fl
