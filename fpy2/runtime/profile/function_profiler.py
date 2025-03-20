"""
Profiler for numerical accuracy.
"""

import math
import numpy as np

from typing import Any, Optional
from titanfp.arithmetic.ieee754 import Float, IEEECtx
from titanfp.arithmetic.mpmf import MPMF

from ..function import Function, Interpreter, get_default_interpreter
from ..metric import ordinal_error
from ..real.interpreter import RealInterpreter
from ..real.rival_manager import PrecisionLimitExceeded

class FunctionProfileResult:
    """
    Result of `FunctionProfiler::profile()`.

    Tracks recorded errors and provides statistics for convenience.
    """

    errors: list[Optional[float]]
    """errors computed or None if evaluator failed"""

    np_errors: np.typing.NDArray
    """numerical errors only"""

    invalid: bool
    """were no numerical errors computed"""

    def __init__(self, errors: list[Optional[float]]):
        self.errors = list(errors)
        self.np_errors = np.array(tuple(filter(lambda e: e is not None, self.errors)))
        self.invalid = all(map(lambda e: e is None, errors))

    def average(self):
        if self.invalid:
            raise ValueError('cannot average: all evaluations failed to produce a result')
        return float(np.average(self.np_errors))

    def min(self):
        if self.invalid:
            raise ValueError('cannot average: all evaluations failed to produce a result')
        return float(np.min(self.np_errors))

    def max(self):
        if self.invalid:
            raise ValueError('cannot average: all evaluations failed to produce a result')
        return float(np.max(self.np_errors))

    def sample_size(self):
        return len(self.errors)



class FunctionProfiler:
    """
    Function profiler.

    Profiles a function's numerical accuracy on a set of inputs.
    Compare the actual output against the real number result.
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
        """Profile the function on a list of input points"""
        # select the interpreter
        if self.interpreter is None:
            interpreter = get_default_interpreter()
        else:
            interpreter = self.interpreter

        # select the reference function
        if 'spec' in func.ir.ctx and isinstance(func.ir.ctx['spec'], Function):
            ref_fn = func.ir.ctx['spec']
        else:
            ref_fn = func

        # evaluate for every input and compute error if possible
        errors: list[Optional[float]] = []
        for input in inputs:
            try:
                # evaluate in both interpreters
                ref_output = self.reference.eval(ref_fn, input)
                fl_output = interpreter.eval(func, input)
                # cast to expected type
                ref_output = self._normalize(ref_output, fl_output)
                # compute errors
                ord_err = ordinal_error(fl_output, ref_output)
                log_ord_err = math.log2(ord_err + 1)
                # append to errors
                errors.append(log_ord_err)
                if self.logging:
                    print('.', end='', flush=True)
            except PrecisionLimitExceeded:
                errors.append(None)
                if self.logging:
                    print('X', end='', flush=True)

        return FunctionProfileResult(errors)


    def _normalize(self, ref, fl):
        """Returns `ref` rounded to the same context as `fl`."""
        if not isinstance(fl, Float | MPMF):
            raise TypeError(f'Expected Float or MPMF for {fl}, got {type(fl)}')
        if not isinstance(fl.ctx, IEEECtx):
            raise TypeError(f'Expected IEEECtx for {fl}, got {type(fl.ctx)}')
        return Float(ref, ctx=fl.ctx)
