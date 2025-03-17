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

        skipped_inputs: list[Any] = []
        fl_outputs: list[Any] = []
        ref_outputs: list[Any] = []

        # evaluate for every input
        for input in inputs:
            try:
                # evaluate in both interpreters
                ref_output = self.reference.eval(func, input)
                fl_output = interpreter.eval(func, input)
                # add to set of points
                ref_outputs.append(self._normalize(ref_output, fl_output))
                fl_outputs.append(fl_output)
                if self.logging:
                    print('.', end='', flush=True)
            except PrecisionLimitExceeded:
                skipped_inputs.append(input)
                if self.logging:
                    print('X', end='', flush=True)

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
            return (errors, len(skipped_inputs))


    def _normalize(self, ref, fl):
        """Returns `ref` rounded to the same context as `fl`."""
        if not isinstance(fl, Float | MPMF):
            raise TypeError(f'Expected Float or MPMF for {fl}, got {type(fl)}')
        if not isinstance(fl.ctx, IEEECtx):
            raise TypeError(f'Expected IEEECtx for {fl}, got {type(fl.ctx)}')
        return Float(ref, ctx=fl.ctx)
