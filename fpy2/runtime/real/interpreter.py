"""
This module defines an FPy interpreter that uses the Rival interval library
to compute the true real number result.
"""

from titanfp.arithmetic.evalctx import EvalCtx
from titanfp.titanic.ndarray import NDArray

from .interval import BoolInterval, RealInterval

from ..function import BaseInterpreter, Function
from ...ir import *


ScalarVal = BoolInterval | RealInterval
"""Type of scalar values in FPy programs."""
TensorVal = NDArray
"""Type of tensor values in FPy programs."""

ScalarArg = ScalarVal | str | int | float
"""Type of scalar arguments in FPy programs; includes native Python types"""
TensorArg = NDArray | tuple | list
"""Type of tensor arguments in FPy programs; includes native Python types"""


class _Interpreter(ReduceVisitor):
    """Single-use real number interpreter"""

    def eval(self,
        func: FunctionDef,
        args: Sequence[Any],
        ctx: Optional[EvalCtx] = None
    ):
        if not isinstance(func, FunctionDef):
            raise TypeError(f'Expected Function, got {type(func)}')
        raise NotImplementedError()

    def _visit_var(self, e, ctx):
        raise NotImplementedError

    def _visit_decnum(self, e, ctx):
        raise NotImplementedError

    def _visit_hexnum(self, e, ctx):
        raise NotImplementedError

    def _visit_integer(self, e, ctx):
        raise NotImplementedError

    def _visit_rational(self, e, ctx):
        raise NotImplementedError

    def _visit_constant(self, e, ctx):
        raise NotImplementedError

    def _visit_digits(self, e, ctx):
        raise NotImplementedError

    def _visit_unknown(self, e, ctx):
        raise NotImplementedError

    def _visit_nary_expr(self, e, ctx):
        raise NotImplementedError

    def _visit_compare(self, e, ctx):
        raise NotImplementedError

    def _visit_tuple_expr(self, e, ctx):
        raise NotImplementedError

    def _visit_tuple_ref(self, e, ctx):
        raise NotImplementedError

    def _visit_tuple_set(self, e, ctx):
        raise NotImplementedError

    def _visit_comp_expr(self, e, ctx):
        raise NotImplementedError

    def _visit_if_expr(self, e, ctx):
        raise NotImplementedError

    def _visit_var_assign(self, stmt, ctx):
        raise NotImplementedError

    def _visit_tuple_assign(self, stmt, ctx):
        raise NotImplementedError

    def _visit_ref_assign(self, stmt, ctx):
        raise NotImplementedError

    def _visit_if1_stmt(self, stmt, ctx):
        raise NotImplementedError

    def _visit_if_stmt(self, stmt, ctx):
        raise NotImplementedError

    def _visit_while_stmt(self, stmt, ctx):
        raise NotImplementedError

    def _visit_for_stmt(self, stmt, ctx):
        raise NotImplementedError

    def _visit_context(self, stmt, ctx):
        raise NotImplementedError

    def _visit_return(self, stmt, ctx):
        raise NotImplementedError

    def _visit_phis(self, phis, lctx, rctx):
        raise NotImplementedError

    def _visit_loop_phis(self, phis, lctx, rctx):
        raise NotImplementedError

    def _visit_block(self, block, ctx):
        raise NotImplementedError

    def _visit_function(self, func, ctx):
        raise NotImplementedError
 
 

class RealInterpreter(BaseInterpreter):
    """
    Real-number interpreter for FPy functions.

    Computes the true real number result of a function,
    rounded to the nearest floating-point value at some precision.
    This interpreter leverages the Rival interval library developed by the Herbie project.
    More information on the Rival library and the Herbie project can
    be found here: https://herbie.uwplse.org/.
    """

    def eval(
        self,
        func: Function,
        args: Sequence[Any],
        ctx: Optional[EvalCtx] = None
    ):
        if not isinstance(func, Function):
            raise TypeError(f'Expected Function, got {func}')
        return _Interpreter().eval(func.ir, args, ctx)
