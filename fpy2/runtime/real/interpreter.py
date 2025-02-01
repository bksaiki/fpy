"""
This module defines an FPy interpreter that uses the Rival interval library
to compute the true real number result.
"""

from titanfp.arithmetic.evalctx import EvalCtx
from titanfp.arithmetic.ieee754 import ieee_ctx
from titanfp.titanic.ndarray import NDArray
from titanfp.titanic.digital import Digital
from titanfp.titanic import gmpmath

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
    env: dict[str, BoolInterval | RealInterval | NDArray]

    # TODO: what are the semantics of arguments
    def _arg_to_mpmf(self, arg: Any, ctx: EvalCtx):
        if isinstance(arg, str | int | float):
            x = gmpmath.mpfr(arg, ctx.p)
            return gmpmath.mpfr_to_digital(x)
        elif isinstance(arg, Digital):
            x = gmpmath.mpfr(arg, ctx.p)
            return gmpmath.mpfr_to_digital(x)
        elif isinstance(arg, tuple | list):
            raise NotImplementedError()
        else:
            raise NotImplementedError(f'unknown argument type {arg}')

    def eval(self,
        func: FunctionDef,
        args: Sequence[Any],
        ctx: Optional[EvalCtx] = None
    ):
        if not isinstance(func, FunctionDef):
            raise TypeError(f'Expected Function, got {type(func)}')

        # check arity
        args = tuple(args)
        if len(args) != len(func.args):
            raise TypeError(f'Expected {len(func.args)} arguments, got {len(args)}')

        # default context if none is specified
        if ctx is None:
            ctx = ieee_ctx(11, 64)

        for val, arg in zip(args, func.args):
            match arg.ty:
                case AnyType():
                    x = self._arg_to_mpmf(val, ctx)
                    self.env[arg.name] = RealInterval.from_val(x)
                case RealType():
                    x = self._arg_to_mpmf(val, ctx)
                    if not isinstance(x, RealInterval):
                        raise TypeError(f'Expected real value, got {val}')
                    self.env[arg.name] = RealInterval.from_val(x)
                case _:
                    raise NotImplementedError(f'unsupported argument type {arg.ty}')

        return self._visit_block(func.body, ctx)


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
