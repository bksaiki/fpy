"""
This module defines an FPy interpreter that uses the Rival interval library
to compute the true real number result.
"""

from typing import Any, Optional, Sequence, TypeAlias

from titanfp.arithmetic.evalctx import EvalCtx
from titanfp.arithmetic.ieee754 import ieee_ctx
from titanfp.titanic.ndarray import NDArray
from titanfp.titanic.digital import Digital
from titanfp.titanic import gmpmath

from .interval import BoolInterval, RealInterval

from ..function import Interpreter, Function
from ...ir import *

from titanfp.titanic.gmpmath import mpfr_to_digital, mpfr
from fpy2.runtime.real.interval import RealInterval
from .rival_manager import RivalManager


ScalarVal: TypeAlias = BoolInterval | RealInterval
"""Type of scalar values in FPy programs."""
TensorVal: TypeAlias = NDArray
"""Type of tensor values in FPy programs."""

ScalarArg: TypeAlias = ScalarVal | str | int | float
"""Type of scalar arguments in FPy programs; includes native Python types"""
TensorArg: TypeAlias = NDArray | tuple | list
"""Type of tensor arguments in FPy programs; includes native Python types"""

"""Maps python operator to the corresponding operator in Rival"""
_method_table: dict[str, str] = {
    '+': '+',
    '-': '-',
    '*': '*',
    '/': '/',
    'fabs': 'fabs',
    'sqrt': 'sqrt',
    'fma': 'fma',
    'neg': 'neg',
    'copysign': 'copysign',
    'fdim': 'fdim',
    'fmax': 'fmax',
    'fmin': 'fmin',
    'fmod': 'fmod',
    'remainder': 'remainder',
    'hypot': 'hypot',
    'cbrt': 'cbrt',
    'ceil': 'ceil',
    'floor': 'floor',
    'nearbyint': 'nearbyint',
    'round': 'round',
    'trunc': 'trunc',
    'acos': 'acos',
    'asin': 'asin',
    'atan': 'atan',
    'atan2': 'atan2',
    'cos': 'cos',
    'sin': 'sin',
    'tan': 'tan',
    'acosh': 'acosh',
    'asinh': 'asinh',
    'atanh': 'atanh',
    'cosh': 'cosh',
    'sinh': 'sinh',
    'tanh': 'tanh',
    'exp': 'exp',
    'exp2': 'exp2',
    'expm1': 'expm1',
    'log': 'log',
    'log10': 'log10',
    'log1p': 'log1p',
    'log2': 'log2',
    'pow': 'pow',
    'erf': 'erf',
    'erfc': 'erfc',
    'lgamma': 'lgamma',
    'tgamma': 'tgamma',
    'isfinite': 'isfinite',
    'isinf': 'isinf',
    'isnan': 'isnan',
    'isnormal': 'isnormal',
    'signbit': 'signbit',
}


class _Interpreter(ReduceVisitor):
    """Single-use real number interpreter"""
    env: dict[NamedId, BoolInterval | RealInterval | NDArray]
    
    def __init__(self, p = 4):
        self.env = {}
        self.rival = RivalManager()
        self.p = p
        self.rival.set_precision(p)

    def _digital_to_ival(self, x: Digital):
        """
        Converts `x` into an interval that represents the rounding envelope of `x`,
        i.e., the tightest set of values that would round to `x` at the
        current precision of `x`.
        """
        y = x.round_new(max_p=x.p + 1)  # increase precision by 1
        prev = y.prev_float()        # next floating-point number (toward zero)
        next = y.next_float()        # previous floating-point number (away zero)
        if x.negative:
            return RealInterval(next, prev)
        else:
            return RealInterval(prev, next)

    def _real_to_digital(self, x: str | int | float, prec: int):
        """
        Converts `x` into a `Digital` type without loss of accuracy.
        Raises an exception if `x` cannot be represented exactly at the given precision.
        """
        rto_round = mpfr(x, prec=prec)
        rounded = mpfr_to_digital(rto_round).round_new(max_p=prec)
        if rounded.inexact:
            raise ValueError(f"cannot represent {x} exactly at precision {prec}")
        return rounded

    # TODO: what are the semantics of arguments
    def _arg_to_mpmf(self, arg: Any, ctx: EvalCtx):
        if isinstance(arg, str | int | float):
            x = gmpmath.mpfr(arg, ctx.p)
            return gmpmath.mpfr_to_digital(x)
        elif isinstance(arg, int):
            return Digital(m=arg, exp=0)
        elif isinstance(arg, float):
            x = gmpmath.mpfr(arg, 53)
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
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = RealInterval.from_val(x)
                case RealType():
                    x = self._arg_to_mpmf(val, ctx)
                    if not isinstance(x, RealInterval):
                        raise TypeError(f'Expected real value, got {val}')
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = RealInterval.from_val(x)
                case _:
                    raise NotImplementedError(f'unsupported argument type {arg.ty}')

        return self._visit_block(func.body, ctx)


    def _lookup(self, name: NamedId):
        if name not in self.env:
            raise RuntimeError(f'unbound variable {name}')
        return self.env[name]

    def _visit_var(self, e: Var, ctx: EvalCtx):
        return self._lookup(e.name)

    def _visit_bool(self, e: Bool, ctx: Any):
        return e.val

    def _visit_decnum(self, e: Decnum, ctx: EvalCtx):
        return self.rival.eval_expr(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: EvalCtx):
        return self.rival.eval_expr(e.val)

    def _visit_integer(self, e: Integer, ctx: EvalCtx):
        return self.rival.eval_expr(e.val)

    def _visit_rational(self, e: Rational, ctx: EvalCtx):
        return self.rival.eval_expr(e.val)

    def _visit_digits(self, e: Digits, ctx: EvalCtx):
        self.rival.define_function(f'(f m e b) (* m (pow b e))')
        return self.rival.eval_expr(f"f {e.m} {e.e} {e.b}")

    def _visit_nary_expr(self, e: NaryExpr, ctx: EvalCtx):
        if e.name in _method_table:
            return self._apply_method(e, ctx)
        elif isinstance(e, Cast):
            return self._visit_expr(e.children[0], ctx)
        elif isinstance(e, Not):
            return self._apply_not(e, ctx)
        elif isinstance(e, And):
            return self._apply_and(e, ctx)
        elif isinstance(e, Or):
            return self._apply_or(e, ctx)
        else:
            raise NotImplementedError('unknown n-ary expression', e)
    
    def _apply_method(self, e: NaryExpr, ctx: EvalCtx):
        fn = _method_table[e.name]
        args: list[RealInterval] = []
        for arg in e.children:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, RealInterval):
                raise TypeError(f'expected a real number argument for {e.name}, got {val}')
            args.append(val)

        arg_names = [f"arg{i+1}" for i in range(len(args))]
        arg_list_str = " ".join(arg_names)
        arg_values_str = " ".join(map(str, args))
        function_def = f"(f {arg_list_str}) ({fn} {arg_list_str})"
        function_call = f"f {arg_values_str}"

        # TODO: debug function call
        print(function_def)
        print(function_call)

        self.rival.define_function(function_def)
        return self.rival.eval_expr(function_call)

        
    def _apply_not(self, e: Not, ctx: EvalCtx):
        arg = self._visit_expr(e.children[0], ctx)
        if not isinstance(arg, bool):
            raise TypeError(f'expected a boolean argument, got {arg}')
        return not arg

    def _apply_and(self, e: And, ctx: EvalCtx):
        args: list[bool] = []
        for arg in e.children:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean argument, got {val}')
            args.append(val)
        return all(args)

    def _apply_or(self, e: Or, ctx: EvalCtx):
        args: list[bool] = []
        for arg in e.children:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean argument, got {val}')
            args.append(val)
        return any(args)

    def _visit_compare(self, e: Compare, ctx: EvalCtx):
        lhs = self._visit_expr(e.children[0], ctx)
        for op, arg in zip(e.ops, e.children[1:]):
            rhs = self._visit_expr(arg, ctx)
            if not self._apply_cmp2(op, lhs, rhs):
                return False
            lhs = rhs
        return True
    
    def _apply_cmp2(self, op: CompareOp, lhs, rhs):
        op_map = {
            CompareOp.EQ: "==",
            CompareOp.NE: "!=",
            CompareOp.LT: "<",
            CompareOp.LE: "<=",
            CompareOp.GT: ">",
            CompareOp.GE: ">="
        }
        
        if op not in op_map:
            raise NotImplementedError("unknown comparison operator", op)

        self.rival.define_function(f"(f x y) ({op_map[op]} x y)")
        return self.rival.eval_expr(f"f {lhs} {rhs}")


    def _visit_if_expr(self, e: IfExpr, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_var_assign(self, stmt: VarAssign, ctx: EvalCtx):
        val = self._visit_expr(stmt.expr, ctx)
        match stmt.var:
            case NamedId():
                self.env[stmt.var] = val
            case UnderscoreId():
                pass
            case _:
                raise NotImplementedError('unknown variable', stmt.var)

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_if_stmt(self, stmt: IfStmt, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_context(self, stmt: ContextStmt, ctx: EvalCtx):
        return self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: EvalCtx):
        test = self._visit_expr(stmt.test, ctx)
        if not isinstance(test, bool):
            raise TypeError(f'expected a boolean, got {test}')
        if not test:
            raise AssertionError(stmt.msg)
        return ctx

    def _visit_return(self, stmt: Return, ctx: EvalCtx):
        return self._visit_expr(stmt.expr, ctx)

    def _visit_block(self, block: Block, ctx: EvalCtx):
        for stmt in block.stmts:
            if isinstance(stmt, Return):
                return self._visit_return(stmt, ctx)
            else:
                ctx = self._visit_statement(stmt, ctx)

        return None

    
    # Currently have no plan to implement functionalities below
    def _visit_comp_expr(self, e: CompExpr, ctx: EvalCtx):
        raise NotImplementedError
    
    def _visit_unknown(self, e: UnknownCall, ctx: EvalCtx):
        raise NotImplementedError
    
    def _visit_constant(self, e: Constant, ctx: EvalCtx):
        raise NotImplementedError
    
    def _visit_tuple_expr(self, e: TupleExpr, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_tuple_ref(self, e: TupleRef, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_tuple_set(self, e: TupleSet, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_ref_assign(self, stmt: RefAssign, ctx: EvalCtx):
        raise NotImplementedError
    
    def _visit_while_stmt(self, stmt: WhileStmt, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_for_stmt(self, stmt: ForStmt, ctx: EvalCtx):
        raise NotImplementedError
    
    def _visit_phis(self, phis: list[PhiNode], lctx: EvalCtx, rctx: EvalCtx):
        raise NotImplementedError

    def _visit_loop_phis(self, phis: list[PhiNode], lctx: EvalCtx, rctx: EvalCtx):
        raise NotImplementedError

    def _visit_function(self, func: FunctionDef, ctx: EvalCtx):
        raise NotImplementedError


class RealInterpreter(Interpreter):
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
