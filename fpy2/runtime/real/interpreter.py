"""
This module defines an FPy interpreter that uses the Rival interval library
to compute the true real number result.
"""

from fractions import Fraction

from typing import Any, Optional, Sequence, TypeAlias

from titanfp.arithmetic.evalctx import EvalCtx, determine_ctx
from titanfp.arithmetic.ieee754 import Float, ieee_ctx
from titanfp.titanic.ndarray import NDArray
from titanfp.titanic.digital import Digital
from titanfp.titanic.ops import RM

from .interval import BoolInterval, RealInterval
from .rival_manager import RivalManager, InsufficientPrecisionError

from ..function import Interpreter, Function, FunctionReturnException
from ...ir import *

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
    env: dict[NamedId, str | RealInterval | bool]
    
    def __init__(self, logging: bool = False):
        self.env = {}
        self.rival = RivalManager(logging)
        self.rival.set_print_ival(True)

    def _arg_to_mpmf(self, arg: Any, ctx: EvalCtx):
        if isinstance(arg, str | int | float | Digital):
            return str(arg)
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
        ctx = determine_ctx(ctx, func.ctx)
        self.rival.set_precision(ctx.p)

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
                        self.env[arg.name] = x
                case _:
                    raise NotImplementedError(f'unsupported argument type {arg.ty}')

        try:
            self._visit_block(func.body, ctx)
            raise RuntimeError('no return statement encountered')
        except FunctionReturnException as e:
            return e.value
        except InsufficientPrecisionError as e:
            print(f"Insufficient precision, retrying with p={e.prec}")
            raise e


    def _lookup(self, name: NamedId):
        if name not in self.env:
            raise RuntimeError(f'unbound variable {name}')
        return name.base # We return the name rather than the value in expression

    def _visit_var(self, e: Var, ctx: EvalCtx):
        return e.name

    def _visit_bool(self, e: Bool, ctx: Any):
        return e.val

    def _visit_decnum(self, e: Decnum, ctx: EvalCtx):
        return str(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: EvalCtx):
        return str(e.val)

    def _visit_integer(self, e: Integer, ctx: EvalCtx):
        return str(e.val)

    def _visit_rational(self, e: Rational, ctx: EvalCtx):
        return f'{e.p}/{e.q}'

    def _visit_digits(self, e: Digits, ctx: EvalCtx):
        x = Fraction(e.b) ** e.e
        return str(e.m * x)

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
        args = [self._visit_expr(arg, ctx) for arg in e.children]
        arg_values_str = " ".join(map(str, args))
        return f"({fn} {arg_values_str})"

    def _apply_not(self, e: Not, ctx: EvalCtx):
        arg = self._visit_expr(e.children[0], ctx)
        return f'(not {arg})'

    def _nary_to_2ary(self, op: str, args: list) -> str:
        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return f'({op} {" ".join(args)})'
        else:
            return f'({op} {self._nary_to_2ary(op, args[:-1])} {args[-1]})'

    def _apply_and(self, e: And, ctx: EvalCtx):
        args = [self._visit_expr(arg, ctx) for arg in e.children]
        return self._nary_to_2ary('and', args)

    def _apply_or(self, e: Or, ctx: EvalCtx):
        args = [self._visit_expr(arg, ctx) for arg in e.children]
        return self._nary_to_2ary('or', args)

    def _apply_cmp2(self, op: CompareOp, lhs, rhs):
        match op:
            case CompareOp.EQ:
                return f'(== {lhs} {rhs})'
            case CompareOp.NE:
                return f'(!= {lhs} {rhs})'
            case CompareOp.LT:
                return f'(< {lhs} {rhs})'
            case CompareOp.LE:
                return f'(<= {lhs} {rhs})'
            case CompareOp.GT:
                return f'(> {lhs} {rhs})'
            case CompareOp.GE:
                return f'(>= {lhs} {rhs})'
            case _:
                raise NotImplementedError('unknown comparison operator', op)

    def _visit_compare(self, e: Compare, ctx: EvalCtx):
        args: list[str] = []
        lhs = self._visit_expr(e.children[0], ctx)
        for op, arg in zip(e.ops, e.children[1:]):
            rhs = self._visit_expr(arg, ctx)
            args.append(self._apply_cmp2(op, lhs, rhs))
            lhs = rhs
        return self._nary_to_2ary('and', args)

    def _visit_if_expr(self, e: IfExpr, ctx: EvalCtx):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return f'(if {cond} {ift} {iff})'

    def _visit_var_assign(self, stmt: VarAssign, ctx: EvalCtx):
        match stmt.var:
            case NamedId():
                self.env[stmt.var] = self._visit_rival(stmt.expr, ctx)
            case UnderscoreId():
                pass
            case _:
                raise NotImplementedError('unknown variable', stmt.var)

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: EvalCtx):
        cond = self._visit_expr_top(stmt.cond, ctx, force=True)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        elif cond:
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
        else:
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.lhs]

    def _visit_if_stmt(self, stmt: IfStmt, ctx: EvalCtx):
        cond = self._visit_expr_top(stmt.cond, ctx, force=True)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        elif cond:
            self._visit_block(stmt.ift, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.lhs]
        else:
            self._visit_block(stmt.iff, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]

    def _visit_context(self, stmt: ContextStmt, ctx: EvalCtx):
        ctx = determine_ctx(ctx, stmt.props)
        return self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: EvalCtx):
        test = self._visit_expr(stmt.test, ctx)
        if not isinstance(test, bool):
            raise TypeError(f'expected a boolean, got {test}')
        if not test:
            raise AssertionError(stmt.msg)
        return ctx

    def _interval_to_real(self, val: RealInterval) -> float:
        # round the endpoints inwards, see if they converge
        lo = Float(x=val.lo, ctx=ieee_ctx(11, 64, RM.RAZ))
        hi = Float(x=val.hi, ctx=ieee_ctx(11, 64, RM.RTZ))
        if lo != hi:
            raise ValueError(f'interval {val} did not converge')
        else:
            return lo

    def _visit_rival(self, expr: Expr, ctx: EvalCtx):
        val = self._visit_expr(expr, ctx)
        if isinstance(val, NamedId):
            # variable name
            return self.env[val]
        elif isinstance(val, bool):
            # boolean
            return val
        elif val.startswith('('):
            # expression to be evaluated by Rival
            self.rival.define_function(f"(f {' '.join(map(str, self.env.keys()))}) {val}")
            return self.rival.eval_expr(f"f {' '.join(map(str, self.env.values()))}")
        else:
            # numerical constant
            return val

    def _visit_expr_top(self, expr: Expr, ctx: EvalCtx, force: bool = False) -> bool | float:
        val = self._visit_rival(expr, ctx)
        if not force:
            return val
        else:
            match val:
                case bool():
                    return val
                case str():
                    val = self.rival.eval_expr(val)
                    if isinstance(val, RealInterval):
                        return self._interval_to_real(val)
                    else:
                        return val
                case RealInterval():
                    return self._interval_to_real(val)
                case _:
                    raise NotImplementedError('unreachable', val)

    def _visit_return(self, stmt: Return, ctx: EvalCtx) -> bool | float:
        # since we are returning we actually want a value
        return self._visit_expr_top(stmt.expr, ctx, force=True)

    def _visit_block(self, block: Block, ctx: EvalCtx):
        for stmt in block.stmts:
            if isinstance(stmt, Return):
                v = self._visit_return(stmt, ctx)
                raise FunctionReturnException(v)
            else:
                ctx = self._visit_statement(stmt, ctx)

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: EvalCtx) -> None:
        for phi in stmt.phis:
            self.env[phi.name] = self.env[phi.lhs]
            del self.env[phi.lhs]

        cond = self._visit_expr_top(stmt.cond, ctx, force=True)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')

        while cond:
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
                del self.env[phi.rhs]

            cond = self._visit_expr_top(stmt.cond, ctx, force=True)
            if not isinstance(cond, bool):
                raise TypeError(f'expected a boolean, got {cond}')

    # override for typing
    def _visit_expr(self, e: Expr, ctx: EvalCtx) -> NamedId | str | bool:
        return super()._visit_expr(e, ctx)

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
    
    logging: bool
    """enable logging?"""

    def __init__(self, logging: bool = False):
        self.logging = logging

    def eval(
        self,
        func: Function,
        args: Sequence[Any],
        ctx: Optional[EvalCtx] = None
    ):
        if not isinstance(func, Function):
            raise TypeError(f'Expected Function, got {func}')
        return _Interpreter(logging=self.logging).eval(func.ir, args, ctx)
