"""
FPy runtime backed by the Python runtime.
"""

import math
import titanfp.titanic.gmpmath as gmpmath

from typing import Any, Callable, Optional, Sequence, TypeAlias

from titanfp.arithmetic.evalctx import EvalCtx, determine_ctx
from titanfp.arithmetic.ieee754 import Float, IEEECtx, ieee_ctx
from titanfp.titanic.digital import Digital
from titanfp.titanic.ops import RM

from ..function import Interpreter, Function, FunctionReturnException
from ...ir import *

def _safe_div(x: float, y: float):
    if y == 0:
        if x == 0:
            return math.nan
        else:
            return math.copysign(math.inf, x)
    else:
        return x / y

ScalarVal: TypeAlias = bool | float
"""Type of scalar values."""
TensorVal: TypeAlias = tuple
"""Type of tensor values."""
_method_table: dict[str, Callable[..., Any]] = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': _safe_div,
    'fabs': math.fabs,
    'sqrt': math.sqrt,
    # TODO: only available in Python 3.13
    # 'fma': math.fma,
    'neg': lambda x: -x,
    'copysign': math.copysign,
    'fdim': lambda x, y: max(x - y, 0),
    'fmax': max,
    'fmin': min,
    'fmod': math.fmod,
    'remainder': math.remainder,
    'hypot': math.hypot,
    'cbrt': math.cbrt,
    'ceil': math.ceil,
    'floor': math.floor,
    'nearbyint': lambda x: round(x),
    'round': round,
    'trunc': math.trunc,
    'acos': math.acos,
    'asin': math.asin,
    'atan': math.atan,
    'atan2': math.atan2,
    'cos': math.cos,
    'sin': math.sin,
    'tan': math.tan,
    'acosh': math.acosh,
    'asinh': math.asinh,
    'atanh': math.atanh,
    'cosh': math.cosh,
    'sinh': math.sinh,
    'tanh': math.tanh,
    'exp': math.exp,
    'exp2': lambda x: 2 ** x,
    'expm1': math.expm1,
    'log': math.log,
    'log10': math.log10,
    'log1p': math.log1p,
    'log2': math.log2,
    'pow': math.pow,
    'erf': math.erf,
    'erfc': math.erfc,
    'lgamma': math.lgamma,
    'tgamma': math.gamma,
    'isfinite': math.isfinite,
    'isinf': math.isinf,
    'isnan': math.isnan,
    'isnormal': lambda x: math.isfinite(x) and x != 0,
    'signbit': lambda x: math.copysign(1, x) < 0,
}

_Env: TypeAlias = dict[NamedId, ScalarVal | TensorVal]

class _Interpreter(ReduceVisitor):
    """Single-use interpreter for a function."""
    func: FunctionDef
    env: _Env

    def __init__(self, ir: FunctionDef):
        self.func = ir
        self.env = {}

    def _is_python_ctx(self, ctx: EvalCtx):
        return (
            isinstance(ctx, IEEECtx)
            and ctx.es == 11
            and ctx.nbits == 64
            or ctx.rm == RM.RNE
        )

    def _arg_to_float(self, arg: Any):
        if isinstance(arg, str | int | float):
            return float(arg)
        elif isinstance(arg, Digital):
            return float(arg)
        elif isinstance(arg, tuple | list):
            raise NotImplementedError()
        else:
            raise NotImplementedError(f'unknown argument type {arg}')

    def _lookup(self, name: NamedId):
        if name not in self.env:
            raise RuntimeError(f'unbound variable {name}')
        return self.env[name]

    def eval(
        self,
        args: Sequence[Any],
        ctx: Optional[EvalCtx] = None
    ):
        args = tuple(args)
        if len(args) != len(self.func.args):
            raise TypeError(f'Expected {len(self.func.args)} arguments, got {len(args)}')

        # determine context
        if ctx is None:
            ctx = ieee_ctx(11, 64)
        ctx = determine_ctx(ctx, self.func.ctx)

        # Python only has doubles
        if not self._is_python_ctx(ctx):
            raise ValueError(f'Unsupported context {ctx}; Python only has doubles')

        # bind arguments
        for val, arg in zip(args, self.func.args):
            match arg.ty:
                case AnyType():
                    x = self._arg_to_float(val)
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case RealType():
                    x = self._arg_to_float(val)
                    if not isinstance(x, float):
                        raise NotImplementedError(f'argument is a scalar, got data {val}')
                    if isinstance(arg.name, NamedId):
                        self.env[arg.name] = x
                case _:
                    raise NotImplementedError(f'unknown argument type {arg.ty}')

        # evaluate the body
        try:
            self._visit_block(self.func.body, ctx)
            raise RuntimeError('no return statement encountered')
        except FunctionReturnException as e:
            return e.value

    def _visit_var(self, e: Var, ctx: EvalCtx):
        return self._lookup(e.name)

    def _visit_bool(self, e: Bool, ctx: EvalCtx):
        return e.val

    def _visit_decnum(self, e: Decnum, ctx: EvalCtx):
        return float(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: EvalCtx):
        return float.fromhex(e.val)

    def _visit_integer(self, e: Integer, ctx: EvalCtx):
        return float(e.val)

    def _visit_rational(self, e: Rational, ctx: EvalCtx):
        return e.p / e.q

    def _visit_constant(self, e: Constant, ctx: EvalCtx):
        # rely on Titanic for this
        x = gmpmath.compute_constant(e.val, prec=ctx.p)
        d = Float._round_to_context(x, ctx=ctx)
        return float(d)

    def _visit_digits(self, e: Digits, ctx: EvalCtx):
        # rely on Titanic for this
        x = gmpmath.compute_digits(e.m, e.e, e.b, prec=ctx.p)
        d = Float._round_to_context(x, ctx)
        return float(d)

    def _visit_unknown(self, e: UnknownCall, ctx: EvalCtx):
        raise NotImplementedError('unknown call', e)

    def _apply_method(self, e: NaryExpr, ctx: EvalCtx):
        fn = _method_table[e.name]
        args: list[float] = []
        for arg in e.children:
            val = self._visit_expr(arg, ctx)
            if not isinstance(val, float):
                raise TypeError(f'expected a real number argument for {e.name}, got {val}')
            args.append(val)
        try:
            result = fn(*args)
        except OverflowError:
            # We could return an infinity, but we don't know which one
            result = math.nan
        except ValueError:
            # domain error means NaN
            result = math.nan

        return result

    def _apply_cast(self, e: Cast, ctx: EvalCtx):
        x = self._visit_expr(e.children[0], ctx)
        if not isinstance(x, float):
            raise TypeError(f'expected a float, got {x}')
        return x

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

    def _apply_range(self, e: Range, ctx: EvalCtx):
        stop = self._visit_expr(e.children[0], ctx)
        if not isinstance(stop, float):
            raise TypeError(f'expected a real number argument, got {stop}')
        if not stop.is_integer():
            raise TypeError(f'expected an integer argument, got {stop}')
        return tuple([float(i) for i in range(int(stop))])

    def _visit_nary_expr(self, e: NaryExpr, ctx: EvalCtx):
        if e.name in _method_table:
            return self._apply_method(e, ctx)
        elif e.name == 'fma':
            raise NotImplementedError('fma not supported in Python 3.11')
        elif isinstance(e, Cast):
            return self._apply_cast(e, ctx)
        elif isinstance(e, Not):
            return self._apply_not(e, ctx)
        elif isinstance(e, And):
            return self._apply_and(e, ctx)
        elif isinstance(e, Or):
            return self._apply_or(e, ctx)
        elif isinstance(e, Range):
            return self._apply_range(e, ctx)
        else:
            raise NotImplementedError('unknown n-ary expression', e)

    def _apply_cmp2(self, op: CompareOp, lhs, rhs):
        match op:
            case CompareOp.EQ:
                return lhs == rhs
            case CompareOp.NE:
                return lhs != rhs
            case CompareOp.LT:
                return lhs < rhs
            case CompareOp.LE:
                return lhs <= rhs
            case CompareOp.GT:
                return lhs > rhs
            case CompareOp.GE:
                return lhs >= rhs
            case _:
                raise NotImplementedError('unknown comparison operator', op)

    def _visit_compare(self, e: Compare, ctx: EvalCtx):
        lhs = self._visit_expr(e.children[0], ctx)
        for op, arg in zip(e.ops, e.children[1:]):
            rhs = self._visit_expr(arg, ctx)
            if not self._apply_cmp2(op, lhs, rhs):
                return False
            lhs = rhs
        return True

    def _visit_tuple_expr(self, e: TupleExpr, ctx: EvalCtx):
        return tuple([self._visit_expr(x, ctx) for x in e.children])

    def _visit_tuple_ref(self, e: TupleRef, ctx: EvalCtx):
        value = self._visit_expr(e.value, ctx)
        if not isinstance(value, tuple):
            raise TypeError(f'expected a tensor, got {value}')

        elt = value
        for s in e.slices:
            val = self._visit_expr(s, ctx)
            if not isinstance(val, Digital):
                raise TypeError(f'expected a real number slice, got {val}')
            if not val.is_integer():
                raise TypeError(f'expected an integer slice, got {val}')
            elt = elt[int(val)]

        return elt

    def _visit_tuple_set(self, e: TupleSet, ctx: EvalCtx):
        raise NotImplementedError

    def _apply_comp(
        self,
        bindings: list[tuple[Id, Expr]],
        elt: Expr,
        ctx: EvalCtx,
        elts: list[Any]
    ):
        if bindings == []:
            elts.append(self._visit_expr(elt, ctx))
        else:
            var, iterable = bindings[0]
            array = self._visit_expr(iterable, ctx)
            if not isinstance(array, tuple):
                raise TypeError(f'expected a tensor, got {array}')
            for val in array:
                if isinstance(var, NamedId):
                    self.env[var] = val
                self._apply_comp(bindings[1:], elt, ctx, elts)

    def _visit_comp_expr(self, e: CompExpr, ctx: EvalCtx):
        # evaluate comprehension
        elts: list[Any] = []
        bindings = [(var, iterable) for var, iterable in zip(e.vars, e.iterables)]
        self._apply_comp(bindings, e.elt, ctx, elts)

        # remove temporarily bound variables
        for var in e.vars:
            if isinstance(var, NamedId):
                del self.env[var]
 
        # the result
        return tuple(elts)

    def _visit_if_expr(self, e: IfExpr, ctx: EvalCtx):
        cond = self._visit_expr(e.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        return self._visit_expr(e.ift if cond else e.iff, ctx)

    def _visit_var_assign(self, stmt: VarAssign, ctx: EvalCtx):
        val = self._visit_expr(stmt.expr, ctx)
        match stmt.var:
            case NamedId():
                self.env[stmt.var] = val
            case UnderscoreId():
                pass
            case _:
                raise NotImplementedError('unknown variable', stmt.var)

    def _unpack_tuple(self, binding: TupleBinding, val: tuple, ctx: EvalCtx) -> None:
        if len(binding.elts) != len(val):
            raise NotImplementedError(f'unpacking {len(val)} values into {len(binding.elts)}')
        for elt, v in zip(binding.elts, val):
            match elt:
                case NamedId():
                    self.env[elt] = v
                case UnderscoreId():
                    pass
                case TupleBinding():
                    self._unpack_tuple(elt, v, ctx)
                case _:
                    raise NotImplementedError('unknown tuple element', elt)

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: EvalCtx):
        val = self._visit_expr(stmt.expr, ctx)
        if not isinstance(val, tuple):
            raise TypeError(f'expected a tuple, got {val}')
        self._unpack_tuple(stmt.binding, val, ctx)

    def _visit_ref_assign(self, stmt: RefAssign, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: EvalCtx):
        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        elif cond:
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
                del self.env[phi.rhs]
        else:
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.lhs]
                del self.env[phi.lhs]

    def _visit_if_stmt(self, stmt: IfStmt, ctx: EvalCtx):
        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        elif cond:
            self._visit_block(stmt.ift, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.lhs]
                del self.env[phi.lhs]
        else:
            self._visit_block(stmt.iff, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
                del self.env[phi.rhs]

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: EvalCtx):
        for phi in stmt.phis:
            self.env[phi.name] = self.env[phi.lhs]
            del self.env[phi.lhs]

        cond = self._visit_expr(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')

        while cond:
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
                del self.env[phi.rhs]

            cond = self._visit_expr(stmt.cond, ctx)
            if not isinstance(cond, bool):
                raise TypeError(f'expected a boolean, got {cond}')

    def _visit_for_stmt(self, stmt: ForStmt, ctx: EvalCtx):
        for phi in stmt.phis:
            self.env[phi.name] = self.env[phi.lhs]
            del self.env[phi.lhs]

        iterable = self._visit_expr(stmt.iterable, ctx)
        if not isinstance(iterable, tuple):
            raise TypeError(f'expected a tensor, got {iterable}')

        for val in iterable:
            if isinstance(stmt.var, NamedId):
                self.env[stmt.var] = val
            self._visit_block(stmt.body, ctx)
            for phi in stmt.phis:
                self.env[phi.name] = self.env[phi.rhs]
                del self.env[phi.rhs]

    def _visit_context(self, stmt: ContextStmt, ctx: EvalCtx):
        ctx = determine_ctx(ctx, stmt.props)
        if not self._is_python_ctx(ctx):
            raise NotImplementedError(f'unsupported context {ctx}')
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

    def _visit_phis(self, phis, lctx, rctx):
        raise NotImplementedError('do not call directly')

    def _visit_loop_phis(self, phis, lctx, rctx):
        raise NotImplementedError('do not call directly')

    def _visit_block(self, block: Block, ctx: EvalCtx):
        for stmt in block.stmts:
            if isinstance(stmt, Return):
                x = self._visit_return(stmt, ctx)
                raise FunctionReturnException(x)
            self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FunctionDef, ctx: EvalCtx):
        raise NotImplementedError('do not call directly')



class PythonInterpreter(Interpreter):
    """
    Python-backed interpreter for FPy programs.

    Programs are evaluated using Python's `math` library.
    Booleans are Python `bool` values, real numbers are `float` values,
    and tensors are Python `tuple` values.
    """

    def eval(
        self,
        func: Function,
        args: Sequence[Any],
        ctx: Optional[EvalCtx] = None
    ):
        if not isinstance(func, Function):
            raise TypeError(f'Expected Function, got {func}')
        return _Interpreter(func.ir).eval(args, ctx)

    def eval_with_trace(self, func: Function, args: Sequence[Any], ctx = None):
        raise NotImplementedError('not implemented')

    def eval_expr(self, expr: Expr, env: _Env, ctx: EvalCtx):
        raise NotImplementedError('not implemented')
