"""Module containing the interpreter for FPy programs."""

from typing import Any, Callable, Optional, Sequence

from titanfp.arithmetic.evalctx import EvalCtx
from titanfp.arithmetic.ieee754 import ieee_ctx
from titanfp.arithmetic.mpmf import MPMF
from titanfp.titanic.digital import Digital
from titanfp.titanic.ndarray import NDArray
import titanfp.titanic.gmpmath as gmpmath

from ..ir import *

ScalarVal = bool | Digital
"""Type of scalar values in FPy programs."""
TensorVal = NDArray
"""Type of tensor values in FPy programs."""

ScalarArg = ScalarVal | str | int | float
"""Type of scalar arguments in FPy programs; includes native Python types"""
TensorArg = NDArray | tuple | list
"""Type of tensor arguments in FPy programs; includes native Python types"""

def _isinf(x: MPMF) -> bool:
    return x.isinf

def _isnan(x: MPMF) -> bool:
    return x.isnan

_method_table: dict[str, Callable[..., Any]] = {
    '+': MPMF.add,
    '-': MPMF.sub,
    '*': MPMF.mul,
    '/': MPMF.div,
    'fabs': MPMF.fabs,
    'sqrt': MPMF.sqrt,
    'fma': MPMF.fma,
    'neg': MPMF.neg,
    'copysign': MPMF.copysign,
    'fdim': MPMF.fdim,
    'fmax': MPMF.fmax,
    'fmin': MPMF.fmin,
    'fmod': MPMF.fmod,
    'remainder': MPMF.remainder,
    'hypot': MPMF.hypot,
    'cbrt': MPMF.cbrt,
    'ceil': MPMF.ceil,
    'floor': MPMF.floor,
    'nearbyint': MPMF.nearbyint,
    'round': MPMF.round,
    'trunc': MPMF.trunc,
    'acos': MPMF.acos,
    'asin': MPMF.asin,
    'atan': MPMF.atan,
    'atan2': MPMF.atan2,
    'cos': MPMF.cos,
    'sin': MPMF.sin,
    'tan': MPMF.tan,
    'acosh': MPMF.acosh,
    'asinh': MPMF.asinh,
    'atanh': MPMF.atanh,
    'cosh': MPMF.cosh,
    'sinh': MPMF.sinh,
    'tanh': MPMF.tanh,
    'exp': MPMF.exp_,
    'exp2': MPMF.exp2,
    'expm1': MPMF.expm1,
    'log': MPMF.log,
    'log10': MPMF.log10,
    'log1p': MPMF.log1p,
    'log2': MPMF.log2,
    'pow': MPMF.pow,
    'erf': MPMF.erf,
    'erfc': MPMF.erfc,
    'lgamma': MPMF.lgamma,
    'tgamma': MPMF.tgamma,
    'isfinite': MPMF.isfinite,
    'isinf': _isinf,
    'isnan': _isnan,
    'isnormal': MPMF.isnormal,
    'signbit': MPMF.signbit,
}

class Interpreter(ReduceVisitor):
    """Interpreter for FPy programs."""

    # TODO: what are the semantics of arguments
    def _arg_to_mpmf(self, arg: Any, ctx: EvalCtx):
        if isinstance(arg, str | int | float):
            return MPMF(x=arg, ctx=ctx)
        elif isinstance(arg, Digital):
            return MPMF(x=arg, ctx=ctx)
        elif isinstance(arg, tuple | list):
            raise NotImplementedError()
        else:
            raise NotImplementedError(f'unknown argument type {arg}')

    def eval(self,
        func: FunctionDef,
        arg_seq: Sequence[Any],
        ctx: Optional[EvalCtx] = None
    ):
        if not isinstance(func, FunctionDef):
            raise TypeError(f'Expected Function, got {type(func)}')
        args = tuple(arg_seq)
        if len(args) != len(func.args):
            raise TypeError(f'Expected {len(func.args)} arguments, got {len(args)}')
        if ctx is None:
            ctx = ieee_ctx(11, 64)
        for val, arg in zip(args, func.args):
            match arg.ty:
                case AnyType():
                    ctx = ctx.let([(arg.name, self._arg_to_mpmf(val, ctx))])
                case RealType():
                    x = self._arg_to_mpmf(val, ctx)
                    if isinstance(x, Digital):
                        ctx = ctx.let([(arg.name, x)])
                    else:
                        raise NotImplementedError(f'argument is a scalar, got data {val}')
                case _:
                    raise NotImplementedError(f'unknown argument type {arg.ty}')
        return self._visit(func.body, ctx)

    def _visit_var(self, e, ctx: EvalCtx):
        if e.name not in ctx.bindings:
            raise RuntimeError(f'unbound variable {e.name}')
        return ctx.bindings[e.name]

    def _visit_decnum(self, e, ctx: EvalCtx):
        return MPMF(x=e.val, ctx=ctx)

    def _visit_integer(self, e, ctx: EvalCtx):
        return MPMF(x=e.val, ctx=ctx)

    def _visit_digits(self, e, ctx: EvalCtx):
        x = gmpmath.compute_digits(e.m, e.e, e.b, prec=ctx.p)
        return MPMF._round_to_context(x, ctx)

    def _visit_unknown(self, e, ctx: EvalCtx):
        raise NotImplementedError

    def _apply_method(self, e: NaryExpr, ctx: EvalCtx):
        fn = _method_table[e.name]
        args: list[Digital] = []
        for arg in e.children:
            val = self._visit(arg, ctx)
            if not isinstance(val, Digital):
                raise TypeError(f'expected a real number argument for {e.name}, got {val}')
            args.append(val)
        return fn(*args)
    
    def _apply_not(self, e: Not, ctx: EvalCtx):
        arg = self._visit(e.children[0], ctx)
        if not isinstance(arg, bool):
            raise TypeError(f'expected a boolean argument, got {arg}')
        return not arg
    
    def _apply_and(self, e: And, ctx: EvalCtx):
        args: list[bool] = []
        for arg in e.children:
            val = self._visit(arg, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean argument, got {val}')
            args.append(val)
        return all(args)

    def _apply_or(self, e: Or, ctx: EvalCtx):
        args: list[bool] = []
        for arg in e.children:
            val = self._visit(arg, ctx)
            if not isinstance(val, bool):
                raise TypeError(f'expected a boolean argument, got {val}')
            args.append(val)
        return any(args)
    
    def _apply_range(self, e: Range, ctx: EvalCtx):
        stop = self._visit(e.children[0], ctx)
        if not isinstance(stop, Digital):
            raise TypeError(f'expected a real number argument, got {stop}')
        if not stop.is_integer():
            raise TypeError(f'expected an integer argument, got {stop}')
        return NDArray([MPMF(i, ctx) for i in range(int(stop))])

    def _visit_nary_expr(self, e, ctx: EvalCtx):
        if e.name in _method_table:
            return self._apply_method(e, ctx)
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

    def _apply_cmp2(self, op, lhs, rhs):
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

    def _visit_compare(self, e, ctx: EvalCtx):
        lhs = self._visit(e.children[0], ctx)
        for op, arg in zip(e.ops, e.children[1:]):
            rhs = self._visit(arg, ctx)
            if not self._apply_cmp2(op, lhs, rhs):
                return False
            lhs = rhs
        return True

    def _visit_tuple_expr(self, e, ctx: EvalCtx):
        return NDArray([self._visit(x, ctx) for x in e.children])

    def _visit_ref_expr(self, e, ctx: EvalCtx):
        raise NotImplementedError
    
    def _apply_comp(self, bindings: list[tuple[Expr, Expr]], elt: Expr, ctx: EvalCtx, elts: list[Any]):
        if bindings == []:
            elts.append(self._visit(elt, ctx))
        else:
            var, iterable = bindings[0]
            array = self._visit(iterable, ctx)
            if not isinstance(array, NDArray):
                raise TypeError(f'expected a tensor, got {array}')
            for val in array:
                self._apply_comp(bindings[1:], elt, ctx.let([(var, val)]), elts)

    def _visit_comp_expr(self, e, ctx: EvalCtx):
        elts: list[Any] = []
        bindings = [(var, iterable) for var, iterable in zip(e.vars, e.iterables)]
        self._apply_comp(bindings, e.elt, ctx, elts)
        return NDArray(elts)

    def _visit_if_expr(self, e, ctx: EvalCtx):
        cond = self._visit(e.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        return self._visit(e.ift if cond else e.iff, ctx)

    def _visit_var_assign(self, stmt, ctx: EvalCtx):
        val = self._visit(stmt.expr, ctx)
        return ctx.let([(stmt.var, val)])

    def _unpack_tuple(self, binding: TupleBinding, val: NDArray, ctx: EvalCtx):
        if len(binding.elts) != len(val):
            raise NotImplementedError(f'unpacking {len(val)} values into {len(binding.elts)}')
        for elt, v in zip(binding.elts, val):
            match elt:
                case str():
                    ctx = ctx.let([(elt, v)])
                case TupleBinding():
                    ctx = self._unpack_tuple(elt, v, ctx)
                case _:
                    raise NotImplementedError('unknown tuple element', elt)
        return ctx

    def _visit_tuple_assign(self, stmt, ctx: EvalCtx):
        val = self._visit(stmt.expr, ctx)
        if not isinstance(val, NDArray):
            raise TypeError(f'expected a tuple, got {val}')
        return self._unpack_tuple(stmt.binding, val, ctx)

    def _visit_if1_stmt(self, stmt, ctx: EvalCtx):
        cond = self._visit(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        elif cond:
            ctx = self._visit(stmt.body, ctx)
            for phi in stmt.phis:
                ctx = ctx.let([(phi.name, ctx.bindings[phi.rhs])])
        else:
            for phi in stmt.phis:
                ctx = ctx.let([(phi.name, ctx.bindings[phi.lhs])])
        return ctx

    def _visit_if_stmt(self, stmt, ctx: EvalCtx):
        cond = self._visit(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')
        elif cond:
            ctx = self._visit(stmt.ift, ctx)
            for phi in stmt.phis:
                ctx = ctx.let([(phi.name, ctx.bindings[phi.lhs])])
        else:
            ctx = self._visit(stmt.iff, ctx)
            for phi in stmt.phis:
                ctx = ctx.let([(phi.name, ctx.bindings[phi.rhs])])
        return ctx

    def _visit_while_stmt(self, stmt, ctx: EvalCtx):
        for phi in stmt.phis:
            ctx = ctx.let([(phi.name, ctx.bindings[phi.lhs])])
        
        cond = self._visit(stmt.cond, ctx)
        if not isinstance(cond, bool):
            raise TypeError(f'expected a boolean, got {cond}')

        while cond:
            ctx = self._visit(stmt.body, ctx)
            for phi in stmt.phis:
                ctx = ctx.let([(phi.name, ctx.bindings[phi.rhs])])

            cond = self._visit(stmt.cond, ctx)
            if not isinstance(cond, bool):
                raise TypeError(f'expected a boolean, got {cond}')

        return ctx

    def _visit_for_stmt(self, stmt, ctx: EvalCtx):        
        for phi in stmt.phis:
            ctx = ctx.let([(phi.name, ctx.bindings[phi.lhs])])

        iterable = self._visit(stmt.iterable, ctx)
        if not isinstance(iterable, NDArray):
            raise TypeError(f'expected a tensor, got {iterable}')

        for val in iterable:
            ctx = ctx.let([(stmt.var, val)])
            ctx = self._visit(stmt.body, ctx)
            for phi in stmt.phis:
                ctx = ctx.let([(phi.name, ctx.bindings[phi.rhs])])

        return ctx

    def _visit_return(self, stmt, ctx: EvalCtx):
        return self._visit(stmt.expr, ctx)

    def _visit_block(self, block, ctx: EvalCtx) -> EvalCtx:
        for stmt in block.stmts:
            if isinstance(stmt, Return):
                return self._visit_return(stmt, ctx)
            else:
                ctx = self._visit_statement(stmt, ctx)
        return ctx

    def _visit_function(self, func, ctx: EvalCtx):
        raise NotImplementedError('do not call directly')

    # override typing hint
    def _visit_statement(self, stmt, ctx: EvalCtx) -> EvalCtx:
        return super()._visit_statement(stmt, ctx)