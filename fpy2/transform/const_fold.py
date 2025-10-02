"""
Constant folding.
"""

import inspect

from fractions import Fraction
from typing import Any, Callable, TypeAlias

from ..analysis import DefineUse, DefineUseAnalysis, Definition
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..env import ForeignEnv
from ..fpc_context import FPCoreContext
from ..interpret import Interpreter, get_default_interpreter
from ..number import Float, RealFloat, REAL
from ..utils import is_dyadic

from .. import ops

ScalarValue: TypeAlias = bool | Float | Fraction | Context
TupleValue: TypeAlias = tuple['Value', ...]
Value: TypeAlias = ScalarValue | TupleValue

class _ConstFoldInstance(DefaultTransformVisitor):
    """
    Constant folding instance for a function.
    """

    func: FuncDef
    env: ForeignEnv
    def_use: DefineUseAnalysis
    rt: Interpreter
    enable_context: bool
    enable_op: bool

    vals: dict[Definition, Value]
    remap: dict[Var, Var]

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        enable_context: bool,
        enable_op: bool,
    ):
        self.func = func
        self.env = func.env
        self.def_use = def_use
        self.rt = get_default_interpreter()
        self.enable_context = enable_context
        self.enable_op = enable_op
        self.vals = {}
        self.remap = {}

    def _eval_env(self):
        return { d.name: v for d, v in self.vals.items() }

    def _is_value(self, e: Expr) -> bool:
        match e:
            case Var():
                d = self.def_use.find_def_from_use(self.remap.get(e, e))
                return d in self.vals
            case BoolVal() | RationalVal() | ForeignVal():
                return True
            case Attribute():
                return self._is_value(e.value)
            case TupleExpr():
                return all(self._is_value(elt) for elt in e.elts)
            case _:
                return False

    def _rational_as_ast(self, x: Float | Fraction, loc: Location | None):
        if isinstance(x, Float):
            x = x.as_rational()

        if x.denominator == 1:
            return Integer(int(x), loc)
        else:
            # TODO: emitting a rational node requires a name:
            # could be `rational` or `fp.rational` or `fpy2.rational`
            func = Attribute(Var(NamedId('fp'), loc), 'rational', loc)
            return Rational(func, x.numerator, x.denominator, loc)

    def _visit_var(self, e: Var, ctx: Context | None):
        e2 = super()._visit_var(e, ctx)
        self.remap[e2] = e
        return e2

    def _visit_attribute(self, e: Attribute, ctx: Context | None):
        e = super()._visit_attribute(e, ctx)
        if self._is_value(e.value) and ctx is not None:
            # constant folding is possible
            val = self.rt.eval_expr(e, self._eval_env(), ctx)
            return ForeignVal(val, e.loc)
        else:
            return e

    def _visit_nullaryop(self, e: NullaryOp, ctx: Context | None):
        e = super()._visit_nullaryop(e, ctx)
        if self.enable_op and ctx is not None:
            # constant folding is possible
            val = self.rt.eval_expr(e, self._eval_env(), ctx)
            if not isinstance(val, Float | Fraction):
                raise TypeError(f'expected a real number, got `{val}`')
            return self._rational_as_ast(val, e.loc)
        else:
            return e

    def _visit_unaryop(self, e: UnaryOp, ctx: Context | None):
        e = super()._visit_unaryop(e, ctx)
        if self.enable_op and not isinstance(e, Round | RoundExact) and ctx is not None and self._is_value(e.arg):
            # constant folding is possible
            val = self.rt.eval_expr(e, self._eval_env(), ctx)
            if isinstance(val, Float | Fraction):
                # constant folded to a real number
                return self._rational_as_ast(val, e.loc)
            else:
                # TODO: constant folded to a tuple or list
                return e
        else:
            return e

    def _visit_binaryop(self, e: BinaryOp, ctx: Context | None):
        e = super()._visit_binaryop(e, ctx)
        if self.enable_op and not isinstance(e, RoundAt) and ctx is not None and self._is_value(e.first) and self._is_value(e.second):
            # constant folding is possible
            val = self.rt.eval_expr(e, self._eval_env(), ctx)
            match val:
                case Float() | Fraction():
                    return self._rational_as_ast(val, e.loc)
                case bool():
                    return BoolVal(val, e.loc)
                case _:
                    # TODO: constant folded to a tuple or list
                    return e
        else:
            return e

    def _visit_ternaryop(self, e: TernaryOp, ctx: Context | None):
        e = super()._visit_ternaryop(e, ctx)
        if self.enable_op and ctx is not None and self._is_value(e.first) and self._is_value(e.second) and self._is_value(e.third):
            # constant folding is possible
            val = self.rt.eval_expr(e, self._eval_env(), ctx)
            match val:
                case Float() | Fraction():
                    return self._rational_as_ast(val, e.loc)
                case bool():
                    return BoolVal(val, e.loc)
                case _:
                    # TODO: constant folded to a tuple or list
                    return e
        else:
            return e

    def _visit_call(self, e: Call, ctx: Context | None):
        e = super()._visit_call(e, ctx)
        if (
            self.enable_context 
            and isinstance(e.fn, type)
            and issubclass(e.fn, Context)
            and ctx is not None
            and all(self._is_value(arg) for arg in e.args)
            and all(self._is_value(v) for _, v in e.kwargs)
        ):
            # context constructor
            val = self.rt.eval_expr(e, self._eval_env(), ctx)
            if not isinstance(val, Context):
                raise TypeError(f'expected a context, got `{val}`')
            return ForeignVal(val, e.loc)
        else:
            return e

    def _visit_context(self, stmt: ContextStmt, ctx: Context | None):
        ctx_e = self._visit_expr(stmt.ctx, REAL)
        if isinstance(ctx_e, ForeignVal) and isinstance(ctx_e.val, Context):
            # we can determine the context
            new_ctx = ctx_e.val
        else:
            # otherwise, we cannot
            new_ctx = None

        body, _ = self._visit_block(stmt.body, new_ctx)
        s = ContextStmt(stmt.target, ctx_e, body, stmt.loc)
        return s, ctx

    def _visit_function(self, func: FuncDef, ctx: None):
        # extract overriding context
        match func.ctx:
            case None:
                fctx: Context | None = None
            case FPCoreContext():
                fctx = func.ctx.to_context()
            case Context():
                fctx = func.ctx
            case _:
                raise RuntimeError(f'unreachable: {func.ctx}')

        # bind foreign values
        for name in func.free_vars:
            d = self.def_use.find_def_from_site(name, func)
            self.vals[d] = self.env[name.base]

        return super()._visit_function(func, fctx)

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


class ConstFold:
    """
    Constant folding.

    This transform evaluates expressions that can be determined statically:
    - constants constructors
    - operations (math, lists, etc.)
    """

    @staticmethod
    def apply(
        func: FuncDef,
        *,
        def_use: DefineUseAnalysis | None = None,
        enable_context: bool = True,
        enable_op: bool = True,
    ) -> FuncDef:
        """
        Applies constant folding.

        Optionally, specify:
        - `enable_context`: whether to enable constant folding for context constructors [default: True]
        - `enable_op`: whether to enable constant folding for operations [default: True]
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)
        inst = _ConstFoldInstance(func, def_use, enable_context, enable_op)
        return inst.apply()
