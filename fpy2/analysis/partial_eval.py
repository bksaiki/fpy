"""
Partial evaluation.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..fpc_context import FPCoreContext
from ..interpret import Interpreter, get_default_interpreter
from ..number import Float, REAL

from .define_use import DefineUse, DefineUseAnalysis, Definition, DefSite

ScalarValue: TypeAlias = bool | Float | Fraction | Context
TupleValue: TypeAlias = tuple['Value', ...]
Value: TypeAlias = ScalarValue | TupleValue


@dataclass
class PartialEvalInfo:
    by_def: dict[Definition, Value]
    by_expr: dict[Expr, Value]
    def_use: DefineUseAnalysis


class _PartialEvalInstance(DefaultVisitor):
    """
    Partial evaluation instance for a function.
    """

    func: FuncDef
    def_use: DefineUseAnalysis
    rt: Interpreter

    by_def: dict[Definition, Value]
    by_expr: dict[Expr, Value]

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
    ):
        self.func = func
        self.def_use = def_use
        self.rt = get_default_interpreter()
        self.by_def = {}
        self.by_expr = {}

    def apply(self) -> PartialEvalInfo:
        self._visit_function(self.func, None)
        return PartialEvalInfo(self.by_def, self.by_expr, self.def_use)

    def _eval_env(self):
        return { d.name: v for d, v in self.by_def.items() }

    def _is_value(self, e: Expr) -> bool:
        return e in self.by_expr

    def _visit_var(self, e: Var, ctx: Context | None):
        d = self.def_use.find_def_from_use(e)
        if d in self.by_def:
            self.by_expr[e] = self.by_def[d]

    def _visit_bool(self, e: BoolVal, ctx: Context | None):
        self.by_expr[e] = e.val

    def _visit_foreign(self, e: ForeignVal, ctx: Context | None):
        self.by_expr[e] = e.val

    def _visit_decnum(self, e: Decnum, ctx: Context | None):
        self.by_expr[e] = e.as_rational()

    def _visit_integer(self, e: Integer, ctx: Context | None):
        self.by_expr[e] = e.as_rational()

    def _visit_hexnum(self, e: Hexnum, ctx: Context | None):
        self.by_expr[e] = e.as_rational()

    def _visit_rational(self, e: Rational, ctx: Context | None):
        self.by_expr[e] = e.as_rational()

    def _visit_digits(self, e: Digits, ctx: Context | None):
        self.by_expr[e] = e.as_rational()

    def _visit_nullaryop(self, e: NullaryOp, ctx: Context | None):
        if ctx is not None:
            val = self.rt.eval_expr(e, {}, ctx)
            self.by_expr[e] = val

    def _visit_unaryop(self, e: UnaryOp, ctx: Context | None):
        self._visit_expr(e.arg, ctx)
        if self._is_value(e.arg) and ctx is not None:
            e_arg = ForeignVal(self.by_expr[e.arg], None)
            if isinstance(e, NamedUnaryOp):
                e_eval: UnaryOp = type(e)(e.func, e_arg, e.loc)
            else:
                e_eval = type(e)(e_arg, e.loc)
            self.by_expr[e] = self.rt.eval_expr(e_eval, {}, ctx)

    def _visit_binaryop(self, e: BinaryOp, ctx: Context | None):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        if self._is_value(e.first) and self._is_value(e.second) and ctx is not None:
            e_fst = ForeignVal(self.by_expr[e.first], None)
            e_snd = ForeignVal(self.by_expr[e.second], None)
            if isinstance(e, NamedBinaryOp):
                e_eval: BinaryOp = type(e)(e.func, e_fst, e_snd, e.loc)
            else:
                e_eval = type(e)(e_fst, e_snd, e.loc)
            
            self.by_expr[e] = self.rt.eval_expr(e_eval, {}, ctx)
    
    def _visit_ternaryop(self, e: TernaryOp, ctx: Context | None):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        if self._is_value(e.first) and self._is_value(e.second) and self._is_value(e.third) and ctx is not None:
            e_fst = ForeignVal(self.by_expr[e.first], None)
            e_snd = ForeignVal(self.by_expr[e.second], None)
            e_trd = ForeignVal(self.by_expr[e.third], None)
            if isinstance(e, NamedTernaryOp):
                e_eval: TernaryOp = type(e)(e.func, e_fst, e_snd, e_trd, e.loc)
            else:
                e_eval = type(e)(e_fst, e_snd, e_trd, e.loc)
            self.by_expr[e] = self.rt.eval_expr(e_eval, {}, ctx)

    def _visit_call(self, e: Call, ctx: Context | None):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        if (
            ctx is not None
            and isinstance(e.fn, type)
            and issubclass(e.fn, Context)
            and all(self._is_value(arg) for arg in e.args)
            and all(self._is_value(v) for _, v in e.kwargs)
        ):
            arg_vals = [ForeignVal(self.by_expr[arg], None) for arg in e.args]
            kwarg_vals = [ (k, ForeignVal(self.by_expr[v], None)) for k, v in e.kwargs ]
            e_eval = Call(e.func, e.fn, arg_vals, kwarg_vals, e.loc)
            self.by_expr[e] = self.rt.eval_expr(e_eval, {}, ctx)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Context | None):
        for elt in e.elts:
            self._visit_expr(elt, ctx)
        if all(self._is_value(elt) for elt in e.elts):
            self.by_expr[e] = tuple(self.by_expr[elt] for elt in e.elts)

    def _visit_attribute(self, e: Attribute, ctx: Context | None):
        self._visit_expr(e.value, ctx)
        if self._is_value(e.value):
            # constant folding is possible
            val = self.by_expr[e.value]
            if isinstance(val, dict):
                if e.attr not in val:
                    raise RuntimeError(f'unknown attribute {e.attr} for {val}')
                self.by_expr[e] = val[e.attr]
            elif hasattr(val, e.attr):
                self.by_expr[e] = getattr(val, e.attr)
            else:
                raise RuntimeError(f'unknown attribute {e.attr} for {val}')

    def _visit_binding(self, site: DefSite, binding: Id | TupleBinding, val: Value):
        match binding:
            case Id():
                if isinstance(binding, NamedId):
                    d = self.def_use.find_def_from_site(binding, site)
                    self.by_def[d] = val
            case TupleBinding():
                assert isinstance(val, tuple)
                for elt, v in zip(binding.elts, val):
                    self._visit_binding(site, elt, v)

    def _visit_assign(self, stmt: Assign, ctx: Context | None):
        self._visit_expr(stmt.expr, ctx)
        if self._is_value(stmt.expr):
            self._visit_binding(stmt, stmt.target, self.by_expr[stmt.expr])

    def _visit_context(self, stmt: ContextStmt, ctx: Context | None):
        self._visit_expr(stmt.ctx, REAL)

        new_ctx: Context | None = None
        if self._is_value(stmt.ctx):
            val = self.by_expr[stmt.ctx]
            if isinstance(val, Context):
                new_ctx = val

        self._visit_block(stmt.body, new_ctx)


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
            self.by_def[d] = self.func.env[str(name)]

        # visit statements
        self._visit_block(func.body, fctx)


class PartialEval:
    """
    Partial evaluation.

    This analysis evaluates parts of the program that can be determined statically,
    allowing for potential optimizations and simplifications before runtime.
    """

    @staticmethod
    def apply(func: FuncDef, *, def_use: DefineUseAnalysis | None = None):
        """
        Applies partial evaluation.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)

        inst = _PartialEvalInstance(func, def_use)
        return inst.apply()
