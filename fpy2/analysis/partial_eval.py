"""
Partial evaluation.

For each expression and SSA definition, records the statically-known
:data:`Value` (if any) under the active rounding context.  Consumed
by :class:`fpy2.transform.ConstFold` as the single source of truth
for "is this expression a known constant?"; also used by
:class:`fpy2.analysis.ArraySizeInfer`, :class:`fpy2.analysis.ContextUse`,
and :class:`fpy2.transform.LiftContext`.
"""

from dataclasses import dataclass
from fractions import Fraction
from types import ModuleType
from typing import TypeAlias

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..fpc_context import FPCoreContext
from ..interpret import Interpreter, get_default_interpreter
from ..number import Float, REAL

from .define_use import DefineUse, DefineUseAnalysis, Definition, DefSite

ScalarValue: TypeAlias = bool | Float | Fraction | Context
TupleValue: TypeAlias = tuple['Value', ...]
ListValue: TypeAlias = list['Value']
Value: TypeAlias = ScalarValue | TupleValue | ListValue


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

    def _base_env(self) -> dict[NamedId, object]:
        return { 
            NamedId(d): self.func.env[d]
            for d in self.func.env
            if isinstance(self.func.env[d], ModuleType)
        }

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

    def _try_eval(self, e_eval: Expr, ctx: Context):
        """Evaluate *e_eval* via the interpreter, returning the result
        on success or ``None`` if the interpreter raised.

        Partial evaluation is best-effort: a single arithmetic edge
        case (e.g. ``Cast`` of an inexact value, division by zero,
        precision overflow inside MPFR) should not crash the analysis
        for the whole function.  We treat any exception here as "this
        expression isn't statically foldable" and move on.
        """
        try:
            return self.rt.eval_expr(e_eval, self._base_env(), ctx)
        except Exception:
            return None

    def _record(self, e: Expr, val):
        """Record *val* as ``by_expr[e]`` unless ``val is None`` (the
        :meth:`_try_eval` sentinel for "didn't evaluate")."""
        if val is not None:
            self.by_expr[e] = val

    def _visit_nullaryop(self, e: NullaryOp, ctx: Context | None):
        if ctx is not None:
            self._record(e, self._try_eval(e, ctx))

    def _visit_unaryop(self, e: UnaryOp, ctx: Context | None):
        self._visit_expr(e.arg, ctx)
        if self._is_value(e.arg) and ctx is not None:
            e_arg = ForeignVal(self.by_expr[e.arg], None)
            if isinstance(e, NamedUnaryOp):
                e_eval: UnaryOp = type(e)(e.func, e_arg, e.loc)
            else:
                e_eval = type(e)(e_arg, e.loc)
            self._record(e, self._try_eval(e_eval, ctx))

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
            self._record(e, self._try_eval(e_eval, ctx))

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
            self._record(e, self._try_eval(e_eval, ctx))

    def _visit_naryop(self, e: NaryOp, ctx: Context | None):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        # ``empty()`` constructs uninitialized values — never partial-eval it.
        if isinstance(e, Empty):
            return
        if (
            ctx is not None
            and len(e.args) > 0
            and all(self._is_value(arg) for arg in e.args)
        ):
            e_args = [ForeignVal(self.by_expr[arg], None) for arg in e.args]
            if isinstance(e, NamedNaryOp):
                e_eval: NaryOp = type(e)(e.func, e_args, e.loc)
            else:
                e_eval = type(e)(e_args, e.loc)
            self._record(e, self._try_eval(e_eval, ctx))

    def _visit_compare(self, e: Compare, ctx: Context | None):
        """Chained comparisons (``a < b < c``) fold when every operand
        is a known value.  The result is a Python ``bool``; we route
        through the interpreter so chain semantics and per-op handling
        of NaN / signed-zero match runtime behaviour.
        """
        for arg in e.args:
            self._visit_expr(arg, ctx)
        if ctx is not None and all(self._is_value(arg) for arg in e.args):
            e_args = [ForeignVal(self.by_expr[arg], None) for arg in e.args]
            e_eval = Compare(e.ops, e_args, e.loc)
            self._record(e, self._try_eval(e_eval, ctx))

    def _visit_call(self, e: Call, ctx: Context | None):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        for _, v in e.kwargs:
            self._visit_expr(v, ctx)
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
            self._record(e, self._try_eval(e_eval, ctx))

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Context | None):
        for elt in e.elts:
            self._visit_expr(elt, ctx)
        if all(self._is_value(elt) for elt in e.elts):
            self.by_expr[e] = tuple(self.by_expr[elt] for elt in e.elts)

    def _visit_list_expr(self, e: ListExpr, ctx: Context | None):
        """A list literal whose elements are all statically known
        becomes a Python ``list`` in :data:`by_expr`.  Lets parent
        operators (e.g. ``min(xs)``, ``sum(xs)``) see the list as a
        value and fold against it."""
        for elt in e.elts:
            self._visit_expr(elt, ctx)
        if all(self._is_value(elt) for elt in e.elts):
            self.by_expr[e] = [self.by_expr[elt] for elt in e.elts]

    def _visit_list_ref(self, e: ListRef, ctx: Context | None):
        """``xs[i]`` folds when ``xs`` is a known list/tuple and ``i``
        is a known scalar.  Routed through the interpreter so FPy's
        indexing semantics (negative-index handling, etc.) are
        authoritative."""
        self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        if (
            ctx is not None
            and self._is_value(e.value)
            and self._is_value(e.index)
        ):
            v = ForeignVal(self.by_expr[e.value], None)
            i = ForeignVal(self.by_expr[e.index], None)
            e_eval = ListRef(v, i, e.loc)
            self._record(e, self._try_eval(e_eval, ctx))

    def _visit_list_slice(self, e: ListSlice, ctx: Context | None):
        """``xs[a:b]`` folds when ``xs`` is a known list and the
        bounds (if any) are known scalars.  FPy slicing is stricter
        than Python's (exact-length, raises on overflow), so we route
        through the interpreter rather than slicing natively."""
        self._visit_expr(e.value, ctx)
        if e.start is not None:
            self._visit_expr(e.start, ctx)
        if e.stop is not None:
            self._visit_expr(e.stop, ctx)
        if (
            ctx is not None
            and self._is_value(e.value)
            and (e.start is None or self._is_value(e.start))
            and (e.stop is None or self._is_value(e.stop))
        ):
            v = ForeignVal(self.by_expr[e.value], None)
            s = ForeignVal(self.by_expr[e.start], None) if e.start is not None else None
            t = ForeignVal(self.by_expr[e.stop], None) if e.stop is not None else None
            e_eval = ListSlice(v, s, t, e.loc)
            self._record(e, self._try_eval(e_eval, ctx))

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
