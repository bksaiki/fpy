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


class _TopType:
    """SCCP lattice top — internal sentinel for "def was analyzed; not
    a single foldable value".  Stored only in the analysis instance's
    ``by_def``; stripped from the public :class:`PartialEvalInfo`."""

    def __repr__(self):
        return '_TOP'


_TOP: _TopType = _TopType()

# Internal lattice value: a public :data:`Value`, BOT (absent from
# the map), or the ``_TOP`` sentinel.
_Lattice: TypeAlias = Value | _TopType


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

    by_def: dict[Definition, _Lattice]
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
        # Strip ``_TOP`` from the public view — consumers see only
        # foldable :data:`Value` entries.
        public_by_def: dict[Definition, Value] = {
            d: v for d, v in self.by_def.items() if v is not _TOP  # type: ignore[misc]
        }
        return PartialEvalInfo(public_by_def, self.by_expr, self.def_use)

    def _base_env(self) -> dict[NamedId, object]:
        return {
            NamedId(d): self.func.env[d]
            for d in self.func.env
            if isinstance(self.func.env[d], ModuleType)
        }

    def _is_value(self, e: Expr) -> bool:
        return e in self.by_expr

    def _visit_expr(self, e: Expr, ctx: Context | None):
        # Pop any stale entry from a previous pass (matters for the
        # loop fixpoint, which revisits the body — without this, a
        # foldable value from an early iteration sticks around after
        # the phi has promoted to ``_TOP``).
        self.by_expr.pop(e, None)
        super()._visit_expr(e, ctx)

    def _visit_var(self, e: Var, ctx: Context | None):
        d = self.def_use.find_def_from_use(e)
        if d in self.by_def:
            val = self.by_def[d]
            if val is not _TOP:
                self.by_expr[e] = val

    def _meet(self, a, b):
        """SCCP meet — ``None`` is BOT (unit), :data:`_TOP` is top;
        anything else is a :data:`Value`."""
        if a is None:
            return b
        if b is None:
            return a
        if a is _TOP or b is _TOP:
            return _TOP
        return a if a == b else _TOP

    def _merge_branch_phis(self, stmt: Stmt):
        """Merge phis after an ``if`` / ``if-else``: both branches are
        visited unconditionally, so absence from ``by_def`` means "we
        visited and couldn't fold" (i.e. ``_TOP``-equivalent) rather
        than "haven't seen yet"."""
        for phi in self.def_use.phis[stmt]:
            lhs = self.by_def.get(self.def_use.defs[phi.lhs], _TOP)
            rhs = self.by_def.get(self.def_use.defs[phi.rhs], _TOP)
            merged = self._meet(lhs, rhs)
            if merged is not None:
                self.by_def[phi] = merged

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
        """Evaluate via the interpreter; return ``None`` on any
        exception (PE is best-effort)."""
        try:
            return self.rt.eval_expr(e_eval, self._base_env(), ctx)
        except Exception:
            return None

    def _record(self, e: Expr, val):
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
        if isinstance(e, Empty):
            # ``empty()`` constructs uninitialized values
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
        for elt in e.elts:
            self._visit_expr(elt, ctx)
        if all(self._is_value(elt) for elt in e.elts):
            self.by_expr[e] = [self.by_expr[elt] for elt in e.elts]

    def _visit_list_ref(self, e: ListRef, ctx: Context | None):
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
        # FPy slicing is stricter than Python's; route through the
        # interpreter rather than slicing natively.
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

    def _visit_if_expr(self, e: IfExpr, ctx: Context | None):
        # When the condition is statically known, copy the chosen
        # branch's value — the unchosen branch need not be foldable.
        self._visit_expr(e.cond, ctx)
        self._visit_expr(e.ift, ctx)
        self._visit_expr(e.iff, ctx)
        if not self._is_value(e.cond):
            return
        cond_val = self.by_expr[e.cond]
        if not isinstance(cond_val, bool):
            return
        branch = e.ift if cond_val else e.iff
        if self._is_value(branch):
            self.by_expr[e] = self.by_expr[branch]

    def _visit_attribute(self, e: Attribute, ctx: Context | None):
        self._visit_expr(e.value, ctx)
        if self._is_value(e.value):
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
        else:
            # RHS isn't statically foldable — clear any stale by_def
            # entry for the target (matters for loop re-iteration).
            self._clear_binding(stmt, stmt.target)

    def _clear_binding(self, site: DefSite, binding: Id | TupleBinding):
        match binding:
            case NamedId():
                d = self.def_use.find_def_from_site(binding, site)
                self.by_def.pop(d, None)
            case UnderscoreId():
                pass
            case TupleBinding():
                for elt in binding.elts:
                    self._clear_binding(site, elt)

    def _visit_if1(self, stmt: If1Stmt, ctx: Context | None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        self._merge_branch_phis(stmt)

    def _visit_if(self, stmt: IfStmt, ctx: Context | None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)
        self._merge_branch_phis(stmt)

    def _loop_fixpoint(self, stmt: Stmt, run_body):
        """Drive a loop's header-phis to fixpoint.  Seed each phi from
        its lhs (pre-loop) value, then iterate ``run_body`` until phi
        bounds stop changing.

        Termination: the lattice has height 3 (``None`` → :data:`Value`
        → :data:`_TOP`) and :meth:`_meet` is monotone, so every phi
        transitions at most twice before stabilizing.
        """
        phis = self.def_use.phis.get(stmt, set())
        for phi in phis:
            lhs = self.by_def.get(self.def_use.defs[phi.lhs])
            if lhs is None:
                self.by_def.pop(phi, None)
            else:
                self.by_def[phi] = lhs
        while True:
            run_body()
            changed = False
            for phi in phis:
                lhs = self.by_def.get(self.def_use.defs[phi.lhs], _TOP)
                rhs = self.by_def.get(self.def_use.defs[phi.rhs], _TOP)
                new = self._meet(lhs, rhs)
                old = self.by_def.get(phi)
                if new != old:
                    if new is None:
                        self.by_def.pop(phi, None)
                    else:
                        self.by_def[phi] = new
                    changed = True
            if not changed:
                return

    def _visit_while(self, stmt: WhileStmt, ctx: Context | None):
        self._visit_expr(stmt.cond, ctx)
        self._loop_fixpoint(stmt, lambda: self._visit_block(stmt.body, ctx))

    def _visit_for(self, stmt: ForStmt, ctx: Context | None):
        self._visit_expr(stmt.iterable, ctx)
        # The loop variable is fresh-bound per iteration; we don't
        # analyse it specially, just iterate the body so other phis
        # converge.
        self._loop_fixpoint(stmt, lambda: self._visit_block(stmt.body, ctx))

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
    """Partial evaluation — records the statically-known
    :data:`Value` of each expression under the active rounding
    context.

    Coverage: literals, free-variable lookups, all operator kinds
    (nullary through n-ary, plus :class:`Compare`), context-constructor
    :class:`Call` s, :class:`Attribute` access, tuple / list literals,
    list indexing / slicing, and SSA definitions via ``Assign``.
    Operators that raise inside the interpreter (e.g. inexact
    :class:`Cast`, division by zero) are silently dropped: PE is
    best-effort and never crashes the analysis for a single edge case.
    """

    @staticmethod
    def apply(func: FuncDef, *, def_use: DefineUseAnalysis | None = None):
        """Run partial evaluation on *func* and return the
        :class:`PartialEvalInfo`.  Pass an existing ``def_use=`` to
        avoid recomputing the def-use analysis."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)

        inst = _PartialEvalInstance(func, def_use)
        return inst.apply()
