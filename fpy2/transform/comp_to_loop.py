"""
Lower a list comprehension into an allocation plus a loop.

A comprehension is the only *expression* in FPy that binds names, and that costs
the scheduling language reach: every pass needing a statement slot for a preamble
refuses to enter one, so no rounding can be eliminated or inserted inside a
comprehension.  Lowering it is what makes that code schedulable.

.. code-block:: python

    # before
    ys = [f(x) for x in xs]

    # after
    t = xs
    acc = fp.empty(len(t))
    for i in range(len(t)):
        x = t[i]
        acc[i] = f(x)
    ys = acc

Several clauses are a cartesian product -- CPython's nesting, outermost clause
first and the last index varying fastest -- so they become nested loops over the
original targets with a running write index.  The index is *not* linearized into
``i0 * n1 + i1``; the FPCore backend does that and gets it wrong for three or
more clauses.

The allocation needs its length up front, and `fp.empty` fills with ``UNINIT``
rather than zero, so every slot must be written.  Both hold: FPy rejects ``if``
filters in a comprehension, so the length is exactly the product of the clause
lengths.  Where a later clause's iterable is independent of the earlier targets
that product is the size expression; where it is not -- ``[b for a in xs for b in
a]``, a ragged flatten -- the size is a nest of sums instead.

The size expression re-evaluates a *dependent* iterable, so the two traversals
have to agree on the lengths: such an iterable must be pure and must not compute
under a stochastic rounding context.  An independent iterable is bound to a
temporary and evaluated once, so it is unconstrained.

Declined, with a reason, when a dependent iterable is impure or stochastic, and
where the loop cannot be emitted: a ``while`` condition, which is re-evaluated
every iteration; an ``IfExpr`` branch, which is conditional; and inside another
comprehension, until the outer one is lowered and gives the inner a slot.
"""

from typing import Any

from ..analysis import (
    ContextUse,
    ContextUseAnalysis,
    DefineUse,
    DefineUseAnalysis,
    PartialEval,
    PartialEvalInfo,
    Purity,
    SyntaxCheck,
)
from ..ast.fpyast import (
    Add,
    Assign,
    ContextStmt,
    Empty,
    Expr,
    ForStmt,
    FuncDef,
    Id,
    If1Stmt,
    IfExpr,
    IfStmt,
    IndexedAssign,
    Integer,
    Len,
    ListComp,
    ListRef,
    Mul,
    NamedId,
    Range1,
    Stmt,
    StmtBlock,
    Sum,
    TupleBinding,
    Var,
    WhileStmt,
)
from ..ast.visitor import DefaultVisitor
from ..number import Context
from ..utils import Gensym
from .cursor import Cursor, EditLog
from .utils import (
    Declined,
    SiteRewriter,
    check_where,
    clone,
    copy_target,
    integer_ctx,
)


def _bound_names(target: Id | TupleBinding) -> set[NamedId]:
    """The names *target* binds."""
    if isinstance(target, TupleBinding):
        return target.names()
    return {target} if isinstance(target, NamedId) else set()


def _mentions(e: Expr, names: set[NamedId]) -> bool:
    """Whether *e* reads any of *names*."""
    if not names:
        return False
    found = False

    class _C(DefaultVisitor):
        def _visit_var(self, var: Var, ctx: None):
            nonlocal found
            if var.name in names:
                found = True

    _C()._visit_expr(e, None)
    return found


def _dependent(e: ListComp) -> list[int]:
    """The clauses whose iterable reads a target bound by an earlier clause.

    These are the ones the size expression has to evaluate a second time.
    """
    out: list[int] = []
    bound: set[NamedId] = set()
    for j, (target, iterable) in enumerate(zip(e.targets, e.iterables)):
        if j > 0 and _mentions(iterable, bound):
            out.append(j)
        bound |= _bound_names(target)
    return out


class _CompToLoopInstance(SiteRewriter):
    """Lowers selected comprehensions into an allocation plus a loop."""

    _expr_sited = True   # the candidates are comprehensions

    func: FuncDef
    def_use: DefineUseAnalysis
    ctx_use: ContextUseAnalysis
    eval_info: PartialEvalInfo
    temp_id: NamedId
    gensym: Gensym
    where: int | Cursor | None

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        ctx_use: ContextUseAnalysis,
        eval_info: PartialEvalInfo,
        where: int | Cursor | None = None,
        temp_id: NamedId | None = None,
    ):
        self.func = func
        self.def_use = def_use
        self.ctx_use = ctx_use
        self.eval_info = eval_info
        self.temp_id = NamedId('t') if temp_id is None else temp_id
        self.gensym = Gensym(reserved=def_use.names())
        self.where = where

    # ------------------------------------------------------------------
    # Verification

    def _stochastic(self, e: Expr) -> bool:
        """Whether any operation in *e* rounds stochastically.

        Such an expression can produce a different *length* on a second
        evaluation, which would leave the allocation the wrong size.
        """
        outer = self
        found = False

        class _C(DefaultVisitor):
            def _visit_expr(self, sub: Expr, ctx: None):
                nonlocal found
                if outer._rounds_unpredictably(sub):
                    found = True
                super()._visit_expr(sub, ctx)

        _C()._visit_expr(e, None)
        return found

    def _rounds_unpredictably(self, sub: Expr) -> bool:
        """Whether the operation *sub* might round stochastically.

        A scope that does not resolve to a concrete context counts: we cannot
        show it is not stochastic, and a wrong answer here silently mis-sizes
        the allocation.
        """
        try:
            scope = self.ctx_use.find_scope_from_use(sub)   # type: ignore[arg-type]
        except (KeyError, TypeError):
            return False   # not an operation, so it rounds nothing
        found: object = scope.ctx
        if not isinstance(found, Context) and isinstance(scope.site, ContextStmt):
            # a symbolic scope: the introducing `with` may still name a context
            found = self.eval_info.by_expr.get(scope.site.ctx)
        return not isinstance(found, Context) or found.is_stochastic()

    def _verify(self, e: ListComp) -> None | Declined:
        """`None` where *e* may be lowered, else why not."""
        for j in _dependent(e):
            it = e.iterables[j]
            if not Purity.analyze_expr(it, self.def_use):
                return Declined(
                    'a later iterable is impure, and the size expression has to '
                    'evaluate it a second time'
                )
            if self._stochastic(it):
                return Declined(
                    'a later iterable rounds stochastically, so its length may '
                    'differ between the size expression and the loop'
                )
        return None

    # ------------------------------------------------------------------
    # The rewrite

    def _size(self, iters: list[NamedId | None], e: ListComp) -> Expr:
        """An expression for the comprehension's length.

        `iters[j]` is the temporary an *independent* iterable was bound to, or
        `None` where the clause is dependent and must be rebuilt in place.
        """
        loc = e.loc
        if all(t is not None for t in iters):
            # every length is available here: the product
            lens: list[Expr] = [Len(None, Var(t, loc), loc) for t in iters]  # type: ignore[arg-type]
            size: Expr = lens[0]
            for nxt in lens[1:]:
                size = Mul(size, nxt, loc)
            return size

        # ragged: sum over each clause of the next clause's length
        def nest(j: int) -> Expr:
            src = Var(iters[j], loc) if iters[j] is not None else clone(e.iterables[j])  # type: ignore[arg-type]
            if j == len(iters) - 1:
                return Len(None, src, loc)
            inner = ListComp([copy_target(e.targets[j])], [src], nest(j + 1), loc)
            return Sum(None, inner, loc)

        return nest(0)

    def _descend(self, e: ListComp, out: list) -> None:
        """Visit *e*'s children the way :meth:`_lower` would.

        The listing has to reach exactly what the rewrite reaches, or it counts a
        site the rewrite will not take: only an *independent* iterable ends up
        outside the loops, so only it keeps a statement slot.
        """
        dependent = set(_dependent(e))
        for j, iterable in enumerate(e.iterables):
            self._visit_expr(iterable, None if j in dependent else out)
        self._visit_expr(e.elt, None)

    def _lower(self, e: ListComp, out: list) -> Expr:
        """Emit the allocation and loops into *out*; return the result `Var`."""
        loc = e.loc
        dependent = set(_dependent(e))

        # An independent iterable is evaluated once, here.  A dependent one has
        # to stay inside the loop that binds what it reads.
        iters: list[NamedId | None] = []
        for j, iterable in enumerate(e.iterables):
            if j in dependent:
                iters.append(None)
                continue
            t = self.gensym.refresh(self.temp_id)
            out.append(Assign(t, None, self._visit_expr(iterable, out), loc))
            iters.append(t)

        acc = self.gensym.fresh('acc')
        size = self._size(iters, e)
        if isinstance(size, (Len, Var)):
            # a bare length needs no arithmetic, so no exact-integer block
            out.append(Assign(acc, None, Empty(None, [size], loc), loc))
        else:
            n = self.gensym.fresh('n')
            out.append(integer_ctx([Assign(n, None, size, loc)], loc))
            out.append(Assign(acc, None, Empty(None, [Var(n, loc)], loc), loc))

        elt = self._visit_expr(e.elt, None)
        if len(e.targets) == 1 and iters[0] is not None:
            # One clause over a bound temporary: index it, so nothing is
            # loop-carried and the store index is the loop variable itself.
            idx = self.gensym.fresh('i')
            first = iters[0]
            assert first is not None
            src = Var(first, loc)
            body = StmtBlock([
                Assign(copy_target(e.targets[0]), None,
                       ListRef(clone(src), Var(idx, loc), loc), loc),
                IndexedAssign(acc, [Var(idx, loc)], elt, loc),
            ])
            out.append(ForStmt(
                idx, Range1(None, Len(None, clone(src), loc), loc), body, loc,
            ))
            return Var(acc, loc)

        # Several clauses: nest the loops over the original targets and carry a
        # write index, rather than linearizing it.
        j_id = self.gensym.fresh('j')
        out.append(Assign(j_id, None, Integer(0, loc), loc))
        inner: list[Stmt] = [
            IndexedAssign(acc, [Var(j_id, loc)], elt, loc),
            integer_ctx([
                Assign(j_id, None,
                       Add(Var(j_id, loc), Integer(1, loc), loc), loc)
            ], loc),
        ]
        for j in reversed(range(len(e.targets))):
            bound = iters[j]
            nest_src: Expr = (
                Var(bound, loc) if bound is not None
                else self._visit_expr(e.iterables[j], None)
            )
            inner = [ForStmt(copy_target(e.targets[j]), nest_src, StmtBlock(inner), loc)]
        out.extend(inner)
        return Var(acc, loc)

    # ------------------------------------------------------------------
    # Walk

    def _visit_expr(self, e: Expr, ctx: Any) -> Expr:
        if not isinstance(e, ListComp):
            return super()._visit_expr(e, ctx)

        declined = (
            Declined(
                'there is no statement-level position for the loop the rewrite '
                'emits: a `while` condition runs every iteration, a conditional '
                'branch may not run at all, and a comprehension has no slot '
                'until the one around it is lowered'
            )
            if ctx is None
            else self._verify(e)
        )
        if declined is not None:
            self.refused.append((e, declined.reason))
            if self._named_by_cursor(e):
                self.declined.append(declined.reason)
            return super()._visit_expr(e, ctx)

        idx = self.site_idx
        self.site_idx += 1
        if not self._selects_expr(e, idx):
            return super()._visit_expr(e, ctx)

        self._matched += 1
        if self.listing:
            self.found_exprs.append(e)
            self._descend(e, ctx)
            return e

        lowered = self._lower(e, ctx)
        self._replaced = True
        return lowered

    def _visit_if_expr(self, e: IfExpr, ctx: Any) -> IfExpr:
        # A branch is conditional, so a loop hoisted out of it would run either
        # way.  The condition is unconditional and keeps its slot.
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, None)
        iff = self._visit_expr(e.iff, None)
        return IfExpr(cond, ift, iff, e.loc)

    def _visit_while(self, stmt: WhileStmt, ctx: Any):
        # The condition is re-evaluated every iteration, and a loop hoisted
        # before the `while` runs once -- so a comprehension there would be
        # frozen at its first value.  Measured: it turns a terminating loop into
        # an out-of-bounds slice.
        stmt, _ = super()._visit_while(stmt, None)
        return stmt, ctx

    # An `if` condition and a `for` iterable are each evaluated exactly once, so
    # hoisting out of them is sound -- but `SiteRewriter._visit_block` clears
    # `_replaced` for every statement of the nested block, which would lose the
    # edit and mis-forward every statement after this one.  Carry it across.

    def _visit_if1(self, stmt: If1Stmt, ctx: Any):
        cond = self._visit_expr(stmt.cond, ctx)
        rewrote = self._replaced
        body, _ = self._visit_block(stmt.body, ctx)
        self._replaced = self._replaced or rewrote
        return If1Stmt(cond, body, stmt.loc), ctx

    def _visit_if(self, stmt: IfStmt, ctx: Any):
        cond = self._visit_expr(stmt.cond, ctx)
        rewrote = self._replaced
        ift, _ = self._visit_block(stmt.ift, ctx)
        iff, _ = self._visit_block(stmt.iff, ctx)
        self._replaced = self._replaced or rewrote
        return IfStmt(cond, ift, iff, stmt.loc), ctx

    def _visit_for(self, stmt: ForStmt, ctx: Any):
        iterable = self._visit_expr(stmt.iterable, ctx)
        rewrote = self._replaced
        target = self._visit_binding(stmt.target, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        self._replaced = self._replaced or rewrote
        return ForStmt(target, iterable, body, stmt.loc), ctx

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


def _lister(func: FuncDef) -> _CompToLoopInstance:
    """The pass instance a listing walks `func` with."""
    def_use = DefineUse.analyze(func)
    return _CompToLoopInstance(
        func,
        def_use,
        ContextUse.analyze(func, def_use=def_use),
        PartialEval.apply(func),
    )


class CompToLoop:
    """
    Transformation pass to lower a list comprehension into an allocation
    plus a loop.
    """

    @staticmethod
    def sites(func: FuncDef, within: Cursor | None = None) -> list[Cursor]:
        """The comprehensions of `func` this rewrite would lower, in visit
        order -- what a `where` index counts, and what `within` narrows.

        A comprehension this pass refuses is not a site: it neither appears here
        nor takes an index.  :meth:`refusals` says why.
        """
        return _lister(func).list_sites(within)

    @staticmethod
    def refusals(
        func: FuncDef, within: Cursor | None = None
    ) -> list[tuple[Cursor, str]]:
        """Why each comprehension of `func` that is not a site was refused."""
        return _lister(func).list_refusals(within)

    @staticmethod
    def apply(
        func: FuncDef, *, where: int | Cursor | None = None,
        temp_id: NamedId | None = None
    ) -> FuncDef:
        """
        Lowers every qualifying comprehension of `func` into an allocation
        plus a loop.

        `where` selects one comprehension by index in visit order; `None`
        lowers every one that verifies.
        """
        return CompToLoop.apply_with_edits(
            func, where=where, temp_id=temp_id,
        ).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef, *, where: int | Cursor | None = None,
        temp_id: NamedId | None = None
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        check_where(where)

        def_use = DefineUse.analyze(func)
        ctx_use = ContextUse.analyze(func, def_use=def_use)
        vtor = _CompToLoopInstance(
            func, def_use, ctx_use, PartialEval.apply(func), where, temp_id,
        )
        out = vtor.apply()
        vtor.check_site('a comprehension')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=True)

