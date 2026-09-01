"""
Lower a list comprehension into an allocation plus a loop.

A comprehension is the only *expression* in FPy that binds names, and a pass
needing a statement slot cannot enter one -- so no rounding can be eliminated or
inserted inside a comprehension until it is a loop.

.. code-block:: python

    # before                    # after
    ys = [f(x) for x in xs]     t = xs
                                ys = fp.empty(len(t))
                                for t1 in range(len(t)):
                                    x = t[t1]
                                    ys[t1] = f(x)

An assignment target is filled directly; a comprehension with no name to fill
allocates a temporary of its own.  Every name minted takes its prefix from
``temp_id``.

Several clauses are a cartesian product, so they become nested loops over the
original targets with a running write index.  The allocation needs its length up
front and `fp.empty` fills with ``UNINIT``, so every slot must be written --
both hold where the clause lengths multiply, since FPy rejects ``if`` filters.

A **dependent clause list** -- a clause's iterable mentions an earlier clause's
target, as in ``[b for a in xs for b in a]`` -- has a length that is a sum
rather than a product.  That one is built a row at a time and flattened
(:meth:`~._CompToLoopInstance._lower_dependent`), at the cost of a materialised
row per outer element; ``dependent=False`` declines it.

What the pass leaves, without erroring, is a comprehension with no statement
slot: a ``while`` condition, an ``IfExpr`` branch, or one nested in another.
Running it again after :class:`~fpy2.transform.Hoistable` clears each, which is
what makes the two a fixpoint.
"""

from typing import Any

from ..analysis import DefineUse, DefineUseAnalysis, LiveVars, SyntaxCheck
from ..ast.fpyast import (
    Add,
    Assign,
    Empty,
    Expr,
    ForStmt,
    FuncDef,
    Id,
    IfExpr,
    IndexedAssign,
    Integer,
    Len,
    ListComp,
    ListRef,
    Mul,
    NamedId,
    Range1,
    Range2,
    Range3,
    Stmt,
    StmtBlock,
    TupleBinding,
    UnderscoreId,
    ValueExpr,
    Var,
    WhileStmt,
)
from ..utils import Gensym
from .cursor import Cursor, EditLog
from .utils import (
    Declined,
    SiteRewriter,
    check_where,
    clone,
    copy_target,
    integer_ctx,
    operands,
)

_ATOMIC = (Var, ValueExpr)
"""Expressions with no effects and no subexpressions to re-evaluate."""


def _bound_names(target: Id | TupleBinding) -> set[NamedId]:
    """The names *target* binds."""
    if isinstance(target, TupleBinding):
        return target.names()
    return {target} if isinstance(target, NamedId) else set()


def dependent_clauses(e: ListComp) -> list[int]:
    """The clauses whose iterable reads a target bound by an earlier clause.

    Non-empty means the length is not the product of the clause lengths.
    """
    out: list[int] = []
    bound: set[NamedId] = set()
    for j, (target, iterable) in enumerate(zip(e.targets, e.iterables)):
        if j > 0 and LiveVars.analyze(iterable) & bound:
            out.append(j)
        bound |= _bound_names(target)
    return out


class _CompToLoopInstance(SiteRewriter):
    """Lowers selected comprehensions into an allocation plus a loop."""

    _expr_sited = True   # the candidates are comprehensions

    func: FuncDef
    def_use: DefineUseAnalysis
    temp_id: NamedId
    gensym: Gensym
    where: int | Cursor | None
    dependent: bool
    """whether to lower a dependent clause list (:meth:`_lower_dependent`).  On
    unless a consumer opts out, since it costs a materialised row per outer
    element"""
    _fill: tuple[ListComp, NamedId, tuple[Expr, ...]] | None
    """an assignment's right-hand comprehension, and the place its loops may
    write into -- a name, plus the indices of a slot -- instead of minting an
    `acc` and copying it in"""

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        where: int | Cursor | None = None,
        temp_id: NamedId | None = None,
        dependent: bool = True,
    ):
        self.func = func
        self.def_use = def_use
        self.temp_id = NamedId('t') if temp_id is None else temp_id
        self.gensym = Gensym(reserved=def_use.names())
        self.where = where
        self.dependent = dependent
        self._fill = None

    # ------------------------------------------------------------------
    # Verification

    def _verify(self, e: ListComp) -> None | Declined:
        """`None` where *e* may be lowered, else why not."""
        if not self.dependent and dependent_clauses(e):
            # `fp.empty` needs a length first, and a sum of clause lengths has
            # nowhere to come from with no `append`
            return Declined(
                'a later clause\'s iterable mentions an earlier clause\'s '
                'target, so the length is not a product of the clause lengths'
            )
        return None

    # ------------------------------------------------------------------
    # The rewrite

    def _size(self, iters: list[Expr], e: ListComp) -> Expr:
        """The comprehension's length: the product of the clause lengths.

        Only reached for an independent clause list, so every length is
        available here, before the loops.
        """
        loc = e.loc
        size: Expr = Len(None, clone(iters[0]), loc)
        for t in iters[1:]:
            size = Mul(size, Len(None, clone(t), loc), loc)
        return size

    @staticmethod
    def _inlinable(iterable: Expr) -> bool:
        """Whether *iterable* may stand in for a temp bound to it.

        The temp buys evaluation-once and immunity to a rebinding of the source
        name; a `range` over atoms needs neither, and it costs one.  A name
        holds a value, so a bound `range(n)` must be materialized -- and over a
        real bound there is no such list, only a counted loop.
        """
        return (
            isinstance(iterable, (Range1, Range2, Range3))
            and all(isinstance(a, _ATOMIC) for a in operands(iterable))
        )

    def _descend(self, e: ListComp, out: list) -> None:
        """Visit *e*'s children the way :meth:`_lower` would.

        The listing must reach exactly what the rewrite reaches, or it counts a
        site the rewrite will not take: the iterables end up outside the loops
        and keep their statement slot, the element does not.  A dependent clause
        list keeps only its *first* iterable out here; the rest and the element
        go into the nested comprehension :meth:`_lower_dependent` builds.
        """
        keep = 1 if dependent_clauses(e) else len(e.iterables)
        for i, iterable in enumerate(e.iterables):
            self._visit_expr(iterable, out if i < keep else None)
        self._visit_expr(e.elt, None)

    def _fillable(self, e: ListComp, target: NamedId, slot: bool) -> bool:
        """Whether the loops of *e* may write into a place based on *target*.

        The element must not read *target*: the loops overwrite the place before
        it runs.  An iterable may -- bound to a temp first, it still sees what
        the place held.  A slot's indices need no check, since going stale would
        take a target shadowing an existing definition, which FPy rejects.

        A *slot* fill needs more.  ``zs[i] = fp.empty(n)`` mutates the list
        already there, so every alias of that slot sees the uninitialised cells,
        and an alias need not name ``zs``.  Nothing here can rule one out, so
        the element must read nothing the comprehension did not bind.  A plain
        ``z = fp.empty(n)`` *rebinds*, leaving an alias on the old list.
        """
        if target in LiveVars.analyze(e.elt):
            return False
        if not slot:
            return True
        bound: set[NamedId] = set()
        for t in e.targets:
            bound |= _bound_names(t)
        return not (LiveVars.analyze(e.elt) - bound)

    def _take_fill(self, e: ListComp) -> 'tuple[NamedId, tuple[Expr, ...]] | None':
        """The place the loops of *e* may write into, taken once.

        Keyed on the comprehension itself and taken before the iterables are
        visited, so a comprehension nested in one cannot claim the statement's
        place.
        """
        if self._fill is None or self._fill[0] is not e:
            return None
        _, target, indices = self._fill
        self._fill = None
        return target, indices

    def _took_fill(self, offered: bool) -> bool:
        """Whether :meth:`_lower` took the place this statement offered."""
        taken = offered and self._fill is None
        self._fill = None
        return taken

    def _lower_dependent(self, e: ListComp, out: list, fill) -> Expr:
        """A clause list whose length is a sum rather than a product.

        `fp.empty` needs the total up front, with no ``append`` and no counting
        an iterable twice.  So build the rows, add up their lengths, then
        flatten -- ``derived-semantics.rst``'s rewrite.

        Split at the *first* clause only: its iterable can read no target, so
        the outer comprehension is independent and the rest become one nested
        comprehension for a later pass.  `k` clauses peel one at a time and each
        flatten is one level deep.  The rows are read twice, to count and to
        copy, so unlike a plain fill the temporary is a second place.
        """
        loc = e.loc
        # the first clause's iterable, bound like any other: evaluated once
        src = self.gensym.refresh(self.temp_id)
        out.append(Assign(
            src, None, self._visit_expr(e.iterables[0], out), loc,
        ))

        rows = self.gensym.refresh(self.temp_id)
        inner = ListComp(
            list(e.targets[1:]), list(e.iterables[1:]), e.elt, loc,
        )
        out.append(Assign(rows, None, ListComp(
            [copy_target(e.targets[0])], [Var(src, loc)], inner, loc,
        ), loc))

        # the total length: one pass over the rows, adding each length
        n = self.gensym.refresh(self.temp_id)
        row = self.gensym.refresh(self.temp_id)
        out.append(integer_ctx([Assign(n, None, Integer(0, loc), loc)], loc))
        out.append(ForStmt(row, Var(rows, loc), StmtBlock([
            integer_ctx([Assign(n, None, Add(
                Var(n, loc), Len(None, Var(row, loc), loc), loc,
            ), loc)], loc),
        ]), loc))

        acc, at = (self.gensym.refresh(self.temp_id), ()) if fill is None else fill

        def place(*more: Expr) -> list[Expr]:
            return [clone(ix) for ix in at] + list(more)

        alloc = Empty(None, [Var(n, loc)], loc)
        out.append(
            IndexedAssign(acc, place(), alloc, loc) if at
            else Assign(acc, None, alloc, loc)
        )

        # ... then copy every element across, carrying one write index
        j = self.gensym.refresh(self.temp_id)
        elt = self.gensym.refresh(self.temp_id)
        row2 = self.gensym.refresh(self.temp_id)
        out.append(integer_ctx([Assign(j, None, Integer(0, loc), loc)], loc))
        out.append(ForStmt(row2, Var(rows, loc), StmtBlock([
            ForStmt(elt, Var(row2, loc), StmtBlock([
                IndexedAssign(acc, place(Var(j, loc)), Var(elt, loc), loc),
                integer_ctx([Assign(j, None, Add(
                    Var(j, loc), Integer(1, loc), loc,
                ), loc)], loc),
            ]), loc),
        ]), loc))
        return Var(acc, loc)

    def _lower(self, e: ListComp, out: list) -> Expr:
        """Emit the allocation and loops into *out*; return the result `Var`."""
        loc = e.loc
        fill = self._take_fill(e)
        if dependent_clauses(e):
            return self._lower_dependent(e, out, fill)

        # every clause is independent, so each iterable is evaluated once here
        iters: list[Expr] = []
        for iterable in e.iterables:
            src = self._visit_expr(iterable, out)
            if self._inlinable(src):
                iters.append(src)
                continue
            t = self.gensym.refresh(self.temp_id)
            out.append(Assign(t, None, src, loc))
            iters.append(Var(t, loc))

        acc, at = (self.gensym.refresh(self.temp_id), ()) if fill is None else fill

        def place(*more: Expr) -> list[Expr]:
            """The accumulator's own indices, then *more*.  Empty where the
            accumulator is a name rather than a slot."""
            return [clone(ix) for ix in at] + list(more)

        def bind(value: Expr) -> Stmt:
            """Put *value* in the accumulator."""
            if at:
                return IndexedAssign(acc, place(), value, loc)
            return Assign(acc, None, value, loc)

        size = self._size(iters, e)
        if isinstance(size, Len):
            # a bare length needs no arithmetic, so no exact-integer block
            out.append(bind(Empty(None, [size], loc)))
        else:
            n = self.gensym.refresh(self.temp_id)
            out.append(integer_ctx([Assign(n, None, size, loc)], loc))
            out.append(bind(Empty(None, [Var(n, loc)], loc)))

        elt = self._visit_expr(e.elt, None)
        # Index the iterable unless it is inlined *and* binds a target:
        # subscripting a `range` materialises what leaving it inline avoids.
        # Indexing is otherwise the better shape, since the store index is the
        # loop variable and `format_infer` bounds it by the range, where a
        # carried counter widens on an unknown trip count.
        indexed = (
            not self._inlinable(iters[0])
            or isinstance(e.targets[0], UnderscoreId)
        )
        if len(e.targets) == 1 and indexed:
            idx = self.gensym.refresh(self.temp_id)
            src = iters[0]
            stmts: list[Stmt] = []
            if not isinstance(e.targets[0], UnderscoreId):
                # a discarded target binds nothing the element can read, and a
                # subscript has no effect to keep
                stmts.append(Assign(
                    copy_target(e.targets[0]), None,
                    ListRef(clone(src), Var(idx, loc), loc), loc,
                ))
            stmts.append(IndexedAssign(acc, place(Var(idx, loc)), elt, loc))
            body = StmtBlock(stmts)
            out.append(ForStmt(
                idx, Range1(None, Len(None, clone(src), loc), loc), body, loc,
            ))
            return Var(acc, loc)

        # Several clauses, or one whose iterable is not indexed: nest loops
        # over the original targets and carry a write index.
        j_id = self.gensym.refresh(self.temp_id)
        out.append(Assign(j_id, None, Integer(0, loc), loc))
        inner: list[Stmt] = [
            IndexedAssign(acc, place(Var(j_id, loc)), elt, loc),
            integer_ctx([
                Assign(j_id, None,
                       Add(Var(j_id, loc), Integer(1, loc), loc), loc)
            ], loc),
        ]
        for j in reversed(range(len(e.targets))):
            inner = [ForStmt(
                copy_target(e.targets[j]), clone(iters[j]), StmtBlock(inner), loc,
            )]
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

    def _visit_assign(self, stmt: Assign, ctx: Any):
        # `z = [...]` fills `z` itself: an `acc` copied into it leaves two
        # names on one list, and a second name is a second *place*.
        self._fill = None
        if (
            isinstance(stmt.expr, ListComp)
            and isinstance(stmt.target, NamedId)
            and self._fillable(stmt.expr, stmt.target, slot=False)
        ):
            self._fill = (stmt.expr, stmt.target, ())
        offered = self._fill is not None
        s, _ = super()._visit_assign(stmt, ctx)
        if self._took_fill(offered):
            # the loops write into the target, so the assignment left over is
            # `z = z`; the loop stands in its place
            return ctx.pop(), ctx
        return s, ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: Any):
        # `zs[i] = [...]` allocates straight into the slot.  The inner half of
        # `[[...] for ...]` arrives in this shape once the outer is lowered.
        self._fill = None
        if (
            isinstance(stmt.expr, ListComp)
            and isinstance(stmt.var, NamedId)
            and self._fillable(stmt.expr, stmt.var, slot=True)
        ):
            self._fill = (stmt.expr, stmt.var, tuple(stmt.indices))
        offered = self._fill is not None
        s, _ = super()._visit_indexed_assign(stmt, ctx)
        if self._took_fill(offered):
            return ctx.pop(), ctx
        return s, ctx

    def _visit_if_expr(self, e: IfExpr, ctx: Any) -> IfExpr:
        # a branch is conditional, so a loop hoisted out of one runs either
        # way; the condition is unconditional and keeps its slot
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, None)
        iff = self._visit_expr(e.iff, None)
        return IfExpr(cond, ift, iff, e.loc)

    def _visit_while(self, stmt: WhileStmt, ctx: Any):
        # the condition re-runs every iteration where a loop hoisted before the
        # `while` runs once, freezing the comprehension at its first value
        stmt, _ = super()._visit_while(stmt, None)
        return stmt, ctx

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


def _lister(func: FuncDef, dependent: bool = True) -> _CompToLoopInstance:
    """The pass instance a listing walks `func` with."""
    return _CompToLoopInstance(
        func, DefineUse.analyze(func), dependent=dependent,
    )


class CompToLoop:
    """
    Transformation pass to lower a list comprehension into an allocation
    plus a loop.
    """

    @staticmethod
    def sites(
        func: FuncDef, within: Cursor | None = None, *,
        dependent: bool = True,
    ) -> list[Cursor]:
        """The comprehensions of `func` this rewrite would lower, in visit
        order -- what a `where` index counts, and what `within` narrows.

        A comprehension this pass cannot lower is not a site: it neither appears
        here nor takes an index.  :meth:`refusals` says why each was left.
        """
        return _lister(func, dependent).list_sites(within)

    @staticmethod
    def refusals(
        func: FuncDef, within: Cursor | None = None, *,
        dependent: bool = True,
    ) -> list[tuple[Cursor, str]]:
        """Why each comprehension of `func` that is not a site was left alone."""
        return _lister(func, dependent).list_refusals(within)

    @staticmethod
    def apply(
        func: FuncDef, *, where: int | Cursor | None = None,
        temp_id: NamedId | None = None, dependent: bool = True,
    ) -> FuncDef:
        """
        Lowers every comprehension of `func` it can into an allocation plus a
        loop, and leaves the rest alone.

        `where` selects one comprehension by index in visit order; `None` takes
        every one it can lower.

        `dependent=False` opts out of lowering a clause list whose length is a
        sum rather than a product, which costs a materialised row per outer
        element where every other shape allocates once and fills.
        """
        return CompToLoop.apply_with_edits(
            func, where=where, temp_id=temp_id, dependent=dependent,
        ).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef, *, where: int | Cursor | None = None,
        temp_id: NamedId | None = None, dependent: bool = True,
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced.

        A comprehension this pass cannot lower is left exactly as it was, so the
        result is not guaranteed comprehension-free.  A caller that needs it to
        be checks: :meth:`refusals` names what was left and why, and a comprehension
        nested in another needs a further pass rather than being unlowerable at
        all.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        check_where(where)

        def_use = DefineUse.analyze(func)
        vtor = _CompToLoopInstance(
            func, def_use, where, temp_id, dependent=dependent,
        )
        out = vtor.apply()
        vtor.check_site('a comprehension')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=True)

