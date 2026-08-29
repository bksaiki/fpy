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
    ys = fp.empty(len(t))
    for t1 in range(len(t)):
        x = t[t1]
        ys[t1] = f(x)

An assignment target is filled directly; only a comprehension with no name to
fill -- in a ``return``, or an argument -- allocates a temporary of its own.
Every name minted here takes its prefix from ``temp_id``, so a caller owns the
whole namespace the rewrite introduces.

Several clauses are a cartesian product -- CPython's nesting, outermost clause
first and the last index varying fastest -- so they become nested loops over the
original targets with a running write index.  Nested loops need no linearized
``i0 * n1 + i1``, and so cannot get its stride wrong.

The allocation needs its length up front, and `fp.empty` fills with ``UNINIT``
rather than zero, so every slot must be written.  Both hold where the clause
lengths multiply: FPy rejects ``if`` filters in a comprehension, so the length is
exactly that product, and every slot is written.

**This pass lowers what it can and leaves the rest alone.**  It never errors on a
comprehension it cannot lower; :meth:`CompToLoop.refusals` names each one and
why, and a caller that needs a comprehension-free program checks for itself.

What it leaves:

- **A dependent clause list** -- some clause's iterable mentions an earlier
  clause's target, as in ``[b for a in xs for b in a]``.  The length is then a
  sum rather than a product, and `fp.empty` has nowhere to get it: there is no
  ``append``, and computing it up front would mean evaluating that iterable a
  second time.
- **A comprehension with no statement slot** -- a ``while`` condition, which is
  re-evaluated every iteration; an ``IfExpr`` branch, which is conditional; and
  one nested in another comprehension, which gets a slot once the outer one is
  lowered, so a further pass takes it.
"""

from typing import Any

from ..analysis import DefineUse, DefineUseAnalysis, SyntaxCheck
from ..ast.fpyast import (
    Add,
    Assign,
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
    operands,
)


_ATOMIC = (Var, ValueExpr)
"""Expressions with no effects and no subexpressions to re-evaluate."""


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


def dependent_clauses(e: ListComp) -> list[int]:
    """The clauses whose iterable reads a target bound by an earlier clause.

    Non-empty means the length is not the product of the clause lengths.
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
    temp_id: NamedId
    gensym: Gensym
    where: int | Cursor | None
    dependent: bool
    """whether to lower a dependent clause list, which costs a materialised
    row per outer element -- see :meth:`_lower_dependent`"""
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
        dependent: bool = False,
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
        if dependent_clauses(e) and not self.dependent:
            # `fp.empty` needs its length first and there is no `append`, so a
            # length that is not a product of the clause lengths has nowhere to
            # come from.
            return Declined(
                'a later clause\'s iterable mentions an earlier clause\'s '
                'target, so the length is not a product of the clause lengths'
            )
        return None

    # ------------------------------------------------------------------
    # The rewrite

    def _size(self, iters: list[Expr], e: ListComp) -> Expr:
        """The comprehension's length: the product of the clause lengths.

        Every clause is independent -- a dependent one is left alone -- so each
        length is available here, before the loops.
        """
        loc = e.loc
        size: Expr = Len(None, clone(iters[0]), loc)
        for t in iters[1:]:
            size = Mul(size, Len(None, clone(t), loc), loc)
        return size

    @staticmethod
    def _inlinable(iterable: Expr) -> bool:
        """Whether *iterable* may stand in for a temp bound to it.

        The temp buys two things: the iterable is evaluated exactly once, and
        the loop bound is out of reach of a body that rebinds the source name.
        A `range` over atoms needs neither -- it has no effects, and nothing a
        lowered loop binds can shadow an atom of it, since a comprehension
        target may not shadow an existing definition.

        What the temp *costs* is fusion.  A name holds a value, so `range(n)`
        bound to one must be materialized as a list -- and over a real bound
        there is no such list, only a counted loop.
        """
        return (
            isinstance(iterable, (Range1, Range2, Range3))
            and all(isinstance(a, _ATOMIC) for a in operands(iterable))
        )

    def _descend(self, e: ListComp, out: list) -> None:
        """Visit *e*'s children the way :meth:`_lower` would.

        The listing has to reach exactly what the rewrite reaches, or it counts a
        site the rewrite will not take: the iterables end up outside the loops
        and keep their statement slot, the element does not.

        A dependent clause list keeps only its *first* iterable out here.  The
        rest, and the element, go into the nested comprehension
        :meth:`_lower_dependent` builds, so they get a slot when that one is
        lowered and not before.
        """
        keep = 1 if dependent_clauses(e) else len(e.iterables)
        for i, iterable in enumerate(e.iterables):
            self._visit_expr(iterable, out if i < keep else None)
        self._visit_expr(e.elt, None)

    def _fillable(self, e: ListComp, target: NamedId) -> bool:
        """Whether the loops of *e* may write into a place based on *target*.

        The element must not read *target*: the loops overwrite the place before
        it runs.  An iterable may -- it is bound to a temp first, so it still
        sees what the place held.

        A slot's indices need no check of their own.  They would go stale if
        *e*'s targets rebound a name one of them reads, but FPy rejects a
        comprehension target that shadows an existing definition, and an index
        cannot read a name that has none.
        """
        return not _mentions(e.elt, {target})

    def _take_fill(self, e: ListComp) -> 'tuple[NamedId, tuple[Expr, ...]] | None':
        """The place the loops of *e* may write into, taken once.

        Keyed on the comprehension itself, and taken before the iterables are
        visited: a comprehension nested in one of them is not the statement's
        right-hand side and must not claim its place.
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

        Where a clause's iterable reads an earlier clause's target, the rows
        have different lengths and `fp.empty` has nowhere to get the total:
        there is no ``append``, and evaluating that iterable a second time to
        count first is not the same program.  So build the rows, add up their
        lengths, then flatten -- which is what
        ``derived-semantics.rst`` prescribes.

        Split at the *first* clause only.  Its iterable can read no target, so
        the outer comprehension is always independent, and the rest become one
        nested comprehension per row -- which a later pass lowers in turn, since
        it now sits in the loop body that is its statement slot.  `k` clauses
        peel one at a time and each flatten is one level deep.

        The rows are read twice, to count and to copy, so unlike a plain fill
        the temporary really is a second place.  That is the cost, and it is why
        this fires only on the dependent case.

        Every name here is a temporary the caller never wrote, so like the
        iterable bindings they all take `temp_id`.
        """
        loc = e.loc
        # the first clause's iterable, bound here like any other -- evaluated
        # once, and under the caller's `temp_id`
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

        # Every clause is independent, so each iterable is evaluated once, here.
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
        # Index the iterable where the loop reads no element from it, or where
        # it is a name: nothing is then loop-carried and the store index is the
        # loop variable itself, which `format_infer` bounds by the range.  A
        # carried counter widens instead, without bound where the trip count is
        # not static.  The exception is an *inlined* iterable with a target to
        # bind: subscripting a `range` would materialise the very list that
        # leaving it inline avoids.
        indexed = (
            not self._inlinable(iters[0])
            or isinstance(e.targets[0], UnderscoreId)
        )
        if len(e.targets) == 1 and indexed:
            idx = self.gensym.refresh(self.temp_id)
            src = iters[0]
            stmts: list[Stmt] = []
            if not isinstance(e.targets[0], UnderscoreId):
                # a discarded target binds nothing: the element cannot read it,
                # and a subscript has no effect to keep
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

        # Several clauses, or one over an iterable that is not indexed: nest the
        # loops over the original targets and carry a write index, rather than
        # linearizing it.
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
        # `z = [<elt> for <t> in <it>]` fills `z` itself.  Minting an `acc` and
        # copying it in leaves two names on one list, and a second name is a
        # second *place*: the cpp backend's `UnboxMode.STRICT` refuses it.
        #
        # Only where the element cannot read `z`, which the loops overwrite
        # before it runs.  An iterable may -- it is bound to a temp first, so it
        # still sees the list `z` held.
        offered = (
            isinstance(stmt.expr, ListComp)
            and isinstance(stmt.target, NamedId)
            and self._fillable(stmt.expr, stmt.target)
        )
        self._fill = (stmt.expr, stmt.target, ()) if offered else None
        s, _ = super()._visit_assign(stmt, ctx)
        if self._took_fill(offered):
            # `_lower` took the target, so the loops already write into it and
            # the assignment left over is `z = z`.  The loop stands in its place.
            return ctx.pop(), ctx
        return s, ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: Any):
        # `zs[i] = [<elt> for <t> in <it>]` allocates straight into the slot.
        # The nested comprehension of `[[...] for ...]` arrives in exactly this
        # shape once the outer one is lowered, and it is the last place an
        # accumulator would be left over.
        offered = (
            isinstance(stmt.expr, ListComp)
            and self._fillable(stmt.expr, stmt.var)
        )
        self._fill = (
            (stmt.expr, stmt.var, tuple(stmt.indices)) if offered else None
        )
        s, _ = super()._visit_indexed_assign(stmt, ctx)
        if self._took_fill(offered):
            return ctx.pop(), ctx
        return s, ctx

    def _visit_if_expr(self, e: IfExpr, ctx: Any) -> IfExpr:
        # A branch is conditional, so a loop hoisted out of it would run either
        # way.  The condition is unconditional and keeps its slot.
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, None)
        iff = self._visit_expr(e.iff, None)
        return IfExpr(cond, ift, iff, e.loc)

    def _visit_while(self, stmt: WhileStmt, ctx: Any):
        # The condition is re-evaluated every iteration and a loop hoisted before
        # the `while` runs once, freezing the comprehension at its first value --
        # which turns a terminating loop into an out-of-bounds slice.
        stmt, _ = super()._visit_while(stmt, None)
        return stmt, ctx

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


def _lister(func: FuncDef, dependent: bool = False) -> _CompToLoopInstance:
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
        dependent: bool = False,
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
        dependent: bool = False,
    ) -> list[tuple[Cursor, str]]:
        """Why each comprehension of `func` that is not a site was left alone."""
        return _lister(func, dependent).list_refusals(within)

    @staticmethod
    def apply(
        func: FuncDef, *, where: int | Cursor | None = None,
        temp_id: NamedId | None = None, dependent: bool = False,
    ) -> FuncDef:
        """
        Lowers every comprehension of `func` it can into an allocation plus a
        loop, and leaves the rest alone.

        `where` selects one comprehension by index in visit order; `None` takes
        every one it can lower.

        `dependent` also lowers a clause list whose length is a sum rather than
        a product.  Off by default: it materialises a row per outer element,
        where every other shape allocates once and fills, so it is worth it only
        to a caller that needs the program comprehension-free.
        """
        return CompToLoop.apply_with_edits(
            func, where=where, temp_id=temp_id, dependent=dependent,
        ).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef, *, where: int | Cursor | None = None,
        temp_id: NamedId | None = None, dependent: bool = False,
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

