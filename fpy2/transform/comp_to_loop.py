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
    Stmt,
    StmtBlock,
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

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        where: int | Cursor | None = None,
        temp_id: NamedId | None = None,
    ):
        self.func = func
        self.def_use = def_use
        self.temp_id = NamedId('t') if temp_id is None else temp_id
        self.gensym = Gensym(reserved=def_use.names())
        self.where = where

    # ------------------------------------------------------------------
    # Verification

    def _verify(self, e: ListComp) -> None | Declined:
        """`None` where *e* may be lowered, else why not."""
        if dependent_clauses(e):
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

    def _size(self, iters: list[NamedId], e: ListComp) -> Expr:
        """The comprehension's length: the product of the clause lengths.

        Every clause is independent -- a dependent one is left alone -- so each
        length is available here, before the loops.
        """
        loc = e.loc
        size: Expr = Len(None, Var(iters[0], loc), loc)
        for t in iters[1:]:
            size = Mul(size, Len(None, Var(t, loc), loc), loc)
        return size

    def _descend(self, e: ListComp, out: list) -> None:
        """Visit *e*'s children the way :meth:`_lower` would.

        The listing has to reach exactly what the rewrite reaches, or it counts a
        site the rewrite will not take: the iterables end up outside the loops
        and keep their statement slot, the element does not.
        """
        for iterable in e.iterables:
            self._visit_expr(iterable, out)
        self._visit_expr(e.elt, None)

    def _lower(self, e: ListComp, out: list) -> Expr:
        """Emit the allocation and loops into *out*; return the result `Var`."""
        loc = e.loc

        # Every clause is independent, so each iterable is evaluated once, here.
        iters: list[NamedId] = []
        for iterable in e.iterables:
            t = self.gensym.refresh(self.temp_id)
            out.append(Assign(t, None, self._visit_expr(iterable, out), loc))
            iters.append(t)

        acc = self.gensym.fresh('acc')
        size = self._size(iters, e)
        if isinstance(size, Len):
            # a bare length needs no arithmetic, so no exact-integer block
            out.append(Assign(acc, None, Empty(None, [size], loc), loc))
        else:
            n = self.gensym.fresh('n')
            out.append(integer_ctx([Assign(n, None, size, loc)], loc))
            out.append(Assign(acc, None, Empty(None, [Var(n, loc)], loc), loc))

        elt = self._visit_expr(e.elt, None)
        if len(e.targets) == 1:
            # One clause over a bound temporary: index it, so nothing is
            # loop-carried and the store index is the loop variable itself.
            idx = self.gensym.fresh('i')
            src = Var(iters[0], loc)
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
            inner = [ForStmt(
                copy_target(e.targets[j]), Var(iters[j], loc), StmtBlock(inner), loc,
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


def _lister(func: FuncDef) -> _CompToLoopInstance:
    """The pass instance a listing walks `func` with."""
    return _CompToLoopInstance(func, DefineUse.analyze(func))


class CompToLoop:
    """
    Transformation pass to lower a list comprehension into an allocation
    plus a loop.
    """

    @staticmethod
    def sites(func: FuncDef, within: Cursor | None = None) -> list[Cursor]:
        """The comprehensions of `func` this rewrite would lower, in visit
        order -- what a `where` index counts, and what `within` narrows.

        A comprehension this pass cannot lower is not a site: it neither appears
        here nor takes an index.  :meth:`refusals` says why each was left.
        """
        return _lister(func).list_sites(within)

    @staticmethod
    def refusals(
        func: FuncDef, within: Cursor | None = None
    ) -> list[tuple[Cursor, str]]:
        """Why each comprehension of `func` that is not a site was left alone."""
        return _lister(func).list_refusals(within)

    @staticmethod
    def apply(
        func: FuncDef, *, where: int | Cursor | None = None,
        temp_id: NamedId | None = None
    ) -> FuncDef:
        """
        Lowers every comprehension of `func` it can into an allocation plus a
        loop, and leaves the rest alone.

        `where` selects one comprehension by index in visit order; `None` takes
        every one it can lower.
        """
        return CompToLoop.apply_with_edits(
            func, where=where, temp_id=temp_id,
        ).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef, *, where: int | Cursor | None = None,
        temp_id: NamedId | None = None
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
        vtor = _CompToLoopInstance(func, def_use, where, temp_id)
        out = vtor.apply()
        vtor.check_site('a comprehension')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(func, out, tuple(vtor.edits), exprs_preserved=True)

