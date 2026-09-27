"""
Hoistable form: every expression sits where a statement may be inserted above it.

FPy has statements and expressions, so a pass needing a temporary for an
expression has to hoist it into a statement above -- and that is not always
sound.  This pass establishes the invariant that makes it always sound:

    Every expression node is evaluated exactly once, unconditionally, whenever
    its enclosing statement is reached.

Under it the slot immediately before the enclosing statement runs exactly as
often, and under exactly the same condition, as any expression in that
statement, so it is always a legal place for a hoisted temporary.  A pass mints
one on demand and never reasons about conditional evaluation again.

**The non-strict positions.**  Being a statement operand does not imply
strictness -- an ``assert``'s message is one and is evaluated only on failure --
so the list is by enumeration:

- an ``IfExpr`` arm, which becomes an ``IfStmt`` assigning one name;
- an ``and``/``or`` tail, which becomes a flat chain of guarded statements;
- a ``while`` condition, whose loop is *rotated* -- the condition evaluated once
  before the loop and once at the end of the body, FPy's own order;
- a comprehension's element and iterables, which
  :class:`~fpy2.transform.CompToLoop` turns into an allocation and a loop.
  That pass is a caller's job and must run *first*: it creates the loop body
  that is the element's slot;
- an ``assert``'s message, which has no lowering at all;
- a chained comparison's third operand and beyond: ``a < b < c`` is the
  conjunction of the adjacent pairs, so ``c`` runs only where ``a < b`` held.

The last two are left sealed and reported by :meth:`Hoistable.refusals`, so the
invariant holds of everything that report is silent about.

**Why the gates are syntactic.**  A ternary lowers when an arm is not an atom, a
chain when an operand after the first is not one, and a loop rotates when its
condition is not one -- not :func:`~fpy2.transform.anf.needs_slot`, which
predicts what an emitter will want a statement for rather than describing the
program.  Gated on that, this pass would leave ``while x*y > 0`` un-rotated, so
its own output would not be in hoistable form.  A gate that cannot state its own
postcondition is not a normal form.

:class:`~fpy2.transform.ANF` is the strong normalization above this one, and
requires it.

**The ordering hazard.**  Lowering alone is *not* semantics-preserving.  Hoisting
a lowered construct out of an operand moves it above the operands to its left,
which are then evaluated later than they were:

.. code-block:: python

    return g(a) + (h(b) if c else 0.0)   # raises g's assertion

    if c: t = h(b)                       # naive lowering
    else: t = 0.0
    return g(a) + t                      # raises h's assertion -- wrong

So the pass names exactly as much as that costs, and no more: see
:func:`~fpy2.analysis.hoistability.force_names`.  Where it sits in the cpp pipeline is §1 of
``docs/todos/backend-independence.md``.
"""

import dataclasses
from typing import Any

from ..analysis import DefineUse, Reachability, SyntaxCheck
from ..analysis.define_use import DefineUseAnalysis
from ..analysis.hoistability import (
    ATOMIC,
    SEALED_REASON,
    Hoistability,
    force_names,
    lowers,
)
from ..ast.accessors import vars_in
from ..ast.fpyast import (
    And,
    AssertStmt,
    Assign,
    Compare,
    ContextStmt,
    EffectStmt,
    Expr,
    ForeignVal,
    ForStmt,
    FuncDef,
    If1Stmt,
    IfExpr,
    IfStmt,
    IndexedAssign,
    ListComp,
    NamedId,
    NaryOp,
    Not,
    Or,
    ReturnStmt,
    Stmt,
    StmtBlock,
    UnderscoreId,
    Var,
    WhileStmt,
)
from ..ast.visitor import DefaultTransformVisitor
from ..number import REAL
from ..utils import Gensym
from .utils import name_forced

# ----------------------------------------------------------------------
# The rewrite


@dataclasses.dataclass
class _Ctx:
    """Block-walk accumulator.

    ``stmts`` is the block being built: a statement visitor appends the bindings
    it needs, and the block visitor appends the rewritten statement after them.
    ``force`` is :func:`~fpy2.analysis.hoistability.force_names` for the
    statement being visited.
    ``hoistable`` is false inside a sealed position, where nothing may be named.
    """

    stmts: list[Stmt]
    force: frozenset[Expr] = frozenset()
    hoistable: bool = True

    def sealed(self) -> '_Ctx':
        return dataclasses.replace(self, hoistable=False)

    def buffer(self) -> '_Ctx':
        """A copy accumulating into a fresh statement list of its own."""
        return dataclasses.replace(self, stmts=[])


class _HoistableInstance(DefaultTransformVisitor):
    """Single-use instance of the pass."""

    func: FuncDef
    gensym: Gensym
    prefix: str
    """Base name for the temporaries this instance mints."""

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis, prefix: str = 't'):
        self.func = func
        self.gensym = Gensym(reserved=def_use.names())
        self.prefix = prefix

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    # ------------------------------------------------------------------
    # Naming

    def _visit_expr(self, e: Expr, ctx: _Ctx) -> Expr:
        """*e* rebuilt, and bound to a fresh name where the prefix rule says so.

        Nothing is asked about *e*'s type: a left operand keeps its place
        whatever it holds, so this can name an aggregate.
        """
        rebuilt = super()._visit_expr(e, ctx)
        if not ctx.hoistable:
            return rebuilt
        fresh = lambda: self.gensym.fresh(self.prefix)
        return name_forced(e, rebuilt, ctx.force, ctx.stmts, fresh)

    def _lowered(self, e: Expr, ctx: _Ctx) -> bool:
        """Whether *e* becomes statements here.  :func:`lowers` says whether the
        shape calls for it, and ``hoistable`` whether there is a slot to put
        them in."""
        return ctx.hoistable and lowers(e)

    # ------------------------------------------------------------------
    # The two expression lowerings

    def _visit_if_expr(self, e: IfExpr, ctx: _Ctx):
        if self._lowered(e, ctx):
            t = self.gensym.fresh(self.prefix)
            # two steps: `_branch_on` appends the condition's own statements,
            # which belong before the `if`
            stmt = self._branch_on(e, t, ctx)
            ctx.stmts.append(stmt)
            return Var(t, e.loc)
        # left alone, so both arms are atoms
        cond = self._visit_expr(e.cond, ctx)
        sealed = ctx.sealed()
        return IfExpr(
            cond,
            self._visit_expr(e.ift, sealed),
            self._visit_expr(e.iff, sealed),
            e.loc,
        )

    def _branch_on(self, e: IfExpr, target: NamedId, ctx: _Ctx) -> IfStmt:
        """*e* as an ``IfStmt`` assigning *target* in each branch.

        Appends the condition's own statements to *ctx*, since the condition is
        evaluated where the ternary was.
        """
        cond = self._visit_expr(e.cond, ctx)
        return IfStmt(
            cond,
            self._arm(target, e.ift, ctx, e.loc),
            self._arm(target, e.iff, ctx, e.loc),
            e.loc,
        )

    def _bind(self, target: NamedId, e: Expr, ctx: _Ctx, loc) -> Stmt:
        """The statement binding *target* to *e*, appending *e*'s own statements
        to *ctx* first.

        A lowered ternary or chain accumulates into *target* directly, so
        nesting them gives one ladder rather than a chain of copies.
        """
        if self._lowered(e, ctx):
            if isinstance(e, IfExpr):
                return self._branch_on(e, target, ctx)
            assert isinstance(e, (And, Or))
            if not any(v.name == target for a in e.args[1:] for v in vars_in(a)):
                return self._short_circuit(e, ctx, target)
            # a chain assigns its target before the later operands run, so one
            # that reads the target would see the accumulator
            acc = self.gensym.fresh(self.prefix)
            ctx.stmts.append(self._short_circuit(e, ctx, acc))
            return Assign(target, None, Var(acc, loc), loc)
        return Assign(target, None, self._visit_expr(e, ctx), loc)

    def _arm(self, target: NamedId, e: Expr, ctx: _Ctx, loc) -> StmtBlock:
        """A block binding *target* to *e*, with *e*'s own statements inside it
        -- the slot the arm lacked, running exactly when the arm did."""
        inner = ctx.buffer()
        inner.stmts.append(self._bind(target, e, inner, loc))
        return StmtBlock(inner.stmts)

    def _visit_naryop(self, e: NaryOp, ctx: _Ctx):
        if not isinstance(e, (And, Or)):
            return super()._visit_naryop(e, ctx)
        if self._lowered(e, ctx):
            t = self.gensym.fresh(self.prefix)
            ctx.stmts.append(self._short_circuit(e, ctx, t))
            return Var(t, e.loc)
        # Short-circuit: the first operand always runs, the rest do not.
        sealed = ctx.sealed()
        args = [
            self._visit_expr(a, ctx) if i == 0
            else self._visit_expr(a, sealed)
            for i, a in enumerate(e.args)
        ]
        return type(e)(args, e.loc)

    def _short_circuit(self, e: 'And | Or', ctx: _Ctx, target: NamedId) -> Stmt:
        """*e* accumulated into *target*, one guard per operand after the first.

        The guards are *flat* and short-circuit all the same: once an ``or``'s
        accumulator is true every later ``if not t`` fails, and dually for
        ``and``.

        .. code-block:: python

            t = a
            if not t: t = b     # only where `a` was false
            if not t: t = c

        All but the last statement are appended to *ctx* and the last returned,
        so a caller with one statement slot has one to give back.
        """
        stmts: list[Stmt] = [self._bind(target, e.args[0], ctx, e.loc)]
        for arg in e.args[1:]:
            read = Var(target, e.loc)
            guard = read if isinstance(e, And) else Not(read, e.loc)
            stmts.append(If1Stmt(guard, self._arm(target, arg, ctx, e.loc), e.loc))
        ctx.stmts.extend(stmts[:-1])
        return stmts[-1]

    def _visit_compare(self, e: Compare, ctx: _Ctx):
        """``a < b < c`` is the conjunction of the adjacent pairs, so every
        operand after the second runs only where the earlier tests held."""
        sealed = ctx.sealed()
        args = [
            self._visit_expr(a, ctx if i < 2 else sealed)
            for i, a in enumerate(e.args)
        ]
        return Compare(e.ops, args, e.loc)

    def _visit_list_comp(self, e: ListComp, ctx: _Ctx):
        # The element runs once per iteration, and a later clause's iterable may
        # read an earlier clause's target, so the whole comprehension is sealed.
        return super()._visit_list_comp(e, ctx.sealed())

    # ------------------------------------------------------------------
    # Statements

    def _visit_block(self, block: StmtBlock, ctx: Any):
        # a temporary belongs to the block whose statement needs it, and the
        # prefix rule is asked of one statement at a time
        inner = _Ctx(stmts=[])
        for stmt in block.stmts:
            s, _ = self._visit_statement(
                stmt, dataclasses.replace(inner, force=frozenset(force_names(stmt, lowers))),
            )
            inner.stmts.append(s)
        return StmtBlock(inner.stmts), ctx

    def _visit_assign(self, stmt: Assign, ctx: _Ctx):
        if isinstance(stmt.target, NamedId) and stmt.type is None:
            # A lowered right-hand side assigns this name directly rather than a
            # temporary this statement copies.  Not where the assignment carries
            # a type annotation, which has one place to sit and several branches
            # to sit in.
            return self._bind(stmt.target, stmt.expr, ctx, stmt.loc), ctx
        expr = self._visit_expr(stmt.expr, ctx)
        return Assign(stmt.target, stmt.type, expr, stmt.loc), ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _Ctx):
        indices = [self._visit_expr(i, ctx) for i in stmt.indices]
        expr = self._visit_expr(stmt.expr, ctx)
        return IndexedAssign(stmt.var, indices, expr, stmt.loc), ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: _Ctx):
        return ReturnStmt(self._visit_expr(stmt.expr, ctx), stmt.loc), ctx

    def _visit_if1(self, stmt: If1Stmt, ctx: _Ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        return If1Stmt(cond, body, stmt.loc), ctx

    def _visit_if(self, stmt: IfStmt, ctx: _Ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        ift, _ = self._visit_block(stmt.ift, ctx)
        iff, _ = self._visit_block(stmt.iff, ctx)
        return IfStmt(cond, ift, iff, stmt.loc), ctx

    def _visit_while(self, stmt: WhileStmt, ctx: _Ctx):
        if isinstance(stmt.cond, ATOMIC):
            # already a place; nothing in it to hoist
            body, _ = self._visit_block(stmt.body, ctx)
            return WhileStmt(self._visit_expr(stmt.cond, ctx), body, stmt.loc), ctx
        return self._rotate(stmt, ctx), ctx

    def _rotate(self, stmt: WhileStmt, ctx: _Ctx) -> WhileStmt:
        """*stmt* with its condition evaluated through a name, once before the
        loop and once at the end of the body -- FPy's own order, so each copy
        sits in a slot running as often as the condition does.

        The copies share no nodes -- each is rebuilt, so neither needs cloning.
        A body that always returns gets no second copy: the loop runs at most one
        iteration, and a statement after the ``return`` is unreachable, which the
        syntax checker rejects.
        """
        c = self.gensym.fresh('c')
        ctx.stmts.append(
            Assign(c, None, self._visit_expr(stmt.cond, ctx), stmt.loc),
        )
        body, _ = self._visit_block(stmt.body, ctx)
        if Reachability.analyze(body).has_fallthrough:
            # the body's own block is the per-iteration slot; `ctx` carries the
            # `while`'s own `force`, not the body's last statement's
            tail = dataclasses.replace(ctx, stmts=body.stmts)
            again = self._visit_expr(stmt.cond, tail)
            body.stmts.append(Assign(c, None, again, stmt.loc))
        return WhileStmt(Var(c, stmt.loc), body, stmt.loc)

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx):
        iterable = self._visit_expr(stmt.iterable, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        return ForStmt(stmt.target, iterable, body, stmt.loc), ctx

    def _visit_context(self, stmt: ContextStmt, ctx: _Ctx):
        """A ``with`` statement, whose two halves round differently.

        As in :meth:`fpy2.transform.anf._ANFInstance._visit_context`:
        **E-Context** evaluates the context expression under ``REAL``, so
        anything hoisted out of it goes in a ``with fp.REAL:`` block of its own
        rather than the enclosing one.
        """
        under_real = ctx.buffer()
        context = self._visit_expr(stmt.ctx, under_real)
        if under_real.stmts:
            ctx.stmts.append(ContextStmt(
                UnderscoreId(),
                ForeignVal(REAL, stmt.loc),
                StmtBlock(under_real.stmts),
                stmt.loc,
            ))
        body, _ = self._visit_block(stmt.body, ctx)
        return ContextStmt(stmt.target, context, body, stmt.loc), ctx

    def _visit_assert(self, stmt: AssertStmt, ctx: _Ctx):
        # the message is evaluated only where the test fails, and there is no
        # slot that runs then -- so it is sealed, like a comprehension
        test = self._visit_expr(stmt.test, ctx)
        sealed = ctx.sealed()
        msg = None if stmt.msg is None else self._visit_expr(stmt.msg, sealed)
        return AssertStmt(test, msg, stmt.loc), ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: _Ctx):
        return EffectStmt(self._visit_expr(stmt.expr, ctx), stmt.loc), ctx


class Hoistable:
    """
    Transformation pass rewriting a function into hoistable form.

    Every expression node ends up evaluated exactly once, unconditionally,
    whenever its enclosing statement is reached, so the slot before that
    statement is always a legal place for a hoisted temporary.
    """

    @staticmethod
    def refusals(func: FuncDef) -> list[tuple[Expr, str]]:
        """Every sealed position of `func` still holding a non-atom.

        Empty is the invariant: a temporary may be hoisted out of anywhere in
        `func`.  Afterwards only the two positions with no lowering can appear --
        a comprehension, and an ``assert`` message.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got `{func}`')
        return [
            (e, SEALED_REASON[why]) for e, why in Hoistability.analyze(func).sealed
            if not isinstance(e, ATOMIC)
        ]

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Rewrites `func` into hoistable form."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got `{func}`')
        def_use = DefineUse.analyze(func)
        out = _HoistableInstance(func, def_use).apply()
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
