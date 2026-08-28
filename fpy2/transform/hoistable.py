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

**The non-strict positions.**  Four, and the list is closed -- every other
expression position is a statement operand or an operand of a strict operator:

- an ``IfExpr`` arm, which becomes an ``IfStmt`` assigning one name;
- an ``and``/``or`` tail, which becomes a flat chain of guarded statements;
- a ``while`` condition, whose loop is *rotated* -- the condition evaluated once
  before the loop and once at the end of the body, FPy's own order;
- a comprehension's element and iterables, which
  :class:`~fpy2.transform.CompToLoop` turns into an allocation and a loop.
  That pass is a caller's job and must run *first*: it creates the loop body
  that is the element's slot.

**Weaker than ANF, deliberately.**  :class:`~fpy2.transform.ANF` establishes the
same invariant, but as a side effect of also binding every nameable
subexpression to a name -- which the cpp emitter needs and a rewrite does not.
Over ``examples/`` ANF names up to 5096 subexpressions where only 71 positions
need a lowering.  This pass does the lowering half and leaves ``anf.py`` alone.

**The ordering hazard.**  Lowering alone is *not* semantics-preserving.  Hoisting
a lowered construct out of an operand moves it above the operands to its left,
which are then evaluated later than they were:

.. code-block:: python

    return g(a) + (h(b) if c else 0.0)   # raises g's assertion

    if c: t = h(b)                       # naive lowering
    else: t = 0.0
    return g(a) + t                      # raises h's assertion -- wrong

ANF avoids this only because atomization names ``g(a)`` too, left to right.  So
this pass keeps *part* of the naming, and only that part: see
:func:`force_names`.

The design notes are in ``docs/todos/hoistable-form.md``.
"""

import dataclasses
from typing import Any

from ..analysis import DefineUse, Reachability, SyntaxCheck
from ..analysis.define_use import DefineUseAnalysis
from ..ast.fpyast import (
    And,
    AssertStmt,
    Assign,
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
from ..ast.visitor import DefaultTransformVisitor, DefaultVisitor
from ..number import REAL
from ..utils import Gensym
from .anf import _ATOMIC, _SEALED_REASON, _reads
from .path import sub_exprs


def lowers(e: Expr) -> bool:
    """Whether this pass emits a statement *at* `e`.

    A ternary lowers whenever an arm is not an atom, and a chain whenever an
    operand after the first is not one: those are exactly the operands with
    nowhere to put a statement.  Both criteria are stricter than
    :func:`~fpy2.transform.anf.needs_slot`, which asks whether a *particular*
    lowering wants a slot rather than whether one could ever be wanted.
    """
    match e:
        case IfExpr():
            return not (
                isinstance(e.ift, _ATOMIC) and isinstance(e.iff, _ATOMIC)
            )
        case And() | Or():
            return any(not isinstance(a, _ATOMIC) for a in e.args[1:])
        case _:
            return False


def lowers_inside(e: Expr) -> bool:
    """Whether this pass emits a statement anywhere in `e`, `e` itself included.

    A comprehension is the only *sealed* position needing an exception.  The
    other two seal an unlowered ternary's arms and an unlowered chain's tail --
    but unlowered means those operands are atoms, and an atom has no children,
    so the recursion finds nothing there on its own.
    """
    if lowers(e):
        return True
    if isinstance(e, ListComp):
        return False
    return any(lowers_inside(sub) for _field, _i, sub in sub_exprs(e))


def force_names(node: 'Stmt | Expr') -> set[Expr]:
    """The expressions in `node` to bind to a name, so a lowering to their right
    does not overtake them.

    The *prefix rule*: at any node, let ``last`` be the position of the last
    child -- in :func:`~fpy2.transform.path.sub_exprs` order, which is
    evaluation order -- that a lowering fires inside.  Every earlier child that
    is not already an atom is named, since a lowering hoists above the whole
    statement and would otherwise run before them.

    .. code-block:: python

        f(g(y), a if c else b)   # -> {g(y)}: the ternary hoists above it
        f(a if c else b, g(y))   # -> {}: nothing runs before the ternary
        xs[i + 1] = a if c else b  # -> {i + 1}: an index runs before the value

    A lowered ``IfExpr`` or chain is exempt: its condition lands in the
    ``IfStmt`` condition and each arm in a block of its own, so their order is
    preserved structurally -- and naming an arm is the very bug the rule exists
    to prevent.  A comprehension is sealed.

    Identity, not structure: ``Expr`` defines no ``__eq__``, so the set holds
    the nodes themselves and two structurally-equal operands stay distinct.

    Blocks are not entered.  A statement of a nested block gets its own call,
    since its own block is the slot its temporaries belong in.
    """
    out: set[Expr] = set()
    _collect(node, out)
    return out


def _collect(node: 'Stmt | Expr', out: set[Expr]) -> None:
    """Accumulate :func:`force_names` for `node` and everything under it."""
    if isinstance(node, ListComp):
        return
    kids = [sub for _field, _i, sub in sub_exprs(node)]
    # A ternary or chain is exempt: lowered, its condition lands in the
    # `IfStmt` condition and each arm in a block of its own; unlowered, its
    # arms are atoms and the rule would find nothing anyway.
    if not isinstance(node, (IfExpr, And, Or)):
        lowering = [i for i, kid in enumerate(kids) if lowers_inside(kid)]
        if lowering:
            out.update(
                kid for kid in kids[:max(lowering)]
                if not isinstance(kid, _ATOMIC)
            )
    for kid in kids:
        _collect(kid, out)


# ----------------------------------------------------------------------
# The residue


def _list_refusals(func: FuncDef) -> list[tuple[Expr, str]]:
    """The sealed positions of *func* still holding a non-atom.

    See :meth:`Hoistable.refusals`, the public entry point.  ``anf.py`` asks
    :func:`~fpy2.transform.anf.needs_slot` of the same positions; this asks only
    whether the operand is an atom, matching the gates in :func:`lowers`.
    """
    out: list[tuple[Expr, str]] = []

    def check(e: Expr, why: str) -> None:
        if not isinstance(e, _ATOMIC):
            out.append((e, _SEALED_REASON[why]))

    class _Residue(DefaultVisitor):
        def _visit_if_expr(self, e: IfExpr, ctx):
            check(e.ift, 'ternary')
            check(e.iff, 'ternary')
            super()._visit_if_expr(e, ctx)

        def _visit_naryop(self, e: NaryOp, ctx):
            if isinstance(e, (And, Or)):
                for arg in e.args[1:]:
                    check(arg, 'chain')
            super()._visit_naryop(e, ctx)

        def _visit_list_comp(self, e: ListComp, ctx):
            # Reported once, and not descended into: the comprehension *is* why
            # nothing inside it can be hoisted, so a ternary in its element is
            # that entry rather than one of its own.  `CompToLoop` clears both.
            check(e.elt, 'element')
            for iterable in e.iterables:
                check(iterable, 'iterable')

        def _visit_while(self, stmt: WhileStmt, ctx):
            check(stmt.cond, 'condition')
            super()._visit_while(stmt, ctx)

    _Residue()._visit_function(func, None)
    return out


# ----------------------------------------------------------------------
# The rewrite


@dataclasses.dataclass
class _Ctx:
    """Block-walk accumulator.

    ``stmts`` is the block being built: a statement visitor appends the bindings
    it needs, and the block visitor appends the rewritten statement after them.
    ``force`` is :func:`force_names` for the statement being visited, the only
    expressions this pass binds to a name.  ``hoistable`` is false inside a
    sealed position, where nothing may be named.
    """

    stmts: list[Stmt]
    force: frozenset[Expr] = frozenset()
    hoistable: bool = True

    def sealed(self) -> '_Ctx':
        return dataclasses.replace(self, hoistable=False)

    def buffer(self) -> '_Ctx':
        """A copy accumulating into a fresh statement list of its own -- a block
        that did not exist before the lowering that needs it."""
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

        Unlike ANF this asks nothing about *e*'s type: a left operand keeps its
        place whatever it holds, so this pass can name an aggregate.
        """
        rebuilt = super()._visit_expr(e, ctx)
        if not ctx.hoistable or e not in ctx.force or isinstance(rebuilt, _ATOMIC):
            return rebuilt
        t = self.gensym.fresh(self.prefix)
        ctx.stmts.append(Assign(t, None, rebuilt, e.loc))
        return Var(t, e.loc)

    def _in_place(self, e: Expr, ctx: _Ctx) -> Expr:
        """*e* rebuilt without being named itself.

        For an operand nothing follows, where the prefix rule never fires and
        naming would only add a copy.
        """
        return super()._visit_expr(e, ctx)

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
            # which belong *before* the `if`
            stmt = self._branch_on(e, t, ctx)
            ctx.stmts.append(stmt)
            return Var(t, e.loc)
        # Left alone, so both arms are atoms.  The condition is evaluated
        # whenever the ternary is, and takes the ternary's own slot.
        cond = self._visit_expr(e.cond, ctx)
        sealed = ctx.sealed()
        return IfExpr(
            cond,
            self._in_place(e.ift, sealed),
            self._in_place(e.iff, sealed),
            e.loc,
        )

    def _branch_on(self, e: IfExpr, target: NamedId, ctx: _Ctx) -> IfStmt:
        """*e* as an ``IfStmt`` assigning *target* in each branch.

        Appends the condition's own statements to *ctx*, since the condition is
        evaluated where the ternary was.
        """
        cond = self._in_place(e.cond, ctx)
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
            if not _reads(target, e.args[1:]):
                return self._short_circuit(e, ctx, target)
            # A chain accumulates into its target *before* the later operands
            # run, so one that reads the target would see the accumulator.  Only
            # a chain: a ternary arm and an ordinary right-hand side are both
            # evaluated before anything is assigned.
            acc = self.gensym.fresh(self.prefix)
            ctx.stmts.append(self._short_circuit(e, ctx, acc))
            return Assign(target, None, Var(acc, loc), loc)
        return Assign(target, None, self._in_place(e, ctx), loc)

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
            else self._in_place(a, sealed)
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

    def _visit_list_comp(self, e: ListComp, ctx: _Ctx):
        # The element runs once per iteration, and a later clause's iterable may
        # read an earlier clause's target, so the whole comprehension is sealed.
        return super()._visit_list_comp(e, ctx.sealed())

    # ------------------------------------------------------------------
    # Statements

    def _visit_block(self, block: StmtBlock, ctx: Any):
        # A fresh buffer per block, and a fresh `force` per statement: a
        # temporary belongs to the block whose statement needs it, and the
        # prefix rule is asked of one statement at a time.
        inner = _Ctx(stmts=[])
        for stmt in block.stmts:
            s, _ = self._visit_statement(
                stmt, dataclasses.replace(inner, force=frozenset(force_names(stmt))),
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
        expr = self._in_place(stmt.expr, ctx)
        return Assign(stmt.target, stmt.type, expr, stmt.loc), ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _Ctx):
        # `_visit_expr`, not `_in_place`: an index runs before the value, so the
        # prefix rule can name it.
        indices = [self._visit_expr(i, ctx) for i in stmt.indices]
        expr = self._in_place(stmt.expr, ctx)
        return IndexedAssign(stmt.var, indices, expr, stmt.loc), ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: _Ctx):
        return ReturnStmt(self._in_place(stmt.expr, ctx), stmt.loc), ctx

    def _visit_if1(self, stmt: If1Stmt, ctx: _Ctx):
        cond = self._in_place(stmt.cond, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        return If1Stmt(cond, body, stmt.loc), ctx

    def _visit_if(self, stmt: IfStmt, ctx: _Ctx):
        cond = self._in_place(stmt.cond, ctx)
        ift, _ = self._visit_block(stmt.ift, ctx)
        iff, _ = self._visit_block(stmt.iff, ctx)
        return IfStmt(cond, ift, iff, stmt.loc), ctx

    def _visit_while(self, stmt: WhileStmt, ctx: _Ctx):
        if isinstance(stmt.cond, _ATOMIC):
            # already a place; nothing in it to hoist
            body, _ = self._visit_block(stmt.body, ctx)
            return WhileStmt(self._in_place(stmt.cond, ctx), body, stmt.loc), ctx
        return self._rotate(stmt, ctx), ctx

    def _rotate(self, stmt: WhileStmt, ctx: _Ctx) -> WhileStmt:
        """*stmt* with its condition evaluated through a name, once before the
        loop and once at the end of the body -- FPy's own order, so each copy
        sits in a slot running as often as the condition does.

        The copies share no nodes: :meth:`_in_place` rebuilds every one, so
        neither needs cloning.  A body that always returns gets no second copy --
        the loop runs at most one iteration, and a statement after the ``return``
        is unreachable, which the syntax checker rejects.
        """
        c = self.gensym.fresh('c')
        ctx.stmts.append(
            Assign(c, None, self._in_place(stmt.cond, ctx), stmt.loc),
        )
        body, _ = self._visit_block(stmt.body, ctx)
        if Reachability.analyze(body).has_fallthrough:
            # The body's own block is the per-iteration slot.  `ctx` carries the
            # `while` statement's own `force`, not the body's last statement's.
            tail = dataclasses.replace(ctx, stmts=body.stmts)
            again = self._in_place(stmt.cond, tail)
            body.stmts.append(Assign(c, None, again, stmt.loc))
        return WhileStmt(Var(c, stmt.loc), body, stmt.loc)

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx):
        iterable = self._in_place(stmt.iterable, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        return ForStmt(stmt.target, iterable, body, stmt.loc), ctx

    def _visit_context(self, stmt: ContextStmt, ctx: _Ctx):
        """A ``with`` statement, whose two halves round differently.

        **E-Context** evaluates the context expression under ``REAL``, not the
        active context, so anything hoisted out of it goes in a ``with fp.REAL:``
        block of its own rather than the enclosing one.  FPy scopes the context
        but not the store, so the names stay visible.
        """
        under_real = ctx.buffer()
        context = self._in_place(stmt.ctx, under_real)
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
        # `_visit_expr` for the test: the message runs after it, so a lowering
        # there can name the test.
        test = self._visit_expr(stmt.test, ctx)
        msg = None if stmt.msg is None else self._in_place(stmt.msg, ctx)
        return AssertStmt(test, msg, stmt.loc), ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: _Ctx):
        return EffectStmt(self._in_place(stmt.expr, ctx), stmt.loc), ctx


class Hoistable:
    """
    Transformation pass rewriting a function into hoistable form.

    Every expression node ends up evaluated exactly once, unconditionally,
    whenever its enclosing statement is reached -- so the slot before that
    statement is always a legal place for a hoisted temporary.  See the module
    docstring.
    """

    @staticmethod
    def refusals(func: FuncDef) -> list[tuple[Expr, str]]:
        """Every sealed position of `func` still holding a non-atom.

        Empty is the invariant: `not Hoistable.refusals(f)` says a temporary may
        be hoisted out of anywhere in `f`.  After this pass only a comprehension
        can appear, and only one :class:`~fpy2.transform.CompToLoop` declined --
        which is why that pass runs first.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got `{func}`')
        return _list_refusals(func)

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Rewrites `func` into hoistable form."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got `{func}`')
        def_use = DefineUse.analyze(func)
        out = _HoistableInstance(func, def_use).apply()
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
