"""
Administrative normal form: bind every non-atomic subexpression to a name.

FPy expressions nest, so a backend emitting a construct that needs a *place* for
an operand has to invent one.  This pass removes the need rather than predicting
it: afterwards every proper subexpression of a statement is an atom, so an
operand a backend meets is already a name.

.. code-block:: python

    # before
    y = (a * b) + (c * d)

    # after
    t = a * b
    t1 = c * d
    y = t + t1

**Where a temporary goes.**  In a statement slot executed *exactly as often, and
under exactly the same condition, as the expression it names* -- not merely the
nearest enclosing statement, which is wrong for a ``while`` condition (evaluated
once per iteration where the slot before the loop runs once) and for a ternary
arm (conditional where the slot is not).  A statement's own operands qualify, and
so does anything inside a body, which is the same case one level down.

The positions FPy evaluates conditionally or repeatedly do not, and this pass
*seals* them: an ``and``/``or`` tail and a comprehension's element keep their
subexpressions inline, because hoisting one out would evaluate it on a path FPy
never takes.

Two of those positions have a lowering that *creates* the slot they lack, and it
is applied where the position holds something that needs one
(:func:`needs_slot`).  A ``while`` loop is **rotated**, so the condition is
evaluated once before the loop and once at the end of the body -- exactly FPy's
own order -- and each copy has a slot of its own.  An ``IfExpr`` becomes an
``IfStmt`` whose branches assign one name, which is the same trade: each arm's
block runs exactly when the arm did.

.. code-block:: python

    # before
    y = (a * b) if c else d

    # after
    if c:
        y = a * b
    else:
        y = d

A ternary is left alone only where it is already in normal form -- ``x1 if c
else x2`` over atoms -- and a bool chain likewise.  Not for the backend's sake:
the cpp emitter spells either inline and needs no place for one.  It is that a
sealed position is unreachable to every pass that needs a preamble.
:class:`~fpy2.transform.RoundElim` and :class:`~fpy2.transform.RoundInsert`
suppress hoisting inside an ``IfExpr`` branch for exactly the reason this pass
seals it, so no rounding can be eliminated or inserted there --
:mod:`fpy2.transform.comp_to_loop`'s own reason for existing, applied to
ternaries and bool tails.  Flattening them makes them schedulable.

A rotation is the exception, because it *duplicates* its condition rather than
restructuring it.  That cost is real, so it is gated on :func:`needs_slot`
instead: a condition needing no place stays an expression.

The two compose, and that is why they are worth having together: the condition a
rotation lifts sits in a slot, so an ``IfExpr`` inside it can then be lowered
too, and a lowered arm is a block, so an ``IfExpr`` nested in one can be as
well.

.. code-block:: python

    # before
    while max(xs) > 0.0:
        <body>

    # after
    t = max(xs)
    c = t > 0.0
    while c:
        <body>
        t1 = max(xs)
        c = t1 > 0.0

So the invariant is not "no ternary survives", nor textbook ANF: it is that no
expression needing a place sits where there is none, and that every position a
lowering reaches for free is flattened.

**Scalars only.**  An expression that is, or may hold, a list or a tuple is left
unnamed (:data:`_AGGREGATE`): naming it would give it a storage place of its own,
and the C++ backend's storage and sharing analyses answer differently about a
second place.  Their children are still named, so ``f(a * b)`` becomes
``t = a * b; f(t)``.

Idempotent: a second pass finds only atoms in the positions it names.
"""

import dataclasses
from typing import Any

from ..analysis import DefineUse, DefineUseAnalysis, Reachability, SyntaxCheck
from ..ast.fpyast import (
    AllOf,
    AMax,
    AMin,
    And,
    AnyOf,
    AssertStmt,
    Assign,
    Attribute,
    BinaryOp,
    Call,
    Cast,
    Compare,
    ContextStmt,
    EffectStmt,
    Empty,
    Enumerate,
    Expr,
    ForStmt,
    Fst,
    FuncDef,
    If1Stmt,
    IfExpr,
    IfStmt,
    IndexedAssign,
    ListComp,
    ListExpr,
    ListRef,
    ListSlice,
    Max,
    Min,
    NamedId,
    NaryOp,
    Not,
    NullaryOp,
    Or,
    Range1,
    Range2,
    Range3,
    ReturnStmt,
    Round,
    RoundAt,
    Snd,
    Stmt,
    StmtBlock,
    Sum,
    TernaryOp,
    TupleExpr,
    UnaryOp,
    ValueExpr,
    Var,
    WhileStmt,
    Zip,
)
from ..ast.visitor import DefaultTransformVisitor
from ..utils import Gensym
from .path import sub_exprs

_ATOMIC = (Var, ValueExpr, NullaryOp)
"""Expressions that are already a place, or need none.

A ``NullaryOp`` is a constant the active context fixes, so it has no children to
name and naming it would buy nothing."""

_AGGREGATE = (
    ListExpr, TupleExpr, ListComp, ListSlice, Empty, Zip, Enumerate,
    Range1, Range2, Range3, Call, Fst, Snd, ListRef, IfExpr, Attribute,
)
"""Expressions this pass does not name, because each is or may hold an aggregate.

A ``Call`` and an ``Attribute`` because their result type is not syntactic; the
container forms because they are one; ``Fst``/``Snd``/``ListRef`` because a
projection can select one -- and because the cpp emitter folds an ``Fst``/``Snd``
chain into a single ``std::get``, which naming each level would break; ``IfExpr``
because its arms decide, and they are sealed here anyway.
"""


_SLOT_FREE = (
    Var, ValueExpr, NullaryOp,
    Compare, Not, And, Or, IfExpr,
    UnaryOp, BinaryOp, TernaryOp,
)
"""Node kinds whose own lowering needs no statement of its own.

A whitelist, so a kind nobody thought about needs a slot: a false *positive*
lowers a position that did not need it, which costs output size, while a false
negative leaves an operand in a position with nowhere to put its statement --
the shape ``docs/todos/backend-cpp.md`` records as a miscompile.
"""

_NEEDS_SLOT = (
    Round, RoundAt, Cast,
    Sum, AMin, AMax, AnyOf, AllOf,
    Range1, Range2, Range3, Enumerate,
    Fst, Snd,
)
"""The exceptions among :data:`_SLOT_FREE`'s base classes.

A rounding may assert; a fold over a list needs a loop and an accumulator; a
range allocates; and a projection is read through a bound name, since the cpp
emitter folds an ``Fst``/``Snd`` chain into one ``std::get``.  Everything not
matched by either tuple -- a call, a container, a comprehension, ``Min``/``Max``
-- needs a slot by default.
"""


def needs_slot(e: Expr) -> bool:
    """Does emitting *e* plausibly require a statement of its own?

    Asked of a position FPy evaluates conditionally or repeatedly, to decide
    whether it is worth lowering: a condition or an arm holding only names,
    literals and arithmetic can stay an expression, since no backend needs
    anywhere to put a statement.  Conservative in the safe direction -- see
    :data:`_SLOT_FREE`.
    """
    if not isinstance(e, _SLOT_FREE) or isinstance(e, _NEEDS_SLOT):
        return True
    return any(needs_slot(sub) for _field, _i, sub in sub_exprs(e))


@dataclasses.dataclass
class _Ctx:
    """Block-walk accumulator.

    ``stmts`` is the block being built: a statement visitor appends the temporary
    bindings it needs, and the block visitor appends the rewritten statement
    after them.  ``hoistable`` is false inside a sealed position, where nothing
    may be named.
    """

    stmts: list[Stmt]
    hoistable: bool = True

    def sealed(self) -> '_Ctx':
        return dataclasses.replace(self, hoistable=False)


class _ANFInstance(DefaultTransformVisitor):
    """Single-use instance of the ANF pass."""

    func: FuncDef
    gensym: Gensym
    prefix: str
    """Base name for the temporaries this instance mints."""

    def __init__(
        self, func: FuncDef, def_use: DefineUseAnalysis, prefix: str = 't',
    ):
        self.func = func
        self.gensym = Gensym(reserved=def_use.names())
        self.prefix = prefix

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    # ------------------------------------------------------------------
    # Naming

    def _visit_expr(self, e: Expr, ctx: _Ctx) -> Expr:
        """*e* rebuilt, and bound to a fresh name where that is allowed."""
        rebuilt = super()._visit_expr(e, ctx)
        if not ctx.hoistable or isinstance(e, _ATOMIC + _AGGREGATE):
            return rebuilt
        t = self.gensym.fresh(self.prefix)
        ctx.stmts.append(Assign(t, None, rebuilt, e.loc))
        return Var(t, e.loc)

    def _in_place(self, e: Expr, ctx: _Ctx) -> Expr:
        """*e* rebuilt with its children named, but not named itself.

        A statement's own expression already has a place -- the statement -- so
        binding it would only add a copy.
        """
        return super()._visit_expr(e, ctx)

    # ------------------------------------------------------------------
    # Sealed expression positions

    def _visit_if_expr(self, e: IfExpr, ctx: _Ctx):
        if self._lowers(e, ctx):
            t = self.gensym.fresh(self.prefix)
            # Two steps, not one expression: `_branch_on` appends the
            # condition's own temporaries, and those belong *before* the `if`.
            stmt = self._branch_on(e, t, ctx)
            ctx.stmts.append(stmt)
            return Var(t, e.loc)
        # The condition is evaluated whenever the ternary is, so it takes the
        # ternary's own slot; the arms are conditional and are sealed.
        cond = self._visit_expr(e.cond, ctx)
        sealed = ctx.sealed()
        return IfExpr(
            cond,
            self._in_place(e.ift, sealed),
            self._in_place(e.iff, sealed),
            e.loc,
        )

    def _lowers(self, e: Expr, ctx: _Ctx) -> bool:
        """Whether *e* is a ternary this pass turns into an ``IfStmt``.

        Every ternary but ``x1 if c else x2`` over atoms, which is already in
        normal form.  Not :func:`needs_slot`, which asks the weaker question of
        whether a *backend* needs a place -- an arm holding ``x * x`` needs no
        statement to emit.  What an unflattened arm costs is reach: no pass with
        a preamble can rewrite inside one.  See the module docstring.

        Only in a hoistable position: the statement has to go somewhere.
        """
        return (
            ctx.hoistable
            and isinstance(e, IfExpr)
            and not (
                isinstance(e.ift, _ATOMIC) and isinstance(e.iff, _ATOMIC)
            )
        )

    def _branch_on(self, e: IfExpr, target: NamedId, ctx: _Ctx) -> IfStmt:
        """*e* as an ``IfStmt`` assigning *target* in each branch.

        The merge is a phi introducing *target*, which is what a backend already
        handles for a name first assigned in both arms of an ``if``.  Appends the
        condition's temporaries to *ctx* on the way, since the condition is
        evaluated where the ternary was.
        """
        cond = self._in_place(e.cond, ctx)
        return IfStmt(
            cond,
            self._arm(target, e.ift, e.loc),
            self._arm(target, e.iff, e.loc),
            e.loc,
        )

    def _bind(self, target: NamedId, e: Expr, ctx: _Ctx, loc) -> Stmt:
        """The statement binding *target* to *e*, appending *e*'s own
        temporaries to *ctx* first.

        A lowered ternary or bool chain accumulates into *target* directly
        rather than through a temporary, so nesting them gives one ladder rather
        than a chain of copies -- and nothing runs after this pass to remove one.
        """
        if self._lowers(e, ctx):
            assert isinstance(e, IfExpr)
            return self._branch_on(e, target, ctx)
        if isinstance(e, (And, Or)) and self._lowers_chain(e, ctx):
            return self._short_circuit(e, ctx, target)
        return Assign(target, None, self._in_place(e, ctx), loc)

    def _arm(self, target: NamedId, e: Expr, loc) -> StmtBlock:
        """A block binding *target* to *e*, with *e*'s temporaries inside it.

        The block runs exactly when the arm did, which is the whole point: this
        is the slot the arm lacked.
        """
        inner = _Ctx(stmts=[])
        inner.stmts.append(self._bind(target, e, inner, loc))
        return StmtBlock(inner.stmts)

    def _visit_naryop(self, e: NaryOp, ctx: _Ctx):
        if not isinstance(e, (And, Or)):
            return super()._visit_naryop(e, ctx)
        if self._lowers_chain(e, ctx):
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

    def _lowers_chain(self, e: 'And | Or', ctx: _Ctx) -> bool:
        """Whether *e* becomes a chain of guarded statements.

        The same rule as :meth:`_lowers`: every chain but one already in normal
        form, which for a bool chain is every operand an atom.  A degenerate
        one-operand chain has nothing to short-circuit.
        """
        return (
            ctx.hoistable
            and len(e.args) > 1
            and not all(isinstance(a, _ATOMIC) for a in e.args)
        )

    def _short_circuit(
        self, e: 'And | Or', ctx: _Ctx, target: NamedId,
    ) -> Stmt:
        """*e* accumulated into *target*, one guard per operand after the first.

        The guards are *flat*, not nested, and short-circuit all the same: once
        an ``or``'s accumulator is true every later ``if not t`` fails, so no
        further operand is evaluated -- and dually for ``and``.  A nested form
        would indent one level per operand for no gain.

        .. code-block:: python

            t = a
            if not t:
                t = b       # only where `a` was false
            if not t:
                t = c

        Everything but the last statement is appended to *ctx*, and the last is
        returned, so a caller with one statement slot -- an assignment whose
        whole right-hand side this is -- has one to give back.
        """
        stmts: list[Stmt] = [self._bind(target, e.args[0], ctx, e.loc)]
        for arg in e.args[1:]:
            read = Var(target, e.loc)
            guard = read if isinstance(e, And) else Not(read, e.loc)
            stmts.append(If1Stmt(guard, self._arm(target, arg, e.loc), e.loc))
        ctx.stmts.extend(stmts[:-1])
        return stmts[-1]

    def _visit_list_comp(self, e: ListComp, ctx: _Ctx):
        # The element runs once per iteration, and a later clause's iterable may
        # read an earlier clause's target, so the whole comprehension is sealed.
        return super()._visit_list_comp(e, ctx.sealed())

    # ------------------------------------------------------------------
    # Statements.  Each names its own operands and puts them in this block.

    def _visit_block(self, block: StmtBlock, ctx: Any):
        # A fresh buffer per block: a temporary belongs to the block whose
        # statement needs it, never to an enclosing one.
        inner = _Ctx(stmts=[])
        for stmt in block.stmts:
            s, _ = self._visit_statement(stmt, inner)
            inner.stmts.append(s)
        return StmtBlock(inner.stmts), ctx

    def _visit_assign(self, stmt: Assign, ctx: _Ctx):
        if isinstance(stmt.target, NamedId) and stmt.type is None:
            # A lowered right-hand side assigns this name directly rather than a
            # temporary this statement then copies.  Skipped where the
            # assignment carries a type annotation, which has one place to sit
            # and several branches to sit in; the generic path keeps it.
            return self._bind(stmt.target, stmt.expr, ctx, stmt.loc), ctx
        expr = self._in_place(stmt.expr, ctx)
        return Assign(stmt.target, stmt.type, expr, stmt.loc), ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _Ctx):
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
        if not needs_slot(stmt.cond):
            # Nothing in it needs a place, so it stays an expression -- and
            # stays sealed, since no slot runs once per iteration.
            cond = self._in_place(stmt.cond, ctx.sealed())
            body, _ = self._visit_block(stmt.body, ctx)
            return WhileStmt(cond, body, stmt.loc), ctx
        return self._rotate(stmt, ctx), ctx

    def _rotate(self, stmt: WhileStmt, ctx: _Ctx) -> WhileStmt:
        """*stmt* with its condition evaluated through a name, once before the
        loop and once at the end of the body.

        That is FPy's own order -- condition, body, condition -- so each copy
        sits in a slot that runs exactly as often as the condition does, and the
        temporaries it needs go there.  The two copies share no nodes:
        :meth:`_in_place` rebuilds every one, so neither needs cloning.

        A body that always returns gets no second copy: the loop runs at most
        one iteration, so the condition is evaluated exactly once, and code after
        the ``return`` is unreachable -- which the syntax checker rejects.
        """
        c = self.gensym.fresh('c')
        ctx.stmts.append(
            Assign(c, None, self._in_place(stmt.cond, ctx), stmt.loc),
        )
        body, _ = self._visit_block(stmt.body, ctx)
        if Reachability.analyze(body).has_fallthrough:
            # The body's own block is the per-iteration slot.
            tail = _Ctx(stmts=body.stmts)
            again = self._in_place(stmt.cond, tail)
            body.stmts.append(Assign(c, None, again, stmt.loc))
        return WhileStmt(Var(c, stmt.loc), body, stmt.loc)

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx):
        iterable = self._in_place(stmt.iterable, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        return ForStmt(stmt.target, iterable, body, stmt.loc), ctx

    def _visit_context(self, stmt: ContextStmt, ctx: _Ctx):
        # The context expression is evaluated outside its own block, so its
        # temporaries belong to this one.  Nothing inside the block is hoisted
        # out of it: `_visit_block` gives the body its own buffer, which is what
        # keeps a hoisted operand under the rounding it was written under.
        context = self._in_place(stmt.ctx, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        return ContextStmt(stmt.target, context, body, stmt.loc), ctx

    def _visit_assert(self, stmt: AssertStmt, ctx: _Ctx):
        test = self._in_place(stmt.test, ctx)
        msg = None if stmt.msg is None else self._in_place(stmt.msg, ctx)
        return AssertStmt(test, msg, stmt.loc), ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: _Ctx):
        return EffectStmt(self._in_place(stmt.expr, ctx), stmt.loc), ctx


class ANF:
    """
    Transformation pass rewriting a function into administrative normal form.

    Every proper subexpression of a statement becomes an atom -- a name, a
    literal or a nullary constant -- bound by a fresh assignment in a statement
    slot that runs exactly when the expression did.  See the module docstring for
    the positions it seals and the aggregates it leaves alone.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Rewrites `func` into administrative normal form."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got `{func}`')
        def_use = DefineUse.analyze(func)
        out = _ANFInstance(func, def_use).apply()
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
