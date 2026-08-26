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
*seals* them: an ``IfExpr`` arm, an ``and``/``or`` tail and a comprehension's
element keep their subexpressions inline, because hoisting one out would evaluate
it on a path FPy never takes.

A ``while`` condition is the one such position with a lowering: the loop is
*rotated*, so the condition is evaluated once before the loop and once at the end
of the body -- exactly FPy's own order -- and each copy has a slot of its own.

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

Rotation costs a second copy of the condition, so it is applied only where the
condition holds something that needs a place at all -- :func:`needs_slot`.  A
condition built from names, literals and arithmetic is left as it is.

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

    def _visit_naryop(self, e: NaryOp, ctx: _Ctx):
        if not isinstance(e, (And, Or)):
            return super()._visit_naryop(e, ctx)
        # Short-circuit: the first operand always runs, the rest do not.
        sealed = ctx.sealed()
        args = [
            self._visit_expr(a, ctx) if i == 0
            else self._in_place(a, sealed)
            for i, a in enumerate(e.args)
        ]
        return type(e)(args, e.loc)

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
