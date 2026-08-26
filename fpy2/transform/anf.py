"""
Administrative normal form: bind every non-atomic subexpression to a name.

FPy expressions nest, so a backend emitting a construct that needs a *place* for
an operand has to invent one.  This pass removes the need: afterwards every
proper subexpression of a statement is an atom.

.. code-block:: python

    # before                # after
    y = (a * b) + (c * d)   t = a * b
                            t1 = c * d
                            y = t + t1

**Where a temporary goes.**  In a statement slot executed *exactly as often, and
under exactly the same condition, as the expression it names* -- not merely the
nearest enclosing statement, which is wrong for a ``while`` condition (evaluated
once per iteration where the slot before the loop runs once) and for a ternary
arm (conditional where the slot is not).  A statement's own operands qualify, and
so does anything inside a body.

Positions FPy evaluates conditionally or repeatedly do not, and are *sealed*:
their subexpressions stay inline, since hoisting one would evaluate it on a path
FPy never takes.  Two of them have a lowering that creates the slot they lack:

- a ``while`` loop is **rotated**, so the condition is evaluated once before the
  loop and once at the end of the body -- FPy's own order.  Rotation *duplicates*
  the condition, so it is gated on :func:`needs_slot`.
- an ``IfExpr`` becomes an ``IfStmt`` assigning one name.  That restructures
  rather than duplicating, so every ternary but ``x1 if c else x2`` over atoms is
  lowered.

An ``and``/``or`` tail is lowered only where an operand needs a place
(:meth:`_ANFInstance._lowers_chain`), and a comprehension's element and
iterables not at all; :meth:`ANF.refusals` reports what is left.  The lowerings
compose: a rotated condition is a slot, so a ternary inside it lowers too.

**Scalars only, by type.**  An expression is named where
:class:`~fpy2.analysis.TypeInfer` gives it a scalar type; a list, tuple, context
or unresolved type variable is left inline.  A scalar has no identity, so naming
one loses nothing, where a name holding a list is a second *place* -- and the cpp
backend's sharing analysis counts places.  So a chain is named at its outermost
scalar and the aggregate spine stays inline:

.. code-block:: python

    # `g(g(x)) + xss[i + 1][0]` becomes
    t = g(x); t1 = g(t); t2 = i + 1; t3 = xss[t2][0]; return t1 + t3

Idempotent: a second pass finds only atoms in the positions it names.

The design notes are in ``docs/todos/backend-independence.md``.
"""

import dataclasses
from typing import Any

from ..analysis import (
    DefineUse,
    DefineUseAnalysis,
    Reachability,
    SyntaxCheck,
    TypeInfer,
)
from ..analysis.type_infer import TypeAnalysis
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
    ForeignVal,
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
    Pow,
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
    UnderscoreId,
    ValueExpr,
    Var,
    WhileStmt,
    Zip,
)
from ..ast.visitor import DefaultTransformVisitor, DefaultVisitor
from ..number import REAL
from ..types import BoolType, RealType
from ..utils import Gensym
from .path import sub_exprs

_ATOMIC = (Var, ValueExpr, NullaryOp)
"""Expressions that are already a place, or need none."""

_NAMEABLE_TYPES = (RealType, BoolType)
"""Types whose values this pass binds to a name.  A whitelist, so an unresolved
``VarType`` is left inline: naming an aggregate wrongly is the costly direction."""


_SLOT_FREE = _ATOMIC + (Compare, Not, And, Or, IfExpr,
                        UnaryOp, BinaryOp, TernaryOp)
"""Node kinds whose own lowering needs no statement.  A whitelist: an unfamiliar
kind needs a slot, since a false negative leaves an operand where its statement
cannot go."""

_NEEDS_SLOT = (
    Round, RoundAt, Cast,
    Sum, AMin, AMax, AnyOf, AllOf,
    Range1, Range2, Range3, Enumerate,
    Fst, Snd, Pow,
)
"""The exceptions among :data:`_SLOT_FREE`'s base classes: a rounding may
assert, a fold needs a loop, a range allocates, a projection is read through a
bound name, and a power's exponent is bound when it lowers to ``ldexp``."""


def _reads(name: NamedId, exprs: 'list[Expr] | tuple[Expr, ...]') -> bool:
    """Whether any of *exprs* mentions *name*."""
    found = False

    class _Reads(DefaultVisitor):
        def _visit_var(self, e: Var, ctx):
            nonlocal found
            if e.name == name:
                found = True

    for e in exprs:
        _Reads()._visit_expr(e, None)
    return found


def needs_slot(e: Expr) -> bool:
    """Does emitting *e* plausibly require a statement of its own?

    Asked of a conditionally- or repeatedly-evaluated position, to decide
    whether lowering it is worth the cost.  Conservative -- see
    :data:`_SLOT_FREE`.
    """
    if not isinstance(e, _SLOT_FREE) or isinstance(e, _NEEDS_SLOT):
        return True
    return any(needs_slot(sub) for _field, _i, sub in sub_exprs(e))


# ----------------------------------------------------------------------
# The residue


_SEALED_REASON = {
    'ternary': 'a ternary arm is evaluated conditionally',
    'chain': 'a short-circuited operand may not be evaluated',
    'element': "a comprehension's element runs once per iteration",
    'iterable': "a comprehension's iterable may read an earlier target",
    'condition': 'a `while` condition is re-evaluated every iteration',
}


def _list_refusals(func: FuncDef) -> list[tuple[Expr, str]]:
    """The sealed positions of *func* holding something that needs a place.

    See :meth:`ANF.refusals`, the public entry point.
    """
    out: list[tuple[Expr, str]] = []

    def check(e: Expr, why: str) -> None:
        if needs_slot(e):
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
            check(e.elt, 'element')
            for iterable in e.iterables:
                check(iterable, 'iterable')
            super()._visit_list_comp(e, ctx)

        def _visit_while(self, stmt: WhileStmt, ctx):
            check(stmt.cond, 'condition')
            super()._visit_while(stmt, ctx)

    _Residue()._visit_function(func, None)
    return out


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
    types: TypeAnalysis
    prefix: str
    """Base name for the temporaries this instance mints."""

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        types: TypeAnalysis,
        prefix: str = 't',
    ):
        self.func = func
        self.gensym = Gensym(reserved=def_use.names())
        self.types = types
        self.prefix = prefix

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    # ------------------------------------------------------------------
    # Naming

    def _visit_expr(self, e: Expr, ctx: _Ctx) -> Expr:
        """*e* rebuilt, and bound to a fresh name where that is allowed."""
        rebuilt = super()._visit_expr(e, ctx)
        # `rebuilt`, not `e`: a lowered ternary or chain comes back as the name
        # it accumulated into, and naming that again is a pure copy.
        if (
            not ctx.hoistable
            or isinstance(rebuilt, _ATOMIC)
            or not self._nameable(e)
        ):
            return rebuilt
        t = self.gensym.fresh(self.prefix)
        ctx.stmts.append(Assign(t, None, rebuilt, e.loc))
        return Var(t, e.loc)

    def _nameable(self, e: Expr) -> bool:
        """Whether *e*'s value may be bound to a name
        (:data:`_NAMEABLE_TYPES`).  Asked of the original node, which is what
        the type analysis is keyed by."""
        return isinstance(self.types.by_expr.get(e), _NAMEABLE_TYPES)

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
            # two steps: `_branch_on` appends the condition's temporaries, which
            # belong *before* the `if`
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
        """Whether *e* is a ternary this pass turns into an ``IfStmt``: every
        one but ``x1 if c else x2`` over atoms.

        Not :func:`needs_slot` -- an ``IfStmt`` restructures rather than
        duplicating, so there is nothing to weigh against flattening the arms.
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

        Appends the condition's temporaries to *ctx*, since the condition is
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

        A lowered ternary or chain accumulates into *target* directly, so
        nesting them gives one ladder rather than a chain of copies.  Nothing
        runs after this pass to remove one.
        """
        if self._lowers(e, ctx):
            assert isinstance(e, IfExpr)
            return self._branch_on(e, target, ctx)
        if isinstance(e, (And, Or)) and self._lowers_chain(e, ctx):
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

    def _arm(self, target: NamedId, e: Expr, loc) -> StmtBlock:
        """A block binding *target* to *e*, with *e*'s temporaries inside it --
        the slot the arm lacked, running exactly when the arm did."""
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

        Only where an operand after the first needs a place -- the case the
        lowering exists for.  Not :meth:`_lowers`'s rule: lowering a *pure*
        chain would break the guard :class:`~fpy2.analysis.ValueClassInfer`
        reads to drop a runtime check, since it matches the ``And`` and a
        lowered one is statements joined by a phi.
        """
        return (
            ctx.hoistable
            and len(e.args) > 1
            and any(needs_slot(a) for a in e.args[1:])
        )

    def _short_circuit(
        self, e: 'And | Or', ctx: _Ctx, target: NamedId,
    ) -> Stmt:
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
            # temporary this statement copies.  Not where the assignment carries
            # a type annotation, which has one place to sit and several branches
            # to sit in.
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
            # nothing in it needs a place, so it stays an expression -- and
            # sealed, since no slot runs once per iteration
            cond = self._in_place(stmt.cond, ctx.sealed())
            body, _ = self._visit_block(stmt.body, ctx)
            return WhileStmt(cond, body, stmt.loc), ctx
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
        """A ``with`` statement, whose two halves round differently.

        **E-Context** evaluates the context expression under ``REAL``, not the
        active context, so its temporaries go in a ``with fp.REAL:`` block of
        their own rather than the enclosing one.  FPy scopes the context but not
        the store, so the names stay visible.

        .. code-block:: python

            with fp.REAL:               # `with fp.IEEEContext(ES + 2, NB + 2):`
                t = ES + 2
                t1 = NB + 2
            with fp.IEEEContext(t, t1):
                <body>

        The body needs no such care: :meth:`_visit_block` gives it its own
        buffer.
        """
        under_real = _Ctx(stmts=[])
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
        test = self._in_place(stmt.test, ctx)
        msg = None if stmt.msg is None else self._in_place(stmt.msg, ctx)
        return AssertStmt(test, msg, stmt.loc), ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: _Ctx):
        return EffectStmt(self._in_place(stmt.expr, ctx), stmt.loc), ctx


class ANF:
    """
    Transformation pass rewriting a function into administrative normal form.

    Every proper subexpression of a statement becomes an atom -- a name, a
    literal or a nullary constant -- bound in a statement slot that runs exactly
    when the expression did.  See the module docstring.
    """

    @staticmethod
    def refusals(func: FuncDef) -> list[tuple[Expr, str]]:
        """Every sealed position of `func` holding something that needs a place.

        Reported rather than refused: whether it matters is the backend's
        question, and the answer differs by position -- the cpp emitter gives a
        comprehension's element the loop body it generates, and a ``while``
        condition nothing.  One entry per position, since :func:`needs_slot`
        already recurses.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got `{func}`')
        return _list_refusals(func)

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Rewrites `func` into administrative normal form."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got `{func}`')
        def_use = DefineUse.analyze(func)
        types = TypeInfer.check(func, def_use=def_use)
        out = _ANFInstance(func, def_use, types).apply()
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
