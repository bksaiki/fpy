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
FPy never takes.

**This pass does not create the slots it needs.**  It *requires* them:
:class:`~fpy2.transform.Hoistable` restructures a ternary, an ``and``/``or`` tail
and a ``while`` condition so each has one, and this pass raises rather than
proceed without.  The precondition is exactly the positions it would have to name
and could not -- a sealed position holding something :func:`needs_slot` -- so a
program it can already normalize is accepted unchanged.  A comprehension is not
one: the cpp emitter gives the element the loop body it generates and the
iterable the ``for`` header, so declining to normalize inside one is a shape
nothing gets wrong.  :meth:`ANF.refusals` reports those.

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

The design notes are in ``docs/todos/backend-independence.md``, and the
precondition in ``docs/todos/hoistable-form.md``.
"""

import dataclasses
from typing import Any

from ..analysis import (
    DefineUse,
    DefineUseAnalysis,
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
from .error import TransformError
from .hoistable import _ATOMIC, _SEALED_REASON
from .path import sub_exprs

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



_CANNOT_SLOT = frozenset(
    _SEALED_REASON[k] for k in ('ternary', 'chain', 'condition')
)
"""The sealed positions with no lowering of their own.

A comprehension is absent on purpose: the cpp emitter gives the element the loop
body it generates and the iterable the ``for`` header, so this pass declining to
normalize inside one is a shape nothing gets wrong.  These three are shapes
something does -- each is a miscompile in ``docs/todos/backend-cpp.md``.
"""


def _check_precondition(func: FuncDef) -> None:
    """Raise unless every position this pass must name has a slot to name it in.

    The check is a filter on :func:`_list_refusals`, not a second analysis:
    a refusal in one of :data:`_CANNOT_SLOT` *is* a position the pass would have
    to emit a statement into and cannot.  So a program it could already
    normalize passes unchanged, and only one it would have had to lower is
    rejected.
    """
    bad = [(e, why) for e, why in _list_refusals(func) if why in _CANNOT_SLOT]
    if not bad:
        return
    e, why = bad[0]
    rest = '' if len(bad) == 1 else f' (and {len(bad) - 1} more)'
    raise TransformError(
        f'cannot normalize `{func.name}`: {why}, so `{e.format()}` has nowhere '
        f'to put a statement{rest}.  Run `Hoistable` first.'
    )


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
        """The condition is evaluated whenever the ternary is, so it takes the
        ternary's own slot; the arms are conditional and are sealed.

        The precondition means an arm holds nothing needing a place, so nothing
        is lost by leaving it inline -- :class:`~fpy2.transform.Hoistable` has
        already made an ``IfStmt`` of any ternary where that was not true.
        """
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
        # Short-circuit: the first operand always runs, the rest do not.  The
        # precondition means the tail holds nothing needing a place.
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
        # Sealed: no slot runs once per iteration.  The precondition means the
        # condition holds nothing needing one -- a loop where that was not true
        # has already been rotated by `Hoistable`.
        cond = self._in_place(stmt.cond, ctx.sealed())
        body, _ = self._visit_block(stmt.body, ctx)
        return WhileStmt(cond, body, stmt.loc), ctx

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
    when the expression did.

    Requires that such a slot exists wherever one is needed:
    :meth:`apply` raises where a sealed position holds something
    :func:`needs_slot`, and :class:`~fpy2.transform.Hoistable` is the pass that
    makes sure none does.  See the module docstring.
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
        _check_precondition(func)
        def_use = DefineUse.analyze(func)
        types = TypeInfer.check(func, def_use=def_use)
        out = _ANFInstance(func, def_use, types).apply()
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
