"""
Hoistability: where a statement may be inserted above an expression's statement.

FPy has statements and expressions, so a pass needing a temporary for an
expression hoists it into a statement above the one it sits in.  That preserves
meaning when two things hold:

1. The expression is *strict*: evaluated exactly once, unconditionally, whenever
   its statement is (:attr:`HoistabilityAnalysis.strict`).  Being a statement
   operand does not imply strictness, so the *sealed* positions are listed by
   enumeration (:data:`SEALED_REASON`): a ternary arm, an ``and``/``or`` tail, a
   ``while`` condition, a comprehension's element and iterables, an ``assert``
   message, and a chained comparison's third operand and beyond.
2. Nothing the statement evaluates before it is left to run after it.  So a
   hoist carries its evaluation prefix: the operands to its left are bound to
   names first, in order (:func:`force_names`).

:class:`~fpy2.transform.Hoistable` lowers what it can so that every expression
outside the sealed positions it leaves is strict.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ..ast.accessors import subblocks, subexprs
from ..ast.fpyast import (
    And,
    AssertStmt,
    Compare,
    Expr,
    FuncDef,
    IfExpr,
    ListComp,
    NaryOp,
    NullaryOp,
    Or,
    Stmt,
    StmtBlock,
    ValueExpr,
    Var,
    WhileStmt,
)
from ..ast.visitor import DefaultVisitor

ATOMIC = (Var, ValueExpr, NullaryOp)
"""Expressions that are already a place, or need none."""

SEALED_REASON = {
    'ternary': 'a ternary arm is evaluated conditionally',
    'chain': 'a short-circuited operand may not be evaluated',
    'element': "a comprehension's element runs once per iteration",
    'iterable': "a comprehension's iterable may read an earlier target",
    'condition': 'a `while` condition is re-evaluated every iteration',
    'message': 'an assert message is evaluated only on failure',
    'comparison': 'a chained comparison short-circuits after the first pair',
}

def lowers(e: Expr) -> bool:
    """Whether :class:`~fpy2.transform.Hoistable` emits a statement *at* `e`.

    An arm, or an operand after the first, that is not already an atom is an
    operand with nowhere to put a statement.
    """
    match e:
        case IfExpr():
            return not (
                isinstance(e.ift, ATOMIC) and isinstance(e.iff, ATOMIC)
            )
        case And() | Or():
            return any(not isinstance(a, ATOMIC) for a in e.args[1:])
        case _:
            return False


def hoists_inside(e: Expr, hoisted: Callable[[Expr], bool]) -> bool:
    """Whether anything in `e`, `e` itself included, is `hoisted`.

    Only strict operands are searched, since nothing is hoisted out of a sealed
    one.
    """
    return hoisted(e) or any(hoists_inside(child, hoisted) for child in _strict(e))


def _strict(node: Stmt | Expr) -> list[Expr]:
    """The operands of `node` evaluated exactly once whenever it is, in
    evaluation order: always a prefix of :func:`~fpy2.ast.accessors.subexprs`."""
    children = [sub for _field, _i, sub in subexprs(node)]
    match node:
        case IfExpr() | And() | Or() | AssertStmt():
            return children[:1]    # the arms, the tail, or the message may not run
        case Compare():
            return children[:2]    # a chain short-circuits after the first pair
        case ListComp() | WhileStmt():
            return []
        case _:
            return children


def force_names(node: Stmt | Expr, hoisted: Callable[[Expr], bool]) -> set[Expr]:
    """The expressions in `node` to bind to a name, so the `hoisted` ones to
    their right do not overtake them.

    The *prefix rule*: at any node, let ``last`` be the position of the last
    child -- in :func:`~fpy2.ast.accessors.subexprs` order, which is
    evaluation order -- that something is hoisted out of (:func:`hoists_inside`).
    Every earlier child that is not already an atom is named, since a hoist
    lands above the whole statement and would otherwise run before them.  One
    the pass hoists itself needs a name only if what replaces it is not an atom.

    .. code-block:: python

        f(g(y), a if c else b)   # -> {g(y)}: the ternary hoists above it
        f(a if c else b, g(y))   # -> {}: nothing runs before the ternary
        xs[i + 1] = a if c else b  # -> {i + 1}: an index runs before the value

    A ternary or chain is exempt: its condition lands in the ``IfStmt``
    condition and each arm in a block of its own, so order is preserved
    structurally -- and naming an arm is the bug the rule exists to prevent.  A
    comprehension is not entered: its element runs once per iteration.

    The set is keyed by identity: ``Expr`` defines no ``__eq__``, so two
    structurally-equal operands stay distinct.
    """
    out: set[Expr] = set()
    _collect(node, out, hoisted)
    return out


def _collect(node: Stmt | Expr, out: set[Expr], hoisted: Callable[[Expr], bool]) -> None:
    """Accumulate :func:`force_names` for `node` and everything under it."""
    if isinstance(node, ListComp):
        return
    children = [sub for _field, _i, sub in subexprs(node)]
    if isinstance(node, AssertStmt):
        children = children[:1]        # the message is sealed; only the test is strict
    elif isinstance(node, Compare):
        children = children[:2]        # a chain short-circuits after the first pair
    if not isinstance(node, (IfExpr, And, Or)):
        hoisting = [i for i, child in enumerate(children) if hoists_inside(child, hoisted)]
        if hoisting:
            out.update(
                child for child in children[:max(hoisting)]
                if not isinstance(child, ATOMIC)
            )
    for child in children:
        _collect(child, out, hoisted)


@dataclass(frozen=True)
class HoistabilityAnalysis:
    """Result of :meth:`Hoistability.analyze`."""

    sealed: list[tuple[Expr, str]]
    """Every operand in a sealed position with its :data:`SEALED_REASON` key,
    in visit order."""

    strict: set[Expr]
    """Every expression reached from its statement through strict operands
    only."""


def _strict_exprs(block: StmtBlock, out: set[Expr]) -> None:
    for stmt in block.stmts:
        _strict_operands(stmt, out)
        for _field, sub in subblocks(stmt):
            _strict_exprs(sub, out)


def _strict_operands(node: Stmt | Expr, out: set[Expr]) -> None:
    for child in _strict(node):
        out.add(child)
        _strict_operands(child, out)


class _Sealed(DefaultVisitor):
    """Collects :attr:`HoistabilityAnalysis.sealed`."""

    found: list[tuple[Expr, str]]

    def __init__(self) -> None:
        self.found = []

    def _visit_if_expr(self, e: IfExpr, ctx: None) -> None:
        self.found.append((e.ift, 'ternary'))
        self.found.append((e.iff, 'ternary'))
        super()._visit_if_expr(e, ctx)

    def _visit_naryop(self, e: NaryOp, ctx: None) -> None:
        if isinstance(e, (And, Or)):
            self.found.extend((arg, 'chain') for arg in e.args[1:])
        super()._visit_naryop(e, ctx)

    def _visit_list_comp(self, e: ListComp, ctx: None) -> None:
        # not descended into: the comprehension is why nothing inside it can be
        # hoisted, so it is the one entry -- a ternary in the element is given a
        # slot by the loop the comprehension becomes
        self.found.append((e.elt, 'element'))
        self.found.extend((iterable, 'iterable') for iterable in e.iterables)

    def _visit_while(self, stmt: WhileStmt, ctx: None) -> None:
        self.found.append((stmt.cond, 'condition'))
        super()._visit_while(stmt, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: None) -> None:
        # the test is strict, the message is not; not descended into, as with a
        # comprehension
        self._visit_expr(stmt.test, ctx)
        if stmt.msg is not None:
            self.found.append((stmt.msg, 'message'))

    def _visit_compare(self, e: Compare, ctx: None) -> None:
        self.found.extend((arg, 'comparison') for arg in e.args[2:])
        for arg in e.args[:2]:
            self._visit_expr(arg, ctx)


class Hoistability:
    """Where a function is, and is not, in hoistable form."""

    @staticmethod
    def analyze(func: FuncDef) -> HoistabilityAnalysis:
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got `{func}`')
        v = _Sealed()
        v._visit_function(func, None)
        strict: set[Expr] = set()
        _strict_exprs(func.body, strict)
        return HoistabilityAnalysis(v.found, strict)
