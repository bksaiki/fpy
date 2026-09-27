"""
Hoistability: where a statement may be inserted above an expression's statement.

FPy has statements and expressions, so a pass needing a temporary for an
expression hoists it into a statement above.  That is sound where the program
is in *hoistable form*:

    Every expression node is evaluated exactly once, unconditionally, whenever
    its enclosing statement is reached.

**The sealed positions** break it, and being a statement operand does not imply
strictness, so they are listed by enumeration (:data:`SEALED_REASON`): a ternary
arm, an ``and``/``or`` tail, a ``while`` condition, a comprehension's element
and iterables, an ``assert`` message, and a chained comparison's third operand
and beyond.

**The ordering hazard.**  A hoist lands above the whole statement, so it
overtakes the operands to its left unless they are named first: see
:func:`force_names`.  :attr:`HoistabilityAnalysis.slots` says where a hoist is
sound as the program stands.

:class:`~fpy2.transform.Hoistable` rewrites a program into this form.
"""

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


def hoists_inside(e: Expr) -> bool:
    """Whether a statement lands above the enclosing statement from anywhere in
    `e`, `e` itself included: a lowering of :class:`~fpy2.transform.Hoistable`,
    or a comprehension, which :class:`~fpy2.transform.CompToLoop` lowers into
    the same slot.

    Only strict operands are searched, since nothing is hoisted out of a sealed
    one.
    """
    if lowers(e) or isinstance(e, ListComp):
        return True
    return any(hoists_inside(kid) for kid in _strict(e))


def _strict(node: 'Stmt | Expr') -> list[Expr]:
    """The operands of `node` evaluated exactly once whenever it is, in
    evaluation order: always a prefix of :func:`~fpy2.ast.accessors.subexprs`."""
    kids = [sub for _field, _i, sub in subexprs(node)]
    match node:
        case IfExpr() | And() | Or() | AssertStmt():
            return kids[:1]    # the arms, the tail, or the message may not run
        case Compare():
            return kids[:2]    # a chain short-circuits after the first pair
        case ListComp() | WhileStmt():
            return []
        case _:
            return kids


def force_names(node: 'Stmt | Expr') -> set[Expr]:
    """The expressions in `node` to bind to a name, so a hoist to their right
    does not overtake them.

    The *prefix rule*: at any node, let ``last`` be the position of the last
    child -- in :func:`~fpy2.ast.accessors.subexprs` order, which is
    evaluation order -- that something hoists out of (:func:`hoists_inside`).
    Every earlier child that is not already an atom is named, since a hoist
    lands above the whole statement and would otherwise run before them.

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
    _collect(node, out)
    return out


def _collect(node: 'Stmt | Expr', out: set[Expr]) -> None:
    """Accumulate :func:`force_names` for `node` and everything under it."""
    if isinstance(node, ListComp):
        return
    kids = [sub for _field, _i, sub in subexprs(node)]
    if isinstance(node, AssertStmt):
        kids = kids[:1]        # the message is sealed; only the test is strict
    elif isinstance(node, Compare):
        kids = kids[:2]        # a chain short-circuits after the first pair
    if not isinstance(node, (IfExpr, And, Or)):
        hoisting = [i for i, kid in enumerate(kids) if hoists_inside(kid)]
        if hoisting:
            out.update(
                kid for kid in kids[:max(hoisting)]
                if not isinstance(kid, ATOMIC)
            )
    for kid in kids:
        _collect(kid, out)


@dataclass(frozen=True)
class HoistabilityAnalysis:
    """Result of :meth:`Hoistability.analyze`."""

    sealed: list[tuple[Expr, str]]
    """Every operand in a sealed position with its :data:`SEALED_REASON` key,
    in visit order."""

    slots: set[Expr]
    """Every expression a statement may be inserted before its own statement
    for: it is reached through strict operands only, and every operand that runs
    before it is an atom."""


def _slots(block: StmtBlock, out: set[Expr]) -> None:
    for stmt in block.stmts:
        _slot_operands(stmt, out)
        for _field, sub in subblocks(stmt):
            _slots(sub, out)


def _slot_operands(node: 'Stmt | Expr', out: set[Expr]) -> None:
    strict = _strict(node)
    for i, kid in enumerate(strict):
        if not all(isinstance(k, ATOMIC) for k in strict[:i]):
            return             # a hoist here would overtake an earlier operand
        out.add(kid)
        _slot_operands(kid, out)


class _Sealed(DefaultVisitor):
    """Collects :attr:`HoistabilityAnalysis.sealed`."""

    found: list[tuple[Expr, str]]

    def __init__(self):
        self.found = []

    def _visit_if_expr(self, e: IfExpr, ctx):
        self.found.append((e.ift, 'ternary'))
        self.found.append((e.iff, 'ternary'))
        super()._visit_if_expr(e, ctx)

    def _visit_naryop(self, e: NaryOp, ctx):
        if isinstance(e, (And, Or)):
            self.found.extend((arg, 'chain') for arg in e.args[1:])
        super()._visit_naryop(e, ctx)

    def _visit_list_comp(self, e: ListComp, ctx):
        # not descended into: the comprehension is why nothing inside it can be
        # hoisted, so it is the one entry -- a ternary in the element is given a
        # slot by the loop the comprehension becomes
        self.found.append((e.elt, 'element'))
        self.found.extend((iterable, 'iterable') for iterable in e.iterables)

    def _visit_while(self, stmt: WhileStmt, ctx):
        self.found.append((stmt.cond, 'condition'))
        super()._visit_while(stmt, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx):
        # the test is strict, the message is not; not descended into, as with a
        # comprehension
        self._visit_expr(stmt.test, ctx)
        if stmt.msg is not None:
            self.found.append((stmt.msg, 'message'))

    def _visit_compare(self, e: Compare, ctx):
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
        slots: set[Expr] = set()
        _slots(func.body, slots)
        return HoistabilityAnalysis(v.found, slots)
