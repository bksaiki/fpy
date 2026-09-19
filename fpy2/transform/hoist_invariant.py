"""Loop-invariant code motion: which statements may move.

Phase 2 of ``docs/todos/hoist-invariant.md`` — the query only.  The rewrite
that acts on it lands in Phase 3.
"""

from ..analysis import DefineUseAnalysis, Definition, LiveVars, Purity
from ..ast import *


class _Nodes(DefaultVisitor):
    """The identity of every statement and expression under a block."""

    found: set[int]

    def __init__(self):
        self.found = set()

    def _visit_statement(self, stmt: Stmt, ctx: None):
        self.found.add(id(stmt))
        super()._visit_statement(stmt, ctx)

    def _visit_expr(self, e: Expr, ctx: None):
        self.found.add(id(e))
        super()._visit_expr(e, ctx)

    @staticmethod
    def of(block: StmtBlock) -> set[int]:
        inst = _Nodes()
        inst._visit_block(block, None)
        return inst.found


def _from_before(d: Definition | None, loop: ForStmt | WhileStmt, body: set[int]) -> bool:
    """Whether *d* is a definition the loop was entered with.

    A definition sited *at* the loop is not one: that is either the ``for``
    target or a phi merging what the body carried round, and both change from
    iteration to iteration.
    """
    return d is not None and d.site is not loop and id(d.site) not in body


def _read_outside(name: NamedId, body: set[int], def_use: DefineUseAnalysis) -> bool:
    """Whether any definition of *name* is read from outside the loop body.

    Hoisting past a loop that runs zero times makes the assignment happen where
    it previously did not, so a reader after the loop would see the hoisted
    value in place of whatever reached the loop.
    """
    return any(
        id(u) not in body
        for d in def_use.name_to_defs.get(name, set())
        for u in def_use.uses.get(d, set())
    )


def _bound_in(name: NamedId, body: set[int], def_use: DefineUseAnalysis) -> list[Definition]:
    """Every definition of *name* sited inside the loop body."""
    return [d for d in def_use.name_to_defs.get(name, set()) if id(d.site) in body]


def _invariants(
    loop: ForStmt | WhileStmt, def_use: DefineUseAnalysis
) -> list[Assign]:
    """The direct children of *loop*'s body that may be hoisted above it.

    Only direct children: a statement under a ``with`` in the body would land
    outside that ``with`` and be rounded differently.  Confining the query this
    way also settles the question outright, since neither loop form opens a
    context scope — a direct child of the body is already in the scope the loop
    statement itself sits in, which is the scope it would move to.

    One round only.  Where an invariant statement reads another, the second
    stays behind until the first has moved; the caller re-runs the query.
    """
    body = _Nodes.of(loop.body)
    out: list[Assign] = []
    for stmt in loop.body.stmts:
        if not isinstance(stmt, Assign) or not isinstance(stmt.target, NamedId):
            continue
        if not Purity.analyze_expr(stmt.expr, def_use):
            continue
        reaching = def_use.reach[stmt]
        if not all(
            _from_before(reaching.get(name), loop, body)
            for name in LiveVars.analyze(stmt.expr)
        ):
            continue
        target = stmt.target
        if _read_outside(target, body, def_use):
            continue
        bound = _bound_in(target, body, def_use)
        if len(bound) != 1 or bound[0].site is not stmt:
            continue
        out.append(stmt)
    return out
