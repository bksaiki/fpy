"""Loop-invariant code motion: moving a loop body's invariant bindings above it."""

from ..analysis import (
    DefineUse,
    DefineUseAnalysis,
    Definition,
    LiveVars,
    Purity,
    SyntaxCheck,
)
from ..ast import *
from .cursor import Cursor, EditLog, StmtPath
from .error import TransformDeclined
from .utils import SiteRewriter, check_where


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


def _from_before(
    d: Definition | None,
    loop: ForStmt | WhileStmt,
    body: set[int],
    taken: set[int],
) -> bool:
    """Whether *d* holds the same value on every iteration.

    True of a definition the loop was entered with, and of one already taken
    out of the body by this pass.  A definition sited *at* the loop is neither:
    that is the ``for`` target, or a phi merging what the body carried round.
    """
    if d is None:
        return False
    return id(d.site) in taken or (d.site is not loop and id(d.site) not in body)


def _read_outside(name: NamedId, body: set[int], def_use: DefineUseAnalysis) -> bool:
    """Whether any definition of *name* is read from outside the loop body."""
    return any(
        id(u) not in body
        for d in def_use.name_to_defs.get(name, set())
        for u in def_use.uses.get(d, set())
    )


def _bound_in(name: NamedId, body: set[int], def_use: DefineUseAnalysis) -> list[Definition]:
    """Every definition of *name* sited inside the loop body."""
    return [d for d in def_use.name_to_defs.get(name, set()) if id(d.site) in body]


def _why_not(
    stmt: Stmt,
    loop: ForStmt | WhileStmt,
    body: set[int],
    taken: set[int],
    def_use: DefineUseAnalysis,
) -> str | None:
    """Why *stmt* may not move above *loop*, or `None` where it may."""
    if not isinstance(stmt, Assign) or not isinstance(stmt.target, NamedId):
        return 'not a simple assignment to a name'
    target = stmt.target
    if not Purity.analyze_expr(stmt.expr, def_use):
        return f'`{target}` is bound to an impure expression'

    reaching = def_use.reach[stmt]
    varies = sorted(
        str(name) for name in LiveVars.analyze(stmt.expr)
        if not _from_before(reaching.get(name), loop, body, taken)
    )
    if varies:
        return f'{", ".join(f"`{n}`" for n in varies)} varies across iterations'

    # a zero-trip loop would leave whatever reached it in place, where the
    # hoisted binding puts the invariant value instead
    if _read_outside(target, body, def_use):
        return f'`{target}` is read from outside the body'
    bound = _bound_in(target, body, def_use)
    if len(bound) != 1 or bound[0].site is not stmt:
        return f'`{target}` is bound more than once in the body'
    return None


def _invariants(
    loop: ForStmt | WhileStmt, def_use: DefineUseAnalysis
) -> list[Assign]:
    """The direct children of *loop*'s body that may be hoisted above it.

    Only direct children: a statement under a ``with`` in the body would land
    outside that ``with`` and be rounded differently.  Confining the query this
    way settles the question outright rather than merely cheaply, since neither
    loop form opens a context scope -- a direct child of the body is already in
    the scope the loop statement sits in, which is the scope it moves to.

    Taken in body order, each one counting as hoisted for those after it, so a
    chain (``_k``, then ``2 ** -_k``) comes out in a single pass.  A binding
    that becomes invariant only once an *inner* loop has been hoisted out of
    needs another pass.
    """
    body = _Nodes.of(loop.body)
    taken: set[int] = set()
    out: list[Assign] = []
    for stmt in loop.body.stmts:
        if _why_not(stmt, loop, body, taken, def_use) is None:
            assert isinstance(stmt, Assign)
            out.append(stmt)
            taken.add(id(stmt))
    return out


def _refusals(loop: ForStmt | WhileStmt, def_use: DefineUseAnalysis) -> list[str]:
    """Why each statement of *loop*'s body stayed, for a loop that is no site."""
    body = _Nodes.of(loop.body)
    reasons = [_why_not(s, loop, body, set(), def_use) for s in loop.body.stmts]
    return sorted({why for why in reasons if why is not None})


class _HoistInvariant(SiteRewriter):
    """Loop-invariant code motion visitor."""

    func: FuncDef
    def_use: DefineUseAnalysis

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        where: int | Cursor | None,
    ):
        super().__init__()
        self.func = func
        self.def_use = def_use
        self.where = where

    def _claims(self, stmt: ForStmt | WhileStmt, hoistable: list[Assign]) -> bool:
        """Whether to hoist here.  A loop with nothing to hoist is no site, and
        one an explicit `where` named is an error rather than a silent no-op."""
        block, pos = self._site
        if not hoistable:
            why = '; '.join(_refusals(stmt, self.def_use)) or 'the body is empty'
            self.refused.append((stmt, why))
            # only a cursor can name a refusal: an index counts sites, and
            # `where=None` means "every site", not "this one too"
            if self._target is not None and self._selects(block, pos, -1):
                self.declined.append(why)
                if not self.listing:
                    raise TransformDeclined(f'nothing to hoist out of the loop: {why}')
            return False

        idx = self.site_idx
        self.site_idx += 1
        if not self._selects(block, pos, idx):
            return False
        self._matched += 1
        if self.listing:
            self.found.append(StmtPath(self._paths[id(block)], pos))
            return False
        return True

    def _hoist(self, stmt: ForStmt | WhileStmt, ctx) -> StmtBlock | None:
        """Emit the invariant bindings before the loop, returning the body that
        is left, or `None` where this loop is not rewritten."""
        hoistable = _invariants(stmt, self.def_use)
        if not self._claims(stmt, hoistable):
            return None
        taken = {id(s) for s in hoistable}
        self._replaced = True
        ctx.extend(hoistable)
        return StmtBlock([s for s in stmt.body.stmts if id(s) not in taken])

    def _visit_for(self, stmt: ForStmt, ctx):
        body = self._hoist(stmt, ctx)
        if body is not None:
            stmt = ForStmt(stmt.target, stmt.iterable, body, stmt.loc)
        return super()._visit_for(stmt, ctx)

    def _visit_while(self, stmt: WhileStmt, ctx):
        body = self._hoist(stmt, ctx)
        if body is not None:
            stmt = WhileStmt(stmt.cond, body, stmt.loc)
        return super()._visit_while(stmt, ctx)

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


class HoistInvariant:
    """Loop-invariant code motion.

    A binding in a loop body whose value cannot change from one iteration to
    the next is computed once, above the loop::

        for x in xs:                    c = n + 1
            c = n + 1           ->      for x in xs:
            acc = acc + c * x               acc = acc + c * x

    Relocation, not re-association, so it is sound under any rounding context
    -- and only direct children of the body move, which keeps the destination
    in the scope they were already written in.  See `_invariants` for what
    qualifies.

    One pass.  A chain within one body comes out together, but a binding freed
    by hoisting out of an *inner* loop needs the pass to run again;
    :class:`Simplify` provides that fixpoint.
    """

    @staticmethod
    def _instance(
        func: FuncDef,
        where: 'int | Cursor | None',
        def_use: DefineUseAnalysis | None = None,
    ) -> _HoistInvariant:
        if def_use is None:
            def_use = DefineUse.analyze(func)
        return _HoistInvariant(func, def_use, where)

    @staticmethod
    def sites(func: FuncDef, within: 'Cursor | None' = None) -> list[Cursor]:
        """The loops this pass would hoist out of, in visit order -- what a
        `where` index counts, and what `within` narrows."""
        return HoistInvariant._instance(func, None).list_sites(within)

    @staticmethod
    def refusals(
        func: FuncDef, within: 'Cursor | None' = None
    ) -> list[tuple[Cursor, str]]:
        """Why each loop this pass could have hoisted out of is not a site."""
        return HoistInvariant._instance(func, None).list_refusals(within)

    @staticmethod
    def apply(
        func: FuncDef,
        where: 'int | Cursor | None' = None,
        *,
        def_use: DefineUseAnalysis | None = None,
    ) -> FuncDef:
        """Hoist the invariant bindings out of *func*'s loops.

        `where` names one site: an index counting the loops this rewrite acts
        on, in visit order, or a cursor or region, which takes the sites at or
        beneath it.  `None` hoists out of every one.
        """
        return HoistInvariant.apply_with_edits(func, where, def_use=def_use).result

    @staticmethod
    def apply_with_status(
        func: FuncDef,
        where: 'int | Cursor | None' = None,
        *,
        def_use: DefineUseAnalysis | None = None,
    ) -> tuple[FuncDef, bool]:
        """:meth:`apply` with a ``changed`` flag -- `True` iff a binding moved."""
        log = HoistInvariant.apply_with_edits(func, where, def_use=def_use)
        return log.result, bool(log.edits)

    @staticmethod
    def apply_with_edits(
        func: FuncDef,
        where: 'int | Cursor | None' = None,
        *,
        def_use: DefineUseAnalysis | None = None,
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a 'FuncDef', got {func}")
        check_where(where)

        inst = HoistInvariant._instance(func, where, def_use)
        out = inst.apply()
        inst.check_site('a loop with an invariant binding')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(func, out, tuple(inst.edits), exprs_preserved=True)
