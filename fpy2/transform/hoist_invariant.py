"""Loop-invariant code motion: moving a loop body's invariant work above it."""

from collections.abc import Callable
from typing import NamedTuple

from ..analysis import (
    Alias,
    AliasAnalysis,
    AssignDef,
    DefineUse,
    DefineUseAnalysis,
    Definition,
    LiveVars,
    Purity,
    SyntaxCheck,
)
from ..ast import *
from ..utils import Gensym
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


class _Context(NamedTuple):
    """The analyses the query consults, and the one body fact it precomputes."""

    def_use: DefineUseAnalysis
    alias: AliasAnalysis | None
    """what may be the same list as what; computed only where it is consulted,
    since it needs type inference and not every program admits it"""
    mutating: bool
    """whether the function writes into a list in place"""

    @staticmethod
    def of(func: FuncDef, def_use: DefineUseAnalysis) -> '_Context':
        mutating = _mutating(_Nodes.of(func.body), def_use)
        alias = Alias.analyze(func, def_use=def_use) if mutating else None
        return _Context(def_use, alias, mutating)


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


def _other_def_is_read(
    name: NamedId, stmt: Stmt, def_use: DefineUseAnalysis
) -> bool:
    """Whether any definition of *name* other than *stmt*'s is ever read.

    Both directions matter, and an outside-only test catches just one.  A read
    *after* the loop sees whatever reached it when the loop runs zero times.  A
    read *earlier in the body* resolves to the loop's phi, so it sees the
    previous iteration's value -- the pre-loop one on the first pass.  Hoisting
    would answer the invariant value in either case.
    """
    return any(
        def_use.uses.get(d, set())
        for d in def_use.name_to_defs.get(name, set())
        if not (isinstance(d, AssignDef) and d.site is stmt)
    )


def _mutating(body: set[int], def_use: DefineUseAnalysis) -> bool:
    """Whether the loop body writes into a list in place."""
    return any(
        isinstance(u, IndexedAssign) and id(u) in body
        for us in def_use.uses.values() for u in us
    )


def _may_be_mutated(
    e: Expr,
    reaching: dict[NamedId, Definition],
    alias: AliasAnalysis | None,
) -> bool:
    """Whether *e* reads a list, which a body that mutates one may change.

    Reaching definitions model ``zs[i] = v`` as a fresh definition of ``zs``
    alone, so a read of an aliased ``ys`` still looks like it came from before
    the loop although the list it names changes every iteration.  Being blunt
    here -- any list-valued read, once the body mutates anything -- rather than
    comparing regions, which would be sharper and is not needed by any schedule
    yet.
    """
    if alias is None:
        return True
    return any(
        (d := reaching.get(name)) is not None and alias.region_of(d) is not None
        for name in LiveVars.analyze(e)
    )


def _bound_in(name: NamedId, body: set[int], def_use: DefineUseAnalysis) -> list[Definition]:
    """Every definition of *name* sited inside the loop body."""
    return [d for d in def_use.name_to_defs.get(name, set()) if id(d.site) in body]


def _why_not(
    stmt: Stmt,
    loop: ForStmt | WhileStmt,
    body: set[int],
    taken: set[int],
    ctx: '_Context',
) -> str | None:
    """Why *stmt* may not move above *loop*, or `None` where it may."""
    if not isinstance(stmt, Assign) or not isinstance(stmt.target, NamedId):
        return 'not a simple assignment to a name'
    target = stmt.target
    if not Purity.analyze_expr(stmt.expr, ctx.def_use):
        return f'`{target}` is bound to an impure expression'

    reaching = ctx.def_use.reach[stmt]
    varies = sorted(
        str(name) for name in LiveVars.analyze(stmt.expr)
        if not _from_before(reaching.get(name), loop, body, taken)
    )
    if varies:
        return f'{", ".join(f"`{n}`" for n in varies)} varies across iterations'
    if ctx.mutating and _may_be_mutated(stmt.expr, reaching, ctx.alias):
        return f'`{target}` reads a list the body may write into'
    if _other_def_is_read(target, stmt, ctx.def_use):
        return f'another definition of `{target}` is read'
    bound = _bound_in(target, body, ctx.def_use)
    if len(bound) != 1 or bound[0].site is not stmt:
        return f'`{target}` is bound more than once in the body'
    return None


def _own_exprs(stmt: Stmt) -> list[Expr]:
    """The expressions *stmt* evaluates itself, excluding any nested block."""
    match stmt:
        case Assign() | EffectStmt() | ReturnStmt():
            return [stmt.expr]
        case IndexedAssign():
            return [*stmt.indices, stmt.expr]
        case AssertStmt():
            return [stmt.test]
        case _:
            return []


class _Maximal(DefaultVisitor):
    """The largest subexpressions satisfying a predicate, never one inside
    another: a match is taken whole and not descended into.

    A position that is not evaluated every time its statement is reached is not
    descended into either, whether it matches or not -- moving such an
    expression above the loop evaluates it where the original may never have,
    and the guard that stopped it is often the point: in
    ``ok and len(ys) > 0 and ys[0] > x`` the subscript is reached only when the
    length test passes.  :class:`PreambleScoped` seals the same positions for
    the same reason.
    """

    def __init__(self, ok: Callable[[Expr], bool]):
        self.ok = ok
        self.found: list[Expr] = []

    def _visit_expr(self, e: Expr, ctx: None):
        if self.ok(e):
            self.found.append(e)
            return None
        return super()._visit_expr(e, ctx)

    def _visit_naryop(self, e: NaryOp, ctx: None):
        if isinstance(e, And | Or):
            return self._visit_expr(e.args[0], ctx)
        return super()._visit_naryop(e, ctx)

    def _visit_if_expr(self, e: IfExpr, ctx: None):
        return self._visit_expr(e.cond, ctx)

    def _visit_list_comp(self, e: ListComp, ctx: None):
        # the element runs once per element, and a later iterable sees the
        # targets of an earlier one
        return None

    @staticmethod
    def of(e: Expr, ok: Callable[[Expr], bool]) -> list[Expr]:
        inst = _Maximal(ok)
        inst._visit_expr(e, None)
        return inst.found


def _invariant_exprs(
    stmt: Stmt,
    loop: ForStmt | WhileStmt,
    body: set[int],
    taken: set[int],
    ctx: _Context,
) -> list[Expr]:
    """The subexpressions of *stmt* worth computing once above *loop*.

    The statement rule one level down, with two exclusions: a bare name, which
    would only be rebound, and an expression that reads nothing, which is
    :class:`ConstFold`'s to fold rather than this pass's to move.

    A statement whose *whole* right-hand side qualifies is included -- the
    binding itself may be pinned in the body while the work it does is not.
    """
    reaching = ctx.def_use.reach[stmt]

    def ok(e: Expr) -> bool:
        if isinstance(e, Var):
            return False
        names = LiveVars.analyze(e)
        if not names:
            return False
        if not all(_from_before(reaching.get(n), loop, body, taken) for n in names):
            return False
        if ctx.mutating and _may_be_mutated(e, reaching, ctx.alias):
            return False
        return Purity.analyze_expr(e, ctx.def_use)

    return [h for e in _own_exprs(stmt) for h in _Maximal.of(e, ok)]


class _Plan(NamedTuple):
    """What to emit above a loop, and what that changes in its body."""

    emit: list[Stmt]
    """the statements to put before the loop"""
    drop: set[int]
    """ids of the body statements they replace"""
    subst: dict[int, NamedId]
    """subexpressions to substitute a name for, by id"""
    dirty: set[int]
    """positions in the body whose expressions that touches"""


def _plan(loop: ForStmt | WhileStmt, ctx: _Context, gensym: Gensym) -> _Plan:
    """Plan the motion out of *loop*.

    Only direct children of the body are considered: a statement under a
    ``with`` would land outside it and be rounded differently.  That confines
    the rounding question rather than merely cheapening it, since neither loop
    form opens a context scope and nor does any expression -- a direct child,
    and every subexpression of it, is already in the scope the loop sits in,
    which is the scope it moves to.

    One walk in body order, each statement taken out counting as invariant for
    those after it, so a chain comes out in a single pass and an emission may
    read a name an earlier one bound.  A statement that moves whole is not also
    picked over for subexpressions.

    One statement always stays: a loop whose body emptied would not re-parse,
    and :class:`UnnestContext` keeps its blocks non-empty the same way.
    """
    body = _Nodes.of(loop.body)
    taken: set[int] = set()
    plan = _Plan([], set(), {}, set())

    for pos, stmt in enumerate(loop.body.stmts):
        if _why_not(stmt, loop, body, taken, ctx) is None:
            plan.emit.append(stmt)
            plan.drop.add(id(stmt))
            taken.add(id(stmt))
            continue
        for e in _invariant_exprs(stmt, loop, body, taken, ctx):
            name = gensym.fresh('t')
            plan.emit.append(Assign(name, None, e, e.loc))
            plan.subst[id(e)] = name
            plan.dirty.add(pos)

    if len(plan.drop) == len(loop.body.stmts):
        last = loop.body.stmts[-1]
        plan.emit.remove(last)
        plan.drop.discard(id(last))
    return plan


def _refusals(loop: ForStmt | WhileStmt, ctx: _Context) -> list[str]:
    """Why each statement of *loop*'s body stayed, for a loop that is no site."""
    body = _Nodes.of(loop.body)
    reasons = [_why_not(s, loop, body, set(), ctx) for s in loop.body.stmts]
    return sorted({why for why in reasons if why is not None})


class _HoistInvariant(SiteRewriter):
    """Loop-invariant code motion visitor."""

    func: FuncDef
    ctx: _Context
    gensym: Gensym
    _hoisting: set[int]
    """statements already emitted above their loop, to be left out of the body"""
    _subst: dict[int, NamedId]
    """subexpressions already emitted above their loop, and the name each took"""

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        where: int | Cursor | None,
    ):
        super().__init__()
        self.func = func
        self.ctx = _Context.of(func, def_use)
        self.where = where
        self.gensym = Gensym(def_use.names())
        self._hoisting = set()
        self._subst = {}

    def _claims(self, stmt: ForStmt | WhileStmt, hoistable: list[Stmt]) -> bool:
        """Whether to hoist here.  A loop with nothing to hoist is no site, and
        one an explicit `where` named is an error rather than a silent no-op."""
        block, pos = self._site
        if not hoistable:
            why = '; '.join(_refusals(stmt, self.ctx)) or 'the body is empty'
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

    def _hoist(self, stmt: ForStmt | WhileStmt, ctx) -> None:
        """Emit the invariant work before the loop, and mark what that leaves
        out of, or changes in, the body.

        The body keeps its own block rather than being rebuilt here: a block
        this pass synthesized is in no path, so no cursor could reach a loop
        nested inside it.
        """
        plan = _plan(stmt, self.ctx, self.gensym)
        if not self._claims(stmt, plan.emit):
            return
        self._hoisting |= plan.drop
        self._subst |= plan.subst
        for pos in sorted(plan.dirty):
            self._mark_exprs(stmt.body, pos)
        self._replaced = True
        ctx.extend(plan.emit)

    def _visit_expr(self, e: Expr, ctx):
        name = self._subst.get(id(e))
        if name is not None:
            return Var(name, e.loc)
        return super()._visit_expr(e, ctx)

    def _visit_assign(self, stmt: Assign, ctx):
        if id(stmt) in self._hoisting:
            # already emitted above the loop; `_visit_block` leaves it out and
            # records the removal
            self._dropped = True
            self._replaced = True
            return stmt, ctx
        return super()._visit_assign(stmt, ctx)

    def _visit_for(self, stmt: ForStmt, ctx):
        self._hoist(stmt, ctx)
        return super()._visit_for(stmt, ctx)

    def _visit_while(self, stmt: WhileStmt, ctx):
        self._hoist(stmt, ctx)
        return super()._visit_while(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx):
        self._hoisting = set()
        self._subst = {}
        return super()._visit_function(func, ctx)

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


class HoistInvariant:
    """Loop-invariant code motion.

    Work in a loop body whose result cannot change from one iteration to the
    next is done once, above the loop.  A whole binding moves where it can::

        for x in xs:                    c = n + 1
            c = n + 1           ->      for x in xs:
            acc = acc + c * x               acc = acc + c * x

    and otherwise its invariant subexpressions are named and moved, which is
    what reaches an operand that was never a statement::

        for x in xs:                    t = n + 1
            acc = acc + (n + 1) * x  -> for x in xs:
                                            acc = acc + t * x

    Relocation, not re-association, so it is sound under any rounding context
    -- and only direct children of the body are considered, which keeps the
    destination in the scope they were already written in.  See `_why_not` and
    `_invariant_exprs` for what qualifies.

    One pass: a chain within one body comes out together, but work freed by
    hoisting out of an *inner* loop needs the pass applied again.
    :func:`fpy2.strategies.hoist_invariant` documents the rest.
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
        """Hoist the invariant work out of *func*'s loops.

        `where` names one site: an index counting the loops this rewrite acts
        on, in visit order, or a cursor or region, which takes the sites at or
        beneath it.  `None` hoists out of every one.
        """
        return HoistInvariant.apply_with_edits(func, where, def_use=def_use).result

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
        inst.check_site('a loop with invariant work in it')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(
            func, out, tuple(inst.edits),
            exprs_rewritten=tuple(inst.dirty_exprs), exprs_preserved=True,
        )
