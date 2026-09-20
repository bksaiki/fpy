"""Pulling an invariant factor out of a reduction."""

from typing import NamedTuple, TypeAlias

from ..analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    AssignDef,
    DefineUse,
    DefineUseAnalysis,
    Definition,
    LiveVars,
    PhiDef,
    Purity,
    SyntaxCheck,
    ValueClassAnalysis,
    ValueClassInfer,
)
from ..analysis.array_size import ListSize, is_size_eq
from ..analysis.value_class import is_positive_literal
from ..ast import *
from ..number import (
    REAL,
    Context,
    EFloatContext,
    ExpContext,
    MPBFixedContext,
    MPBFloatContext,
    MPFixedContext,
    MPFloatContext,
    MPSFloatContext,
    OverflowMode,
)
from .cursor import Cursor, EditLog
from .error import TransformDeclined
from .hoist_invariant import _from_before, _Nodes, _own_exprs
from .path import walk_blocks
from .utils import RoundingScopes, SiteRewriter, check_where

_Reduction: TypeAlias = 'Sum | AMax | AMin'
"""The reductions this rewrite knows.  `sum` accumulates, so its own rounding
is part of the question; `max` and `min` *select*, so theirs is not -- under
`fp.FP16`, `max([65600.0])` is `65600.0` where rounding it would give `inf`."""


class _Reductions(DefaultVisitor):
    """Every reduction in an expression, outermost first."""

    def __init__(self):
        self.found: list[_Reduction] = []

    def _visit_unaryop(self, e: UnaryOp, ctx: None):
        if isinstance(e, Sum | AMax | AMin):
            self.found.append(e)
        super()._visit_unaryop(e, ctx)

    @staticmethod
    def of(e: Expr) -> list[_Reduction]:
        inst = _Reductions()
        inst._visit_expr(e, None)
        return inst.found


def _not_order_preserving(ctx: 'Context | None') -> 'str | None':
    """Why rounding under *ctx* may not stand in for one rounding per element,
    or `None` where it may.

    What a selection needs in place of exactness.  ``max_i round(c * xᵢ)`` is
    ``round(c * max_i xᵢ)`` only where rounding is monotone and settles the
    same way every time, since the rewrite turns one rounding per element into
    one in total.  Every rounding *mode* qualifies -- each picks one of the two
    bracketing values, and a deterministic bracket rule preserves order.

    A context this does not recognise is refused: not provably order-preserving
    is the safe answer.
    """
    if ctx is None:
        return 'the scope is not known'
    if ctx is REAL:
        return None
    if ctx.is_stochastic():
        # the one rounding would draw differently from the many it replaces
        return 'the scope rounds stochastically'

    match ctx:
        case MPFloatContext() | MPFixedContext() | MPSFloatContext():
            # unbounded: nothing overflows, so rounding alone decides, and
            # every mode of it is monotone
            return None
        case EFloatContext() | ExpContext() | MPBFixedContext() | MPBFloatContext():
            overflow = ctx.overflow
        case _:
            return 'the scope is not a kind this rewrite knows'

    match overflow:
        case OverflowMode.OVERFLOW | OverflowMode.SATURATE:
            return None
        case OverflowMode.WRAP:
            # not monotone at all, and `FixedContext`'s default
            return 'the scope wraps on overflow, which reorders the elements'
        case OverflowMode.ASSERT:
            # the rewrite changes *which* products happen, so it can drop an
            # abort: `c * min(xs)` may overflow where `c * max(xs)` does not
            return 'the scope aborts on overflow, and the rewrite changes which products overflow'
        case _:
            raise RuntimeError(f'unreachable overflow mode: {overflow}')


class _Facts(NamedTuple):
    """The analyses the conditions consult."""

    def_use: DefineUseAnalysis
    scopes: RoundingScopes
    classes: ValueClassAnalysis
    sizes: ArraySizeAnalysis

    @staticmethod
    def of(func: FuncDef) -> '_Facts':
        def_use = DefineUse.analyze(func)
        return _Facts(
            def_use,
            RoundingScopes(func),
            ValueClassInfer.analyze(func),
            ArraySizeInfer.analyze(func),
        )


class _Site(NamedTuple):
    """A reduction over scaled elements, in either of the two shapes it takes.

    ``sum([c * e for x in xs])`` is the one a program is written in, and the
    one this rewrite is really about: a comprehension defines every element, so
    there is nothing to prove about coverage.  ``sum(ts)`` over a list a loop
    filled is the same thing after `comp_to_loop`, which `rescale_fixed`
    requires -- and there the elements not written are the whole difficulty.
    """

    red: _Reduction
    prod: Mul
    """the product to replace with the operand that stays"""
    factor: Expr
    elt: Expr
    fixed: tuple[NamedId, ...]
    """names bound per element, which the factor must not read"""
    red_stmt: 'Stmt | None' = None
    loop: 'ForStmt | None' = None
    write: 'IndexedAssign | None' = None
    prod_stmt: 'Assign | None' = None


def _targets(comp: ListComp) -> tuple[NamedId, ...]:
    """Every name *comp* binds per element."""
    out: list[NamedId] = []
    for t in comp.targets:
        match t:
            case NamedId():
                out.append(t)
            case TupleBinding():
                out.extend(n for n in t.names() if isinstance(n, NamedId))
            case _:
                pass
    return tuple(out)


def _comp_site(red: _Reduction) -> '_Site | None':
    """``sum([c * e for x in xs])``."""
    comp = red.arg
    if not isinstance(comp, ListComp) or not isinstance(comp.elt, Mul):
        return None
    prod = comp.elt
    return _Site(red, prod, prod.args[0], prod.args[1], _targets(comp))


def _writes(loop: ForStmt) -> list[IndexedAssign]:
    """The list writes that are direct children of *loop*'s body.

    Direct children only: a write under an ``if`` runs on some iterations, so a
    matching trip count would no longer mean every element is written.
    """
    return [s for s in loop.body.stmts if isinstance(s, IndexedAssign)]


def _all_writes(loop: ForStmt, name: NamedId) -> int:
    """How many list writes to *name* the whole body holds, nested included.

    `_writes` looks at direct children, which is what coverage needs; this
    counts every one, because a second write -- under an ``if``, say -- can
    leave an element holding a value the factor was never applied to.
    """
    count = 0

    class V(DefaultVisitor):
        def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
            nonlocal count
            if stmt.var == name:
                count += 1
            super()._visit_indexed_assign(stmt, ctx)

    V()._visit_block(loop.body, None)
    return count


def _created_by(stmt: Stmt, name: NamedId, facts: '_Facts') -> 'Definition | None':
    """The definition of *name* that *stmt* introduces."""
    for d in facts.def_use.name_to_defs.get(name, set()):
        if isinstance(d, AssignDef) and d.site is stmt:
            return d
    return None


def _loop_site(red: _Reduction, stmt: Stmt, facts: '_Facts') -> '_Site | None':
    """``sum(ts)`` over a list a loop filled.

    The loop is found through the *definition* the reduction reads, not by
    scanning backwards for one that writes the same name: after the loop the
    name resolves to a phi sited at it, and a later rebinding resolves to that
    rebinding instead, which is how ``ts = ys; return sum(ts)`` is refused.

    The element write reads a name -- `rescale_fixed` binds the scaled value
    first -- so the product is reached through `defining_expr`.
    """
    if not isinstance(red.arg, Var):
        return None
    d = facts.def_use.find_def_from_use(red.arg)
    if not isinstance(d, PhiDef) or not isinstance(d.site, ForStmt):
        return None
    loop = d.site

    writes = _writes(loop)
    if len(writes) != 1 or writes[0].var != red.arg.name:
        return None
    write = writes[0]
    created = _created_by(write, write.var, facts)
    if created is None or facts.def_use.def_to_idx[created] not in (d.lhs, d.rhs):
        return None

    if not isinstance(write.expr, Var):
        return None
    prod = facts.def_use.defining_expr(write.expr)
    if not isinstance(prod, Mul):
        return None
    prod_stmt = facts.def_use.find_def_from_use(write.expr).site
    if not isinstance(prod_stmt, Assign) or prod_stmt not in loop.body.stmts:
        return None
    return _Site(
        red, prod, prod.args[0], prod.args[1], (),
        red_stmt=stmt, loop=loop, write=write, prod_stmt=prod_stmt,
    )


def _site(red: _Reduction, stmt: Stmt, facts: '_Facts') -> '_Site | None':
    """The rewrite's shape at *red*, in whichever form it takes."""
    return _comp_site(red) or _loop_site(red, stmt, facts)


def _indexed_by_target(site: _Site) -> bool:
    """Whether the write's index is exactly the loop's own target.

    Without this a loop of the right trip count may still write one slot over
    and over, leaving the rest to be scaled unwritten.
    """
    assert site.loop is not None and site.write is not None
    target = site.loop.target
    if not isinstance(target, NamedId) or len(site.write.indices) != 1:
        return False
    index = site.write.indices[0]
    return isinstance(index, Var) and index.name == target


def _covers(site: _Site, facts: '_Facts') -> bool:
    """Whether the loop writes every element of the list it fills.

    The trip count is ``range(len(v))`` and the list is the same size as *v*.
    Both sizes must be *known*: `is_size_eq` answers `True` for two unknowns,
    which would read as coverage rather than ignorance.
    """
    assert site.loop is not None and site.write is not None
    it = site.loop.iterable
    if not isinstance(it, Range1):
        return False
    stop = facts.def_use.defining_expr(it.arg)
    if not isinstance(stop, Len):
        return False
    d = facts.def_use.find_def_from_use(site.write)
    a, b = facts.sizes.by_expr.get(stop.arg), facts.sizes.by_def.get(d)
    if not isinstance(a, ListSize) or not isinstance(b, ListSize):
        return False
    if a.size is None or b.size is None:
        return False
    return is_size_eq(a, b)


def _varies(site: _Site, facts: '_Facts') -> list[str]:
    """The names the factor reads that do not hold still across the reduction."""
    names = LiveVars.analyze(site.factor)
    if site.loop is None:
        return sorted(str(n) for n in names if n in site.fixed)
    assert site.prod_stmt is not None
    body = _Nodes.of(site.loop.body)
    reaching = facts.def_use.reach[site.prod_stmt]
    return sorted(
        str(n) for n in names
        if not _from_before(reaching.get(n), site.loop, body, set())
    )


def _why_not(site: _Site, facts: '_Facts') -> 'str | None':
    """Why the factor may not leave *site*'s reduction, or `None` where it may."""
    if isinstance(site.red, Sum):
        # accumulation: the partial sums round, so the two orders disagree
        if not facts.scopes.is_exact(site.red):
            return 'the reduction does not round exactly'
    else:
        # selection: the reduction does not round, but the multiply does, and
        # the rewrite leaves one of it where there were as many as elements
        scope = facts.scopes.scope_ctx(site.prod)
        if scope is not facts.scopes.scope_ctx(site.red):
            return 'the product and the reduction round differently'
        why = _not_order_preserving(scope)
        if why is not None:
            return why
    if not Purity.analyze_expr(site.factor, facts.def_use):
        return 'the factor is not pure'

    varies = _varies(site, facts)
    if varies:
        return f'the factor reads {", ".join(f"`{n}`" for n in varies)}, which varies'

    # only accumulation cancels: `sum([c*1, c*-1, c*1])` is a NaN at `c = inf`
    # where `c * sum(...)` is `+inf`.  A selection returns one element, so an
    # infinite or NaN factor reaches both sides alike.
    if isinstance(site.red, Sum) and not facts.classes.is_finite(site.factor):
        return 'the factor may be an infinity or a NaN'
    base = facts.def_use.defining_expr(site.factor)
    if not isinstance(base, Pow) or not is_positive_literal(base.args[0]):
        return 'the factor is not a power with a positive base, so may be negative'

    if site.loop is None:
        # a comprehension defines every element, and both halves of the rewrite
        # sit in the one expression, so nothing below applies
        return None

    assert site.write is not None and site.prod_stmt is not None
    assert site.red_stmt is not None

    # deleting the multiply deletes its rounding, which is only harmless where
    # it rounded exactly.  A selection has had this asked of it already, in the
    # order-preserving form it needs instead.
    if isinstance(site.red, Sum) and not facts.scopes.is_exact(site.prod):
        return 'the product does not round exactly'

    # the factor is re-emitted at the reduction, so every name it reads must
    # mean the same thing there as it did in the loop
    at_prod = facts.def_use.reach[site.prod_stmt]
    at_red = facts.def_use.reach[site.red_stmt]
    moved = sorted(
        str(n) for n in LiveVars.analyze(site.factor)
        if at_prod.get(n) is not at_red.get(n)
    )
    if moved:
        return f'{", ".join(f"`{n}`" for n in moved)} is rebound before the reduction'

    # dropping the factor changes the product, so nothing but the write may
    # read it
    target = site.prod_stmt.target
    if not isinstance(target, NamedId):
        return 'the product is not bound to a name'
    prod_def = _created_by(site.prod_stmt, target, facts)
    if prod_def is None or facts.def_use.uses.get(prod_def, set()) != {site.write.expr}:
        return f'`{target}` is read by something other than the write'

    # and nothing but the write and the reduction may touch the list
    if any(
        u is not site.write and u is not site.red.arg
        for d in facts.def_use.name_to_defs.get(site.write.var, set())
        for u in facts.def_use.uses.get(d, set())
    ):
        return f'`{site.write.var}` is used somewhere other than the write and the reduction'
    if _all_writes(site.loop, site.write.var) != 1:
        return f'`{site.write.var}` is written more than once in the body'

    if not _indexed_by_target(site):
        return f'`{site.write.var}` is not written at the loop index'
    if not _covers(site, facts):
        return f'the loop may not write every element of `{site.write.var}`'
    return None


class _HoistScale(SiteRewriter):
    """Reduction scale-hoisting visitor."""

    func: FuncDef
    facts: _Facts
    _expr_sited = True
    """the sites are the reductions, so a cursor names one exactly"""
    _drop: dict[int, Expr]
    """products to replace with the operand that stays, by id"""
    _wrap: dict[int, Expr]
    """reductions to multiply by a factor, by id"""

    def __init__(self, func: FuncDef, facts: _Facts, where: int | Cursor | None):
        super().__init__()
        self.func = func
        self.facts = facts
        self.where = where
        self._drop = {}
        self._wrap = {}

    def _claims(self, red: _Reduction, why: str | None) -> bool:
        if why is not None:
            self.refused.append((red, why))
            if self._named_by_cursor(red):
                self.declined.append(why)
                if not self.listing:
                    raise TransformDeclined(f'cannot hoist the factor: {why}')
            return False

        idx = self.site_idx
        self.site_idx += 1
        if not self._selects_expr(red, idx):
            return False
        self._matched += 1
        if self.listing:
            self.found_exprs.append(red)
            return False
        return True

    def _visit_expr(self, e: Expr, ctx):
        if (elt := self._drop.get(id(e))) is not None:
            return self._visit_expr(elt, ctx)
        if (factor := self._wrap.get(id(e))) is not None:
            return Mul(factor, super()._visit_expr(e, ctx), e.loc)
        return super()._visit_expr(e, ctx)

    def _plan(self) -> None:
        """Decide every site before rewriting anything.

        The loop that fills the list comes *before* the reduction that reads
        it, so by the time the reduction is visited its product has already
        been rebuilt.  Both edits are therefore chosen in one pass over the
        original tree and applied in the next.
        """
        for _, block in walk_blocks(self.func):
            for pos, stmt in enumerate(block.stmts):
                self._site = (block, pos)
                for e in _own_exprs(stmt):
                    for red in _Reductions.of(e):
                        site = _site(red, stmt, self.facts)
                        why = (
                            'no scaled list write fills the reduction'
                            if site is None else _why_not(site, self.facts)
                        )
                        if not self._claims(red, why):
                            continue
                        assert site is not None
                        self._drop[id(site.prod)] = site.elt
                        self._wrap[id(site.red)] = site.factor
                        self._mark_exprs(block, pos)
                        if site.loop is not None and site.prod_stmt is not None:
                            body = site.loop.body
                            self._mark_exprs(
                                body, body.stmts.index(site.prod_stmt)
                            )

    def _visit_function(self, func: FuncDef, ctx):
        self._begin(func)
        self._drop = {}
        self._wrap = {}
        self._plan()
        if self.listing:
            return func
        body, _ = self._visit_block(func.body, ctx)
        return FuncDef(func.name, func.args, body, func.meta, loc=func.loc)

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


class HoistScale:
    """Pulling an invariant factor out of a reduction.

    Where a loop fills a list by scaling each element, the scaling is one
    multiply after the reduction rather than one per element::

        for i in range(len(xs)):        for i in range(len(xs)):
            t = c * e               ->      t = e
            ts[i] = t                       ts[i] = t
        return sum(ts)                  return (c * sum(ts))

    Algebra, not relocation, so unlike :class:`HoistInvariant` it needs an
    exact scope -- under a rounding one the partial sums round and the two
    disagree.  It also needs the factor finite (an infinite `c` turns a
    cancellation into a survivor), non-negative (a negative one signs a zero
    the original never signed), invariant across the loop, and the loop to
    write every element, since an unwritten one would be scaled too.

    `max` / `min` are not reductions for this purpose: a monotone selection
    needs its own proof.
    """

    @staticmethod
    def _instance(
        func: FuncDef, where: 'int | Cursor | None'
    ) -> _HoistScale:
        return _HoistScale(func, _Facts.of(func), where)

    @staticmethod
    def sites(func: FuncDef, within: 'Cursor | None' = None) -> list[Cursor]:
        """The reductions this pass would rewrite, in visit order."""
        return HoistScale._instance(func, None).list_sites(within)

    @staticmethod
    def refusals(
        func: FuncDef, within: 'Cursor | None' = None
    ) -> list[tuple[Cursor, str]]:
        """Why each reduction this pass could have rewritten is not a site."""
        return HoistScale._instance(func, None).list_refusals(within)

    @staticmethod
    def apply(func: FuncDef, where: 'int | Cursor | None' = None) -> FuncDef:
        """Hoist the invariant factor out of *func*'s reductions.

        `where` names one site: an index counting the reductions this rewrite
        acts on, in visit order, or a cursor or region, which takes the sites
        at or beneath it.  `None` rewrites every one.
        """
        return HoistScale.apply_with_edits(func, where).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef, where: 'int | Cursor | None' = None
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a 'FuncDef', got {func}")
        check_where(where)

        inst = HoistScale._instance(func, where)
        out = inst.apply()
        inst.check_site('a reduction with an invariant factor')
        SyntaxCheck.check(out, ignore_unknown=True)
        return EditLog(
            func, out, tuple(inst.edits),
            exprs_rewritten=tuple(inst.dirty_exprs), exprs_preserved=True,
        )
