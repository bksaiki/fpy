"""Pulling an invariant factor out of a reduction."""

from typing import NamedTuple

from ..analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    DefineUse,
    DefineUseAnalysis,
    LiveVars,
    Purity,
    SyntaxCheck,
    ValueClassAnalysis,
    ValueClassInfer,
)
from ..analysis.array_size import is_size_eq
from ..analysis.value_class import _positive_literal
from ..ast import *
from .cursor import Cursor, EditLog
from .error import TransformDeclined
from .hoist_invariant import _from_before, _Nodes, _own_exprs
from .path import walk_blocks
from .utils import RoundingScopes, SiteRewriter, check_where


class _Reductions(DefaultVisitor):
    """Every `sum(...)` in an expression, outermost first."""

    def __init__(self):
        self.found: list[Sum] = []

    def _visit_unaryop(self, e: UnaryOp, ctx: None):
        if isinstance(e, Sum):
            self.found.append(e)
        super()._visit_unaryop(e, ctx)

    @staticmethod
    def of(e: Expr) -> list[Sum]:
        inst = _Reductions()
        inst._visit_expr(e, None)
        return inst.found


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

    red: Sum
    prod: Mul
    """the product to replace with the operand that stays"""
    factor: Expr
    elt: Expr
    fixed: tuple[NamedId, ...]
    """names bound per element, which the factor must not read"""
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


def _comp_site(red: Sum) -> '_Site | None':
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


def _find(block: StmtBlock, pos: int, name: NamedId) -> 'ForStmt | None':
    """The nearest loop above ``block[pos]`` that fills *name*, or `None`.

    Interference in between needs no check here: a statement that reads or
    writes *name* is a use of it, and `_why_not` refuses any use outside the
    loop body other than the reduction itself -- a second filling loop
    included.
    """
    for i in range(pos - 1, -1, -1):
        stmt = block.stmts[i]
        if isinstance(stmt, ForStmt):
            writes = _writes(stmt)
            if len(writes) == 1 and writes[0].var == name:
                return stmt
    return None


def _loop_site(
    block: StmtBlock, pos: int, red: Sum, facts: '_Facts'
) -> '_Site | None':
    """``sum(ts)`` over a list a loop above it filled.

    The element write reads a *name* -- `rescale_fixed` binds the scaled value
    first -- so both the product and the factor's own definition are reached
    through `defining_expr`.
    """
    if not isinstance(red.arg, Var):
        return None
    loop = _find(block, pos, red.arg.name)
    if loop is None:
        return None
    write = _writes(loop)[0]
    prod = facts.def_use.defining_expr(write.expr)
    if not isinstance(prod, Mul) or not isinstance(write.expr, Var):
        return None
    prod_stmt = facts.def_use.find_def_from_use(write.expr).site
    if not isinstance(prod_stmt, Assign):
        return None
    return _Site(
        red, prod, prod.args[0], prod.args[1], (),
        loop=loop, write=write, prod_stmt=prod_stmt,
    )


def _site(block: StmtBlock, pos: int, red: Sum, facts: '_Facts') -> '_Site | None':
    """The rewrite's shape at *red*, in whichever form it takes."""
    return _comp_site(red) or _loop_site(block, pos, red, facts)


def _covers(site: _Site, facts: '_Facts') -> bool:
    """Whether the loop writes every element of the list it fills.

    The trip count is ``range(len(v))`` and the list is the same size as *v*,
    which the size union-find answers even where neither length is concrete.
    """
    assert site.loop is not None and site.write is not None
    it = site.loop.iterable
    if not isinstance(it, Range1):
        return False
    stop = facts.def_use.defining_expr(it.arg)
    if not isinstance(stop, Len):
        return False
    d = facts.def_use.find_def_from_use(site.write)
    return is_size_eq(facts.sizes.by_expr.get(stop.arg), facts.sizes.by_def.get(d))


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
    if not facts.scopes.is_exact(site.red):
        return 'the reduction does not round exactly'
    if not Purity.analyze_expr(site.factor, facts.def_use):
        return 'the factor is not pure, and would be evaluated once instead of once per element'

    varies = _varies(site, facts)
    if varies:
        return f'the factor reads {", ".join(f"`{n}`" for n in varies)}, which varies'

    if not facts.classes.is_finite(site.factor):
        return 'the factor may be an infinity or a NaN'
    base = facts.def_use.defining_expr(site.factor)
    if not isinstance(base, Pow) or not _positive_literal(base.args[0]):
        return 'the factor is not a power with a positive base, so may be negative'

    if site.loop is None:
        # a comprehension defines every element; there is nothing to cover
        return None

    assert site.write is not None
    body = _Nodes.of(site.loop.body)
    # the reduction's own use is the `Var` it reads, not the `Sum` around it
    if any(
        u is not site.red.arg and id(u) not in body
        for d in facts.def_use.name_to_defs.get(site.write.var, set())
        for u in facts.def_use.uses.get(d, set())
    ):
        return f'`{site.write.var}` is read somewhere other than the reduction'
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

    def _claims(self, red: Sum, why: str | None) -> bool:
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
                        site = _site(block, pos, red, self.facts)
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
