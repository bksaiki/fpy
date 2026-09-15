"""
Shared machinery for the transforms: the loop rewrites and the rounding
rewrites.
"""

from dataclasses import dataclass
from typing import Any

from ..analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    ContextUse,
    DefineUse,
    ListSize,
    concrete_size,
)
from ..analysis.format_infer import FormatAnalysis, FormatInfer
from ..ast.fpyast import (
    Assign,
    Attribute,
    BinaryOp,
    Cast,
    ConstInf,
    ConstNan,
    ContextStmt,
    Decnum,
    Expr,
    ForeignVal,
    ForStmt,
    FuncDef,
    Id,
    If1Stmt,
    IfExpr,
    IfStmt,
    Integer,
    ListComp,
    Location,
    NamedBinaryOp,
    NamedId,
    NamedNaryOp,
    NamedTernaryOp,
    NamedUnaryOp,
    NaryOp,
    Neg,
    NullaryOp,
    Rational,
    Round,
    Signbit,
    Stmt,
    StmtBlock,
    TernaryOp,
    TupleBinding,
    UnaryOp,
    UnderscoreId,
    Var,
    WhileStmt,
)
from ..ast.visitor import DefaultTransformVisitor
from ..number import (
    INTEGER,
    REAL,
    Context,
    Float,
    RealFloat,
    same_value,
)
from ..utils import Gensym
from .cursor import (
    BlockCursor,
    Cursor,
    Edit,
    ExprCursor,
    StmtCursor,
    contains,
    expr_sites,
    not_a_statement,
    region_of,
    stmt_sites,
)
from .error import TransformDeclined, TransformReferenceError
from .path import BlockPath, StmtPath, beneath, block_paths


def infer_array_size(func: FuncDef) -> ArraySizeAnalysis | None:
    """Run the array-size analysis as an *auxiliary* input: a failure
    only disables a static optimization, so it never breaks the
    transformation."""
    try:
        return ArraySizeInfer.analyze(func)
    except Exception:  # noqa: BLE001 -- auxiliary analysis; failure only disables an optimization
        return None


def static_size(array_size: ArraySizeAnalysis | None, iterable: Expr) -> int | None:
    """The statically-known length of *iterable* (the original AST node,
    which is what the analysis indexes), or ``None`` if the analysis
    could not pin it down."""
    if array_size is None:
        return None
    bound = array_size.by_expr.get(iterable)
    if isinstance(bound, ListSize):
        return concrete_size(bound.size)
    return None


def integer_ctx(stmts: list[Stmt], loc: Location | None) -> ContextStmt:
    """A ``with fp.INTEGER:`` block: the exact integer context under
    which a loop transform's synthesized loop-control and index
    arithmetic must be evaluated (see the rounding-context-safety
    section of the transform's module docstring)."""
    return ContextStmt(UnderscoreId(), ForeignVal(INTEGER, None), StmtBlock(stmts), loc)


def clone_block(block: StmtBlock) -> StmtBlock:
    """A structurally-fresh copy of *block*, so each emitted copy of a
    loop body occupies distinct AST nodes (a plain transform visit
    rebuilds every node)."""
    block, _ = DefaultTransformVisitor()._visit_block(block, None)
    return block


def clone(e: Expr) -> Expr:
    """A structurally fresh copy of *e*, so no AST node is shared between two
    places (a plain transform visit rebuilds every node)."""
    return DefaultTransformVisitor()._visit_expr(e, None)


def copy_target(target: Id | TupleBinding) -> Id | TupleBinding:
    """A fresh copy of a loop target with the *same* names.  ``Id``s are
    value-like and shared verbatim; a ``TupleBinding`` is rebuilt so no
    node is shared between the copies it appears in."""
    match target:
        case Id():
            return target
        case TupleBinding():
            return TupleBinding([copy_target(e) for e in target.elts], target.loc)
        case _:
            raise RuntimeError(f'Unexpected target {target}')


def attribute(alias: str, *names: str, loc: Location | None = None) -> Attribute:
    """The dotted name `alias.names[0].names[1]...`."""
    e: Expr = Var(NamedId(alias), loc)
    for name in names:
        e = Attribute(e, name, loc)
    assert isinstance(e, Attribute)
    return e


def number_literal(x: RealFloat, loc: Location | None) -> Expr:
    """`x` as an exact literal: every `RealFloat` is a dyadic rational."""
    if x.is_integer():
        return Integer(int(x), loc)
    r = x.as_rational()
    return Rational(None, r.numerator, r.denominator, loc)


def value_literal(v: Float, loc: Location | None) -> Expr:
    """`v` as a literal, whatever kind of value it is."""
    if v.is_zero() and v.s:
        # a negative zero has no rational form
        return Decnum('-0.0', loc)
    if not v.is_nar():
        return number_literal(v.as_real(), loc)
    e: Expr = ConstNan(None, loc) if v.isnan else ConstInf(None, loc)
    return Neg(e, loc) if v.s else e


def shift(x: RealFloat, k: int) -> RealFloat:
    """`x * 2**k`, exactly."""
    return RealFloat(s=x.s, exp=x.exp + k, c=x.c)


def try_round(ctx: Context, x: Float | RealFloat) -> Float | None:
    """`x` under `ctx`, or `None` where the format has no value for it.

    A fixed-point format commonly rejects NaN and the infinities outright."""
    try:
        return ctx.round(x)
    except (ValueError, OverflowError):
        return None


def agrees(a: Float | None, b: Float | None) -> bool:
    """Whether two rounding outcomes match, a refusal counting as an outcome."""
    if a is None or b is None:
        return a is None and b is None
    return same_value(a, b)


def sign_choice(pos: Float, neg: Float, operand: Expr, loc: Location | None) -> Expr:
    """
    The result for a positive or negative `operand`, chosen by its sign.

    When a format makes the same value of both, the choice collapses to that
    value and the operand is not tested at all.
    """
    if same_value(pos, neg):
        return value_literal(pos, loc)
    return IfExpr(
        Signbit(None, operand, loc),
        value_literal(neg, loc), value_literal(pos, loc), loc,
    )


def operands(e: Expr) -> list[Expr]:
    """The direct operands, left to right, of an operation.

    Shared by the rounding rewrites so they cannot disagree about the shape of
    what they lift.  Every arity is handled, not just arithmetic: the rounding
    rules hold for any real-valued function, so `sqrt` and `fma` are lifted the
    same way a multiply is.
    """
    match e:
        case NullaryOp():
            return []
        case UnaryOp():
            return [e.arg]
        case BinaryOp():
            return [e.first, e.second]
        case TernaryOp():
            return [e.first, e.second, e.third]
        case NaryOp():
            return list(e.args)
        case _:
            raise RuntimeError(f'not an operation: {e!r}')


def rebuild(e: Expr, args: list[Expr]) -> Expr:
    """*e* with its operands replaced: the inverse of :func:`operands`."""
    # a `Named*` op carries the symbol it was written with; a nullary one always
    # does, since it has nothing else to identify it.  The arities are settled by
    # the match, which mypy cannot follow through the star-args.
    named = isinstance(
        e, (NullaryOp, NamedUnaryOp, NamedBinaryOp, NamedTernaryOp, NamedNaryOp),
    )
    head = (e.func,) if named else ()   # type: ignore[attr-defined]
    ctor: Any = type(e)
    match e:
        case NullaryOp():
            return ctor(*head, e.loc)
        case NaryOp():
            return ctor(*head, args, e.loc)
        case UnaryOp() | BinaryOp() | TernaryOp():
            return ctor(*head, *args, e.loc)
        case _:
            raise RuntimeError(f'not an operation: {e!r}')


def check_where(where: int | Cursor | None) -> None:
    """Rejects a `where` that names nothing of the kind."""
    if isinstance(where, bool) or (
        where is not None and not isinstance(where, (int, Cursor))
    ):
        raise TypeError(
            f'expected an \'int\', a cursor or None for where, got `{where}`'
        )


def _target_of(
    where: int | Cursor | None, func: FuncDef
) -> tuple[BlockPath, range] | None:
    """The block path and indices an explicit cursor or region names."""
    if where is None or isinstance(where, int):
        return None
    if where.func is not func:
        raise TransformReferenceError(f'`{where}` names a statement of another program')
    match where:
        case StmtCursor() | BlockCursor():
            return region_of(where)
        case ExprCursor():
            raise not_a_statement(where)


@dataclass(frozen=True)
class Declined:
    """A verification refusal: why a candidate was not rewritten."""
    reason: str


class RoundingScopes:
    """What context each operation of a function is evaluated under.

    Shared by the rounding rewrites so they agree on the question, and so a
    `where` index counts the same operations for a listing and for the rewrite.
    """

    def __init__(self, func: FuncDef):
        self.func = func
        self.def_use = DefineUse.analyze(func)
        self.ctx_use = ContextUse.analyze(func, def_use=self.def_use)
        self._format_info: FormatAnalysis | None = None

    @property
    def format_info(self) -> FormatAnalysis:
        """Format inference, run on first use: the rounding rewrites need the
        context scopes alone and must not pay for this."""
        if self._format_info is None:
            self._format_info = FormatInfer.analyze(
                self.func, def_use=self.def_use, ctx_use=self.ctx_use,
            )
        return self._format_info

    def scope_ctx_expr(self, e: Expr) -> Expr | None:
        """The context expression of the `with` introducing *e*'s scope.

        `None` where the scope is the function's own annotation, which states
        its context outside the body: a rewrite that rebuilds a context by
        editing its constructor call has none to edit and declines.
        """
        site = self.ctx_use.find_scope_from_use(e).site
        return site.ctx if isinstance(site, ContextStmt) else None

    def scope_ctx(self, e: Expr) -> Context | None:
        """*e*'s active context, or `None` where the scope stays symbolic.

        A function-level annotation is already resolved by `ContextUse`, so a
        `None` here means genuinely unknown.
        """
        scope = self.ctx_use.find_scope_from_use(e)
        return scope.ctx if isinstance(scope.ctx, Context) else None

    def is_exact(self, e: Expr) -> bool:
        """Whether *e*'s active scope rounds exactly, so it has no rounding yet."""
        return self.scope_ctx(e) is REAL


class SiteRewriter(DefaultTransformVisitor):
    """
    The site vocabulary a rewrite with countable sites shares: where it is
    aimed, and what it replaced.

    `where` aims the rewrite: an index picks one candidate, counting in visit
    order, outermost-first; a :class:`StmtCursor` or :class:`BlockCursor` picks
    every candidate at or beneath the program point it names; an
    :class:`ExprCursor` picks exactly one, and only where the candidates are
    expressions (`_expr_sited`); `None` takes them all.

    Every rewrite is recorded in `edits`, which is what forwards a cursor
    across the pass.

    A subclass that overrides `_visit_function` must call `_begin` itself.
    """

    func: FuncDef
    """the program being walked; set by the subclass"""
    where: int | Cursor | None
    site_idx: int
    edits: list[Edit]
    dirty_exprs: list[StmtPath]
    """statements this rewrite changed the expressions of without replacing"""
    declined: list[str]
    _matched: int
    _replaced: bool
    """set by a statement visitor that replaced the statement it was handed;
    `_visit_block` turns it into an edit"""
    _dropped: bool = False
    """set by a statement visitor whose statement is wholly subsumed by what it
    emitted; `_visit_block` then leaves the statement out"""
    _site: tuple[StmtBlock, int]
    """the block and index of the statement being visited, for a visitor whose
    context carries something else"""
    _paths: dict[int, BlockPath]
    _target: tuple[BlockPath, range] | None
    _target_expr: Expr | None
    """the expression an explicit cursor names, where the sites are expressions"""
    _expr_sited: bool = False
    """whether this rewrite's candidates are expressions rather than statements;
    only such a rewrite can be aimed with an :class:`ExprCursor`"""
    listing: bool = False
    """report the sites this rewrite would act on, instead of acting on them"""
    found: list[StmtPath]
    """the statement sites, while listing"""
    found_exprs: list[Expr]
    """the expression sites, while listing"""
    refused: list[tuple[object, str]]
    """each candidate that is not a site, and why: the AST node and the reason"""

    def _begin(self, func: FuncDef) -> None:
        """Set up against the tree about to be walked: both the paths an edit
        is recorded with and the target a cursor names are nodes of it."""
        self.site_idx = 0
        self.edits = []
        self.dirty_exprs = []
        self.declined = []
        self.found = []
        self.found_exprs = []
        self.refused = []
        self._matched = 0
        self._replaced = False
        self._dropped = False
        self._paths = block_paths(func)
        self._target = None
        self._target_expr = None
        if self._expr_sited and isinstance(self.where, ExprCursor):
            if self.where.func is not func:
                raise TransformReferenceError(
                    f'`{self.where}` names an expression of another program'
                )
            self._target_expr = self.where.resolve()
        else:
            self._target = _target_of(self.where, func)

    def _visit_function(self, func: FuncDef, ctx):
        self._begin(func)
        return super()._visit_function(func, ctx)

    def _list(self) -> None:
        """Walk without rewriting, so `found` / `found_exprs` / `refused` hold
        what the pass would have done."""
        self.where = None
        self.listing = True
        self._visit_function(self.func, None)

    def list_sites(self, within: Cursor | None = None) -> list[Cursor]:
        """The sites this pass would rewrite, in visit order -- what a `where`
        index counts, and what `within` narrows.

        The pass's own walk, so a listing and an `apply` cannot disagree about
        what a site is.  Reports whichever kind of site the pass has.
        """
        self._list()
        if self._expr_sited:
            marked = {id(e) for e in self.found_exprs}
            return list(expr_sites(self.func, lambda e: id(e) in marked, within))
        if within is not None:
            # checked even when nothing was found, so an empty listing rejects a
            # `within` naming nothing of the kind as a populated one would
            if within.func is not self.func:
                raise TransformReferenceError(
                    f'`{within}` names part of another program'
                )
            if isinstance(within, ExprCursor):
                raise not_a_statement(within)
        cursors: list[Cursor] = [StmtCursor(self.func, q) for q in self.found]
        if within is None:
            return cursors
        return [c for c in cursors if contains(within, c)]

    def list_refusals(
        self, within: Cursor | None = None
    ) -> list[tuple[Cursor, str]]:
        """Why each program point this pass could have acted on is not a site,
        in visit order.

        A refusal takes no index and appears in no listing, so this is the only
        way to find one without already knowing where it is.
        """
        self._list()
        reasons = {id(node): why for node, why in self.refused}
        found: list[Cursor] = (
            list(expr_sites(self.func, lambda e: id(e) in reasons, within))
            if self._expr_sited
            else list(stmt_sites(self.func, lambda s: id(s) in reasons, within))
        )
        return [(c, reasons[id(c.resolve())]) for c in found]

    def _named_by_cursor(self, e: Expr) -> bool:
        """Whether an explicit cursor names the expression *e*, ignoring the
        index: what decides whether a refusal is reported or merely counted."""
        if self._target_expr is not None:
            return self._target_expr is e
        return self._target is not None and self._selects(*self._site, -1)

    def check_site(self, what: str) -> None:
        """Rejects an explicit `where` that named no candidate, or one whose
        candidates all declined: fail rather than silently no-op."""
        where = self.where
        if where is None:
            return
        if isinstance(where, int):
            if not 0 <= where < self.site_idx:
                refused = (
                    f'; {len(self.refused)} candidate(s) were refused: '
                    + '; '.join(why for _, why in self.refused)
                    if self.refused else ''
                )
                raise TransformReferenceError(
                    f'where={where} does not correspond to {what}; '
                    f'the function has {self.site_idx} site(s){refused}'
                )
        elif self.declined and not self.edits:
            # a refused candidate is not a site, so a cursor naming one matches
            # nothing -- but saying why beats saying it named nothing
            raise TransformDeclined(f'`{where}`: ' + '; '.join(self.declined))
        elif self._matched == 0:
            raise TransformReferenceError(f'`{where}` does not name {what}')

    def _selects(self, block: StmtBlock, pos: int, idx: int, count: int = 1) -> bool:
        """Whether the candidate at `block[pos:pos+count]`, the `idx`th of the
        program, is one this rewrite is aimed at.

        A cursor or region names a piece of program, and the candidates it
        selects are the ones *at or beneath* it -- so the statement an earlier
        rewrite left behind names the site now nested inside it.
        """
        return self._selects_at(self._paths.get(id(block)), pos, idx, count)

    def _selects_at(
        self, here: BlockPath | None, pos: int, idx: int, count: int = 1
    ) -> bool:
        """:meth:`_selects`, where the caller already has the block's path."""
        if self._target is None:
            return self.where is None or idx == self.where
        if here is None:
            # a block the rewrite synthesized: no cursor can name it
            return False

        path, span = self._target
        # a multi-statement candidate is selected only in full, so a rewrite
        # never reaches past what the caller named
        return all(
            beneath(StmtPath(here, p), path, span) for p in range(pos, pos + count)
        )

    def _selects_expr(self, e: Expr, idx: int) -> bool:
        """Whether the candidate expression *e* of the current statement is one
        this rewrite is aimed at.  An expression cursor names it exactly; a
        statement cursor or region names every candidate at or beneath it."""
        if self._target_expr is not None:
            return e is self._target_expr
        return self._selects(*self._site, idx)

    def _record(self, block: StmtBlock, pos: int, inserted: int, *, removed: int = 1) -> None:
        """Record that `inserted` statements took the place of `removed` at
        `block[pos]`; `removed=0` records an insertion, which replaces nothing.
        """
        self._record_at(self._paths[id(block)], pos, inserted, removed=removed)

    def _record_at(
        self, path: BlockPath, pos: int, inserted: int, *, removed: int = 1
    ) -> None:
        """:meth:`_record`, where the caller already has the block's path.

        A rewrite of an enclosing statement subsumes anything recorded inside
        it -- nothing under a rebuilt statement forwards anyway, and the edits
        of one pass have to stay disjoint.
        """
        if removed:
            replaced = range(pos, pos + removed)
            self.edits = [
                e for e in self.edits if not beneath(e.block_path, path, replaced)
            ]
        self.edits.append(Edit(path, pos, removed, inserted))

    def _visit_block(self, block: StmtBlock, ctx):
        """Visit a block, recording what each statement was replaced by.

        A statement visitor emits its replacement into the list handed to it as
        the context and returns the last statement, so the count is the growth
        of that list, and signals the rewrite with `_replaced`.  One whose
        emission already covers the statement signals `_dropped` instead, and
        the statement is left out.
        """
        out: list[Stmt] = []
        # a nested block must not lose an edit the enclosing statement already
        # made -- e.g. one hoisted out of the `if` condition above this block
        outer = self._replaced
        for pos, stmt in enumerate(block.stmts):
            self._site = (block, pos)
            self._replaced = False
            self._dropped = False
            before = len(out)
            s, _ = self._visit_statement(stmt, out)
            if not self._dropped:
                out.append(s)
            if self._replaced:
                self._record(block, pos, len(out) - before)
                self._replaced = False
            # cleared before returning, or a nested block whose last statement
            # was dropped would drop the compound statement around it too
            self._dropped = False
        self._replaced = outer
        return StmtBlock(out), None

    def _mark_exprs(self, block: StmtBlock, pos: int) -> None:
        """Record that `block[pos]` survives with its expressions rewritten, so
        an expression cursor in it does not forward."""
        self.dirty_exprs.append(StmtPath(self._paths[id(block)], pos))


class PreambleScoped(SiteRewriter):
    """A rewrite that emits statements before the one it is visiting.

    `_visit_block` hands each statement visitor the list to emit into and
    `DefaultTransformVisitor` threads it down to every sub-expression, but only
    some of those positions are evaluated where that list runs.  This passes
    `None` for a compound statement's own sub-expression, which is how a
    subclass knows there is no slot to use.

    For a `while` condition that is soundness: the condition is re-evaluated
    every iteration and a preamble before the loop computes it once, which does
    not terminate.  The rest is scope -- those positions are evaluated exactly
    once, so a subclass may lift the seal where it can use one (`CompToLoop`
    and the derived-iterable unfolds do).
    """

    def _visit_if1(self, stmt: If1Stmt, ctx):
        return super()._visit_if1(stmt, None)[0], ctx

    def _visit_if(self, stmt: IfStmt, ctx):
        return super()._visit_if(stmt, None)[0], ctx

    def _visit_while(self, stmt: WhileStmt, ctx):
        return super()._visit_while(stmt, None)[0], ctx

    def _visit_for(self, stmt: ForStmt, ctx):
        return super()._visit_for(stmt, None)[0], ctx

    def _visit_context(self, stmt: ContextStmt, ctx):
        return super()._visit_context(stmt, None)[0], ctx

    def _visit_list_comp(self, e: ListComp, ctx) -> ListComp:
        # the element sees the loop targets and later iterables see earlier
        # ones, so no statement-level preamble reaches inside a comprehension
        targets = [self._visit_binding(t, ctx) for t in e.targets]
        iterables = [self._visit_expr(i, None) for i in e.iterables]
        elt = self._visit_expr(e.elt, None)
        return ListComp(targets, iterables, elt, e.loc)

    def _visit_if_expr(self, e: IfExpr, ctx) -> IfExpr:
        # the condition is evaluated unconditionally; the branches are not, so
        # hoisting one of them out would evaluate it either way
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, None)
        iff = self._visit_expr(e.iff, None)
        return IfExpr(cond, ift, iff, e.loc)


class ExprSiteRewriter(PreambleScoped):
    """A rewrite whose sites are expressions and whose replacement needs a
    statement slot.

    Holds the walk the rounding rewrites share: decide the refusal before an
    index is spent, count the site, and either list it or emit.  A subclass
    says which expressions it considers (`_candidate`), whether one may be
    rewritten (`_check`), and what replaces it (`_emit`).
    """

    _expr_sited = True   # the sites are expressions, not statements
    _no_slot: str
    """the refusal for a position no statement-level preamble reaches"""

    func: FuncDef
    scopes: 'RoundingScopes'
    gensym: Gensym
    where: 'Cursor | int | None'

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _candidate(self, e: Expr) -> bool:
        """Whether *e* is an expression this rewrite considers at all."""
        raise NotImplementedError

    def _check(self, e: Expr):
        """What `_emit` needs for *e*, or a `Declined` saying why it cannot be
        rewritten."""
        raise NotImplementedError

    def _emit(self, e: Expr, info, out: list) -> Expr:
        """What replaces *e*, with whatever it needs appended to `out`."""
        raise NotImplementedError

    def _visit_expr(self, e: Expr, ctx) -> Expr:
        if not self._candidate(e):
            return super()._visit_expr(e, ctx)

        # a refusal is not a site, so it is decided before an index is spent:
        # `ctx` is `None` where no statement-level preamble reaches
        info = Declined(self._no_slot) if ctx is None else self._check(e)
        if isinstance(info, Declined):
            self.refused.append((e, info.reason))
            if self._named_by_cursor(e):
                # a cursor named it: say why, rather than that it named nothing
                self.declined.append(info.reason)
            return super()._visit_expr(e, ctx)

        idx = self.site_idx
        self.site_idx += 1
        if not self._selects_expr(e, idx):
            return super()._visit_expr(e, ctx)

        self._matched += 1
        if self.listing:
            self.found_exprs.append(e)
            return super()._visit_expr(e, ctx)

        self._replaced = True
        return self._emit(e, info, ctx)


class RoundingRewriter(ExprSiteRewriter):
    """Shared machinery for the rounding rewrites that lift one operation into
    a block of its own.

    :class:`fpy2.transform.RoundInsert` gives an exact operation a format;
    :class:`fpy2.transform.SplitRound` splits a rounded one through an
    intermediate.  A subclass says which operations it considers
    (:meth:`_candidate`), whether one may be rewritten (:meth:`_verify`), and
    what stands in its place (:meth:`_wrap`).
    """

    _no_slot = (
        'the operation has no statement-level position for the block the '
        'rewrite emits'
    )

    ctx: Context

    def __init__(
        self,
        func: FuncDef,
        ctx: Context,
        scopes: 'RoundingScopes',
        where=None,
    ):
        self.func = func
        self.ctx = ctx
        self.scopes = scopes
        self.gensym = Gensym(reserved=scopes.def_use.names())
        self.where = where

    def _check(self, e: Expr) -> 'Declined | None':
        """`None` where *e* may be rewritten, else why not."""
        raise NotImplementedError

    def _wrap(self, t: NamedId, loc: Location | None) -> Expr:
        """What replaces the operation, given the temporary holding it."""
        raise NotImplementedError

    def _emit(self, e: Expr, info, out: list) -> Expr:
        """Compute *e* alone under `self.ctx`; return :meth:`_wrap` of it.

        Each operand that survives the visit as a non-``Var`` is bound under the
        *original* scope first, so the emitted block covers this operation and
        nothing else.
        """
        loc = e.loc
        args: list[Expr] = []
        for operand in operands(e):
            new = self._visit_expr(operand, out)
            if isinstance(new, Var):
                # a name lookup rounds nothing, so the bind would be a pure copy
                args.append(new)
                continue
            t = self.gensym.fresh('_t')
            out.append(Assign(t, None, new, loc))
            args.append(Var(t, loc))

        result = self.gensym.fresh('_t')
        block = StmtBlock([Assign(result, None, rebuild(e, args), loc)])
        out.append(ContextStmt(
            UnderscoreId(), ForeignVal(self.ctx, loc), block, loc,
        ))
        return self._wrap(result, loc)


class ScopedRoundingRewriter(ExprSiteRewriter):
    """Shared machinery for the rewrites that restate one rounding as program
    text: :class:`fpy2.transform.UnfoldSpecial`,
    :class:`fpy2.transform.UnfoldNegZero`,
    :class:`fpy2.transform.UnfoldOverflow`,
    :class:`fpy2.transform.FloatToFixed` and
    :class:`fpy2.transform.RescaleFixed`.

    A site is a `Round` -- or a `Cast`, where `_casts` -- wherever the scope it
    runs under can be restated, which `ContextUse` answers and the shape of the
    program around it does not.  A subclass says whether a given context can be
    restated (`_verify`) and appends the statements that restate it
    (`_ladder`).

    Sibling of :class:`RoundingRewriter`, which lifts an operation into a block
    of its own rather than replacing it: same seal on positions with no
    statement slot, different thing emitted into that slot.
    """

    _no_slot = (
        'the rounding has no statement-level position for the statements the '
        'rewrite emits'
    )
    _casts: bool = False
    """whether a `fp.cast` is a candidate as well as a `fp.round`"""

    where: 'Cursor | int | None'
    _direct: 'tuple[NamedId, Expr] | None'
    """the name a site may assign directly, and the expression that may take
    it: the right-hand side of the assignment being visited"""

    def _begin(self, func: FuncDef) -> None:
        super()._begin(func)
        self._direct = None

    # ------------------------------------------------------------------
    # What a subclass supplies

    def _verify(self, e: Expr, ctx: Context | None):
        """What `_ladder` needs to restate *e*'s rounding, or a `Declined`
        saying why it cannot be.  `ctx` is the scope *e* runs under, or `None`
        where that scope stays symbolic."""
        raise NotImplementedError

    def _check(self, e: Expr):
        return self._verify(e, self.scopes.scope_ctx(e))

    def _ladder(self, e: Expr, target: NamedId, out: list, info) -> None:
        """Append to `out` the statements assigning `target` what *e* rounds
        to."""
        raise NotImplementedError

    # ------------------------------------------------------------------

    def _arg_name(self, e: 'Round | Cast', out: list) -> NamedId:
        """The name holding *e*'s operand, binding it where it is not one.

        A ladder names its operand several times, from inside the
        ``with fp.REAL:`` it wraps itself in.  So the bind goes before that
        block, in the current one -- the scope the operand was written in, and
        the one the rounding runs under, which makes it an identity.
        """
        arg = e.arg
        if isinstance(arg, Var):
            return arg.name
        t = self.gensym.fresh('_a')
        out.append(Assign(t, None, self._visit_expr(arg, out), e.loc))
        return t

    def _candidate(self, e: Expr) -> bool:
        """Whether *e* is a rounding this rewrite considers at all.

        Every rounding is, whatever its scope: a scope that cannot be restated
        is a refusal with a reason, which is the only way one gets reported.
        """
        return isinstance(e, (Round, Cast) if self._casts else Round)

    def _emit(self, e: Expr, info, out: list) -> Expr:
        """*e*'s ladder, appended to `out`; returns what reads its result."""
        direct = (
            self._direct[0]
            if self._direct is not None and self._direct[1] is e
            else None
        )
        target = direct if direct is not None else self.gensym.fresh('t')
        self._ladder(e, target, out, info)
        if direct is not None:
            # the ladder assigned the statement's own target; what is left of
            # the assignment would copy that name onto itself
            self._dropped = True
        return Var(target, e.loc)

    def _visit_assign(self, stmt: Assign, ctx):
        # a site that is the whole right-hand side assigns this name directly,
        # rather than a temporary the statement copies onto it.  Not where the
        # assignment is annotated: the ladder has several branches to sit in
        # and the annotation one place to sit.
        self._direct = (
            (stmt.target, stmt.expr)
            if ctx is not None and stmt.type is None
            and isinstance(stmt.target, NamedId) and self._candidate(stmt.expr)
            else None
        )
        try:
            return super()._visit_assign(stmt, ctx)
        finally:
            self._direct = None
