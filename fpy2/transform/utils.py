"""
Shared machinery for the transforms: the loop rewrites and the rounding
rewrites.
"""

from dataclasses import dataclass

from ..analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    ContextUse,
    DefineUse,
    ListSize,
    concrete_size,
)
from ..analysis.format_infer import FormatInfer
from ..ast.fpyast import (
    Abs,
    Add,
    Assign,
    Attribute,
    Cast,
    ConstInf,
    ConstNan,
    ContextStmt,
    Decnum,
    Expr,
    ForeignVal,
    FuncDef,
    Id,
    IfExpr,
    Integer,
    Location,
    Mul,
    NamedId,
    Neg,
    Rational,
    ReturnStmt,
    Round,
    Signbit,
    Stmt,
    StmtBlock,
    Sub,
    TupleBinding,
    UnderscoreId,
    Var,
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


def is_rounding_block(stmt: Stmt, *, casts: bool) -> bool:
    """Whether *stmt* is a candidate rounding block: what :meth:`sites` lists."""
    return isinstance(stmt, ContextStmt) and rounding_block(stmt, casts=casts) is not None


def operands(e: Expr) -> list[Expr]:
    """The direct operands, left to right, of an operation that carries a
    context-driven rounding.

    Shared by the two halves of the rounding axis, :class:`.RoundElim` and
    :class:`.RoundInsert`, so they cannot disagree about which operations one
    eliminates and the other inserts.
    """
    match e:
        case Add() | Sub() | Mul():
            return [e.first, e.second]
        case Abs() | Neg() | Round() | Cast():
            return [e.arg]
        case _:
            raise RuntimeError(f'not a rounded operation: {e!r}')


def rebuild(e: Expr, args: list[Expr]) -> Expr:
    """*e* with its operands replaced: the inverse of :func:`operands`."""
    match e:
        case Add() | Sub() | Mul():
            return type(e)(args[0], args[1], e.loc)
        case Abs() | Neg():
            return type(e)(args[0], e.loc)
        case Round() | Cast():
            return type(e)(e.func, args[0], e.loc)
        case _:
            raise RuntimeError(f'not a rounded operation: {e!r}')


def check_where(where: int | Cursor | None) -> None:
    """Rejects a `where` that names nothing of the kind."""
    if isinstance(where, bool) or (
        where is not None and not isinstance(where, (int, Cursor))
    ):
        raise TypeError(
            f'expected an \'int\', a cursor or None for where, got `{where}`'
        )


def rounding_block(stmt: ContextStmt, *, casts: bool) -> list[Var] | None:
    """The rounded operands of a structurally-matching block: an
    underscore-bound context whose every statement assigns or returns a
    round (or a cast too, where `casts`) of a variable.  `None` otherwise.
    Pure syntax: this is what a `where` index counts.  An annotated assign
    is no match: the rewrites cannot carry the annotation.
    """
    # a bound context is visible to the body as a value, which a rewrite changes
    if not isinstance(stmt.target, UnderscoreId):
        return None
    args: list[Var] = []
    for s in stmt.body.stmts:
        match s:
            case Assign(target=NamedId(), type=None) | ReturnStmt():
                match s.expr:
                    case Round(arg=Var() as v):
                        args.append(v)
                    case Cast(arg=Var() as v) if casts:
                        args.append(v)
                    case _:
                        return None
            case _:
                return None
    return args


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
    """A verification refusal: why a candidate block was not rewritten."""
    reason: str


class RoundingScopes:
    """What context each operation of a function is evaluated under.

    Shared by the rounding rewrites so they agree on the question, and so a
    `where` index counts the same operations for a listing and for the rewrite.
    """

    def __init__(self, func: FuncDef):
        self.def_use = DefineUse.analyze(func)
        self.ctx_use = ContextUse.analyze(func, def_use=self.def_use)
        self.format_info = FormatInfer.analyze(
            func, def_use=self.def_use, ctx_use=self.ctx_use,
        )

    def scope_ctx(self, e: Expr) -> Context | None:
        """*e*'s active context, or `None` where the scope stays symbolic.

        A function-level annotation is already resolved by `ContextUse`, so a
        `None` here means genuinely unknown.
        """
        scope = self.ctx_use.find_scope_from_use(e)   # type: ignore[arg-type]
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
        of that list, and signals the rewrite with `_replaced`.
        """
        out: list[Stmt] = []
        # a nested block must not lose an edit the enclosing statement already
        # made -- e.g. one hoisted out of the `if` condition above this block
        outer = self._replaced
        for pos, stmt in enumerate(block.stmts):
            self._site = (block, pos)
            self._replaced = False
            before = len(out)
            s, _ = self._visit_statement(stmt, out)
            out.append(s)
            if self._replaced:
                self._record(block, pos, len(out) - before)
                self._replaced = False
        self._replaced = outer
        return StmtBlock(out), None

    def _mark_exprs(self, block: StmtBlock, pos: int) -> None:
        """Record that `block[pos]` survives with its expressions rewritten, so
        an expression cursor in it does not forward."""
        self.dirty_exprs.append(StmtPath(self._paths[id(block)], pos))


class BlockRewriter(SiteRewriter):
    """
    Rewrites selected `with` blocks, each into several statements.

    A subclass says which blocks structurally match (`_candidate`), whether a
    match may be rewritten (`_verify`), and what to put in its place
    (`_rewrite`).  A candidate `_verify` declines is skipped, except that an
    index naming one raises :class:`TransformDeclined`, as does a cursor or
    region whose candidates *all* declined.
    """

    def _candidate(self, stmt: ContextStmt):
        """What `_verify` needs for this block, or `None` where it does not
        structurally match.  Only matches count toward `where`."""
        raise NotImplementedError

    def _verify(self, stmt: ContextStmt, info):
        """What `_rewrite` needs for this match, or a `Declined` saying why
        it cannot be rewritten.  By default every match verifies."""
        return info

    def _rewrite(self, stmt: ContextStmt, info) -> list[Stmt]:
        """The statements that replace `stmt`."""
        raise NotImplementedError

    def _visit_block(self, block: StmtBlock, ctx):
        # a rewritten block expands into several statements, so the splice
        # happens here rather than in `_visit_context`
        stmts: list[Stmt] = []
        for pos, s in enumerate(block.stmts):
            if isinstance(s, ContextStmt):
                info = self._candidate(s)
                if info is not None:
                    # every candidate is verified, whether or not it is the one
                    # aimed at: a refusal is not a site, so it must not consume
                    # an index that a listing would not report
                    verified = self._verify(s, info)
                    if isinstance(verified, Declined):
                        self.refused.append((s, verified.reason))
                        if self._target is not None and self._selects(block, pos, -1):
                            # a cursor named this candidate: say why, rather
                            # than report that it named nothing
                            self.declined.append(verified.reason)
                    else:
                        idx = self.site_idx
                        self.site_idx += 1
                        if self._selects(block, pos, idx):
                            self._matched += 1
                            if self.listing:
                                self.found.append(
                                    StmtPath(self._paths[id(block)], pos)
                                )
                                stmts.append(s)
                                continue
                            emitted = self._rewrite(s, verified)
                            self._record(block, pos, len(emitted))
                            stmts.extend(emitted)
                            continue
            new_s, ctx = self._visit_statement(s, ctx)
            stmts.append(new_s)
        return StmtBlock(stmts), ctx
