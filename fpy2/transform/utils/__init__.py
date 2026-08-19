"""
Shared machinery for the transforms: the loop rewrites and the rounding
rewrites.

The vocabulary a transform refers to a site with lives in the submodules:
:mod:`~fpy2.transform.utils.error` for what it raises, and
:mod:`~fpy2.transform.utils.cursor` for what it points with.  Both are
re-exported here, so a transform needs one import.
"""

from dataclasses import dataclass

from ...analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    ListSize,
    concrete_size,
)
from ...ast.fpyast import (
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
    NamedId,
    Neg,
    Rational,
    ReturnStmt,
    Round,
    Signbit,
    Stmt,
    StmtBlock,
    TupleBinding,
    UnderscoreId,
    Var,
)
from ...ast.visitor import DefaultTransformVisitor
from ...number import (
    INTEGER,
    Context,
    Float,
    MPBFixedContext,
    MPFixedContext,
    RealFloat,
)
from .cursor import Block, Cursor, Edit, EditLog, Path, block_paths
from .error import TransformDeclined, TransformReferenceError


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


def same_value(a: Float, b: Float) -> bool:
    """Whether two values are the same, sign and all."""
    if a.is_nar():
        return a.isnan == b.isnan and a.isinf == b.isinf and a.s == b.s
    return not b.is_nar() and a.as_real() == b.as_real() and a.s == b.s


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


def fixed_probes(
    ctx: MPFixedContext | MPBFixedContext,
) -> list[Float | RealFloat]:
    """
    The edge values of a fixed-point format, on which a rewrite of its
    rounding could disagree with it: both zeros, values rounding to zero from
    either side, the specials, and — for a bounded format — operands at and
    past the bound, including one full trip around the wrapped range, which
    lands back on zero.
    """
    nan = Float(isnan=True)
    xs: list[Float | RealFloat] = [
        nan, Float(x=nan, s=True),
        Float(isinf=True), Float(isinf=True, s=True),
        Float(c=0), Float(c=0, s=True),
    ]
    grid = [
        RealFloat(exp=ctx.nmin - 3, c=1),   # far below the grid
        RealFloat(exp=ctx.nmin, c=1),       # the tie at half a step
        RealFloat(exp=ctx.nmin, c=3),
        RealFloat(exp=ctx.nmin + 1, c=1),   # the grid's finest step
    ]
    if isinstance(ctx, MPBFixedContext):
        step = RealFloat(exp=ctx.nmin + 1, c=1)
        span = ctx.pos_maxval - ctx.neg_maxval + step
        grid += [
            ctx.pos_maxval, ctx.neg_maxval,
            shift(ctx.pos_maxval, 1), shift(ctx.pos_maxval, 64),
            span, span + step, shift(span, 1),
        ]
    for g in grid:
        xs.append(RealFloat(exp=g.exp, c=g.c))
        xs.append(RealFloat(s=True, exp=g.exp, c=g.c))
    return xs


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


def check_where(where: int | Cursor | Block | None) -> None:
    """Rejects a `where` that names nothing of the kind."""
    if where is not None and not isinstance(where, (int, Cursor, Block)):
        raise TypeError(
            f'expected an \'int\', \'Cursor\', \'Block\' or None for where, got `{where}`'
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


def _target_of(where: int | Cursor | Block | None, func: FuncDef) -> tuple[Path, range] | None:
    """The block path and indices an explicit cursor or region names."""
    if where is None or isinstance(where, int):
        return None
    if where.func is not func:
        raise TransformReferenceError(f'`{where}` names a statement of another program')
    if isinstance(where, Cursor):
        where.resolve()  # a cursor of this program still has to name something
        return where.block_path, range(where.index, where.index + 1)
    return where.block_path, where.span


@dataclass(frozen=True)
class Declined:
    """A verification refusal: why a candidate block was not rewritten."""
    reason: str


class SiteRewriter(DefaultTransformVisitor):
    """
    The site vocabulary a rewrite with countable sites shares: where it is
    aimed, and what it replaced.

    `where` aims the rewrite: an index picks one candidate, counting in visit
    order, outermost-first; a :class:`Cursor` or :class:`Block` picks every
    candidate at or beneath the program point it names; `None` takes them all.

    Every rewrite is recorded in `edits`, which is what forwards a cursor
    across the pass.  The record is structural -- a statement span replaced by
    another -- so no subclass says anything about what its output means.

    A subclass that overrides `_visit_function` must call `_begin` itself.
    """

    where: int | Cursor | Block | None
    site_idx: int
    edits: list[Edit]
    declined: list[str]
    _matched: int
    _replaced: bool
    """set by a statement visitor that replaced the statement it was handed;
    `_visit_block` turns it into an edit, since it is what knows where the
    statement was"""
    _site: tuple[StmtBlock, int]
    """the block and index of the statement being visited, for a visitor whose
    context carries something else"""
    _paths: dict[int, Path]
    _target: tuple[Path, range] | None
    """the block path and indices an explicit cursor or region names"""

    def _begin(self, func: FuncDef) -> None:
        """Set up against the tree about to be walked: both the paths an edit
        is recorded with and the target a cursor names are nodes of it."""
        self.edits = []
        self.declined = []
        self._matched = 0
        self._replaced = False
        self._paths = block_paths(func)
        self._target = _target_of(self.where, func)

    def _visit_function(self, func: FuncDef, ctx):
        self._begin(func)
        return super()._visit_function(func, ctx)

    def check_site(self, what: str) -> None:
        """Rejects an explicit `where` that named no candidate, or one whose
        candidates all declined: fail rather than silently no-op."""
        where = self.where
        if where is None:
            return
        if isinstance(where, int):
            if not 0 <= where < self.site_idx:
                raise TransformReferenceError(
                    f'where={where} does not correspond to {what}; '
                    f'the function has {self.site_idx} candidate site(s)'
                )
        elif self._matched == 0:
            raise TransformReferenceError(f'`{where}` does not name {what}')
        elif self.declined and not self.edits:
            raise TransformDeclined(f'`{where}`: ' + '; '.join(self.declined))

    def _selects(self, block: StmtBlock, pos: int, idx: int) -> bool:
        """Whether the candidate at `block[pos]`, the `idx`th of the program,
        is one this rewrite is aimed at.

        A cursor or region names a piece of program, and the candidates it
        selects are the ones *at or beneath* it -- which is what makes a
        forwarded site usable: the statement a rewrite leaves behind is a
        wrapper, and the next operator's site sits inside it.
        """
        if self._target is None:
            return self.where is None or idx == self.where

        path, span = self._target
        here = self._paths.get(id(block))
        if here is None:
            return False  # a block the rewrite synthesized, not one it was aimed at
        if here == path:
            return pos in span
        n = len(path)
        return len(here) > n and here[:n] == path and here[n] in span

    def _record(self, block: StmtBlock, pos: int, inserted: int) -> None:
        """Record that `block[pos]` was replaced by `inserted` statements.

        A rewrite of an enclosing statement subsumes anything recorded inside
        it -- nothing under a rebuilt statement forwards anyway, and the edits
        of one pass have to stay disjoint.
        """
        path = self._paths[id(block)]
        inner = (*path, pos)
        self.edits = [e for e in self.edits if e.block_path[:len(inner)] != inner]
        self.edits.append(Edit(path, pos, 1, inserted))


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
                    idx = self.site_idx
                    self.site_idx += 1
                    if self._selects(block, pos, idx):
                        self._matched += 1
                        verified = self._verify(s, info)
                        if isinstance(verified, Declined):
                            self.declined.append(verified.reason)
                            if isinstance(self.where, int):
                                raise TransformDeclined(
                                    f'where={idx}: {verified.reason}'
                                )
                            # a region, or the whole program, skips it
                        else:
                            emitted = self._rewrite(s, verified)
                            self._record(block, pos, len(emitted))
                            stmts.extend(emitted)
                            continue
            new_s, ctx = self._visit_statement(s, ctx)
            stmts.append(new_s)
        return StmtBlock(stmts), ctx
