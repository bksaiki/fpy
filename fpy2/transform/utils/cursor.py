"""
A reference to a statement that survives the rewrites around it.

Every transform is a :class:`DefaultTransformVisitor`, which rebuilds every node
it visits, so node identity dies at the first rewrite and cannot be the
reference.  A cursor is a *path* instead -- see
:mod:`~fpy2.transform.utils.path` for the shape.

A cursor is owned by one program version -- it holds the :class:`FuncDef` it
resolved against -- so aiming it at another program is a bad reference rather
than a silent mis-hit.  Holding that reference also keeps the tree alive, so no
`id` is recycled underneath a live cursor.
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import TypeAlias

from ...ast.fpyast import Expr, FuncDef, Stmt, StmtBlock
from .error import TransformReferenceError
from .path import (
    BlockPath,
    ExprPath,
    FuncBody,
    StmtPath,
    SubBlock,
    bad_path,
    beneath,
    format_path,
    rebase_expr,
    resolve_block,
    resolve_expr,
    resolve_stmt,
    walk_exprs,
    walk_stmts,
)


@dataclass(frozen=True)
class StmtCursor:
    """A statement of a program; validated on construction."""

    func: FuncDef
    """the program this cursor names a statement of"""
    path: StmtPath
    """where the statement sits"""

    def __post_init__(self):
        if not isinstance(self.func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {self.func}')
        if not isinstance(self.path, StmtPath):
            raise TypeError(f'expected a \'StmtPath\', got {self.path}')
        self.resolve()

    @property
    def block_path(self) -> BlockPath:
        """The path of the block holding the statement."""
        return self.path.parent

    @property
    def index(self) -> int:
        """The statement's index within its block."""
        return self.path.index

    def resolve(self) -> Stmt:
        """The statement this cursor names."""
        return resolve_stmt(self.func, self.path)

    def __str__(self):
        loc = self.resolve().loc
        where = '' if loc is None else f' at {loc.format()}'
        return f'{format_path(self.path)}{where}'


@dataclass(frozen=True)
class BlockCursor:
    """A run of consecutive statements of one block: what a statement replaced
    by several forwards to, and what a rewrite aimed at a region names."""

    func: FuncDef
    """the program this region belongs to"""
    block_path: BlockPath
    """the block holding the run"""
    span: range
    """the indices the run covers"""

    def __post_init__(self):
        if not isinstance(self.func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {self.func}')
        if not isinstance(self.block_path, (FuncBody, SubBlock)):
            raise TypeError(f'expected a \'BlockPath\', got {self.block_path}')
        if not isinstance(self.span, range) or self.span.step != 1:
            raise TypeError(f'expected a contiguous \'range\', got {self.span}')
        block = resolve_block(self.func, self.block_path)
        if not (0 <= self.span.start and self.span.stop <= len(block.stmts)):
            raise bad_path(
                self.block_path,
                f'has no statements {self.span.start}:{self.span.stop}; '
                f'the block holds {len(block.stmts)}'
            )

    def __len__(self):
        return len(self.span)

    def __iter__(self):
        for idx in self.span:
            yield StmtCursor(self.func, StmtPath(self.block_path, idx))

    def __getitem__(self, i: int) -> StmtCursor:
        return StmtCursor(self.func, StmtPath(self.block_path, self.span[i]))

    def one(self) -> StmtCursor:
        """The single statement of this region."""
        if len(self.span) != 1:
            raise bad_path(self.block_path, f'names {len(self.span)} statements, not one')
        return self[0]

    def resolve(self) -> list[Stmt]:
        """The statements this region covers."""
        block = resolve_block(self.func, self.block_path)
        return block.stmts[self.span.start:self.span.stop]

    def __str__(self):
        return f'{format_path(self.block_path)}[{self.span.start}:{self.span.stop}]'


@dataclass(frozen=True)
class ExprCursor:
    """An expression of a program: what a rewrite whose sites *are* expressions
    is aimed at, `inline` being the one."""

    func: FuncDef
    """the program this cursor names an expression of"""
    path: ExprPath
    """where the expression sits"""

    def __post_init__(self):
        if not isinstance(self.func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {self.func}')
        if not isinstance(self.path, ExprPath):
            raise TypeError(f'expected an \'ExprPath\', got {self.path}')
        self.resolve()

    def resolve(self) -> Expr:
        """The expression this cursor names."""
        return resolve_expr(self.func, self.path)

    def stmt(self) -> StmtCursor:
        """The statement the expression belongs to."""
        return StmtCursor(self.func, self.path.stmt())

    def __str__(self):
        loc = self.resolve().loc
        where = '' if loc is None else f' at {loc.format()}'
        return f'{format_path(self.path)}{where}'


Cursor: TypeAlias = ExprCursor | StmtCursor | BlockCursor
"""What a rewrite can be aimed at, and what forwarding hands back."""


@dataclass(frozen=True)
class Edit:
    """One run of statements replaced by another, in the *old* program's terms."""

    block_path: BlockPath
    """the block, in the old program, the rewrite happened in"""
    index: int
    """the old index of the first statement consumed"""
    removed: int
    """how many statements the rewrite consumed"""
    inserted: int
    """how many statements it emitted in their place"""

    def __post_init__(self):
        if self.index < 0 or self.removed < 0 or self.inserted < 0:
            raise ValueError(f'ill-formed edit: {self}')

    @property
    def span(self) -> range:
        """The indices consumed, in the old program."""
        return range(self.index, self.index + self.removed)


@dataclass(frozen=True)
class EditLog:
    """What one pass did to a program, and the forwarding it supports.

    An empty log is the identity: a pass that leaves the statement tree alone
    still rebases its cursors, since the program is a new object.
    """

    source: FuncDef
    """the program the pass was given"""
    result: FuncDef
    """the program it produced"""
    edits: tuple[Edit, ...] = ()
    """the rewrites, disjoint, in the source program's terms"""
    exprs_rewritten: tuple[StmtPath, ...] = ()
    """statements whose expressions the pass rewrote without replacing them"""
    exprs_preserved: bool = False
    """whether the pass left every expression *outside* those rewrites alone.

    An expression cursor forwards only under this claim: a pass may rewrite
    expressions in statements it never replaces -- folding a constant, pinning a
    context -- and no edit would record it.  False is the safe default, and makes
    forwarding an :class:`ExprCursor` across the pass fail rather than mis-aim.
    """

    def __post_init__(self):
        for e in self.edits:
            block = resolve_block(self.source, e.block_path)
            if e.index + e.removed > len(block.stmts):
                raise ValueError(
                    f'edit consumes statements {e.span.start}:{e.span.stop} of a '
                    f'block of {len(block.stmts)}: {e}'
                )
        for a in self.edits:
            for b in self.edits:
                if a is not b and _overlaps(a, b):
                    raise ValueError(f'edits are not disjoint: {a} and {b}')

    def forward(self, cursor: Cursor) -> Cursor:
        """*cursor*, in the program this pass produced.

        A statement the pass rewrote forwards to the region that replaced it.  A
        statement *inside* one it rewrote does not forward: that subtree was
        rebuilt, and only the pass could say what became of it.
        """
        if isinstance(cursor, BlockCursor):
            return self._forward_region(cursor)
        if isinstance(cursor, ExprCursor):
            return self._forward_expr(cursor)
        if not isinstance(cursor, StmtCursor):
            raise TypeError(f'expected a \'StmtCursor\' or \'BlockCursor\', got {cursor}')
        if cursor.func is not self.source:
            raise TransformReferenceError(
                f'`{cursor}` names a statement of another program'
            )

        block, index, edit = _forward_stmt(cursor.path, self.edits, cursor.path)
        if edit is None or edit.inserted == 1:
            return StmtCursor(self.result, StmtPath(block, index))
        if edit.inserted == 0:
            raise TransformReferenceError(f'`{cursor}` was deleted')
        return BlockCursor(self.result, block, range(index, index + edit.inserted))

    def _forward_expr(self, cursor: ExprCursor) -> ExprCursor:
        """*cursor*, under its statement's image."""
        if cursor.func is not self.source:
            raise TransformReferenceError(
                f'`{cursor}` names an expression of another program'
            )
        if not self.exprs_preserved:
            raise TransformReferenceError(
                f'`{cursor}` does not forward: the pass does not say what it did '
                'to expressions outside the statements it replaced'
            )

        stmt = cursor.path.stmt()
        if stmt in self.exprs_rewritten:
            raise TransformReferenceError(
                f'`{cursor}` is in `{format_path(stmt)}`, whose expressions the '
                'pass rewrote'
            )
        block, index, edit = _forward_stmt(stmt, self.edits, cursor.path)
        if edit is not None:
            raise TransformReferenceError(
                f'`{cursor}` is inside `{format_path(stmt)}`, which was rewritten'
            )
        return ExprCursor(self.result, rebase_expr(cursor.path, StmtPath(block, index)))

    def _forward_region(self, region: BlockCursor) -> Cursor:
        """Every statement of *region*, forwarded and re-joined.

        An edit replaces a run of statements in place, so the images stay in
        one block and stay adjacent; a region that comes apart means the log
        is wrong, not that there is a case to handle."""
        if len(region) == 0:
            raise TransformReferenceError(f'`{region}` holds no statements')
        spans: list[range] = []
        paths: set[BlockPath] = set()
        for img in (self.forward(c) for c in region):
            assert isinstance(img, (StmtCursor, BlockCursor))  # never an expression
            spans.append(
                img.span if isinstance(img, BlockCursor)
                else range(img.index, img.index + 1)
            )
            paths.add(img.block_path)
        # members one edit consumed together share its image, so `==` counts
        adjacent = all(b.start in (a.stop, a.start) for a, b in pairwise(spans))
        if len(paths) != 1 or not adjacent:
            raise TransformReferenceError(f'`{region}` no longer lies in one run')

        span = range(spans[0].start, max(s.stop for s in spans))
        block_path = paths.pop()
        if len(span) == 1:
            return StmtCursor(self.result, StmtPath(block_path, span.start))
        return BlockCursor(self.result, block_path, span)


def region_of(where: StmtCursor | BlockCursor) -> tuple[BlockPath, range]:
    """The block and indices *where* names: one statement, or a run of them."""
    if isinstance(where, StmtCursor):
        return where.path.parent, range(where.index, where.index + 1)
    return where.block_path, where.span


def not_a_statement(cursor: ExprCursor) -> TransformReferenceError:
    """An expression cursor handed to a statement-sited rewrite or listing."""
    return TransformReferenceError(
        f'`{cursor}` names an expression, and these sites are statements: '
        'no statement sits beneath an expression'
    )


def _restrict(func: FuncDef, within: Cursor | None, *, stmts: bool):
    """What `within` narrows a listing to, as a predicate over paths."""
    if within is None:
        return lambda path: True
    if within.func is not func:
        raise TransformReferenceError(f'`{within}` names part of another program')
    if isinstance(within, ExprCursor):
        if stmts:
            raise not_a_statement(within)
        under = within.path
        return lambda path: path == under or _under_expr(path, under)
    block, span = region_of(within)
    return lambda path: beneath(path, block, span)


def _under_expr(path: ExprPath, ancestor: ExprPath) -> bool:
    """Whether *path* lies under *ancestor*, both expressions."""
    p = path.parent
    while isinstance(p, ExprPath):
        if p == ancestor:
            return True
        p = p.parent
    return False


def stmt_sites(
    func: FuncDef,
    match: Callable[[Stmt], bool],
    within: Cursor | None = None,
) -> list[StmtCursor]:
    """The statements of *func* that *match*, in visit order.

    What a `where` index counts, and what `within` narrows: a cursor or region
    keeps the sites at or beneath it.
    """
    keep = _restrict(func, within, stmts=True)
    return [
        StmtCursor(func, path)
        for path, stmt in walk_stmts(func)
        if match(stmt) and keep(path)
    ]


def expr_sites(
    func: FuncDef,
    match: Callable[[Expr], bool],
    within: Cursor | None = None,
) -> list[ExprCursor]:
    """The expressions of *func* that *match*, in visit order.

    Outermost first within each statement, and each statement's own expressions
    before the blocks it holds -- the order a visitor reaches them in.
    """
    keep = _restrict(func, within, stmts=False)
    return [
        ExprCursor(func, path)
        for path, e in walk_exprs(func)
        if match(e) and keep(path)
    ]


def _overlaps(a: Edit, b: Edit) -> bool:
    """Whether *b* sits within what *a* consumed -- in the same block, or in a
    block below one of the statements *a* replaced."""
    if a.block_path == b.block_path:
        return b.index in a.span or a.index in b.span
    return beneath(b.block_path, a.block_path, a.span)


def _forward_block(block: BlockPath, edits: tuple[Edit, ...], leaf) -> BlockPath:
    """*block*, after the edits: its enclosing statements shift with them."""
    match block:
        case FuncBody():
            return block
        case SubBlock(parent, field):
            new_block, new_index, edit = _forward_stmt(parent, edits, leaf)
            if edit is not None:
                raise TransformReferenceError(
                    f'`{format_path(leaf)}` is inside `{format_path(parent)}`, '
                    'which was rewritten'
                )
            return SubBlock(StmtPath(new_block, new_index), field)


def _forward_stmt(
    path: StmtPath, edits: tuple[Edit, ...], leaf
) -> tuple[BlockPath, int, Edit | None]:
    """The block and index *path*'s statement lands at, and the edit that
    replaced it, where one did.

    An edit shifts every later statement of its own block by
    ``inserted - removed``; the shifts of a block accumulate, and the enclosing
    blocks shift first, since a path descends through them.  *leaf* is the path
    the caller asked about, for the message where an ancestor was rewritten.
    """
    block = _forward_block(path.parent, edits, leaf)
    shift = 0
    containing: Edit | None = None
    for e in edits:
        if e.block_path != path.parent:
            continue
        if path.index >= e.index + e.removed:
            shift += e.inserted - e.removed
        elif path.index >= e.index:
            containing = e
    if containing is not None:
        return block, containing.index + shift, containing
    return block, path.index + shift, None
