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

from dataclasses import dataclass
from itertools import pairwise

from ...ast.fpyast import Expr, FuncDef, Stmt, StmtBlock
from .error import TransformReferenceError
from .path import (
    BlockPath,
    FuncBody,
    StmtPath,
    SubBlock,
    bad_path,
    beneath,
    format_path,
    resolve_block,
    resolve_stmt,
)


@dataclass(frozen=True)
class Cursor:
    """A statement of a program, named so that a rewrite can forward it.

    Validated on construction: a cursor always names a statement of its own
    program.  Use it against any other program and the transform rejects it.
    """

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

    def parent(self) -> tuple[StmtBlock, int]:
        """The block holding the statement, and its index within it."""
        return resolve_block(self.func, self.block_path), self.index

    def __str__(self):
        loc = self.resolve().loc
        where = '' if loc is None else f' at {loc.format()}'
        return f'{format_path(self.path)}{where}'


@dataclass(frozen=True)
class Block:
    """A run of consecutive statements of one block.

    What a rewritten statement forwards to: a transform replaces it with
    several statements, and the region they occupy is the honest image.  A
    region is also what a transform aims at when told to rewrite every
    candidate *within* one.
    """

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
            yield Cursor(self.func, StmtPath(self.block_path, idx))

    def __getitem__(self, i: int) -> Cursor:
        return Cursor(self.func, StmtPath(self.block_path, self.span[i]))

    def one(self) -> Cursor:
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
class Edit:
    """One run of statements replaced by another, in the *old* program's terms.

    Purely structural -- how many statements the rewrite consumed and how many
    it emitted -- so no transform has to say what its output *means*.
    """

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

    def forward(self, cursor: Cursor | Block) -> Cursor | Block:
        """*cursor*, in the program this pass produced.

        A statement the pass rewrote forwards to the region that replaced it --
        a :class:`Cursor` where that is a single statement, a :class:`Block`
        otherwise.  A statement *inside* one it rewrote does not forward at
        all: that subtree was rebuilt, and only the pass could say what became
        of it.
        """
        if isinstance(cursor, Block):
            return self._forward_region(cursor)
        if not isinstance(cursor, Cursor):
            raise TypeError(f'expected a \'Cursor\' or \'Block\', got {cursor}')
        if cursor.func is not self.source:
            raise TransformReferenceError(
                f'`{cursor}` names a statement of another program'
            )

        block, index, edit = _forward_stmt(cursor.path, self.edits, cursor.path)
        if edit is None or edit.inserted == 1:
            return Cursor(self.result, StmtPath(block, index))
        if edit.inserted == 0:
            raise TransformReferenceError(f'`{cursor}` was deleted')
        return Block(self.result, block, range(index, index + edit.inserted))

    def _forward_region(self, region: Block) -> Cursor | Block:
        """Every statement of *region*, forwarded and re-joined.

        An edit replaces a run of statements in place, so the images stay in
        one block and stay adjacent; a region that comes apart means the log
        is wrong, not that there is a case to handle."""
        if len(region) == 0:
            raise TransformReferenceError(f'`{region}` holds no statements')
        images = [self.forward(c) for c in region]
        spans = [
            img.span if isinstance(img, Block) else range(img.index, img.index + 1)
            for img in images
        ]
        paths = {img.block_path for img in images}
        adjacent = all(b.start == a.stop for a, b in pairwise(spans))
        if len(paths) != 1 or not adjacent:
            raise TransformReferenceError(f'`{region}` no longer lies in one run')

        span = range(spans[0].start, spans[-1].stop)
        block_path = paths.pop()
        if len(span) == 1:
            return Cursor(self.result, StmtPath(block_path, span.start))
        return Block(self.result, block_path, span)


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
