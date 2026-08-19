"""
A reference to a statement that survives the rewrites around it.

Every transform is a :class:`DefaultTransformVisitor`, which rebuilds every node
it visits, so node identity dies at the first rewrite and cannot be the
reference.  A cursor is the statement's *path* from the :class:`FuncDef`: an
alternating tuple of block field and statement index, ``('body', 1, 'ift', 0)``
for the first statement of the true arm of the second statement.  A path of odd
length names a block, one of even length a statement.

A cursor is owned by one program version -- it holds the :class:`FuncDef` it
resolved against -- so aiming it at another program is a bad reference rather
than a silent mis-hit.  Holding that reference also keeps the tree alive, so no
`id` is recycled underneath a live cursor.
"""

from dataclasses import dataclass
from typing import TypeAlias

from ...ast.fpyast import (
    ContextStmt,
    ForStmt,
    FuncDef,
    If1Stmt,
    IfStmt,
    Stmt,
    StmtBlock,
    WhileStmt,
)
from .error import TransformReferenceError

Path: TypeAlias = tuple[str | int, ...]
"""A path from a :class:`FuncDef`: block field, statement index, block field, ...

Odd length names a block, even length a statement.
"""


def format_path(path: Path) -> str:
    """A path as `body[1].ift[0]`."""
    out: list[str] = []
    for step in path:
        if isinstance(step, int):
            out.append(f'[{step}]')
        else:
            out.append(f'.{step}' if out else step)
    return ''.join(out)


def sub_blocks(stmt: Stmt) -> tuple[tuple[str, StmtBlock], ...]:
    """The blocks *stmt* encloses, each with the field that names it."""
    match stmt:
        case IfStmt():
            return ('ift', stmt.ift), ('iff', stmt.iff)
        case If1Stmt() | WhileStmt() | ForStmt() | ContextStmt():
            return ('body', stmt.body),
        case _:
            return ()


def block_paths(func: FuncDef) -> dict[int, Path]:
    """The path of every block in *func*, keyed by `id`.

    The inverse of :func:`resolve_block`, for the transforms: a visitor knows
    the block object it is rewriting in, and needs its path to record an edit.
    Valid only while *func* is alive.
    """
    paths: dict[int, Path] = {}

    def walk(block: StmtBlock, path: Path) -> None:
        paths[id(block)] = path
        for i, stmt in enumerate(block.stmts):
            for field, sub in sub_blocks(stmt):
                walk(sub, (*path, i, field))

    walk(func.body, ('body',))
    return paths


def _bad(path: Path, why: str) -> TransformReferenceError:
    return TransformReferenceError(f'`{format_path(path)}` {why}')


def resolve_block(func: FuncDef, path: Path) -> StmtBlock:
    """The block *path* names in *func*."""
    if len(path) % 2 != 1:
        raise _bad(path, 'is not a block path')
    if path[0] != 'body':
        raise _bad(path, f'does not start at the function body: `{path[0]}`')

    block = func.body
    for i in range(1, len(path), 2):
        idx, field = path[i], path[i + 1]
        if not isinstance(idx, int):
            raise _bad(path, f'has a field where a statement index belongs: `{idx}`')
        if not 0 <= idx < len(block.stmts):
            raise _bad(path, f'names statement {idx} of a block of {len(block.stmts)}')
        stmt = block.stmts[idx]
        for name, sub in sub_blocks(stmt):
            if name == field:
                block = sub
                break
        else:
            raise _bad(path, f'names no `{field}` block of a `{type(stmt).__name__}`')
    return block


def resolve_stmt(func: FuncDef, path: Path) -> Stmt:
    """The statement *path* names in *func*."""
    if path == () or len(path) % 2 != 0:
        raise _bad(path, 'is not a statement path')
    block = resolve_block(func, path[:-1])
    idx = path[-1]
    assert isinstance(idx, int)  # parity: an even-length path ends in an index
    if not 0 <= idx < len(block.stmts):
        raise _bad(path, f'names statement {idx} of a block of {len(block.stmts)}')
    return block.stmts[idx]


@dataclass(frozen=True)
class Cursor:
    """A statement of a program, named so that a rewrite can forward it.

    Validated on construction: a cursor always names a statement of its own
    program.  Use it against any other program and the transform rejects it.
    """

    func: FuncDef
    """the program this cursor names a statement of"""
    path: Path
    """the statement's path (see the module docstring)"""

    def __post_init__(self):
        if not isinstance(self.func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {self.func}')
        if not isinstance(self.path, tuple):
            raise TypeError(f'expected a \'tuple\' path, got {self.path}')
        self.resolve()

    @property
    def block_path(self) -> Path:
        """The path of the block holding the statement."""
        return self.path[:-1]

    @property
    def index(self) -> int:
        """The statement's index within its block."""
        idx = self.path[-1]
        assert isinstance(idx, int)
        return idx

    def resolve(self) -> Stmt:
        """The statement this cursor names."""
        return resolve_stmt(self.func, self.path)

    def parent(self) -> tuple[StmtBlock, int]:
        """The block holding the statement, and its index within it.

        What a transform matches against: the visitor walks these very nodes,
        so identity is a valid test *within* one traversal.
        """
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
    block_path: Path
    """the path of the block holding the run"""
    span: range
    """the indices the run covers"""

    def __post_init__(self):
        if not isinstance(self.func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {self.func}')
        if not isinstance(self.span, range) or self.span.step != 1:
            raise TypeError(f'expected a contiguous \'range\', got {self.span}')
        block = resolve_block(self.func, self.block_path)
        if not (0 <= self.span.start and self.span.stop <= len(block.stmts)):
            raise _bad(
                self.block_path,
                f'has no statements {self.span.start}:{self.span.stop}; '
                f'the block holds {len(block.stmts)}'
            )

    def __len__(self):
        return len(self.span)

    def __iter__(self):
        for idx in self.span:
            yield Cursor(self.func, (*self.block_path, idx))

    def __getitem__(self, i: int) -> Cursor:
        return Cursor(self.func, (*self.block_path, self.span[i]))

    def one(self) -> Cursor:
        """The single statement of this region."""
        if len(self.span) != 1:
            raise _bad(self.block_path, f'names {len(self.span)} statements, not one')
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

    block_path: Path
    """the path, in the old program, of the block the rewrite happened in"""
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

    def forward(self, cursor: Cursor) -> Cursor | Block:
        """*cursor*, in the program this pass produced.

        A statement the pass rewrote forwards to the region that replaced it --
        a :class:`Cursor` where that is a single statement, a :class:`Block`
        otherwise.  A statement *inside* one it rewrote does not forward at
        all: that subtree was rebuilt, and only the pass could say what became
        of it.
        """
        if not isinstance(cursor, Cursor):
            raise TypeError(f'expected a \'Cursor\', got {cursor}')
        if cursor.func is not self.source:
            raise TransformReferenceError(
                f'`{cursor}` names a statement of another program'
            )

        path, edit, start = _forward(cursor.path, self.edits)
        if edit is None:
            return Cursor(self.result, path)
        if edit.inserted == 0:
            raise TransformReferenceError(f'`{cursor}` was deleted')
        if edit.inserted == 1:
            return Cursor(self.result, (*path, start))
        return Block(self.result, path, range(start, start + edit.inserted))


def _overlaps(a: Edit, b: Edit) -> bool:
    """Whether *b* sits within what *a* consumed -- in the same block, or in a
    block below one of the statements *a* replaced."""
    if a.block_path == b.block_path:
        return b.index in a.span or a.index in b.span
    n = len(a.block_path)
    return (
        len(b.block_path) > n
        and b.block_path[:n] == a.block_path
        and b.block_path[n] in a.span
    )


def _forward(path: Path, edits: tuple[Edit, ...]) -> tuple[Path, Edit | None, int]:
    """*path* under *edits*, level by level.

    Returns the forwarded path, and -- where the path's last step lands in what
    an edit replaced -- that edit and the new index its replacement starts at.
    An edit shifts every later statement of its own block by
    ``inserted - removed``; the shifts of a block accumulate, and a path
    descending through a rewritten statement does not forward at all.
    """
    out: list[str | int] = []
    prefix: list[str | int] = []
    for i in range(0, len(path), 2):
        field = path[i]
        out.append(field)
        prefix.append(field)
        if i + 1 == len(path):
            break  # a block path: no index at this level

        idx = path[i + 1]
        assert isinstance(idx, int)
        shift = 0
        containing: Edit | None = None
        for e in edits:
            if e.block_path != tuple(prefix):
                continue
            if idx >= e.index + e.removed:
                shift += e.inserted - e.removed
            elif idx >= e.index:
                containing = e

        if containing is not None:
            if i + 2 < len(path):
                raise TransformReferenceError(
                    f'`{format_path(path)}` is inside `'
                    f'{format_path((*prefix, idx))}`, which was rewritten'
                )
            return tuple(out), containing, containing.index + shift

        out.append(idx + shift)
        prefix.append(idx)
    return tuple(out), None, 0
