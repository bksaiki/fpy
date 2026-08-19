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
