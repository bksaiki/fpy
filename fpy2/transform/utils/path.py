"""
Where a block or a statement sits in a program.

A path is the grammar the AST already has, written down::

    BlockPath ::= FuncBody | StmtPath . field       -- 'body' | 'ift' | 'iff'
    StmtPath  ::= BlockPath [ index ]

as parent-linked frozen dataclasses, so every ill-formed path is unrepresentable:
a sub-block hangs off a statement, and a statement sits in a block.  Linking to
the *parent* is what makes a path's type the type of its leaf, which is what a
cursor is named by -- a :class:`~fpy2.transform.utils.cursor.Cursor` takes a
`StmtPath` and mypy checks it.

Build one by descending::

    FuncBody().stmt(1).block('ift').stmt(0)          # body[1].ift[0]
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

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

BlockField: TypeAlias = Literal['body', 'ift', 'iff']
"""The fields a statement can hold a block in."""


@dataclass(frozen=True)
class FuncBody:
    """The function's own body: where every path starts."""

    def stmt(self, index: int) -> 'StmtPath':
        """The statement at *index* of this block."""
        return StmtPath(self, index)


@dataclass(frozen=True)
class SubBlock:
    """A block belonging to a statement: a loop body, an arm of an `if`."""

    parent: 'StmtPath'
    field: BlockField

    def stmt(self, index: int) -> 'StmtPath':
        """The statement at *index* of this block."""
        return StmtPath(self, index)


BlockPath: TypeAlias = FuncBody | SubBlock
"""A block of a program: the function's body, or one held by a statement."""


@dataclass(frozen=True)
class StmtPath:
    """A statement of a block."""

    parent: BlockPath
    index: int

    def block(self, field: BlockField) -> SubBlock:
        """The block this statement holds in *field*."""
        return SubBlock(self, field)


Path: TypeAlias = BlockPath | StmtPath
"""Either kind of path, for the operations that take both."""


def format_path(path: Path) -> str:
    """A path as `body[1].ift[0]`."""
    match path:
        case FuncBody():
            return 'body'
        case SubBlock(parent, field):
            return f'{format_path(parent)}.{field}'
        case StmtPath(parent, index):
            return f'{format_path(parent)}[{index}]'


def bad_path(path: Path, why: str) -> TransformReferenceError:
    """A path that names nothing, said with the path."""
    return TransformReferenceError(f'`{format_path(path)}` {why}')


def sub_blocks(stmt: Stmt) -> tuple[tuple[BlockField, StmtBlock], ...]:
    """The blocks *stmt* encloses, each with the field that names it."""
    match stmt:
        case IfStmt():
            return ('ift', stmt.ift), ('iff', stmt.iff)
        case If1Stmt() | WhileStmt() | ForStmt() | ContextStmt():
            return ('body', stmt.body),
        case _:
            return ()


def resolve_block(func: FuncDef, path: BlockPath) -> StmtBlock:
    """The block *path* names in *func*."""
    match path:
        case FuncBody():
            return func.body
        case SubBlock(parent, field):
            stmt = resolve_stmt(func, parent)
            for name, block in sub_blocks(stmt):
                if name == field:
                    return block
            raise bad_path(path, f'names no `{field}` block of a `{type(stmt).__name__}`')


def resolve_stmt(func: FuncDef, path: StmtPath) -> Stmt:
    """The statement *path* names in *func*."""
    block = resolve_block(func, path.parent)
    if not 0 <= path.index < len(block.stmts):
        raise bad_path(path, f'names statement {path.index} of a block of {len(block.stmts)}')
    return block.stmts[path.index]


def beneath(path: Path, block: BlockPath, span: range) -> bool:
    """Whether *path* lies at or under one of *block*'s statements in *span*.

    The upward walk a parent-linked path is for: what a rewrite aimed at a
    region selects, and what an edit inside a replaced statement fails.
    """
    p = path
    while True:
        match p:
            case FuncBody():
                return False
            case SubBlock(parent, _):
                p = parent
            case StmtPath(parent, index):
                if parent == block and index in span:
                    return True
                p = parent


def block_paths(func: FuncDef) -> dict[int, BlockPath]:
    """The path of every block in *func*, keyed by `id`.

    The inverse of :func:`resolve_block`, for the transforms: a visitor knows
    the block object it is rewriting in, and needs its path to record an edit.
    Valid only while *func* is alive.
    """
    paths: dict[int, BlockPath] = {}

    def walk(block: StmtBlock, path: BlockPath) -> None:
        paths[id(block)] = path
        for i, stmt in enumerate(block.stmts):
            for field, sub in sub_blocks(stmt):
                walk(sub, SubBlock(StmtPath(path, i), field))

    walk(func.body, FuncBody())
    return paths
