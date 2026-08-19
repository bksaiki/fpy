"""
Where a block or a statement sits in a program.

A path is the grammar the AST already has, written down::

    BlockPath ::= FuncBody                  -- base: the function's own body
                | StmtPath . field          -- a block held by a statement
    StmtPath  ::= BlockPath [ index ]       -- a statement of a block
    ExprPath  ::= StmtPath . field [ i? ]   -- an expression of a statement
                | ExprPath . field [ i? ]   -- ... or of another expression

as parent-linked frozen dataclasses, so every ill-formed path is unrepresentable.
`FuncBody` is the only constructor without a parent, so every path is absolute; a
path's type is the type of its leaf, which is what a cursor is named by.

Build one by descending::

    FuncBody().stmt(1).block('ift').stmt(0)          # body[1].ift[0]
    FuncBody().stmt(0).expr('expr').expr('args', 1)  # body[0].expr.args[1]
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, TypeAlias

from ...ast.fpyast import (
    AssertStmt,
    Assign,
    Attribute,
    BinaryOp,
    Call,
    Compare,
    ContextStmt,
    EffectStmt,
    Expr,
    ForStmt,
    FuncDef,
    If1Stmt,
    IfExpr,
    IfStmt,
    IndexedAssign,
    ListComp,
    ListExpr,
    ListRef,
    ListSlice,
    NaryOp,
    NullaryOp,
    ReturnStmt,
    Stmt,
    StmtBlock,
    TernaryOp,
    TupleExpr,
    UnaryOp,
    WhileStmt,
)
from .error import TransformReferenceError

BlockField: TypeAlias = Literal['body', 'ift', 'iff']
"""The fields a statement can hold a block in."""

ExprField: TypeAlias = Literal[
    # of a statement
    'expr', 'indices', 'cond', 'iterable', 'ctx', 'test', 'msg',
    # of an expression
    'args', 'kwargs', 'elts', 'value', 'index', 'start', 'stop',
    'iterables', 'elt', 'ift', 'iff',
]
"""The fields a statement or expression can hold an expression in.

A typo'd field is then a type error, and :func:`sub_exprs` is checked against
this list.
"""


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

    def expr(self, field: ExprField, index: int | None = None) -> 'ExprPath':
        """The expression this statement holds in *field*."""
        return ExprPath(self, field, index)


@dataclass(frozen=True)
class ExprPath:
    """An expression of a statement, or of another expression.

    *index* positions it where the field holds several (`args`, `elts`); it is
    `None` where the field holds one (`arg`, `cond`).
    """

    parent: 'StmtPath | ExprPath'
    field: ExprField
    index: int | None = None

    def expr(self, field: ExprField, index: int | None = None) -> 'ExprPath':
        """The expression this one holds in *field*."""
        return ExprPath(self, field, index)

    def stmt(self) -> StmtPath:
        """The statement this expression belongs to."""
        p = self.parent
        while isinstance(p, ExprPath):
            p = p.parent
        return p


Path: TypeAlias = BlockPath | StmtPath | ExprPath
"""Any path, for the operations that take more than one kind."""


def format_path(path: Path) -> str:
    """A path as `body[1].ift[0]` or `body[0].expr.args[1]`."""
    match path:
        case FuncBody():
            return 'body'
        case SubBlock(parent, field):
            return f'{format_path(parent)}.{field}'
        case StmtPath(parent, index):
            return f'{format_path(parent)}[{index}]'
        case ExprPath(parent, field, index):
            at = '' if index is None else f'[{index}]'
            return f'{format_path(parent)}.{field}{at}'


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


def sub_exprs(node: Stmt | Expr) -> tuple[tuple[ExprField, int | None, Expr], ...]:
    """The expressions *node* holds, each with the field and position naming it.

    The only place the AST's expression field names appear.
    """
    def at(field: ExprField, es) -> tuple[tuple[ExprField, int | None, Expr], ...]:
        return tuple((field, i, e) for i, e in enumerate(es))

    match node:
        # statements
        case Assign() | EffectStmt() | ReturnStmt():
            return ('expr', None, node.expr),
        case IndexedAssign():
            return *at('indices', node.indices), ('expr', None, node.expr)
        case If1Stmt() | IfStmt() | WhileStmt():
            return ('cond', None, node.cond),
        case ForStmt():
            return ('iterable', None, node.iterable),
        case ContextStmt():
            return ('ctx', None, node.ctx),
        case AssertStmt():
            if node.msg is None:
                return ('test', None, node.test),
            return ('test', None, node.test), ('msg', None, node.msg)
        # expressions -- every operator holds its operands in `args`, whatever
        # its arity, and `arg` / `first` / `second` are properties over that
        case Call():
            return *at('args', node.args), *at('kwargs', [v for _, v in node.kwargs])
        case NullaryOp() | UnaryOp() | BinaryOp() | TernaryOp() | NaryOp() | Compare():
            return at('args', node.args)
        case TupleExpr() | ListExpr():
            return at('elts', node.elts)
        case ListRef():
            return ('value', None, node.value), ('index', None, node.index)
        case ListSlice():
            out: list[tuple[ExprField, int | None, Expr]] = [
                ('value', None, node.value)
            ]
            if node.start is not None:
                out.append(('start', None, node.start))
            if node.stop is not None:
                out.append(('stop', None, node.stop))
            return tuple(out)
        case ListComp():
            return *at('iterables', node.iterables), ('elt', None, node.elt)
        case IfExpr():
            return (('cond', None, node.cond), ('ift', None, node.ift),
                    ('iff', None, node.iff))
        case Attribute():
            return ('value', None, node.value),
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


def resolve_expr(func: FuncDef, path: ExprPath) -> Expr:
    """The expression *path* names in *func*."""
    match path.parent:
        case ExprPath() as parent:
            node: Stmt | Expr = resolve_expr(func, parent)
        case parent:
            node = resolve_stmt(func, parent)
    for field, index, e in sub_exprs(node):
        if field == path.field and index == path.index:
            return e
    raise bad_path(
        path, f'names no `{path.field}` expression of a `{type(node).__name__}`'
    )


def rebase_expr(path: ExprPath, stmt: StmtPath) -> ExprPath:
    """*path*, hanging off *stmt* instead of the statement it named."""
    match path.parent:
        case ExprPath() as parent:
            return ExprPath(rebase_expr(parent, stmt), path.field, path.index)
        case _:
            return ExprPath(stmt, path.field, path.index)


def beneath(path: Path, block: BlockPath, span: range) -> bool:
    """Whether *path* lies at or under one of *block*'s statements in *span*."""
    p: BlockPath | StmtPath = path.stmt() if isinstance(path, ExprPath) else path
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


def walk_stmts(func: FuncDef) -> Iterator[tuple[StmtPath, Stmt]]:
    """Every statement of *func* with its path, in visit order.

    A statement comes before the blocks it holds: the order a `where` index
    counts candidates in.
    """
    def walk(block: StmtBlock, path: BlockPath) -> Iterator[tuple[StmtPath, Stmt]]:
        for i, stmt in enumerate(block.stmts):
            here = StmtPath(path, i)
            yield here, stmt
            for field, sub in sub_blocks(stmt):
                yield from walk(sub, SubBlock(here, field))

    yield from walk(func.body, FuncBody())


def block_paths(func: FuncDef) -> dict[int, BlockPath]:
    """The path of every block in *func*, keyed by `id`.

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
