"""
Where an AST node holds its blocks, sub-expressions and variables.
"""

from collections.abc import Iterable
from typing import Literal, TypeAlias

from .fpyast import (
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
    If1Stmt,
    IfExpr,
    IfStmt,
    IndexedAssign,
    ListComp,
    ListExpr,
    ListRef,
    ListSlice,
    NamedId,
    NaryOp,
    NullaryOp,
    ReturnStmt,
    Stmt,
    StmtBlock,
    TernaryOp,
    TupleExpr,
    UnaryOp,
    Var,
    WhileStmt,
)
from .visitor import DefaultVisitor

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

A typo'd field is then a type error, and :func:`subexprs` is checked against
this list.
"""


def subblocks(stmt: Stmt) -> tuple[tuple[BlockField, StmtBlock], ...]:
    """The blocks *stmt* encloses, each with the field that names it."""
    match stmt:
        case IfStmt():
            return ('ift', stmt.ift), ('iff', stmt.iff)
        case If1Stmt() | WhileStmt() | ForStmt() | ContextStmt():
            return ('body', stmt.body),
        case _:
            return ()


def subexprs(node: Stmt | Expr) -> tuple[tuple[ExprField, int | None, Expr], ...]:
    """The expressions *node* holds, each with the field and position naming
    it, in evaluation order."""
    def at(
        field: ExprField, es: Iterable[Expr]
    ) -> tuple[tuple[ExprField, int | None, Expr], ...]:
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


class _Vars(DefaultVisitor):
    found: list[Var]

    def __init__(self) -> None:
        self.found = []

    def _visit_var(self, e: Var, ctx: None) -> None:
        self.found.append(e)


def vars_in(node: Expr | Stmt) -> list[Var]:
    """Every `Var` in *node*, in visit order.

    Syntactic, so a comprehension's own target counts -- unlike
    :class:`~fpy2.analysis.LiveVars`.
    """
    v = _Vars()
    if isinstance(node, Expr):
        v._visit_expr(node, None)
    else:
        v._visit_statement(node, None)
    return v.found


def names_in(node: Expr | Stmt) -> set[NamedId]:
    """The names of :func:`vars_in`."""
    return {v.name for v in vars_in(node)}
