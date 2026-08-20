"""
Unit tests for :mod:`fpy2.transform.path`.

A cursor is a path from the :class:`FuncDef`, so the tests pin the two
directions -- :func:`block_paths` writes paths for the blocks a visitor holds,
:func:`resolve_block` / :func:`resolve_stmt` read them back -- and the failures,
since a path that names nothing must say so rather than land somewhere.
"""

import pytest

import fpy2 as fp

from fpy2.ast import Assign, ContextStmt, ForStmt, IfStmt, ReturnStmt
from fpy2.transform import (
    FuncBody,
    StmtCursor,
    StmtPath,
    SubBlock,
    TransformReferenceError,
)
from fpy2.ast.visitor import DefaultTransformVisitor
from fpy2.transform.path import (
    block_paths,
    format_path,
    resolve_block,
    resolve_expr,
    resolve_stmt,
    sub_blocks,
    walk_blocks,
    walk_exprs,
    walk_stmts,
)


@fp.fpy(ctx=fp.REAL)
def nested(x: fp.Real, xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    if x > 0:
        with fp.FP32:
            y = fp.round(x)
    else:
        for v in xs:
            acc = acc + v
        y = acc
    return y


@fp.fpy(ctx=fp.REAL)
def flat(x: fp.Real) -> fp.Real:
    with fp.FP16:
        y = fp.round(x)
    return y


# ----------------------------------------------------------------------
# Paths


def test_format_path():
    assert format_path(FuncBody()) == 'body'
    assert format_path(FuncBody().stmt(1).block('ift').stmt(0)) == 'body[1].ift[0]'


def test_a_path_is_built_by_descending():
    """Each step's type says what may follow it: a block holds statements, a
    statement holds blocks."""
    p = FuncBody().stmt(1).block('ift').stmt(0)
    assert p == StmtPath(SubBlock(StmtPath(FuncBody(), 1), 'ift'), 0)
    assert p.parent.parent.index == 1


def test_block_paths_round_trip():
    """Every block `block_paths` names resolves back to that same block."""
    ast = nested.ast
    paths = block_paths(ast)
    assert len(paths) == 5  # body, ift, iff, the `with` body, the `for` body
    for ident, path in paths.items():
        assert id(resolve_block(ast, path)) == ident


def test_block_paths_shape():
    ast = nested.ast
    paths = set(block_paths(ast).values())
    assert paths == {
        FuncBody(),
        FuncBody().stmt(1).block('ift'),
        FuncBody().stmt(1).block('ift').stmt(0).block('body'),
        FuncBody().stmt(1).block('iff'),
        FuncBody().stmt(1).block('iff').stmt(0).block('body'),
    }


def test_resolve_stmt():
    ast = nested.ast
    assert isinstance(resolve_stmt(ast, FuncBody().stmt(0)), Assign)
    assert isinstance(resolve_stmt(ast, FuncBody().stmt(1)), IfStmt)
    assert isinstance(resolve_stmt(ast, FuncBody().stmt(2)), ReturnStmt)
    assert isinstance(resolve_stmt(ast, FuncBody().stmt(1).block('ift').stmt(0)), ContextStmt)
    assert isinstance(resolve_stmt(ast, FuncBody().stmt(1).block('iff').stmt(0)), ForStmt)


def test_sub_blocks():
    ast = nested.ast
    assert sub_blocks(resolve_stmt(ast, FuncBody().stmt(0))) == ()
    fields = [f for f, _ in sub_blocks(resolve_stmt(ast, FuncBody().stmt(1)))]
    assert fields == ['ift', 'iff']


# ----------------------------------------------------------------------
# The walks agree with the visitor


@fp.fpy(ctx=fp.REAL)
def busy(x: fp.Real, xs: list[fp.Real], n: fp.Real) -> fp.Real:
    a = x * 2 + 1
    acc = 0.0
    for v in xs[0:3]:
        with fp.MPBFixedContext(-4, 1024, overflow=fp.OverflowMode.WRAP):
            acc = fp.round(acc + v * a)
    if x > 0:
        b = min(x, n)
    else:
        b = -x
    i = 0.0
    while i < n:
        i = i + 1
    ys = [v * 2 for v in xs]
    assert len(ys) >= 0, 'nonneg'
    return acc + b + i + a


class _Recorder(DefaultTransformVisitor):
    """What the visitor reaches, in the order it reaches it.

    The *transform* visitor: it is the one every rewrite is built on, so it is
    the traversal a path has to agree with.  (`DefaultVisitor` differs — see
    `_visit_call`, which skips keyword arguments.)
    """

    def __init__(self):
        self.exprs: list = []
        self.stmts: list = []
        self.blocks: list = []

    def _visit_expr(self, e, ctx):
        self.exprs.append(e)
        return super()._visit_expr(e, ctx)

    def _visit_statement(self, s, ctx):
        self.stmts.append(s)
        return super()._visit_statement(s, ctx)

    def _visit_block(self, b, ctx):
        self.blocks.append(b)
        return super()._visit_block(b, ctx)


def test_the_walks_agree_with_the_visitor():
    """`sub_blocks` / `sub_exprs` name the fields the visitor descends through
    without naming, so the two encode the same tree shape and order.  A `where`
    index counts in the visitor's order and a listing reports in the walks', so a
    drift between them would silently aim one place and report another.
    """
    seen = _Recorder()
    seen._visit_function(busy.ast, None)

    assert [e for _, e in walk_exprs(busy.ast)] == seen.exprs
    assert [s for _, s in walk_stmts(busy.ast)] == seen.stmts
    assert [b for _, b in walk_blocks(busy.ast)] == seen.blocks


def test_every_walked_path_resolves_to_what_was_walked():
    for path, e in walk_exprs(busy.ast):
        assert resolve_expr(busy.ast, path) is e
    for path, s in walk_stmts(busy.ast):
        assert resolve_stmt(busy.ast, path) is s
    for path, b in walk_blocks(busy.ast):
        assert resolve_block(busy.ast, path) is b


# ----------------------------------------------------------------------
# Bad references


# only resolvable-but-wrong paths remain; the ill-formed ones are type errors


@pytest.mark.parametrize('path', [
    FuncBody().stmt(9).block('ift'),        # index past the end
    FuncBody().stmt(1).block('body'),       # an `IfStmt` has no `body`
    FuncBody().stmt(0).block('body'),       # an `Assign` has no block at all
])
def test_resolve_block_rejects(path):
    with pytest.raises(TransformReferenceError):
        resolve_block(nested.ast, path)


@pytest.mark.parametrize('path', [
    FuncBody().stmt(9),                          # index past the end
    FuncBody().stmt(1).block('ift').stmt(3),     # past the end of a nested block
])
def test_resolve_stmt_rejects(path):
    with pytest.raises(TransformReferenceError):
        resolve_stmt(nested.ast, path)


# ----------------------------------------------------------------------
# The cursor itself


def test_cursor_parts():
    cur = StmtCursor(nested.ast, FuncBody().stmt(1).block('ift').stmt(0))
    assert cur.block_path == FuncBody().stmt(1).block('ift')
    assert cur.index == 0
    assert isinstance(cur.resolve(), ContextStmt)

    block = resolve_block(nested.ast, cur.block_path)
    assert block.stmts[cur.index] is cur.resolve()


def test_cursor_validated_on_construction():
    """A cursor always names a statement of its own program."""
    with pytest.raises(TransformReferenceError):
        StmtCursor(nested.ast, FuncBody().stmt(9))
    with pytest.raises(TypeError):
        StmtCursor(nested.ast, FuncBody())   # type: ignore[arg-type]
    with pytest.raises(TypeError):
        StmtCursor(nested, FuncBody().stmt(0))      # type: ignore[arg-type]


def test_cursor_equality_is_per_program():
    """Same path, different program: different cursor."""
    a = StmtCursor(nested.ast, FuncBody().stmt(0))
    assert a == StmtCursor(nested.ast, FuncBody().stmt(0))
    assert a != StmtCursor(nested.ast, FuncBody().stmt(1))
    assert a != StmtCursor(flat.ast, FuncBody().stmt(0))


def test_cursor_str_names_the_source():
    cur = StmtCursor(flat.ast, FuncBody().stmt(0))
    text = str(cur)
    assert text.startswith('body[0] at ')
    assert 'test_cursor.py' in text
