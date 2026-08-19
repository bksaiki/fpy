"""
Unit tests for :mod:`fpy2.transform.cursor`.

A cursor is a path from the :class:`FuncDef`, so the tests pin the two
directions -- :func:`block_paths` writes paths for the blocks a visitor holds,
:func:`resolve_block` / :func:`resolve_stmt` read them back -- and the failures,
since a path that names nothing must say so rather than land somewhere.
"""

import pytest

import fpy2 as fp

from fpy2.ast import Assign, ContextStmt, ForStmt, IfStmt, ReturnStmt
from fpy2.transform import Cursor, TransformReferenceError
from fpy2.transform.utils.cursor import (
    block_paths,
    format_path,
    resolve_block,
    resolve_stmt,
    sub_blocks,
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
    assert format_path(('body',)) == 'body'
    assert format_path(('body', 1, 'ift', 0)) == 'body[1].ift[0]'


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
        ('body',),
        ('body', 1, 'ift'),
        ('body', 1, 'ift', 0, 'body'),
        ('body', 1, 'iff'),
        ('body', 1, 'iff', 0, 'body'),
    }


def test_resolve_stmt():
    ast = nested.ast
    assert isinstance(resolve_stmt(ast, ('body', 0)), Assign)
    assert isinstance(resolve_stmt(ast, ('body', 1)), IfStmt)
    assert isinstance(resolve_stmt(ast, ('body', 2)), ReturnStmt)
    assert isinstance(resolve_stmt(ast, ('body', 1, 'ift', 0)), ContextStmt)
    assert isinstance(resolve_stmt(ast, ('body', 1, 'iff', 0)), ForStmt)


def test_sub_blocks():
    ast = nested.ast
    assert sub_blocks(resolve_stmt(ast, ('body', 0))) == ()
    fields = [f for f, _ in sub_blocks(resolve_stmt(ast, ('body', 1)))]
    assert fields == ['ift', 'iff']


# ----------------------------------------------------------------------
# Bad references


@pytest.mark.parametrize('path', [
    ('body', 1),                      # even: not a block path
    ('ift',),                         # does not start at the body
    ('body', 9, 'ift'),               # index past the end
    ('body', 1, 'body'),              # an `IfStmt` has no `body`
    ('body', 0, 'body'),              # an `Assign` has no block at all
    ('body', 'ift', 'ift'),           # a field where an index belongs
])
def test_resolve_block_rejects(path):
    with pytest.raises(TransformReferenceError):
        resolve_block(nested.ast, path)


@pytest.mark.parametrize('path', [
    (),                               # empty
    ('body',),                        # odd: not a statement path
    ('body', 9),                      # index past the end
    ('body', 1, 'ift', 3),            # index past the end of a nested block
])
def test_resolve_stmt_rejects(path):
    with pytest.raises(TransformReferenceError):
        resolve_stmt(nested.ast, path)


# ----------------------------------------------------------------------
# The cursor itself


def test_cursor_parts():
    cur = Cursor(nested.ast, ('body', 1, 'ift', 0))
    assert cur.block_path == ('body', 1, 'ift')
    assert cur.index == 0
    assert isinstance(cur.resolve(), ContextStmt)

    block, idx = cur.parent()
    assert block.stmts[idx] is cur.resolve()
    assert block is resolve_block(nested.ast, ('body', 1, 'ift'))


def test_cursor_validated_on_construction():
    """A cursor always names a statement of its own program."""
    with pytest.raises(TransformReferenceError):
        Cursor(nested.ast, ('body', 9))
    with pytest.raises(TypeError):
        Cursor(nested.ast, ['body', 0])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Cursor(nested, ('body', 0))      # type: ignore[arg-type]


def test_cursor_equality_is_per_program():
    """Same path, different program: different cursor."""
    a = Cursor(nested.ast, ('body', 0))
    assert a == Cursor(nested.ast, ('body', 0))
    assert a != Cursor(nested.ast, ('body', 1))
    assert a != Cursor(flat.ast, ('body', 0))


def test_cursor_str_names_the_source():
    cur = Cursor(flat.ast, ('body', 0))
    text = str(cur)
    assert text.startswith('body[0] at ')
    assert 'test_cursor.py' in text
