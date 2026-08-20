"""
The cursor a match carries.

A match holds a cursor, so its location can be resolved, compared and
forwarded.
"""

import pytest

from fpy2 import fpy, pattern
from fpy2.ast import Add, Mul
from fpy2.rewrite.matcher import Matcher
from fpy2.transform import BlockCursor, ExprCursor, FuncBody, StmtCursor


@pattern
def mul_add(a, b, c):
    a * b + c


@pattern
def one_stmt(a):
    y = a * 2


@pattern
def two_stmts(a):
    y = a * 2
    z = y + 1


@fpy
def twice(x, y, z):
    t = x * y + z
    u = z * y + x
    return t + u


@fpy
def nested(x, y, z):
    if x > 0:
        t = x * y + z
    else:
        t = z * y + x
    return t


@fpy
def two_then_one(x):
    y = x * 2
    z = y + 1
    return z


# ----------------------------------------------------------------------
# Expression patterns


def test_an_expression_match_carries_an_expression_cursor():
    matches = Matcher(mul_add).match(twice)
    assert len(matches) == 2
    for m in matches:
        assert isinstance(m.cursor, ExprCursor)
        # the cursor names the very expression that matched
        assert isinstance(m.cursor.resolve(), Add)
        assert m.cursor.func is twice.ast


def test_expression_cursors_are_in_traversal_order():
    """Match `i` is the `i`th in visit order, which is what `where=i` names."""
    matches = Matcher(mul_add).match(twice)
    assert [m.cursor.path.stmt() for m in matches] == [
        FuncBody().stmt(0), FuncBody().stmt(1)
    ]


def test_a_match_inside_a_branch_is_named_by_its_path():
    matches = Matcher(mul_add).match(nested)
    assert [m.cursor.path.stmt() for m in matches] == [
        FuncBody().stmt(0).block('ift').stmt(0),
        FuncBody().stmt(0).block('iff').stmt(0),
    ]


def test_the_cursor_agrees_with_the_substitution():
    """`a * b + c` against `x * y + z`: the cursor's own subexpression is what
    `a * b` bound to."""
    m, _ = Matcher(mul_add).match(twice)
    add = m.cursor.resolve()
    assert isinstance(add, Add)
    left = ExprCursor(twice.ast, m.cursor.path.expr('args', 0))
    assert isinstance(left.resolve(), Mul)


# ----------------------------------------------------------------------
# Statement patterns


def test_a_one_statement_match_carries_a_statement_cursor():
    matches = Matcher(one_stmt).match(two_then_one)
    assert len(matches) == 1
    assert matches[0].cursor == StmtCursor(two_then_one.ast, FuncBody().stmt(0))


def test_a_k_statement_match_carries_a_region():
    """A pattern of two statements matches a run of two, which is a region."""
    matches = Matcher(two_stmts).match(two_then_one)
    assert len(matches) == 1
    cursor = matches[0].cursor
    assert cursor == BlockCursor(two_then_one.ast, FuncBody(), range(0, 2))
    assert [s.format() for s in cursor.resolve()] == [
        s.format() for s in two_then_one.ast.body.stmts[0:2]
    ]


def test_a_pattern_longer_than_the_block_matches_nothing():
    @fpy
    def one(x):
        return x

    assert Matcher(two_stmts).match(one) == []


# ----------------------------------------------------------------------
# The overlap the sliding window can produce


@pattern
def two_adds(a, b):
    y = a + 1
    z = b + 1


@fpy
def three_adds(x):
    a = x + 1
    b = x + 1
    c = x + 1
    return a + b + c


def test_the_window_produces_overlapping_matches():
    """Windows at 0 and 1 both match, sharing statement 1 -- fine for a cursor,
    but a rewrite cannot apply to both."""
    matches = Matcher(two_adds).match(three_adds)
    spans = [m.cursor.span for m in matches if isinstance(m.cursor, BlockCursor)]
    assert spans == [range(0, 2), range(1, 3)]

    first, second = spans
    assert set(first) & set(second) == {1}


def test_a_substitution_still_reaches_its_bindings():
    """`subst` is what the applier reads."""
    m, = Matcher(one_stmt).match(two_then_one)
    assert 'a' in {str(v) for v in m.subst.vars()}
    assert m.pattern is one_stmt


def test_a_match_is_a_value():
    matches = Matcher(mul_add).match(twice)
    assert matches[0] == matches[0]
    assert matches[0] != matches[1]
    with pytest.raises(Exception):
        matches[0].cursor = matches[1].cursor  # frozen
