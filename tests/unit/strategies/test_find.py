"""
Naming a location by what it looks like.

`find_all` lists every match; `find` insists on one, because a pattern that
matches three places has not identified anything and indexing the first would
be the quiet wrong-site a cursor exists to prevent.
"""

import pytest

import fpy2 as fp

from fpy2.ast import Add, Assign
from fpy2.rewrite import find, find_all
from fpy2.strategies import (
    BlockCursor,
    ExprCursor,
    FuncBody,
    StmtCursor,
    TransformReferenceError,
    unfold_special,
)


@fp.pattern
def mul_add(a, b, c):
    a * b + c


@fp.pattern
def scale(a):
    y = a * 2


@fp.pattern
def scale_then_bump(a):
    y = a * 2
    z = y + 1


@fp.fpy
def twice(x, y, z):
    t = x * y + z
    u = z * y + x
    return t + u


@fp.fpy
def once(x, y, z):
    return x * y + z


@fp.fpy
def branchy(x, y, z):
    if x > 0:
        t = x * y + z
    else:
        t = z
    return t


@fp.fpy
def stmts(x):
    y = x * 2
    z = y + 1
    return z


# ----------------------------------------------------------------------
# find_all


def test_find_all_lists_expression_matches_in_order():
    found = find_all(mul_add, twice)
    assert all(isinstance(c, ExprCursor) for c in found)
    assert [c.path.stmt() for c in found] == [FuncBody().stmt(0), FuncBody().stmt(1)]
    assert all(isinstance(c.resolve(), Add) for c in found)


def test_find_all_reaches_into_a_branch():
    found = find_all(mul_add, branchy)
    assert [c.path.stmt() for c in found] == [FuncBody().stmt(0).block('ift').stmt(0)]


def test_find_all_lists_a_one_statement_match_as_a_statement():
    found = find_all(scale, stmts)
    assert found == [StmtCursor(stmts.ast, FuncBody().stmt(0))]
    assert isinstance(found[0].resolve(), Assign)


def test_find_all_lists_a_k_statement_match_as_a_region():
    found = find_all(scale_then_bump, stmts)
    assert found == [BlockCursor(stmts.ast, FuncBody(), range(0, 2))]


def test_find_all_returns_nothing_where_it_matches_nothing():
    @fp.fpy
    def plain(x):
        return x

    assert find_all(mul_add, plain) == []
    assert find_all(scale_then_bump, plain) == []


# ----------------------------------------------------------------------
# find


def test_find_returns_the_one_match():
    cursor = find(mul_add, once)
    assert isinstance(cursor, ExprCursor)
    assert cursor.path.stmt() == FuncBody().stmt(0)


def test_find_refuses_when_nothing_matches():
    @fp.fpy
    def plain(x):
        return x

    with pytest.raises(TransformReferenceError, match='matches nothing'):
        find(mul_add, plain)


def test_find_refuses_when_several_match_and_says_how_many():
    with pytest.raises(TransformReferenceError, match=r'matches 2 places') as exc:
        find(mul_add, twice)
    assert 'find_all' in str(exc.value)
    assert 'mul_add' in str(exc.value)


def test_find_narrowed_by_within_becomes_unambiguous():
    """The escape hatch the error suggests: narrow, rather than index."""
    second = StmtCursor(twice.ast, FuncBody().stmt(1))
    cursor = find(mul_add, twice, second)
    assert cursor.path.stmt() == FuncBody().stmt(1)


# ----------------------------------------------------------------------
# within


def test_within_keeps_the_matches_beneath_a_region():
    part = BlockCursor(twice.ast, FuncBody(), range(0, 1))
    assert [c.path.stmt() for c in find_all(mul_add, twice, part)] == [
        FuncBody().stmt(0)
    ]


def test_within_keeps_the_matches_beneath_an_expression():
    """A statement's whole expression contains the match; its left operand does
    not."""
    whole = ExprCursor(once.ast, FuncBody().stmt(0).expr('expr'))
    assert find_all(mul_add, once, whole) == [whole]

    left = ExprCursor(once.ast, FuncBody().stmt(0).expr('expr').expr('args', 0))
    assert find_all(mul_add, once, left) == []


def test_a_statement_pattern_cannot_be_narrowed_by_an_expression():
    cur = ExprCursor(stmts.ast, FuncBody().stmt(0).expr('expr'))
    with pytest.raises(TransformReferenceError, match='these sites are statements'):
        find_all(scale, stmts, cur)


def test_within_is_forwarded_from_an_earlier_program():
    """A site listed against one program narrows a search against a later one,
    without the caller forwarding it by hand."""
    site = StmtCursor(twice.ast, FuncBody().stmt(1))
    out = unfold_special(twice)          # nothing to rewrite, but a new program

    assert [c.path.stmt() for c in find_all(mul_add, out, site)] == [
        FuncBody().stmt(1)
    ]


def test_within_of_an_unrelated_program_is_a_bad_reference():
    other = StmtCursor(stmts.ast, FuncBody().stmt(0))
    with pytest.raises(TransformReferenceError, match='unrelated program'):
        find_all(mul_add, twice, other)


# ----------------------------------------------------------------------
# Overlap, which the caller has to know about


@fp.pattern
def two_bumps(a, b):
    y = a + 1
    z = b + 1


def test_overlapping_matches_are_all_listed():
    @fp.fpy
    def three(x):
        a = x + 1
        b = x + 1
        c = x + 1
        return a + b + c

    found = find_all(two_bumps, three)
    assert [c.span for c in found if isinstance(c, BlockCursor)] == [
        range(0, 2), range(1, 3)
    ]
