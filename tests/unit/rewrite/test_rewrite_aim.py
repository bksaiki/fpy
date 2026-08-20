"""
Which match a `where` names.

An index and the cursor at that index must name the same match, a region must
not reach past what it names, and a plural aim must not quietly rewrite a
subset.  Nested programs are the cases that separate these.
"""

import pytest

import fpy2 as fp

from fpy2.rewrite import Rewrite, find_all
from fpy2.strategies import (
    BlockCursor,
    FuncBody,
    StmtCursor,
    TransformDeclined,
    TransformReferenceError,
)


@fp.pattern
def fma_l(a, b, c):
    a * b + c


@fp.pattern
def fma_r(a, b, c):
    fp.fma(a, b, c)


@fp.pattern
def bump_l(a):
    y = a + 1


@fp.pattern
def bump_r(a):
    y = a + 2


@fp.pattern
def pair_l(a, b):
    y = a + 1
    z = b + 1


@fp.pattern
def pair_r(a, b):
    y = a + 1
    z = b + 1
    z = z + y


fma = Rewrite(fma_l, fma_r)
bump = Rewrite(bump_l, bump_r)
widen = Rewrite(pair_l, pair_r)


# ----------------------------------------------------------------------
# An index names the match `find_all` lists at that index


def test_an_index_and_its_cursor_agree_on_nested_expressions():
    """The visit rebuilds children first, so a counter would number the inner
    match zero while `find_all` numbers the outer one zero."""
    @fp.fpy
    def nest(x, y, z, w, v):
        return (x * y + z) * w + v

    for i, cursor in enumerate(find_all(fma_l, nest)):
        assert fma.apply(nest, i).format() == fma.apply(nest, cursor).format()


def test_an_index_and_its_cursor_agree_across_nested_blocks():
    @fp.fpy
    def h(x):
        y = x + 1
        if y > 0:
            z = x + 1
            w = x + 1
        q = x + 1
        return y

    cursors = find_all(bump_l, h)
    assert len(cursors) == 4
    for i, cursor in enumerate(cursors):
        assert bump.apply(h, i).format() == bump.apply(h, cursor).format()


def test_the_first_index_is_the_outermost_expression():
    @fp.fpy
    def nest(x, y, z, w, v):
        return (x * y + z) * w + v

    # the outer match: the inner one survives as an argument
    assert 'fp.fma(((x * y) + z), w, v)' in fma.apply(nest, 0).format()


# ----------------------------------------------------------------------
# A region does not reach past itself


def test_a_region_does_not_select_a_window_it_only_partly_holds():
    """`within` and `where` have to agree: the match is `body[1:3]`, so a region
    naming `body[0:2]` names no match rather than rewriting statement 2."""
    @fp.fpy
    def f(x):
        a = x * 2
        b = x + 1
        c = x + 1
        return a + b + c

    region = BlockCursor(f.ast, FuncBody(), range(0, 2))
    assert find_all(pair_l, f, region) == []
    with pytest.raises(TransformReferenceError, match='does not correspond'):
        widen.apply(f, region)


def test_a_region_holding_the_whole_window_still_selects_it():
    @fp.fpy
    def f(x):
        a = x * 2
        b = x + 1
        c = x + 1
        return a + b + c

    region = BlockCursor(f.ast, FuncBody(), range(1, 3))
    out = widen.apply(f, region)
    edit, = out.edits.edits
    assert (edit.index, edit.removed, edit.inserted) == (1, 2, 3)


# ----------------------------------------------------------------------
# A plural aim declines rather than doing less than asked


def test_a_cursor_selecting_overlapping_matches_declines():
    @fp.fpy
    def three(x):
        if x > 0:
            y = x + 1
            z = x + 1
            w = x + 1
        return x

    with pytest.raises(TransformDeclined, match='overlap'):
        widen.apply(three, StmtCursor(three.ast, FuncBody().stmt(0)))


def test_an_aim_at_one_of_two_overlapping_matches_is_fine():
    """Only the selected match matters, so naming one of an overlapping pair
    does not decline."""
    @fp.fpy
    def three(x):
        y = x + 1
        z = x + 1
        w = x + 1
        return y + z + w

    found = find_all(pair_l, three)
    assert len(found) == 2
    out = widen.apply(three, found[0])
    assert len(out.edits.edits) == 1


# ----------------------------------------------------------------------
# `repeat`


def test_repeat_expands_an_expression_rule():
    @fp.pattern
    def dbl_l(a):
        a * 2

    @fp.pattern
    def dbl_r(a):
        (a + a) * 2

    @fp.fpy
    def g(x):
        return x * 2

    rule = Rewrite(dbl_l, dbl_r)
    counts = [rule.apply(g, repeat=r).format().count('+') for r in (1, 2, 3)]
    assert counts == [1, 3, 7]


def test_repeat_leaves_the_rule_it_expanded_alone():
    """The replacement may be a module-level `@fp.pattern` the caller still
    holds, so expansion must not rewrite it in place."""
    @fp.pattern
    def sl(a):
        y = a * 2

    @fp.pattern
    def sr(a):
        y = (a + a) * 2

    @fp.fpy
    def g(x):
        y = x * 2
        return y

    before = sr.to_ast().format()
    Rewrite(sl, sr).apply(g, repeat=3)
    assert sr.to_ast().format() == before


# ----------------------------------------------------------------------
# Node sharing


def test_a_cursor_rewrites_one_of_two_identical_siblings():
    """A replacement using a variable twice must not put one node in two
    places, or a single cursor would rewrite both."""
    @fp.pattern
    def dup_l(a):
        a * 2

    @fp.pattern
    def dup_r(a):
        a + a

    @fp.pattern
    def sub_l(a, b):
        a - b

    @fp.pattern
    def sub_r(a, b):
        fp.fma(a, 1, -b)

    @fp.fpy
    def f(x, y):
        return (x - y) * 2

    out = Rewrite(dup_l, dup_r).apply(f)
    found = find_all(sub_l, out)
    assert len(found) == 2
    once = Rewrite(sub_l, sub_r).apply(out, found[0])
    assert once.format().count('fp.fma') == 1
