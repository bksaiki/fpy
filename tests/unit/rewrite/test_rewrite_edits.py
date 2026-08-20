"""
What a user rewrite reports, and what a cursor makes of it.

A rewrite is a pass like any other: it says which statements it replaced, so
a cursor crosses it, and a schedule can put a user rule between two built-ins.
"""

import pytest

import fpy2 as fp

from fpy2.rewrite import Rewrite, find, find_all
from fpy2.strategies import (
    BlockCursor,
    ExprCursor,
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
    # three statements from two, rebinding `z` rather than introducing a name:
    # a rule's replacement may only use variables its pattern bound
    y = a + 1
    z = b + 1
    z = z + y


fma = Rewrite(fma_l, fma_r)
bump = Rewrite(bump_l, bump_r)
widen = Rewrite(pair_l, pair_r)


@fp.fpy
def expr_prog(x, y, z):
    a = x * y + z
    b = z + 1
    return a + b


@fp.fpy
def stmt_prog(x):
    y = x + 1
    z = y * 3
    return y + z


# ----------------------------------------------------------------------
# An expression rule changes a statement without replacing it


def test_an_expression_rule_records_no_edit():
    out = fma.apply(expr_prog)
    assert out.edits is not None
    assert out.edits.edits == ()


def test_a_statement_cursor_crosses_an_expression_rule():
    """The statement is still there, so the cursor still names it."""
    later = StmtCursor(expr_prog.ast, FuncBody().stmt(1))
    out = fma.apply(expr_prog)

    moved = out.forward(later)
    assert moved == StmtCursor(out.ast, FuncBody().stmt(1))
    assert moved.resolve().format() == later.resolve().format()


def test_an_expression_cursor_in_the_rewritten_statement_does_not_cross():
    site = find(fma_l, expr_prog)
    assert isinstance(site, ExprCursor)
    out = fma.apply(expr_prog)
    with pytest.raises(TransformReferenceError, match='whose expressions'):
        out.forward(site)


# ----------------------------------------------------------------------
# A statement rule replaces a window


def test_a_statement_rule_records_the_window_it_replaced():
    out = bump.apply(stmt_prog)
    assert out.edits is not None
    edit, = out.edits.edits
    assert (edit.block_path, edit.index, edit.removed, edit.inserted) == (
        FuncBody(), 0, 1, 1
    )


def test_a_rule_that_grows_the_block_shifts_what_follows():
    """`pair_l -> pair_r` replaces two statements with three."""
    @fp.fpy
    def two_bumps(x):
        y = x + 1
        z = x + 1
        return y + z

    out = widen.apply(two_bumps)
    assert out.edits is not None
    edit, = out.edits.edits
    assert (edit.removed, edit.inserted) == (2, 3)

    after = StmtCursor(two_bumps.ast, FuncBody().stmt(2))
    assert out.forward(after) == StmtCursor(out.ast, FuncBody().stmt(3))


def test_a_region_of_the_replaced_window_forwards_to_its_replacement():
    """The `removed > 1` path: a two-statement match is a region, and its image
    is the three statements that replaced it."""
    @fp.fpy
    def two_bumps(x):
        y = x + 1
        z = x + 1
        return y + z

    site = find(pair_l, two_bumps)
    assert site == BlockCursor(two_bumps.ast, FuncBody(), range(0, 2))

    out = widen.apply(two_bumps)
    image = out.forward(site)
    assert isinstance(image, BlockCursor)
    assert image.span == range(0, 3)


def test_a_cursor_inside_a_replaced_window_does_not_forward():
    site = StmtCursor(stmt_prog.ast, FuncBody().stmt(0))
    out = bump.apply(stmt_prog)
    # the statement was replaced one-for-one, so the cursor lands on its image
    assert isinstance(out.forward(site), StmtCursor)
    assert 'x + 2' in out.forward(site).resolve().format()


# ----------------------------------------------------------------------
# A rule in the middle of a schedule


def test_a_cursor_crosses_a_user_rule_between_two_built_ins():
    from fpy2.strategies import unfold_special

    @fp.fpy(ctx=fp.REAL)
    def prog(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
        a = x * y + z
        with fp.FP16:
            r = fp.round(a)
        return r

    site = StmtCursor(prog.ast, FuncBody().stmt(1))
    out = fma.apply(prog)                       # a user rule, in the middle
    out = unfold_special(out, where=site)       # aimed with the original cursor

    assert out.edits is not None and len(out.edits.edits) == 1
    assert 'fp.fma' in out.format()


# ----------------------------------------------------------------------
# Overlap


def test_overlapping_matches_decline_the_whole_application():
    @fp.fpy
    def three(x):
        y = x + 1
        z = x + 1
        w = x + 1
        return y + z + w

    assert len(find_all(pair_l, three)) == 2      # windows [0:2] and [1:3]
    with pytest.raises(TransformDeclined, match='overlap'):
        widen.apply(three)


def test_a_named_match_is_unaffected_by_other_overlaps():
    """Only one match is rewritten, so nothing conflicts."""
    @fp.fpy
    def three(x):
        y = x + 1
        z = x + 1
        w = x + 1
        return y + z + w

    out = widen.apply(three, 0)
    assert out.edits is not None and len(out.edits.edits) == 1


# ----------------------------------------------------------------------
# Aiming


def test_a_cursor_names_the_match_to_rewrite():
    @fp.fpy
    def two(x):
        y = x + 1
        z = y * 3
        y = x + 1
        return y + z

    second = find_all(bump_l, two)[1]
    out = bump.apply(two, second)

    assert out.edits is not None
    edit, = out.edits.edits
    assert edit.index == 2


def test_an_index_and_the_cursor_it_lists_aim_alike():
    @fp.fpy
    def two(x):
        y = x + 1
        z = y * 3
        y = x + 1
        return y + z

    for i, cursor in enumerate(find_all(bump_l, two)):
        assert bump.apply(two, cursor).format() == bump.apply(two, i).format()


def test_a_cursor_naming_no_match_is_a_bad_reference():
    site = StmtCursor(stmt_prog.ast, FuncBody().stmt(1))   # `z = y * 3`
    with pytest.raises(TransformReferenceError, match='does not correspond'):
        bump.apply(stmt_prog, site)


def test_an_expression_cursor_names_one_match_of_an_expression_rule():
    @fp.fpy
    def two(x, y, z):
        a = x * y + z
        b = z * y + x
        return a + b

    second = find_all(fma_l, two)[1]
    out = fma.apply(two, second)

    assert out.format().count('fp.fma') == 1
    assert 'z * y + x' not in out.format()


def test_a_stale_cursor_is_forwarded_on_arrival():
    """A cursor chosen against the original program aims a rewrite of a later
    one, without the caller rebasing it."""
    @fp.fpy
    def two(x):
        y = x + 1
        z = y * 3
        y = x + 1
        return y + z

    site = find_all(bump_l, two)[1]
    once = bump.apply(two, 0)          # rewrite the first match
    twice_ = bump.apply(once, site)    # the second, named against the original

    assert twice_.format().count('x + 2') == 2


def test_where_defaults_to_every_match():
    @fp.fpy
    def two(x, y, z):
        a = x * y + z
        b = z * y + x
        return a + b

    assert fma.apply(two).format().count('fp.fma') == 2


def test_an_unselected_match_keeps_its_statements():
    """A window that matches but is not the chosen match keeps its statements."""
    @fp.fpy
    def two(x):
        y = x + 1
        z = y * 3
        y = x + 1
        return y + z

    out = bump.apply(two, 1)
    assert out.format().count('x + 1') == 1
    assert out.format().count('x + 2') == 1
    assert len(out.ast.body.stmts) == len(two.ast.body.stmts)
