"""
Listing the sites a strategy can be aimed at.

The claim that matters is the correspondence: `sites(strategy, f)[i]` names the
same site as `where=i`, so a cursor can be chosen rather than counted out by hand.
"""

import pytest

import fpy2 as fp

from fpy2.ast import Call
from fpy2.strategies import (
    BlockCursor,
    ExprCursor,
    FuncBody,
    StmtCursor,
    TransformDeclined,
    TransformReferenceError,
    float_to_fixed,
    inline,
    insert_round,
    rescale_fixed,
    simplify,
    sites,
    split,
    unfold_neg_zero,
    unfold_overflow,
    unfold_special,
    unroll_for,
    unroll_while,
)


@fp.fpy(ctx=fp.REAL)
def two_sites(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP16:
        p = fp.round(x)
    with fp.FP16:
        q = fp.round(y)
    return p + q


@fp.fpy(ctx=fp.REAL)
def nested(x: fp.Real) -> fp.Real:
    if x > 0:
        with fp.FP16:
            y = fp.round(x)
    else:
        y = -x
    return y


@fp.fpy(ctx=fp.REAL)
def cast_and_round(a: fp.Real) -> fp.Real:
    with fp.FixedContext(True, -4, 16):
        aq = fp.cast(a)
    with fp.FixedContext(True, -8, 16):
        bq = fp.round(a)
    return aq + bq


@fp.fpy(ctx=fp.REAL)
def cast_and_round_fp16(a: fp.Real) -> fp.Real:
    with fp.FP16:
        aq = fp.cast(a)
    with fp.FP16:
        bq = fp.round(a)
    return aq + bq


@fp.fpy(ctx=fp.REAL)
def declining(x: fp.Real) -> fp.Real:
    with fp.REAL:      # a candidate that `unfold_special` refuses
        y = fp.round(x)
    return y


@fp.fpy
def loops(xs: list[fp.Real], n: fp.Real) -> fp.Real:
    a = 0.0
    for x in xs:
        a = a + x
    i = 0.0
    while i < n:
        i = i + 1
    return a + i


@fp.fpy
def sq(x: fp.Real) -> fp.Real:
    return x * x


@fp.fpy
def cube(x: fp.Real) -> fp.Real:
    return x * x * x


@fp.fpy
def calls(x: fp.Real, y: fp.Real) -> fp.Real:
    return sq(x) + cube(y)


# ----------------------------------------------------------------------
# What a listing names


def test_the_rounding_strategies_list_their_blocks():
    """The three that apply to a float format list both of its rounds."""
    for strategy in (unfold_special, unfold_overflow, float_to_fixed):
        found = sites(strategy, two_sites)
        assert [c.path for c in found] == [FuncBody().stmt(0), FuncBody().stmt(1)]
        assert all(isinstance(c, StmtCursor) for c in found)


def test_a_strategy_that_applies_to_nothing_lists_nothing():
    """`two_sites` rounds to `FP16`, which has no negative-zero rule to shed and
    nothing fixed-point to rescale, so neither strategy has a site in it."""
    for strategy in (unfold_neg_zero, rescale_fixed):
        assert sites(strategy, two_sites) == []


def test_a_listing_is_outermost_first():
    found = sites(unfold_special, nested)
    assert [c.path for c in found] == [FuncBody().stmt(0).block('ift').stmt(0)]


def test_the_loop_strategies_list_their_loops():
    assert [c.path for c in sites(split, loops)] == [FuncBody().stmt(1)]
    assert [c.path for c in sites(unroll_for, loops)] == [FuncBody().stmt(1)]
    assert [c.path for c in sites(unroll_while, loops)] == [FuncBody().stmt(3)]


def _callee(cursor: ExprCursor) -> str:
    call = cursor.resolve()
    assert isinstance(call, Call)
    return call.fn.name


def test_insert_round_lists_operations():
    """Its candidates are operations whose scope rounds exactly, so the two
    `fp.round` blocks here are not sites at all -- only the final add, which is
    under the function's own `fp.REAL` scope."""
    found = sites(insert_round, two_sites)
    assert all(isinstance(c, ExprCursor) for c in found)
    assert [type(c.resolve()).__name__ for c in found] == ['Add']


def test_inline_lists_expressions():
    found = sites(inline, calls)
    assert all(isinstance(c, ExprCursor) for c in found)
    assert [_callee(c) for c in found] == ['sq', 'cube']


def test_inline_honours_the_funcs_filter():
    only = sites(inline, calls, funcs=[cube])
    assert [_callee(c) for c in only] == ['cube']


def test_a_listing_is_semantic():
    """A candidate the strategy refuses is not a site: it neither appears in a
    listing nor consumes an index.  A cursor naming it still says why."""
    assert sites(unfold_special, declining) == []
    at_block = StmtCursor(declining.ast, FuncBody().stmt(0))
    with pytest.raises(TransformDeclined, match='rounds exactly'):
        unfold_special(declining, where=at_block)


def test_a_strategy_that_takes_no_where_has_no_sites():
    with pytest.raises(ValueError, match='takes no `where`'):
        sites(simplify, two_sites)


# ----------------------------------------------------------------------
# The correspondence with `where=<index>`


def _aims_alike(strategy, func, i, cursor):
    """Whether `where=i` and `where=cursor` do the same thing, a refusal
    counting as an outcome."""
    try:
        expect = strategy(func, where=i).format()
    except TransformDeclined:
        with pytest.raises(TransformDeclined):
            strategy(func, where=cursor)
        return
    assert strategy(func, where=cursor).format() == expect


@pytest.mark.parametrize('strategy', [
    unfold_special, unfold_neg_zero, unfold_overflow, float_to_fixed, rescale_fixed,
])
def test_a_listed_site_aims_the_same_as_its_index(strategy):
    for i, cursor in enumerate(sites(strategy, two_sites)):
        _aims_alike(strategy, two_sites, i, cursor)


@pytest.mark.parametrize('strategy,func', [
    (unfold_special, cast_and_round_fp16),
    (rescale_fixed, cast_and_round),
])
def test_a_cast_block_is_listed_where_it_counts(strategy, func):
    """A `cast` block is a candidate for these two, so it has to appear in the
    listing at the index `where` gives it."""
    listed = sites(strategy, func)
    assert [c.index for c in listed] == [0, 1]
    for i, cursor in enumerate(listed):
        _aims_alike(strategy, func, i, cursor)


def test_a_cast_block_is_not_listed_where_it_does_not_count():
    """...and must not, for the two that only take a round.  `unfold_neg_zero`
    is absent: it refuses a float format outright, so there is no program where
    it both verifies and sees a cast."""
    for strategy in (unfold_overflow, float_to_fixed):
        assert [c.index for c in sites(strategy, cast_and_round_fp16)] == [1]


def test_a_listed_insert_round_site_aims_the_same_as_its_index():
    """`insert_round` takes a `ctx` as well, so it binds one rather than
    joining the parametrization above."""
    def aim(func, where):
        return insert_round(func, fp.FP64, where=where)

    for i, cursor in enumerate(sites(insert_round, two_sites)):
        _aims_alike(aim, two_sites, i, cursor)


def test_a_listed_call_aims_the_same_as_its_index():
    for i, cursor in enumerate(sites(inline, calls)):
        assert inline(calls, cursor).format() == inline(calls, i).format()


# ----------------------------------------------------------------------
# `within`


def test_within_narrows_to_a_region():
    part = BlockCursor(two_sites.ast, FuncBody(), range(1, 2))
    assert [c.path for c in sites(unfold_special, two_sites, part)] == [
        FuncBody().stmt(1)
    ]


def test_within_narrows_to_what_a_cursor_holds():
    """The `if` is one statement, and the rounding is beneath it."""
    branch = StmtCursor(nested.ast, FuncBody().stmt(0))
    assert len(sites(unfold_special, nested, branch)) == 1
    assert sites(unfold_special, nested, StmtCursor(nested.ast, FuncBody().stmt(1))) == []


def test_within_asks_a_forwarded_site_what_it_now_holds():
    """The step a schedule takes: rewrite at a site, then look inside its image."""
    site = sites(unfold_special, two_sites)[0]
    out = unfold_special(two_sites, where=site)

    inner = sites(unfold_overflow, out, out.rebase(site))
    assert len(inner) == 1
    # ... and it is inside the wrapper the rewrite left behind
    assert inner[0].path != site.path


def test_within_of_another_program_is_a_bad_reference():
    other = StmtCursor(nested.ast, FuncBody().stmt(0))
    with pytest.raises(TransformReferenceError, match='unrelated program'):
        sites(unfold_special, two_sites, other)


def test_within_is_forwarded_like_a_where():
    """A site listed against one program narrows a listing against a later
    one, without the caller forwarding it by hand."""
    site = sites(unfold_special, two_sites)[0]
    out = unfold_special(two_sites, where=site)
    assert len(sites(unfold_overflow, out, site)) == 1


def test_an_expression_cannot_narrow_a_statement_listing():
    cur = ExprCursor(calls.ast, FuncBody().stmt(0).expr('expr'))
    with pytest.raises(TransformReferenceError, match='these sites are statements'):
        sites(unfold_special, calls, cur)


def test_an_expression_narrows_a_call_listing():
    """Both calls are under the returned sum; only one is under each operand."""
    whole = ExprCursor(calls.ast, FuncBody().stmt(0).expr('expr'))
    assert len(sites(inline, calls, whole)) == 2

    left = ExprCursor(calls.ast, FuncBody().stmt(0).expr('expr').expr('args', 0))
    assert [c.path for c in sites(inline, calls, left)] == [left.path]
