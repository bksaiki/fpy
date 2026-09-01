"""The contract every aimable strategy owes `where`.

If a strategy has `k` sites in a program, then `where=None` rewrites all `k` and
`where=j` for ``0 <= j < k`` rewrites the `j`th.  `k` is `len(sites(...))`.

Six of the ten strategies used to break this: `sites` reported *structural*
candidates while a refusal consumed an index, so `where=j` could raise and
`where=None` could be a no-op with `k > 0`.  The `refuses` rows are what catch
that, and every strategy in `_SITES` has to appear here at all --
`test_every_aimable_strategy_is_covered` fails if one is added without a row.
"""

import pytest

import fpy2 as fp
from fpy2.analysis.format_infer import derive_intermediate
from fpy2.ast.fpyast import Integer
from fpy2.strategies import (
    ExprCursor,
    TransformReferenceError,
    comp_to_loop,
    float_to_fixed,
    inline,
    insert_round,
    monomorphize,
    refusals,
    rescale_fixed,
    sites,
    split,
    split_round,
    unfold_neg_zero,
    unfold_overflow,
    unfold_special,
    unroll_for,
    unroll_while,
)
from fpy2.strategies.sites import _SITES
from fpy2.transform import ForUnrollStrategy, SplitLoopStrategy, contains
from fpy2.types import RealType

# ----------------------------------------------------------------------
# Programs, chosen so each strategy has at least one site in its own


@fp.fpy(ctx=fp.REAL)
def _two_floats(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP16:
        p = fp.round(x)
    with fp.FP16:
        q = fp.round(y)
    return p + q


@fp.fpy(ctx=fp.REAL)
def _rounds_exactly(x: fp.Real) -> fp.Real:
    with fp.REAL:
        y = fp.round(x)
    return y


@fp.fpy(ctx=fp.REAL)
def _refuses_then_acts(x: fp.Real, y: fp.Real) -> fp.Real:
    """A refused candidate before a site, so index 0 is `body[1]`."""
    with fp.REAL:
        p = fp.round(x)
    with fp.FP16:
        q = fp.round(y)
    return p + q


@fp.fpy(ctx=fp.REAL)
def _two_fixed(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.MPFixedContext(-8):
        p = fp.round(x)
    with fp.MPFixedContext(-4):
        q = fp.round(y)
    return p + q


@fp.fpy(ctx=fp.REAL)
def _two_scaled(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FixedContext(True, -16, 32):
        p = fp.round(x)
    with fp.FixedContext(True, -8, 32):
        q = fp.round(y)
    return p + q


@fp.fpy
def _two_for(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    a = 0.0
    for x in xs:
        a = a + x
    for y in ys:
        a = a + y
    return a


@fp.fpy
def _nested_for(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    a = 0.0
    for x in xs:
        for y in ys:
            a = a + x * y
    return a


@fp.fpy
def _two_while(n: fp.Real, m: fp.Real) -> fp.Real:
    i = 0.0
    while i < n:
        i = i + 1
    j = 0.0
    while j < m:
        j = j + 1
    return i + j


@fp.fpy
def _odd_trip() -> fp.Real:
    """Three iterations, so a STRICT split or unroll by 2 is refused."""
    a = 0.0
    for _i in range(3):
        a = a + 1
    return a


@fp.fpy
def _nested_while(n: fp.Real, m: fp.Real) -> fp.Real:
    i = 0.0
    while i < n:
        j = 0.0
        while j < m:
            j = j + 1
        i = i + 1
    return i


@fp.fpy
def _leaf(x: fp.Real) -> fp.Real:
    return x * x


@fp.fpy
def _mid(x: fp.Real) -> fp.Real:
    return _leaf(x) + 1.0


@fp.fpy
def _two_calls(x: fp.Real, y: fp.Real) -> fp.Real:
    return _leaf(x) + _leaf(y)


@fp.fpy
def _refuses_inline(n: fp.Real) -> fp.Real:
    """The call sits in a `while` condition, where splicing the body ahead of
    the loop would evaluate it once instead of every iteration."""
    i = 0.0
    while _leaf(i) < n:
        i = i + 1
    return i


@fp.fpy
def _nested_calls(x: fp.Real) -> fp.Real:
    return _mid(x) * _mid(x)


@fp.fpy(ctx=fp.FP64)
def _sum_of_squares(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.REAL:
        t = (x * x) + (y * y)
    return t


@fp.fpy(ctx=fp.FP32)
def _two_rounded(x: fp.Real, y: fp.Real) -> fp.Real:
    """Two operations that already round, so `split_round` has two sites."""
    t = x * y
    s = x + y
    return t + s


@fp.fpy(ctx=fp.FP64)
def _comprehension(xs: list[fp.Real]) -> list[fp.Real]:
    return [x * x for x in xs]


@fp.fpy(ctx=fp.FP64)
def _dependent_comprehension(xss: list[list[fp.Real]]) -> list[fp.Real]:
    """A later iterable mentions an earlier target, so the length is a sum and
    `fp.empty` has nowhere to get it."""
    return [b for a in xss for b in a]


@fp.fpy(ctx=fp.FP64)
def _nested_ops(x: fp.Real) -> fp.Real:
    with fp.REAL:
        t = abs(x * x)
    return t


def _pin(func, n: int):
    return monomorphize(func, fp.FP64, [RealType(fp.FP32)] * n)


# ----------------------------------------------------------------------
# The table: (strategy, program, kwargs for `sites`/`refusals`)
#
# `nested` rows carry candidates that contain other candidates, where a rewrite
# of the outer subsumes the inner in the edit log but not in the rewrite.

_FP64 = {'ctx': fp.FP64}
_VIA32 = {'ctx': derive_intermediate(fp.FP32)}
_STRICT_SPLIT = {'factor': Integer(2, None), 'strategy': SplitLoopStrategy.STRICT}
_STRICT_UNROLL = {'times': 1, 'strategy': ForUnrollStrategy.STRICT}

# Rows where the strategy has sites.
ACTS = [
    ('unfold_special', unfold_special, _two_floats, {}),
    ('unfold_special/mixed', unfold_special, _refuses_then_acts, {}),
    ('unfold_neg_zero', unfold_neg_zero, _two_fixed, {}),
    ('unfold_overflow', unfold_overflow, _two_floats, {}),
    ('float_to_fixed', float_to_fixed, _two_floats, {}),
    ('rescale_fixed', rescale_fixed, _two_scaled, {}),
    ('split', split, _two_for, {}),
    ('split/nested', split, _nested_for, {}),
    ('unroll_for', unroll_for, _two_for, {}),
    ('unroll_for/nested', unroll_for, _nested_for, {}),
    ('unroll_while', unroll_while, _two_while, {}),
    ('unroll_while/nested', unroll_while, _nested_while, {}),
    ('inline', inline, _two_calls, {}),
    ('inline/nested', inline, _nested_calls, {}),
    ('insert_round', insert_round, _pin(_sum_of_squares, 2), _FP64),
    ('insert_round/nested', insert_round, _pin(_nested_ops, 1), _FP64),
    ('comp_to_loop', comp_to_loop, _comprehension, {}),
    ('split_round', split_round, _two_rounded, _VIA32),
]

# Rows where it has none: a program it refuses outright.  These are where the
# divergence lived -- `sites` reported structural candidates and `where=None`
# then did nothing -- so a table of only the rows above does not catch it.
REFUSES = [
    ('unfold_special/refuses', unfold_special, _rounds_exactly, {}),
    ('unfold_neg_zero/refuses', unfold_neg_zero, _two_floats, {}),
    ('unfold_overflow/refuses', unfold_overflow, _two_fixed, {}),
    ('float_to_fixed/refuses', float_to_fixed, _two_fixed, {}),
    ('rescale_fixed/refuses', rescale_fixed, _two_floats, {}),
    ('inline/refuses', inline, _refuses_inline, {}),
    ('split/refuses', split, _odd_trip, _STRICT_SPLIT),
    ('unroll_for/refuses', unroll_for, _odd_trip, _STRICT_UNROLL),
    ('insert_round/refuses', insert_round, _pin(_sum_of_squares, 2), {'ctx': fp.FP16}),
    ('comp_to_loop/refuses', comp_to_loop, _dependent_comprehension,
     {'dependent': False}),
    # an intermediate no wider than the target: no rule admits it, generic or
    # operation-specific
    ('split_round/refuses', split_round, _two_rounded, {'ctx': fp.FP32}),
]

# `unroll_while` has no row: it refuses nothing at all.

CASES = ACTS + REFUSES

_rows = lambda cs: [(strategy, func, kw) for _, strategy, func, kw in cs]
ACT_IDS = [n for n, *_ in ACTS]
ACT_ROWS = _rows(ACTS)
REFUSE_IDS = [n for n, *_ in REFUSES]
REFUSE_ROWS = _rows(REFUSES)
IDS = [n for n, *_ in CASES]
ROWS = _rows(CASES)


def _apply(strategy, func, where, kw):
    """Run *strategy* aimed at *where*, passing whatever else it requires."""
    if strategy is insert_round or strategy is split_round:
        return strategy(func, kw['ctx'], where=where)
    if strategy is inline:
        return strategy(func, where)
    if strategy is split:
        return strategy(func, 2, where=where, **_no_factor(kw))
    if strategy is unroll_for and 'strategy' in kw:
        return strategy(func, where, kw['times'], strategy=kw['strategy'])
    return strategy(func, where=where, **kw)


def _no_factor(kw):
    """*kw* as the `split` rewrite takes it: `factor` is positional there."""
    return {k: v for k, v in kw.items() if k != 'factor'}


# ----------------------------------------------------------------------
# The three properties


@pytest.mark.parametrize('strategy,func,kw', ACT_ROWS, ids=ACT_IDS)
def test_every_index_in_range_rewrites(strategy, func, kw):
    """`where=j` for `0 <= j < k` rewrites, and rewrites something.

    This is the half that broke when a refusal consumed an index: `where=j`
    raised `TransformDeclined` for a `j` the listing had just reported.
    """
    listed = sites(strategy, func, **kw)
    assert listed, 'the program should give this strategy at least one site'
    for j in range(len(listed)):
        out = _apply(strategy, func, j, kw)
        assert not out.ast.is_equiv(func.ast), f'where={j} changed nothing'


@pytest.mark.parametrize('strategy,func,kw', ROWS, ids=IDS)
def test_where_none_acts_exactly_when_there_are_sites(strategy, func, kw):
    """`where=None` rewrites all `k`, so it is a no-op if and only if `k` is 0.

    This is the half that broke worse: `sites(rescale_fixed, two_floats)`
    reported two sites in a program holding no fixed-point context at all, and
    `where=None` did nothing.
    """
    listed = sites(strategy, func, **kw)
    out = _apply(strategy, func, None, kw)
    assert out.ast.is_equiv(func.ast) == (not listed)

    # "all `k`", not "at least one": doing only some of them would pass the
    # check above, so `where=None` has to differ from every single aim
    for j in range(len(listed)) if len(listed) > 1 else ():
        assert not out.ast.is_equiv(_apply(strategy, func, j, kw).ast), (
            f'where=None did no more than where={j}'
        )


@pytest.mark.parametrize('strategy,func,kw', ROWS, ids=IDS)
def test_a_listed_cursor_aims_the_same_as_its_index(strategy, func, kw):
    """`sites(...)[j]` and `where=j` name the same site -- where that cursor
    names only that site.

    An index names one site.  A *statement* cursor names every site at or
    beneath it, so on a candidate containing another the two are deliberately
    not interchangeable and the cursor does strictly more.  An *expression*
    cursor names one exactly, so for those the two always agree.
    """
    listed = sites(strategy, func, **kw)
    for j, cursor in enumerate(listed):
        under = [
            c for i, c in enumerate(listed)
            if i != j and not isinstance(cursor, ExprCursor) and contains(cursor, c)
        ]
        by_index = _apply(strategy, func, j, kw)
        by_cursor = _apply(strategy, func, cursor, kw)
        if under:
            assert not by_cursor.ast.is_equiv(by_index.ast), (
                f'a cursor on site {j}, which contains {len(under)} more, '
                f'should take them too'
            )
        else:
            assert by_cursor.ast.is_equiv(by_index.ast)


# ----------------------------------------------------------------------
# What falls out of the contract


@pytest.mark.parametrize('strategy,func,kw', REFUSE_ROWS, ids=REFUSE_IDS)
def test_a_strategy_that_applies_to_nothing_lists_nothing(strategy, func, kw):
    """The divergence, stated directly: a program the strategy refuses has no
    sites, so `where=None` is a no-op and `where=0` is out of range.  It used to
    report a site per structural candidate and then rewrite none of them."""
    assert sites(strategy, func, **kw) == []
    assert refusals(strategy, func, **kw), 'it should say why, not stay silent'
    assert _apply(strategy, func, None, kw).ast.is_equiv(func.ast)
    with pytest.raises(TransformReferenceError):
        _apply(strategy, func, 0, kw)


@pytest.mark.parametrize('strategy,func,kw', ROWS, ids=IDS)
def test_an_index_past_the_end_is_an_error(strategy, func, kw):
    """Never a silent no-op: an index outside `range(k)` names nothing."""
    k = len(sites(strategy, func, **kw))
    with pytest.raises(TransformReferenceError):
        _apply(strategy, func, k, kw)


@pytest.mark.parametrize('strategy,func,kw', ROWS, ids=IDS)
def test_sites_and_refusals_are_disjoint(strategy, func, kw):
    """A program point is a site or a refusal, never both."""
    # `str` rather than `.path`: a region has a span instead, and no listing
    # reports one anyway
    listed = {str(c) for c in sites(strategy, func, **kw)}
    refused = {str(c) for c, _ in refusals(strategy, func, **kw)}
    assert listed.isdisjoint(refused)


def test_every_aimable_strategy_is_covered():
    """A new aimable strategy has to earn a row here, or this fails."""
    assert {strategy for _, strategy, _, _ in ACTS} == set(_SITES)
