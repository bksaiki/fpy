"""
Aiming a rewrite with a cursor or a region.

The claim under test is the one the `where` index cannot make: a schedule pins
a program point *once* and every operator in the sequence hits that point, with
the rest of the program untouched.
"""

import pytest

import fpy2 as fp

from fpy2.strategies import (
    Block,
    Cursor,
    TransformDeclined,
    TransformReferenceError,
    float_to_fixed,
    rescale_fixed,
    unfold_overflow,
    unfold_special,
)


@fp.fpy(ctx=fp.REAL)
def two_sites(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP16:
        p = fp.round(x)
    with fp.FP16:
        q = fp.round(y)
    return p + q


@fp.fpy(ctx=fp.REAL)
def branched(x: fp.Real) -> fp.Real:
    # both arms round a bare variable, which is what a candidate block is
    if x > 0:
        with fp.FP16:
            y = fp.round(x)
    else:
        with fp.FP16:
            y = fp.round(x)
    return y


@fp.fpy(ctx=fp.REAL)
def exact(x: fp.Real) -> fp.Real:
    with fp.REAL:
        y = fp.round(x)
    return y


def _rounding_ctxs(func) -> list[str]:
    """The rounding contexts still written as `with fp.FP16:` blocks."""
    return [s.format() for s in func.ast.body.stmts if 'fp.FP16' in s.format()]


# ----------------------------------------------------------------------
# One cursor, the whole sequence


def test_a_pinned_cursor_aims_every_operator_in_the_sequence():
    site = Cursor(two_sites.ast, ('body', 0))

    f = unfold_special(two_sites, where=site)
    f = unfold_overflow(f, where=site, early_check=True)
    f = float_to_fixed(f, where=site)
    f = rescale_fixed(f, where=site)

    # the pinned block is gone, lowered to fixed-point rounding
    assert 'MPBFixedContext' in f.format()
    assert 'fp.logb' in f.format()
    # the other one is exactly as it was
    assert _rounding_ctxs(f) == _rounding_ctxs(two_sites)[1:]
    assert len(_rounding_ctxs(f)) == 1


def test_a_cursor_takes_candidates_beneath_it():
    """After the first rewrite the cursor names a wrapper, and the rounding it
    left behind sits inside — so the next operator still lands."""
    site = Cursor(two_sites.ast, ('body', 0))
    f1 = unfold_special(two_sites, where=site)

    wrapper = f1.rebase(site)
    assert isinstance(wrapper, Cursor)
    assert 'fp.FP16' not in wrapper.resolve().format().splitlines()[0]

    f2 = unfold_overflow(f1, where=site, early_check=True)
    assert f2.edits is not None and len(f2.edits.edits) == 1
    # the edit is *inside* the wrapper, not at the top level
    assert len(f2.edits.edits[0].block_path) > 1


def test_a_cursor_selects_one_of_two_sites():
    first = unfold_special(two_sites, where=Cursor(two_sites.ast, ('body', 0)))
    second = unfold_special(two_sites, where=Cursor(two_sites.ast, ('body', 1)))

    assert first.edits is not None and first.edits.edits[0].index == 0
    assert second.edits is not None and second.edits.edits[0].index == 1
    assert first.format() != second.format()


def test_a_cursor_reaches_a_site_in_a_branch():
    site = Cursor(branched.ast, ('body', 0, 'iff', 0))
    out = unfold_special(branched, where=site)

    edit, = out.edits.edits if out.edits else ()
    assert edit.block_path == ('body', 0, 'iff')
    # the other arm is untouched
    assert 'fp.FP16' in out.ast.body.stmts[0].format()


# ----------------------------------------------------------------------
# Regions


def test_a_region_takes_every_candidate_within_it():
    whole = Block(two_sites.ast, ('body',), range(0, 2))
    out = unfold_special(two_sites, where=whole)
    assert out.edits is not None
    assert [e.index for e in out.edits.edits] == [0, 1]


def test_a_region_scopes_the_walk():
    part = Block(two_sites.ast, ('body',), range(1, 2))
    out = unfold_special(two_sites, where=part)
    assert out.edits is not None
    assert [e.index for e in out.edits.edits] == [1]


def test_a_region_reaches_candidates_nested_in_it():
    """The `if` is one statement; both arms' roundings are beneath it."""
    whole = Block(branched.ast, ('body',), range(0, 1))
    out = unfold_special(branched, where=whole)
    assert out.edits is not None
    assert {e.block_path for e in out.edits.edits} == {
        ('body', 0, 'ift'), ('body', 0, 'iff'),
    }


# ----------------------------------------------------------------------
# Bad references and refusals


def test_a_cursor_naming_no_candidate_is_a_bad_reference():
    with pytest.raises(TransformReferenceError, match='does not name'):
        unfold_special(two_sites, where=Cursor(two_sites.ast, ('body', 2)))


def test_a_region_with_no_candidate_is_a_bad_reference():
    """An explicit `where` that matches nothing fails; only `None` may hit
    zero sites."""
    with pytest.raises(TransformReferenceError, match='does not name'):
        unfold_special(two_sites, where=Block(two_sites.ast, ('body',), range(2, 3)))


def test_a_cursor_of_an_unrelated_program_is_a_bad_reference():
    with pytest.raises(TransformReferenceError, match='unrelated program'):
        unfold_special(two_sites, where=Cursor(branched.ast, ('body', 0)))


def test_a_cursor_whose_every_candidate_declines_says_why():
    site = Cursor(exact.ast, ('body', 0))
    with pytest.raises(TransformDeclined, match='rounds exactly'):
        unfold_special(exact, where=site)


def test_where_rejects_something_that_is_not_a_site():
    with pytest.raises(TypeError, match='for where'):
        unfold_special(two_sites, where='body[0]')  # type: ignore[arg-type]


def test_the_transform_layer_takes_only_cursors_of_its_own_program():
    """The wrapper forwards; the transform underneath does not guess."""
    from fpy2.transform import UnfoldSpecial

    stale = Cursor(two_sites.ast, ('body', 0))
    once = UnfoldSpecial.apply(two_sites.ast, where=stale)
    with pytest.raises(TransformReferenceError, match='another program'):
        UnfoldSpecial.apply(once, where=stale)
