"""
Cursors across the rounding strategies: what a rewrite reports, and how a
cursor reaches the program a later pass produced.

The point being pinned is composition — a location chosen against one program
still names the same program point after the rewrites around it, which is what
`where=<index>` cannot do.
"""

import pytest

import fpy2 as fp

from fpy2.strategies import (
    BlockCursor,
    StmtCursor,
    FuncBody,
    TransformReferenceError,
    float_to_fixed,
    rescale_fixed,
    simplify,
    unfold_overflow,
    unfold_special,
)


@fp.fpy(ctx=fp.REAL)
def two_sites(x: fp.Real, y: fp.Real) -> fp.Real:
    a = x
    with fp.FP16:
        p = fp.round(x)
    with fp.FP16:
        q = fp.round(y)
    return p + q + a


@fp.fpy(ctx=fp.REAL)
def two_rounds(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP16:
        p = fp.round(x)
        q = fp.round(y)
    z = p + q
    return z


@fp.fpy(ctx=fp.REAL)
def exact(x: fp.Real) -> fp.Real:
    with fp.REAL:
        y = fp.round(x)
    return y


@fp.fpy(ctx=fp.REAL)
def one_site(x: fp.Real) -> fp.Real:
    with fp.FP16:
        y = fp.round(x)
    return y


def _first(func) -> StmtCursor:
    """The first rounding block of `two_sites`-shaped programs."""
    return StmtCursor(func.ast, FuncBody().stmt(1))


def _text(site: StmtCursor | BlockCursor) -> str:
    """The source of what a cursor or region names."""
    if isinstance(site, BlockCursor):
        return '\n'.join(s.format() for s in site.resolve())
    return site.resolve().format()


# ----------------------------------------------------------------------
# What a rewrite reports


def test_a_rewrite_reports_the_site_it_replaced():
    out = unfold_special(two_sites, where=0)
    assert out.edits is not None
    assert out.edits.source is two_sites.ast
    assert out.edits.result is out.ast
    edit, = out.edits.edits
    assert (edit.block_path, edit.index, edit.removed) == (FuncBody(), 1, 1)


def test_apply_everywhere_reports_every_site():
    out = unfold_special(two_sites)
    assert out.edits is not None
    assert [e.index for e in out.edits.edits] == [1, 2]


def test_a_declined_site_is_not_an_edit():
    """`REAL` rounds exactly, so the candidate is skipped and nothing moves."""
    out = unfold_special(exact)
    assert out.edits is not None and out.edits.edits == ()
    assert out.forward(StmtCursor(exact.ast, FuncBody().stmt(1))) == StmtCursor(out.ast, FuncBody().stmt(1))


# ----------------------------------------------------------------------
# Forwarding


def test_untouched_statements_survive_a_rewrite_that_grew_the_block():
    """Two rounds become two statements, so what followed them shifts."""
    after = StmtCursor(two_rounds.ast, FuncBody().stmt(1))
    out = unfold_special(two_rounds, where=0)
    assert out.edits is not None and out.edits.edits[0].inserted == 2

    moved = out.forward(after)
    assert isinstance(moved, StmtCursor)
    assert moved.index == 2
    assert moved.resolve().format() == after.resolve().format()


def test_a_rewritten_statement_forwards_to_its_replacement():
    out = unfold_special(two_sites, where=0)
    image = out.forward(_first(two_sites))
    # one rounding in the block, so one statement replaced it
    assert isinstance(image, StmtCursor)
    assert 'round(' in _text(image)


def test_forwarding_composes_across_two_passes():
    """A cursor chosen against the *original* program reaches the program two
    rewrites later — the thing a `where` index cannot do."""
    untouched = StmtCursor(two_sites.ast, FuncBody().stmt(2))
    f1 = unfold_special(two_sites, where=0)
    f2 = unfold_overflow(f1, where=0, early_check=True)

    image = f2.forward(_first(two_sites))
    assert image.func is f2.ast
    assert 'round(' in _text(image)

    # and the site neither pass was aimed at is still the block it was
    assert _text(f2.forward(untouched)) == _text(untouched)


def test_a_region_forwards_as_a_region():
    """A block of two rounds lowers to two statements, so the image of its
    site is a region — which forwards on through the next pass as one."""
    site = StmtCursor(two_rounds.ast, FuncBody().stmt(0))
    f1 = float_to_fixed(two_rounds, where=0)

    region = f1.forward(site)
    assert isinstance(region, BlockCursor)
    assert len(region) == 2

    f2 = rescale_fixed(f1)
    again = f2.forward(site)
    assert isinstance(again, BlockCursor)
    assert again.func is f2.ast and len(again) == 2


def test_a_cursor_inside_a_rewritten_statement_does_not_forward():
    inside = StmtCursor(two_sites.ast, FuncBody().stmt(1).block('body').stmt(0))
    out = unfold_special(two_sites, where=0)
    with pytest.raises(TransformReferenceError, match='which was rewritten'):
        out.forward(inside)


def test_a_cursor_of_an_unrelated_program_does_not_forward():
    out = unfold_special(two_sites, where=0)
    with pytest.raises(TransformReferenceError, match='unrelated program'):
        out.forward(StmtCursor(one_site.ast, FuncBody().stmt(0)))


def test_an_opaque_pass_stops_the_walk():
    """`simplify` does not report what it rewrote, so a cursor cannot cross
    it — that is the answer, not a guess that the statement survived."""
    site = _first(two_sites)
    out = simplify(unfold_special(two_sites, where=0))
    with pytest.raises(TransformReferenceError, match='does not report'):
        out.forward(site)


def test_forwarding_to_the_same_program_is_the_identity():
    site = _first(two_sites)
    assert two_sites.forward(site) == site


# ----------------------------------------------------------------------
# The chain itself


def test_the_chain_records_each_step():
    f1 = unfold_special(two_sites, where=0)
    f2 = unfold_overflow(f1, where=0, early_check=True)
    assert f2.parent is f1
    assert f1.parent is two_sites
    assert two_sites.parent is None


def test_with_edits_rejects_a_log_from_another_program():
    from fpy2.transform import UnfoldSpecial
    log = UnfoldSpecial.apply_with_edits(two_sites.ast, where=0)
    with pytest.raises(ValueError, match='not produced from this program'):
        one_site.with_edits(log)


def test_a_runtime_does_not_break_the_chain():
    """`with_rt` is the same program, so it keeps its place in the chain."""
    from fpy2.interpret import get_default_interpreter

    f1 = unfold_special(two_sites, where=0)
    same = f1.with_rt(get_default_interpreter())
    assert same.parent is f1.parent and same.edits is f1.edits
    assert same.forward(_first(two_sites)) == f1.forward(_first(two_sites))
