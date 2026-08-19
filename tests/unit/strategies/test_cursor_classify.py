"""
What the passes that are not site-rewrites do to a cursor.

Each claim is per *kind*: a pass can leave the statement tree alone while
rewriting expressions in it, so one verdict per pass would make one of the two
answers a lie.
"""

import pytest

import fpy2 as fp

from fpy2.strategies import (
    ExprCursor,
    FuncBody,
    StmtCursor,
    TransformReferenceError,
    close,
    elim_iter,
    elim_round,
    fuse,
    lift_context,
    monomorphize,
    simplify,
    unfold_special,
)
from fpy2.types import RealType

SCALE = 2.0


@fp.fpy(ctx=fp.REAL)
def rounded(x: fp.Real) -> fp.Real:
    a = x * SCALE
    with fp.FP16:
        y = fp.round(a)
    return y


@fp.fpy(ctx=fp.REAL)
def in_a_loop(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x in xs:
        with fp.IEEEContext(8, 32):
            acc = fp.round(x)
    return acc


def _site(func) -> StmtCursor:
    """The rounding block of `rounded`."""
    return StmtCursor(func.ast, FuncBody().stmt(1))


# ----------------------------------------------------------------------
# Structure-preserving: the statement tree is untouched


def test_monomorphize_carries_every_kind_of_cursor():
    """It rewrites argument annotations and the function's own context; the body
    passes through untouched, so statements *and* expressions survive."""
    site = _site(rounded)
    expr = ExprCursor(rounded.ast, FuncBody().stmt(0).expr('expr'))
    out = monomorphize(rounded, args=[RealType(fp.FP32)])

    assert out.edits is not None and out.edits.edits == ()
    assert out.forward(site) == StmtCursor(out.ast, site.path)
    assert out.forward(expr) == ExprCursor(out.ast, expr.path)


def test_a_cursor_survives_monomorphize_into_the_rest_of_a_schedule():
    """The first step of the lowering recipe, so a site chosen against the
    original program still aims what follows."""
    site = _site(rounded)
    f = monomorphize(rounded, args=[RealType(fp.FP32)])
    f = unfold_special(f, where=site)

    assert f.edits is not None and len(f.edits.edits) == 1
    assert f.edits.edits[0].index == 1


# ----------------------------------------------------------------------
# Prepending: leading assignments, and everything below shifts


def test_close_shifts_by_its_prelude():
    site = _site(rounded)
    out = close(rounded)

    assert out.edits is not None
    edit, = out.edits.edits
    assert (edit.index, edit.removed, edit.inserted) == (0, 0, 1)   # `SCALE = 2`

    moved = out.forward(site)
    assert isinstance(moved, StmtCursor)
    assert moved.index == site.index + 1
    assert moved.resolve().format() == site.resolve().format()


def test_close_carries_expression_cursors_too():
    """The body's statements are reused verbatim, so expressions survive."""
    expr = ExprCursor(rounded.ast, FuncBody().stmt(0).expr('expr'))
    out = close(rounded)

    moved = out.forward(expr)
    assert isinstance(moved, ExprCursor)
    assert moved.resolve().format() == expr.resolve().format()


def test_close_with_nothing_to_bind_is_the_identity():
    site = StmtCursor(in_a_loop.ast, FuncBody().stmt(1))
    out = close(in_a_loop)
    assert out.edits is not None and out.edits.edits == ()
    assert out.forward(site) == StmtCursor(out.ast, site.path)


def test_lift_context_shifts_statements_but_not_expressions():
    """It hoists the context to a leading binding — a shift — and replaces the
    context expression in place, which no edit records."""
    site = StmtCursor(in_a_loop.ast, FuncBody().stmt(1))
    out = lift_context(in_a_loop)

    assert out.edits is not None
    edit, = out.edits.edits
    assert (edit.index, edit.removed, edit.inserted) == (0, 0, 1)
    assert out.forward(site).index == site.index + 1

    expr = ExprCursor(in_a_loop.ast, FuncBody().stmt(0).expr('expr'))
    with pytest.raises(TransformReferenceError, match='does not say what it did'):
        out.forward(expr)


# ----------------------------------------------------------------------
# Opaque: a cursor cannot cross


@pytest.mark.parametrize('strategy', [simplify, elim_round, fuse, elim_iter])
def test_an_opaque_pass_stops_a_cursor(strategy):
    """These rewrite at sites they do not report, so forwarding says so rather
    than guessing."""
    site = _site(rounded)
    out = strategy(rounded)

    assert out.edits is None
    with pytest.raises(TransformReferenceError, match='does not report'):
        out.forward(site)
