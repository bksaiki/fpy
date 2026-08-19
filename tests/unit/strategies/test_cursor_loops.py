"""
Cursors for the loop and call strategies.

Same contract as the rounding operators — a cursor or region names a program
point and the rewrite takes every site at or beneath it — over transforms whose
sites are loops and calls rather than rounding blocks.
"""

import pytest

import fpy2 as fp

from fpy2.strategies import (
    BlockCursor,
    FuncBody,
    StmtCursor,
    TransformReferenceError,
    inline,
    sites,
    split,
    unroll_for,
    unroll_while,
)


@fp.fpy
def two_loops(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    a = 0.0
    for x in xs:
        a = a + x
    b = 0.0
    for y in ys:
        b = b + y
    return a + b


@fp.fpy
def counted(n: fp.Real) -> fp.Real:
    i = 0.0
    while i < n:
        i = i + 1
    return i


@fp.fpy
def sq(x: fp.Real) -> fp.Real:
    return x * x


@fp.fpy
def two_calls(x: fp.Real, y: fp.Real) -> fp.Real:
    return sq(x) + sq(y)


@fp.fpy
def call_each(x: fp.Real, y: fp.Real) -> fp.Real:
    a = sq(x)
    b = sq(y)
    return a + b


def _loops(func) -> int:
    """How many `for` loops the program has, at any depth."""
    from fpy2.ast.visitor import DefaultVisitor

    n = [0]

    class _C(DefaultVisitor):
        def _visit_for(self, stmt, ctx):
            n[0] += 1
            super()._visit_for(stmt, ctx)

    _C()._visit_function(func.ast, None)
    return n[0]


# ----------------------------------------------------------------------
# for loops


def test_a_cursor_unrolls_one_loop():
    out = unroll_for(two_loops, StmtCursor(two_loops.ast, FuncBody().stmt(1)), times=1)
    assert out.edits is not None
    edit, = out.edits.edits
    assert (edit.block_path, edit.index) == (FuncBody(), 1)


def test_a_split_shifts_what_followed_it():
    """A split expands one loop into several statements, so the second loop
    moves — and its cursor moves with it."""
    second = StmtCursor(two_loops.ast, FuncBody().stmt(3))
    out = split(two_loops, 2, StmtCursor(two_loops.ast, FuncBody().stmt(1)))

    moved = out.forward(second)
    assert isinstance(moved, StmtCursor)
    assert moved.index > 3
    assert moved.resolve().format() == second.resolve().format()


def test_a_cursor_inside_an_unrolled_body_does_not_forward():
    """The body now exists in several copies, so no single statement is its
    image."""
    inside = StmtCursor(two_loops.ast, FuncBody().stmt(1).block('body').stmt(0))
    out = unroll_for(two_loops, StmtCursor(two_loops.ast, FuncBody().stmt(1)), times=1)
    with pytest.raises(TransformReferenceError, match='which was rewritten'):
        out.forward(inside)


def test_one_cursor_drives_a_loop_schedule():
    """Split the pinned loop, then unroll the loops the split produced — the
    region a split forwards to is what names them."""
    site = StmtCursor(two_loops.ast, FuncBody().stmt(1))
    f = split(two_loops, 4, site)
    assert _loops(f) > _loops(two_loops)

    f = unroll_for(f, site, times=1)
    # the other loop was never touched
    assert 'for y in ys' in f.format()


def test_a_region_takes_every_loop_within_it():
    whole = BlockCursor(two_loops.ast, FuncBody(), range(0, 4))
    out = unroll_for(two_loops, whole, times=1)
    assert out.edits is not None
    assert [e.index for e in out.edits.edits] == [1, 3]


def test_a_cursor_naming_no_loop_is_a_bad_reference():
    with pytest.raises(TransformReferenceError, match='does not name'):
        unroll_for(two_loops, StmtCursor(two_loops.ast, FuncBody().stmt(0)), times=1)


# ----------------------------------------------------------------------
# while loops


def test_a_while_unroll_replaces_the_loop_with_one_statement():
    site = StmtCursor(counted.ast, FuncBody().stmt(1))
    out = unroll_while(counted, site, times=1)

    assert out.edits is not None
    edit, = out.edits.edits
    assert (edit.removed, edit.inserted) == (1, 1)
    image = out.forward(site)
    assert isinstance(image, StmtCursor)
    assert image.resolve().format().startswith('if ')


def test_a_zero_times_unroll_records_nothing():
    """Nothing changed, so cursors inside the body still name what they did."""
    from fpy2.transform import WhileUnroll

    log = WhileUnroll.apply_with_edits(counted.ast, None, 0)
    assert log.edits == ()


# ----------------------------------------------------------------------
# call sites


def test_a_cursor_inlines_the_call_in_one_statement():
    out = inline(call_each, StmtCursor(call_each.ast, FuncBody().stmt(0)))
    assert out.format().count('sq(') == 1  # the second call survives


def test_a_cursor_is_coarser_than_the_index_for_inline():
    """One statement, two candidate calls: the index separates them and a
    statement cursor does not."""
    by_index = inline(two_calls, 0)
    assert by_index.format().count('sq(') == 1

    by_cursor = inline(two_calls, StmtCursor(two_calls.ast, FuncBody().stmt(0)))
    assert 'sq(' not in by_cursor.format()


def test_inlining_shifts_what_followed_the_call():
    after = StmtCursor(call_each.ast, FuncBody().stmt(1))
    out = inline(call_each, StmtCursor(call_each.ast, FuncBody().stmt(0)))

    moved = out.forward(after)
    assert isinstance(moved, StmtCursor)
    assert moved.index > 1
    assert moved.resolve().format() == after.resolve().format()


@fp.fpy
def call_in_a_loop(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for v in xs:
        acc = acc + sq(v)
    return acc


def test_inlining_only_prefixes_the_statement_that_held_the_call():
    """The callee's body goes in *ahead* of that statement, which survives — so
    a cursor beneath it forwards, and an edit recorded inside it stands."""
    loop = StmtCursor(call_in_a_loop.ast, FuncBody().stmt(1))
    inside = StmtCursor(call_in_a_loop.ast, FuncBody().stmt(1).block('body').stmt(0))

    out = inline(call_in_a_loop, recursive=False)
    edit, = out.edits.edits if out.edits else ()
    assert edit.removed == 0                     # an insertion, not a replacement

    assert out.forward(loop).resolve().format().startswith('for ')
    assert isinstance(out.forward(inside), StmtCursor)


def test_an_expression_of_an_inlined_statement_does_not_forward():
    """That statement's call became a variable, so an expression cursor in it
    names something else now."""
    call = sites(inline, call_in_a_loop)[0]
    out = inline(call_in_a_loop, recursive=False)
    with pytest.raises(TransformReferenceError, match='whose expressions'):
        out.forward(call)


def test_two_pinned_calls_inline_one_at_a_time():
    """Each cursor is chosen against the original program, so the second has to
    survive the first rewrite."""
    first, second = sites(inline, call_each)
    out = inline(inline(call_each, first), second)
    assert 'sq(' not in out.format()


def test_inline_everywhere_still_reports_its_edits():
    """The bottom-up path inlines every reachable function; only this
    program's own rewrites forward its cursors."""
    out = inline(call_each)
    assert out.edits is not None
    assert [e.index for e in out.edits.edits] == [0, 1]
    assert out.forward(StmtCursor(call_each.ast, FuncBody().stmt(2))).func is out.ast
