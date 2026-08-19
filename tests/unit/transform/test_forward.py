"""
Unit tests for the edit log and forwarding (:mod:`fpy2.transform.cursor`).

No transform is involved: each test pairs two hand-written programs with an
:class:`EditLog` describing the difference, so the forwarding rules are pinned
on their own, before anything depends on them.
"""

import pytest

import fpy2 as fp

from fpy2.transform import Block, Cursor, Edit, EditLog, TransformReferenceError

# ----------------------------------------------------------------------
# One statement of a flat block, replaced by two


@fp.fpy(ctx=fp.REAL)
def flat(x: fp.Real) -> fp.Real:
    a = x + 1
    b = a * 2
    c = b - 3
    return c


@fp.fpy(ctx=fp.REAL)
def flat_split(x: fp.Real) -> fp.Real:
    a = x + 1
    t = a * 2      # `b = a * 2` became two statements
    b = t
    c = b - 3
    return c


_SPLIT = EditLog(flat.ast, flat_split.ast, (Edit(('body',), 1, 1, 2),))


def test_untouched_statement_before_the_edit():
    assert _SPLIT.forward(Cursor(flat.ast, ('body', 0))) == Cursor(flat_split.ast, ('body', 0))


def test_later_siblings_shift():
    for old, new in ((2, 3), (3, 4)):
        cur = _SPLIT.forward(Cursor(flat.ast, ('body', old)))
        assert cur == Cursor(flat_split.ast, ('body', new))


def test_rewritten_statement_forwards_to_its_region():
    region = _SPLIT.forward(Cursor(flat.ast, ('body', 1)))
    assert isinstance(region, Block)
    assert region == Block(flat_split.ast, ('body',), range(1, 3))
    assert len(region) == 2
    assert [c.path for c in region] == [('body', 1), ('body', 2)]
    assert region[0] == Cursor(flat_split.ast, ('body', 1))
    assert region.resolve() == flat_split.ast.body.stmts[1:3]
    assert str(region) == 'body[1:3]'
    with pytest.raises(TransformReferenceError):
        region.one()


def test_forwarding_is_per_program():
    """A cursor of another program is a bad reference, not a coincidence."""
    with pytest.raises(TransformReferenceError):
        _SPLIT.forward(Cursor(flat_split.ast, ('body', 0)))


def test_empty_log_rebases():
    """A pass that leaves the tree alone still hands back a live cursor."""
    log = EditLog(flat.ast, flat_split.ast)
    assert log.forward(Cursor(flat.ast, ('body', 2))) == Cursor(flat_split.ast, ('body', 2))


# ----------------------------------------------------------------------
# A rewrite that consumes a compound statement


@fp.fpy(ctx=fp.REAL)
def wrapped(x: fp.Real) -> fp.Real:
    with fp.FP32:
        y = fp.round(x)
    return y


@fp.fpy(ctx=fp.REAL)
def wrapped_after(x: fp.Real) -> fp.Real:
    t = x          # the `with` block became two statements
    y = t
    return y


_CONSUMED = EditLog(wrapped.ast, wrapped_after.ast, (Edit(('body',), 0, 1, 2),))


def test_statement_inside_a_rewritten_one_does_not_forward():
    inside = Cursor(wrapped.ast, ('body', 0, 'body', 0))
    with pytest.raises(TransformReferenceError, match='which was rewritten'):
        _CONSUMED.forward(inside)


def test_deleted_statement_does_not_forward():
    log = EditLog(flat.ast, flat_split.ast, (Edit(('body',), 1, 1, 0),))
    with pytest.raises(TransformReferenceError, match='was deleted'):
        log.forward(Cursor(flat.ast, ('body', 1)))


# ----------------------------------------------------------------------
# Nesting: shifts at one level, and of an ancestor


@fp.fpy(ctx=fp.REAL)
def nested(x: fp.Real) -> fp.Real:
    a = x
    if x > 0:
        with fp.FP32:
            y = fp.round(a)
        z = y * 2
    else:
        z = -x
    return z


@fp.fpy(ctx=fp.REAL)
def nested_after(x: fp.Real) -> fp.Real:
    a = x
    b = a          # inserted before the `if`
    if x > 0:
        t = b      # the `with` block became two statements
        with fp.FP32:
            y = fp.round(t)
        z = y * 2
    else:
        z = -x
    return z


_NESTED = EditLog(nested.ast, nested_after.ast, (
    Edit(('body',), 1, 0, 1),               # a pure insertion before the `if`
    Edit(('body', 1, 'ift'), 0, 1, 2),      # the `with` block, in the true arm
))


def test_insertion_shifts_an_ancestor():
    """The `if` moved, so a cursor *under* it moves with it."""
    cur = _NESTED.forward(Cursor(nested.ast, ('body', 1, 'iff', 0)))
    assert cur == Cursor(nested_after.ast, ('body', 2, 'iff', 0))


def test_shifts_at_two_levels_compose():
    """`z = y * 2` shifts once for the insertion above the `if`, once for the
    two statements that replaced the `with` block beside it."""
    cur = _NESTED.forward(Cursor(nested.ast, ('body', 1, 'ift', 1)))
    assert cur == Cursor(nested_after.ast, ('body', 2, 'ift', 2))


def test_region_lands_in_the_shifted_block():
    region = _NESTED.forward(Cursor(nested.ast, ('body', 1, 'ift', 0)))
    assert region == Block(nested_after.ast, ('body', 2, 'ift'), range(0, 2))


def test_statement_after_both_edits():
    cur = _NESTED.forward(Cursor(nested.ast, ('body', 2)))
    assert cur == Cursor(nested_after.ast, ('body', 3))


# ----------------------------------------------------------------------
# Several edits in one block


@fp.fpy(ctx=fp.REAL)
def five(x: fp.Real) -> fp.Real:
    a = x + 1
    b = a * 2
    c = b - 3
    d = c * 4
    return d


@fp.fpy(ctx=fp.REAL)
def five_after(x: fp.Real) -> fp.Real:
    a = x + 1
    t1 = a * 2     # `b` became two
    b = t1
    c = b - 3
    t2 = c * 4     # `d` became two
    d = t2
    return d


def test_shifts_accumulate_within_a_block():
    log = EditLog(five.ast, five_after.ast, (
        Edit(('body',), 1, 1, 2),
        Edit(('body',), 3, 1, 2),
    ))
    assert log.forward(Cursor(five.ast, ('body', 2))) == Cursor(five_after.ast, ('body', 3))
    assert log.forward(Cursor(five.ast, ('body', 4))) == Cursor(five_after.ast, ('body', 6))
    assert log.forward(Cursor(five.ast, ('body', 3))) == Block(
        five_after.ast, ('body',), range(4, 6)
    )


def test_single_statement_image_is_a_cursor():
    """A one-for-one rewrite forwards to a cursor, not a region of one."""
    log = EditLog(five.ast, five_after.ast, (Edit(('body',), 1, 1, 1),))
    assert log.forward(Cursor(five.ast, ('body', 1))) == Cursor(five_after.ast, ('body', 1))


# ----------------------------------------------------------------------
# Ill-formed logs


def test_log_rejects_an_edit_past_the_end():
    with pytest.raises(ValueError, match='consumes statements'):
        EditLog(flat.ast, flat_split.ast, (Edit(('body',), 3, 2, 1),))


def test_log_rejects_an_unresolvable_block():
    with pytest.raises(TransformReferenceError):
        EditLog(flat.ast, flat_split.ast, (Edit(('body', 0, 'body'), 0, 1, 1),))


@pytest.mark.parametrize('other', [
    Edit(('body',), 1, 1, 3),                 # the same statement, twice
    Edit(('body',), 1, 0, 1),                 # an insertion into what was consumed
    Edit(('body', 1, 'ift'), 0, 1, 1),        # a block under a consumed statement
])
def test_log_rejects_overlapping_edits(other):
    with pytest.raises(ValueError, match='not disjoint'):
        EditLog(nested.ast, nested_after.ast, (Edit(('body',), 1, 1, 2), other))


def test_edit_rejects_negative_counts():
    with pytest.raises(ValueError, match='ill-formed'):
        Edit(('body',), 0, -1, 1)
