"""
Unit tests for expression paths and cursors.

An expression path is the third arm of the grammar, and the one place forwarding
rests on a *claim* rather than on structure: a pass may rewrite expressions in
statements it never replaces, so an expression cursor forwards only where the
pass says it left them alone.
"""

import pytest

import fpy2 as fp

from fpy2.ast import Add, Call, ForStmt, IfStmt, Integer, Mul
from fpy2.transform import (
    Edit,
    EditLog,
    ExprCursor,
    ExprPath,
    FuncBody,
    StmtCursor,
    TransformReferenceError,
)
from fpy2.transform.utils.path import format_path, resolve_expr, sub_exprs


@fp.fpy(ctx=fp.REAL)
def arith(x: fp.Real, y: fp.Real) -> fp.Real:
    a = (x * 2) + y
    return a


@fp.fpy(ctx=fp.REAL)
def branchy(x: fp.Real, xs: list[fp.Real]) -> fp.Real:
    y = 0.0
    if x > 0:
        for v in xs:
            y = y + v * x
    else:
        y = -x
    return y


# ----------------------------------------------------------------------
# Paths into expressions


def test_a_path_descends_into_an_expression():
    p = FuncBody().stmt(0).expr('expr')
    assert format_path(p) == 'body[0].expr'
    assert isinstance(resolve_expr(arith.ast, p), Add)

    left = p.expr('args', 0)
    assert format_path(left) == 'body[0].expr.args[0]'
    assert isinstance(resolve_expr(arith.ast, left), Mul)
    assert isinstance(resolve_expr(arith.ast, left.expr('args', 1)), Integer)


def test_an_expression_path_knows_its_statement():
    p = FuncBody().stmt(0).expr('expr').expr('args', 0).expr('args', 1)
    assert p.stmt() == FuncBody().stmt(0)


def test_sub_exprs_covers_the_statement_kinds():
    stmts = branchy.ast.body.stmts
    fields = {
        type(s).__name__: [f for f, _, _ in sub_exprs(s)]
        for s in stmts
    }
    assert fields == {
        'Assign': ['expr'], 'IfStmt': ['cond'], 'ReturnStmt': ['expr'],
    }

    branch = stmts[1]
    assert isinstance(branch, IfStmt)
    loop = branch.ift.stmts[0]
    assert isinstance(loop, ForStmt)
    assert [f for f, _, _ in sub_exprs(loop)] == ['iterable']
    assert [f for f, _, _ in sub_exprs(loop.body.stmts[0])] == ['expr']


def test_a_path_naming_no_expression_is_a_bad_reference():
    with pytest.raises(TransformReferenceError, match='names no `cond` expression'):
        resolve_expr(arith.ast, FuncBody().stmt(0).expr('cond'))
    with pytest.raises(TransformReferenceError, match='names no `args` expression'):
        resolve_expr(arith.ast, FuncBody().stmt(0).expr('expr').expr('args', 9))


def test_the_cursor_validates_and_prints():
    cur = ExprCursor(arith.ast, FuncBody().stmt(0).expr('expr').expr('args', 0))
    assert isinstance(cur.resolve(), Mul)
    assert cur.stmt() == StmtCursor(arith.ast, FuncBody().stmt(0))
    assert str(cur).startswith('body[0].expr.args[0] at ')

    with pytest.raises(TransformReferenceError):
        ExprCursor(arith.ast, FuncBody().stmt(0).expr('cond'))
    with pytest.raises(TypeError):
        ExprCursor(arith.ast, FuncBody().stmt(0))  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Forwarding


@fp.fpy(ctx=fp.REAL)
def two_stmts(x: fp.Real, y: fp.Real) -> fp.Real:
    a = x + y
    b = a * 2
    return b


@fp.fpy(ctx=fp.REAL)
def two_stmts_after(x: fp.Real, y: fp.Real) -> fp.Real:
    t = x          # `a = x + y` became two statements
    a = t + y
    b = a * 2
    return b


def _log(**kwargs) -> EditLog:
    return EditLog(
        two_stmts.ast, two_stmts_after.ast, (Edit(FuncBody(), 0, 1, 2),), **kwargs
    )


def test_an_expression_shifts_with_its_statement():
    """`b = a * 2` only moved, so the expression path still names the same
    expression."""
    cur = ExprCursor(two_stmts.ast, FuncBody().stmt(1).expr('expr').expr('args', 0))
    out = _log(exprs_preserved=True).forward(cur)

    assert isinstance(out, ExprCursor)
    assert out.path == FuncBody().stmt(2).expr('expr').expr('args', 0)
    assert out.resolve().format() == cur.resolve().format()


def test_an_expression_of_a_rewritten_statement_does_not_forward():
    cur = ExprCursor(two_stmts.ast, FuncBody().stmt(0).expr('expr'))
    with pytest.raises(TransformReferenceError, match='which was rewritten'):
        _log(exprs_preserved=True).forward(cur)


def test_an_expression_does_not_forward_without_the_claim():
    """The default is that a pass says nothing about expressions it did not
    replace a statement for — so a cursor fails rather than mis-aiming."""
    cur = ExprCursor(two_stmts.ast, FuncBody().stmt(1).expr('expr'))
    with pytest.raises(TransformReferenceError, match='does not say what it did'):
        _log().forward(cur)


def test_an_expression_of_another_program_does_not_forward():
    cur = ExprCursor(arith.ast, FuncBody().stmt(0).expr('expr'))
    with pytest.raises(TransformReferenceError, match='another program'):
        _log(exprs_preserved=True).forward(cur)


# ----------------------------------------------------------------------
# Aiming


@fp.fpy
def sq(x: fp.Real) -> fp.Real:
    return x * x


@fp.fpy
def two_calls(x: fp.Real, y: fp.Real) -> fp.Real:
    return sq(x) + sq(y)


def test_an_expression_cursor_aims_inline_at_one_call():
    """The coarseness a statement cursor has here: `inline` counts calls, and an
    expression cursor names one of the two in this statement."""
    from fpy2.strategies import inline

    second = ExprCursor(two_calls.ast, FuncBody().stmt(0).expr('expr').expr('args', 1))
    assert isinstance(second.resolve(), Call)

    out = inline(two_calls, second)
    assert out.format().count('sq(') == 1
    # ... and it is the *first* call that survived
    assert 'sq(x)' in out.format()


def test_a_statement_sited_rewrite_refuses_an_expression_cursor():
    """No statement sits beneath an expression, and the message says so rather
    than reporting no candidate."""
    from fpy2.strategies import unfold_special

    @fp.fpy(ctx=fp.REAL)
    def rounded(x: fp.Real) -> fp.Real:
        with fp.FP16:
            y = fp.round(x)
        return y

    cur = ExprCursor(rounded.ast, FuncBody().stmt(1).expr('expr'))
    with pytest.raises(TransformReferenceError, match='aimed at statements'):
        unfold_special(rounded, where=cur)


def test_rebasing_an_expression_cursor_is_the_identity_on_its_own_program():
    cur = ExprCursor(two_calls.ast, FuncBody().stmt(0).expr('expr').expr('args', 0))
    assert two_calls.rebase(cur) == cur


def test_an_expression_path_is_built_only_under_a_statement():
    """The ADT's third arm: an expression hangs off a statement or another
    expression, never off a block."""
    p = FuncBody().stmt(0).expr('expr')
    assert isinstance(p, ExprPath)
    assert not hasattr(FuncBody(), 'expr')
