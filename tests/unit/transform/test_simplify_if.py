"""
Unit tests for the :class:`fpy2.transform.SimplifyIf` transform.

The pass rewrites `if` statements into `if` expressions, hoisting both branch
bodies and merging each phi with an `IfExpr`.  It mints fresh names via
``Gensym``, so a hand-written golden AST is brittle; these tests assert

1. **Structural shape** — no `IfStmt` or `If1Stmt` survives, and an `IfExpr`
   appears for each merged variable.
2. **Semantic equivalence** via the interpreter, on inputs taking each branch.

This file is the regression net for the refusal conditions that follow.  Every
program here is pure and total, so it stays accepted under every mode the pass
grows.
"""

import re

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.ast.fpyast import If1Stmt, IfExpr, IfStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import SimplifyIf, TransformDeclined

# ----------------------------------------------------------------------
# Helpers


def _count(ast, kind) -> int:
    """How many *kind* nodes are in *ast*."""
    n = 0

    class _C(DefaultVisitor):
        def _visit_if(self, stmt, ctx):
            nonlocal n
            if kind is IfStmt:
                n += 1
            super()._visit_if(stmt, ctx)

        def _visit_if1(self, stmt, ctx):
            nonlocal n
            if kind is If1Stmt:
                n += 1
            super()._visit_if1(stmt, ctx)

        def _visit_if_expr(self, e, ctx):
            nonlocal n
            if kind is IfExpr:
                n += 1
            super()._visit_if_expr(e, ctx)

    _C()._visit_function(ast, None)
    return n


def _names(ast) -> set:
    """Every identifier bound by an assignment in *ast*."""
    out = set()

    class _N(DefaultVisitor):
        def _visit_assign(self, stmt, ctx):
            out.add(stmt.target)
            super()._visit_assign(stmt, ctx)

    _N()._visit_function(ast, None)
    return out


def _apply(f: Function) -> Function:
    return Function(SimplifyIf.apply(f.ast), runtime=f.runtime)


def _agrees(f: Function, *args):
    """The rewrite computes what the original did, on *args*."""
    assert repr(_apply(f)(*args)) == repr(f(*args))


def _no_if_statements(f: Function) -> None:
    ast = SimplifyIf.apply(f.ast)
    assert _count(ast, IfStmt) == 0
    assert _count(ast, If1Stmt) == 0


# ----------------------------------------------------------------------
# Programs


@fp.fpy
def one_armed(x):
    y = x
    if x > 0:
        y = x * 2
    return y


@fp.fpy
def two_armed(x):
    if x > 0:
        y = x * 2
    else:
        y = -x
    return y


@fp.fpy
def mutated_in_both(x):
    acc = 1.0
    if x > 0:
        acc = acc + x
    else:
        acc = acc - x
    return acc


@fp.fpy
def two_variables(x):
    if x > 0:
        a = x
        b = x * 2
    else:
        a = -x
        b = 0.0
    return a + b


@fp.fpy
def nested(x, y):
    if x > 0:
        if y > 0:
            z = 1.0
        else:
            z = 2.0
    else:
        z = 3.0
    return z


@fp.fpy
def condition_is_a_var(x):
    c = x > 0
    if c:
        y = 1.0
    else:
        y = 2.0
    return y


_UNARY = [one_armed, two_armed, mutated_in_both, two_variables, condition_is_a_var]


# ----------------------------------------------------------------------
# Tests


class TestIfStatementsAreEliminated:
    @pytest.mark.parametrize('f', _UNARY + [nested])
    def test_the_input_has_if_statements(self, f):
        """Otherwise the assertions below hold trivially."""
        assert _count(f.ast, IfStmt) + _count(f.ast, If1Stmt) > 0

    @pytest.mark.parametrize('f', _UNARY)
    def test_unary(self, f):
        _no_if_statements(f)

    def test_nested(self):
        _no_if_statements(nested)

    def test_an_if_expression_is_introduced(self):
        assert _count(SimplifyIf.apply(two_armed.ast), IfExpr) == 1

    def test_one_if_expression_per_merged_variable(self):
        assert _count(SimplifyIf.apply(two_variables.ast), IfExpr) == 2


class TestSemanticsArePreserved:
    @pytest.mark.parametrize('f', _UNARY)
    @pytest.mark.parametrize('x', [1.0, -1.0, 0.0])
    def test_unary(self, f, x):
        _agrees(f, x)

    @pytest.mark.parametrize('x', [1.0, -1.0])
    @pytest.mark.parametrize('y', [1.0, -1.0])
    def test_nested(self, x, y):
        _agrees(nested, x, y)


class TestTheConditionTemporary:
    def test_a_compound_condition_is_bound_once(self):
        """An `IfExpr` reads its condition twice, so a non-atom is named."""
        ast = SimplifyIf.apply(two_armed.ast)
        assert 'cond' in {str(n) for n in _names(ast)}

    def test_a_var_condition_needs_no_temporary(self):
        ast = SimplifyIf.apply(condition_is_a_var.ast)
        assert 'cond' not in {str(n) for n in _names(ast)}



# ----------------------------------------------------------------------
# Refusals that hold under every mode


@fp.fpy
def returns_in_branch(x):
    if x > 0:
        return 1.0
    return x * 2


@fp.fpy
def asserts_in_branch(x):
    if x > 0:
        assert x > 10, 'too small'
        y = x
    else:
        y = 0.0
    return y


@fp.fpy
def effect_in_branch(x):
    y = 0.0
    if x > 0:
        fp.round(x)
        y = 1.0
    return y


@fp.fpy
def writes_in_branch(xs: list[fp.Real], x):
    if x > 0:
        xs[0] = x
        y = x
    else:
        y = 0.0
    return y


@fp.fpy
def while_in_branch(x, n: int):
    if x > 0:
        y = 0.0
        i = 0
        while i < n:
            y = y + x
            i = i + 1
    else:
        y = 0.0
    return y


@fp.fpy
def for_in_branch(x, n: int):
    if x > 0:
        y = 0.0
        for _i in range(n):
            y = y + x
    else:
        y = 0.0
    return y


@fp.fpy
def cast_in_branch(x):
    if x > 0:
        y = fp.cast(x)
    else:
        y = 0.0
    return y


@fp.fpy
def nested_unhoistable(x, y):
    if x > 0:
        if y > 0:
            assert y > 10, 'too small'
            z = 1.0
        else:
            z = 2.0
    else:
        z = 3.0
    return z


@fp.fpy
def guarded_read(xs: list[fp.Real], i: int):
    if i < len(xs):
        y = xs[i]
    else:
        y = 0.0
    return y


_REFUSED = [
    (returns_in_branch, '`return` escapes'),
    (asserts_in_branch, '`assert` would run unconditionally'),
    (effect_in_branch, 'effect would run unconditionally'),
    (writes_in_branch, 'list write would run unconditionally'),
    (while_in_branch, '`while` would run unconditionally'),
    (for_in_branch, '`for` would run unconditionally'),
    (cast_in_branch, 'asserts its result is exact'),
    (nested_unhoistable, '`assert` would run unconditionally'),
]


class TestUnconditionalRefusals:
    """Constructs that can change whether, or which, value comes out.  No
    evaluation strategy makes these legal, so no mode admits them."""

    @pytest.mark.parametrize('f,why', _REFUSED, ids=lambda v: getattr(v, 'name', ''))
    def test_declines(self, f, why):
        with pytest.raises(TransformDeclined, match=re.escape(why)):
            SimplifyIf.apply(f.ast)

    def test_a_return_no_longer_raises_a_syntax_error(self):
        """It used to fail as `FPySyntaxError: unbound variable`, naming a
        variable the user never wrote."""
        with pytest.raises(TransformDeclined):
            SimplifyIf.apply(returns_in_branch.ast)

    def test_an_inner_refusal_declines_the_outer_if(self):
        with pytest.raises(TransformDeclined):
            SimplifyIf.apply(nested_unhoistable.ast)


class TestTheRefusalsDoNotOverreach:
    def test_a_guarded_read_is_still_accepted(self):
        """Partial *reads* are the keyword's business, not this phase's."""
        SimplifyIf.apply(guarded_read.ast)



# ----------------------------------------------------------------------
# What the `strict` keyword governs


@fp.fpy
def rounds_under_symbolic_ctx(x):
    if x > 0:
        y = fp.round(x)
    else:
        y = 0.0
    return y


@fp.fpy(ctx=fp.FP32)
def rounds_under_fp32(x):
    if x > 0:
        y = fp.round(x)
    else:
        y = 0.0
    return y


@fp.fpy
def rounds_under_assert_overflow(x):
    if x > 0:
        with fp.MPBFixedContext(-1, 128, overflow=fp.OverflowMode.ASSERT):
            y = fp.round(x)
    else:
        y = 0.0
    return y


@fp.fpy
def slices_in_branch(xs: list[fp.Real], i: int):
    if i < len(xs):
        ys = xs[0:i]
    else:
        ys = xs[0:0]
    return len(ys)


_UNPROVEN = [guarded_read, slices_in_branch, rounds_under_symbolic_ctx]


class TestStrictGovernsUnprovenEffects:
    """Value preserved, observable effects possibly not."""

    @pytest.mark.parametrize('f', _UNPROVEN)
    def test_the_default_hoists(self, f):
        SimplifyIf.apply(f.ast)

    @pytest.mark.parametrize('f', _UNPROVEN)
    def test_strict_declines(self, f):
        with pytest.raises(TransformDeclined):
            SimplifyIf.apply(f.ast, strict=True)

    def test_a_resolved_safe_context_is_accepted_under_strict(self):
        """`strict` refuses what cannot be *shown* safe, not every rounding."""
        SimplifyIf.apply(rounds_under_fp32.ast, strict=True)


class TestAbortsRefuseUnderEveryMode:
    @pytest.mark.parametrize('strict', [False, True])
    def test_assert_overflow_rounding(self, strict):
        with pytest.raises(TransformDeclined, match='ASSERT` overflow'):
            SimplifyIf.apply(rounds_under_assert_overflow.ast, strict=strict)

    @pytest.mark.parametrize('f,why', _REFUSED, ids=lambda v: getattr(v, 'name', ''))
    @pytest.mark.parametrize('strict', [False, True])
    def test_phase_2_refusals_hold_under_strict(self, f, why, strict):
        with pytest.raises(TransformDeclined, match=re.escape(why)):
            SimplifyIf.apply(f.ast, strict=strict)


class TestStrictDoesNotDisturbTotalPrograms:
    @pytest.mark.parametrize('f', _UNARY + [nested])
    @pytest.mark.parametrize('strict', [False, True])
    def test_accepted(self, f, strict):
        SimplifyIf.apply(f.ast, strict=strict)
