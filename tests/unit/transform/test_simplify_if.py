"""
Unit tests for the :class:`fpy2.transform.SimplifyIf` transform.

The pass rewrites `if` statements into `if` expressions, hoisting both branch
bodies and merging each phi with an `IfExpr`.  It mints fresh names via
``Gensym``, so a hand-written golden AST is brittle; these tests assert

1. **Structural shape** — no `IfStmt` or `If1Stmt` survives, and an `IfExpr`
   appears for each merged variable.
2. **Semantic equivalence** via the interpreter, on inputs taking each branch.

Every program in the first section is pure and total, so it stays accepted
under both modes.
"""

import re

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.ast.fpyast import Assign, BinaryOp, If1Stmt, IfExpr, IfStmt, ReturnStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import OverflowMode, RealFloat
from fpy2.transform.cursor import expr_sites, stmt_sites
from fpy2.transform import (
    BlockCursor,
    SimplifyIf,
    TransformDeclined,
    TransformReferenceError,
)

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

    def test_a_return_declines_rather_than_erroring(self):
        """A `return` in a branch has no expression form; the refusal has to
        come before the rewrite, which would fail on a name it had renamed."""
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
    def test_refusals_hold_under_strict(self, f, why, strict):
        with pytest.raises(TransformDeclined, match=re.escape(why)):
            SimplifyIf.apply(f.ast, strict=strict)


@fp.fpy(ctx=fp.FP64)
def pinned_two_armed(x):
    if x > 0:
        y = x * 2
    else:
        y = -x
    return y


class TestStrictNeedsAResolvedContext:
    """`strict` promises observational equivalence, and an unresolved context
    is exactly what denies it.

    A function with no ``ctx=`` inherits its caller's, so every rounded
    operation in it -- all arithmetic, not just `fp.round` -- might be under a
    context that aborts on overflow.  `strict` therefore declines an
    unannotated branch and accepts a pinned one; the default hoists both.
    """

    @pytest.mark.parametrize('f', _UNARY + [nested])
    def test_the_default_accepts_an_unannotated_program(self, f):
        SimplifyIf.apply(f.ast)

    @pytest.mark.parametrize(
        'f', [one_armed, two_armed, mutated_in_both, two_variables]
    )
    def test_strict_declines_unannotated_arithmetic(self, f):
        with pytest.raises(TransformDeclined, match='unresolved context'):
            SimplifyIf.apply(f.ast, strict=True)

    @pytest.mark.parametrize('f', [condition_is_a_var, nested])
    def test_a_literal_only_branch_is_accepted(self, f):
        """`ContextUse` records no use site for a literal, and a literal does
        not overflow its context -- so there is nothing here `strict` cannot
        prove."""
        SimplifyIf.apply(f.ast, strict=True)

    @pytest.mark.parametrize('strict', [False, True])
    def test_a_pinned_context_is_accepted_under_both(self, strict):
        SimplifyIf.apply(pinned_two_armed.ast, strict=strict)


# ----------------------------------------------------------------------
# Aiming the rewrite


@fp.fpy
def two_ifs(x, y):
    if x > 0:
        a = 1.0
    else:
        a = 2.0
    if y > 0:
        b = 3.0
    else:
        b = 4.0
    return a + b


def _n_ifs(ast) -> int:
    return _count(ast, IfStmt) + _count(ast, If1Stmt)


class TestSites:
    def test_listing_counts_every_if(self):
        assert len(SimplifyIf.sites(two_ifs.ast)) == 2

    def test_a_listing_matches_what_where_none_rewrites(self):
        """The pass's own walk, so a listing and an apply cannot disagree."""
        assert len(SimplifyIf.sites(two_ifs.ast)) == _n_ifs(two_ifs.ast)
        assert _n_ifs(SimplifyIf.apply(two_ifs.ast)) == 0

    def test_a_declined_branch_is_not_a_site(self):
        assert SimplifyIf.sites(asserts_in_branch.ast) == []

    def test_but_it_is_a_refusal(self):
        refused = SimplifyIf.refusals(asserts_in_branch.ast)
        assert len(refused) == 1 and 'assert' in refused[0][1]

    def test_strict_changes_what_is_a_site(self):
        """`strict` decides what the pass declines, so it decides what a
        `where` index counts."""
        assert len(SimplifyIf.sites(guarded_read.ast)) == 1
        assert SimplifyIf.sites(guarded_read.ast, strict=True) == []


class TestWhere:
    @pytest.mark.parametrize('where', [0, 1])
    def test_an_index_rewrites_exactly_one(self, where):
        assert _n_ifs(SimplifyIf.apply(two_ifs.ast, where)) == 1

    def test_none_rewrites_every_one(self):
        assert _n_ifs(SimplifyIf.apply(two_ifs.ast, None)) == 0

    def test_a_cursor_rewrites_the_one_it_names(self):
        cursor = SimplifyIf.sites(two_ifs.ast)[1]
        assert _n_ifs(SimplifyIf.apply(two_ifs.ast, cursor)) == 1

    def test_an_out_of_range_index_is_a_bad_reference(self):
        with pytest.raises(TransformReferenceError, match='does not correspond'):
            SimplifyIf.apply(two_ifs.ast, 5)

    @pytest.mark.parametrize('where', [0, 1])
    def test_semantics_are_preserved(self, where):
        out = SimplifyIf.apply(two_ifs.ast, where)
        g = Function(out, runtime=two_ifs.runtime)
        for x in (1.0, -1.0):
            for y in (1.0, -1.0):
                assert repr(g(x, y)) == repr(two_ifs(x, y))


class TestNestedIfs:
    """A cursor takes the sites beneath it; an index takes exactly one.

    Leaving a nested `if` behind is sound rather than merely tolerated: it
    becomes unconditional, but a branch body is effect-free by this pass's own
    refusals, so the value it computes is discarded by the enclosing `IfExpr`.
    """

    def test_an_index_takes_only_the_outer_if(self):
        assert _n_ifs(nested.ast) == 2
        assert _n_ifs(SimplifyIf.apply(nested.ast, 0)) == 1

    def test_a_cursor_takes_the_subtree(self):
        outer = SimplifyIf.sites(nested.ast)[0]
        assert _n_ifs(SimplifyIf.apply(nested.ast, outer)) == 0

    @pytest.mark.parametrize('x', [1.0, -1.0])
    @pytest.mark.parametrize('y', [1.0, -1.0])
    def test_semantics_are_preserved(self, x, y):
        out = SimplifyIf.apply(nested.ast, 0)
        assert repr(Function(out, runtime=nested.runtime)(x, y)) == repr(nested(x, y))

    def test_an_inner_refusal_removes_the_outer_site(self):
        """The subtree rule runs both ways: an `if` whose branch holds an
        unhoistable statement at any depth is not a site."""
        assert SimplifyIf.sites(nested_unhoistable.ast) == []

    def test_an_index_then_names_nothing_but_says_why(self):
        """`check_site` reports an index naming no site as a bad reference,
        carrying the reasons its candidates were refused for."""
        with pytest.raises(TransformReferenceError, match='assert'):
            SimplifyIf.apply(nested_unhoistable.ast, 0)

    def test_a_cursor_naming_a_refused_candidate_declines(self):
        """Deliberately not a reference error: a cursor names a real place, so
        saying why beats saying it named nothing."""
        cursor, _why = SimplifyIf.refusals(nested_unhoistable.ast)[0]
        with pytest.raises(TransformDeclined, match='assert'):
            SimplifyIf.apply(nested_unhoistable.ast, cursor)


# ----------------------------------------------------------------------
# Forwarding


@fp.fpy
def around_an_if(x, y):
    a = x + 1.0
    if x > 0:
        b = 1.0
    else:
        b = 2.0
    c = y * 2.0
    return a + b + c


def _stmts(ast, kind):
    return stmt_sites(ast, lambda s: isinstance(s, kind))


class TestTheEditLog:
    def test_an_if_is_one_edit(self):
        log = SimplifyIf.apply_with_edits(around_an_if.ast)
        assert len(log.edits) == 1
        (e,) = log.edits
        assert (e.index, e.removed) == (1, 1)
        assert e.inserted > 1

    def test_apply_is_the_log_result(self):
        log = SimplifyIf.apply_with_edits(around_an_if.ast)
        assert log.result.is_equiv(SimplifyIf.apply(around_an_if.ast))


class TestStatementCursorsForward:
    def test_one_before_the_rewrite_is_unmoved(self):
        log = SimplifyIf.apply_with_edits(around_an_if.ast)
        before = _stmts(around_an_if.ast, Assign)[0]
        assert log.forward(before).path.index == before.path.index

    def test_one_after_shifts_by_the_growth(self):
        log = SimplifyIf.apply_with_edits(around_an_if.ast)
        (e,) = log.edits
        after = _stmts(around_an_if.ast, ReturnStmt)[0]
        moved = log.forward(after).path.index
        assert moved == after.path.index + e.inserted - e.removed

    def test_the_if_forwards_to_the_region_that_replaced_it(self):
        log = SimplifyIf.apply_with_edits(around_an_if.ast)
        got = log.forward(_stmts(around_an_if.ast, IfStmt)[0])
        assert isinstance(got, BlockCursor)

    def test_one_inside_a_branch_fails_loudly(self):
        """That subtree was rebuilt and renamed; only the pass could say what
        became of it, so forwarding refuses rather than mis-aiming."""
        log = SimplifyIf.apply_with_edits(around_an_if.ast)
        inner = [c for c in _stmts(around_an_if.ast, Assign)
                 if c.path.parent != _stmts(around_an_if.ast, Assign)[0].path.parent]
        assert inner
        with pytest.raises(TransformReferenceError):
            log.forward(inner[0])


class TestExpressionCursorsForward:
    def test_expressions_outside_the_rewrite_are_preserved(self):
        """What `exprs_preserved=True` claims.  The only rewrite reaching past
        a replaced statement is the closing `CopyPropagate`, restricted to
        names this pass minted."""
        log = SimplifyIf.apply_with_edits(around_an_if.ast)
        outside = expr_sites(around_an_if.ast, lambda e: isinstance(e, BinaryOp))
        checked = 0
        for c in outside:
            try:
                got = log.forward(c)
            except TransformReferenceError:
                continue
            assert c.resolve().is_equiv(got.resolve())
            checked += 1
        assert checked, 'no expression cursor forwarded, so nothing was checked'


@fp.fpy
def mutated_in_one_arm_each(c):
    a = 0.0
    b = 0.0
    if c > 0:
        a = 1.0
    else:
        b = 2.0
    return a + b


@fp.fpy
def mutated_in_one_arm_only(c, x):
    y = x
    if c > 0:
        y = x * 2
    else:
        pass
    return y


class TestAsymmetricMutation:
    """A variable mutated in one arm only still needs a merge.

    The merge once ran over the `if` arm's names alone, so a variable touched
    only in the `else` arm kept its pre-`if` value -- a wrong answer with no
    refusal.  Every other program here assigns the same names in both arms,
    which is why the suite could not see it.
    """

    @pytest.mark.parametrize('c', [1.0, -1.0])
    def test_a_variable_per_arm(self, c):
        _agrees(mutated_in_one_arm_each, c)

    @pytest.mark.parametrize('c', [1.0, -1.0])
    def test_an_empty_else(self, c):
        _agrees(mutated_in_one_arm_only, c, 3.0)

    def test_both_variables_are_merged(self):
        assert _count(SimplifyIf.apply(mutated_in_one_arm_each.ast), IfExpr) == 2


# ----------------------------------------------------------------------
# Aborts reachable other than through `Round`


@fp.fpy
def arithmetic_under_assert_overflow(x):
    if x < 2:
        with fp.MPBFixedContext(-1, 128, overflow=fp.OverflowMode.ASSERT):
            y = x * x
    else:
        y = 0.0
    return y


@fp.fpy
def round_at_under_assert_overflow(x):
    if x < 0:
        with fp.MPBFixedContext(-1, 128, overflow=fp.OverflowMode.ASSERT):
            y = fp.round_at(x, 2)
    else:
        y = 0.0
    return y


@fp.fpy
def _asserting_callee(x):
    assert x > 0, 'positive'
    return x


@fp.fpy
def calls_an_asserting_function(x):
    if x > 0:
        y = _asserting_callee(x)
    else:
        y = 0.0
    return y


class TestAbortsNotReachedThroughRound:
    """The check is keyed on whether an expression consults the rounding
    context, not on its node class.

    Keying it on `isinstance(e, Round | Cast)` let three shapes through, each
    of which diverged from the interpreter under *both* modes: arithmetic
    (every rounded operation consults the context), `fp.round_at`, and an
    abort reached through a callee.
    """

    @pytest.mark.parametrize('strict', [False, True])
    @pytest.mark.parametrize('f', [
        arithmetic_under_assert_overflow,
        round_at_under_assert_overflow,
    ])
    def test_assert_overflow_is_found_whatever_rounds(self, f, strict):
        with pytest.raises(TransformDeclined, match='ASSERT` overflow'):
            SimplifyIf.apply(f.ast, strict=strict)

    @pytest.mark.parametrize('strict', [False, True])
    def test_a_callee_is_refused_rather_than_scanned(self, strict):
        with pytest.raises(TransformDeclined, match='callee is not scanned'):
            SimplifyIf.apply(calls_an_asserting_function.ast, strict=strict)

    def test_a_context_constructor_is_not_a_refused_call(self):
        """`fp.MPBFixedContext(...)` in a `with` header is a `Call` too;
        refusing every call would decline any branch that opens a context."""
        assert SimplifyIf.refusals(arithmetic_under_assert_overflow.ast)
        reasons = [w for _c, w in
                   SimplifyIf.refusals(arithmetic_under_assert_overflow.ast)]
        assert not any('foreign' in w for w in reasons)


# ----------------------------------------------------------------------
# A context that cannot hold what an operation produces


@fp.fpy(ctx=fp.INTEGER)
def logb_guarded_from_zero(xs: list[fp.Real]):
    largest = fp.round(0)
    for x in xs:
        if x != 0:
            largest = max(largest, fp.logb(x))
    return largest


class TestUnrepresentableResults:
    """`fp.logb(0)` is an infinity for a finite operand, and `INTEGER` holds
    no infinity -- so *hoisting* it past `x != 0` would turn a returning
    program into a raising one.

    Two things keep that from happening.  The arm inlines, so `fp.logb` stays
    inside the lazy `IfExpr` and never runs on the input the guard excluded --
    which is why the default mode rewrites these and still agrees.  Where an
    arm cannot inline, `strict` declines: the trap is the format having
    nowhere to put the result, not an abort the program asked for, so it is
    `unproven` rather than `aborts`.

    The condition is representability in the active context, not the operator
    -- the same `fp.logb` under FP64 yields `-inf` and is fine.
    """

    def test_strict_declines(self):
        with pytest.raises(TransformDeclined, match='infinity or NaN'):
            SimplifyIf.apply(logb_guarded_from_zero.ast, strict=True)

    def test_the_default_rewrites_and_agrees(self):
        """Including on `0`, the input the guard excluded."""
        _no_if_statements(logb_guarded_from_zero)
        for xs in ([], [0.0], [0.0, 4.0], [8.0, 0.0, 2.0]):
            _agrees(logb_guarded_from_zero, xs)

    @pytest.mark.parametrize('strict', [False, True])
    def test_arithmetic_under_the_same_context_is_accepted(self, strict):
        """Only an operation that can *produce* a special is refused.  `x + y`
        under `INTEGER` cannot -- the context is unbounded, so there is no
        overflow either -- and the guard there is not load-bearing."""
        @fp.fpy(ctx=fp.INTEGER)
        def adds(xs: list[fp.Real]):
            total = fp.round(0)
            for x in xs:
                if x != 0:
                    total = total + x
            return total

        SimplifyIf.apply(adds.ast, strict=strict)

    def test_strict_declines_an_inverse_trig_pole(self):
        """`acos` is a pole op by IEEE 754 §7.2: `acos(2)` is NaN."""
        @fp.fpy(ctx=fp.INTEGER)
        def guarded(x: fp.Real):
            y = fp.round(0)
            if x <= 1:
                y = fp.acos(x)
            return y

        with pytest.raises(TransformDeclined, match='infinity or NaN'):
            SimplifyIf.apply(guarded.ast, strict=True)

    def test_strict_declines_overflow_under_a_bounded_context(self):
        """The other route to a special is IEEE 754 §7.4 overflow, which turns
        on the context rather than the operation: under a bounded format that
        rounds an overflow to infinity, even `x * y` needs its guard."""
        small = fp.MPBFloatContext(4, -6, RealFloat(m=15, exp=4),
                                   enable_inf=False, enable_nan=False)

        @fp.fpy(ctx=small)
        def guarded(x: fp.Real, y: fp.Real):
            z = fp.round(1)
            if x < 10:
                z = x * y
            return z

        with pytest.raises(TransformDeclined, match='overflow to an infinity'):
            SimplifyIf.apply(guarded.ast, strict=True)

    @pytest.mark.parametrize('strict', [False, True])
    def test_a_saturating_context_is_accepted(self, strict):
        """`SATURATE` clamps instead of rounding to infinity, so the same
        program has nothing to raise."""
        small = fp.MPBFloatContext(4, -6, RealFloat(m=15, exp=4),
                                   overflow=OverflowMode.SATURATE,
                                   enable_inf=False, enable_nan=False)

        @fp.fpy(ctx=small)
        def guarded(x: fp.Real, y: fp.Real):
            z = fp.round(1)
            if x < 10:
                z = x * y
            return z

        SimplifyIf.apply(guarded.ast, strict=strict)

    def test_the_same_shape_under_fp64_is_accepted(self):
        """The refusal is about the context, not the operation."""
        @fp.fpy(ctx=fp.FP64)
        def under_fp64(xs: list[fp.Real]):
            largest = fp.round(0)
            for x in xs:
                if x != 0:
                    largest = max(largest, fp.logb(x))
            return largest

        SimplifyIf.apply(under_fp64.ast, strict=True)


class TestArmInlining:
    """An arm that reduces to expressions goes inside the `IfExpr`, which is
    lazy, so it keeps its guard instead of being hoisted."""

    def test_the_operation_stays_in_the_arm(self):
        @fp.fpy(ctx=fp.FP64)
        def guarded(x: fp.Real):
            y = fp.round(0)
            if x != 0:
                y = fp.logb(x)
            return y

        src = SimplifyIf.apply(guarded.ast).format()
        # `fp.logb` appears only as an `IfExpr` arm, never on its own line
        assert 'fp.logb(x) if' in src
        assert not re.search(r'=\s*fp\.logb\(x\)\s*$', src, re.M)

    def test_merges_happen_at_once(self):
        """A merge reads pre-`if` names, so one merge must not see a name an
        earlier merge already overwrote."""
        @fp.fpy(ctx=fp.FP64)
        def two(c: bool, p: fp.Real, q: fp.Real):
            if c:
                q = p + 1
                p = q * 2
            return (p, q)

        for c in (True, False):
            for p, q in [(1.0, 0.0), (-2.5, 7.0), (0.0, 0.0)]:
                _agrees(two, c, p, q)

    def test_a_nested_arm_inlines_through(self):
        """The `max_e` shape: an inner `if` becomes expressions, which lets the
        outer arm inline too, so `fp.logb` never leaves its guard."""
        @fp.fpy(ctx=fp.FP64)
        def max_e(xs: list[fp.Real]):
            largest_e = fp.round(0)
            any_non_zero: bool = False
            for x in xs:
                if fp.isfinite(x) and x != 0:
                    if any_non_zero:
                        largest_e = max(largest_e, fp.logb(x))
                    else:
                        largest_e = fp.logb(x)
                        any_non_zero = True
            return (largest_e, any_non_zero)

        for xs in ([], [0.0], [0.0, 2.0], [4.0, 0.0, 16.0], [1.0, 1.0]):
            _agrees(max_e, xs)
