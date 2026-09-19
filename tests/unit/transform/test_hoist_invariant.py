"""
Regression net for `HoistInvariant` — Phase 1 of `docs/todos/hoist-invariant.md`.

Nothing hoists yet.  These assertions pin what the tree does *today*, so that the
diffs in Phase 3 (the transform) and Phase 5 (its entry into `Simplify`) show
exactly which statements moved.  Every assertion that is meant to flip later
says so on the line above it.

The sweeps include the empty list on purpose: a zero-trip loop is where hoisting
changes *whether* an invariant expression is evaluated at all, and the decision
recorded in the plan is to hoist anyway.
"""

import fpy2 as fp
import fpy2.strategies as st
import pytest

from fpy2.analysis import DefineUse
from fpy2.ast import Assign, ForStmt, StmtBlock, WhileStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import HoistInvariant
from fpy2.transform.hoist_invariant import _invariants
from fpy2.utils import NamedId

# zero-trip, one-trip, and a spread that straddles the FP16 subnormal and
# overflow boundaries, so a hoist that changed rounding would show up
_VALUES = ([], [1.0], [1.1, 3.7, 1e-5], [65600.0, 1e-8, 2.0, -4.5])


def _text(func, ast) -> str:
    return ' '.join(fp.Function(ast, runtime=func.runtime).format().split())


def _block_names(block: StmtBlock) -> set[str]:
    """Names assigned by a direct child of *block*."""
    return {
        str(s.target) for s in block.stmts
        if isinstance(s, Assign) and isinstance(s.target, NamedId)
    }


def _body_names(ast) -> set[str]:
    """Names assigned directly in some loop body."""
    found: set[str] = set()

    class V(DefaultVisitor):
        def _visit_for(self, stmt: ForStmt, ctx):
            found.update(_block_names(stmt.body))
            super()._visit_for(stmt, ctx)

        def _visit_while(self, stmt: WhileStmt, ctx):
            found.update(_block_names(stmt.body))
            super()._visit_while(stmt, ctx)

    V()._visit_function(ast, None)
    return found


def _agrees(func, ast, values=_VALUES) -> bool:
    """The rewritten function computes what the original computes, bit for
    bit — same value, same flags, same context tag on the result."""
    rewritten = fp.Function(ast, runtime=func.runtime)
    return all(repr(rewritten(v)) == repr(func(v)) for v in values)


def _agrees_by_value(func, ast, values=_VALUES) -> bool:
    """Same, but comparing only the number.

    `rescale_fixed` moves where the rounding happens, so a result can come back
    carrying a different context tag: on a one-element list the source's `sum`
    returns the rounded element still tagged `MPFixedContext`, while the
    rescaled schedule's final unscaling multiply re-tags it `RealContext`.
    Same `exp`, same `c`, same flags.
    """
    rewritten = fp.Function(ast, runtime=func.runtime)
    return all(rewritten(v) == func(v) for v in values)


@fp.fpy(ctx=fp.REAL)
def scaled_sum(xs: list[fp.Real]) -> fp.Real:
    n = len(xs)
    acc = 0.0
    for x in xs:
        c = n + 1
        acc = acc + c * x
    return acc


@fp.fpy(ctx=fp.REAL)
def scaled_while(xs: list[fp.Real]) -> fp.Real:
    n = len(xs)
    acc = 0.0
    i = 0
    while i < n:
        c = n + 1
        acc = acc + c * xs[i]
        i = i + 1
    return acc


class TestTheLoopsThemselves:
    """`c` is invariant in both bodies and nothing moves it."""

    def test_a_for_body_keeps_its_invariant(self):
        # the "before" side; `TestTheTransform` asserts the "after"
        assert 'c' in _body_names(scaled_sum.ast)

    def test_a_while_body_keeps_its_invariant(self):
        # the "before" side; `TestTheTransform` asserts the "after"
        assert 'c' in _body_names(scaled_while.ast)


class TestSimplify:
    """`Simplify` has no pass that relocates a statement."""

    def test_it_leaves_a_for_invariant_alone(self):
        out = st.simplify(scaled_sum)
        # flips in Phase 5: `simplify` gains `HoistInvariant`
        assert 'c' in _body_names(out.ast)
        assert _agrees(scaled_sum, out.ast)

    def test_it_leaves_a_while_invariant_alone(self):
        out = st.simplify(scaled_while)
        # flips in Phase 5: `simplify` gains `HoistInvariant`
        assert 'c' in _body_names(out.ast)
        assert _agrees(scaled_while, out.ast)


class TestRescaleFixedOutput:
    """The motivating schedule, and the wart this PR exists to remove."""

    P = 12

    @staticmethod
    @fp.fpy(ctx=fp.REAL)
    def fused_sum(xs: list[fp.Real]) -> fp.Real:
        if all([fp.isfinite(x) for x in xs]):
            e = max([fp.logb(x) for x in xs])
            with fp.MPFixedContext(e - 12, rm=fp.RM.RTZ, enable_neg_zero=False):
                ts = [fp.round(x) for x in xs]
            return sum(ts)
        else:
            with fp.FP32:
                return sum(xs)

    @staticmethod
    def _schedule(func):
        return st.simplify(st.rescale_fixed(st.comp_to_loop(st.fuse(func))))

    def test_the_scale_is_recomputed_every_iteration(self):
        out = self._schedule(self.fused_sum)
        # the "before" side; `TestTheTransform` asserts the "after"
        assert '_k' in _body_names(out.ast)

    def test_the_scale_is_written_as_a_power_of_two(self):
        """What `HoistScale` will look for in PR 2: an invariant factor
        multiplying each element, written into the result list."""
        src = _text(self.fused_sum, self._schedule(self.fused_sum).ast)
        assert '_k = ((e - 12) + 1)' in src
        assert '(2 ** _k)' in src

    def test_the_schedule_preserves_the_source_semantics(self):
        """Not about hoisting — the baseline the later phases are measured
        against.  The empty list is excluded: `max([])` has no value."""
        out = self._schedule(self.fused_sum)
        assert _agrees_by_value(self.fused_sum, out.ast, values=_VALUES[1:])


# ----------------------------------------------------------------------
# Phase 2: the invariance query, and Phase 3: the transform that acts on it


def _loops(ast) -> list:
    """Every `for` and `while` in *ast*, outermost first."""
    found = []

    class V(DefaultVisitor):
        def _visit_for(self, stmt: ForStmt, ctx):
            found.append(stmt)
            super()._visit_for(stmt, ctx)

        def _visit_while(self, stmt: WhileStmt, ctx):
            found.append(stmt)
            super()._visit_while(stmt, ctx)

    V()._visit_function(ast, None)
    return found


def _query(func, which: int = 0) -> list[str]:
    """The names `_invariants` says may leave the *which*-th loop of *func*."""
    def_use = DefineUse.analyze(func.ast)
    return [str(s.target) for s in _invariants(_loops(func.ast)[which], def_use)]


def _hoisted(func):
    """*func* with every invariant binding moved above its loop."""
    return func.with_ast(HoistInvariant.apply(func.ast))


def _foreign(x):
    return x


# The programs the query refuses, one per condition.  Shared so that the
# transform is tested against exactly what the query rejects.


@fp.fpy(ctx=fp.REAL)
def reads_the_loop_target(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x in xs:
        c = x + 1
        acc = acc + c
    return acc


@fp.fpy(ctx=fp.REAL)
def reads_a_name_the_body_rebinds(xs: list[fp.Real]) -> fp.Real:
    m = 1.0
    acc = 0.0
    for x in xs:
        c = m + 1
        m = m + 1
        acc = acc + c * x
    return acc


@fp.fpy(ctx=fp.REAL)
def binds_the_target_twice(xs: list[fp.Real]) -> fp.Real:
    n = len(xs)
    acc = 0.0
    for x in xs:
        c = n + 1
        acc = acc + c * x
        c = n + 2
    return acc


@fp.fpy(ctx=fp.REAL)
def reads_the_target_after_the_loop(xs: list[fp.Real]) -> fp.Real:
    n = len(xs)
    c = 0.0
    acc = 0.0
    for x in xs:
        c = n + 1
        acc = acc + c * x
    return acc + c


@fp.fpy(ctx=fp.REAL)
def binds_a_tuple(xs: list[fp.Real]) -> fp.Real:
    n = len(xs)
    acc = 0.0
    for x in xs:
        a, b = (n + 1, n + 2)
        acc = acc + (a + b) * x
    return acc


@fp.fpy(ctx=fp.REAL)
def binds_under_a_nested_with(xs: list[fp.Real]) -> fp.Real:
    n = len(xs)
    acc = 0.0
    for x in xs:
        with fp.FP32:
            c = n + 1
        acc = acc + c * x
    return acc


@fp.fpy(ctx=fp.REAL)
def binds_an_impure_expression(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x in xs:
        c = _foreign(1.0)
        acc = acc + c * x
    return acc


# `binds_an_impure_expression` is not among them: the interpreter refuses to
# call a foreign Python function, so there is nothing to run a sweep against.
REFUSED_RUNNABLE = (
    reads_the_loop_target,
    reads_a_name_the_body_rebinds,
    binds_the_target_twice,
    reads_the_target_after_the_loop,
    binds_a_tuple,
    binds_under_a_nested_with,
)

REFUSED = (*REFUSED_RUNNABLE, binds_an_impure_expression)


@fp.fpy(ctx=fp.REAL)
def chained(xs: list[fp.Real]) -> fp.Real:
    n = len(xs)
    acc = 0.0
    for x in xs:
        a = n + 1
        b = a * 2
        acc = acc + b * x
    return acc


class TestTheQueryFinds:

    def test_an_invariant_in_a_for_body(self):
        assert _query(scaled_sum) == ['c']

    def test_an_invariant_in_a_while_body(self):
        assert _query(scaled_while) == ['c']

    def test_a_chain_in_one_pass(self):
        """`b` reads `a`, which this same pass is taking out, so both go."""
        assert _query(chained) == ['a', 'b']


class TestTheQueryRefuses:

    @pytest.mark.parametrize('func', REFUSED, ids=lambda f: f.name)
    def test_it(self, func):
        assert _query(func) == []


class TestTheTransform:

    def test_it_hoists_out_of_a_for(self):
        out = _hoisted(scaled_sum)
        assert 'c' not in _body_names(out.ast)
        assert 'c' in _block_names(out.ast.body)
        assert _agrees(scaled_sum, out.ast)

    def test_it_hoists_out_of_a_while(self):
        out = _hoisted(scaled_while)
        assert 'c' not in _body_names(out.ast)
        assert 'c' in _block_names(out.ast.body)
        assert _agrees(scaled_while, out.ast)

    def test_it_hoists_a_chain_in_one_pass(self):
        out = _hoisted(chained)
        assert _body_names(out.ast) == {'acc'}
        assert {'a', 'b'} <= _block_names(out.ast.body)
        assert _agrees(chained, out.ast)

    @pytest.mark.parametrize('func', REFUSED, ids=lambda f: f.name)
    def test_it_leaves_a_refused_loop_alone(self, func):
        assert _hoisted(func).ast.is_equiv(func.ast)

    @pytest.mark.parametrize('func', REFUSED_RUNNABLE, ids=lambda f: f.name)
    def test_a_refused_loop_still_computes_what_it_did(self, func):
        assert _agrees(func, _hoisted(func).ast)

    def test_a_refused_loop_is_no_site_and_says_why(self):
        assert HoistInvariant.sites(reads_the_loop_target.ast) == []
        (_, why), = HoistInvariant.refusals(reads_the_loop_target.ast)
        assert '`x` varies across iterations' in why

    def test_the_zero_trip_case_is_pinned(self):
        """The decision recorded in the plan: an invariant binding is hoisted
        out of a loop that may never run.  The value is unchanged — what moves
        is *when* the expression is evaluated, not what the function returns."""
        out = _hoisted(scaled_sum)
        assert 'c' in _block_names(out.ast.body)
        assert repr(fp.Function(out.ast, runtime=scaled_sum.runtime)([])) == repr(
            scaled_sum([])
        )


class TestTheQueryOnTheMotivatingExample:

    def test_it_finds_the_scale(self):
        out = TestRescaleFixedOutput._schedule(TestRescaleFixedOutput.fused_sum)
        assert '_k' in _query(out, which=2)

    def test_the_transform_takes_it_out(self):
        sched = TestRescaleFixedOutput._schedule(TestRescaleFixedOutput.fused_sum)
        out = _hoisted(sched)
        assert '_k' not in _body_names(out.ast)
        assert _agrees_by_value(sched, out.ast, values=_VALUES[1:])

    def test_anf_after_rescaling_frees_both_scales(self):
        """`to_anf` must run *after* `rescale_fixed`: it is the rescaling that
        introduces `2 ** -_k` and `2 ** _k`, and naming them is what lets this
        pass take them out.  Both leave in a single pass, since each hoisted
        binding counts as invariant for the ones after it."""
        f = st.rescale_fixed(st.comp_to_loop(st.fuse(TestRescaleFixedOutput.fused_sum)))
        f = st.simplify(st.to_anf(f))
        out = _hoisted(f)
        body = ' '.join(_text(f, out.ast).split())
        assert '(2 ** _k)' not in _loop_bodies_text(out.ast)
        assert '(2 ** _k)' in body
        assert _agrees_by_value(f, out.ast, values=_VALUES[1:])


def _loop_bodies_text(ast) -> str:
    """The formatted text of every loop body, for asserting what is *not* in
    one without pinning the surrounding temporary names."""
    return ' '.join(
        ' '.join(s.format().split()) for loop in _loops(ast) for s in loop.body.stmts
    )
