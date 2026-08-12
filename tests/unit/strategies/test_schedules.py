"""End-to-end tests for composed schedules.

These pin the two guiding schedules from
``docs/todos/scheduling-operators.md``: a loop schedule (inline the
helper, bake the captured constant, eliminate the derived iterable,
unroll, clean up) and a precision schedule (pin formats, delete the
roundings the pinned context made identities, clean up).
"""

import fpy2 as fp

from fpy2.ast import AnyOf, ContextStmt, ForeignVal, ListComp, Zip
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.number import REAL
from fpy2.strategies import (
    close,
    elim_iter,
    elim_round,
    fuse,
    inline,
    lift_context,
    monomorphize,
    simplify,
    unroll_for,
)
from fpy2.types import RealType


def _has_node(ast, predicate) -> bool:
    """True iff some statement or expression in *ast* satisfies *predicate*."""
    hit = [False]

    class _C(DefaultVisitor):
        def _visit_statement(self, stmt, ctx):
            if predicate(stmt):
                hit[0] = True
            return super()._visit_statement(stmt, ctx)

        def _visit_expr(self, e, ctx):
            if predicate(e):
                hit[0] = True
            return super()._visit_expr(e, ctx)

    _C()._visit_function(ast, None)
    return hit[0]


def _count_fpy_calls(ast) -> int:
    """Number of remaining calls to user-defined FPy functions."""
    n = [0]

    class _C(DefaultVisitor):
        def _visit_call(self, e, ctx):
            if isinstance(e.fn, Function):
                n[0] += 1
            super()._visit_call(e, ctx)

    _C()._visit_function(ast, None)
    return n[0]


SCALE = 2.0


@fp.fpy
def _mul_add(acc: fp.Real, x: fp.Real, y: fp.Real) -> fp.Real:
    return acc + SCALE * (x * y)


@fp.fpy
def _dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x, y in zip(xs, ys):
        acc = _mul_add(acc, x, y)  # type: ignore[assignment]
    return acc


@fp.fpy
def _prod3(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
    return (x * y) * z


@fp.fpy
def _any_small(xs: list[fp.Real]) -> bool:
    return any([abs(x) < 1e-6 for x in xs])


class TestLoopSchedule:

    def test_end_to_end(self):
        sched = inline(_dot)
        sched = close(sched)
        sched = elim_iter(sched)
        sched = unroll_for(sched, times=1)
        sched = simplify(sched)

        # no user-function calls, no derived iterable, no captures
        assert _count_fpy_calls(sched.ast) == 0
        assert not _has_node(sched.ast, lambda n: isinstance(n, Zip))
        assert not sched.ast.free_vars

        # runtime-equivalent, including an odd length (unroll remainder)
        for k in (0, 1, 3, 4):
            xs = [float(i + 1) for i in range(k)]
            ys = [0.5 * i - 1.0 for i in range(k)]
            assert _dot(xs, ys) == sched(xs, ys), f'diverged at len {k}'


class TestPrecisionSchedule:

    def test_end_to_end(self):
        pinned = monomorphize(_prod3, fp.FP64, [RealType(fp.FP32)] * 3)
        sched = elim_round(pinned)
        sched = lift_context(sched)
        sched = simplify(sched)

        # the inner FP32*FP32 multiply is now unrounded
        assert _has_node(
            sched.ast,
            lambda n: (
                isinstance(n, ContextStmt)
                and isinstance(n.ctx, ForeignVal)
                and n.ctx.val is REAL
            ),
        )

        for xyz in ((1.5, 2.5, 3.5), (0.1, -0.25, 4.0), (-3.0, 0.0, 1.0)):
            assert pinned(*xyz) == sched(*xyz)

    def test_fuse(self):
        sched = fuse(_any_small)
        assert not _has_node(sched.ast, lambda n: isinstance(n, (AnyOf, ListComp)))
        for xs in ([1.0, 2.0], [1.0, 1e-7], []):
            assert _any_small(xs) == sched(xs)
