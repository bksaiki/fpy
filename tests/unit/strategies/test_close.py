"""Unit tests for :func:`fpy2.strategies.close`."""


import fpy2 as fp

from fpy2.ast import Assign
from fpy2.strategies import close, inline


SCALE = 2.0
WEIGHTS = (0.25, 0.75)
MY_CTX = fp.IEEEContext(8, 32)


@fp.fpy
def _scaled(x: fp.Real) -> fp.Real:
    return SCALE * x


@fp.fpy
def _weighted(x: fp.Real, y: fp.Real) -> fp.Real:
    a, b = WEIGHTS
    return a * x + b * y


@fp.fpy
def _ctx_capture(x: fp.Real) -> fp.Real:
    # a rounding context has no literal form — stays free
    with MY_CTX:
        return x + 1.0


@fp.fpy
def _no_free(x: fp.Real) -> fp.Real:
    return x + 1.0


@fp.fpy
def _mul_add(acc: fp.Real, x: fp.Real, y: fp.Real) -> fp.Real:
    return acc + SCALE * (x * y)


@fp.fpy
def _dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x, y in zip(xs, ys):
        acc = _mul_add(acc, x, y)  # type: ignore[assignment]
    return acc


class TestClose:

    def test_number_baked(self):
        out = close(_scaled)
        assert isinstance(out.ast.body.stmts[0], Assign)
        assert not out.ast.free_vars
        for x in (0.0, 1.5, -3.25):
            assert _scaled(x) == out(x)
        # the input is not mutated
        assert _scaled.ast.free_vars

    def test_tuple_baked(self):
        out = close(_weighted)
        assert not out.ast.free_vars
        assert _weighted(1.0, 3.0) == out(1.0, 3.0)

    def test_context_stays_free(self):
        out = close(_ctx_capture)
        assert out.ast.is_equiv(_ctx_capture.ast)
        assert out.ast.free_vars == _ctx_capture.ast.free_vars

    def test_no_free_vars_noop(self):
        out = close(_no_free)
        assert out.ast.is_equiv(_no_free.ast)

    def test_after_inline(self):
        # inlining `_mul_add` makes SCALE free in `_dot`; `close` bakes it
        sched = close(inline(_dot))
        assert not sched.ast.free_vars
        xs, ys = [1.0, 2.0, 3.0], [0.5, -1.5, 2.5]
        assert _dot(xs, ys) == sched(xs, ys)

