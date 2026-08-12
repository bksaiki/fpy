"""Unit tests for :func:`fpy2.strategies.close`."""

import pytest

import fpy2 as fp

from fpy2.ast.fpyast import Assign
from fpy2.strategies import close, inline, simplify
from fpy2.transform.free_var_elim import unclosed_data_free_vars


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
        assert unclosed_data_free_vars(out.ast) == []
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
        # the guiding loop schedule: inlining `_mul_add` makes SCALE
        # free in `_dot`; `close` then bakes it
        sched = close(inline(_dot))
        assert unclosed_data_free_vars(sched.ast) == []
        # `inline` leaves the inlined callee's name as a stale free-var
        # entry (the body no longer references it); a `Function` has no
        # literal form, so `close` leaves it and DCE prunes it
        assert {str(v) for v in sched.ast.free_vars} == {'_mul_add'}
        assert not simplify(sched).ast.free_vars
        xs, ys = [1.0, 2.0, 3.0], [0.5, -1.5, 2.5]
        assert _dot(xs, ys) == sched(xs, ys)

    def test_type_error(self):
        with pytest.raises(TypeError):
            close(_scaled.ast)  # type: ignore[arg-type]  # FuncDef, not Function
