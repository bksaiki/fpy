"""Unit tests for :func:`fpy2.strategies.lift_context`."""

import pytest

import fpy2 as fp

from fpy2.ast.fpyast import Assign, ContextStmt, ForStmt, Var
from fpy2.strategies import lift_context


def _find_stmt(ast, node_type):
    """Return the first statement of *node_type* in *ast*, descending
    into compound-statement bodies."""
    def walk(stmts):
        for s in stmts:
            if isinstance(s, node_type):
                return s
            body = getattr(s, 'body', None)
            if body is not None and hasattr(body, 'stmts'):
                hit = walk(body.stmts)
                if hit is not None:
                    return hit
        return None
    return walk(ast.body.stmts)


@fp.fpy
def _with_ctor(x: fp.Real) -> fp.Real:
    with fp.IEEEContext(11, 64):
        return x + 1.0


@fp.fpy
def _ctor_in_loop(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x in xs:
        with fp.IEEEContext(8, 32):
            acc = acc + x
    return acc


@fp.fpy
def _ctx_param(x: fp.Real, ctx: fp.Context) -> fp.Real:
    # already a variable — nothing to lift
    with ctx:
        return x + 1.0


@fp.fpy
def _no_ctx(x: fp.Real) -> fp.Real:
    return x + 1.0


class TestLiftContext:

    def test_ctor_lifted(self):
        out = lift_context(_with_ctor)
        assert isinstance(out.ast.body.stmts[0], Assign)
        with_stmt = _find_stmt(out.ast, ContextStmt)
        assert isinstance(with_stmt.ctx, Var)
        for x in (0.0, 1.5, -3.25):
            assert _with_ctor(x) == out(x)
        # the input is not mutated
        assert not isinstance(_with_ctor.ast.body.stmts[0], Assign)

    def test_hoisted_out_of_loop(self):
        out = lift_context(_ctor_in_loop)
        # the binding precedes the loop; the loop body references it
        assert isinstance(out.ast.body.stmts[0], Assign)
        loop = _find_stmt(out.ast, ForStmt)
        with_stmt = _find_stmt(out.ast, ContextStmt)
        assert loop is not None and isinstance(with_stmt.ctx, Var)
        xs = [1.0, 2.5, -0.5]
        assert _ctor_in_loop(xs) == out(xs)

    def test_var_ctx_noop(self):
        out = lift_context(_ctx_param)
        assert out.ast.is_equiv(_ctx_param.ast)

    def test_no_ctx_noop(self):
        out = lift_context(_no_ctx)
        assert out.ast.is_equiv(_no_ctx.ast)

    def test_idempotent(self):
        once = lift_context(_with_ctor)
        twice = lift_context(once)
        assert once.ast.is_equiv(twice.ast)

    def test_type_error(self):
        with pytest.raises(TypeError):
            lift_context(_with_ctor.ast)  # type: ignore[arg-type]  # FuncDef, not Function
