"""Unit tests for the :class:`fpy2.transform.LiftContext` transform."""


from typing import TypeVar

import fpy2 as fp

from fpy2.ast import Assign, ContextStmt, ForStmt, FuncDef, Stmt, Var
from fpy2.transform import LiftContext
from fpy2.transform.path import walk_stmts

_S = TypeVar('_S', bound=Stmt)


def _find_stmt(ast: FuncDef, node_type: type[_S]) -> _S:
    """The first statement of *node_type* in *ast*."""
    return next(s for _, s in walk_stmts(ast) if isinstance(s, node_type))


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
        out = LiftContext.apply(_with_ctor.ast)
        assert isinstance(out.body.stmts[0], Assign)
        with_stmt = _find_stmt(out, ContextStmt)
        assert isinstance(with_stmt.ctx, Var)
        for x in (0.0, 1.5, -3.25):
            assert _with_ctor(x) == _with_ctor.with_ast(out)(x)
        # the input is not mutated
        assert not isinstance(_with_ctor.ast.body.stmts[0], Assign)

    def test_hoisted_out_of_loop(self):
        out = LiftContext.apply(_ctor_in_loop.ast)
        # the binding precedes the loop; the loop body references it
        assert isinstance(out.body.stmts[0], Assign)
        loop = _find_stmt(out, ForStmt)
        with_stmt = _find_stmt(out, ContextStmt)
        assert loop is not None and isinstance(with_stmt.ctx, Var)
        xs = [1.0, 2.5, -0.5]
        assert _ctor_in_loop(xs) == _ctor_in_loop.with_ast(out)(xs)

    def test_var_ctx_noop(self):
        assert LiftContext.apply(_ctx_param.ast).is_equiv(_ctx_param.ast)

    def test_no_ctx_noop(self):
        assert LiftContext.apply(_no_ctx.ast).is_equiv(_no_ctx.ast)

    def test_idempotent(self):
        once = LiftContext.apply(_with_ctor.ast)
        assert LiftContext.apply(once).is_equiv(once)
