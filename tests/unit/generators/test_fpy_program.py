"""
Smoke tests for the type-directed FPy program generators.

These tests don't check any specific FPy behavior — they verify that the
generator emits well-typed programs that downstream passes accept. They
double as fuzz tests for those passes.
"""

import fpy2 as fp

from hypothesis import given, settings, strategies as st

from fpy2.analysis.type_infer import TypeInfer
from fpy2.ast.fpyast import (
    BoolVal, Expr, FuncDef, FuncMeta, Integer, ListExpr, ReturnStmt,
    StmtBlock, TupleExpr,
)
from fpy2.env import ForeignEnv
from fpy2.types import BoolType, ListType, RealType, TupleType

from . import (
    bool_expr,
    expr,
    fpy_real_funcdef,
    fpy_real_function,
    list_expr,
    real_expr,
    stmt_block,
    tuple_expr,
)
from .number import real_floats


def _wrap_as_funcdef(body_expr: Expr) -> FuncDef:
    body = StmtBlock([ReturnStmt(body_expr, None)])
    meta = FuncMeta(set(), None, None, {}, ForeignEnv.default())
    return FuncDef('f', [], body, meta)


class TestGeneratedTypeChecks:
    """Every generated function should type-check to ``real -> ... -> real``."""

    @given(fpy_real_funcdef())
    def test_typechecks(self, fd: FuncDef) -> None:
        analysis = TypeInfer.check(fd)
        for at in analysis.arg_types:
            assert isinstance(at, RealType), f"arg type {at.format()} is not real"
        assert isinstance(analysis.return_type, RealType), (
            f"return type {analysis.return_type.format()} is not real"
        )

    @given(fpy_real_funcdef(num_args=st.just(0), max_depth=st.just(0)))
    def test_zero_arg_zero_depth_is_just_a_literal(self, fd: FuncDef) -> None:
        # With no params and no recursion budget, the body must be a literal.
        # Guards against the generator accidentally drawing a Var of a name
        # not in env (which would crash type inference) when env is empty.
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)


class TestGeneratedRuns:
    """Generated functions should evaluate under ``fp.FP64`` without crashing.

    FP64 (not ``fp.REAL``) because ``Div`` isn't implemented for real-number
    arithmetic; FP64 handles div-by-zero / sqrt of negative / etc. by
    producing ``±inf`` or ``nan`` rather than raising.
    """

    @given(
        fpy_real_function(num_args=st.integers(0, 3), max_depth=st.integers(0, 3)),
        st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_runs_under_FP64(self, f: fp.Function, data: st.DataObject) -> None:
        n = len(f.args)
        inputs = [data.draw(real_floats(prec_max=8, exp_min=-4, exp_max=4))
                  for _ in range(n)]
        f(*inputs, ctx=fp.FP64)


class TestRealExprStrategyDirectly:
    """Exercise ``real_expr`` outside the ``FuncDef`` wrapper."""

    @given(real_expr({}, depth=0))
    def test_empty_env_leaf_is_a_literal(self, e: Expr) -> None:
        # No vars in env + depth=0 ⇒ leaves only ⇒ must be a literal node.
        assert isinstance(e, Integer)


class TestBoolExprStrategyDirectly:
    """Exercise ``bool_expr`` outside the ``FuncDef`` wrapper."""

    @given(bool_expr({}, depth=0))
    def test_empty_env_leaf_is_a_literal(self, e: Expr) -> None:
        assert isinstance(e, BoolVal)

    @given(bool_expr({}, depth=3))
    def test_typechecks_as_bool_inside_if_expr(self, cond: Expr) -> None:
        # Wrap as `return cond if True else 0` — if `cond` doesn't type as
        # bool the surrounding function fails to type-check.
        from fpy2.ast.fpyast import IfExpr
        fd = _wrap_as_funcdef(
            IfExpr(cond, Integer(1, None), Integer(0, None), None)
        )
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)


class TestListExprStrategyDirectly:
    """Exercise ``list_expr`` and verify produced lists are well-typed."""

    @given(list_expr(RealType(), {}, depth=0))
    def test_depth0_is_a_list_literal(self, e: Expr) -> None:
        assert isinstance(e, ListExpr)
        assert len(e.elts) >= 1

    @given(list_expr(RealType(), {}, depth=3))
    def test_typechecks_as_list_real(self, e: Expr) -> None:
        fd = _wrap_as_funcdef(e)
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, ListType)
        assert isinstance(analysis.return_type.elt, RealType)

    @given(list_expr(BoolType(), {}, depth=2))
    def test_typechecks_as_list_bool(self, e: Expr) -> None:
        fd = _wrap_as_funcdef(e)
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, ListType)
        assert isinstance(analysis.return_type.elt, BoolType)


class TestTupleExprStrategyDirectly:
    """Exercise ``tuple_expr`` and verify produced tuples are well-typed."""

    @given(tuple_expr((RealType(), RealType()), {}, depth=0))
    def test_pair_of_reals(self, e: Expr) -> None:
        assert isinstance(e, TupleExpr)
        assert len(e.elts) == 2
        fd = _wrap_as_funcdef(e)
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, TupleType)
        assert all(isinstance(t, RealType) for t in analysis.return_type.elts)

    @given(tuple_expr((RealType(), BoolType()), {}, depth=2))
    def test_heterogeneous_pair_typechecks(self, e: Expr) -> None:
        fd = _wrap_as_funcdef(e)
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, TupleType)
        assert isinstance(analysis.return_type.elts[0], RealType)
        assert isinstance(analysis.return_type.elts[1], BoolType)


class TestStmtBlock:
    """Statement-block generator: ``Assign``* ``ReturnStmt``."""

    @given(stmt_block({}, RealType(), depth=2, max_assigns=3))
    def test_ends_with_return(self, block) -> None:
        from fpy2.ast.fpyast import Assign, ReturnStmt
        assert isinstance(block.stmts[-1], ReturnStmt)
        # everything before the return is an Assign in this phase
        for s in block.stmts[:-1]:
            assert isinstance(s, Assign)

    @given(stmt_block({}, RealType(), depth=2, max_assigns=3))
    def test_typechecks_when_wrapped(self, block) -> None:
        from fpy2.ast.fpyast import FuncMeta
        from fpy2.env import ForeignEnv
        fd = FuncDef('f', [], block,
                     FuncMeta(set(), None, None, {}, ForeignEnv.default()))
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)


class TestAssignsAreVisible:
    """A generated function with locals should still type-check and run.

    ``max_assigns`` is a *cap*, not a floor, so a body with zero locals is
    a valid draw — we just verify the body always type-checks regardless.
    """

    @given(fpy_real_funcdef(
        num_args=st.integers(0, 2),
        max_depth=st.integers(0, 2),
        max_assigns=st.just(3),
    ))
    def test_typechecks_with_locals_in_scope(self, fd: FuncDef) -> None:
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)


class TestRangeArgKwarg:
    """``range_arg_min`` / ``range_arg_max`` constrain ``Range1`` / ``Range2`` args."""

    @given(list_expr(
        RealType(), {}, depth=2,
        range_arg_min=42, range_arg_max=42,
    ))
    def test_pinned_range_arg_only_uses_42(self, e: Expr) -> None:
        # Walk the generated tree; every Range1/Range2 must carry Integer(42).
        from fpy2.ast.fpyast import Range1, Range2
        stack = [e]
        while stack:
            node = stack.pop()
            if isinstance(node, (Range1, Range2)):
                for arg in node.args:
                    assert isinstance(arg, Integer) and arg.val == 42, (
                        f"expected Integer(42), got {arg}"
                    )
            for attr in ('args', 'elts'):
                if hasattr(node, attr):
                    children = getattr(node, attr)
                    if isinstance(children, (list, tuple)):
                        stack.extend(c for c in children if isinstance(c, Expr))


class TestTypeDispatch:
    """The ``expr`` entry point dispatches on the target type."""

    @given(expr(RealType(), {}, depth=0))
    def test_real_dispatch_yields_real_leaf(self, e: Expr) -> None:
        assert isinstance(e, Integer)

    @given(expr(BoolType(), {}, depth=0))
    def test_bool_dispatch_yields_bool_leaf(self, e: Expr) -> None:
        assert isinstance(e, BoolVal)

    @given(expr(ListType(RealType()), {}, depth=0))
    def test_list_dispatch_yields_list_literal(self, e: Expr) -> None:
        assert isinstance(e, ListExpr)

    @given(expr(TupleType(RealType(), BoolType()), {}, depth=0))
    def test_tuple_dispatch_yields_tuple_expr(self, e: Expr) -> None:
        assert isinstance(e, TupleExpr)
        assert len(e.elts) == 2
