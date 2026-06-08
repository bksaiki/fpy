"""
Smoke tests for the type-directed FPy program generators.

These tests don't check any specific FPy behavior — they verify that the
generator emits well-typed programs that downstream passes accept. They
double as fuzz tests for those passes.
"""

import fpy2 as fp

from hypothesis import given, settings, strategies as st

from fpy2.analysis.type_infer import TypeInfer
from fpy2.ast.fpyast import BoolVal, Expr, FuncDef, Integer
from fpy2.types import BoolType, RealType

from . import (
    bool_expr,
    expr,
    fpy_real_funcdef,
    fpy_real_function,
    real_expr,
)
from .number import real_floats


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
        from fpy2.ast.fpyast import (
            FuncMeta, IfExpr, ReturnStmt, StmtBlock,
        )
        from fpy2.env import ForeignEnv
        body = StmtBlock([ReturnStmt(
            IfExpr(cond, Integer(1, None), Integer(0, None), None),
            None,
        )])
        meta = FuncMeta(set(), None, None, {}, ForeignEnv.default())
        fd = FuncDef('f', [], body, meta)
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)


class TestTypeDispatch:
    """The ``expr`` entry point dispatches on the target type."""

    @given(expr(RealType(), {}, depth=0))
    def test_real_dispatch_yields_real_leaf(self, e: Expr) -> None:
        assert isinstance(e, Integer)

    @given(expr(BoolType(), {}, depth=0))
    def test_bool_dispatch_yields_bool_leaf(self, e: Expr) -> None:
        assert isinstance(e, BoolVal)
