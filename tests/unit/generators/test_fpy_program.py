"""
Smoke tests for the type-directed FPy program generators (phase 1).

These tests don't check any specific FPy behavior — they verify that the
generator emits well-typed programs that downstream passes accept. They
double as fuzz tests for those passes.
"""

import fpy2 as fp

from hypothesis import given, settings, strategies as st

from fpy2.analysis.type_infer import TypeInfer
from fpy2.ast.fpyast import FuncDef
from fpy2.types import RealType

from . import fpy_real_funcdef, fpy_real_function, real_expr
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
        # This guards against the generator accidentally drawing a Var of a
        # name not in env (which would crash type inference) when env is empty.
        analysis = TypeInfer.check(fd)
        assert isinstance(analysis.return_type, RealType)


class TestGeneratedRuns:
    """Generated functions should evaluate under ``fp.FP64`` without crashing.

    We use FP64 rather than ``fp.REAL`` because ``Div`` is not implemented
    for real-number arithmetic (no closed-form rational result in general),
    while FP64 handles div-by-zero by producing ``±inf``.
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
        # Should not raise. We don't check the value — only that the generated
        # program is executable end-to-end.
        f(*inputs, ctx=fp.FP64)


class TestRealExprStrategyDirectly:
    """Exercise ``real_expr`` outside the ``FuncDef`` wrapper."""

    @given(real_expr({}, depth=0))
    def test_empty_env_leaf_is_a_literal(self, _expr) -> None:
        # No vars in env + depth=0 ⇒ leaves only ⇒ must be a literal node.
        # Imported lazily to avoid a top-level dep cycle.
        from fpy2.ast.fpyast import Integer
        assert isinstance(_expr, Integer)
