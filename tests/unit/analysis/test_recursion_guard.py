"""
The recursion guard shared by the call-graph-walking analyses and
transforms (``TypeInfer``, ``FormatInfer``, ``Purity``, ``FuncInline``).

FPy forbids recursion, so each runs a :class:`CallGraph` acyclicity
check at its entry point before lazily descending into callees.  The
parser rejects forward references, so a cycle can't be built through
``@fp.fpy`` decoration; we patch a ``Call.fn`` to close one (the case
that matters for programmatically-built ASTs).
"""

import pytest

import fpy2 as fp

from fpy2.analysis import (
    CallGraphError, FormatInfer, Purity, TypeInfer, TypeInferError,
)
from fpy2.ast import DefaultVisitor
from fpy2.transform import FuncInline


def _first_call(func):
    """The first ``Call`` node in a decorated function's body."""
    calls = []

    class _C(DefaultVisitor):
        def _visit_call(self, e, ctx):
            calls.append(e)
            super()._visit_call(e, ctx)

    _C()._visit_function(func.ast, None)
    return calls[0]


def _make_self_cycle():
    @fp.fpy
    def leaf(x: fp.Real) -> fp.Real:
        return x + 1

    @fp.fpy
    def m(x: fp.Real) -> fp.Real:
        return leaf(x)

    # Redirect m's only call from `leaf` to `m`: m -> m.
    _first_call(m).fn = m
    return m


class TestTypeInferGuard:
    def test_recursion_raises_type_infer_error(self):
        m = _make_self_cycle()
        with pytest.raises(TypeInferError):
            TypeInfer.check(m.ast)

    def test_acyclic_multi_function_still_checks(self):
        @fp.fpy
        def g(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return g(x) * 2

        # no exception
        TypeInfer.check(f.ast)


class TestFormatInferGuard:
    def test_recursion_raises_call_graph_error(self):
        m = _make_self_cycle()
        # FormatInfer has no dedicated error type; the descriptive
        # CallGraphError propagates directly.
        with pytest.raises(CallGraphError):
            FormatInfer.analyze(m.ast)

    def test_acyclic_multi_function_still_infers(self):
        @fp.fpy
        def g(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return g(x) * 2

        # no exception
        FormatInfer.analyze(f.ast)


class TestPurityGuard:
    def test_recursion_raises_call_graph_error(self):
        m = _make_self_cycle()
        # Purity returns a bool; a structural cycle is an error, not a
        # purity verdict — CallGraphError propagates.
        with pytest.raises(CallGraphError):
            Purity.analyze(m.ast)

    def test_acyclic_multi_function_still_analyzes(self):
        @fp.fpy
        def g(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return g(x) * 2

        assert Purity.analyze(f.ast) is True


class TestFuncInlineGuard:
    def test_recursion_raises_call_graph_error(self):
        m = _make_self_cycle()
        with pytest.raises(CallGraphError):
            FuncInline.apply(m.ast)
