"""
Tests for the :class:`fpy2.analysis.CallGraph` analysis.
"""

import pytest

import fpy2 as fp

from fpy2.analysis import CallGraph, CallGraphError
from fpy2.ast import DefaultVisitor


def _cg(func):
    """Build the call graph from a decorated ``Function``."""
    return CallGraph.analyze(func.ast)


def _first_call(func):
    """The first ``Call`` node in a decorated function's body."""
    calls = []

    class _C(DefaultVisitor):
        def _visit_call(self, e, ctx):
            calls.append(e)
            super()._visit_call(e, ctx)

    _C()._visit_function(func.ast, None)
    return calls[0]


class TestStructure:
    """Nodes, edges, and reverse edges."""

    def test_single_function_no_calls(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return x + 1

        cg = _cg(f)
        assert cg.nodes == {f.ast}
        assert cg.callees_of(f.ast) == []
        assert cg.callers_of(f.ast) == []
        assert cg.order == [f.ast]
        assert len(cg) == 1
        assert f.ast in cg

    def test_linear_chain(self):
        @fp.fpy
        def c(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def b(x: fp.Real) -> fp.Real:
            return c(x) * 2

        @fp.fpy
        def a(x: fp.Real) -> fp.Real:
            return b(x) - 1

        cg = _cg(a)
        assert cg.nodes == {a.ast, b.ast, c.ast}
        assert cg.callees_of(a.ast) == [b.ast]
        assert cg.callees_of(b.ast) == [c.ast]
        assert cg.callees_of(c.ast) == []
        assert cg.callers_of(c.ast) == [b.ast]
        assert cg.callers_of(b.ast) == [a.ast]
        assert cg.callers_of(a.ast) == []

    def test_diamond(self):
        @fp.fpy
        def d(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def b(x: fp.Real) -> fp.Real:
            return d(x) * 2

        @fp.fpy
        def c(x: fp.Real) -> fp.Real:
            return d(x) - 2

        @fp.fpy
        def a(x: fp.Real) -> fp.Real:
            return b(x) + c(x)

        cg = _cg(a)
        assert cg.nodes == {a.ast, b.ast, c.ast, d.ast}
        # callees in source order
        assert cg.callees_of(a.ast) == [b.ast, c.ast]
        # d reached from both b and c, deduplicated; both recorded as callers
        assert set(cg.callers_of(d.ast)) == {b.ast, c.ast}

    def test_duplicate_calls_dedup_in_callees_but_not_sites(self):
        @fp.fpy
        def g(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return g(x) * g(x + 1)

        cg = _cg(f)
        # one edge, two call sites
        assert cg.callees_of(f.ast) == [g.ast]
        assert len(cg.call_sites[f.ast]) == 2

    def test_nested_call_in_argument_discovered(self):
        @fp.fpy
        def inner(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def outer(x: fp.Real) -> fp.Real:
            return inner(inner(x))

        cg = _cg(outer)
        assert cg.nodes == {outer.ast, inner.ast}
        assert cg.callees_of(outer.ast) == [inner.ast]
        assert len(cg.call_sites[outer.ast]) == 2

    def test_nested_call_in_kwarg_value_discovered(self):
        # ``DefaultVisitor._visit_call`` only walks positional args; the
        # collector must also descend into keyword-argument values, or a
        # callee reached only through a kwarg is missed entirely.
        @fp.fpy
        def leaf(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def mid(x: fp.Real) -> fp.Real:
            return x * 2

        @fp.fpy
        def top(x: fp.Real) -> fp.Real:
            # `leaf` is reachable ONLY through the kwarg value.
            return mid(x=leaf(x))

        cg = _cg(top)
        assert cg.nodes == {top.ast, mid.ast, leaf.ast}
        assert cg.callees_of(top.ast) == [mid.ast, leaf.ast]
        assert len(cg.call_sites[top.ast]) == 2


class TestLeavesFirst:
    """The ``order`` field / iteration is callee-before-caller."""

    def test_order_chain(self):
        @fp.fpy
        def c(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def b(x: fp.Real) -> fp.Real:
            return c(x) * 2

        @fp.fpy
        def a(x: fp.Real) -> fp.Real:
            return b(x) - 1

        cg = _cg(a)
        assert cg.order == [c.ast, b.ast, a.ast]
        # iteration matches order
        assert list(cg) == [c.ast, b.ast, a.ast]

    def test_order_respects_dependencies_in_diamond(self):
        @fp.fpy
        def d(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def b(x: fp.Real) -> fp.Real:
            return d(x) * 2

        @fp.fpy
        def c(x: fp.Real) -> fp.Real:
            return d(x) - 2

        @fp.fpy
        def a(x: fp.Real) -> fp.Real:
            return b(x) + c(x)

        order = _cg(a).order
        # every callee precedes its caller
        assert order.index(d.ast) < order.index(b.ast)
        assert order.index(d.ast) < order.index(c.ast)
        assert order.index(b.ast) < order.index(a.ast)
        assert order.index(c.ast) < order.index(a.ast)
        assert order[-1] is a.ast


class TestFormat:
    """``format()`` renders an indented tree rooted at the entry."""

    def test_single_node(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return x + 1

        assert _cg(f).format() == 'f'

    def test_chain(self):
        @fp.fpy
        def c(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def b(x: fp.Real) -> fp.Real:
            return c(x) * 2

        @fp.fpy
        def a(x: fp.Real) -> fp.Real:
            return b(x) - 1

        assert _cg(a).format() == 'a\n└─ b\n   └─ c'

    def test_diamond_marks_revisit(self):
        @fp.fpy
        def d(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def b(x: fp.Real) -> fp.Real:
            return d(x) * 2

        @fp.fpy
        def c(x: fp.Real) -> fp.Real:
            return d(x) - 2

        @fp.fpy
        def a(x: fp.Real) -> fp.Real:
            return b(x) + c(x)

        assert _cg(a).format() == (
            'a\n'
            '├─ b\n'
            '│  └─ d\n'
            '└─ c\n'
            '   └─ d (*)'
        )


class TestDot:
    """``dot()`` renders a Graphviz digraph with collision-safe ids."""

    def test_diamond(self):
        @fp.fpy
        def d(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def b(x: fp.Real) -> fp.Real:
            return d(x) * 2

        @fp.fpy
        def c(x: fp.Real) -> fp.Real:
            return d(x) - 2

        @fp.fpy
        def a(x: fp.Real) -> fp.Real:
            return b(x) + c(x)

        out = _cg(a).dot()
        assert out.startswith('digraph call_graph {')
        assert out.rstrip().endswith('}')
        # one node declaration per function
        assert out.count('[label=') == 4
        # one edge per (deduplicated) call-graph edge: a->b, a->c, b->d, c->d
        assert out.count(' -> ') == 4
        # labels are the function names
        for name in ('a', 'b', 'c', 'd'):
            assert f'[label="{name}"]' in out


class TestExternalLeaves:
    """Primitives / builtins / context constructors are not nodes."""

    def test_primitive_call_excluded(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return fp.sqrt(x)

        cg = _cg(f)
        assert cg.nodes == {f.ast}
        assert cg.callees_of(f.ast) == []
        assert cg.call_sites[f.ast] == []


class TestRecursion:
    """FPy forbids recursion — cycles raise ``CallGraphError``.

    The parser already rejects forward references, so a cycle can't be
    built through ``@fp.fpy`` decoration; we construct one by patching a
    ``Call.fn`` to exercise the defensive guard (the case that matters
    for programmatically-built ASTs, e.g. FPCore import or transforms).
    """

    def test_direct_recursion(self):
        @fp.fpy
        def leaf(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def m(x: fp.Real) -> fp.Real:
            return leaf(x)

        # Make `m` call itself: m -> m.
        _first_call(m).fn = m

        with pytest.raises(CallGraphError):
            _cg(m)

    def test_mutual_recursion(self):
        @fp.fpy
        def leaf(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def b(x: fp.Real) -> fp.Real:
            return leaf(x)

        @fp.fpy
        def a(x: fp.Real) -> fp.Real:
            return b(x)

        # Redirect b's call from `leaf` to `a`, closing a -> b -> a.
        _first_call(b).fn = a

        with pytest.raises(CallGraphError):
            _cg(a)


class TestInputValidation:
    def test_rejects_non_funcdef(self):
        with pytest.raises(TypeError):
            CallGraph.analyze('not a FuncDef')  # type: ignore[arg-type]
