"""
Unit tests for function inlining
"""

import fpy2 as fp

from fpy2.ast import Call
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function


def _count_fpy_calls(ast) -> int:
    """Number of remaining calls to user-defined FPy functions."""
    n = 0

    class _C(DefaultVisitor):
        def _visit_call(self, e, ctx):
            nonlocal n
            if isinstance(e.fn, Function):
                n += 1
            super()._visit_call(e, ctx)

    _C()._visit_function(ast, None)
    return n


class TestFuncInline():

    def assertASTEquiv(self, f: fp.ast.FuncDef, g: fp.ast.FuncDef, msg: str = ''):
        assert f.is_equiv(g), f'AST not equivalent:\nexpect:\n{g.format()}\nactual:\n{f.format()}\n{msg}'

    def test_example1(self):
        @fp.fpy
        def f():
            return fp.rational(1, 3) + fp.rational(1, 3)

        @fp.fpy
        def g():
            return fp.round(2) * f()

        @fp.fpy
        def expect():
            t = (fp.rational(1, 3) + fp.rational(1, 3))
            return (fp.round(2) * t)


        h = fp.transform.FuncInline.apply(g.ast, recursive=False)
        h.name = expect.name
        self.assertASTEquiv(h, expect.ast, 'inlining failed')

    def test_example2(self):
        @fp.fpy
        def f(x: fp.Real):
            return x + fp.rational(1, 3)

        @fp.fpy
        def g(x: fp.Real):
            return fp.round(2) * f(x)

        @fp.fpy
        def expect(x: fp.Real):
            x2 = x
            t = (x2 + fp.rational(1, 3))
            return (fp.round(2) * t)

        h = fp.transform.FuncInline.apply(g.ast, recursive=False)
        h.name = expect.name
        self.assertASTEquiv(h, expect.ast, 'inlining failed')


    def test_example3(self):
        @fp.fpy(ctx=fp.REAL)
        def select_ctx(xs: list[fp.Real], p: fp.Real):
            e, _ = fp.libraries.core.max_e(xs)
            n = e - p
            return fp.MPFixedContext(n, fp.RM.RTZ)

        @fp.fpy
        def bna(xs: list[fp.Real], p: fp.Real):
            with select_ctx(xs, p):
                return [fp.round(x) for x in xs]

        @fp.fpy
        def expect(xs: list[fp.Real], p: fp.Real):
            xs4 = xs
            p5 = p
            with fp.REAL:
                e, _ = fp.libraries.core.max_e(xs4)
                n = (e - p5)
                t = fp.MPFixedContext(n, fp.RM.RTZ)
            with t:
                return [fp.round(x) for x in xs]


        h = fp.transform.FuncInline.apply(bna.ast, recursive=False)
        h.name = expect.name

        h = fp.transform.ConstFold.apply(h, enable_op=False)
        e = fp.transform.ConstFold.apply(expect.ast, enable_op=False)
        self.assertASTEquiv(h, e, 'inlining failed')

    def test_recursive_bottom_up(self):
        """`recursive=True` flattens a multi-level call graph with a
        shared callee, leaving no user-function calls and preserving
        the computed value."""

        @fp.fpy
        def c(x: fp.Real) -> fp.Real:
            return x + 1

        @fp.fpy
        def b(x: fp.Real) -> fp.Real:
            return c(x) * 2

        @fp.fpy
        def a(x: fp.Real) -> fp.Real:
            # `c` is shared: reached via `b` and directly.
            return b(x) + c(x)

        inlined = fp.transform.FuncInline.apply(a.ast, recursive=True)
        # everything user-defined is inlined away
        assert _count_fpy_calls(inlined) == 0
        # value is preserved
        inlined_fn = Function(inlined)
        for xv in (0.0, 1.5, -3.25, 10.0):
            assert a(xv) == inlined_fn(xv)
