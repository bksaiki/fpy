"""
Unit tests for function inlining
"""

import fpy2 as fp
import unittest

class TestFuncInline(unittest.TestCase):

    def assertASTEquiv(self, f: fp.ast.FuncDef, g: fp.ast.FuncDef, msg: str = ''):
        self.assertTrue(
            f.is_equiv(g),
            f'AST not equivalent:\nexpect:\n{g.format()}\nactual:\n{f.format()}\n{msg}'
        )

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


        h = fp.transform.FuncInline.apply(g.ast)
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

        h = fp.transform.FuncInline.apply(g.ast)
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


        h = fp.transform.FuncInline.apply(bna.ast)
        h.name = expect.name

        h = fp.transform.ConstFold.apply(h, enable_op=False)
        e = fp.transform.ConstFold.apply(expect.ast, enable_op=False)
        self.assertASTEquiv(h, e, 'inlining failed')
