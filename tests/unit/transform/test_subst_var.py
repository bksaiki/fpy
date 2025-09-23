"""
Unit tests for variable substitution.
"""

import fpy2 as fp
import unittest


class TestSubstVar(unittest.TestCase):

    def test_example1(self):
        @fp.fpy
        def test():
            x = 1
            return x + 1

        @fp.fpy
        def test_expect():
            x = 1
            return 2 + 1

        subst: dict[fp.analysis.AssignDef, fp.ast.Expr] = {}
        def_use = fp.analysis.DefineUse.analyze(test.ast)
        for d in def_use.name_to_defs[fp.ast.NamedId('x')]:
            if isinstance(d, fp.analysis.AssignDef):
                subst[d] = fp.ast.Integer(2, None)
        assert len(subst) == 1

        f = fp.transform.SubstVar.apply(test.ast, def_use, subst)
        f.name = test_expect.name
        self.assertTrue(f.is_equiv(test_expect.ast), f'expect:\n{test_expect.format()}\nactual:\n{f.format()}')


    def test_example2(self):
        @fp.fpy
        def test(t):
            x = 1
            if t < 0:
                z = x + 1
            else:
                z = x - 1
            return x + z

        @fp.fpy
        def test_expect(t):
            x = 1
            if t < 0:
                z = 2 + 1
            else:
                z = 2 - 1
            return 2 + z

        subst: dict[fp.analysis.AssignDef, fp.ast.Expr] = {}
        def_use = fp.analysis.DefineUse.analyze(test.ast)
        for d in def_use.name_to_defs[fp.ast.NamedId('x')]:
            if isinstance(d, fp.analysis.AssignDef):
                subst[d] = fp.ast.Integer(2, None)
        assert len(subst) == 1

        f = fp.transform.SubstVar.apply(test.ast, def_use, subst)
        f.name = test_expect.name
        self.assertTrue(f.is_equiv(test_expect.ast), f'expect:\n{test_expect.format()}\nactual:\n{f.format()}')

    def test_example3(self):
        @fp.fpy
        def test(t):
            x = 1
            if t < 0:
                z = x + 1
            else:
                z = x - 1
            x = z
            return x + z

        @fp.fpy
        def test_expect(t):
            x = 1
            if t < 0:
                z = 2 + 1
            else:
                z = 2 - 1
            x = z
            return x + z

        subst: dict[fp.analysis.AssignDef, fp.ast.Expr] = {}
        def_use = fp.analysis.DefineUse.analyze(test.ast)
        for d in def_use.name_to_defs[fp.ast.NamedId('x')]:
            if isinstance(d, fp.analysis.AssignDef) and d.prev is None:
                subst[d] = fp.ast.Integer(2, None)
        assert len(subst) == 1

        f = fp.transform.SubstVar.apply(test.ast, def_use, subst)
        f.name = test_expect.name
        self.assertTrue(f.is_equiv(test_expect.ast), f'expect:\n{test_expect.format()}\nactual:\n{f.format()}')
