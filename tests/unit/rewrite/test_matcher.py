import unittest

from fpy2 import fpy, pattern, Function
from fpy2.ast import Add, Var, NamedId, Integer, Expr, Stmt, StmtBlock, FuncDef
from fpy2.rewrite.matcher import Matcher


@pattern
def insert_fma_l(a, b, c):
    a * b + c

@fpy
def f(x, y, z):
    return x * y + z

@fpy
def g(x, y, z):
    return y * z + x

@fpy
def h(x, y, z):
    t = x * y + z
    for x in range(10):
        t += 10 * y + z
    return t


class _MatcherTestCase(unittest.TestCase):

    def assertAstEqual(
        self,
        a: Expr | Stmt | StmtBlock | FuncDef,
        b: Expr | Stmt | StmtBlock | FuncDef
    ):
        self.assertTrue(a.is_equiv(b), f'\n### AST 1 ###\n{a.format()}\n### AST 2 ###\n{b.format()}\n')

class MatchExprTestCase(_MatcherTestCase):
    """Testing `Matcher.match()` for expressions"""

    def test_compare(self):
        @pattern
        def compare_pattern1(a, b):
            a == b

        @fpy
        def f1(x, y):
            return x == y

        assert isinstance(f1, Function)
        m = Matcher(compare_pattern1)
        matches = m.match(f1)
        self.assertEqual(len(matches), 1)
        self.assertAstEqual(matches[0].subst['a'], Var(NamedId('x'), None))
        self.assertAstEqual(matches[0].subst['b'], Var(NamedId('y'), None))


    def test_fma_example_1(self):
        assert isinstance(f, Function)
        m = Matcher(insert_fma_l)
        matches = m.match(f)
        self.assertEqual(len(matches), 1)
        self.assertAstEqual(matches[0].subst['a'], Var(NamedId('x'), None))
        self.assertAstEqual(matches[0].subst['b'], Var(NamedId('y'), None))
        self.assertAstEqual(matches[0].subst['c'], Var(NamedId('z'), None))

    def test_fma_example_2(self):
        assert isinstance(g, Function)
        m = Matcher(insert_fma_l)
        matches = m.match(g)
        self.assertEqual(len(matches), 1)
        self.assertAstEqual(matches[0].subst['a'], Var(NamedId('y'), None))
        self.assertAstEqual(matches[0].subst['b'], Var(NamedId('z'), None))
        self.assertAstEqual(matches[0].subst['c'], Var(NamedId('x'), None))

    def test_fma_example_3(self):
        assert isinstance(h, Function)
        m = Matcher(insert_fma_l)
        matches = m.match(h)
        self.assertEqual(len(matches), 2)

        m0 = matches[0]
        self.assertAstEqual(m0.subst['a'], Var(NamedId('x'), None))
        self.assertAstEqual(m0.subst['b'], Var(NamedId('y'), None))
        self.assertAstEqual(m0.subst['c'], Var(NamedId('z'), None))

        m1 = matches[1]
        self.assertAstEqual(m1.subst['a'], Integer(10, None))
        self.assertAstEqual(m1.subst['b'], Var(NamedId('y'), None))
        self.assertAstEqual(m1.subst['c'], Var(NamedId('z'), None))


@fpy
def f2(lst):
    sum = 0
    for n in lst:
        sum += n
    return sum

@pattern
def insert_sum_l(xs):
    y = 0
    for x in xs:
        y += x


class MatchStmtTestCase(_MatcherTestCase):
    """Testing `Matcher.match()` for statements"""

    def test_unpack(self):
        @pattern
        def unpack_pattern1():
            a, b = 1, 2

        @pattern
        def unpack_pattern2():
            a, _ = 1, 2

        @fpy
        def f1():
            x, y = 1, 2
            return x + y

        assert isinstance(f1, Function)
        m = Matcher(unpack_pattern1)
        matches = m.match(f1)
        self.assertEqual(len(matches), 1)
        self.assertAstEqual(matches[0].subst['a'], Var(NamedId('x'), None))
        self.assertAstEqual(matches[0].subst['b'], Var(NamedId('y'), None))

        m = Matcher(unpack_pattern2)
        matches = m.match(f1)
        self.assertEqual(len(matches), 1)
        self.assertAstEqual(matches[0].subst['a'], Var(NamedId('x'), None))

    def test_if(self):
        @pattern
        def if_pattern1(c1, c2):
            if True:
                t = c1
            else:
                t = c2

        @fpy
        def f():
            if True:
                x = 1
            else:
                x = 2
            return x

        assert isinstance(f, Function)
        m = Matcher(if_pattern1)
        matches = m.match(f)
        self.assertEqual(len(matches), 1)
        self.assertAstEqual(matches[0].subst['t'], Var(NamedId('x'), None))
        self.assertAstEqual(matches[0].subst['c1'], Integer(1, None))
        self.assertAstEqual(matches[0].subst['c2'], Integer(2, None))

    def test_while(self):
        @pattern
        def while_pattern1(N, e):
            t = 0
            while t < N:
                t = e

        @fpy
        def f():
            x = 0
            while x < 100:
                x += 1
            return x

        assert isinstance(f, Function)
        m = Matcher(while_pattern1)
        matches = m.match(f)
        self.assertEqual(len(matches), 1)
        self.assertAstEqual(matches[0].subst['t'], Var(NamedId('x'), None))
        self.assertAstEqual(matches[0].subst['N'], Integer(100, None))
        self.assertAstEqual(matches[0].subst['e'], Add(Var(NamedId('x'), None), Integer(1, None), None))


    def test_fma_example_1(self):
        assert isinstance(f2, Function)
        m = Matcher(insert_sum_l)
        matches = m.match(f2)

        self.assertEqual(len(matches), 1)
        self.assertAstEqual(matches[0].subst['y'], Var(NamedId('sum'), None))
        self.assertAstEqual(matches[0].subst['x'], Var(NamedId('n'), None))
        self.assertAstEqual(matches[0].subst['xs'], Var(NamedId('lst'), None))
