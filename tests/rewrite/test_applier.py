import unittest

from fpy2 import fpy, pattern, Function
from fpy2.ast import Expr, Stmt, StmtBlock, FuncDef, Fma, Var, NamedId, SimpleAssign, Call
from fpy2.rewrite.applier import Applier
from fpy2.rewrite.matcher import Matcher
from fpy2.typing import *


@pattern
def insert_fma_l(a, b, c):
    a * b + c

@pattern
def insert_fma_r(a, b, c):
    fma(a, b, c)

@fpy
def f(x, y, z):
    return x * y + z

@fpy
def g(x, y, z):
    t = x * y + z
    for x in range(10):
        t += 10 * y + z
    return t


class _ApplierTestCase(unittest.TestCase):

    def assertAstEqual(
        self,
        a: Expr | Stmt | StmtBlock | FuncDef,
        b: Expr | Stmt | StmtBlock | FuncDef
    ):
        self.assertTrue(a.is_equiv(b), f'\n### AST 1 ###\n{a.format()}\n### AST 2 ###\n{b.format()}\n')


class ApplierExprTestCase(_ApplierTestCase):
    """Testing `Applier` for expressions"""

    def test_fma_example_1(self):
        assert isinstance(f, Function)
        m = Matcher(insert_fma_l)
        a = Applier(insert_fma_r)
        matches = m.match(f)
        f2 = a.apply(matches[0])
        self.assertAstEqual(f2, Fma(
            Var(NamedId('x'), None),
            Var(NamedId('y'), None),
            Var(NamedId('z'), None),
            None
        ))

    def test_fma_example_2(self):
        assert isinstance(g, Function)
        m = Matcher(insert_fma_l)
        a = Applier(insert_fma_r)
        matches = m.match(g)
        f2 = a.apply(matches[0])
        self.assertAstEqual(f2, Fma(
            Var(NamedId('x'), None),
            Var(NamedId('y'), None),
            Var(NamedId('z'), None),
            None
        ))


@fpy
def h(lst):
    sum = 0
    for n in lst:
        sum += n
    return sum

@pattern
def insert_sum_l(xs):
    y = 0
    for x in xs:
        y += x

@pattern
def insert_sum_r(xs):
    y = sum(xs)

class ApplierStmtTestCase(_ApplierTestCase):
    """Testing `Applier` for sum expressions"""

    def test_sum_example_1(self):
        assert isinstance(h, Function)
        m = Matcher(insert_sum_l)
        a = Applier(insert_sum_r)
        matches = m.match(h)
        h2 = a.apply(matches[0])
        self.assertAstEqual(
            h2, 
            StmtBlock([
                SimpleAssign(
                    NamedId('sum'),
                    Call('sum', [Var(NamedId('lst'), None)], None), None, None),
            ])
        )
