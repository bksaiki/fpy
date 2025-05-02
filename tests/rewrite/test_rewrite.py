import unittest

from fpy2 import *
from fpy2.typing import *
from fpy2.ast import *
from fpy2.rewrite.rewrite import Rewrite


@pattern
def insert_fma_l(a, b, c):
    a * b + c

@pattern
def insert_fma_r(a, b, c):
    fma(a, b, c)

rw_fma = Rewrite(insert_fma_l, insert_fma_r)

@pattern
def insert_sum_l(xs):
    y = 0
    for x in xs:
        y += x

@pattern
def insert_sum_r(xs):
    y = sum(xs)

@pattern
def expand_sum_bad(xs):
    y = 0
    for _ in xs:
        y += 0
    for _ in xs:
        y *= 1
    for x in xs:
        y += x


rw_sum = Rewrite(insert_sum_l, insert_sum_r)
rw_sum_bad = Rewrite(insert_sum_l, expand_sum_bad)

@fpy
def f(x, y, z):
    t = x * y + z
    for x in range(10):
        t += 10 * y + z
    return t

@fpy
def f1(x, y, z):
    t = fma(x, y, z)
    for x in range(10):
        t += 10 * y + z
    return t

@fpy
def f2(x, y, z):
    t = fma(x, y, z)
    for x in range(10):
        t += fma(10, y, z)
    return t

@fpy
def g(lst):
    acc = 0
    for n in lst:
        acc += n
    return acc

@fpy
def g1(lst):
    acc = sum(lst)
    return acc

@fpy
def g2(lst):
    acc = 0
    for _ in lst:
        acc += 0
    for _ in lst:
        acc *= 1
    for n in lst:
        acc += n
    return acc


@pattern
def unroll_for_l(N):
    t = 0
    for i in range(N):
        t += i

@pattern
def unroll_for_r(N):
    t = 0
    t += 0
    for i in range(N - 1):
        t += i

rw_unroll_for = Rewrite(unroll_for_l, unroll_for_r)

@fpy
def h(x):
    x = 0
    for c in range(10):
        x += c
    return x

@fpy
def h1(x):
    x = 0
    x += 0
    for c in range(10 - 1):
        x += c
    return x

@pattern
def unroll_while_l(t, e):
    while t > 0:
        t = e

@pattern
def unroll_while_r(t, e):
    t = e
    while t > 0:
        t = e

rw_unroll_while = Rewrite(unroll_while_l, unroll_while_r)


@fpy
def k(N):
    x = N
    while x > 0:
        x -= x / 2
    return x

@fpy
def k1(N):
    x = N
    x -= x / 2
    while x > 0:
        x -= x / 2
    return x


class RewriteTestCase(unittest.TestCase):
    """Testing `Pattern` parsing for examples"""

    def assertAstEqual(self, a: Ast, b: Ast):
        self.assertEqual(a, b, f'\n### AST 1 ###\n{a.format()}\n### AST 2 ###\n{b.format()}\n')

    def test_fma_example1(self):
        assert isinstance(f, Function)
        assert isinstance(f1, Function)

        f_rw = rw_fma.apply(f)
        self.assertIsInstance(f_rw, Function)
        self.assertAstEqual(f_rw.ast.body, f1.ast.body)

    def test_fma_example2(self):
        assert isinstance(f, Function)
        assert isinstance(f2, Function)

        f_rw = rw_fma.apply(f)
        f_rw = rw_fma.apply(f_rw)
        self.assertIsInstance(f_rw, Function)
        self.assertAstEqual(f_rw.ast.body, f2.ast.body)

    def test_fma_example3(self):
        assert isinstance(f, Function)
        assert isinstance(f2, Function)

        f_rw = rw_fma.apply_all(f)
        self.assertIsInstance(f_rw, Function)
        self.assertAstEqual(f_rw.ast.body, f2.ast.body)

    def test_sum_example1(self):
        assert isinstance(g, Function)
        assert isinstance(g1, Function)

        g_rw = rw_sum.apply(g)
        self.assertIsInstance(g_rw, Function)
        self.assertAstEqual(g_rw.ast.body, g1.ast.body)

    def test_sum_example2(self):
        assert isinstance(g, Function)
        assert isinstance(g2, Function)

        g_rw = rw_sum_bad.apply(g)
        self.assertIsInstance(g_rw, Function)
        self.assertAstEqual(g_rw.ast.body, g2.ast.body)

    def test_unroll_for_example1(self):
        assert isinstance(h, Function)
        assert isinstance(h1, Function)

        h_rw = rw_unroll_for.apply(h)
        self.assertIsInstance(h_rw, Function)
        self.assertAstEqual(h_rw.ast.body, h1.ast.body)

    def test_unroll_while_example1(self):
        assert isinstance(k, Function)
        assert isinstance(k1, Function)

        k_rw = rw_unroll_while.apply(k)
        self.assertIsInstance(k_rw, Function)
        self.assertAstEqual(k_rw.ast.body, k1.ast.body)
