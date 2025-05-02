import unittest

from fpy2 import pattern
from fpy2.rewrite.pattern import Pattern
from fpy2.typing import *

class ParseSimpleTestCase(unittest.TestCase):
    """Testing `Pattern` parsing"""

    def test_const_pat(self):
        @pattern
        def const_pat():
            1
        self.assertIsInstance(const_pat, Pattern)

    def test_identity_pat(self):
        @pattern
        def identity_pat(x):
            x
        self.assertIsInstance(identity_pat, Pattern)


class ParseExampleTestCase(unittest.TestCase):
    """Testing `Pattern` parsing for examples"""

    def test_fma_example(self):
        @pattern
        def insert_mad_pat(a, b, c):
            a * b + c
        @pattern
        def insert_fma_pat(a, b, c):
            fma(a, b, c)

        self.assertIsInstance(insert_mad_pat, Pattern)
        self.assertIsInstance(insert_fma_pat, Pattern)

    def test_sum_example(self):
        @pattern
        def insert_sum_pat(xs):
            y = 0
            for x in xs:
                y += x
        @pattern
        def insert_sum_op_pat(xs):
            y = sum(xs)

        self.assertIsInstance(insert_sum_pat, Pattern)
        self.assertIsInstance(insert_sum_op_pat, Pattern)

    def test_sum_bad_example(self):
        @pattern
        def expand_sum_bad(xs):
            y = 0
            for _ in xs:
                y += 0
            for _ in xs:
                y *= 1
            for x in xs:
                y += x

        self.assertIsInstance(expand_sum_bad, Pattern)
