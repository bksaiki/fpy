
from fpy2 import *
from fpy2.rewrite.pattern import Pattern

class ParseSimpleTestCase():
    """Testing `Pattern` parsing"""

    def test_const_pat(self):
        @pattern
        def const_pat():
            1
        assert isinstance(const_pat, Pattern)

    def test_identity_pat(self):
        @pattern
        def identity_pat(x):
            x
        assert isinstance(identity_pat, Pattern)


class ParseExampleTestCase():
    """Testing `Pattern` parsing for examples"""

    def test_fma_example(self):
        @pattern
        def insert_mad_pat(a, b, c):
            a * b + c
        @pattern
        def insert_fma_pat(a, b, c):
            fma(a, b, c)

        assert isinstance(insert_mad_pat, Pattern)
        assert isinstance(insert_fma_pat, Pattern)

    def test_sum_example(self):
        @pattern
        def insert_sum_pat(xs):
            y = 0
            for x in xs:
                y += x
        @pattern
        def insert_sum_op_pat(xs):
            y = sum(xs)

        assert isinstance(insert_sum_pat, Pattern)
        assert isinstance(insert_sum_op_pat, Pattern)

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

        assert isinstance(expand_sum_bad, Pattern)


    def test_while_unroll(self):
        @pattern
        def unroll_while_l(t, e):
            while t > 0:
                t = e

        @pattern
        def unroll_while_r(t, e):
            t = e
            while t > 0:
                t = e

        assert isinstance(unroll_while_l, Pattern)
        assert isinstance(unroll_while_r, Pattern)
