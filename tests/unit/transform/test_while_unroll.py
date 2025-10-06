"""
Unit tests for loop unrolling.
"""

import fpy2 as fp
import unittest

class TestForUnroll(unittest.TestCase):

    def test_example1(self):
        @fp.fpy
        def test(t: fp.Real):
            x: fp.Real = 0
            while t > 0:
                x += t
                t -= 1
            return x

        @fp.fpy
        def test_expect(t: fp.Real):
            x: fp.Real = 0
            while t > 0:
                x += t
                t -= 1
            return x

        h = fp.transform.WhileUnroll.apply(test.ast, times=0)
        h.name = test_expect.name
        self.assertTrue(
            h.is_equiv(test_expect.ast),
            f'expect:\n{test_expect.ast.format()}\nactual:\n{h.format()}'
        )

    def test_example2(self):
        @fp.fpy
        def test(t: fp.Real):
            x: fp.Real = 0
            while t > 0:
                x += t
                t -= 1
            return x

        @fp.fpy
        def test_expect(t: fp.Real):
            x: fp.Real = 0
            if t > 0:
                x += t
                t -= 1
                while t > 0:
                    x += t
                    t -= 1
            return x

        h = fp.transform.WhileUnroll.apply(test.ast, times=1)
        h.name = test_expect.name
        self.assertTrue(
            h.is_equiv(test_expect.ast),
            f'expect:\n{test_expect.ast.format()}\nactual:\n{h.format()}'
        )

    def test_example3(self):
        @fp.fpy
        def test(t: fp.Real):
            x: fp.Real = 0
            while t > 0:
                x += t
                t -= 1
            return x

        @fp.fpy
        def test_expect(t: fp.Real):
            x: fp.Real = 0
            if t > 0:
                x += t
                t -= 1
                if t > 0:
                    x += t
                    t -= 1
                    while t > 0:
                        x += t
                        t -= 1
            return x

        h = fp.transform.WhileUnroll.apply(test.ast, times=2)
        h.name = test_expect.name
        self.assertTrue(
            h.is_equiv(test_expect.ast),
            f'expect:\n{test_expect.ast.format()}\nactual:\n{h.format()}'
        )
