"""
Unit tests for loop unrolling.
"""

import fpy2 as fp
import unittest

class TestForUnroll(unittest.TestCase):

    def test_example1(self):
        @fp.fpy
        def test():
            x = 0
            for i in range(32):
                x += i
            return x

        @fp.fpy
        def test_expect():
            x = 0
            for i in range(32):
                x += i
            return x

        h = fp.transform.ForUnroll.apply(test.ast, times=1)
        h.name = test_expect.name

        h = fp.transform.ConstFold.apply(h, enable_op=False)
        e = fp.transform.ConstFold.apply(test_expect.ast, enable_op=False)
        self.assertTrue(
            h.is_equiv(e),
            f'expect:\n{e.format()}\nactual:\n{h.format()}'
        )

    def test_example2(self):
        @fp.fpy
        def test():
            x = 0
            for i in range(32):
                x += i
            return x

        @fp.fpy
        def test_expect():
            x = 0
            with fp.INTEGER:
                t = range(32)
                n = len(t)
                assert fp.fmod(n, 2) == 0
            for i2 in range(0, n, 2):
                with fp.INTEGER:
                    i = t[i2]
                    i3 = t[i2 + 1]
                x += i
                x += i3
            return x

        h = fp.transform.ForUnroll.apply(test.ast, times=2)
        h.name = test_expect.name

        h = fp.transform.ConstFold.apply(h, enable_op=False)
        e = fp.transform.ConstFold.apply(test_expect.ast, enable_op=False)
        self.assertTrue(
            h.is_equiv(e),
            f'expect:\n{e.format()}\nactual:\n{h.format()}'
        )

    def test_example3(self):
        @fp.fpy
        def test():
            x = 0
            for i in range(32):
                x += i
            return x

        @fp.fpy
        def test_expect():
            x = 0
            with fp.INTEGER:
                t = range(32)
                n = len(t)
                assert fp.fmod(n, 4) == 0
            for i2 in range(0, n, 4):
                with fp.INTEGER:
                    i = t[i2]
                    i3 = t[i2 + 1]
                    i4 = t[i2 + 2]
                    i5 = t[i2 + 3]
                x += i
                x += i3
                x += i4
                x += i5
            return x

        h = fp.transform.ForUnroll.apply(test.ast, times=4)
        h.name = test_expect.name

        h = fp.transform.ConstFold.apply(h, enable_op=False)
        e = fp.transform.ConstFold.apply(test_expect.ast, enable_op=False)
        self.assertTrue(
            h.is_equiv(e),
            f'expect:\n{e.format()}\nactual:\n{h.format()}'
        )
