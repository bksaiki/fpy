"""
Unit tests for constant propagation.
"""

import fpy2 as fp


class TestConstProp():

    def test_example1(self):
        @fp.fpy
        def test():
            x = True
            return x and False

        @fp.fpy
        def test_expect():
            x = True
            return True and False

        f = fp.transform.ConstPropagate.apply(test.ast)
        f.name = test_expect.name
        assert f.is_equiv(test_expect.ast), f'expect:\n{test_expect.format()}\nactual:\n{f.format()}'


    def test_example2(self):
        @fp.fpy
        def test():
            ES = 2
            NB = 8
            with fp.IEEEContext(ES, NB):
                return fp.round(1)

        @fp.fpy
        def test_expect():
            ES = 2
            NB = 8
            with fp.IEEEContext(2, 8):
                return fp.round(1)

        f = fp.transform.ConstPropagate.apply(test.ast)
        f.name = test_expect.name
        assert f.is_equiv(test_expect.ast), f'expect:\n{test_expect.format()}\nactual:\n{f.format()}'
