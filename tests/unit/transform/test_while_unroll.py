"""
Unit tests for loop unrolling.
"""

import fpy2 as fp

class TestWhileUnroll():

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
        assert h.is_equiv(test_expect.ast), f'expect:\n{test_expect.ast.format()}\nactual:\n{h.format()}'

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
        assert h.is_equiv(test_expect.ast), f'expect:\n{test_expect.ast.format()}\nactual:\n{h.format()}'

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
        assert h.is_equiv(test_expect.ast), f'expect:\n{test_expect.ast.format()}\nactual:\n{h.format()}'


class TestWhileUnrollWhere():
    """`where` names a single loop by pre-order index; an index that names no
    loop is a caller error, not a silent no-op."""

    def test_out_of_range_raises(self):
        @fp.fpy
        def one_loop(t: fp.Real):
            while t > 0:      # the only while loop -> index 0
                t -= 1
            return t

        for bad in (1, 2, 5):
            try:
                fp.transform.WhileUnroll.apply(one_loop.ast, where=bad, times=1)
                assert False, f'expected ValueError for where={bad}'
            except ValueError:
                pass

    def test_negative_raises(self):
        @fp.fpy
        def one_loop(t: fp.Real):
            while t > 0:
                t -= 1
            return t

        try:
            fp.transform.WhileUnroll.apply(one_loop.ast, where=-1, times=1)
            assert False, 'expected ValueError for where=-1'
        except ValueError:
            pass

    def test_no_loops_raises(self):
        @fp.fpy
        def no_loop(x: fp.Real):
            return x + 1

        try:
            fp.transform.WhileUnroll.apply(no_loop.ast, where=0, times=1)
            assert False, 'expected ValueError: no loop at index 0'
        except ValueError:
            pass

    def test_valid_where_selects_one_loop(self):
        # Two sibling loops; where=1 selects the second and is in range.
        @fp.fpy
        def two_loops(t: fp.Real):
            x: fp.Real = 0
            while t > 0:
                x += t
                t -= 1
            while x > 0:
                x -= 1
            return x

        # where=1 is in range -> no error, semantics preserved
        out = fp.transform.WhileUnroll.apply(two_loops.ast, where=1, times=1)
        u = two_loops.with_ast(out)
        for v in (0.0, 1.0, 4.0):
            assert two_loops(v) == u(v)
