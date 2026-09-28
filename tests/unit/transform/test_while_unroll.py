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
