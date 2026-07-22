"""
Unit tests for loop unrolling.
"""

import fpy2 as fp

from fpy2.transform import ForUnroll, ForUnrollStrategy

PEEL = ForUnrollStrategy.PEEL


def _check_peel(fn, inputs, times_range=(1, 2, 3, 4)):
    """Unroll `fn` with the PEEL strategy for several factors and assert the
    result is semantically identical to the original on every input (a
    name-independent correctness check, executed via the interpreter)."""
    for times in times_range:
        out = ForUnroll.apply(fn.ast, times=times, strategy=PEEL)
        unrolled = fn.with_ast(out)
        for args in inputs:
            expect = fn(*args)
            actual = unrolled(*args)
            assert expect == actual, (
                f'times={times} args={args}: expected {expect}, got {actual}'
            )


class TestForUnrollPeel():
    """PEEL strategy: unroll the multiple-of-k prefix, run the remainder
    separately. Correct for any length, so verified by execution."""

    def test_unknown_length(self):
        # A list parameter has no statically-known length, so this exercises
        # the residual-loop path.
        @fp.fpy
        def sumlist(xs: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for x in xs:
                acc = acc + x
            return acc

        _check_peel(sumlist, [
            ([],),
            ([1.0],),
            ([1.0, 2.0, 3.0],),           # non-divisible by several factors
            ([1.0, 2.0, 3.0, 4.0],),
            ([float(i) for i in range(10)],),
        ])

    def test_range_iterable(self):
        @fp.fpy
        def sumrange() -> fp.Real:
            x = 0
            for i in range(32):
                x = x + i
            return x

        _check_peel(sumrange, [()], times_range=(1, 3, 7))

    def test_non_divisible_length(self):
        @fp.fpy
        def sumrange5() -> fp.Real:
            x = 0
            for i in range(5):
                x = x + i
            return x

        _check_peel(sumrange5, [()])

    def test_tuple_target(self):
        @fp.fpy
        def dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for a, b in zip(xs, ys):
                acc = acc + a * b
            return acc

        _check_peel(dot, [
            ([], []),
            ([1.0, 2.0], [3.0, 4.0]),
            ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        ])

    def test_iterable_evaluated_at_ambient_context(self):
        # The iterable computes rounded values inline; it must be materialized
        # under the ambient context, not the (integer) index context.
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            s = 0.0
            for x in [a * b, a + b]:
                s = s + x
            return s

        _check_peel(f, [(1.5, 2.5), (0.1, 0.3), (3.0, 7.0)], times_range=(1,))


class TestForUnroll():

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

        h = fp.transform.ForUnroll.apply(test.ast, times=0)
        h.name = test_expect.name

        h = fp.transform.ConstFold.apply(h, enable_op=False)
        e = fp.transform.ConstFold.apply(test_expect.ast, enable_op=False)
        assert h.is_equiv(e), f'expect:\n{e.format()}\nactual:\n{h.format()}'

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

        h = fp.transform.ForUnroll.apply(test.ast, times=1)
        h.name = test_expect.name

        h = fp.transform.ConstFold.apply(h, enable_op=False)
        e = fp.transform.ConstFold.apply(test_expect.ast, enable_op=False)
        assert h.is_equiv(e), f'expect:\n{e.format()}\nactual:\n{h.format()}'

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

        h = fp.transform.ForUnroll.apply(test.ast, times=3)
        h.name = test_expect.name

        h = fp.transform.ConstFold.apply(h, enable_op=False)
        e = fp.transform.ConstFold.apply(test_expect.ast, enable_op=False)
        assert h.is_equiv(e), f'expect:\n{e.format()}\nactual:\n{h.format()}'
