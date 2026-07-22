"""
Unit tests for loop unrolling.
"""

import fpy2 as fp

from fpy2.transform import ForUnroll, ForUnrollStrategy

PEEL = ForUnrollStrategy.PEEL
STRICT = ForUnrollStrategy.STRICT


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

    def test_static_divisible_no_runtime_check(self):
        # A statically-known length divisible by k needs no `len`/`fmod` and
        # no remainder region — just the main loop over a literal bound.
        @fp.fpy
        def r() -> fp.Real:
            x = 0
            for i in range(8):
                x = x + i
            return x

        out = ForUnroll.apply(r.ast, times=1, strategy=PEEL)   # k=2, 8 % 2 == 0
        txt = out.format()
        assert 'len(' not in txt and 'fmod(' not in txt
        assert txt.count('for ') == 1                          # main loop only
        assert r() == r.with_ast(out)()

    def test_static_non_divisible_peels_remainder(self):
        # A known length not divisible by k: literal main bound plus the
        # leftover peeled straight-line (still no residual loop, no len/fmod).
        @fp.fpy
        def r() -> fp.Real:
            x = 0
            for i in range(7):
                x = x + i
            return x

        out = ForUnroll.apply(r.ast, times=1, strategy=PEEL)   # k=2, 7 % 2 == 1
        txt = out.format()
        assert 'len(' not in txt and 'fmod(' not in txt
        assert txt.count('for ') == 1                          # main loop; remainder is straight-line
        assert r() == r.with_ast(out)()

    def test_static_full_unroll(self):
        # When k exceeds the known length, there is no main region at all:
        # the loop becomes fully straight-line.
        @fp.fpy
        def r() -> fp.Real:
            x = 0
            for i in range(3):
                x = x + i
            return x

        out = ForUnroll.apply(r.ast, times=7, strategy=PEEL)   # k=8 > 3
        txt = out.format()
        assert txt.count('for ') == 0                          # no loop
        assert r() == r.with_ast(out)()

    def test_static_empty(self):
        @fp.fpy
        def r() -> fp.Real:
            x = 0
            for i in range(0):
                x = x + i
            return x

        out = ForUnroll.apply(r.ast, times=1, strategy=PEEL)
        assert out.format().count('for ') == 0                 # nothing to iterate
        assert r() == r.with_ast(out)()


class TestForUnrollMutation():
    """Regression: a body that mutates the iterable in place. Reads must stay
    interleaved with their bodies (grouping reads ahead of bodies miscompiles
    this), so both strategies must match the original."""

    def test_in_place_mutation(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            s = 0.0
            for x in xs:
                xs[1] = 99.0     # in-place mutation of the iterable
                s = s + x
            return s

        for strategy in (STRICT, PEEL):
            for times in (1, 3):
                # length 4 keeps STRICT's divisibility precondition satisfied
                out = ForUnroll.apply(f.ast, times=times, strategy=strategy)
                expect = f([10.0, 20.0, 30.0, 40.0])
                actual = f.with_ast(out)([10.0, 20.0, 30.0, 40.0])
                assert expect == actual, (strategy, times, expect, actual)


class TestForUnrollStrict():
    """STRICT strategy specifics (beyond the example round-trip tests)."""

    def test_static_non_divisible_raises(self):
        @fp.fpy
        def r() -> fp.Real:
            x = 0
            for i in range(5):
                x = x + i
            return x

        # 5 is not a multiple of k=2, and the length is statically known.
        try:
            ForUnroll.apply(r.ast, times=1, strategy=STRICT)
            assert False, 'expected a ValueError for a provably-indivisible length'
        except ValueError:
            pass

    def test_unknown_length_keeps_assert(self):
        @fp.fpy
        def sumlist(xs: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for x in xs:
                acc = acc + x
            return acc

        txt = ForUnroll.apply(sumlist.ast, times=1, strategy=STRICT).format()
        assert 'fmod(' in txt and 'len(' in txt          # runtime divisibility assert
        # divisible input runs correctly
        assert sumlist([1.0, 2.0, 3.0, 4.0]) == \
            sumlist.with_ast(ForUnroll.apply(sumlist.ast, times=1, strategy=STRICT))([1.0, 2.0, 3.0, 4.0])


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
            t = range(32)                # length 32 is statically known:
            for i2 in range(0, 32, 2):   # no len/assert, literal bound
                with fp.INTEGER:         # offset indices grouped
                    i3 = i2 + 1
                i = t[i2]                # reads interleaved with bodies,
                x += i                   # target reassigned (not renamed)
                i = t[i3]
                x += i
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
            t = range(32)
            for i2 in range(0, 32, 4):
                with fp.INTEGER:
                    i3 = i2 + 1
                    i4 = i2 + 2
                    i5 = i2 + 3
                i = t[i2]
                x += i
                i = t[i3]
                x += i
                i = t[i4]
                x += i
                i = t[i5]
                x += i
            return x

        h = fp.transform.ForUnroll.apply(test.ast, times=3)
        h.name = test_expect.name

        h = fp.transform.ConstFold.apply(h, enable_op=False)
        e = fp.transform.ConstFold.apply(test_expect.ast, enable_op=False)
        assert h.is_equiv(e), f'expect:\n{e.format()}\nactual:\n{h.format()}'
