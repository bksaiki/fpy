"""
`UnfoldZip` and `UnfoldEnumerate` — stating a derived iterable as the
comprehension `derived-semantics.rst` defines it to be.
"""

import pytest

import fpy2 as fp
from fpy2.ast import NamedId
from fpy2.transform import (
    Hoistable,
    TransformDeclined,
    TransformReferenceError,
    UnfoldEnumerate,
    UnfoldZip,
)


def _text(func, ast):
    return ' '.join(func.with_ast(ast).format().split())


class TestZip:
    def test_the_shape_it_states(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], ys: list[fp.Real]):
            return zip(xs, ys)

        out = _text(f, UnfoldZip.apply(f.ast))
        assert 'assert len(ys) == len(xs)' in out
        assert 'return [(xs[t], ys[t]) for t in range(len(xs))]' in out

    def test_one_assert_per_extra_iterable(self):
        """The length is the *first* iterable's, so it needs no assert of its
        own and every other one does."""
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], ys: list[fp.Real], zs: list[fp.Real]):
            return zip(xs, ys, zs)

        out = _text(f, UnfoldZip.apply(f.ast))
        assert out.count('assert') == 2
        assert 'len(ys) == len(xs)' in out and 'len(zs) == len(xs)' in out

    def test_it_agrees_with_the_surface_form(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for x, y in zip(xs, ys):
                acc = acc + x * y
            return acc

        xs, ys = [1.5, 2.0, 3.25], [0.5, 4.0, 1.0]
        assert f(xs, ys) == f.with_ast(UnfoldZip.apply(f.ast))(xs, ys)


class TestEnumerate:
    def test_the_shape_it_states(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            return enumerate(xs)

        out = _text(f, UnfoldEnumerate.apply(f.ast))
        assert 'return [(t, xs[t]) for t in range(len(xs))]' in out
        # nothing to carry: the index is `INTEGER` either way and the length
        # comes from the same list
        assert 'assert' not in out

    def test_it_agrees_with_the_surface_form(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for i, x in enumerate(xs):
                acc = acc + x
            return acc

        xs = [1.5, 2.0, 3.25]
        assert f(xs) == f.with_ast(UnfoldEnumerate.apply(f.ast))(xs)


class TestArgumentsAreBoundOnce:
    def test_a_nested_form_is_named(self):
        """Both rewrites read their argument twice, in ``xs[i]`` and in
        ``len(xs)``, so anything but a name is bound above: rebuilding it would
        make two lists where the program had one."""
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], ys: list[fp.Real]):
            return enumerate(zip(xs, ys))

        ast = UnfoldEnumerate.apply(UnfoldZip.apply(f.ast))
        out = _text(f, ast)
        # the inner comprehension is built once, into a name the outer reads
        assert out.count('for t') == 1 or out.count('range(len(xs))') == 1
        assert 'range(len(t3))' in out or 'range(len(t' in out

    def test_either_order_works(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], ys: list[fp.Real]):
            return enumerate(zip(xs, ys))

        a = UnfoldEnumerate.apply(UnfoldZip.apply(f.ast))
        b = UnfoldZip.apply(UnfoldEnumerate.apply(f.ast))
        for ast in (a, b):
            out = _text(f, ast)
            assert 'zip' not in out and 'enumerate' not in out

    def test_temp_id_is_respected(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            return enumerate(xs)

        out = _text(f, UnfoldEnumerate.apply(f.ast, temp_id=NamedId('idx')))
        assert 'for idx in range' in out


class TestSealedPositions:
    """A position with no statement slot is refused, not silently rewritten:
    the rewrite has bindings to emit, and a `while` condition would compute
    them once for a loop that re-evaluates."""

    @staticmethod
    def _refusals(func):
        return [r for _, r in UnfoldZip.refusals(func.ast)]

    def test_a_while_condition(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            n = 0
            while n < len(zip(xs, ys)):
                n = n + 1
            return n

        assert UnfoldZip.sites(f.ast) == []
        assert len(self._refusals(f)) == 1

    def test_a_comprehension_element(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], rs: list[list[fp.Real]]):
            return [zip(xs, r) for r in rs]

        assert UnfoldZip.sites(f.ast) == []
        assert len(self._refusals(f)) == 1

    def test_hoistable_gives_a_slot(self):
        """`Hoistable` moves the sealed expression into a statement of its own,
        which is why the cpp pipeline runs the two to a fixpoint together.  A
        `while` condition becomes *two* -- one before the loop and one at the
        end of the body -- which is how it is evaluated each iteration without
        a slot inside it."""
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            n = 0
            while n < len(zip(xs, ys)):
                n = n + 1
            return n

        assert len(UnfoldZip.sites(Hoistable.apply(f.ast))) == 2


class TestOnceEvaluatedPositions:
    """The three positions evaluated exactly once at the point the preamble
    runs.  A `for` iterable is where a derived iterable actually appears, so
    sealing it would refuse the site that matters."""

    def test_a_for_iterable(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for x, y in zip(xs, ys):
                acc = acc + x * y
            return acc

        assert len(UnfoldZip.sites(f.ast)) == 1
        assert 'zip' not in _text(f, UnfoldZip.apply(f.ast))

    def test_an_if_condition(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            if len(zip(xs, ys)) > 0:
                return 1.0
            return 0.0

        assert len(UnfoldZip.sites(f.ast)) == 1


class TestWhere:
    def test_an_index_takes_one(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], ys: list[fp.Real]):
            a = zip(xs, ys)
            b = zip(ys, xs)
            return (a, b)

        assert len(UnfoldZip.sites(f.ast)) == 2
        out = _text(f, UnfoldZip.apply(f.ast, where=0))
        assert out.count('zip') == 1

    def test_a_cursor_takes_the_one_it_names(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], ys: list[fp.Real]):
            a = zip(xs, ys)
            b = zip(ys, xs)
            return (a, b)

        second = UnfoldZip.sites(f.ast)[1]
        out = _text(f, UnfoldZip.apply(f.ast, where=second))
        assert out.count('zip') == 1
        assert 'assert len(xs) == len(ys)' in out    # the second one's operands

    def test_a_cursor_naming_a_refusal_says_why(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], rs: list[list[fp.Real]]):
            return [zip(xs, r) for r in rs]

        cursor, _ = UnfoldZip.refusals(f.ast)[0]
        with pytest.raises(TransformDeclined, match='statement-level'):
            UnfoldZip.apply(f.ast, where=cursor)

    def test_an_index_past_the_end(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], ys: list[fp.Real]):
            return zip(xs, ys)

        with pytest.raises(TransformReferenceError):
            UnfoldZip.apply(f.ast, where=3)

    def test_nothing_to_do_is_not_an_error(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]) -> fp.Real:
            return xs[0]

        assert UnfoldZip.apply(f.ast) is not None
        assert UnfoldEnumerate.sites(f.ast) == []
