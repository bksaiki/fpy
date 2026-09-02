"""
The `unfold_zip` / `unfold_enumerate` strategies: the scheduling-language
surface over `UnfoldZip` / `UnfoldEnumerate`.
"""

import pytest

import fpy2 as fp
import fpy2.strategies as st
from fpy2.strategies import TransformDeclined


@fp.fpy(ctx=fp.FP64)
def dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x, y in zip(xs, ys):
        acc = acc + x * y
    return acc


@fp.fpy(ctx=fp.FP64)
def total(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for i, x in enumerate(xs):
        acc = acc + x
    return acc


@fp.fpy(ctx=fp.FP64)
def sealed(xs: list[fp.Real], rs: list[list[fp.Real]]):
    return [zip(xs, r) for r in rs]


def _text(func):
    return ' '.join(func.format().split())


class TestSurface:
    def test_unfold_zip(self):
        out = _text(st.unfold_zip(dot))
        assert 'zip' not in out
        assert 'assert len(ys) == len(xs)' in out

    def test_unfold_enumerate(self):
        out = _text(st.unfold_enumerate(total))
        assert 'enumerate' not in out

    def test_values_agree(self):
        xs, ys = [1.5, 2.0, 3.25], [0.5, 4.0, 1.0]
        assert st.unfold_zip(dot)(xs, ys) == dot(xs, ys)
        assert st.unfold_enumerate(total)(xs) == total(xs)

    def test_a_bare_function_is_refused(self):
        with pytest.raises(TypeError, match='Function'):
            st.unfold_zip(dot.ast)   # type: ignore[arg-type]


class TestAimed:
    def test_sites_and_refusals_are_registered(self):
        assert len(st.sites(st.unfold_zip, dot)) == 1
        assert len(st.sites(st.unfold_enumerate, total)) == 1
        assert st.sites(st.unfold_zip, sealed) == []
        assert len(st.refusals(st.unfold_zip, sealed)) == 1

    def test_a_cursor_aims_it(self):
        where = st.sites(st.unfold_zip, dot)[0]
        assert 'zip' not in _text(st.unfold_zip(dot, where))

    def test_a_refused_cursor_says_why(self):
        cursor, why = st.refusals(st.unfold_zip, sealed)[0]
        assert 'statement-level' in why
        with pytest.raises(TransformDeclined, match='statement-level'):
            st.unfold_zip(sealed, cursor)


class TestAgainstTheFuse:
    """`elim_iter` is the opposite trade: it fuses the derived iterable into an
    indexed loop so no list of tuples is built at all.  Unfolding first leaves
    it nothing to match, which is why a pipeline runs the fuse ahead."""

    def test_the_fuse_no_longer_matches_an_unfolded_program(self):
        unfolded = st.unfold_zip(dot)
        assert _text(st.elim_iter(unfolded)) == _text(unfolded)

    def test_the_fuse_leaves_nothing_to_unfold(self):
        fused = st.elim_iter(dot)
        assert st.sites(st.unfold_zip, fused) == []
