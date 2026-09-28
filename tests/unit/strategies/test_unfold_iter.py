"""
The `unfold_zip` / `unfold_enumerate` strategies: the scheduling-language
surface over `UnfoldZip` / `UnfoldEnumerate`.
"""


import fpy2 as fp
import fpy2.strategies as st


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
