"""Unit tests for :func:`fpy2.strategies.elim_iter`.

The transforms are tested exhaustively in
``tests/unit/transform/test_zip_elim.py`` and
``tests/unit/transform/test_enumerate_elim.py``; these tests pin the
wrapper's flags, its internal ordering, and its composition with the
loop operators.
"""


import fpy2 as fp

from fpy2.ast import Enumerate, Zip
from fpy2.ast.visitor import DefaultVisitor
from fpy2.strategies import elim_iter, inline, unroll_for


def _has_node(ast, node_type) -> bool:
    """True iff any reachable expression in *ast* is a *node_type*."""
    found = [False]

    class _C(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            if isinstance(e, node_type):
                found[0] = True
            super()._visit_expr(e, ctx)

    _C()._visit_function(ast, None)
    return found[0]


SCALE = 2.0


@fp.fpy
def _mul_add(acc: fp.Real, x: fp.Real, y: fp.Real) -> fp.Real:
    return acc + SCALE * (x * y)


@fp.fpy
def _dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x, y in zip(xs, ys):
        acc = _mul_add(acc, x, y)  # type: ignore[assignment]
    return acc


@fp.fpy
def _weighted_sum(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for i, x in enumerate(xs):
        acc = acc + i * x
    return acc


@fp.fpy
def _enum_zip(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for i, (x, y) in enumerate(zip(xs, ys)):
        acc = acc + i * (x + y)
    return acc


_XS = [1.0, 2.0, 3.0, 4.0]
_YS = [0.5, -1.5, 2.5, -3.5]


class TestElimIter:

    def test_zip_eliminated(self):
        out = elim_iter(_dot)
        assert not _has_node(out.ast, Zip)
        assert _dot(_XS, _YS) == out(_XS, _YS)
        # the input is not mutated
        assert _has_node(_dot.ast, Zip)

    def test_enumerate_eliminated(self):
        out = elim_iter(_weighted_sum)
        assert not _has_node(out.ast, Enumerate)
        assert _weighted_sum(_XS) == out(_XS)

    def test_enumerate_of_zip_eliminated(self):
        # both intermediates collapse at once — this is why the bundle
        # runs EnumerateElim before ZipElim
        out = elim_iter(_enum_zip)
        assert not _has_node(out.ast, Enumerate)
        assert not _has_node(out.ast, Zip)
        assert _enum_zip(_XS, _YS) == out(_XS, _YS)

    def test_flags(self):
        out = elim_iter(_dot, enable_zip=False)
        assert _has_node(out.ast, Zip)

        out = elim_iter(_weighted_sum, enable_enumerate=False)
        assert _has_node(out.ast, Enumerate)

    def test_enumerate_flag_owns_enum_zip(self):
        # `enumerate(zip(...))` is handled as a unit by EnumerateElim,
        # so the inner zip goes even with `enable_zip=False`
        out = elim_iter(_enum_zip, enable_zip=False)
        assert not _has_node(out.ast, Enumerate)
        assert not _has_node(out.ast, Zip)
        assert _enum_zip(_XS, _YS) == out(_XS, _YS)

    def test_loop_schedule_composition(self):
        # inline, then eliminate the derived iterable, then unroll the
        # resulting indexed loop
        sched = inline(_dot)
        sched = elim_iter(sched)
        sched = unroll_for(sched, times=1)
        assert not _has_node(sched.ast, Zip)
        assert _dot(_XS, _YS) == sched(_XS, _YS)
        # odd length exercises the PEEL remainder
        assert _dot(_XS[:3], _YS[:3]) == sched(_XS[:3], _YS[:3])

