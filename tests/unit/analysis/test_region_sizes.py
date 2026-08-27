"""Unit tests for :func:`fpy2.analysis.alias.region_sizes`.

One proven length per alias region, or ``None``.  The meet poisons on any doubt,
so most of these check that a region a consumer *could* have sized is refused
when some contributor disagrees -- the direction a bug would take.

Nothing here imports a backend: a region's length is a fact about the program.
"""

import fpy2 as fp

from fpy2.analysis import Alias, ArraySizeInfer, DefineUse
from fpy2.analysis.alias import region_sizes


def _sizes(func: fp.Function):
    """``(alias, def_use, sizes)`` for *func*."""
    du = DefineUse.analyze(func.ast)
    alias = Alias.analyze(func.ast, def_use=du)
    array_size = ArraySizeInfer.analyze(func.ast)
    return alias, du, region_sizes(alias, array_size)


def _def(du, name: str):
    return next(d for d in du.defs if str(d.name) == name)


class TestRegionSizes:

    def test_a_literal_has_its_length(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                xs = [x, x, x]
                return xs[0]

        alias, du, sizes = _sizes(f)
        assert sizes[alias.region_of(_def(du, 'xs'))] == 3

    def test_two_lengths_in_one_region_poison_it(self):
        """Both arms bind the same name, so one region holds a 2 and a 3 --
        no single fixed length can hold it."""

        @fp.fpy
        def f(c: bool, x: fp.Real) -> fp.Real:
            with fp.FP64:
                if c:
                    xs = [x, x]
                else:
                    xs = [x, x, x]
                return xs[0]

        alias, du, sizes = _sizes(f)
        regions = {alias.region_of(d) for d in du.defs if str(d.name) == 'xs'}
        assert regions and None not in regions
        # present-and-poisoned, not merely absent
        assert all(r in sizes and sizes[r] is None for r in regions)

    def test_a_parameter_length_is_unknown(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return xs[0]

        alias, du, sizes = _sizes(f)
        r = alias.region_of(_def(du, 'xs'))
        assert r in sizes and sizes[r] is None

    def test_a_symbolic_length_is_refused(self):
        """A length that varies per run is never a fixed length, even though the
        size analysis can name it."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = [x * 2 for x in xs]
                return ys[0]

        alias, du, sizes = _sizes(f)
        r = alias.region_of(_def(du, 'ys'))
        assert r in sizes and sizes[r] is None

    def test_nesting_is_sized_per_level(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                xss = [[x, x], [x, x]]
                return xss[0][0]

        alias, du, sizes = _sizes(f)
        d = _def(du, 'xss')
        assert sizes[alias.region_of(d, 0)] == 2
        assert sizes[alias.region_of(d, 1)] == 2

    def test_an_inner_disagreement_leaves_the_outer_proven(self):
        """The rows differ, so the element region is poisoned -- but the spine
        still has two of them."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                xss = [[x, x], [x, x, x]]
                return xss[0][0]

        alias, du, sizes = _sizes(f)
        d = _def(du, 'xss')
        assert sizes[alias.region_of(d, 0)] == 2
        assert sizes[alias.region_of(d, 1)] is None

    def test_a_returned_literal_is_sized_from_by_expr(self):
        """It has a region but no definition, so only the expression-side bound
        can size it."""

        @fp.fpy
        def f(x: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                return [x, x, x]

        alias, _du, sizes = _sizes(f)
        assert 3 in sizes.values()
