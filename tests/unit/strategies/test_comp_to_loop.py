"""Unit tests for :func:`fpy2.strategies.comp_to_loop`.

The transform itself is tested exhaustively in
``tests/unit/transform/test_comp_to_loop.py``; these tests pin the wrapper's
behavior, its aiming, and the thing the rewrite exists for — that a lowered
comprehension is reachable by the rounding rewrites, and an unlowered one is not.
"""

import pytest

import fpy2 as fp
from fpy2.ast.visitor import DefaultVisitor
from fpy2.strategies import (
    ExprCursor,
    TransformDeclined,
    TransformReferenceError,
    comp_to_loop,
    elim_round,
    insert_round,
    monomorphize,
    refusals,
    simplify,
    sites,
)
from fpy2.types import ListType, RealType


def _comps(ast) -> int:
    n = 0

    class _C(DefaultVisitor):
        def _visit_list_comp(self, e, ctx):
            nonlocal n
            n += 1
            super()._visit_list_comp(e, ctx)

    _C()._visit_function(ast, None)
    return n


@fp.fpy(ctx=fp.FP64)
def _scale(xs: list[fp.Real], k: fp.Real) -> list[fp.Real]:
    return [k * x for x in xs]


@fp.fpy(ctx=fp.FP64)
def _sq(xs: list[fp.Real]) -> list[fp.Real]:
    return [x * x for x in xs]


@fp.fpy(ctx=fp.FP64)
def _ragged(xss: list[list[fp.Real]]) -> list[fp.Real]:
    return [b for a in xss for b in a]


_SAMPLE = [1.5, -3.0, 0.5]


class TestCompToLoop:
    def test_returns_a_function_that_agrees(self):
        out = comp_to_loop(_scale)
        assert out is not _scale
        assert _comps(out.ast) == 0
        for k in (2.0, 0.5):
            assert [float(v) for v in out(_SAMPLE, k)] \
                == [float(v) for v in _scale(_SAMPLE, k)]

    def test_does_not_mutate_the_input(self):
        comp_to_loop(_scale)
        assert _comps(_scale.ast) == 1

    def test_is_idempotent(self):
        once = comp_to_loop(_scale)
        assert comp_to_loop(once).ast.is_equiv(once.ast)

    def test_temp_id_names_the_iterable_binding(self):
        assert 'src = xs' in comp_to_loop(_scale, temp_id='src').format()

    def test_composes_with_simplify(self):
        out = simplify(comp_to_loop(_scale))
        assert [float(v) for v in out(_SAMPLE, 2.0)] \
            == [float(v) for v in _scale(_SAMPLE, 2.0)]

    def test_rejects_non_function(self):
        with pytest.raises(TypeError):
            comp_to_loop(_scale.ast)  # type: ignore[arg-type]


class TestLeftAlone:
    def test_a_dependent_clause_list_is_left_alone_without_error(self):
        """Declined, the pass returns its input rather than raising -- a
        leftover comprehension is still a valid program."""
        assert sites(comp_to_loop, _ragged, dependent=False) == []
        out = comp_to_loop(_ragged, dependent=False)      # no exception
        assert out.ast.is_equiv(_ragged.ast)
        why = refusals(comp_to_loop, _ragged, dependent=False)
        assert len(why) == 1 and 'mentions an earlier' in why[0][1]

    def test_naming_one_by_cursor_says_why(self):
        cursor, _ = refusals(comp_to_loop, _ragged, dependent=False)[0]
        with pytest.raises(TransformDeclined, match='mentions an earlier'):
            comp_to_loop(_ragged, cursor, dependent=False)

    def test_it_lowers_by_default(self):
        """A consumer opts *out* of unfolding, not in."""
        assert sites(comp_to_loop, _ragged) != []


class TestWhere:
    def test_sites_are_comprehensions(self):
        found = sites(comp_to_loop, _scale)
        assert all(isinstance(c, ExprCursor) for c in found)
        assert [type(c.resolve()).__name__ for c in found] == ['ListComp']

    def test_a_cursor_aims_the_same_as_its_index(self):
        for j, cursor in enumerate(sites(comp_to_loop, _scale)):
            expect = comp_to_loop(_scale, j).format()
            assert comp_to_loop(_scale, cursor).format() == expect

    def test_a_where_naming_nothing(self):
        with pytest.raises(TransformReferenceError):
            comp_to_loop(_scale, 7)

    def test_a_cursor_of_an_unrelated_program(self):
        other = sites(comp_to_loop, _sq)[0]
        with pytest.raises(TransformReferenceError):
            comp_to_loop(_scale, other)


class TestUnblocksTheRoundingAxis:
    """Why the rewrite exists: a comprehension's element has no statement slot,
    so no rounding rewrite can reach it."""

    @staticmethod
    def _pinned():
        return monomorphize(_sq, fp.FP64, [ListType(RealType(fp.FP32), 3)])

    def test_a_comprehension_hides_its_element_from_elim_round(self):
        g = self._pinned()
        assert elim_round(g).ast.is_equiv(g.ast)
        assert sites(insert_round, g, ctx=fp.FP64) == []

    def test_lowering_first_exposes_it(self):
        g = self._pinned()
        lowered = comp_to_loop(g)
        hoisted = elim_round(lowered)
        # the exact FP32 product now sits in a statement, so `elim_round` can
        # hoist it to `fp.REAL` and `insert_round` has somewhere to aim
        assert not hoisted.ast.is_equiv(lowered.ast)
        assert len(sites(insert_round, hoisted, ctx=fp.FP64)) == 1
        for x in ([1.5, 2.0, -3.0],):
            assert [float(v) for v in hoisted(x)] == [float(v) for v in g(x)]
