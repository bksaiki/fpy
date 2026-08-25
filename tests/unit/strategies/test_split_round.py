"""Unit tests for :func:`fpy2.strategies.split_round`.

The transform itself is tested exhaustively in
``tests/unit/transform/test_split_round.py``; these tests pin the wrapper's
behavior, its aiming, and the thing the operator exists for -- computing an
operation in a format the environment does have and re-rounding to the one it
does not.
"""

import random

import pytest

import fpy2 as fp
from fpy2.analysis.format_infer import DoubleRoundOp, derive_intermediate
from fpy2.strategies import (
    ExprCursor,
    TransformDeclined,
    TransformReferenceError,
    monomorphize,
    refusals,
    simplify,
    sites,
    split_round,
)
from fpy2.types import RealType

VIA32 = derive_intermediate(fp.FP32)


@fp.fpy(ctx=fp.FP32)
def _product(x: fp.Real, y: fp.Real) -> fp.Real:
    return x * y


@fp.fpy(ctx=fp.FP32)
def _two_ops(x: fp.Real, y: fp.Real) -> fp.Real:
    t = x * y
    s = x + y
    return t + s


def _sweep(n: int = 500):
    rng = random.Random(0)
    for i in range(n):
        a = rng.uniform(-1e3, 1e3) if i % 3 else rng.uniform(-1e-30, 1e-30)
        b = rng.uniform(-1e3, 1e3) if i % 2 else rng.uniform(-1e30, 1e30)
        yield a, b


class TestSplitRound:
    def test_returns_a_function_that_agrees(self):
        out = split_round(_product, VIA32)
        assert out is not _product
        for a, b in _sweep():
            assert str(out(a, b)) == str(_product(a, b))

    def test_does_not_mutate_the_input(self):
        before = _product.format()
        split_round(_product, VIA32)
        assert _product.format() == before

    def test_composes_with_simplify(self):
        out = simplify(split_round(_product, VIA32))
        for a, b in _sweep(200):
            assert str(out(a, b)) == str(_product(a, b))

    def test_rejects_non_function(self):
        with pytest.raises(TypeError):
            split_round(_product.ast, VIA32)  # type: ignore[arg-type]

    def test_rejects_a_non_context_intermediate(self):
        with pytest.raises(TypeError):
            split_round(_product, fp.FP32.format())  # type: ignore[arg-type]


class TestRefused:
    def test_round_to_nearest_twice_with_unknown_operands(self):
        """Nearest over nearest is outside Figure 8 at every width, and the two
        narrower rules both need the operand formats -- which an unannotated
        `fp.Real` argument does not give.  `monomorphize` is what unlocks it; see
        :class:`TestTheOperationRules`."""
        assert sites(split_round, _product, ctx=fp.FP64) == []
        out = split_round(_product, fp.FP64)          # no exception
        assert out.ast.is_equiv(_product.ast)
        why = refusals(split_round, _product, ctx=fp.FP64)
        assert len(why) == 1 and 'is not the same as' in why[0][1]

    def test_naming_a_refused_operation_says_why(self):
        cursor, _ = refusals(split_round, _product, ctx=fp.FP64)[0]
        with pytest.raises(TransformDeclined, match='is not the same as'):
            split_round(_product, fp.FP64, cursor)


class TestWhere:
    def test_sites_are_operations(self):
        found = sites(split_round, _two_ops, ctx=VIA32)
        assert len(found) == 3      # two operands and the outer add
        assert all(isinstance(c, ExprCursor) for c in found)

    def test_a_cursor_aims_the_same_as_its_index(self):
        for j, cursor in enumerate(sites(split_round, _two_ops, ctx=VIA32)):
            assert split_round(_two_ops, VIA32, cursor).format() \
                == split_round(_two_ops, VIA32, j).format()

    def test_a_where_naming_nothing(self):
        with pytest.raises(TransformReferenceError):
            split_round(_product, VIA32, 7)

    def test_a_cursor_forwards_across_a_split(self):
        """A cursor taken before the first split still names its operation
        after, which is what `func.rebase` and `exprs_preserved` are for."""
        listed = sites(split_round, _two_ops, ctx=VIA32)
        once = split_round(_two_ops, VIA32, listed[0])
        twice = split_round(once, VIA32, listed[1])
        assert twice.format().count('RoundingMode.RTO') == 2

    def test_a_cursor_of_an_unrelated_program(self):
        other = sites(split_round, _two_ops, ctx=VIA32)[0]
        with pytest.raises(TransformReferenceError):
            split_round(_product, VIA32, other)


class TestTheRecipe:
    """What the operator is for: §5.3's modular-library step.  An environment
    with no FP32 multiply computes the product wide under round-to-odd and
    re-rounds, getting exactly what the FP32 multiply would have given."""

    def test_the_wide_computation_reproduces_the_narrow_one(self):
        out = split_round(_product, VIA32)
        text = out.format()
        assert 'RoundingMode.RTO' in text and 'fp.round' in text
        # including the cases the premise's `next(b)` bump exists for
        edges = [(1e-40, 1e-40), (1e30, 1e30), (0.0, -1.0),
                 (float('nan'), 1.0), (1.0, 1.0)]
        for a, b in edges:
            assert str(out(a, b)) == str(_product(a, b)), (a, b)

    def test_the_intermediate_is_wider_than_the_target(self):
        assert VIA32.format().pmax > 24
        assert VIA32.rounding_mode() is fp.RoundingMode.RTO

class TestTheOperationRules:
    """The other half of the recipe: an intermediate the *environment actually
    has*.  Round-to-odd is not a hardware mode, so a split that has to run on
    real hardware needs one of the operation-specific rules instead."""

    @staticmethod
    def _pinned(f):
        return monomorphize(f, args=[RealType(fp.FP32), RealType(fp.FP32)])

    def test_a_product_splits_through_fp64(self):
        """The exact FP32 product is 48 digits, which FP64 holds, so the
        intermediate rounding is the identity and both contexts keep RNE."""
        pinned = self._pinned(_product)
        out = split_round(pinned, fp.FP64)
        assert 'RoundingMode.RTO' not in out.format()
        for a, b in _sweep(200):
            assert str(out(a, b)) == str(pinned(a, b))

    def test_a_sum_splits_through_a_derived_intermediate(self):
        """An exact sum is 278 digits, so this is the addition rule rather than
        exactness -- and `derive_intermediate` sizes it."""

        @fp.fpy(ctx=fp.FP32)
        def total(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        pinned = monomorphize(total, args=[RealType(fp.FP32), RealType(fp.FP32)])
        via = derive_intermediate(fp.FP32, DoubleRoundOp.ADD)
        assert via.rounding_mode() is fp.RoundingMode.RNE
        out = split_round(pinned, via)
        assert len(sites(split_round, pinned, ctx=via)) == 1
        for a, b in _sweep(200):
            assert str(out(a, b)) == str(pinned(a, b))

    def test_without_monomorphize_it_refuses(self):
        """The operand formats are the premise, so this is the order the two
        operators go in."""
        assert sites(split_round, _product, ctx=fp.FP64) == []
        assert len(sites(split_round, self._pinned(_product), ctx=fp.FP64)) == 1

