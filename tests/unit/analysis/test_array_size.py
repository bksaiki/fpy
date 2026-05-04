"""
Unit tests for :class:`ArraySizeInfer`.
"""

import fpy2 as fp
import pytest

from fpy2.analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    ListSize,
    TupleSize,
)


class TestArraySizeInfer:
    """Unit tests for the array-size analysis."""

    @staticmethod
    def _run(func: fp.Function) -> ArraySizeAnalysis:
        return ArraySizeInfer.analyze(func.ast)

    # ------------------------------------------------------------------
    # Public API surface

    def test_analyze_rejects_non_funcdef(self):
        """``ArraySizeInfer.analyze`` raises ``TypeError`` on non-FuncDef."""
        with pytest.raises(TypeError, match='Expected `FuncDef`'):
            ArraySizeInfer.analyze('not a FuncDef')  # type: ignore[arg-type]

    def test_analysis_result_shape(self):
        """``ArraySizeAnalysis`` exposes the documented public fields."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return x

        info = self._run(f)
        assert isinstance(info, ArraySizeAnalysis)
        assert hasattr(info, 'by_expr')
        assert hasattr(info, 'by_def')
        assert hasattr(info, 'ret_size')
        assert hasattr(info, 'def_use')

    # ------------------------------------------------------------------
    # Argument typing

    def test_scalar_argument_has_none_bound(self):
        """A scalar real argument has no array-size info."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return x

        info = self._run(f)
        x_bounds = [b for d, b in info.by_def.items() if d.name.base == 'x']
        assert x_bounds == [None]

    def test_list_argument_has_listsize(self):
        """A list-of-real argument has a ListSize with empty size set."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return xs[0]

        info = self._run(f)
        xs_bounds = [b for d, b in info.by_def.items() if d.name.base == 'xs']
        assert len(xs_bounds) == 1
        bound = xs_bounds[0]
        assert isinstance(bound, ListSize)
        assert bound.elt is None
        assert bound.size is None

    def test_tuple_argument_has_tuplesize(self):
        """A tuple argument is a TupleSize with one entry per element."""

        @fp.fpy
        def f(p: tuple[fp.Real, fp.Real, fp.Real]) -> tuple[fp.Real, fp.Real, fp.Real]:
            return p

        info = self._run(f)
        p_bounds = [b for d, b in info.by_def.items() if d.name.base == 'p']
        assert len(p_bounds) == 1
        bound = p_bounds[0]
        assert isinstance(bound, TupleSize)
        assert len(bound.elts) == 3
        assert all(e is None for e in bound.elts)

    # ------------------------------------------------------------------
    # Expression rules

    def test_list_literal_size_is_known(self):
        """A list literal carries a singleton ArraySize equal to its length."""

        @fp.fpy
        def f() -> list[fp.Real]:
            return [1.0, 2.0, 3.0]

        info = self._run(f)
        list_bounds = [
            b for b in info.by_expr.values()
            if isinstance(b, ListSize) and b.size == 3
        ]
        assert list_bounds, 'expected a ListSize with size {3}'

    def test_empty_list_literal_size_is_zero(self):
        """An empty list literal has size {0}."""

        @fp.fpy
        def f() -> list[fp.Real]:
            return []

        info = self._run(f)
        list_bounds = [
            b for b in info.by_expr.values()
            if isinstance(b, ListSize) and b.size == 0
        ]
        assert list_bounds, 'expected a ListSize with size {0}'

    def test_tuple_literal(self):
        """A tuple literal records per-element bounds."""

        @fp.fpy
        def f() -> tuple[fp.Real, fp.Real]:
            return (1.0, 2.0)

        info = self._run(f)
        tup_bounds = [
            b for b in info.by_expr.values()
            if isinstance(b, TupleSize) and len(b.elts) == 2
        ]
        assert tup_bounds, 'expected a TupleSize of arity 2'

    def test_range1_known_size(self):
        """``range(n)`` with a partial-eval-known ``n`` yields a known size."""

        @fp.fpy
        def f() -> list[fp.Real]:
            return [0.0 for _ in range(5)]

        info = self._run(f)
        range_bounds = [
            b for b in info.by_expr.values()
            if isinstance(b, ListSize) and b.size == 5
        ]
        assert range_bounds, f'expected a ListSize with size 5, got {info.by_expr.values()}'

    def test_range2_known_size(self):
        """``range(start, stop)`` with both static yields ``stop - start``."""

        @fp.fpy
        def f() -> list[fp.Real]:
            return [0.0 for _ in range(2, 7)]

        info = self._run(f)
        range_bounds = [
            b for b in info.by_expr.values()
            if isinstance(b, ListSize) and b.size == 5
        ]
        assert range_bounds

    def test_list_comprehension_size(self):
        """A comprehension's size is the iterable's size."""

        @fp.fpy
        def f() -> list[fp.Real]:
            return [x * 2.0 for x in [1.0, 2.0, 3.0, 4.0]]

        info = self._run(f)
        comp_bounds = [
            b for b in info.by_expr.values()
            if isinstance(b, ListSize) and b.size == 4
        ]
        assert comp_bounds

    def test_list_ref_returns_element_bound(self):
        """``xs[i]`` exposes the list's element bound."""

        @fp.fpy
        def f() -> fp.Real:
            xs = [1.0, 2.0, 3.0]
            return xs[1]

        info = self._run(f)
        # xs has element bound None (scalar reals); xs[1] also has None.
        ref_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'ListRef'
        ]
        assert ref_bounds == [None]

    def test_list_slice_with_known_bounds(self):
        """``xs[1:3]`` from a size-5 list resolves to size 2."""

        @fp.fpy
        def f() -> list[fp.Real]:
            xs = [1.0, 2.0, 3.0, 4.0, 5.0]
            return xs[1:3]

        info = self._run(f)
        slice_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'ListSlice'
        ]
        assert len(slice_bounds) == 1
        bound = slice_bounds[0]
        assert isinstance(bound, ListSize)
        assert bound.size == 2

    # ------------------------------------------------------------------
    # Control-flow merges

    def test_if_phi_unequal_sizes_widens_to_unknown(self):
        """
        An ``if`` merge of two lists with different known sizes joins
        to ``None`` (unknown) — the flat lattice has no
        "set-of-possible-sizes" representation.
        """

        @fp.fpy
        def f(c: bool) -> list[fp.Real]:
            if c:
                xs = [1.0, 2.0]
            else:
                xs = [1.0, 2.0, 3.0]
            return xs

        info = self._run(f)
        xs_bounds = [b for d, b in info.by_def.items() if d.name.base == 'xs']
        merged = [
            b for b in xs_bounds
            if isinstance(b, ListSize) and b.size is None
        ]
        assert merged, f'expected a phi with size=None, got {xs_bounds}'

    def test_if_phi_equal_sizes_preserves_size(self):
        """
        An ``if`` merge of two lists with the same known size keeps
        that size on the phi.
        """

        @fp.fpy
        def f(c: bool) -> list[fp.Real]:
            if c:
                xs = [1.0, 2.0, 3.0]
            else:
                xs = [4.0, 5.0, 6.0]
            return xs

        info = self._run(f)
        xs_bounds = [b for d, b in info.by_def.items() if d.name.base == 'xs']
        merged = [
            b for b in xs_bounds
            if isinstance(b, ListSize) and b.size == 3
        ]
        assert merged, f'expected a phi with size=3, got {xs_bounds}'

    def test_for_loop_phi_keeps_listsize(self):
        """A loop-carried list maintains its (single-source) size."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            ys = [0.0, 0.0, 0.0]
            for x in xs:
                ys = ys
            return ys

        info = self._run(f)
        ys_bounds = [b for d, b in info.by_def.items() if d.name.base == 'ys']
        # All ys bounds (initial + phi) must be ListSize with size 3.
        assert all(
            isinstance(b, ListSize) and b.size == 3
            for b in ys_bounds
        ), f'expected all ys bounds to be ListSize size 3, got {ys_bounds}'

    def test_for_loop_body_revisit_propagates_widened_phi(self):
        """
        A body-internal read of one phi'd variable depends on another
        phi'd variable's value, and the latter only widens to ``None``
        after the back-edge is processed.  Without a fixpoint the analysis
        would lock in the pre-loop size for the dependent variable; with
        the fixpoint, the widening propagates through.

        Iter 1: phi(ys) = size 2 (pre-loop).  ``xs = ys`` records xs as
                size 2.  ``ys = [...]`` (size 3) widens phi(ys) → None.
        Iter 2: phi(ys) = None.  ``xs = ys`` widens phi(xs) → None.
        """

        @fp.fpy
        def f(n: fp.Real) -> list[fp.Real]:
            xs = [1.0, 2.0]
            ys = xs
            for _ in range(0, 1):
                xs = ys
                ys = [1.0, 2.0, 3.0]
            return xs

        info = self._run(f)
        xs_bounds = [b for d, b in info.by_def.items() if d.name.base == 'xs']
        # xs's loop-phi must have widened to size=None (would be stuck at
        # 2 without a fixpoint).
        widened = [
            b for b in xs_bounds
            if isinstance(b, ListSize) and b.size is None
        ]
        assert widened, (
            f'expected the back-edge to widen xs to size=None, got {xs_bounds}'
        )

    def test_while_loop_body_revisit_propagates_widened_phi(self):
        """``while`` analogue of the for-loop fixpoint test."""

        @fp.fpy
        def f(cond: bool) -> list[fp.Real]:
            xs = [1.0, 2.0]
            ys = xs
            while cond:
                xs = ys
                ys = [1.0, 2.0, 3.0]
            return xs

        info = self._run(f)
        xs_bounds = [b for d, b in info.by_def.items() if d.name.base == 'xs']
        widened = [
            b for b in xs_bounds
            if isinstance(b, ListSize) and b.size is None
        ]
        assert widened, (
            f'expected the back-edge to widen xs to size=None, got {xs_bounds}'
        )

    # ------------------------------------------------------------------
    # Return-size capture

    def test_ret_size_captures_list_return(self):
        """``ret_size`` records the inferred bound of the returned list."""

        @fp.fpy
        def f() -> list[fp.Real]:
            return [1.0, 2.0, 3.0]

        info = self._run(f)
        assert isinstance(info.ret_size, ListSize)
        assert info.ret_size.size == 3

    def test_ret_size_none_for_scalar_return(self):
        """``ret_size`` stays ``None`` when the return value isn't a list."""

        @fp.fpy
        def f() -> fp.Real:
            return 1.0

        info = self._run(f)
        assert info.ret_size is None
