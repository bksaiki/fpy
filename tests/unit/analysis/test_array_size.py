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

    def test_empty_nested_list_literal_does_not_crash(self):
        """
        Regression: an empty list literal annotated as ``list[list[…]]``
        must not raise ``IndexError`` when the analysis builds the
        element bound (the previous implementation indexed
        ``elt_sizes[0]`` unconditionally).  The analysis runs to
        completion and reports a ``ListSize`` with size 0; the element
        bound's precision depends on whether the type checker resolved
        the inner type variable.
        """

        @fp.fpy
        def f() -> list[list[fp.Real]]:
            xs: list[list[fp.Real]] = []
            return xs

        # The fact that ``self._run(f)`` completes is half the test —
        # before the fix, this raised IndexError.
        info = self._run(f)
        empty_bounds = [
            b for b in info.by_expr.values()
            if isinstance(b, ListSize) and b.size == 0
        ]
        assert empty_bounds, (
            f'expected an outer ListSize of size 0, got {list(info.by_expr.values())}'
        )

    def test_nested_list_literal_with_differing_inner_sizes(self):
        """
        ``[[1.0], [1.0, 2.0]]``: inner sizes 1 and 2 conflict — the
        element bound's size must widen to ``None`` rather than retain
        only the first element's size.
        """

        @fp.fpy
        def f() -> list[list[fp.Real]]:
            return [[1.0], [1.0, 2.0]]

        info = self._run(f)
        # Find the outer ListExpr by its nested-ListSize element bound.
        outer_bounds = [
            b for b in info.by_expr.values()
            if isinstance(b, ListSize)
            and b.size == 2
            and isinstance(b.elt, ListSize)
        ]
        assert outer_bounds, (
            f'expected outer ListSize of size 2 with nested element, '
            f'got {list(info.by_expr.values())}'
        )
        # Inner sizes are 1 and 2 — must widen to None.
        assert outer_bounds[0].elt.size is None, (
            f'expected widened inner size None, got {outer_bounds[0].elt.size}'
        )

    def test_nested_list_literal_with_matching_inner_sizes(self):
        """
        ``[[1.0, 2.0], [3.0, 4.0]]``: inner sizes both 2 — the element
        bound's size survives the unify.
        """

        @fp.fpy
        def f() -> list[list[fp.Real]]:
            return [[1.0, 2.0], [3.0, 4.0]]

        info = self._run(f)
        outer_bounds = [
            b for b in info.by_expr.values()
            if isinstance(b, ListSize)
            and b.size == 2
            and isinstance(b.elt, ListSize)
            and b.elt.size == 2
        ]
        assert outer_bounds, (
            f'expected outer size 2 with inner size 2, got {list(info.by_expr.values())}'
        )

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

    def test_range3_positive_step(self):
        """``range(0, 10, 3)`` has size 4 (== ``len(range(0, 10, 3))``)."""

        @fp.fpy
        def f() -> list[fp.Real]:
            return [0.0 for _ in range(0, 10, 3)]

        info = self._run(f)
        range_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'Range3'
        ]
        assert range_bounds and range_bounds[0].size == 4

    def test_range3_negative_step(self):
        """``range(10, 0, -3)`` walks down: size 4."""

        @fp.fpy
        def f() -> list[fp.Real]:
            return [0.0 for _ in range(10, 0, -3)]

        info = self._run(f)
        range_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'Range3'
        ]
        assert range_bounds and range_bounds[0].size == 4

    def test_range3_empty_when_step_overshoots(self):
        """``range(5, 5, 1)`` is empty: size 0."""

        @fp.fpy
        def f() -> list[fp.Real]:
            return [0.0 for _ in range(5, 5, 1)]

        info = self._run(f)
        range_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'Range3'
        ]
        assert range_bounds and range_bounds[0].size == 0

    def test_range3_zero_step_unknown(self):
        """A zero step is invalid; size falls back to unknown."""

        @fp.fpy
        def f() -> list[fp.Real]:
            return [0.0 for _ in range(0, 5, 0)]

        info = self._run(f)
        range_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'Range3'
        ]
        assert range_bounds and range_bounds[0].size is None

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

    def test_list_slice_omitted_stop_uses_list_size(self):
        """
        ``xs[1:]`` (omitted stop) on a size-5 list has size 4.  Falling
        back to the original list size (5) would be unsound — downstream
        passes (e.g., Sum expansion) could overrun the slice.
        """

        @fp.fpy
        def f() -> list[fp.Real]:
            xs = [1.0, 2.0, 3.0, 4.0, 5.0]
            return xs[1:]

        info = self._run(f)
        slice_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'ListSlice'
        ]
        assert len(slice_bounds) == 1 and slice_bounds[0].size == 4, (
            f'expected slice size 4, got {slice_bounds}'
        )

    def test_list_slice_omitted_start_uses_zero(self):
        """``xs[:3]`` on a size-5 list has size 3."""

        @fp.fpy
        def f() -> list[fp.Real]:
            xs = [1.0, 2.0, 3.0, 4.0, 5.0]
            return xs[:3]

        info = self._run(f)
        slice_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'ListSlice'
        ]
        assert len(slice_bounds) == 1 and slice_bounds[0].size == 3

    def test_list_slice_unknown_list_size_falls_back_to_none(self):
        """
        When the underlying list's size is unknown, the slice size is
        also unknown — *not* the original list's size (which would be
        unsoundly equal to ``None``-and-also-claim-precision).
        """

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[1:]

        info = self._run(f)
        slice_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'ListSlice'
        ]
        assert len(slice_bounds) == 1 and slice_bounds[0].size is None

    def test_list_slice_start_past_end_is_unknown(self):
        """
        ``xs[10:]`` on a size-5 list is invalid under FPy's strict
        slicing semantics (start past end of list).  The runtime raises
        ``IndexError``; the static size analysis reports ``None``
        (unknown) rather than committing to Python's clamping value of 0.
        """

        @fp.fpy
        def f() -> list[fp.Real]:
            xs = [1.0, 2.0, 3.0, 4.0, 5.0]
            return xs[10:]

        info = self._run(f)
        slice_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'ListSlice'
        ]
        assert len(slice_bounds) == 1 and slice_bounds[0].size is None

    def test_list_slice_concrete_bounds_resolves_without_list_size(self):
        """
        With strict semantics, ``xs[1:3]`` has size exactly ``3 - 1 = 2``
        when both bounds are concrete — *regardless of whether the
        underlying list's size is statically known*.  (At runtime, if
        ``len(xs) < 3``, the slice raises; the static analysis trusts
        the runtime check and reports the size the slice *would* have.)
        """

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[1:3]

        info = self._run(f)
        slice_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'ListSlice'
        ]
        assert len(slice_bounds) == 1 and slice_bounds[0].size == 2

    # ------------------------------------------------------------------
    # IndexedAssign as a fresh SSA def

    def test_indexed_assign_creates_fresh_def(self):
        """
        ``xs[i] = e`` is treated as ``xs = update(xs, [i], e)``: the
        mutation produces a *new* SSA definition of ``xs``.  Both the
        pre- and post-mutation defs appear in ``by_def`` with their own
        bounds.
        """

        @fp.fpy
        def f() -> list[list[fp.Real]]:
            xs = [[1.0], [1.0]]
            xs[0] = [1.0, 2.0]
            return xs

        info = self._run(f)
        xs_bounds = [b for d, b in info.by_def.items() if d.name.base == 'xs']
        # Two distinct defs for xs: original and post-IndexedAssign.
        assert len(xs_bounds) == 2, f'expected 2 xs defs, got {xs_bounds}'

    def test_indexed_assign_widens_inner_element_size(self):
        """
        Element mutation that inserts a list of a different size widens
        the inner-element bound at the new def to ``None`` (the flat
        lattice's "unknown" top), while the outer size is preserved.
        """

        @fp.fpy
        def f() -> list[list[fp.Real]]:
            xs = [[1.0], [1.0]]      # outer 2, inner 1
            xs[0] = [1.0, 2.0]       # widens inner size to None
            return xs

        info = self._run(f)
        xs_bounds = [b for d, b in info.by_def.items() if d.name.base == 'xs']
        # Original def: inner size 1, outer size 2.
        original = [
            b for b in xs_bounds
            if isinstance(b, ListSize)
            and b.size == 2
            and isinstance(b.elt, ListSize)
            and b.elt.size == 1
        ]
        # Post-mutation def: inner size widened to None, outer still 2.
        widened = [
            b for b in xs_bounds
            if isinstance(b, ListSize)
            and b.size == 2
            and isinstance(b.elt, ListSize)
            and b.elt.size is None
        ]
        assert original, f'expected pre-mutation def with inner size=1, got {xs_bounds}'
        assert widened, f'expected post-mutation def with inner size=None, got {xs_bounds}'

    def test_indexed_assign_in_loop_propagates_through_phi(self):
        """
        ``xs[0] = …`` inside a loop body must be observed by the loop
        phi.  Without the SSA-fresh-def treatment of ``IndexedAssign``,
        the body would produce no new def and the phi would degenerate
        to ``phi(xs, xs)``.  With the fresh def + the loop fixpoint, the
        widening propagates through.
        """

        @fp.fpy
        def f(n: fp.Real) -> list[list[fp.Real]]:
            xs = [[1.0], [1.0]]
            for _ in range(0, 1):
                xs[0] = [1.0, 2.0, 3.0]
            return xs

        info = self._run(f)
        xs_bounds = [b for d, b in info.by_def.items() if d.name.base == 'xs']
        # The loop phi (and post-loop reads) must observe the inner
        # element widening induced by the body's IndexedAssign.
        widened = [
            b for b in xs_bounds
            if isinstance(b, ListSize)
            and b.size == 2
            and isinstance(b.elt, ListSize)
            and b.elt.size is None
        ]
        assert widened, (
            f'expected loop phi/post-loop xs to widen inner size to None, '
            f'got {xs_bounds}'
        )

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
