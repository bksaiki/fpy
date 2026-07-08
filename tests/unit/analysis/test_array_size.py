"""
Unit tests for :class:`ArraySizeInfer`.
"""

import fpy2 as fp
import pytest

from hypothesis import given, settings, strategies as st

from fpy2.analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    ListSize,
    TupleSize,
)
from fpy2.analysis.array_size import concrete_size, is_size_eq
from fpy2.utils import NamedId


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
        """A list-of-real argument has a ListSize whose size is a fresh
        size variable (its length is fixed per call, just unknown)."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return xs[0]

        info = self._run(f)
        xs_bounds = [b for d, b in info.by_def.items() if d.name.base == 'xs']
        assert len(xs_bounds) == 1
        bound = xs_bounds[0]
        assert isinstance(bound, ListSize)
        assert bound.elt is None
        assert isinstance(bound.size, NamedId)

    def test_arg_concrete_length_annotation_seeds_size(self):
        """A ``list[Real]`` argument annotated with a concrete length
        (e.g. an FPCore fixed dimension) seeds a concrete array-size."""
        from fpy2.ast.fpyast import ListTypeAnn, RealTypeAnn

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return xs[0]

        f.ast.args[0].type = ListTypeAnn(RealTypeAnn(None, None), 16, None)
        info = self._run(f)
        xs_bound = [b for d, b in info.by_def.items() if d.name.base == 'xs'][0]
        assert concrete_size(xs_bound.size) == 16

    def test_shared_symbolic_length_annotation_unifies(self):
        """Two list args annotated with the *same* symbolic length share a
        size class (provably-equal lengths)."""
        from fpy2.ast.fpyast import ListTypeAnn, RealTypeAnn

        n = NamedId('N')

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            return xs[0]

        f.ast.args[0].type = ListTypeAnn(RealTypeAnn(None, None), n, None)
        f.ast.args[1].type = ListTypeAnn(RealTypeAnn(None, None), n, None)
        info = self._run(f)
        xs_b = [b for d, b in info.by_def.items() if d.name.base == 'xs'][0]
        ys_b = [b for d, b in info.by_def.items() if d.name.base == 'ys'][0]
        assert is_size_eq(xs_b, ys_b)

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

    def test_range1_negative_is_empty(self):
        """``range(n)`` with ``n <= 0`` is empty (size 0), mirroring
        Python — the size must be clamped, not negative."""

        @fp.fpy
        def f() -> list[fp.Real]:
            return [0.0 for _ in range(-5)]

        info = self._run(f)
        range_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'Range1'
        ]
        assert range_bounds and range_bounds[0].size == 0

    def test_range2_flipped_is_empty(self):
        """``range(start, stop)`` with ``start >= stop`` is empty (size 0)."""

        @fp.fpy
        def f() -> list[fp.Real]:
            return [0.0 for _ in range(5, 0)]

        info = self._run(f)
        range_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'Range2'
        ]
        assert range_bounds and range_bounds[0].size == 0

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

    def test_zip_equal_known_sizes(self):
        """``zip`` of equal-length known lists has that size, with a
        per-input tuple element bound."""

        @fp.fpy
        def f() -> list[tuple[fp.Real, fp.Real]]:
            return [t for t in zip([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])]

        info = self._run(f)
        zip_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'Zip'
        ]
        assert len(zip_bounds) == 1
        bound = zip_bounds[0]
        assert isinstance(bound, ListSize)
        assert bound.size == 3
        # element is a 2-tuple (one slot per input list)
        assert isinstance(bound.elt, TupleSize)
        assert len(bound.elt.elts) == 2

    def test_zip_unequal_known_sizes_is_unknown(self):
        """``zip`` is strict (raises on length mismatch), so it must NOT
        report the *min* of conflicting known sizes — the result is
        unreachable, hence unknown."""

        @fp.fpy
        def f() -> list[tuple[fp.Real, fp.Real]]:
            return [t for t in zip([1.0, 2.0, 3.0], [4.0, 5.0])]

        info = self._run(f)
        zip_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'Zip'
        ]
        assert len(zip_bounds) == 1
        bound = zip_bounds[0]
        assert isinstance(bound, ListSize)
        # NOT 2 (the min) — strict zip raises on mismatch.
        assert bound.size is None

    def test_zip_unknown_with_known_pins_to_known(self):
        """``zip`` is strict, so an unknown-size input zipped with a
        known-size one must equal it (or the zip raises): the result size
        is the known length, and the symbolic input is pinned to it."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[tuple[fp.Real, fp.Real]]:
            return [t for t in zip(xs, [1.0, 2.0, 3.0])]

        info = self._run(f)
        zip_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'Zip'
        ]
        assert len(zip_bounds) == 1
        bound = zip_bounds[0]
        assert isinstance(bound, ListSize)
        assert bound.size == 3
        # the strict zip also pins `xs` to length 3
        xs_bound = [b for d, b in info.by_def.items() if d.name.base == 'xs'][0]
        assert concrete_size(xs_bound.size) == 3

    def test_zip_single_arg_preserves_size(self):
        """A 1-argument ``zip`` preserves the input's known size."""

        @fp.fpy
        def f() -> list[tuple[fp.Real]]:
            return [t for t in zip([1.0, 2.0, 3.0, 4.0])]

        info = self._run(f)
        zip_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'Zip'
        ]
        assert len(zip_bounds) == 1
        bound = zip_bounds[0]
        assert isinstance(bound, ListSize)
        assert bound.size == 4

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

    def _slice_bound(self, f):
        info = self._run(f)
        slice_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'ListSlice'
        ]
        assert len(slice_bounds) == 1
        assert isinstance(slice_bounds[0], ListSize)
        return slice_bounds[0]

    def test_list_slice_symbolic_offset_under_real(self):
        """``x[i : i + 16]`` on an unknown-size list has size 16: the
        symbolic base ``i`` cancels in ``(i + 16) - i``.  Sound only
        because the ``+`` runs under the exact ``REAL`` context."""

        @fp.fpy
        def f(x: list[fp.Real], i: fp.Real) -> list[fp.Real]:
            with fp.REAL:
                y = x[i:i + 16]
            return y

        assert self._slice_bound(f).size == 16

    def test_list_slice_symbolic_offset_subtraction_under_real(self):
        """Both endpoints offset from the same base: ``x[i-4 : i+4]`` -> 8."""

        @fp.fpy
        def f(x: list[fp.Real], i: fp.Real) -> list[fp.Real]:
            with fp.REAL:
                y = x[i - 4:i + 4]
            return y

        assert self._slice_bound(f).size == 8

    def test_list_slice_symbolic_offset_constant_on_left(self):
        """``x[i : 16 + i]`` -> 16 (constant operand on either side)."""

        @fp.fpy
        def f(x: list[fp.Real], i: fp.Real) -> list[fp.Real]:
            with fp.REAL:
                y = x[i:16 + i]
            return y

        assert self._slice_bound(f).size == 16

    def test_list_slice_symbolic_offset_not_under_real_is_unknown(self):
        """Without an exact context, rounding could perturb ``i + 16``,
        so the difference is not provably constant -> unknown."""

        @fp.fpy
        def f(x: list[fp.Real], i: fp.Real) -> list[fp.Real]:
            return x[i:i + 16]

        assert self._slice_bound(f).size is None

    def test_list_slice_symbolic_offset_inverted_is_unknown(self):
        """``x[i+16 : i]`` has start > stop: strict slicing always raises,
        so report unknown rather than a negative size."""

        @fp.fpy
        def f(x: list[fp.Real], i: fp.Real) -> list[fp.Real]:
            with fp.REAL:
                y = x[i + 16:i]
            return y

        assert self._slice_bound(f).size is None

    def test_list_slice_symbolic_offset_nonconstant_is_unknown(self):
        """``x[i : i + k]`` with symbolic ``k`` can't be pinned."""

        @fp.fpy
        def f(x: list[fp.Real], i: fp.Real, k: fp.Real) -> list[fp.Real]:
            with fp.REAL:
                y = x[i:i + k]
            return y

        assert self._slice_bound(f).size is None

    # ------------------------------------------------------------------
    # range() with symbolic-offset bounds (shares the slice affine logic)

    def _range_bound(self, f, tyname):
        info = self._run(f)
        bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == tyname
        ]
        assert len(bounds) == 1
        assert isinstance(bounds[0], ListSize)
        return bounds[0]

    def test_range2_symbolic_offset_under_real(self):
        """``range(i, i + 16)`` -> 16: the symbolic base cancels, and the
        ``+`` runs under the exact ``REAL`` context."""

        @fp.fpy
        def f(i: fp.Real) -> list[fp.Real]:
            with fp.REAL:
                ys = [0.0 for _ in range(i, i + 16)]
            return ys

        assert self._range_bound(f, 'Range2').size == 16

    def test_range2_symbolic_offset_not_under_real_is_unknown(self):
        """Without an exact context, ``i + 16`` may round, so the span
        isn't provably constant -> unknown."""

        @fp.fpy
        def f(i: fp.Real) -> list[fp.Real]:
            return [0.0 for _ in range(i, i + 16)]

        assert self._range_bound(f, 'Range2').size is None

    def test_range2_symbolic_offset_inverted_is_empty(self):
        """``range(i + 16, i)`` is empty (start > stop) for any ``i`` —
        range clamps to 0 (unlike slicing, which raises)."""

        @fp.fpy
        def f(i: fp.Real) -> list[fp.Real]:
            with fp.REAL:
                ys = [0.0 for _ in range(i + 16, i)]
            return ys

        assert self._range_bound(f, 'Range2').size == 0

    def test_range3_symbolic_offset_with_step(self):
        """``range(i, i + 16, 2)`` -> 8: span 16 stepped by 2."""

        @fp.fpy
        def f(i: fp.Real) -> list[fp.Real]:
            with fp.REAL:
                ys = [0.0 for _ in range(i, i + 16, 2)]
            return ys

        assert self._range_bound(f, 'Range3').size == 8

    def test_range1_len_of_known_list(self):
        """``range(len(xs))`` -> size of ``xs`` when statically known."""

        @fp.fpy
        def f() -> list[fp.Real]:
            xs = [1.0, 2.0, 3.0, 4.0]
            return [0.0 for _ in range(len(xs))]

        # the inner range has the size; the comprehension echoes it
        assert self._range_bound(f, 'Range1').size == 4

    def test_slice_len_of_known_list(self):
        """``xs[0:len(xs)]`` -> full size via ``len`` of a known list."""

        @fp.fpy
        def f() -> list[fp.Real]:
            xs = [1.0, 2.0, 3.0, 4.0, 5.0]
            return xs[0:len(xs)]

        assert self._slice_bound(f).size == 5

    def test_range1_len_of_unknown_list_is_unknown(self):
        """``range(len(xs))`` for an unknown-size ``xs`` stays unknown."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            return [0.0 for _ in range(len(xs))]

        assert self._range_bound(f, 'Range1').size is None

    def test_range1_dim_is_known_structurally(self):
        """``dim(xs)`` is the nesting depth — known even when sizes aren't,
        so ``range(dim(xs))`` is a known size."""

        @fp.fpy
        def f(xs: list[list[fp.Real]]) -> list[fp.Real]:
            return [0.0 for _ in range(fp.dim(xs))]

        assert self._range_bound(f, 'Range1').size == 2

    def test_range1_size_of_known_dimension(self):
        """``size(xs, d)`` folds to the length of dimension ``d`` when
        statically known."""

        @fp.fpy
        def f() -> list[fp.Real]:
            xs = fp.empty(4, 3)
            return [0.0 for _ in range(fp.size(xs, 1))]

        assert self._range_bound(f, 'Range1').size == 3

    def test_range1_size_of_unknown_dimension_is_unknown(self):
        """``size(xs, d)`` of an unknown-size dimension stays unknown."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            return [0.0 for _ in range(fp.size(xs, 0))]

        assert self._range_bound(f, 'Range1').size is None

    def test_slice_dim_of_known_list(self):
        """``dim`` folds inside a slice bound too."""

        @fp.fpy
        def f(xs: list[list[fp.Real]]) -> list[list[fp.Real]]:
            return xs[0:fp.dim(xs)]

        assert self._slice_bound(f).size == 2

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

    # ------------------------------------------------------------------
    # Multi-return: ``_visit_return`` unifies the list-size bound
    # across every reachable return site.

    def test_ret_size_multi_return_equal_sizes(self):
        """Two return paths producing lists of the same concrete
        size: the unified ``ret_size`` keeps the concrete size."""

        @fp.fpy
        def f(c: bool) -> list[fp.Real]:
            if c:
                return [1.0, 2.0, 3.0]
            else:
                return [4.0, 5.0, 6.0]

        info = self._run(f)
        assert isinstance(info.ret_size, ListSize)
        assert info.ret_size.size == 3

    def test_ret_size_multi_return_unequal_sizes(self):
        """Two return paths producing lists of *different* concrete
        sizes: the unified ``ret_size`` widens the size dimension to
        ``None`` (size unknown at the join point) while keeping the
        ``ListSize`` shape."""

        @fp.fpy
        def f(c: bool) -> list[fp.Real]:
            if c:
                return [1.0, 2.0]
            else:
                return [1.0, 2.0, 3.0]

        info = self._run(f)
        assert isinstance(info.ret_size, ListSize)
        assert info.ret_size.size is None

    def test_ret_size_early_return_unifies_with_trailing(self):
        """Early-return pattern (``if c: return [..]; return [..]``).
        Both paths produce same-size lists; ``ret_size`` reflects
        the join."""

        @fp.fpy
        def f(c: bool) -> list[fp.Real]:
            if c:
                return [1.0, 2.0]
            return [3.0, 4.0]

        info = self._run(f)
        assert isinstance(info.ret_size, ListSize)
        assert info.ret_size.size == 2

    # ------------------------------------------------------------------
    # Call-result size propagation

    def test_call_propagates_known_return_size(self):
        """A call to a callee whose return size is statically known
        adopts that size at the call site (and onto a binding of it)."""

        @fp.fpy
        def callee() -> list[fp.Real]:
            return [1.0, 2.0, 3.0]

        @fp.fpy
        def caller() -> list[fp.Real]:
            ys = callee()
            return ys

        info = self._run(caller)
        call_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'Call'
        ]
        assert len(call_bounds) == 1
        assert isinstance(call_bounds[0], ListSize)
        assert call_bounds[0].size == 3
        # ...and it survives onto the binding / return.
        assert isinstance(info.ret_size, ListSize)
        assert info.ret_size.size == 3

    def test_call_with_arg_dependent_size_stays_unknown(self):
        """A callee whose return size depends on its arguments has no
        statically-known size; the call site must not invent one."""

        @fp.fpy
        def callee(xs: list[fp.Real]) -> list[fp.Real]:
            return [x for x in xs]

        @fp.fpy
        def caller(xs: list[fp.Real]) -> list[fp.Real]:
            return callee(xs)

        info = self._run(caller)
        call_bounds = [
            b for e, b in info.by_expr.items()
            if type(e).__name__ == 'Call'
        ]
        assert len(call_bounds) == 1
        assert isinstance(call_bounds[0], ListSize)
        assert call_bounds[0].size is None

    def test_call_propagates_transitively(self):
        """A known size flows through a chain of calls."""

        @fp.fpy
        def leaf() -> list[fp.Real]:
            return [1.0, 2.0, 3.0, 4.0, 5.0]

        @fp.fpy
        def mid() -> list[fp.Real]:
            return leaf()

        @fp.fpy
        def top() -> list[fp.Real]:
            return mid()

        info = self._run(top)
        assert isinstance(info.ret_size, ListSize)
        assert info.ret_size.size == 5

    def test_call_propagates_nested_sizes(self):
        """Nested known sizes (``empty(2, 3)``) propagate per-dimension
        through a call via the structural overlay."""

        @fp.fpy
        def grid() -> list[list[fp.Real]]:
            return fp.empty(2, 3)

        @fp.fpy
        def use_grid() -> list[list[fp.Real]]:
            return grid()

        info = self._run(use_grid)
        outer = info.ret_size
        assert isinstance(outer, ListSize)
        assert outer.size == 2
        assert isinstance(outer.elt, ListSize)
        assert outer.elt.size == 3

    def test_primitive_call_has_no_size(self):
        """Calling a primitive (non-FPy-function) doesn't crash and
        yields no list-size info."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return fp.sqrt(x)

        info = self._run(f)
        assert info.ret_size is None

    # ------------------------------------------------------------------
    # Symbolic sizes (union-find): equal-but-unknown lengths

    @staticmethod
    def _def_size(info, name):
        bounds = [b for d, b in info.by_def.items() if d.name.base == name]
        assert bounds, f'no def found for {name}'
        return bounds[0]

    def test_rebind_preserves_symbol(self):
        """``ys = xs`` gives ``ys`` the same symbolic length as ``xs``."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            ys = xs
            return ys

        info = self._run(f)
        xs_b, ys_b = self._def_size(info, 'xs'), self._def_size(info, 'ys')
        assert is_size_eq(xs_b, ys_b)

    def test_comprehension_over_argument_is_equivalent(self):
        """``[f(x) for x in xs]`` has the same length as ``xs``."""
        from fpy2.utils import NamedId

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                ys = [x for x in xs]
            return ys

        info = self._run(f)
        assert isinstance(info.ret_size, ListSize)
        assert isinstance(info.ret_size.size, NamedId)
        assert is_size_eq(info.ret_size, self._def_size(info, 'xs'))

    def test_enumerate_over_argument_is_equivalent(self):
        """``enumerate(xs)`` preserves ``xs``'s symbolic length."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            ys = [x for _, x in enumerate(xs)]
            return ys

        info = self._run(f)
        assert is_size_eq(info.ret_size, self._def_size(info, 'xs'))

    def test_full_slice_preserves_symbol(self):
        """``xs[:]`` spans the whole list, keeping its symbolic length."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[:]

        info = self._run(f)
        assert is_size_eq(info.ret_size, self._def_size(info, 'xs'))

    def test_zip_two_unknowns_merges_classes(self):
        """``zip(xs, ys)`` is strict, so on the non-raising path
        ``len(xs) == len(ys)``: their symbolic classes are merged."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for a, b in zip(xs, ys):
                with fp.FP64:
                    acc = acc + a
            return acc

        info = self._run(f)
        xs_b, ys_b = self._def_size(info, 'xs'), self._def_size(info, 'ys')
        assert xs_b.size == ys_b.size

    def test_distinct_arguments_are_not_equivalent(self):
        """Two independent list arguments get distinct, non-equivalent
        symbols (no false equality)."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            return xs[0]

        info = self._run(f)
        xs_b, ys_b = self._def_size(info, 'xs'), self._def_size(info, 'ys')
        assert not is_size_eq(xs_b, ys_b)

    def test_loop_rebind_keeps_symbol(self):
        """A loop that re-binds a list via a size-preserving op keeps the
        symbolic length across the fixpoint."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            ys = xs
            for _ in range(3):
                ys = ys
            return ys

        info = self._run(f)
        assert is_size_eq(info.ret_size, self._def_size(info, 'xs'))

    def test_concrete_size_helper(self):
        """``concrete_size`` resolves ints directly and pinned symbols."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            acc = 0.0
            for t in zip(xs, [1.0, 2.0]):   # pins xs to 2
                with fp.FP64:
                    acc = acc + 1.0
            return acc

        info = self._run(f)
        xs_b = self._def_size(info, 'xs')
        assert concrete_size(xs_b.size) == 2
        assert concrete_size(7) == 7
        assert concrete_size(None) is None

    # ------------------------------------------------------------------
    # Assertion-seeded size equalities

    def test_assert_len_eq_merges(self):
        """``assert len(xs) == len(ys)`` makes their sizes equivalent."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            assert len(xs) == len(ys)
            return xs[0]

        info = self._run(f)
        assert is_size_eq(self._def_size(info, 'xs'),
                          self._def_size(info, 'ys'))

    def test_assert_len_eq_constant_pins(self):
        """``assert len(xs) == 16`` pins ``xs`` to a concrete size."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            assert len(xs) == 16
            return xs[0]

        info = self._run(f)
        assert concrete_size(self._def_size(info, 'xs').size) == 16

    def test_assert_len_eq_transitive(self):
        """Two asserts chain: ``xs`` and ``zs`` become equivalent."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real], zs: list[fp.Real]) -> fp.Real:
            assert len(xs) == len(ys)
            assert len(ys) == len(zs)
            return xs[0]

        info = self._run(f)
        assert is_size_eq(self._def_size(info, 'xs'),
                          self._def_size(info, 'zs'))

    def test_conditional_assert_does_not_merge(self):
        """An assert inside an ``if`` need not hold on every execution, so
        it must NOT constrain sizes globally (soundness)."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real], c: bool) -> fp.Real:
            if c:
                assert len(xs) == len(ys)
            return xs[0]

        info = self._run(f)
        assert not is_size_eq(self._def_size(info, 'xs'),
                              self._def_size(info, 'ys'))

    def test_loop_assert_does_not_merge(self):
        """An assert inside a loop body is conditional (the loop may run
        zero times), so it must not constrain sizes globally."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            for _ in range(0):
                assert len(xs) == len(ys)
            return xs[0]

        info = self._run(f)
        assert not is_size_eq(self._def_size(info, 'xs'),
                              self._def_size(info, 'ys'))

    def test_assert_under_context_still_merges(self):
        """A ``with`` block executes unconditionally, so an assert inside
        one still seeds the equality."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                assert len(xs) == len(ys)
            return xs[0]

        info = self._run(f)
        assert is_size_eq(self._def_size(info, 'xs'),
                          self._def_size(info, 'ys'))


# ----------------------------------------------------------------------
# Differential soundness: inferred sizes vs. actual runtime lengths.
#
# Each function is analyzed statically, then *run* through the
# interpreter on concrete inputs; the inferred sizes are checked against
# the real runtime list lengths.  Two invariants must hold for **every**
# input (a single counterexample is a soundness bug):
#
#   1. concrete-size soundness — a static concrete ``int`` size (directly
#      or via a pinned size variable) equals the runtime list length;
#   2. equivalence soundness — ``is_size_eq`` bounds have equal runtime
#      lengths.
#
# Only the ``static-fact => runtime-fact`` direction is checked (the
# analysis is conservative, so the converse may legitimately fail).
# Observables are the list-typed arguments and the return value.  Inputs
# that raise at runtime (strict-``zip`` mismatch, out-of-range slice) are
# skipped — the static claim about them is vacuous.

@fp.fpy
def _d_literal() -> list[fp.Real]:
    return [1.0, 2.0, 3.0]


@fp.fpy
def _d_empty_literal() -> list[fp.Real]:
    return []


@fp.fpy
def _d_nested_literal() -> list[list[fp.Real]]:
    return [[1.0, 2.0], [3.0, 4.0]]


@fp.fpy
def _d_range1() -> list[fp.Real]:
    return [0.0 for _ in range(5)]


@fp.fpy
def _d_range1_negative() -> list[fp.Real]:
    return [0.0 for _ in range(-5)]


@fp.fpy
def _d_range2() -> list[fp.Real]:
    return [0.0 for _ in range(2, 7)]


@fp.fpy
def _d_range2_flipped() -> list[fp.Real]:
    return [0.0 for _ in range(7, 2)]


@fp.fpy
def _d_range3() -> list[fp.Real]:
    return [0.0 for _ in range(0, 10, 3)]


@fp.fpy
def _d_range_len_literal() -> list[fp.Real]:
    xs = [1.0, 2.0, 3.0, 4.0]
    return [0.0 for _ in range(len(xs))]


@fp.fpy
def _d_comp_over_arg(xs: list[fp.Real]) -> list[fp.Real]:
    return [x for x in xs]


@fp.fpy
def _d_comp_product() -> list[fp.Real]:
    return [0.0 for _ in range(3) for _ in range(4)]


@fp.fpy
def _d_enumerate(xs: list[fp.Real]) -> list[fp.Real]:
    return [x for _, x in enumerate(xs)]


@fp.fpy
def _d_zip_args(xs: list[fp.Real], ys: list[fp.Real]) -> list[tuple[fp.Real, fp.Real]]:
    return [t for t in zip(xs, ys)]


@fp.fpy
def _d_zip_literal(xs: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for _ in zip(xs, [1.0, 2.0, 3.0]):
        with fp.FP64:
            acc = acc + 1.0
    return acc


@fp.fpy
def _d_slice_concrete() -> list[fp.Real]:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    return xs[1:4]


@fp.fpy
def _d_slice_full(xs: list[fp.Real]) -> list[fp.Real]:
    return xs[:]


@fp.fpy
def _d_slice_prefix(xs: list[fp.Real]) -> list[fp.Real]:
    return xs[0:2]


@fp.fpy
def _d_slice_sym_offset(xs: list[fp.Real], i: fp.Real) -> list[fp.Real]:
    with fp.REAL:
        ys = xs[i:i + 2]
    return ys


@fp.fpy
def _d_range_sym_offset(i: fp.Real) -> list[fp.Real]:
    with fp.REAL:
        ys = [0.0 for _ in range(i, i + 4)]
    return ys


@fp.fpy
def _d_empty_2d() -> list[list[fp.Real]]:
    return fp.empty(2, 3)


@fp.fpy
def _d_cond_equal(b: bool) -> list[fp.Real]:
    if b:
        return [1.0, 2.0]
    else:
        return [3.0, 4.0]


@fp.fpy
def _d_cond_unequal(b: bool) -> list[fp.Real]:
    if b:
        return [1.0]
    else:
        return [2.0, 3.0]


@fp.fpy
def _d_rebind(xs: list[fp.Real]) -> list[fp.Real]:
    ys = xs
    return ys


@fp.fpy
def _d_loop_preserve(xs: list[fp.Real]) -> list[fp.Real]:
    ys = xs
    for _ in range(3):
        ys = ys
    return ys


@fp.fpy
def _d_indexed_assign(xs: list[fp.Real]) -> list[fp.Real]:
    ys = [0.0 for _ in xs]
    for i, x in enumerate(xs):
        ys[i] = x
    return ys


@fp.fpy
def _d_callee_known() -> list[fp.Real]:
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@fp.fpy
def _d_call_known() -> list[fp.Real]:
    return _d_callee_known()


@fp.fpy
def _d_assert_eq(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
    assert len(xs) == len(ys)   # merges xs ~ ys; only equal-length inputs return
    return ys


def _floats(n):
    return [float(k) for k in range(n)]


_DIFFERENTIAL_SPECS = [
    (_d_literal, [()]),
    (_d_empty_literal, [()]),
    (_d_nested_literal, [()]),
    (_d_range1, [()]),
    (_d_range1_negative, [()]),
    (_d_range2, [()]),
    (_d_range2_flipped, [()]),
    (_d_range3, [()]),
    (_d_range_len_literal, [()]),
    (_d_comp_over_arg, [(_floats(0),), (_floats(1),), (_floats(3),), (_floats(10),)]),
    (_d_comp_product, [()]),
    (_d_enumerate, [(_floats(0),), (_floats(1),), (_floats(7),)]),
    (_d_zip_args, [(_floats(0), _floats(0)), (_floats(4), _floats(4)), (_floats(9), _floats(9))]),
    (_d_zip_literal, [(_floats(3),)]),
    (_d_slice_concrete, [()]),
    (_d_slice_full, [(_floats(0),), (_floats(1),), (_floats(6),)]),
    (_d_slice_prefix, [(_floats(2),), (_floats(5),)]),
    (_d_slice_sym_offset, [(_floats(5), 1), (_floats(10), 3), (_floats(2), 0)]),
    (_d_range_sym_offset, [(0,), (5,), (-3,)]),
    (_d_empty_2d, [()]),
    (_d_cond_equal, [(True,), (False,)]),
    (_d_cond_unequal, [(True,), (False,)]),
    (_d_rebind, [(_floats(0),), (_floats(1),), (_floats(8),)]),
    (_d_loop_preserve, [(_floats(0),), (_floats(4),)]),
    (_d_indexed_assign, [(_floats(0),), (_floats(1),), (_floats(6),)]),
    (_d_call_known, [()]),
    # equal-length inputs (unequal would raise the assert and be skipped)
    (_d_assert_eq, [(_floats(0), _floats(0)), (_floats(3), _floats(3)), (_floats(8), _floats(8))]),
]


class TestArraySizeDifferential:
    """Compare inferred sizes against runtime list lengths."""

    @classmethod
    def _runtime_tree(cls, v):
        """Nested-length shape of a runtime value: ``('list', n,
        inner|None)`` / ``('tuple', (sub, ...))`` / ``('scalar',)``."""
        if isinstance(v, list):
            return ('list', len(v), cls._runtime_tree(v[0]) if v else None)
        if isinstance(v, tuple):
            return ('tuple', tuple(cls._runtime_tree(e) for e in v))
        return ('scalar',)

    def _check_concrete(self, bound, tree, path):
        """Every concrete static size must match the runtime length."""
        if bound is None:
            return
        if isinstance(bound, ListSize):
            assert tree[0] == 'list', f'{path}: static list vs runtime {tree[0]}'
            c = concrete_size(bound.size)
            if c is not None:
                assert c == tree[1], (
                    f'{path}: static size {c} != runtime length {tree[1]}'
                )
            if tree[2] is not None:
                self._check_concrete(bound.elt, tree[2], path + '[0]')
        elif isinstance(bound, TupleSize):
            assert tree[0] == 'tuple', f'{path}: static tuple vs runtime {tree[0]}'
            assert len(bound.elts) == len(tree[1]), f'{path}: tuple arity'
            for k, (b, t) in enumerate(zip(bound.elts, tree[1])):
                self._check_concrete(b, t, f'{path}.{k}')

    @staticmethod
    def _observables(func, info, inputs, result):
        obs = []
        for arg, val in zip(func.ast.args, inputs):
            if isinstance(arg.name, NamedId):
                d = info.def_use.find_def_from_site(arg.name, arg)
                obs.append((str(arg.name), info.by_def[d], val))
        obs.append(('<ret>', info.ret_size, result))
        return obs

    def _assert_sound(self, func, input_sets):
        info = ArraySizeInfer.analyze(func.ast)
        ran_any = False
        for inputs in input_sets:
            try:
                result = func(*inputs)
            except Exception:
                continue  # invalid input -> static claim is vacuous
            ran_any = True
            obs = self._observables(func, info, inputs, result)

            # (1) concrete-size soundness on every observable
            for label, bound, val in obs:
                self._check_concrete(bound, self._runtime_tree(val), label)

            # (2) equivalence soundness: equal bounds => equal runtime lengths
            for i in range(len(obs)):
                for j in range(i + 1, len(obs)):
                    _, bi, vi = obs[i]
                    _, bj, vj = obs[j]
                    if (isinstance(vi, list) and isinstance(vj, list)
                            and is_size_eq(bi, bj)):
                        assert len(vi) == len(vj), (
                            f'{obs[i][0]} ~ {obs[j][0]} are is_size_eq but '
                            f'runtime lengths differ: {len(vi)} != {len(vj)}'
                        )
        assert ran_any, f'no input set ran without raising for {func.name}'

    @pytest.mark.parametrize(
        'func,input_sets', _DIFFERENTIAL_SPECS,
        ids=[f.name for f, _ in _DIFFERENTIAL_SPECS],
    )
    def test_sound(self, func, input_sets):
        self._assert_sound(func, input_sets)

    # -- hypothesis: fuzz argument lengths for size-linked functions ----

    @settings(max_examples=60)
    @given(n=st.integers(min_value=0, max_value=64))
    def test_fuzz_comp_over_arg(self, n):
        self._assert_sound(_d_comp_over_arg, [(_floats(n),)])

    @settings(max_examples=60)
    @given(n=st.integers(min_value=0, max_value=64))
    def test_fuzz_enumerate(self, n):
        self._assert_sound(_d_enumerate, [(_floats(n),)])

    @settings(max_examples=60)
    @given(n=st.integers(min_value=0, max_value=64))
    def test_fuzz_rebind(self, n):
        self._assert_sound(_d_rebind, [(_floats(n),)])

    @settings(max_examples=60)
    @given(n=st.integers(min_value=0, max_value=64))
    def test_fuzz_full_slice(self, n):
        self._assert_sound(_d_slice_full, [(_floats(n),)])

    @settings(max_examples=60)
    @given(n=st.integers(min_value=0, max_value=64))
    def test_fuzz_indexed_assign(self, n):
        self._assert_sound(_d_indexed_assign, [(_floats(n),)])

    @settings(max_examples=60)
    @given(n=st.integers(min_value=0, max_value=64))
    def test_fuzz_zip_args_equal(self, n):
        self._assert_sound(_d_zip_args, [(_floats(n), _floats(n))])

    @settings(max_examples=60)
    @given(i=st.integers(min_value=0, max_value=32))
    def test_fuzz_slice_sym_offset(self, i):
        self._assert_sound(_d_slice_sym_offset, [(_floats(i + 2), i)])

    @settings(max_examples=60)
    @given(i=st.integers(min_value=-8, max_value=32))
    def test_fuzz_range_sym_offset(self, i):
        self._assert_sound(_d_range_sym_offset, [(i,)])


class TestTupleAccessorSizes:
    """``fst`` / ``snd`` project the operand's :class:`TupleSize`, so a
    known list length survives a tuple access."""

    @staticmethod
    def _run(func: fp.Function) -> ArraySizeAnalysis:
        return ArraySizeInfer.analyze(func.ast)

    @staticmethod
    def _pin(func: fp.Function, lengths: dict[str, int]) -> None:
        """Pin concrete list lengths onto the named list arguments."""
        from fpy2.ast.fpyast import ListTypeAnn, RealTypeAnn
        for arg in func.ast.args:
            n = lengths.get(arg.name.base)
            if n is not None:
                arg.type = ListTypeAnn(RealTypeAnn(None, None), n, None)

    @staticmethod
    def _bound(info: ArraySizeAnalysis, name: str):
        return [b for d, b in info.by_def.items() if d.name.base == name][0]

    def test_fst_projects_first_component_size(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]):
            t = (xs, ys)
            u = fp.fst(t)
            return u

        self._pin(f, {'xs': 16, 'ys': 8})
        u = self._bound(self._run(f), 'u')
        assert isinstance(u, ListSize)
        assert concrete_size(u.size) == 16

    def test_snd_of_pair_projects_second_size(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]):
            t = (xs, ys)
            u = fp.snd(t)        # pair -> bare second element
            return u

        self._pin(f, {'xs': 16, 'ys': 8})
        u = self._bound(self._run(f), 'u')
        assert isinstance(u, ListSize)
        assert concrete_size(u.size) == 8
