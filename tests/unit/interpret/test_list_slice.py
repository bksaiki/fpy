"""
Unit tests for FPy's strict list-slicing semantics.

Unlike Python's clamping behaviour, FPy requires ``xs[start:stop]`` to
extract a block of *exactly* ``stop - start`` elements — out-of-range
bounds raise ``IndexError`` at runtime instead of silently producing a
truncated or empty list.
"""

import fpy2 as fp
import pytest


class TestStrictListSlice:
    """Runtime tests for the ``__fpy_list_slice`` helper."""

    def test_valid_inner_slice(self):
        """A fully-in-bounds slice extracts exactly ``stop - start`` items."""

        @fp.fpy
        def f(xs: list[fp.Real], a: fp.Real, b: fp.Real) -> list[fp.Real]:
            return xs[a:b]

        result = f([1.0, 2.0, 3.0, 4.0, 5.0], 1, 3)
        assert len(result) == 2

    def test_full_slice_no_bounds(self):
        """``xs[:]`` returns the whole list."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[:]

        result = f([1.0, 2.0, 3.0])
        assert len(result) == 3

    def test_omitted_start_defaults_to_zero(self):
        """``xs[:k]`` defaults start to 0."""

        @fp.fpy
        def f(xs: list[fp.Real], k: fp.Real) -> list[fp.Real]:
            return xs[:k]

        assert len(f([1.0, 2.0, 3.0, 4.0, 5.0], 3)) == 3

    def test_omitted_stop_defaults_to_length(self):
        """``xs[k:]`` defaults stop to ``len(xs)``."""

        @fp.fpy
        def f(xs: list[fp.Real], k: fp.Real) -> list[fp.Real]:
            return xs[k:]

        assert len(f([1.0, 2.0, 3.0, 4.0, 5.0], 2)) == 3

    def test_empty_slice_at_endpoint(self):
        """``xs[k:k]`` (start == stop, in bounds) returns an empty list."""

        @fp.fpy
        def f(xs: list[fp.Real], k: fp.Real) -> list[fp.Real]:
            return xs[k:k]

        assert f([1.0, 2.0, 3.0], 2) == []

    def test_full_slice_at_endpoint(self):
        """``xs[0:len(xs)]`` is valid and returns the whole list."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[0:len(xs)]

        assert len(f([1.0, 2.0, 3.0])) == 3

    # ------------------------------------------------------------------
    # Strict-bounds violations: each raises IndexError.

    def test_negative_start_raises(self):
        """``xs[-1:k]`` is invalid — strict semantics reject negatives."""

        @fp.fpy
        def f(xs: list[fp.Real], a: fp.Real, b: fp.Real) -> list[fp.Real]:
            return xs[a:b]

        with pytest.raises(IndexError, match='out of range'):
            f([1.0, 2.0, 3.0], -1, 2)

    def test_stop_past_end_raises(self):
        """``xs[a:b]`` with ``b > len(xs)`` is invalid."""

        @fp.fpy
        def f(xs: list[fp.Real], a: fp.Real, b: fp.Real) -> list[fp.Real]:
            return xs[a:b]

        with pytest.raises(IndexError, match='out of range'):
            f([1.0, 2.0, 3.0], 0, 5)

    def test_start_greater_than_stop_raises(self):
        """``xs[2:1]`` (start > stop) is invalid even when both are in range."""

        @fp.fpy
        def f(xs: list[fp.Real], a: fp.Real, b: fp.Real) -> list[fp.Real]:
            return xs[a:b]

        with pytest.raises(IndexError, match='start .* > stop'):
            f([1.0, 2.0, 3.0], 2, 1)

    def test_start_past_end_with_omitted_stop_raises(self):
        """``xs[10:]`` on a size-3 list — strict semantics raise.

        Python clamps this to an empty list; FPy doesn't.
        """

        @fp.fpy
        def f(xs: list[fp.Real], a: fp.Real) -> list[fp.Real]:
            return xs[a:]

        with pytest.raises(IndexError):
            f([1.0, 2.0, 3.0], 10)

    def test_stop_past_end_with_omitted_start_raises(self):
        """``xs[:10]`` on a size-3 list — strict semantics raise."""

        @fp.fpy
        def f(xs: list[fp.Real], b: fp.Real) -> list[fp.Real]:
            return xs[:b]

        with pytest.raises(IndexError):
            f([1.0, 2.0, 3.0], 10)

    def test_non_integer_bound_raises(self):
        """A non-integer slice bound raises ``TypeError``."""

        @fp.fpy
        def f(xs: list[fp.Real], a: fp.Real, b: fp.Real) -> list[fp.Real]:
            return xs[a:b]

        with pytest.raises(TypeError):
            f([1.0, 2.0, 3.0], 0.5, 2)
