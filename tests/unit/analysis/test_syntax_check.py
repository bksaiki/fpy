"""
Unit tests for `SyntaxCheck`.

- ``TestDuplicateBinding``: a single binding may not bind the same name twice.
"""

import fpy2 as fp
import pytest

from fpy2.analysis.syntax_check import FPySyntaxError


class TestDuplicateBinding:
    def test_duplicate_tuple_assign(self):
        with pytest.raises(FPySyntaxError, match='duplicate identifier'):
            @fp.fpy
            def f(t: tuple[fp.Real, fp.Real]):
                x, x = t
                return x

    def test_duplicate_nested_tuple_assign(self):
        with pytest.raises(FPySyntaxError, match='duplicate identifier'):
            @fp.fpy
            def f(t: tuple[fp.Real, tuple[fp.Real, fp.Real]]):
                a, (b, a) = t
                return a

    def test_duplicate_for_target(self):
        with pytest.raises(FPySyntaxError, match='duplicate identifier'):
            @fp.fpy
            def f(xs: list[tuple[fp.Real, fp.Real]]):
                s = 0
                for a, a in xs:
                    s = s + a
                return s

    def test_duplicate_comprehension_target(self):
        with pytest.raises(FPySyntaxError, match='duplicate identifier'):
            @fp.fpy
            def f(xs: list[tuple[fp.Real, fp.Real]]):
                return [a for a, a in xs]

    def test_repeated_wildcard_allowed(self):
        @fp.fpy
        def f(t: tuple[fp.Real, fp.Real]):
            _, _ = t
            return 1

        assert f((1, 2)) == 1

    def test_rebinding_across_assignments_allowed(self):
        @fp.fpy
        def f(t: tuple[fp.Real, fp.Real]):
            x, y = t
            x, z = t
            return x + y + z

        assert f((1, 2)) == 5

    def test_distinct_comprehension_targets_may_repeat(self):
        # separate bindings, so the shared name is a rebind, not a duplicate
        @fp.fpy
        def f(xs: list[fp.Real]):
            return [a for a in xs for a in xs]

        assert f([1, 2]) == [1, 2, 1, 2]
