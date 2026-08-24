"""
Unit tests for `SyntaxCheck`.

- ``TestDuplicateBinding``: a single binding may not bind the same name twice.
- ``TestTargetScope``: `for`/comprehension targets are fresh and loop-scoped.
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


class TestTargetScope:
    """A `for`/comprehension target must be fresh and does not escape its loop."""

    def test_target_shadowing_argument(self):
        with pytest.raises(FPySyntaxError, match='shadows an existing definition'):
            @fp.fpy
            def f(x: fp.Real, xs: list[fp.Real]):
                for x in xs:
                    x = x + 1
                return x

    def test_target_shadowing_assignment(self):
        with pytest.raises(FPySyntaxError, match='shadows an existing definition'):
            @fp.fpy
            def f(xs: list[fp.Real]):
                a = 0
                for a in xs:
                    pass
                return a

    def test_nested_comprehension_clauses_must_differ(self):
        with pytest.raises(FPySyntaxError, match='shadows an existing definition'):
            @fp.fpy
            def f(xs: list[fp.Real]):
                return [a for a in xs for a in xs]

    def test_comprehension_target_shadowing_enclosing_loop(self):
        with pytest.raises(FPySyntaxError, match='shadows an existing definition'):
            @fp.fpy
            def f(xs: list[fp.Real]):
                ys = [0.0]
                for v in xs:
                    ys = [v * 2 for v in xs]
                return ys

    def test_target_not_visible_after_the_loop(self):
        with pytest.raises(FPySyntaxError, match='unbound variable `v`'):
            @fp.fpy
            def f(xs: list[fp.Real]):
                for v in xs:
                    pass
                return v

    def test_sibling_loops_may_reuse_a_target(self):
        # the first loop's target went out of scope, so this is not a shadow
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]):
            s = 0
            for v in xs:
                s = s + v
            for v in ys:
                s = s + v
            return s

        assert f([1, 2], [3, 4]) == 10

    def test_sibling_comprehension_may_reuse_a_loop_target(self):
        @fp.fpy
        def f(xs: list[fp.Real]):
            s = 0
            for v in xs:
                s = s + v
            return [v * s for v in xs]

        assert f([1, 2]) == [3, 6]
