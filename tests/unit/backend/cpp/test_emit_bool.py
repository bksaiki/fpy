"""
Phase 3b tests for the cpp emitter — booleans and comparisons.
"""

import re

import fpy2 as fp

from fpy2.backend.cpp import CppCompiler
from fpy2.types import BoolType, ListType, RealType


class TestBoolAndCompare:
    """Phase 3b — bool literals and comparison expressions."""

    def test_bool_literal_true(self):
        @fp.fpy
        def f() -> bool:
            return True

        out = CppCompiler().compile(f)
        assert out == 'bool f() {\n    return true;\n}'

    def test_bool_literal_false(self):
        @fp.fpy
        def f() -> bool:
            return False

        out = CppCompiler().compile(f)
        assert out == 'bool f() {\n    return false;\n}'

    def test_pairwise_lt(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> bool:
            with fp.FP64:
                return x < y

        cc = CppCompiler()
        out = cc.compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'return (x < y);' in out
        assert out.startswith('bool f(double x, double y)')

    def test_all_six_comparison_ops(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> bool:
            with fp.FP64:
                a = x < y
                b = x <= y
                c = x > y
                d = x >= y
                e = x == y
                g = x != y
                return a

        cc = CppCompiler()
        out = cc.compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64), RealType(fp.FP64)],
        )
        assert 'a = (x < y);' in out
        assert 'b = (x <= y);' in out
        assert 'c = (x > y);' in out
        assert 'd = (x >= y);' in out
        assert 'e = (x == y);' in out
        assert 'g = (x != y);' in out

    def test_chained_comparison(self):
        """``x < y < z`` expands to a conjunction."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real, z: fp.Real) -> bool:
            with fp.FP64:
                return x < y < z

        cc = CppCompiler()
        out = cc.compile(
            f, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64)] * 3,
        )
        assert '((x < y) && (y < z))' in out


class TestBooleanReduce:
    """``any`` / ``all`` lower to ``std::any_of`` / ``std::all_of`` with an
    identity predicate — no hoisted loop and no casting, unlike ``AMin`` /
    ``AMax``."""

    @staticmethod
    def _compile(f, elt_ty, **kwargs):
        return CppCompiler(**kwargs).compile(
            f, ctx=fp.FP64, arg_types=[ListType(elt_ty)],
        )

    def test_any_over_bool_list_arg(self):
        @fp.fpy
        def f(bs: list[bool]) -> bool:
            with fp.FP64:
                return any(bs)

        out = self._compile(f, BoolType())
        assert out.startswith('bool f(const std::vector<bool>& bs)')
        assert 'std::any_of(' in out

    def test_all_over_bool_list_arg(self):
        @fp.fpy
        def f(bs: list[bool]) -> bool:
            with fp.FP64:
                return all(bs)

        out = self._compile(f, BoolType())
        assert out.startswith('bool f(const std::vector<bool>& bs)')
        assert 'std::all_of(' in out

    def test_identity_predicate_is_a_bool_lambda(self):
        @fp.fpy
        def f(bs: list[bool]) -> bool:
            with fp.FP64:
                return all(bs)

        out = self._compile(f, BoolType())
        m = re.search(r'\[\]\(bool (\w+)\) \{ return (\w+); \}', out)
        assert m, f'no identity predicate in:\n{out}'
        assert m.group(1) == m.group(2), 'predicate must return its parameter'

    def test_prvalue_operand_is_bound_before_iterating(self):
        """``begin()``/``end()`` on a prvalue would name iterators into two
        different temporaries — an invalid range.  Same reason ``Sum`` binds."""
        @fp.fpy
        def f(bs: list[bool]) -> bool:
            with fp.FP64:
                return any(bs[1:])

        out = self._compile(f, BoolType())
        bound = re.search(r'auto&& (\w+) = std::vector', out)
        assert bound, out
        t = bound.group(1)
        assert f'std::any_of({t}.begin(), {t}.end()' in out

    def test_named_operand_is_not_bound(self):
        """A name is evaluated once already, so there is nothing to bind — and
        a list is a handle, so there was never a copy to avoid either."""
        @fp.fpy
        def f(bs: list[bool]) -> bool:
            with fp.FP64:
                return any(bs)

        out = self._compile(f, BoolType())
        assert 'std::any_of(bs.begin(), bs.end()' in out
        assert 'auto&&' not in out

    def test_comprehension_operand_is_fused_away(self):
        """``ReduceFusion`` runs in the default (optimizing) pipeline, so the
        idiomatic form emits an accumulator loop with no intermediate
        vector."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> bool:
            with fp.FP64:
                return all([x < 0 for x in xs])

        out = self._compile(f, RealType(fp.FP64))
        assert 'std::vector<bool>' not in out
        assert 'std::all_of(' not in out
        assert '&&' in out          # folded with `and` in the loop body

    def test_comprehension_operand_unfused_without_optimize(self):
        """With ``optimize=False`` the reduction is not fused: the
        comprehension materializes a ``std::vector<bool>`` that
        ``std::all_of`` then scans.

        The comprehension is still *lowered* -- ``CompToLoop`` is a
        normalization and runs either way -- so the vector is filled by index
        rather than grown."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> bool:
            with fp.FP64:
                return all([x < 0 for x in xs])

        out = self._compile(f, RealType(fp.FP64), optimize=False)
        assert 'std::vector<bool>' in out
        assert 'acc[static_cast<size_t>(i)] = (x < static_cast<double>(0));' in out
        assert 'std::all_of(' in out
