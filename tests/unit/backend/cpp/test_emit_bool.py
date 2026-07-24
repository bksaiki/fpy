"""
Phase 3b tests for the cpp emitter — booleans and comparisons.
"""

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
    """``any`` / ``all`` lower to ``std::any_of`` / ``std::all_of``.

    No hoisted loop and no casting, unlike ``AMin``/``AMax``: the operand is
    ``std::vector<bool>`` and the result is ``bool``, so the algorithm applies
    directly with an identity predicate.  The empty range is well defined and
    agrees with FPy (``all_of`` -> ``true``, ``any_of`` -> ``false``), so there
    is no unguarded ``xs[0]`` as there is for ``min``/``max``.
    """

    def test_any_over_bool_list_arg(self):
        @fp.fpy
        def f(bs: list[bool]) -> bool:
            with fp.FP64:
                return any(bs)

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[ListType(BoolType())],
        )
        assert out.startswith('bool f(const std::vector<bool>& bs)')
        assert 'std::any_of(' in out
        assert '.begin(), ' in out and '.end(), ' in out
        assert 'return ' in out

    def test_all_over_bool_list_arg(self):
        @fp.fpy
        def f(bs: list[bool]) -> bool:
            with fp.FP64:
                return all(bs)

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[ListType(BoolType())],
        )
        assert out.startswith('bool f(const std::vector<bool>& bs)')
        assert 'std::all_of(' in out

    def test_identity_predicate_is_a_bool_lambda(self):
        @fp.fpy
        def f(bs: list[bool]) -> bool:
            with fp.FP64:
                return all(bs)

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[ListType(BoolType())],
        )
        # `[](bool <t>) { return <t>; }` -- same temp in both positions
        import re
        m = re.search(r'\[\]\(bool (\w+)\) \{ return (\w+); \}', out)
        assert m, f'no identity predicate in:\n{out}'
        assert m.group(1) == m.group(2)

    def test_operand_bound_to_auto_ref_before_iterating(self):
        """``begin()``/``end()`` on a prvalue would name iterators into two
        different temporaries -- an invalid range.  Same reason ``Sum`` binds."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> bool:
            with fp.FP64:
                return any([x < 0 for x in xs])

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[ListType(RealType(fp.FP64))],
        )
        assert 'auto&& ' in out
        # the iterated name is the bound reference, not the raw expression
        import re
        bound = re.search(r'auto&& (\w+) = ', out)
        assert bound, out
        assert f'std::any_of({bound.group(1)}.begin(), {bound.group(1)}.end()' in out

    def test_over_comprehension_of_comparisons(self):
        """The idiomatic form: the comprehension materializes a
        ``std::vector<bool>``, then the algorithm scans it."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> bool:
            with fp.FP64:
                return all([x < 0 for x in xs])

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[ListType(RealType(fp.FP64))],
        )
        assert 'std::vector<bool>' in out
        assert 'push_back(' in out
        assert 'std::all_of(' in out
