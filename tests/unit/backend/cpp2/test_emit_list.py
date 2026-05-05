"""
Phase 4a tests for the cpp2 emitter — list literals, indexing, ``len``.
"""

import fpy2 as fp

from fpy2.backend.cpp2 import Cpp2Compiler
from fpy2.types import ListType, RealType


def _compile_list_arg(func) -> str:
    return Cpp2Compiler().compile(
        func, ctx=fp.FP64,
        arg_types=[ListType(RealType(fp.FP64)) for _ in func.args],
    )


class TestListLiteral:
    """Phase 4a — ``[a, b, c]`` list literals."""

    def test_list_of_ints(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                xs = [1, 2, 3]
                return xs[0]

        out = Cpp2Compiler().compile(f, ctx=fp.FP64, arg_types=[])
        assert 'std::vector<uint8_t>' in out
        assert '{1, 2, 3}' in out

    def test_list_passed_through_args(self):
        """A list-typed argument keeps its ``std::vector<T>`` storage
        in the function signature."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return xs[0]

        out = _compile_list_arg(f)
        assert out.startswith('double f(std::vector<double> xs)')


class TestListRef:
    """Phase 4a — ``xs[i]`` indexing."""

    def test_constant_index(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return xs[0] + xs[1]

        out = _compile_list_arg(f)
        assert 'xs[0]' in out
        assert 'xs[1]' in out

    def test_variable_index(self):
        @fp.fpy
        def f(xs: list[fp.Real], i: fp.Real) -> fp.Real:
            with fp.FP64:
                return xs[i]

        cc = Cpp2Compiler()
        out = cc.compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64)), RealType(fp.FP64)],
        )
        assert 'xs[i]' in out


class TestLen:
    """Phase 4a — ``len(xs)``."""

    def test_len_used(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                n = len(xs)
                return xs[n - 1]

        out = _compile_list_arg(f)
        # ``len`` lowers to ``size()`` cast to the inferred integer type.
        assert 'static_cast<int64_t>(xs.size())' in out


class TestListComp:
    """Phase 4b — ``[expr for x in iter]`` list comprehensions."""

    def test_var_iterable(self):
        """Iterating a list-typed Var emits a range-based for loop
        with the target's storage type folded into the header."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = [x + 1 for x in xs]
                return ys[0]

        out = _compile_list_arg(f)
        assert 'for (double x : xs) {' in out
        assert '.push_back((x + 1));' in out

    def test_range_iterable(self):
        """``range(...)`` in a comprehension expands to a counter loop."""

        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                sq = [i * i for i in range(5)]
                return sq[0]

        out = Cpp2Compiler().compile(f, ctx=fp.FP64, arg_types=[])
        assert 'for (int64_t i = 0; i < 5; ++i) {' in out
        assert '.push_back((i * i));' in out

    def test_range2_iterable(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                ks = [k for k in range(3, 9)]
                return ks[0]

        out = Cpp2Compiler().compile(f, ctx=fp.FP64, arg_types=[])
        # ``range(3, 9)`` widens through the unbounded-integer fallback
        # to ``int64_t`` (no tighter ladder entry covers it).
        assert 'for (int64_t k = 3; k < 9; ++k) {' in out

    def test_nested_comp(self):
        """Multiple ``for`` clauses produce nested loops within a
        single result vector."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                zs = [x * y for x in xs for y in ys]
                return zs[0]

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64)), ListType(RealType(fp.FP64))],
        )
        # Inner loop nested directly inside outer loop's body.
        assert (
            'for (double x : xs) {\n'
            '        for (double y : ys) {'
        ) in out
        assert '.push_back((x * y));' in out

    def test_tuple_binding_target(self):
        """Tuple-binding targets in the for-clause destructure each
        element via ``std::get<i>`` inside the loop body — see
        ``test_emit_tuple.TestTupleDestructure`` for the detailed
        shape; this test just confirms list-comp wiring composes."""

        @fp.fpy
        def f(xs: list[tuple[fp.Real, fp.Real]]) -> fp.Real:
            with fp.FP64:
                ys = [a + b for (a, b) in xs]
                return ys[0]

        from fpy2.types import TupleType
        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[
                ListType(TupleType(RealType(fp.FP64), RealType(fp.FP64)))
            ],
        )
        assert 'std::get<0>' in out
        assert 'std::get<1>' in out
        assert '.push_back((a + b));' in out


class TestListSlice:
    """Phase 4d — ``xs[a:b]`` slicing."""

    def test_both_bounds(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = xs[1:4]
                return ys[0]

        out = _compile_list_arg(f)
        # Value bound to a temp, then both endpoints emit as iterator
        # arithmetic with size_t casts.
        assert 'auto __cpp2_tmp1 = xs;' in out
        assert (
            '__cpp2_tmp1.begin() + static_cast<size_t>(1), '
            '__cpp2_tmp1.begin() + static_cast<size_t>(4)'
        ) in out

    def test_open_stop(self):
        """``xs[a:]`` defaults the stop endpoint to ``__tmp.size()``."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = xs[2:]
                return ys[0]

        out = _compile_list_arg(f)
        assert (
            '__cpp2_tmp1.begin() + static_cast<size_t>(2), '
            '__cpp2_tmp1.begin() + __cpp2_tmp1.size()'
        ) in out

    def test_open_start(self):
        """``xs[:b]`` defaults the start endpoint to ``0``."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = xs[:3]
                return ys[0]

        out = _compile_list_arg(f)
        assert (
            '__cpp2_tmp1.begin() + 0, '
            '__cpp2_tmp1.begin() + static_cast<size_t>(3)'
        ) in out

    def test_full_slice(self):
        """``xs[:]`` produces a full-list copy."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                ys = xs[:]
                return ys[0]

        out = _compile_list_arg(f)
        assert (
            '__cpp2_tmp1.begin() + 0, '
            '__cpp2_tmp1.begin() + __cpp2_tmp1.size()'
        ) in out
