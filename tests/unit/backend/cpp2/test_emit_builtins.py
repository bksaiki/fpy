"""
Phase 4g tests for the cpp2 emitter — list built-ins.

``sum``, ``enumerate``, and ``zip`` are FPy-side functions over
lists that lower to standard C++ idioms:

- ``sum(xs)`` → ``std::accumulate`` with the result type inferred
  by format inference.
- ``enumerate(xs)`` → a ``std::vector<std::tuple<I, T>>`` populated
  by an indexed for-loop.
- ``zip(xs, ys, ...)`` → a ``std::vector<std::tuple<T1, T2, ...>>``
  populated similarly.

The temporaries the emitter allocates use ``__cpp2_tmpN`` names.
"""

import fpy2 as fp

from fpy2.backend.cpp2 import Cpp2Compiler
from fpy2.types import ListType, RealType


class TestSum:
    """``sum(xs)`` → ``std::accumulate``."""

    def test_sum_returns_accumulate(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return sum(xs)

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        assert (
            'std::accumulate(xs.begin(), xs.end(), '
            'static_cast<double>(0))'
        ) in out


class TestEnumerate:
    """``enumerate(xs)`` builds a ``std::vector<std::tuple<I, T>>``."""

    def test_enumerate_in_for_loop(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                # The body intentionally uses ``x`` only — mixing the
                # int64 ``i`` into FP64 arithmetic would require an
                # explicit ``fp.round(i)`` under the strict cast policy.
                for i, x in enumerate(xs):
                    acc = acc + x
                return acc

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        # Result-vector type and per-element tuple shape.
        assert 'std::vector<std::tuple<int64_t, double>>' in out
        # Loop populates the result with (size_t-cast index, source elt).
        assert (
            'std::make_tuple(static_cast<int64_t>(__cpp2_tmp3), '
            '__cpp2_tmp1[__cpp2_tmp3]);'
        ) in out
        # Then the outer for-loop destructures into ``i``/``x``.
        assert 'int64_t i = std::get<0>' in out
        assert 'double x = std::get<1>' in out


class TestZip:
    """``zip(xs, ys, ...)`` builds a ``std::vector<std::tuple<...>>``."""

    def test_zip_two_args(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for x, y in zip(xs, ys):
                    acc = acc + x * y
                return acc

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[
                ListType(RealType(fp.FP64)),
                ListType(RealType(fp.FP64)),
            ],
        )
        # Both iterables bound to temps.
        assert 'auto __cpp2_tmp1 = xs;' in out
        assert 'auto __cpp2_tmp2 = ys;' in out
        # Per-element tuple draws from both temps.
        assert (
            'std::make_tuple(__cpp2_tmp1[__cpp2_tmp4], '
            '__cpp2_tmp2[__cpp2_tmp4]);'
        ) in out
        # Loop body destructures back to ``x``/``y``.
        assert 'double x = std::get<0>' in out
        assert 'double y = std::get<1>' in out

    def test_zip_three_args(self):
        @fp.fpy
        def f(
            xs: list[fp.Real], ys: list[fp.Real], zs: list[fp.Real]
        ) -> fp.Real:
            with fp.FP64:
                acc = 0
                for x, y, z in zip(xs, ys, zs):
                    acc = acc + x * y * z
                return acc

        out = Cpp2Compiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))] * 3,
        )
        assert 'std::vector<std::tuple<double, double, double>>' in out
        # Three iterables bound, three subscript reads in make_tuple.
        assert 'auto __cpp2_tmp1 = xs;' in out
        assert 'auto __cpp2_tmp2 = ys;' in out
        assert 'auto __cpp2_tmp3 = zs;' in out
