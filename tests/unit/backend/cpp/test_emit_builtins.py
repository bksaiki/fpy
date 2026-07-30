"""
Phase 4g tests for the cpp emitter — list built-ins.

``sum``, ``enumerate``, and ``zip`` are FPy-side functions over
lists that lower to standard C++ idioms:

- ``sum(xs)`` → ``std::accumulate`` with the result type inferred
  by format inference.
- ``enumerate(xs)`` → a ``fpy::list<std::tuple<I, T>>`` populated
  by an indexed for-loop.
- ``zip(xs, ys, ...)`` → a ``fpy::list<std::tuple<T1, T2, ...>>``
  populated similarly.

The temporaries the emitter allocates use ``_tmpN`` names.
"""

import fpy2 as fp

from fpy2.backend.cpp import CppCompiler
from fpy2.types import ListType, RealType


class TestSum:
    """``sum(xs)`` → ``std::accumulate``."""

    def test_sum_returns_accumulate(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return sum(xs)

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        # A named operand is read directly; only a prvalue needs binding so
        # that begin()/end() name the same object (see
        # ``test_prvalue_operand_is_bound_before_iterating`` in test_emit_bool).
        assert 'auto&&' not in out
        assert (
            'std::accumulate(xs->begin(), xs->end(), '
            'static_cast<double>(0))'
        ) in out


class TestEnumerate:
    """``enumerate(xs)`` builds a ``fpy::list<std::tuple<I, T>>``."""

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

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        # Result-vector type and per-element tuple shape.
        assert 'fpy::list<std::tuple<int64_t, double>>' in out
        # Loop populates the result with (size_t-cast index, source elt).
        assert (
            'std::make_tuple(static_cast<int64_t>(_tmp2), '
            '(*xs)[_tmp2]);'
        ) in out
        # Then the outer for-loop destructures into ``i``/``x``.
        assert 'int64_t i = std::get<0>' in out
        assert 'double x = std::get<1>' in out


class TestZip:
    """``zip(xs, ys, ...)`` lowers to a ``fpy::list<std::tuple<...>>``
    by default when optimizations are disabled.  With the default
    ``optimize=True``, :class:`ZipElim` rewrites the pattern to a
    plain indexed loop instead — see :meth:`test_zip_optimized_skips_tuple_vector`.
    """

    def test_zip_two_args_unoptimized(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for x, y in zip(xs, ys):
                    acc = acc + x * y
                return acc

        out = CppCompiler(optimize=False).compile(
            f, ctx=fp.FP64,
            arg_types=[
                ListType(RealType(fp.FP64)),
                ListType(RealType(fp.FP64)),
            ],
        )
        # Both iterables are names, so neither needs a temp; the per-element
        # tuple indexes them directly.
        assert 'auto&&' not in out
        assert 'std::make_tuple((*xs)[_tmp2], (*ys)[_tmp2]);' in out
        # Loop body destructures back to ``x``/``y``.
        assert 'double x = std::get<0>' in out
        assert 'double y = std::get<1>' in out

    def test_zip_three_args_unoptimized(self):
        @fp.fpy
        def f(
            xs: list[fp.Real], ys: list[fp.Real], zs: list[fp.Real]
        ) -> fp.Real:
            with fp.FP64:
                acc = 0
                for x, y, z in zip(xs, ys, zs):
                    acc = acc + x * y * z
                return acc

        out = CppCompiler(optimize=False).compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))] * 3,
        )
        assert 'fpy::list<std::tuple<double, double, double>>' in out
        # Three named iterables, so three direct subscript reads in make_tuple.
        assert 'auto&&' not in out
        assert 'std::make_tuple((*xs)[' in out
        assert '(*ys)[' in out and '(*zs)[' in out

    def test_zip_optimized_skips_tuple_vector(self):
        """Default ``CppCompiler()`` has ``optimize=True``, so
        :class:`ZipElim` runs first and ``for ... in zip(...)``
        lowers to a plain indexed loop — no intermediate
        ``fpy::list<std::tuple<...>>``."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for x, y in zip(xs, ys):
                    acc = acc + x * y
                return acc

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))] * 2,
        )
        # No tuple-vector machinery at all.
        assert 'std::tuple' not in out
        assert 'std::make_tuple' not in out
        # The two sources are bound to read-only ``_src`` aliases (const
        # references — no copy) and indexed directly.
        assert 'const auto& _src' in out
