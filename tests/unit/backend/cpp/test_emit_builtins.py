"""
Phase 4g tests for the cpp emitter — list built-ins.

``sum``, ``enumerate``, and ``zip`` are FPy-side functions over
lists that lower to standard C++ idioms:

- ``sum(xs)`` → ``std::accumulate`` with the result type inferred
  by format inference.
- ``enumerate(xs)`` → a ``std::vector<std::tuple<I, T>>`` populated
  by an indexed for-loop.
- ``zip(xs, ys, ...)`` → a ``std::vector<std::tuple<T1, T2, ...>>``
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
            'std::accumulate(xs.begin(), xs.end(), '
            'static_cast<double>(0))'
        ) in out


class TestEnumerate:
    """``enumerate(xs)`` lowers to a ``std::vector<std::tuple<I, T>>`` when
    optimizations are disabled.  With the default ``optimize=True``,
    :class:`EnumerateElim` rewrites it to a plain indexed loop instead.
    """

    def test_enumerate_in_for_loop_unoptimized(self):
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

        out = CppCompiler(optimize=False).compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        # Result-vector type and per-element tuple shape.
        assert 'std::vector<std::tuple<int64_t, double>>' in out
        # Loop populates the result with (size_t-cast index, source elt).
        assert (
            'std::make_tuple(static_cast<int64_t>(_tmp2), '
            'xs[_tmp2]);'
        ) in out
        # Then the outer for-loop destructures into ``i``/``x``.
        assert 'int64_t i = std::get<0>' in out
        assert 'double x = std::get<1>' in out

    def test_enumerate_optimized_skips_tuple_vector(self):
        """With the default ``optimize=True``, :class:`EnumerateElim` lowers
        this to a plain indexed loop: no intermediate
        ``std::vector<std::tuple<...>>``, and ``i`` becomes the loop counter
        rather than a destructured tuple element."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, x in enumerate(xs):
                    acc = acc + x
                return acc

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        # No tuple machinery at all.
        assert 'std::tuple' not in out
        assert 'std::make_tuple' not in out
        # The source is bound to a read-only ``_src`` alias (a const
        # reference — no copy) and indexed directly.
        assert 'const auto& _src' in out
        # ``i`` is the loop counter itself.
        assert 'for (int64_t i = 0;' in out

    def test_enumerate_of_zip_optimized_skips_both_vectors(self):
        """Unoptimized this materializes *two* vectors — the zip's tuples and
        the enumerate's (index, tuple) pairs.  :class:`EnumerateElim` collapses
        both into direct indexing of the zip's own arguments, which is why it
        must run before :class:`ZipElim`."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, (a, b) in enumerate(zip(xs, ys)):
                    acc = acc + a * b
                return acc

        arg_types = [ListType(RealType(fp.FP64))] * 2
        unopt = CppCompiler(optimize=False).compile(
            f, ctx=fp.FP64, arg_types=arg_types,
        )
        # Unoptimized: both intermediate vectors, the outer one nesting the
        # inner tuple type.
        assert 'std::vector<std::tuple<double, double>>' in unopt
        assert (
            'std::vector<std::tuple<int64_t, std::tuple<double, double>>>'
        ) in unopt

        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=arg_types)
        # Optimized: neither vector, and both sources indexed directly.
        assert 'std::tuple' not in out
        assert 'std::make_tuple' not in out
        assert 'for (int64_t i = 0;' in out
        assert out.count('const auto& _src') == 2

    def test_enumerate_of_zip_whole_tuple_slot(self):
        """An element slot bound to the whole zipped tuple still avoids both
        vectors: the tuple is rebuilt per iteration from the indexed
        sources, so only a stack tuple is ever constructed."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, p in enumerate(zip(xs, ys)):
                    acc = acc + fp.fst(p) * fp.snd(p)
                return acc

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[ListType(RealType(fp.FP64))] * 2,
        )
        # No list-of-tuples anywhere ...
        assert 'fpy::list<std::tuple' not in out
        # ... just the one per-iteration tuple.
        assert out.count('std::make_tuple') == 1
        assert 'for (int64_t i = 0;' in out


class TestZip:
    """``zip(xs, ys, ...)`` lowers to a ``std::vector<std::tuple<...>>``
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
        assert 'std::make_tuple(xs[_tmp2], ys[_tmp2]);' in out
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
        assert 'std::vector<std::tuple<double, double, double>>' in out
        # Three named iterables, so three direct subscript reads in make_tuple.
        assert 'auto&&' not in out
        assert 'std::make_tuple(xs[' in out
        assert 'ys[' in out and 'zs[' in out

    def test_zip_optimized_skips_tuple_vector(self):
        """Default ``CppCompiler()`` has ``optimize=True``, so
        :class:`ZipElim` runs first and ``for ... in zip(...)``
        lowers to a plain indexed loop — no intermediate
        ``std::vector<std::tuple<...>>``."""

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
