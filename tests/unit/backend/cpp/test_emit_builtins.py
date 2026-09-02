"""
Phase 4g tests for the cpp emitter — list built-ins.

``sum(xs)`` lowers to ``std::accumulate``, with the result type inferred by
format inference.

``enumerate`` and ``zip`` no longer reach the emitter at all: `UnfoldEnumerate`
and `UnfoldZip` state each as the comprehension `derived-semantics.rst` defines
it to be, inside `_to_statement_form`'s fixpoint, and `CompToLoop` lowers that.
So the tuple list they used to build is now built by the comprehension's own
fill loop — same object, one fewer emitter case.

The temporaries the emitter allocates use ``_tmpN`` names.
"""

import contextlib

import pytest

import fpy2 as fp
import fpy2.backend.cpp.compiler as _compiler
from fpy2.backend.cpp import CppCompileError, CppCompiler
from fpy2.backend.cpp.emitter import CppEmitter
from fpy2.transform import CompToLoop, Hoistable
from fpy2.types import ListType, RealType


@contextlib.contextmanager
def _no_unfold():
    """`_to_statement_form` without the unfolds, which is how a `zip` reaches
    the emitter at all."""
    def plain(fd):
        while True:
            fd = Hoistable.apply(fd)
            log = CompToLoop.apply_with_edits(fd)
            if not log.edits:
                return fd
            fd = log.result

    original = _compiler._to_statement_form
    _compiler._to_statement_form = plain
    try:
        yield
    finally:
        _compiler._to_statement_form = original


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
        # Seeded from the first element over ``begin() + 1``, which is the fold
        # `_eval_sum` performs -- *n-1* additions from an unrounded seed, and an
        # exact ``+0`` for the empty list.  See ``test_emit_sum.py``.
        assert 'std::accumulate(xs.begin() + 1, xs.end(), ' in out
        assert 'xs.size() == 0 ? static_cast<double>(0)' in out


class TestEnumerate:
    """Unoptimized, the unfolded comprehension materializes a list of tuples.
    With the default ``optimize=True``, :class:`EnumerateElim` fuses it into a
    plain indexed loop before the unfold ever sees it — which is why the fuse
    has to run first.
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
        # The fill loop pairs the index with the source element.
        assert 'std::make_tuple(' in out
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
        # the source is indexed directly -- copy propagation reaches the
        # `_src` alias `EnumerateElim` binds, so not even a reference is left
        assert 'xs[static_cast<size_t>(i)]' in out
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
        assert 'xs[static_cast<size_t>(i)]' in out
        assert 'ys[static_cast<size_t>(i)]' in out

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
        assert 'std::vector<std::tuple' not in out
        # ... just the one per-iteration tuple.
        assert out.count('std::make_tuple') == 1
        assert 'for (int64_t i = 0;' in out


class TestZip:
    """Unoptimized, the unfolded comprehension materializes the tuple list —
    plus the length assertion the unfolding claims, which is where the surface
    node's strictness went.  With the default ``optimize=True``,
    :class:`ZipElim` fuses the pattern first: see
    :meth:`test_zip_optimized_skips_tuple_vector`.
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
        # the length the unfolding claims, one per iterable past the first
        assert out.count('assert(') == 1
        assert 'ys.size()) == static_cast<int64_t>(xs.size())' in out
        # the fill loop indexes both sources into the tuple
        assert 'std::make_tuple(xs[' in out and 'ys[' in out
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
        # two extra iterables, so two length claims
        assert out.count('assert(') == 2
        # three direct subscript reads in make_tuple
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
        # both sources are indexed directly
        assert 'xs[static_cast<size_t>(' in out
        assert 'ys[static_cast<size_t>(' in out


class TestTheEmitterNoLongerHasThem:
    """`_emit_zip` and `_emit_enumerate` are gone: `_to_statement_form` unfolds
    both, so a node reaching the emitter is a backend bug.

    Measured over the corpus: the two fired 6 and 3 times before the unfolds
    joined the fixpoint, and 15 and 3 unoptimized.  Running the unfolds *once*
    instead of to a fixpoint leaves 2 and 9 -- a `zip` only gets its statement
    slot after `CompToLoop` opens the comprehension around it.
    """

    def test_the_methods_are_gone(self):
        for name in ('_emit_zip', '_emit_enumerate'):
            assert not hasattr(CppEmitter, name)

    def test_a_zip_reaching_the_emitter_is_a_tripwire(self):
        """Reached by taking the unfold out, which is the only way in."""
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]):
            with fp.FP64:
                return zip(xs, ys)

        arg_types = [ListType(RealType(fp.FP64))] * 2
        with _no_unfold(), pytest.raises(CppCompileError, match='`zip` reach'):
            CppCompiler().compile(f, ctx=fp.FP64, arg_types=arg_types)

    def test_an_enumerate_reaching_the_emitter_is_a_tripwire(self):
        @fp.fpy
        def f(xs: list[fp.Real]):
            with fp.FP64:
                return enumerate(xs)

        with _no_unfold(), pytest.raises(
            CppCompileError, match='`enumerate` reach',
        ):
            CppCompiler().compile(
                f, ctx=fp.FP64, arg_types=[ListType(RealType(fp.FP64))],
            )
