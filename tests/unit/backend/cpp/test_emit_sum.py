"""``sum(xs)`` must be the fold the interpreter performs.

``_eval_sum`` seeds the accumulator with ``xs[0]`` **unrounded** and performs
*n-1* additions, each rounded to the active context; an empty list is an exact
``+0``.  Emitting ``accumulate(begin, end, T(0))`` instead did *n* additions from
a typed zero, which differs twice over:

- ``sum([-0.0])`` came out ``+0.0``, since ``0.0 + -0.0`` is ``+0.0``;
- a seed in a narrower format rounded the first element away, so ``sum(xs)``
  under FP32 with ``xs = [5e-324]`` gave ``0`` where the interpreter gives the
  element untouched.

``accumulate`` takes its seed and its range separately, so the interpreter's
shape is just a range starting one past ``begin``.

**Why the accumulator may be wider but not narrower.**  ``accumulate``'s step is
``init = init + *first``, so the addition is ``T + E`` under the usual arithmetic
conversions, not ``T + T``.  When ``E`` converts to ``T`` exactly the common type
*is* ``T``, so the step promotes exactly, adds once and rounds once to ``T`` --
uni-precision at the accumulator, which is what the interpreter does.  When ``E``
is wider the step computes in ``E`` and narrows on assignment, rounding twice;
and widening ``T`` to hold the unrounded seed instead would round every addition
at the wrong format.  No accumulator type gives both, so that case is refused.
"""

import fpy2 as fp
import pytest

from fpy2.backend.cpp import CppCompiler, CppCompileError
from fpy2.types import ListType, RealType

_L64 = ListType(RealType(fp.FP64))


class TestTheEmittedFold:
    def test_seeds_from_the_first_element(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]) -> fp.Real:
            return sum(xs)

        out = CppCompiler(optimize=False).compile(f, arg_types=[_L64])
        assert 'begin() + 1' in out, out
        assert 'std::accumulate' in out, out

    def test_guards_the_empty_list(self):
        """``begin() + 1`` and ``xs[0]`` are both undefined on an empty vector,
        and the differential harness runs length zero."""

        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]) -> fp.Real:
            return sum(xs)

        out = CppCompiler(optimize=False).compile(f, arg_types=[_L64])
        assert 'size() == 0' in out, out
        # ...and the empty answer is a positive zero, per `_eval_sum`
        assert 'static_cast<double>(0)' in out, out


class TestAccumulatorWidth:
    def test_a_wider_accumulator_is_allowed_and_seeds_with_a_cast(self):
        """``int8`` elements into FP64: the conversion is exact, so the fold is
        uni-precision at ``double``.

        The cast on the seed is load-bearing -- ``accumulate`` deduces ``T``
        from it, so without one the whole fold would run in the element type.
        """

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.SINT8:
                q = [fp.round(x) for x in xs]
            with fp.FP64:
                return sum(q)

        out = CppCompiler(optimize=False).compile(
            f, ctx=fp.FP64, arg_types=[_L64],
        )
        assert 'std::accumulate' in out, out
        assert 'static_cast<double>(' in out, out

    def test_a_narrower_accumulator_is_refused(self):
        """FP64 elements into an FP32 accumulator.

        The seed would round, which is the ``[5e-324]`` case, and no accumulator
        type avoids it without breaking the per-addition rounding.  A refusal is
        an acceptable answer; a wrong sum is not.
        """

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP32:
                return sum(xs)

        with pytest.raises(CppCompileError, match='cannot hold one exactly'):
            CppCompiler(optimize=False).compile(
                f, ctx=fp.FP32, arg_types=[_L64],
            )
