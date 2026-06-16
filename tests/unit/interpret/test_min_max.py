"""
Interpreter behaviour of ``min`` and ``max`` for IEEE 754 special values.

Two cases the interpreter must handle to match the cpp / FPCore backends:

1. **NaN propagation.** FPy's contract is that any NaN input yields a
   NaN result.  Python's built-in ``min`` / ``max`` are based on ``<``
   comparisons and pick whichever operand they see first, so
   ``max(1.0, nan) == 1.0`` and ``max(nan, 1.0)`` is ``nan`` — an
   order-dependent quirk that no IEEE 754 variant defines.  The
   interpreter wraps Python's builtins with a NaN-aware helper to fix
   this.

2. **Signed zero.** IEEE 754 specifies ``min(-0, +0) = -0`` and
   ``max(-0, +0) = +0`` regardless of argument order; Python's
   ``min(+0.0, -0.0) == +0.0`` violates the order-independence.

Each test class covers both the variadic form (``min(a, b, …)`` →
:class:`Min` / :class:`Max`) and the reduce form (``min(xs)`` →
:class:`AMin` / :class:`AMax`).
"""

import math
import pytest

import fpy2 as fp


def _is_nan(x) -> bool:
    """True iff *x* is a NaN of any precision."""
    return math.isnan(float(x))


def _is_neg_zero(x) -> bool:
    """True iff *x* is the negative-zero representative."""
    f = float(x)
    return f == 0.0 and math.copysign(1.0, f) == -1.0


def _is_pos_zero(x) -> bool:
    """True iff *x* is the positive-zero representative."""
    f = float(x)
    return f == 0.0 and math.copysign(1.0, f) == 1.0


# ---------------------------------------------------------------------------
# Variadic form — ``min(a, b, ...)`` → ``Min``, ``max(a, b, ...)`` → ``Max``
# ---------------------------------------------------------------------------


class TestVariadicNaN:
    """``Min`` / ``Max`` propagate NaN regardless of argument position."""

    def test_min_nan_first(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return min(a, b)
        assert _is_nan(f(math.nan, 1.0, ctx=fp.FP64))

    def test_min_nan_second(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return min(a, b)
        assert _is_nan(f(1.0, math.nan, ctx=fp.FP64))

    def test_max_nan_first(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return max(a, b)
        assert _is_nan(f(math.nan, 1.0, ctx=fp.FP64))

    def test_max_nan_second(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return max(a, b)
        assert _is_nan(f(1.0, math.nan, ctx=fp.FP64))

    def test_min_both_nan(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return min(a, b)
        assert _is_nan(f(math.nan, math.nan, ctx=fp.FP64))

    def test_max_nan_in_middle_of_three(self):
        """Three-arg min/max with NaN sandwiched between finite values."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
            return max(a, b, c)
        assert _is_nan(f(1.0, math.nan, 2.0, ctx=fp.FP64))

    def test_min_nan_at_end_of_three(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
            return min(a, b, c)
        assert _is_nan(f(1.0, 2.0, math.nan, ctx=fp.FP64))


class TestVariadicSignedZero:
    """``Min`` / ``Max`` respect IEEE 754 signed-zero ordering."""

    def test_min_neg_pos_zero(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return min(a, b)
        assert _is_neg_zero(f(-0.0, 0.0, ctx=fp.FP64))

    def test_min_pos_neg_zero(self):
        """Order-independent: ``min(+0, -0)`` must still be ``-0``."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return min(a, b)
        assert _is_neg_zero(f(0.0, -0.0, ctx=fp.FP64))

    def test_max_neg_pos_zero(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return max(a, b)
        assert _is_pos_zero(f(-0.0, 0.0, ctx=fp.FP64))

    def test_max_pos_neg_zero(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return max(a, b)
        assert _is_pos_zero(f(0.0, -0.0, ctx=fp.FP64))


class TestVariadicSanity:
    """``Min`` / ``Max`` still behave correctly on ordinary inputs —
    the NaN / signed-zero handling must not break the finite case."""

    def test_min_pairwise(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return min(a, b)
        assert float(f(1.0, 2.0, ctx=fp.FP64)) == 1.0
        assert float(f(2.0, 1.0, ctx=fp.FP64)) == 1.0

    def test_max_pairwise(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return max(a, b)
        assert float(f(1.0, 2.0, ctx=fp.FP64)) == 2.0
        assert float(f(2.0, 1.0, ctx=fp.FP64)) == 2.0

    def test_min_three_args(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
            return min(a, b, c)
        assert float(f(3.0, 1.0, 2.0, ctx=fp.FP64)) == 1.0

    def test_max_with_inf(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return max(a, b)
        assert math.isinf(float(f(math.inf, 1.0, ctx=fp.FP64)))

    def test_min_with_neg_inf(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            return min(a, b)
        result = float(f(-math.inf, 1.0, ctx=fp.FP64))
        assert math.isinf(result) and result < 0


# ---------------------------------------------------------------------------
# Reduce form — ``min(xs)`` → ``AMin``, ``max(xs)`` → ``AMax``
# ---------------------------------------------------------------------------


class TestReduceNaN:
    """``AMin`` / ``AMax`` propagate NaN from any list position."""

    def test_amin_nan_first(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return min(xs)
        assert _is_nan(f([math.nan, 1.0, 2.0], ctx=fp.FP64))

    def test_amin_nan_last(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return min(xs)
        assert _is_nan(f([1.0, 2.0, math.nan], ctx=fp.FP64))

    def test_amax_nan_middle(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return max(xs)
        assert _is_nan(f([1.0, math.nan, 2.0], ctx=fp.FP64))

    def test_amin_singleton_nan(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return min(xs)
        assert _is_nan(f([math.nan], ctx=fp.FP64))


class TestReduceSignedZero:
    """``AMin`` / ``AMax`` respect IEEE 754 signed-zero ordering."""

    def test_amin_neg_then_pos_zero(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return min(xs)
        assert _is_neg_zero(f([-0.0, 0.0], ctx=fp.FP64))

    def test_amin_pos_then_neg_zero(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return min(xs)
        assert _is_neg_zero(f([0.0, -0.0], ctx=fp.FP64))

    def test_amax_neg_then_pos_zero(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return max(xs)
        assert _is_pos_zero(f([-0.0, 0.0], ctx=fp.FP64))

    def test_amax_pos_then_neg_zero(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return max(xs)
        assert _is_pos_zero(f([0.0, -0.0], ctx=fp.FP64))

    def test_amin_zeros_among_positives(self):
        """``min([1.0, -0.0, 0.0, 2.0])`` → -0 (not just any zero)."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return min(xs)
        assert _is_neg_zero(f([1.0, -0.0, 0.0, 2.0], ctx=fp.FP64))


class TestReduceSanity:
    """``AMin`` / ``AMax`` correctness on ordinary lists."""

    def test_amin_basic(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return min(xs)
        assert float(f([3.0, 1.0, 4.0, 1.0, 5.0], ctx=fp.FP64)) == 1.0

    def test_amax_basic(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return max(xs)
        assert float(f([3.0, 1.0, 4.0, 1.0, 5.0], ctx=fp.FP64)) == 5.0

    def test_amin_singleton(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return min(xs)
        assert float(f([42.0], ctx=fp.FP64)) == 42.0

    def test_amin_empty_raises(self):
        """Empty list is UB at the FPy contract level; the interpreter
        surfaces it as a Python ``ValueError`` (matches ``min([])``)."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            return min(xs)
        with pytest.raises(ValueError):
            f([], ctx=fp.FP64)
