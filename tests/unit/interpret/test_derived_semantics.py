"""
Interpreter checks backing ``docs/source/dev/derived-semantics.rst``.

Every AST node outside the core fragment is documented there as either
(i) evaluating like a core rule or (ii) desugaring to a core FPy program.
These tests make each claim executable:

* **desugarings** are checked by running the node form against the *exact
  baseline FPy function given in the doc* and asserting they agree **under
  several rounding contexts** (``REAL``, ``FP16``, ``FP64``) — so the check
  catches any place where the rounding context would make the two diverge.
  The ``*_desugar`` / ``*_node`` pairs below mirror the doc's functions.
* **rounding primitives** are checked for the documented rounding behaviour
  (exact under ``REAL``, correctly rounded under a finite context).

Grouped by the headings of the doc.  ``min`` / ``max`` (``test_min_max``),
list slicing (``test_list_slice``), and ``fst`` / ``snd``
(``test_tuple_accessors``) have dedicated files; they are re-touched lightly
here only so every group is self-contained.
"""

import math
from fractions import Fraction

import pytest

import fpy2 as fp

REAL = fp.REAL
FP16 = fp.FP16
FP64 = fp.FP64

# contexts spanning exact + two finite precisions
_CTXS = (REAL, FP16, FP64)


def _same(a, b) -> bool:
    """Structural equality over interpreter results (scalars / lists / tuples)."""
    a_seq = isinstance(a, (list, tuple))
    b_seq = isinstance(b, (list, tuple))
    if a_seq or b_seq:
        return (
            a_seq and b_seq
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    return a == b


def _fp_eq(a, b) -> bool:
    """Equality that treats NaNs as equal and distinguishes signed zeros."""
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return fa == fb and math.copysign(1.0, fa) == math.copysign(1.0, fb)


def _agree(node_fn, desugar_fn, args, ctxs=_CTXS):
    """Assert a node and its desugaring agree under each context."""
    for ctx in ctxs:
        a = node_fn(*args, ctx=ctx)
        b = desugar_fn(*args, ctx=ctx)
        assert _same(a, b), f'{node_fn.__name__} != desugaring under {ctx}'


# ---------------------------------------------------------------------------
# Literals and values  (Decnum, Hexnum, Integer, Rational, Digits, BoolVal,
# Var, ForeignVal)
# ---------------------------------------------------------------------------

_FOREIGN = 7  # closed-over Python value -> ForeignVal


class TestLiterals:

    def test_decnum_exact(self):
        @fp.fpy(ctx=REAL)
        def f() -> fp.Real:
            return 0.1  # denotes exactly 1/10 under REAL
        assert f() == Fraction(1, 10)

    def test_integer(self):
        @fp.fpy(ctx=REAL)
        def f() -> fp.Real:
            return 3
        assert f() == 3

    def test_rational_exact(self):
        @fp.fpy(ctx=REAL)
        def f() -> fp.Real:
            return fp.rational(1, 3)
        assert f() == Fraction(1, 3)

    def test_digits(self):
        @fp.fpy(ctx=REAL)
        def f() -> fp.Real:
            return fp.digits(3, -1, 2)  # 3 * 2**-1 == 1.5
        assert f() == Fraction(3, 2)

    def test_hexfloat(self):
        @fp.fpy(ctx=REAL)
        def f() -> fp.Real:
            return fp.hexfloat('0x1.8p+0')  # 1.5
        assert f() == Fraction(3, 2)

    def test_bool_and_var(self):
        @fp.fpy
        def f(x: fp.Real) -> bool:
            y = x > 0  # Var read below
            return y
        assert f(1.0, ctx=FP64) is True

    def test_foreign_value(self):
        @fp.fpy(ctx=REAL)
        def f() -> fp.Real:
            return _FOREIGN  # free variable -> ForeignVal
        assert f() == 7


# ---------------------------------------------------------------------------
# Constants  (ConstPi ... ConstSqrt1_2, ConstNan, ConstInf)
# ---------------------------------------------------------------------------

class TestConstants:

    def test_named_constants_round_under_context(self):
        cases = {
            fp.const_pi: math.pi,
            fp.const_e: math.e,
            fp.const_log2e: math.log2(math.e),
            fp.const_log10e: math.log10(math.e),
            fp.const_ln2: math.log(2),
            fp.const_pi_2: math.pi / 2,
            fp.const_pi_4: math.pi / 4,
            fp.const_1_pi: 1 / math.pi,
            fp.const_2_pi: 2 / math.pi,
            fp.const_2_sqrt_pi: 2 / math.sqrt(math.pi),
            fp.const_sqrt2: math.sqrt(2),
            fp.const_sqrt1_2: math.sqrt(0.5),
        }
        for const, expected in cases.items():
            @fp.fpy
            def f() -> fp.Real:
                return const()
            assert float(f(ctx=FP64)) == pytest.approx(expected), const

    def test_nan(self):
        @fp.fpy
        def f() -> fp.Real:
            return fp.nan()
        assert math.isnan(float(f(ctx=FP64)))

    def test_inf(self):
        @fp.fpy
        def f() -> fp.Real:
            return fp.inf()
        assert math.isinf(float(f(ctx=FP64)))


# ---------------------------------------------------------------------------
# Arithmetic  (rounds the exact result under C, like E-Add)
# ---------------------------------------------------------------------------

class TestArithmetic:

    def test_algebraic_exact_under_real(self):
        # under REAL the rounding is the identity, so the exact result is kept
        # (division is excluded: REAL cannot represent non-terminating ratios)
        @fp.fpy(ctx=REAL)
        def mul_() -> fp.Real:
            return 0.1 * 0.1  # exact 1/100 under REAL
        @fp.fpy(ctx=REAL)
        def sub_() -> fp.Real:
            return 0.3 - 0.1  # exact 1/5 under REAL
        assert mul_() == Fraction(1, 100)
        assert sub_() == Fraction(1, 5)

    def test_algebraic_rounds_under_finite_context(self):
        # the same operations are rounded to representable values under FP64
        @fp.fpy
        def mul_(a: fp.Real, b: fp.Real) -> fp.Real:
            return a * b
        @fp.fpy
        def div_(a: fp.Real, b: fp.Real) -> fp.Real:
            return a / b
        assert float(mul_(0.1, 0.1, ctx=FP64)) == 0.1 * 0.1
        assert float(div_(1.0, 3.0, ctx=FP64)) == 1.0 / 3.0

    def test_binary_ops_match_ieee(self):
        @fp.fpy
        def sub_(a: fp.Real, b: fp.Real) -> fp.Real:
            return a - b
        @fp.fpy
        def div_(a: fp.Real, b: fp.Real) -> fp.Real:
            return a / b
        @fp.fpy
        def pow_(a: fp.Real, b: fp.Real) -> fp.Real:
            return fp.pow(a, b)
        @fp.fpy
        def mod_(a: fp.Real, b: fp.Real) -> fp.Real:
            return a % b
        @fp.fpy
        def copysign_(a: fp.Real, b: fp.Real) -> fp.Real:
            return fp.copysign(a, b)
        @fp.fpy
        def fmod_(a: fp.Real, b: fp.Real) -> fp.Real:
            return fp.fmod(a, b)
        @fp.fpy
        def remainder_(a: fp.Real, b: fp.Real) -> fp.Real:
            return fp.remainder(a, b)
        @fp.fpy
        def atan2_(a: fp.Real, b: fp.Real) -> fp.Real:
            return fp.atan2(a, b)
        assert float(sub_(1.0, 2.0, ctx=FP64)) == -1.0
        assert float(div_(3.0, 4.0, ctx=FP64)) == 0.75
        assert float(pow_(2.0, 10.0, ctx=FP64)) == 1024.0
        assert float(mod_(7.0, 3.0, ctx=FP64)) == 1.0        # sign of divisor
        assert float(copysign_(3.0, -1.0, ctx=FP64)) == -3.0
        assert float(fmod_(-7.0, 3.0, ctx=FP64)) == -1.0     # sign of dividend
        assert float(remainder_(7.0, 3.0, ctx=FP64)) == 1.0  # nearest to zero
        assert float(atan2_(1.0, 1.0, ctx=FP64)) == pytest.approx(math.pi / 4)

    def test_fma_single_rounding(self):
        @fp.fpy
        def fma_(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
            return fp.fma(a, b, c)
        assert float(fma_(2.0, 3.0, 4.0, ctx=FP64)) == 10.0

    def test_unary_algebraic(self):
        @fp.fpy
        def abs_(x: fp.Real) -> fp.Real:
            return abs(x)
        @fp.fpy
        def neg_(x: fp.Real) -> fp.Real:
            return -x
        @fp.fpy
        def sqrt_(x: fp.Real) -> fp.Real:
            return fp.sqrt(x)
        @fp.fpy
        def cbrt_(x: fp.Real) -> fp.Real:
            return fp.cbrt(x)
        assert float(abs_(-2.5, ctx=FP64)) == 2.5
        assert float(neg_(2.5, ctx=FP64)) == -2.5
        assert float(sqrt_(9.0, ctx=FP64)) == 3.0
        assert float(cbrt_(27.0, ctx=FP64)) == 3.0

    def test_elementary_functions_sweep(self):
        # every remaining rounded elementary function: exercised and checked
        # against the host math library (correctly-rounded, within tolerance).
        cases = [
            (fp.sin, math.sin, 0.5),
            (fp.cos, math.cos, 0.5),
            (fp.tan, math.tan, 0.5),
            (fp.asin, math.asin, 0.5),
            (fp.acos, math.acos, 0.5),
            (fp.atan, math.atan, 0.5),
            (fp.sinh, math.sinh, 0.5),
            (fp.cosh, math.cosh, 0.5),
            (fp.tanh, math.tanh, 0.5),
            (fp.asinh, math.asinh, 0.5),
            (fp.acosh, math.acosh, 2.0),
            (fp.atanh, math.atanh, 0.5),
            (fp.exp, math.exp, 0.5),
            (fp.exp2, lambda x: 2 ** x, 0.5),
            (fp.expm1, math.expm1, 0.5),
            (fp.log, math.log, 2.0),
            (fp.log10, math.log10, 2.0),
            (fp.log1p, math.log1p, 2.0),
            (fp.log2, math.log2, 2.0),
            (fp.erf, math.erf, 0.5),
            (fp.erfc, math.erfc, 0.5),
            (fp.lgamma, math.lgamma, 2.5),
            (fp.tgamma, math.gamma, 2.5),
        ]
        for op, ref, x in cases:
            @fp.fpy
            def f(v: fp.Real) -> fp.Real:
                return op(v)
            assert float(f(x, ctx=FP64)) == pytest.approx(ref(x), rel=1e-9), op

    def test_composite_desugarings_across_contexts(self):
        # Composite ops round ONCE: compute the body exactly (under REAL), then
        # round the result.  The baselines below are the doc's functions; the
        # inputs include cases where a naive per-step-rounding form diverges.
        # (fdim / hypot are unimplemented under REAL, so checked on finite ctxs.)
        @fp.fpy
        def fdim_node(x: fp.Real, y: fp.Real) -> fp.Real:
            return fp.fdim(x, y)
        @fp.fpy
        def fdim_desugar(x: fp.Real, y: fp.Real) -> fp.Real:
            with REAL:
                t = max(x - y, 0)
            return fp.round(t)

        @fp.fpy
        def hypot_node(x: fp.Real, y: fp.Real) -> fp.Real:
            return fp.hypot(x, y)
        @fp.fpy
        def hypot_desugar(x: fp.Real, y: fp.Real) -> fp.Real:
            with REAL:
                t = x * x + y * y
            return fp.sqrt(t)
        # a naive form that rounds every intermediate -- must NOT match, else
        # the single-rounding check is vacuous
        @fp.fpy
        def hypot_naive(x: fp.Real, y: fp.Real) -> fp.Real:
            return fp.sqrt(x * x + y * y)

        inputs = [(5.3, 2.1), (2.1, 5.3), (0.7, 0.9), (3.7, 5.9), (1.3, 8.1)]
        for args in inputs:
            _agree(fdim_node, fdim_desugar, args, ctxs=(FP16, FP64))
            _agree(hypot_node, hypot_desugar, args, ctxs=(FP16, FP64))

        # guard: single-rounding genuinely differs from the naive form, so the
        # agreement above is meaningful and not a coincidence of the inputs
        assert hypot_node(0.7, 0.9, ctx=FP16) != hypot_naive(0.7, 0.9, ctx=FP16)
        assert hypot_node(3.7, 5.9, ctx=FP64) != hypot_naive(3.7, 5.9, ctx=FP64)

    def test_selection_no_rounding(self):
        # Max / Min return an operand exactly, even under a finite context
        @fp.fpy
        def max_(a: fp.Real, b: fp.Real) -> fp.Real:
            return max(a, b)
        @fp.fpy
        def min_(a: fp.Real, b: fp.Real) -> fp.Real:
            return min(a, b)
        assert float(max_(0.1, 0.2, ctx=FP64)) == 0.2
        assert float(min_(0.1, 0.2, ctx=FP64)) == 0.1

    def test_min_max_desugarings(self):
        # baselines mirror the doc: NaN propagates, ±0 ties break by sign
        @fp.fpy
        def max_node(x: fp.Real, y: fp.Real) -> fp.Real:
            return max(x, y)
        @fp.fpy
        def max_desugar(x: fp.Real, y: fp.Real) -> fp.Real:
            if fp.isnan(x) or fp.isnan(y):
                return x if fp.isnan(x) else y
            return x if x > y or (x == y and not fp.signbit(x)) else y

        @fp.fpy
        def min_node(x: fp.Real, y: fp.Real) -> fp.Real:
            return min(x, y)
        @fp.fpy
        def min_desugar(x: fp.Real, y: fp.Real) -> fp.Real:
            if fp.isnan(x) or fp.isnan(y):
                return x if fp.isnan(x) else y
            return x if x < y or (x == y and fp.signbit(x)) else y

        nan = math.nan
        cases = [
            (1.0, 2.0), (2.0, 1.0),
            (nan, 1.0), (1.0, nan), (nan, nan),   # NaN propagation
            (-0.0, 0.0), (0.0, -0.0),             # signed-zero ties
            (-3.0, -3.0),
        ]
        for x, y in cases:
            assert _fp_eq(max_node(x, y, ctx=FP64), max_desugar(x, y, ctx=FP64)), ('max', x, y)
            assert _fp_eq(min_node(x, y, ctx=FP64), min_desugar(x, y, ctx=FP64)), ('min', x, y)

        # guard: NaN propagates regardless of position (unlike Python's builtin
        # max, which is order-dependent for NaN)
        assert math.isnan(float(max_node(nan, 1.0, ctx=FP64)))
        assert math.isnan(float(max_node(1.0, nan, ctx=FP64)))
        assert math.isnan(float(min_node(nan, 1.0, ctx=FP64)))
        assert math.isnan(float(min_node(1.0, nan, ctx=FP64)))

    def test_round_to_integer(self):
        @fp.fpy
        def ceil_(x: fp.Real) -> fp.Real:
            return fp.ceil(x)
        @fp.fpy
        def floor_(x: fp.Real) -> fp.Real:
            return fp.floor(x)
        @fp.fpy
        def trunc_(x: fp.Real) -> fp.Real:
            return fp.trunc(x)
        @fp.fpy
        def roundint_(x: fp.Real) -> fp.Real:
            return fp.roundint(x)
        @fp.fpy
        def nearbyint_(x: fp.Real) -> fp.Real:
            return fp.nearbyint(x)
        assert float(ceil_(2.3, ctx=FP64)) == 3.0
        assert float(floor_(2.7, ctx=FP64)) == 2.0
        assert float(trunc_(-2.7, ctx=FP64)) == -2.0
        assert float(roundint_(2.5, ctx=FP64)) == 3.0    # round-half-away
        assert float(nearbyint_(2.5, ctx=FP64)) == 2.0   # uses context RNE


# ---------------------------------------------------------------------------
# Reductions  (Sum, AMax, AMin)
# ---------------------------------------------------------------------------

class TestReductions:

    def test_sum_is_fold_across_contexts(self):
        @fp.fpy
        def sum_node(xs: list[fp.Real]) -> fp.Real:
            return sum(xs)
        @fp.fpy
        def sum_desugar(xs: list[fp.Real]) -> fp.Real:
            acc = 0
            for x in xs:
                acc = acc + x
            return acc
        # values that round differently at each step under a finite context
        _agree(sum_node, sum_desugar, ([0.1, 0.2, 0.3, 0.4],))

    def test_amax_amin_reduce_form(self):
        @fp.fpy
        def amax_(xs: list[fp.Real]) -> fp.Real:
            return max(xs)
        @fp.fpy
        def amin_(xs: list[fp.Real]) -> fp.Real:
            return min(xs)
        xs = [3.0, 1.0, 2.0]
        assert float(amax_(xs, ctx=FP64)) == 3.0
        assert float(amin_(xs, ctx=FP64)) == 1.0


# ---------------------------------------------------------------------------
# Classification / inspection  (IsFinite, IsInf, IsNan, IsNormal, Signbit,
# Logb) -- booleans / integers, no rounding
# ---------------------------------------------------------------------------

class TestClassification:

    def test_predicates(self):
        @fp.fpy
        def isfinite_(x: fp.Real) -> bool:
            return fp.isfinite(x)
        @fp.fpy
        def isinf_(x: fp.Real) -> bool:
            return fp.isinf(x)
        @fp.fpy
        def isnan_(x: fp.Real) -> bool:
            return fp.isnan(x)
        @fp.fpy
        def isnormal_(x: fp.Real) -> bool:
            return fp.isnormal(x)
        @fp.fpy
        def signbit_(x: fp.Real) -> bool:
            return fp.signbit(x)
        assert isfinite_(1.0, ctx=FP64) is True
        assert isfinite_(math.inf, ctx=FP64) is False
        assert isinf_(math.inf, ctx=FP64) is True
        assert isnan_(math.nan, ctx=FP64) is True
        assert isnormal_(1.0, ctx=FP64) is True
        assert signbit_(-1.0, ctx=FP64) is True
        assert signbit_(1.0, ctx=FP64) is False

    def test_logb_integer_exponent(self):
        @fp.fpy
        def logb_(x: fp.Real) -> fp.Real:
            return fp.logb(x)
        assert float(logb_(8.0, ctx=FP64)) == 3.0
        assert float(logb_(1.0, ctx=FP64)) == 0.0


# ---------------------------------------------------------------------------
# Logical operators  (Not, And, Or) -- short-circuiting, boolean
# ---------------------------------------------------------------------------

class TestLogical:

    def test_not(self):
        @fp.fpy
        def f(x: bool) -> bool:
            return not x
        assert f(True, ctx=FP64) is False
        assert f(False, ctx=FP64) is True

    def test_and_or_match_conditional_desugaring(self):
        @fp.fpy
        def and_node(a: bool, b: bool) -> bool:
            return a and b
        @fp.fpy
        def and_desugar(a: bool, b: bool) -> bool:
            return b if a else False
        @fp.fpy
        def or_node(a: bool, b: bool) -> bool:
            return a or b
        @fp.fpy
        def or_desugar(a: bool, b: bool) -> bool:
            return True if a else b
        for a in (True, False):
            for b in (True, False):
                assert and_node(a, b, ctx=FP64) == and_desugar(a, b, ctx=FP64)
                assert or_node(a, b, ctx=FP64) == or_desugar(a, b, ctx=FP64)


# ---------------------------------------------------------------------------
# Comparisons  (Compare) -- chained -> conjunction of pairwise E-Lt
# ---------------------------------------------------------------------------

class TestComparison:

    def test_chain_matches_conjunction(self):
        @fp.fpy
        def chain(a: fp.Real, b: fp.Real, c: fp.Real) -> bool:
            return a < b <= c
        @fp.fpy
        def conj(a: fp.Real, b: fp.Real, c: fp.Real) -> bool:
            return (a < b) and (b <= c)
        for triple in [(1.0, 2.0, 2.0), (1.0, 2.0, 1.5), (2.0, 2.0, 3.0)]:
            assert chain(*triple, ctx=FP64) == conj(*triple, ctx=FP64)

    def test_all_operators(self):
        @fp.fpy
        def ops_(a: fp.Real, b: fp.Real) -> bool:
            return (a == a) and (a != b) and (a < b) and (a <= b) \
                and (b > a) and (b >= a)
        assert ops_(1.0, 2.0, ctx=FP64) is True


# ---------------------------------------------------------------------------
# Rounding operators  (Round, RoundAt, Cast)
# ---------------------------------------------------------------------------

class TestRoundingOperators:

    def test_round_applies_context(self):
        # round(1/3): identity under REAL, rounds under FP64
        @fp.fpy(ctx=REAL)
        def exact() -> fp.Real:
            return fp.round(fp.rational(1, 3))
        @fp.fpy(ctx=FP64)
        def rounded() -> fp.Real:
            return fp.round(fp.rational(1, 3))
        assert exact() == Fraction(1, 3)
        assert float(rounded()) == 1.0 / 3.0
        assert rounded() != Fraction(1, 3)  # rounding lost information

    def test_round_at_position(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return fp.round_at(x, -1)  # least retained digit 2**0 -> integer
        assert float(f(2.6, ctx=FP64)) == 3.0

    def test_cast_exact_is_identity(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return fp.cast(x)
        assert float(f(1.5, ctx=FP64)) == 1.5


# ---------------------------------------------------------------------------
# Compound data  (TupleExpr, ListExpr, ListRef, Fst, Snd, IfExpr, ListSlice,
# ListComp, Zip, Enumerate, Empty, Len, Size, Dim, Range1/2/3, Attribute)
# ---------------------------------------------------------------------------

class TestCompoundData:

    def test_tuple_list_ref(self):
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            xs = [a, b]
            t = (a, b)
            return xs[0] + fp.snd(t)
        assert float(f(3.0, 4.0, ctx=FP64)) == 7.0

    def test_fst_snd_match_tuple_pattern(self):
        @fp.fpy
        def fst_node(t: tuple[fp.Real, fp.Real]) -> fp.Real:
            return fp.fst(t)
        @fp.fpy
        def fst_desugar(t: tuple[fp.Real, fp.Real]) -> fp.Real:
            a, b = t
            return a
        @fp.fpy
        def snd_node(t: tuple[fp.Real, fp.Real]) -> fp.Real:
            return fp.snd(t)
        @fp.fpy
        def snd_desugar(t: tuple[fp.Real, fp.Real]) -> fp.Real:
            a, b = t
            return b
        _agree(fst_node, fst_desugar, ((5.0, 2.0),))
        _agree(snd_node, snd_desugar, ((5.0, 2.0),))

    def test_if_expr_matches_if_stmt(self):
        @fp.fpy
        def expr_(x: fp.Real) -> fp.Real:
            return x if x > 0 else -x
        @fp.fpy
        def stmt_(x: fp.Real) -> fp.Real:
            if x > 0:
                y = x
            else:
                y = -x
            return y
        for args in [(3.0,), (-3.0,)]:
            _agree(expr_, stmt_, args)

    def test_list_slice_matches_comprehension(self):
        @fp.fpy
        def slice_(xs: list[fp.Real]) -> list[fp.Real]:
            return xs[1:3]
        @fp.fpy
        def comp_(xs: list[fp.Real]) -> list[fp.Real]:
            return [xs[i] for i in range(1, 3)]
        _agree(slice_, comp_, ([10.0, 11.0, 12.0, 13.0],))

    def test_list_comp_builds_list_across_contexts(self):
        # tuple-binding comprehension over a zipped iterable, vs. its loop form
        @fp.fpy
        def comp_(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
            return [x * y for x, y in zip(xs, ys)]
        @fp.fpy
        def loop_(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
            pairs = zip(xs, ys)
            acc = fp.empty(len(pairs))
            j = 0
            for x, y in pairs:
                acc[j] = x * y
                j = j + 1
            return acc
        _agree(comp_, loop_, ([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))

    def test_zip_matches_index_comprehension(self):
        @fp.fpy
        def zip_node(xs: list[fp.Real], ys: list[fp.Real]) -> list[tuple[fp.Real, fp.Real]]:
            return zip(xs, ys)
        @fp.fpy
        def zip_desugar(xs: list[fp.Real], ys: list[fp.Real]) -> list[tuple[fp.Real, fp.Real]]:
            return [(xs[i], ys[i]) for i in range(len(xs))]
        _agree(zip_node, zip_desugar, ([1.0, 2.0], [3.0, 4.0]))

    def test_enumerate_matches_index_comprehension(self):
        @fp.fpy
        def enum_node(xs: list[fp.Real]) -> list[tuple[fp.Real, fp.Real]]:
            return enumerate(xs)
        @fp.fpy
        def enum_desugar(xs: list[fp.Real]) -> list[tuple[fp.Real, fp.Real]]:
            return [(i, xs[i]) for i in range(len(xs))]
        _agree(enum_node, enum_desugar, ([10.0, 20.0, 30.0],))

    def test_empty_len_dim_size(self):
        @fp.fpy
        def build(n: fp.Real) -> list[fp.Real]:
            xs = fp.empty(n)
            for i in range(n):
                xs[i] = i
            return xs
        @fp.fpy
        def length(xs: list[fp.Real]) -> fp.Real:
            return len(xs)
        @fp.fpy
        def dims(xs: list[fp.Real]) -> fp.Real:
            return fp.dim(xs)
        @fp.fpy
        def size0(xs: list[fp.Real]) -> fp.Real:
            return fp.size(xs, 0)
        xs = build(4, ctx=FP64)
        assert [float(v) for v in xs] == [0.0, 1.0, 2.0, 3.0]
        assert float(length(xs, ctx=FP64)) == 4.0
        assert float(dims(xs, ctx=FP64)) == 1.0
        assert float(size0(xs, ctx=FP64)) == 4.0

    def test_range_forms(self):
        @fp.fpy
        def r1() -> list[fp.Real]:
            return [i for i in range(3)]
        @fp.fpy
        def r2() -> list[fp.Real]:
            return [i for i in range(2, 5)]
        @fp.fpy
        def r3() -> list[fp.Real]:
            return [i for i in range(0, 6, 2)]
        assert [float(v) for v in r1(ctx=FP64)] == [0.0, 1.0, 2.0]
        assert [float(v) for v in r2(ctx=FP64)] == [2.0, 3.0, 4.0]
        assert [float(v) for v in r3(ctx=FP64)] == [0.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# Polymorphism: the structural ops move values without inspecting them, so they
# work on any element type (the doc annotates these with `Any`, not fp.Real).
# ---------------------------------------------------------------------------

class TestPolymorphism:

    def test_tuple_and_if_on_bools(self):
        @fp.fpy
        def fst_(t: tuple[bool, bool]) -> bool:
            return fp.fst(t)
        @fp.fpy
        def snd_(t: tuple[bool, bool]) -> bool:
            return fp.snd(t)
        @fp.fpy
        def if_(c: bool, a: bool, b: bool) -> bool:
            return a if c else b
        assert fst_((True, False), ctx=FP64) is True
        assert snd_((True, False), ctx=FP64) is False
        assert if_(False, True, False, ctx=FP64) is False

    def test_list_ops_on_bools(self):
        @fp.fpy
        def slice_(xs: list[bool], a: int, b: int) -> list[bool]:
            return xs[a:b]
        @fp.fpy
        def enum_(xs: list[bool]) -> list[tuple[fp.Real, bool]]:
            return enumerate(xs)
        assert list(slice_([True, False, True], 0, 2, ctx=FP64)) == [True, False]
        out = enum_([True, False], ctx=FP64)
        assert [(float(i), v) for i, v in out] == [(0.0, True), (1.0, False)]


# ---------------------------------------------------------------------------
# Statements  (StmtBlock, Assign, IndexedAssign, If1Stmt, IfStmt, WhileStmt,
# ForStmt, ContextStmt, AssertStmt, EffectStmt, ReturnStmt, PassStmt) plus
# Call (E-App) and a declared-context override (FuncMeta.ctx)
# ---------------------------------------------------------------------------

@fp.fpy
def _identity(x: fp.Real) -> fp.Real:
    return x


class TestStatements:

    def test_seq_assign_return(self):
        @fp.fpy
        def f(a: fp.Real) -> fp.Real:
            x = a
            y = x + 1
            return y
        assert float(f(2.0, ctx=FP64)) == 3.0

    def test_indexed_assign(self):
        @fp.fpy
        def f() -> list[fp.Real]:
            xs = fp.empty(3)
            xs[0] = 1
            xs[1] = 2
            xs[2] = 3
            return xs
        assert [float(v) for v in f(ctx=FP64)] == [1.0, 2.0, 3.0]

    def test_if1_and_if(self):
        @fp.fpy
        def if1(x: fp.Real) -> fp.Real:
            y = x
            if x < 0:
                y = -x
            return y
        @fp.fpy
        def if2(x: fp.Real) -> fp.Real:
            if x < 0:
                y = -x
            else:
                y = x
            return y
        for args in [(2.0,), (-2.0,)]:
            _agree(if1, if2, args)

    def test_while_and_for_agree(self):
        @fp.fpy
        def while_(n: fp.Real) -> fp.Real:
            acc = 0
            i = 0
            while i < n:
                acc = acc + i
                i = i + 1
            return acc
        @fp.fpy
        def for_(n: fp.Real) -> fp.Real:
            acc = 0
            for i in range(n):
                acc = acc + i
            return acc
        _agree(while_, for_, (5,))

    def test_assert_and_pass(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            assert x > 0, 'must be positive'
            pass
            return x
        assert float(f(2.0, ctx=FP64)) == 2.0

    def test_effect_statement(self):
        # a bare call evaluated for effect; result discarded, evaluation
        # continues normally
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            _identity(x)
            return x + 1
        assert float(f(1.0, ctx=FP64)) == 2.0

    def test_call_is_application(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return _identity(x) + 1
        assert float(f(4.0, ctx=FP64)) == 5.0

    def test_context_statement_scopes_rounding(self):
        # the inner context governs only its body; the outer context resumes
        @fp.fpy(ctx=REAL)
        def f() -> fp.Real:
            x = fp.rational(1, 3)               # exact under REAL
            with FP64 as c:
                y = fp.round(fp.rational(1, 3))  # rounded under FP64
            return x - y                         # exact - rounded != 0
        assert f() != 0

    def test_declared_context_override(self):
        # a function with a declared context runs its body under that context
        # regardless of the caller's context (FuncMeta.ctx)
        @fp.fpy(ctx=FP64)
        def callee() -> fp.Real:
            return fp.round(fp.rational(1, 3))
        @fp.fpy(ctx=REAL)
        def caller() -> fp.Real:
            return callee()
        assert float(caller()) == 1.0 / 3.0     # rounded under FP64...
        assert caller() != Fraction(1, 3)       # ...not the exact REAL value
