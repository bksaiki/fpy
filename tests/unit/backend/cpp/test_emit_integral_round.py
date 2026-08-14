"""
Rounding under an integer context, emitted as a libm call.

C++ rounds to an integral *value* in a floating-point type — ``std::trunc``
and friends are ``double -> double`` — so an integer rounding needs no integer
type.  That keeps the signed zero, needs no integer wide enough for the value,
and covers every rounding mode rather than only ``RTZ``.

The context's edges become assertions bracketing the call: an operand it has no
result for, and a result past its bound.
"""

import pytest

import fpy2 as fp
from fpy2.backend.cpp import CppCompiler
from fpy2.backend.cpp.compiler import CppCompileError
from fpy2.number import MPBFixedContext
from fpy2.types import RealType

RM = fp.RoundingMode
ASSERT = fp.OverflowMode.ASSERT


def _emit(ctx, src=fp.FP64, body='round'):
    """Compile ``y = round(x)`` (or ``cast``) under *ctx*."""
    if body == 'round':
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            with ctx:
                y = fp.round(x)
            return y
    else:
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            with ctx:
                y = fp.cast(x)
            return y
    return CppCompiler().compile(f, arg_types=[RealType(src)])


class TestModeTable:
    """Each FPy rounding mode maps to the libm function with the same
    semantics, verified against the interpreter in
    ``tests/unit/analysis`` and against compiled output here."""

    @pytest.mark.parametrize('rm, fn', [
        (RM.RTZ, 'std::trunc'),
        (RM.RTN, 'std::floor'),
        (RM.RTP, 'std::ceil'),
        (RM.RNA, 'std::round'),
        (RM.RNE, 'std::nearbyint'),
    ], ids=['rtz', 'rtn', 'rtp', 'rna', 'rne'])
    def test_mode_picks_its_function(self, rm, fn):
        out = _emit(MPBFixedContext(-1, fp.RealFloat(exp=10, c=1), rm=rm, overflow=ASSERT))
        assert fn in out
        # not a cast to an integer type: the value stays in a float
        assert 'static_cast<int' not in out

    @pytest.mark.parametrize('rm', [RM.RAZ, RM.RTO, RM.RTE],
                             ids=['raz', 'rto', 'rte'])
    def test_mode_without_a_function_is_refused(self, rm):
        """A refusal with a location, not a crash."""
        with pytest.raises(CppCompileError, match='no libm function'):
            _emit(MPBFixedContext(-1, fp.RealFloat(exp=10, c=1), rm=rm, overflow=ASSERT))


class TestAssertions:
    """A context states which values it has no result for; each statement
    becomes an assertion."""

    def test_operand_guard_when_specials_are_refused(self):
        """Neither NaN nor infinity representable and no substitute stated, so
        both collapse to one finiteness test."""
        out = _emit(MPBFixedContext(-1, fp.RealFloat(exp=10, c=1), overflow=ASSERT))
        assert 'assert(std::isfinite(' in out
        assert 'rounding is undefined for this value' in out

    def test_only_nan_refused(self):
        """With infinity representable, the guard narrows to NaN alone."""
        out = _emit(MPBFixedContext(
            -1, fp.RealFloat(exp=10, c=1), overflow=ASSERT, enable_inf=True))
        assert 'assert(!std::isnan(' in out
        assert 'std::isfinite' not in out

    def test_no_guard_when_both_are_representable(self):
        """libm passes NaN and the infinities through unchanged, so a context
        that admits them needs nothing said."""
        out = _emit(MPBFixedContext(
            -1, fp.RealFloat(exp=10, c=1), overflow=ASSERT,
            enable_nan=True, enable_inf=True))
        assert 'isfinite' not in out and 'isnan' not in out

    def test_bound_assertion_for_overflow_assert(self):
        out = _emit(MPBFixedContext(-1, fp.RealFloat(exp=11, c=1), overflow=ASSERT))
        assert 'assert(std::fabs(' in out
        assert '2048' in out
        assert 'overflow occurred so rounding is undefined' in out

    def test_no_bound_assertion_without_assert_mode(self):
        """``SATURATE`` is an edge *rule*, not a claim; it has no assertion to
        make and is not this lowering's business."""
        out = _emit(MPBFixedContext(
            -1, fp.RealFloat(exp=11, c=1), overflow=fp.OverflowMode.SATURATE))
        assert 'overflow occurred' not in out


class TestDeclines:
    """Shapes the libm lowering must not claim."""

    def test_non_zero_position_is_refused(self):
        """A position other than zero rounds to a multiple of ``2 ** n``, which
        needs the operand scaled first — `rescale_fixed`'s job, not the
        backend's, since doing it here would reintroduce an inexact ``exp2``."""
        with pytest.raises(CppCompileError, match='position zero'):
            _emit(MPBFixedContext(-4, fp.RealFloat(exp=10, c=1), overflow=ASSERT))

    def test_integer_storage_still_casts(self):
        """Without a signed zero the format lands in an integer type, where a
        cast is the better lowering and ``RTZ`` is what C++ does."""
        out = _emit(MPBFixedContext(
            -1, fp.RealFloat(exp=10, c=1), rm=RM.RTZ, overflow=ASSERT,
            enable_neg_zero=False))
        assert 'static_cast<int16_t>' in out
        assert 'std::trunc' not in out

    def test_signed_zero_decides_the_lowering(self):
        """The two lowerings want opposite answers: libm keeps a signed zero
        and an integer type cannot, so the flag picks between them."""
        kw = dict(overflow=ASSERT, rm=RM.RTZ)
        with_nz = _emit(MPBFixedContext(-1, fp.RealFloat(exp=10, c=1),
                                        enable_neg_zero=True, **kw))
        without = _emit(MPBFixedContext(-1, fp.RealFloat(exp=10, c=1),
                                        enable_neg_zero=False, **kw))
        assert 'std::trunc' in with_nz and 'static_cast<int' not in with_nz
        assert 'static_cast<int' in without and 'std::trunc' not in without


class TestFloatContextUnaffected:
    """A genuine float context still goes through ``fesetround``; the
    restructured validation must not have changed that."""

    def test_float_context_sets_the_mode(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.IEEEContext(8, 32, RM.RTZ):
                return fp.round(x)

        out = CppCompiler().compile(f, arg_types=[RealType(fp.FP64)])
        assert 'fesetround' in out
        assert 'static_cast<float>' in out

    def test_non_float_context_in_float_storage_is_reported(self):
        """``storage.is_float()`` does not imply a float context — that used to
        be a bare ``assert`` and is now a diagnostic."""
        # a fixed-point context at a non-zero position lands here
        with pytest.raises(CppCompileError):
            _emit(MPBFixedContext(-8, fp.RealFloat(exp=4, c=1), overflow=ASSERT))


class TestScaleByPowerOfTwo:
    """``2 ** n * v`` becomes ``std::ldexp(v, n)``.

    Not an optimization: ``std::pow`` is not required to return ``2 ** n``
    exactly (C11 F.10 requires correct rounding of no math function, and IEEE
    754 only *recommends* it for ``exp2``), and the product it feeds rounds a
    second time.  ``ldexp`` is IEEE 754's ``scaleB`` -- multiplication by an
    integral power of two, exact but for overflow and underflow.
    """

    @staticmethod
    def _lowered(src=fp.FP32, target=fp.FP16):
        import fpy2.strategies as st

        @fp.fpy(ctx=fp.REAL)
        def q(x: fp.Real) -> fp.Real:
            with target:
                y = fp.round(x)
            return y

        ref = st.monomorphize(q, args=[RealType(src)])
        low = st.rescale_fixed(st.float_to_fixed(
            st.unfold_overflow(ref, early_check=True)))
        return CppCompiler().compile(low)

    def test_the_lowering_uses_ldexp_not_pow(self):
        out = self._lowered()
        assert 'std::ldexp(' in out
        # `pow` survives only as the non-finite arm of the guard, never as the
        # way a finite scale is computed
        for line in out.splitlines():
            if 'std::pow(' in line:
                assert 'std::isfinite(' in line, line

    def test_a_possibly_nonfinite_exponent_falls_back_to_a_product(self):
        """``ldexp`` takes an ``int``, and converting a NaN or an infinity to
        one is undefined -- on x86-64 it gives ``INT_MIN``, so ``2 ** inf``
        would come back ``0`` where FPy says an infinity.

        An assertion would not do: it compiles out under ``NDEBUG``, leaving
        the undefined conversion in a release build.  ``std::pow`` defines all
        three cases exactly as FPy does, so the product is the faithful
        lowering precisely where the exponent is not finite.
        """
        out = self._lowered()
        assert 'std::isfinite(' in out
        assert 'std::ldexp(' in out
        assert 'std::pow(2.0,' in out, 'the non-finite arm must be a product'
        assert 'scaling is undefined' not in out, (
            'an assertion is not enough: NDEBUG would erase it'
        )

    def test_a_constant_scale_stays_a_multiply(self):
        """A constant power of two needs no call: the literal multiply is
        already exact, and folding it is better than either."""
        out = self._lowered()
        # the subnormal branch scales by a literal 2**24
        assert '16777216' in out

    def test_a_bare_power_of_two_also_uses_ldexp(self):
        """Not only as a multiply's operand: a power on its own would otherwise
        go through ``std::pow`` and land on the wrong grid."""
        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.FP64:
                return 2 ** n

        out = CppCompiler().compile(f, arg_types=[RealType(fp.SINT16)])
        assert 'std::ldexp' in out
        assert 'std::pow' not in out

    def test_both_operand_orders(self):
        """Multiplication commutes, so the scale may sit on either side."""
        @fp.fpy
        def left(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP64:
                return (2 ** n) * x

        @fp.fpy
        def right(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP64:
                return x * (2 ** n)

        tys = [RealType(fp.FP64), RealType(fp.SINT16)]
        for f in (left, right):
            out = CppCompiler().compile(f, arg_types=tys)
            assert out.count('std::ldexp') == 1, f.name
            assert 'std::pow' not in out, f.name

    def test_a_product_of_two_powers(self):
        """Both halves are exact: the multiply peephole takes the outer scale
        and the bare-power rule takes the inner one."""
        @fp.fpy
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                return (2 ** a) * (2 ** b)

        out = CppCompiler().compile(
            f, arg_types=[RealType(fp.SINT16), RealType(fp.SINT16)])
        assert out.count('std::ldexp') == 2
        assert 'std::pow' not in out

    def test_a_non_integer_exponent_is_left_alone(self):
        """``2 ** 0.5`` has no integral exponent, so the product stands."""
        @fp.fpy
        def f(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP64:
                return (2 ** n) * x

        out = CppCompiler().compile(
            f, arg_types=[RealType(fp.FP64), RealType(fp.FP64)])
        assert 'std::ldexp' not in out
        assert 'std::pow' in out

    def test_an_integer_exponent_needs_no_guard(self):
        """An exponent the analysis knows is finite by its format costs neither
        a branch nor a fallback -- only one derived from ``logb`` does."""
        @fp.fpy
        def f(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP64:
                return (2 ** n) * x

        out = CppCompiler().compile(
            f, arg_types=[RealType(fp.FP64), RealType(fp.SINT16)])
        assert 'std::ldexp' in out
        assert 'std::isfinite' not in out
        assert 'std::pow' not in out

    def test_declines_when_the_context_rounds_the_product(self):
        """``ldexp`` computes the *exact* product, so it may only stand in for
        ``round_C`` where that rounding is the identity.  Under a context
        narrower than the operands it would skip a rounding -- and skipping it
        gave `1 + 2**-20` back unchanged where FP16 says `1`.

        The op-table dispatch refuses this program for the same reason, so the
        peephole must decline rather than quietly answer.
        """
        @fp.fpy
        def f(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP16:
                return (2 ** n) * x

        with pytest.raises(CppCompileError, match='no matching signature'):
            CppCompiler().compile(
                f, arg_types=[RealType(fp.FP64), RealType(fp.SINT16)])
