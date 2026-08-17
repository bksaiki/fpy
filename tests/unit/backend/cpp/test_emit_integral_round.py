"""
Rounding under a fixed-point context, emitted in floating-point.

C++ rounds to an integral *value* in a floating-point type — ``std::trunc`` and
friends are ``double -> double`` — so an integer rounding needs no integer type.
That keeps the signed zero and needs no integer wide enough for the value.

All eight FPy modes are covered.  Five are one libm call; ``RAZ`` takes three,
and ``RTO``/``RTE`` ask for the parity of the *result*, which no libm function
reports — those are composed from the same calls, so the lowered program still
depends on ``std::`` alone.

The context's edges become assertions around it: an operand it has no result
for, and a result past its bound.
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
    semantics.  This pins the mapping; ``test_lowered_roundtrip.py`` is where
    the compiled result is diffed against the interpreter."""

    @pytest.mark.parametrize('rm, fn', [
        (RM.RTZ, 'std::trunc'),
        (RM.RTN, 'std::floor'),
        (RM.RTP, 'std::ceil'),
        (RM.RNA, 'std::round'),
        (RM.RNE, 'std::nearbyint'),
        (RM.RAZ, 'std::copysign'),
    ], ids=['rtz', 'rtn', 'rtp', 'rna', 'rne', 'raz'])
    def test_mode_picks_its_function(self, rm, fn):
        out = _emit(MPBFixedContext(-1, fp.RealFloat(exp=10, c=1), rm=rm, overflow=ASSERT))
        assert fn in out
        # not a cast to an integer type: the value stays in a float
        assert 'static_cast<int' not in out

    def test_away_from_zero_takes_three_calls(self):
        """``ceil`` rounds away from zero only above zero, so the sign comes off
        and goes back on.  ``copysign`` also carries a ``-0.0`` through, which the
        other five modes do natively and this lowering requires.

        The operand is named twice, so it has to be bound first -- otherwise a
        temporary-producing argument would be evaluated twice.
        """
        out = _emit(MPBFixedContext(-1, fp.RealFloat(exp=10, c=1), rm=RM.RAZ,
                                    overflow=ASSERT))
        assert 'std::copysign(std::ceil(std::fabs(x)), x)' in out

    def test_the_table_covers_every_mode(self):
        """The table is complete, so no rounding mode is refused any more.

        Asked as a question rather than assumed: a mode added to
        :class:`RoundingMode` later gets a refusal with a location from
        `_INTEGRAL_MODES` instead of being silently mis-lowered.
        """
        from fpy2.backend.cpp.emitter import CppEmitter

        missing = set(RM) - CppEmitter._INTEGRAL_MODES
        assert not missing, f'no integral spelling for {missing}'

    @pytest.mark.parametrize('rm', [RM.RTO, RM.RTE], ids=['rto', 'rte'])
    def test_parity_modes_need_no_support_library(self, rm):
        """Both ask for the parity of the *result*, which no libm function
        reports, so each is composed -- but out of ``std::`` alone.

        In particular not C23 ``roundeven``, which no compiler is required to
        have; `RTE` uses ``std::nearbyint`` for it, the same stand-in `RNE`
        already makes.
        """
        out = _emit(MPBFixedContext(-1, fp.RealFloat(exp=10, c=1), rm=rm,
                                    overflow=ASSERT))
        assert 'fpy::' not in out
        assert 'roundeven' not in out
        # a conditional, not a single call
        assert '?' in out

    def test_toward_even_halves_rounds_to_nearest_and_doubles(self):
        """`mpfx`'s ``round_to_integral``, with ``std::nearbyint`` standing in for
        C23 ``roundeven``.  The ``fabs`` comparison separates the one case that
        must not move -- an odd integer, already exact and a full step from that
        even neighbour."""
        out = _emit(MPBFixedContext(-1, fp.RealFloat(exp=10, c=1), rm=RM.RTE,
                                    overflow=ASSERT))
        assert 'std::nearbyint(x * 0.5)' in out
        assert 'std::fabs(x - ' in out

    def test_toward_even_inherits_the_fe_tonearest_precondition(self):
        """``std::nearbyint`` follows the dynamic mode, so `RTE` is refused where
        `RNE` is -- under an enclosing scope that set another one."""
        outer = fp.IEEEContext(8, 32, RM.RTZ)
        inner = MPBFixedContext(-1, fp.RealFloat(exp=10, c=1), rm=RM.RTE,
                                overflow=ASSERT)

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            with outer:
                t = fp.round(x)
                with inner:
                    y = fp.round(t)
            return y

        with pytest.raises(CppCompileError, match='FE_TONEAREST'):
            CppCompiler().compile(f, arg_types=[RealType(fp.FP64)])

    def test_the_new_modes_are_not_aliases_of_the_old(self):
        """Keeps the differential from being vacuous.

        One value cannot separate all four directed modes -- at 2.5, ``RTZ`` and
        ``RTE`` agree and so do ``RAZ`` and ``RTO``.  It takes 3.5 as well, where
        the parity pair swaps and the other two do not move.
        """
        def r(rm, x):
            ctx = MPBFixedContext(-1, fp.RealFloat(exp=10, c=1), rm=rm,
                                  overflow=ASSERT)

            @fp.fpy(ctx=fp.REAL)
            def f(v: fp.Real) -> fp.Real:
                with ctx:
                    y = fp.round(v)
                return y
            return float(f(x))

        assert [r(rm, 2.5) for rm in (RM.RTZ, RM.RAZ, RM.RTO, RM.RTE)] == \
            [2.0, 3.0, 3.0, 2.0]
        assert [r(rm, 3.5) for rm in (RM.RTZ, RM.RAZ, RM.RTO, RM.RTE)] == \
            [3.0, 4.0, 3.0, 4.0]


class TestAssertions:
    """A context states which values it has no result for; each statement becomes
    an assertion.  The *bound* assertions live in `test_round_fixed_bound.py`,
    with the differential that checks which values they admit."""

    def test_operand_guard_when_specials_are_refused(self):
        """Neither NaN nor infinity representable and no substitute stated, so
        both collapse to one finiteness test."""
        out = _emit(MPBFixedContext(-1, fp.RealFloat(exp=10, c=1), overflow=ASSERT))
        assert 'assert((std::isfinite(' in out
        assert 'rounding is undefined for this value' in out

    def test_only_nan_refused(self):
        """With infinity representable, the guard narrows to NaN alone."""
        out = _emit(MPBFixedContext(
            -1, fp.RealFloat(exp=10, c=1), overflow=ASSERT, enable_inf=True))
        assert 'assert((!std::isnan(' in out

    def test_no_operand_guard_when_both_are_representable(self):
        """libm passes NaN and the infinities through unchanged, so a context
        admitting both needs no guard on its operand.

        The bound assertion still has to *exempt* them: no magnitude test admits
        an infinity, and this context represents one.
        """
        out = _emit(MPBFixedContext(
            -1, fp.RealFloat(exp=10, c=1), overflow=ASSERT,
            enable_nan=True, enable_inf=True))
        assert 'rounding is undefined for this value' not in out
        assert '!std::isfinite(x) || std::fabs(' in out

    def test_bound_assertion_for_overflow_assert(self):
        out = _emit(MPBFixedContext(-1, fp.RealFloat(exp=11, c=1), overflow=ASSERT))
        assert 'assert((std::fabs(' in out
        assert '2048' in out
        assert 'overflow occurred so rounding is undefined' in out

    def test_asymmetric_bounds_get_two_comparisons(self):
        """``fabs`` states one magnitude, but the two bounds are independent --
        a two's-complement format runs to -128 and only to 127, and the legal
        most-negative value must not trip the assertion."""
        out = _emit(MPBFixedContext(
            -1, fp.RealFloat(exp=0, c=127),
            neg_maxval=fp.RealFloat(s=True, exp=0, c=128), overflow=ASSERT))
        assert 'fabs' not in out
        assert '-128' in out and '127' in out
        assert 'overflow occurred so rounding is undefined' in out


class TestDeclines:
    """Shapes the libm lowering must not claim."""

    def test_non_zero_position_is_refused(self):
        """A position other than zero rounds to a multiple of ``2 ** n``, which
        needs the operand scaled first — `rescale_fixed`'s job, not the
        backend's, since doing it here would reintroduce an inexact ``exp2``."""
        # float storage, so `_validate_context_rm` names it first
        with pytest.raises(CppCompileError, match='digits at position zero'):
            _emit(MPBFixedContext(-4, fp.RealFloat(exp=10, c=1), overflow=ASSERT))

    def test_signed_zero_decides_the_lowering(self):
        """The two lowerings want opposite answers: libm keeps a signed zero
        and an integer type cannot, so the flag picks between them."""
        kw = dict(overflow=ASSERT, rm=RM.RTZ)
        with_nz = _emit(MPBFixedContext(-1, fp.RealFloat(exp=10, c=1),
                                        enable_neg_zero=True, **kw))
        without = _emit(MPBFixedContext(-1, fp.RealFloat(exp=10, c=1),
                                        enable_neg_zero=False, **kw))
        # libm computes the result under one; the cast computes it under the other
        assert 'float _tmp1 = std::trunc(' in with_nz
        assert 'static_cast<int' not in with_nz
        assert 'static_cast<int' in without
        assert not any(
            'std::trunc' in ln and 'assert' not in ln
            for ln in without.splitlines()
        )


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
        """No ``pow`` at all: value classes prove both exponents finite, so even
        the guard's fallback arm is gone."""
        out = self._lowered()
        assert 'std::ldexp(' in out
        assert 'std::pow(' not in out

    def test_the_lowering_widens_before_scaling(self):
        """``std::ldexp`` is overloaded on its first argument, so scaling the
        `FP32` source directly would overflow in ``float`` where the `double`
        result is representable.  The value is widened first."""
        out = self._lowered()
        scale = [ln for ln in out.splitlines() if 'std::ldexp(' in ln]
        assert scale
        for ln in scale:
            assert 'float ' not in ln, ln
        assert 'static_cast<double>(x)' in out

    def test_a_possibly_nonfinite_exponent_falls_back_to_a_product(self):
        """``ldexp`` takes an ``int``, and converting a NaN or an infinity to
        one is undefined -- on x86-64 it gives ``INT_MIN``, so ``2 ** inf``
        would come back ``0`` where FPy says an infinity.

        An assertion would not do: it compiles out under ``NDEBUG``, leaving
        the undefined conversion in a release build.  ``std::pow`` defines all
        three cases exactly as FPy does, so the product is the faithful
        lowering precisely where the exponent is not finite.

        Reached by an exponent whose *format* admits both specials while still
        representing only integers, since neither a lowered rounding nor an
        integer-typed exponent leaves the question open any more.
        """
        exp_ctx = MPBFixedContext(
            -1, fp.RealFloat(exp=0, c=100), enable_nan=True, enable_inf=True)

        @fp.fpy(ctx=fp.REAL)
        def q(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP64:
                y = (2 ** n) * x
            return y

        out = CppCompiler().compile(
            q, arg_types=[RealType(fp.FP64), RealType(exp_ctx)])
        assert 'std::isfinite(n)' in out
        assert 'std::ldexp(' in out
        assert 'std::pow(2.0,' in out, 'the non-finite arm must be a product'
        assert 'scaling is undefined' not in out, (
            'an assertion is not enough: NDEBUG would erase it'
        )

    def test_a_guarding_branch_is_what_removes_the_select(self):
        """Two programs differing only in a branch, so the removal is due to the
        branch and not to the exponent's format -- which admits both specials
        either way.  Reading the branch is what value classes add; the lowered
        rounding above gets the same treatment from its ``elif`` ladder."""
        exp_ctx = MPBFixedContext(
            -1, fp.RealFloat(exp=0, c=100), enable_nan=True, enable_inf=True)

        @fp.fpy(ctx=fp.REAL)
        def bare(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP64:
                y = (2 ** n) * x
            return y

        @fp.fpy(ctx=fp.REAL)
        def guarded(x: fp.Real, n: fp.Real) -> fp.Real:
            if fp.isfinite(n):
                with fp.FP64:
                    y = (2 ** n) * x
            else:
                y = 0
            return y

        tys = [RealType(fp.FP64), RealType(exp_ctx)]
        assert 'std::pow(2.0,' in CppCompiler().compile(bare, arg_types=tys)
        assert 'std::pow(' not in CppCompiler().compile(guarded, arg_types=tys)

    def test_a_constant_scale_stays_a_multiply(self):
        """A constant power of two needs no call: the literal multiply is
        already exact, and folding it is better than either."""
        out = self._lowered()
        # the subnormal branch scales by a literal 2**24
        assert '16777216' in out

    def test_a_bare_power_of_two_also_uses_ldexp(self):
        """Not only as a multiply's operand: a power on its own would otherwise
        go through ``std::pow``, which may not return the exact power."""
        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.FP64:
                return 2 ** n

        out = CppCompiler().compile(f, arg_types=[RealType(fp.SINT8)])
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

        tys = [RealType(fp.FP64), RealType(fp.SINT8)]
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
            f, arg_types=[RealType(fp.SINT8), RealType(fp.SINT8)])
        assert out.count('std::ldexp') == 2
        assert 'std::pow' not in out

    def test_declines_when_the_power_itself_is_rounded(self):
        """One ``ldexp`` replaces *two* rounded steps, so the intermediate
        ``2 ** n`` must be exact under the context too.

        With a ``SINT16`` exponent the power spans ``2 ** 32767``, which `FP64`
        rounds -- and `2 ** -1080` is already zero before it reaches the
        product, a value ``ldexp`` would never form.  Inference records the
        *clipped* format for the power, so this cannot be read off the recorded
        bound; :func:`fpy2.analysis.format_infer.exact_exp2` is asked instead.
        """
        @fp.fpy
        def f(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP64:
                return (2 ** n) * x

        out = CppCompiler().compile(
            f, arg_types=[RealType(fp.FP64), RealType(fp.SINT16)])
        assert 'std::ldexp' not in out
        assert 'std::pow' in out

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
            f, arg_types=[RealType(fp.FP64), RealType(fp.SINT8)])
        assert 'std::ldexp' in out
        assert 'std::isfinite' not in out
        assert 'std::pow' not in out

    def test_does_not_rescue_a_program_the_dispatch_refuses(self):
        """``ldexp`` computes the *exact* product, so it may only stand in for
        ``round_C`` where that rounding is the identity.  The op-table refuses
        this program because `FP16` has no storage matching its `FP64` operand;
        were the peephole to fire it would answer first, and the emitted
        ``ldexp`` would skip the narrowing rounding the context asked for.

        So the error escaping is the assertion: the peephole declined.
        """
        @fp.fpy
        def f(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP16:
                return (2 ** n) * x

        with pytest.raises(CppCompileError, match='no matching signature'):
            CppCompiler().compile(
                f, arg_types=[RealType(fp.FP64), RealType(fp.SINT8)])
