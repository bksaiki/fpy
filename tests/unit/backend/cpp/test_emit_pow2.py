"""
A power of two in the cpp emitter: ``2 ** n`` scales by ``std::ldexp``.
"""

import re

import pytest

import fpy2 as fp
from fpy2.backend.cpp import CppCompiler
from fpy2.backend.cpp.compiler import CppCompileError
from fpy2.number import MPBFixedContext
from fpy2.types import BoolType, RealType


class TestTheLoweredScaleInStaysNarrow:
    """The normal branch scales by ``2 ** -exp`` in the operand's own type.

    `FloatToFixed` takes that branch on ``abs(x) >= 2 ** emin``, which tells
    inference the finest digit ``x`` can carry (`_implied_magnitude`).  Without
    that the operand's format keeps a digit it cannot have, and the scale-in
    widens to ``double`` -- correct, and a type wider than the value needs.
    """

    def test_the_scale_in_takes_the_operand_unwidened(self):
        import fpy2.strategies as strat

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            with fp.FP16:
                return fp.round(x)

        g = strat.simplify(strat.rescale_fixed(strat.float_to_fixed(
            strat.unfold_overflow(strat.unfold_special(f), early_check=True))))
        out = CppCompiler().compile(g, arg_types=[RealType(fp.FP32)])
        assert re.search(r'float \w+ = std::ldexp\(x,', out), out


class TestScaleByPowerOfTwo:
    """``2 ** n * v`` becomes ``std::ldexp(v, n)``.

    Not an optimization: ``std::pow`` is not required to return ``2 ** n``
    exactly (C11 F.10 requires correct rounding of no math function, and IEEE
    754 only *recommends* it for ``exp2``), and the product it feeds rounds a
    second time.  ``ldexp`` is IEEE 754's ``scaleB`` -- multiplication by an
    integral power of two, exact but for overflow and underflow.
    """

    @pytest.fixture(scope='class')
    def lowered(self) -> str:
        """An `FP16` rounding of an `FP32` value, lowered to fixed point."""
        import fpy2.strategies as st

        @fp.fpy(ctx=fp.REAL)
        def q(x: fp.Real) -> fp.Real:
            with fp.FP16:
                y = fp.round(x)
            return y

        ref = st.monomorphize(q, args=[RealType(fp.FP32)])
        low = st.rescale_fixed(st.float_to_fixed(
            st.unfold_overflow(ref, early_check=True)))
        return CppCompiler().compile(low)

    def test_the_lowering_uses_ldexp_not_pow(self, lowered):
        """No ``pow`` at all: value classes prove both exponents finite, so even
        the guard's fallback arm is gone."""
        out = lowered
        assert 'std::ldexp(' in out
        assert 'std::pow(' not in out

    def test_the_scale_needs_no_widening(self, lowered):
        """``std::ldexp`` is overloaded on its first argument, so the scale runs
        in whatever type the value is stored in.

        That used to force a widening to ``double``: the scale-in's bound was
        inferred at ``2 ** 287``, far past what ``float`` holds, even though the
        true value is in ``[2 ** 10, 2 ** 11)``.  Branch refinement now reads the
        guards and reports ``prec=24, exp=-53, bound ~ 2 ** 40``, which ``float``
        does hold -- so the cast is gone and the scale still lands exactly, which
        ``test_lowered_roundtrip.py`` checks bit-for-bit across fourteen
        formats.

        The peephole fuses the power into the scale, so there is no separate
        product to widen either."""
        out = lowered
        scale = [ln for ln in out.splitlines() if 'std::ldexp(' in ln]
        assert scale
        # nothing here is widened: the scale runs in `float`, on `float`
        assert all('static_cast<double>' not in ln for ln in scale), scale
        assert 'std::ldexp(x, ' in out, out
        # the power is fused in, not multiplied separately
        assert 'std::pow(' not in out
        assert not [ln for ln in out.splitlines() if '* x)' in ln], out

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
        """The program above behind a branch, so the removal is due to the
        branch and not to the exponent's format -- which admits both specials
        either way.  Reading the branch is what value classes add; the lowered
        rounding above gets the same treatment from its ``elif`` ladder."""
        exp_ctx = MPBFixedContext(
            -1, fp.RealFloat(exp=0, c=100), enable_nan=True, enable_inf=True)

        @fp.fpy(ctx=fp.REAL)
        def guarded(x: fp.Real, n: fp.Real) -> fp.Real:
            if fp.isfinite(n):
                with fp.FP64:
                    y = (2 ** n) * x
            else:
                y = 0
            return y

        out = CppCompiler().compile(
            guarded, arg_types=[RealType(fp.FP64), RealType(exp_ctx)])
        assert 'std::pow(' not in out

    def test_a_constant_scale_stays_a_multiply(self, lowered):
        """A constant power of two needs no call: the literal multiply is
        already exact, and folding it is better than either."""
        out = lowered
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

    def test_the_exponent_is_not_re_read_at_the_product(self):
        """The scale must not reach its exponent through a name: a backend gives
        two definitions of one source name a single C++ variable, so reading it
        at the product reads whatever a later branch put there."""

        @fp.fpy
        def f(x: fp.Real, n: fp.Real, m: fp.Real, c: bool) -> fp.Real:
            with fp.FP64:
                k = n
                p = 2 ** k
                if c:
                    k = m
                return x * p

        out = CppCompiler().compile(f, arg_types=[
            RealType(fp.FP64), RealType(fp.SINT8), RealType(fp.SINT8),
            BoolType(),
        ])
        assert 'std::ldexp(x' not in out, out

    def test_a_compound_exponent_in_a_while_condition(self):
        """`_ldexp_call` binds a compound exponent, so the condition needs a
        statement -- which the emitter has nowhere to put.  Statement form has
        to rotate the loop, and refuses at codegen if it did not."""

        @fp.fpy
        def f(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while (2 ** (n + 1)) > 1.0:
                    y = y - 1.0
                return y

        out = CppCompiler().compile(
            f, arg_types=[RealType(fp.FP64), RealType(fp.SINT8)])
        assert 'while' in out

    def test_both_operand_orders(self):
        """Multiplication commutes, so the scale may sit on either side.  An
        exponent the analysis knows is finite by its format costs neither a
        branch nor a fallback -- only one derived from ``logb`` does."""
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
            assert 'std::isfinite' not in out, f.name

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
