"""
A ``static_cast`` is only the context's rounding where the two formats agree.

Storage is chosen to *contain* a format, not to equal it, so casting into it
rounds to the storage's format.  `FP16` gets ``float``, and
``static_cast<float>(1024.5)`` is ``1024.5`` where `FP16.round` says ``1024`` --
so every context narrower than its storage silently answered in the wrong format,
as did a saturating or stochastic one of the *same* width.

Arithmetic never had the problem: the op table matches whole contexts
(:meth:`CppOp.matches`), so ``x + y`` under `FP16` was already refused.
``Round`` and ``Cast`` bypass the table, and this pins that they no longer do.
"""

import re

import pytest

import fpy2 as fp
import fpy2.strategies as st
from fpy2.backend.cpp import CppCompiler
from fpy2.backend.cpp.compiler import CppCompileError
from fpy2.backend.cpp.target import is_native_ctx
from fpy2.types import RealType

# `C++` would read as a regex quantifier
_REFUSAL = re.escape('has no C++ analogue')

_SATURATING = fp.IEEEContext(8, 32, fp.RM.RNE, fp.OverflowMode.SATURATE)
_STOCHASTIC = fp.IEEEContext(8, 32, fp.RM.RNE, fp.OverflowMode.OVERFLOW, 4)

# every reason a float context is not what a cast into its storage does
_REFUSED = [
    pytest.param(fp.FP16, id='fp16'),
    pytest.param(fp.IEEEContext(5, 16), id='ieee_5_16'),
    pytest.param(fp.MX_E5M2, id='e5m2'),
    pytest.param(fp.MX_E4M3, id='e4m3'),
    pytest.param(fp.MX_E2M1, id='e2m1'),
    pytest.param(_SATURATING, id='saturating'),
    pytest.param(_STOCHASTIC, id='stochastic'),
]

# contexts a cast *does* implement: the storage's own format, any `fesetround`
# mode, plus the integer contexts whose cast rounds toward zero
_ALLOWED = [
    pytest.param(fp.FP32, id='fp32'),
    pytest.param(fp.FP64, id='fp64'),
    pytest.param(fp.IEEEContext(8, 32, fp.RM.RTZ), id='fp32_rtz'),
    pytest.param(fp.IEEEContext(8, 32, fp.RM.RTP), id='fp32_rtp'),
    pytest.param(fp.IEEEContext(11, 64, fp.RM.RTN), id='fp64_rtn'),
    pytest.param(fp.SINT32, id='sint32'),
    pytest.param(fp.UINT16, id='uint16'),
]


def _compile(target, *, cast: bool):
    """``round``/``cast`` into *target* from an `FP64` argument."""
    if cast:
        @fp.fpy(ctx=fp.REAL)
        def q(x: fp.Real) -> fp.Real:
            with target:
                y = fp.cast(x)
            return y
    else:
        @fp.fpy(ctx=fp.REAL)
        def q(x: fp.Real) -> fp.Real:
            with target:
                y = fp.round(x)
            return y

    return CppCompiler().compile(q, arg_types=[RealType(fp.FP64)])


@pytest.mark.parametrize('cast', [False, True], ids=['round', 'cast'])
class TestRefusedContexts:
    @pytest.mark.parametrize('target', _REFUSED)
    def test_a_context_a_cast_cannot_perform_is_refused(self, target, cast):
        with pytest.raises(CppCompileError, match=_REFUSAL):
            _compile(target, cast=cast)

    @pytest.mark.parametrize('target', _ALLOWED)
    def test_a_context_a_cast_does_perform_still_compiles(self, target, cast):
        assert _compile(target, cast=cast)


class TestFormatEqualityIsNotEnough:
    def test_fp16_no_longer_answers_in_fp32(self):
        """``static_cast<float>`` left these four unchanged where `FP16` rounds
        them; refusing is the fix, since the backend cannot round to `FP16`."""
        for x in (1 + 2 ** -11, 1 + 2 ** -12, 1 + 3 * 2 ** -12, 1024.5):
            # the interpreter moves each one; a `float` cast would not
            assert float(fp.FP16.round(x)) != x

        with pytest.raises(CppCompileError, match=_REFUSAL):
            _compile(fp.FP16, cast=False)

    def test_saturation_is_refused_despite_matching_fp32s_format(self):
        """Why the guard compares contexts and not formats: this context is
        format-equal to `FP32`, so a format test would let it through."""
        assert _SATURATING.format() == fp.FP32.format()
        assert float(_SATURATING.round(1e300)) < float('inf')  # a cast gives inf
        assert not is_native_ctx(_SATURATING)

        with pytest.raises(CppCompileError, match=_REFUSAL):
            _compile(_SATURATING, cast=False)


class TestTheSupportedPathSurvives:
    def test_the_lowered_fp16_rounding_still_compiles(self):
        """The guard must refuse only what was wrong.  `FP16` is reachable
        through the lowering pipeline, which replaces the rounding before the
        emitter sees it -- that path is what `test_lowered_roundtrip.py` checks
        bit-for-bit, and it must keep compiling."""
        @fp.fpy(ctx=fp.REAL)
        def q(x: fp.Real) -> fp.Real:
            with fp.FP16:
                y = fp.round(x)
            return y

        ref = st.monomorphize(q, args=[RealType(fp.FP32)])
        low = st.rescale_fixed(st.float_to_fixed(
            st.unfold_overflow(ref, early_check=True)))
        assert CppCompiler().compile(low)

    def test_a_fixed_point_context_is_exempt(self):
        """`_validate_context_rm` has already checked that a libm call or an
        integer cast reproduces a fixed-point rounding, so the guard defers to
        it rather than second-guessing."""
        ctx = fp.MPBFixedContext(
            -1, fp.RealFloat(exp=11, c=1), rm=fp.RM.RTZ,
            overflow=fp.OverflowMode.ASSERT)
        assert not is_native_ctx(ctx)
        assert 'std::trunc' in _compile(ctx, cast=False)


class TestIsNativeCtx:
    def test_it_agrees_with_what_the_op_table_dispatches(self):
        """Reading the same list the op table is built from makes the two agree
        by construction; this checks that they *do*, via the table itself.

        ``Add`` stands in for the table: a context it has a signature for is one
        a cast can perform, and one it does not is not.
        """
        from fpy2.ast import Add
        from fpy2.backend.cpp.target import make_op_table

        sigs = make_op_table().binary[Add]
        dispatched = {s.out_ctx for s in sigs}
        for ctx in dispatched:
            assert is_native_ctx(ctx), f'dispatched but not native: {ctx}'
        # and the converse, on a context deliberately absent from the table
        assert not any(_SATURATING == s.out_ctx for s in sigs)
        assert not is_native_ctx(_SATURATING)
