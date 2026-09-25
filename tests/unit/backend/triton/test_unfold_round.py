"""`unfold_round`: which roundings Triton cannot spell, and their lowering."""

import pytest

import fpy2 as fp
from fpy2.backend.triton import TritonCompiler, TritonEmitError
from fpy2.backend.triton.unfold_round import UnfoldKind, UnfoldMode, sites, unfold
from fpy2.number import MPFixedContext, RealFloat
from fpy2.transform import Monomorphize
from fpy2.types import ListType, RealType
from fpy2.utils import NamedId

RZ_FP16 = fp.IEEEContext(5, 16, fp.RM.RTZ)
BOUNDED = fp.MPBFixedContext(-1, RealFloat.from_int(2 ** 17), fp.RM.RTZ, overflow=fp.OverflowMode.ASSERT)


@fp.fpy
def rz_fp16(x: fp.Real):
    with RZ_FP16:
        return fp.round(x)


@fp.fpy
def bounded(x: fp.Real):
    with BOUNDED:
        return fp.round(x)


def _mono(fn, ctx):
    return Monomorphize.apply(fn.ast, fp.REAL, [RealType(ctx)])


def _kinds(func):
    return [(s.kind, s.ctx) for s in sites(func)]


def test_classify():
    assert _kinds(_mono(rz_fp16, fp.FP64)) == [(UnfoldKind.FLOAT_ROUND, RZ_FP16)]
    # a bounded context is never lowered as it stands: its check cannot raise
    assert _kinds(_mono(bounded, fp.FP16)) == [(UnfoldKind.FIXED_ROUND, BOUNDED)]


@fp.fpy
def rz_e8m13(x: fp.Real):
    with fp.IEEEContext(8, 22, fp.RM.RTZ):
        return fp.round(x)


def test_a_directed_round_with_f32_exponents_is_no_site():
    """A cast spells it from any source, so it is not lowered."""
    assert _kinds(_mono(rz_e8m13, fp.FP64)) == []


def test_unfold_float_round_leaves_no_site():
    """The ladder's bounds are proven, so none is left for the emitter."""
    out = unfold(_mono(rz_fp16, fp.FP64), UnfoldMode.ROUNDINGS)
    assert sites(out) == []


def test_a_proven_bound_is_dropped():
    # every FP16 value is below 2 ** 17
    out = unfold(_mono(bounded, fp.FP16), UnfoldMode.ROUNDINGS)
    assert sites(out) == []
    assert 'MPBFixedContext' not in out.format()
    assert repr(MPFixedContext(-1, fp.RM.RTZ)) in out.format()


def test_an_unproven_bound_is_kept():
    # an FP64 value is not
    out = unfold(_mono(bounded, fp.FP64), UnfoldMode.ROUNDINGS)
    assert _kinds(out) == [(UnfoldKind.FIXED_ROUND, BOUNDED)]


@fp.fpy(ctx=fp.FP32)
def _wrap8(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
    for i in range(len(xs)):
        with fp.SINT8:
            y = fp.round(xs[i])
        out[i] = y
    return out


def _compile_wrap8(src: fp.Context):
    n = NamedId('n')
    return TritonCompiler(drop_asserts=True).compile(
        _wrap8, ctx=fp.FP32, arg_types=[
            ListType(RealType(src), n), ListType(RealType(fp.SINT8), n),
            RealType(fp.INTEGER)])


def test_a_wrapping_round_from_an_integer_is_a_cast():
    assert '.to(tl.int8)' in _compile_wrap8(fp.SINT16).source


def test_a_float_into_sint8_is_refused():
    with pytest.raises(TritonEmitError, match='saturates where the context wraps'):
        _compile_wrap8(fp.FP32)


@pytest.mark.parametrize('mode', [UnfoldMode.NONE, UnfoldMode.ROUNDINGS])
def test_a_wrapping_round_from_a_float_is_refused(mode):
    @fp.fpy(ctx=fp.FP32)
    def k(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
        for i in range(len(xs)):
            with fp.FixedContext(True, -3, 8, fp.RM.RTZ, fp.OV.WRAP):
                y = fp.round(xs[i])
            out[i] = y
        return out

    with pytest.raises(TritonEmitError, match='saturates where the context wraps'):
        TritonCompiler(drop_asserts=True, unfold=mode).compile(
            k, ctx=fp.FP32, arg_types=[
                ListType(RealType(fp.FP32), NamedId('n'))] * 2 + [RealType(fp.INTEGER)])
