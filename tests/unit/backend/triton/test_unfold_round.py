"""`unfold_round`: which roundings Triton cannot spell, and their lowering."""

import fpy2 as fp
from fpy2.backend.triton.unfold_round import UnfoldKind, UnfoldMode, sites, unfold
from fpy2.number import MPFixedContext, RealFloat
from fpy2.transform import Monomorphize
from fpy2.types import RealType

RZ_FP32 = fp.IEEEContext(8, 32, fp.RM.RTZ)
BOUNDED = fp.MPBFixedContext(-1, RealFloat.from_int(2 ** 17), fp.RM.RTZ, overflow=fp.OverflowMode.ASSERT)


@fp.fpy
def rz_fp32(x: fp.Real):
    with RZ_FP32:
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
    assert _kinds(_mono(rz_fp32, fp.FP64)) == [(UnfoldKind.FLOAT_ROUND, RZ_FP32)]
    # a bounded context is never lowered as it stands: its check cannot raise
    assert _kinds(_mono(bounded, fp.FP16)) == [(UnfoldKind.FIXED_ROUND, BOUNDED)]


def test_unfold_float_round_leaves_no_site():
    """The ladder's bounds are proven, so none is left for the emitter."""
    out = unfold(_mono(rz_fp32, fp.FP64), UnfoldMode.ROUNDINGS)
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
