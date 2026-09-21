"""
`fused_sum` across the four schedules that lower it.

Two knobs with nothing to do with the program decide what comes out of it:
whether `fuse` ran before `comp_to_loop`, and whether the function was
monomorphized.  Each costs a fact an analysis reads in only one spelling --
`fuse` decides whether the `isfinite` guard arrives as a fold or as a
materialised mask, and `monomorphize` whether the scaling loop's trip count is
`len(xs)` or a literal.  The second no longer costs anything -- `trip_count`
reads both spellings -- and the `mono` column is pinned so that it stays that
way.  The first still does: without `fuse` the scale factor is not known finite.

This module pins all four cells, so a phase that closes one of the gaps has to
say which cell it changed.  See `docs/todos/finiteness-refinement.md`.
"""

import pytest

import fpy2 as fp
import fpy2.strategies as st

from fpy2.analysis import FormatInfer, ValueClass, ValueClassInfer
from fpy2.ast import Round
from fpy2.backend.cpp.storage import CppScalar, choose_storage
from fpy2.transform import HoistScale, walk_exprs
from fpy2.types import ListType, RealType

from .test_hoist_invariant import fused_sum

K = 32
EMIN = fp.FP16.emin
_ARGS = [ListType(RealType(fp.FP16), K)]

TOP = ValueClass.TOP
FINITE = ValueClass.ZERO | ValueClass.FINITE


@fp.fpy(ctx=fp.REAL)
def fused_sum_clamped(xs: list[fp.Real]) -> fp.Real:
    """`fused_sum` as `digit-bound`'s `rescale_fixed` leaves it: the exponent is
    clamped at `emin`, which is what rules `logb(0)`'s `-inf` out of `e`."""
    if all([fp.isfinite(x) for x in xs]):
        e = max([max(fp.logb(x), EMIN) for x in xs])
        with fp.MPFixedContext(e - 12, rm=fp.RM.RTZ, enable_neg_zero=False):
            ts = [fp.round(x) for x in xs]
        return sum(ts)
    else:
        with fp.FP32:
            return sum(xs)


_PROGRAMS = {'plain': fused_sum, 'clamped': fused_sum_clamped}


def _sched(func, *, fuse: bool, mono: bool):
    """The motivating schedule, with the two knobs."""
    if mono:
        func = st.monomorphize(func, args=_ARGS)
    if fuse:
        func = st.fuse(func)
    func = st.simplify(st.rescale_fixed(st.comp_to_loop(func)))
    return st.hoist_invariant(func)


def _why(func) -> list[str]:
    return [why for _, why in HoistScale.refusals(func.ast)]


def _exponent(func):
    """The format and class of `e`, the exponent the scale is built from."""
    fmts = FormatInfer.analyze(func.ast).by_def
    classes = ValueClassInfer.analyze(func.ast).by_def
    d = next(d for d in fmts if str(getattr(d, 'name', '')) == 'e')
    return fmts[d], classes[d]


def _rounded(func) -> ValueClass:
    """The class of what the rescaled round is handed -- the operand of the
    `std::isfinite` assertion the emitter writes when it is not known finite."""
    info = ValueClassInfer.analyze(func.ast)
    rounds = [e for _, e in walk_exprs(func.ast) if isinstance(e, Round)]
    assert len(rounds) == 1
    return info.classify(rounds[0].arg)


# ----------------------------------------------------------------------
# The grid

_NO_WRITE = 'no scaled list write fills the reduction'
_NOT_FINITE = 'the factor may be an infinity or a NaN'

_GRID = {
    # (fuse, mono): sites, every refusal in visit order.  The two `_NO_WRITE`s
    # are the program's other reductions and are not what this page is about.
    (False, False): (0, [_NO_WRITE, _NOT_FINITE, _NO_WRITE]),
    (False, True): (0, [_NO_WRITE, _NOT_FINITE, _NO_WRITE]),
    (True, False): (1, [_NO_WRITE, _NO_WRITE]),
    (True, True): (1, [_NO_WRITE, _NO_WRITE]),
}


@pytest.mark.parametrize('program', _PROGRAMS)
@pytest.mark.parametrize('fuse, mono', _GRID)
class TestGrid:
    """What `HoistScale` makes of each schedule.  The clamp does not enter into
    it: both programs refuse in the same places, for the same reasons."""

    def test_sites(self, program, fuse, mono):
        out = _sched(_PROGRAMS[program], fuse=fuse, mono=mono)
        assert len(HoistScale.sites(out.ast)) == _GRID[(fuse, mono)][0]

    def test_refusals(self, program, fuse, mono):
        out = _sched(_PROGRAMS[program], fuse=fuse, mono=mono)
        assert _why(out) == _GRID[(fuse, mono)][1]


class TestFiniteness:
    """What the guard is worth, on the monomorphized schedules -- the only ones
    where a storage question has an answer."""

    def test_the_mask_refines_nothing(self):
        """Unfused, the guard is a materialised mask and no element is known
        finite, so nothing downstream of it is either."""
        for program in _PROGRAMS.values():
            out = _sched(program, fuse=False, mono=True)
            fmt, cls = _exponent(out)
            assert ValueClass.INF & cls
            assert choose_storage(fmt, cls) is CppScalar.F32
            assert _rounded(out) is TOP

    def test_the_fold_refines_the_elements(self):
        """Fused, the guard is a fold the analysis reads, and the clamp is what
        turns that into an integer exponent: without it `logb(0)`'s `-inf`
        survives the max and no integer storage holds `e`."""
        plain = _sched(fused_sum, fuse=True, mono=True)
        fmt, cls = _exponent(plain)
        assert cls == ValueClass.NEG_INF | FINITE
        assert choose_storage(fmt, cls) is CppScalar.F32

        clamped = _sched(fused_sum_clamped, fuse=True, mono=True)
        fmt, cls = _exponent(clamped)
        assert cls == FINITE
        assert choose_storage(fmt, cls) is CppScalar.S8
        assert _rounded(clamped) == FINITE
