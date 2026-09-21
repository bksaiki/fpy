"""
`fused_sum` across the four schedules that lower it.

Two knobs with nothing to do with the program decide what comes out of it:
whether `fuse` ran before `comp_to_loop`, and whether the function was
monomorphized.  Each costs a fact an analysis reads in only one spelling --
`fuse` decides whether the `isfinite` guard arrives as a fold or as a
materialised mask, and `monomorphize` whether the scaling loop's trip count is
`len(xs)` or a literal.  Neither costs anything now -- `trip_count` reads both
trip counts and `_implied_mask` both guards -- and every cell of the grid gives
the same answer.

This module pins all four, so a change that costs one of the facts again has to
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

ZERO, FINITE = ValueClass.ZERO, ValueClass.FINITE


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

_CELLS = [(False, False), (False, True), (True, False), (True, True)]


@pytest.mark.parametrize('program', _PROGRAMS)
@pytest.mark.parametrize('fuse, mono', _CELLS)
class TestGrid:
    """What `HoistScale` makes of each schedule: one site, in every cell.  The
    two refusals left are the program's other two reductions, which is a
    separate question.  The clamp does not enter into it -- both programs
    refuse in the same places, for the same reasons."""

    def test_it_hoists_the_scale(self, program, fuse, mono):
        out = _sched(_PROGRAMS[program], fuse=fuse, mono=mono)
        site, = HoistScale.sites(out.ast)
        assert site.resolve().format() == 'sum(ts)'

    def test_the_other_reductions_are_refused(self, program, fuse, mono):
        out = _sched(_PROGRAMS[program], fuse=fuse, mono=mono)
        assert _why(out) == [_NO_WRITE, _NO_WRITE]


@pytest.mark.parametrize('fuse', [False, True])
class TestFiniteness:
    """What the guard is worth, on the monomorphized schedules -- the only ones
    where a storage question has an answer.  Both spellings of it answer alike,
    so what is left to see is the clamp."""

    def test_the_clamped_exponent_is_an_integer(self, fuse):
        out = _sched(fused_sum_clamped, fuse=fuse, mono=True)
        fmt, cls = _exponent(out)
        assert cls == ZERO | FINITE
        assert choose_storage(fmt, cls) is CppScalar.S8
        # and so the emitter's `std::isfinite` assertion goes
        assert _rounded(out) == ZERO | FINITE

    def test_without_the_clamp_logb_of_zero_survives(self, fuse):
        """`logb(0)` is `-inf` whatever the elements are, and no integer
        storage holds one -- so the clamp is load-bearing for the exponent's
        type, guard or no guard."""
        out = _sched(fused_sum, fuse=fuse, mono=True)
        fmt, cls = _exponent(out)
        assert cls == ValueClass.NEG_INF | ZERO | FINITE
        assert choose_storage(fmt, cls) is CppScalar.F32
