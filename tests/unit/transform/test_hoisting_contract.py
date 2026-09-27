"""The contract every pass that hoists owes evaluation order.

A pass that inserts statements before the one an expression sits in must hoist
only a strict expression -- not out of an ``and``/``or`` tail, a comparison's
tail, or an ``assert`` message -- and must name what its statement evaluates
before it (see :mod:`fpy2.analysis.hoistability`).  Each row runs a pass on a
program that breaks if either is dropped, and checks the outcome -- the value,
or which assertion fires -- is unchanged.
"""

from fractions import Fraction
from typing import Any

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.number import OverflowMode
from fpy2.transform import (
    FloatToFixed,
    Monomorphize,
    RescaleFixed,
    RoundInsert,
    SplitRound,
    UnfoldNegZero,
    UnfoldOverflow,
    UnfoldSpecial,
)
from fpy2.types import RealType


@fp.fpy
def _pos(x: fp.Real) -> fp.Real:
    assert x > 0, 'pos'
    return x


@fp.fpy
def _neg(x: fp.Real) -> fp.Real:
    assert x < 0, 'neg'
    return x


@fp.fpy
def _bump(xs: list[fp.Real]) -> fp.Real:
    xs[0] = 4
    return 3


@fp.fpy
def _get(xs: list[fp.Real]) -> fp.Real:
    return xs[0]


def _outcome(ast, runtime, args):
    try:
        return repr(Function(ast, runtime=runtime)(*args))
    except Exception as e:  # noqa: BLE001 -- the exception is the outcome
        return f'{type(e).__name__}({e})'


def _arithmetic(S):
    """Programs whose site is an operation, run under scope `S`."""
    @fp.fpy(ctx=fp.REAL)
    def and_tail(x, y):
        with S:
            return x > 0 and _pos(x) * y > 0

    @fp.fpy(ctx=fp.REAL)
    def comparison_tail(x, y):
        with S:
            return 0 < x < _pos(x) * y

    @fp.fpy(ctx=fp.REAL)
    def message(x, y):
        with S:
            assert x > 0, _neg(x) * y
        return x

    @fp.fpy(ctx=fp.REAL)
    def order(x, y):
        with S:
            t = _neg(x) < _pos(x) * y
        return t

    @fp.fpy(ctx=fp.REAL)
    def order_through_the_heap(x, y):
        xs = [x]
        with S:
            t = _bump(xs) < _get(xs) * y
        return t

    return [(and_tail, (-1, 2)), (comparison_tail, (-1, 2)), (message, (1, 2)),
            (order, (0, 2)), (order_through_the_heap, (1, 2))]


def _rounding(C):
    """Programs whose site is a rounding under `C`."""
    @fp.fpy(ctx=fp.REAL)
    def and_tail(x):
        with C:
            return x > 0 and fp.round(_pos(x)) > 0

    @fp.fpy(ctx=fp.REAL)
    def comparison_tail(x):
        with C:
            return 0 < x < fp.round(_pos(x))

    @fp.fpy(ctx=fp.REAL)
    def message(x):
        with C:
            assert x > 0, fp.round(_neg(x))
        return x

    @fp.fpy(ctx=fp.REAL)
    def order(x):
        with C:
            t = _neg(x) < fp.round(_pos(x))
        return t

    @fp.fpy(ctx=fp.REAL)
    def order_through_the_heap(x):
        xs = [x]
        with C:
            t = _bump(xs) < fp.round(_get(xs))
        return t

    return [(and_tail, (-1,)), (comparison_tail, (-1,)), (message, (1,)),
            (order, (0,)), (order_through_the_heap, (1,))]


def _guarded(C, op):
    """A site whose own statements fault where the guard held it back."""
    @fp.fpy(ctx=fp.REAL)
    def guarded(x):
        with C:
            return x == 0.5 and op(x) > 0
    return guarded


_F32 = RealType(fp.FP32)

_SCOPED: list[tuple[Any, fp.Context]] = [
    (UnfoldSpecial, fp.MPFixedContext(-8, enable_nan=True, enable_inf=True)),
    (UnfoldNegZero, fp.MPFixedContext(-8)),
    (FloatToFixed, fp.FP16),
    (RescaleFixed, fp.FixedContext(True, -16, 32)),
    (UnfoldOverflow, fp.FP16),
]

_GUARDED: list[tuple[Any, fp.Context, Any, Any]] = [
    (UnfoldSpecial, fp.MPFixedContext(-8, enable_nan=True, enable_inf=True), fp.cast, Fraction(1, 3)),
    (UnfoldSpecial, fp.IEEEContext(5, 16, overflow=OverflowMode.ASSERT), fp.round, 1e10),
    (UnfoldNegZero, fp.MPFixedContext(-8), fp.round, float('nan')),
    (UnfoldNegZero, fp.MPBFixedContext(-8, fp.RealFloat.from_int(100), overflow=OverflowMode.ASSERT),
     fp.round, 1e10),
    (RescaleFixed, fp.FixedContext(True, -16, 32), fp.cast, Fraction(1, 3)),
    (RescaleFixed, fp.FixedContext(True, -16, 32, overflow=OverflowMode.ASSERT), fp.round, 1e10),
    (RescaleFixed, fp.FixedContext(True, -16, 32), fp.round, float('nan')),
]

_ROWS = [
    *[(f'round_insert-{f.name}', f, args, lambda a: RoundInsert.apply(a, fp.FP64), True)
      for f, args in _arithmetic(fp.REAL)],
    *[(f'split_round-{f.name}', f, args, lambda a: SplitRound.apply(a, fp.FP64), True)
      for f, args in _arithmetic(fp.FP32)],
    *[(f'{P.__name__}-{f.name}', f, args, P.apply, False)
      for P, C in _SCOPED for f, args in _rounding(C)],
    *[(f'{P.__name__}-guarded-{op.__name__}-{v}', _guarded(C, op), (v,), P.apply, False)
      for P, C, op, v in _GUARDED],
]


@pytest.mark.parametrize('f,args,apply,mono', [r[1:] for r in _ROWS], ids=[r[0] for r in _ROWS])
def test_the_outcome_is_unchanged(f, args, apply, mono):
    ast = Monomorphize.apply(f.ast, fp.REAL, [_F32] * len(args)) if mono else f.ast
    assert _outcome(apply(ast), f.runtime, args) == _outcome(ast, f.runtime, args)
