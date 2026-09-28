"""The contract every pass that hoists owes evaluation order.

A pass that inserts statements before the one an expression sits in must hoist
only a strict expression -- not out of an ``and``/``or`` tail, a comparison's
tail, or an ``assert`` message -- and must name what its statement evaluates
before it (see :mod:`fpy2.analysis.hoistability`).  Each row runs a pass on a
program that breaks if either is dropped, and checks the outcome -- the value,
or which assertion fires -- is unchanged.
"""

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.ast import FuncDef
from fpy2.backend.cpp import CppCompiler
from fpy2.number import OverflowMode
from fpy2.transform import (
    FloatToFixed,
    Monomorphize,
    RescaleFixed,
    RoundElim,
    RoundInsert,
    SplitRound,
    StatementForm,
    UnfoldEnumerate,
    UnfoldNegZero,
    UnfoldOverflow,
    UnfoldSpecial,
    UnfoldZip,
)
from fpy2.types import ListType, RealType


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


@fp.fpy
def _wrap(xs: list[fp.Real]) -> list[fp.Real]:
    return [xs[0]]


@fp.fpy
def _nonempty(xs: list[fp.Real]) -> list[fp.Real]:
    assert len(xs) > 0, 'nonempty'
    return xs


def _outcome(ast: FuncDef, runtime: Any, args: tuple) -> str:
    try:
        return repr(Function(ast, runtime=runtime)(*args))
    except Exception as e:  # noqa: BLE001 -- the exception is the outcome
        return f'{type(e).__name__}({e})'


def _arithmetic(S: fp.Context) -> list[tuple[Function, tuple]]:
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


def _rounding(C: fp.Context) -> list[tuple[Function, tuple]]:
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


def _guarded(C: fp.Context, op: Callable[..., Any]) -> Function:
    """A site whose own statements fault where the guard held it back."""
    @fp.fpy(ctx=fp.REAL)
    def guarded(x):
        with C:
            return x == 0.5 and op(x) > 0
    return guarded


# derived iterables: their unfolding binds arguments and asserts lengths
@fp.fpy(ctx=fp.FP64)
def _zip_and_tail(xs: list[fp.Real], ys: list[fp.Real]) -> bool:
    return len(xs) == len(ys) and len(zip(xs, ys)) > 0


@fp.fpy(ctx=fp.FP64)
def _zip_comparison_tail(xs: list[fp.Real], ys: list[fp.Real]) -> bool:
    return len(xs) == len(ys) == len(zip(xs, ys))


@fp.fpy(ctx=fp.FP64)
def _zip_message(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    assert len(xs) > 0, len(zip(xs, ys))
    return 0


@fp.fpy(ctx=fp.FP64)
def _zip_order_through_the_heap(xs: list[fp.Real], ys: list[fp.Real]) -> bool:
    return _bump(xs) < fp.fst(zip(_wrap(xs), ys)[0])


@fp.fpy(ctx=fp.FP64)
def _enumerate_and_tail(xs: list[fp.Real], ys: list[fp.Real]) -> bool:
    return len(xs) > 0 and len(enumerate(_nonempty(xs))) > 0


@fp.fpy(ctx=fp.FP64)
def _enumerate_comparison_tail(xs: list[fp.Real], ys: list[fp.Real]) -> bool:
    return 0 < len(xs) < len(enumerate(_nonempty(xs))) + 1


@fp.fpy(ctx=fp.FP64)
def _enumerate_message(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    assert len(ys) > 0, len(enumerate(_nonempty(xs)))
    return 0


@fp.fpy(ctx=fp.FP64)
def _enumerate_order_through_the_heap(xs: list[fp.Real], ys: list[fp.Real]) -> bool:
    return _bump(xs) < fp.snd(enumerate(_wrap(xs))[0])


# the cpp pipeline: `StatementForm`, `RoundElim`, then the unfold ladder
_Q = fp.FixedContext(True, -8, 16)


@fp.fpy(ctx=fp.FP64)
def _cpp_comparison_tail(x: fp.Real, y: fp.Real) -> bool:
    return 0 < x < _pos(x) * y


@fp.fpy(ctx=fp.FP64)
def _cpp_message(x: fp.Real, y: fp.Real) -> fp.Real:
    assert x > 0, _neg(x) * y
    return x


@fp.fpy(ctx=fp.FP64)
def _cpp_order_through_the_heap(x: fp.Real, y: fp.Real) -> bool:
    xs = [x]
    return _bump(xs) < _get(xs) * y


@fp.fpy(ctx=fp.FP64)
def _cpp_ladder_comparison_tail(x: fp.Real, y: fp.Real) -> bool:
    with _Q:
        return 0 < x < fp.round(_pos(x))


@fp.fpy(ctx=fp.FP64)
def _cpp_ladder_order_through_the_heap(x: fp.Real, y: fp.Real) -> bool:
    xs = [x]
    with _Q:
        t = _bump(xs) < fp.round(_get(xs))
    return t


_F32 = RealType(fp.FP32)
_L32 = ListType(_F32)


def _specialize(name: str, unfold: str) -> Callable[[FuncDef], FuncDef]:
    """`CppCompiler.specialize` as a pass over one function."""
    def apply(ast: FuncDef) -> FuncDef:
        m = fp.Module()
        m.add(Function(ast), arg_types=[_F32, _F32])
        cc = CppCompiler(unfold=CppCompiler.UnfoldMode[unfold])
        return next(s for s in cc.specialize(m) if s.name.startswith(name)).ast
    return apply

_ITERABLES = [
    (UnfoldZip, [(_zip_and_tail, ([1, 2], [1])), (_zip_comparison_tail, ([1, 2], [1])),
                 (_zip_message, ([1, 2], [1])), (_zip_order_through_the_heap, ([1], [1]))]),
    (UnfoldEnumerate, [(_enumerate_and_tail, ([], [1])), (_enumerate_comparison_tail, ([], [1])),
                       (_enumerate_message, ([], [1])), (_enumerate_order_through_the_heap, ([1], [1]))]),
]

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
    *[(f'{P.__name__}-{f.name}', f, args, P.apply, True)
      for P, cases in _ITERABLES for f, args in cases],
    *[(f'StatementForm-{f.name}', f, args, StatementForm.apply, True)
      for _, cases in _ITERABLES for f, args in cases],
    *[(f'round_elim-{f.name}', f, args, RoundElim.apply, True)
      for f, args in _arithmetic(fp.FP64)],
    *[(f'cpp-{unfold}-{f.name}', f, (-1, 2) if 'tail' in f.name else (1, 2),
       _specialize(f.name, unfold), False)
      for f, unfold in [(_cpp_comparison_tail, 'NONE'), (_cpp_message, 'NONE'),
                        (_cpp_order_through_the_heap, 'NONE'),
                        (_cpp_ladder_comparison_tail, 'ROUNDINGS'),
                        (_cpp_ladder_order_through_the_heap, 'ROUNDINGS')]],
]


@pytest.mark.parametrize('f,args,apply,mono', [r[1:] for r in _ROWS], ids=[r[0] for r in _ROWS])
def test_the_outcome_is_unchanged(
    f: Function, args: tuple, apply: Callable[[FuncDef], FuncDef], mono: bool,
) -> None:
    types = [_L32 if isinstance(a, list) else _F32 for a in args]
    ast = Monomorphize.apply(f.ast, f.ast.ctx or fp.REAL, types) if mono else f.ast
    assert _outcome(apply(ast), f.runtime, args) == _outcome(ast, f.runtime, args)
