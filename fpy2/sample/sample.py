"""
This module defines sampling methods.
"""

import random

from typing import Optional
from titanfp.arithmetic.evalctx import determine_ctx
from titanfp.arithmetic import ieee754

from ..ast import AnyTypeAnn, RealTypeAnn, FuncDef
from ..env import ForeignEnv
from ..function import Function
from ..interpret import DefaultInterpreter

from .table import RangeTable

_DEFAULT_FUEL = 32

class SamplingFailure(Exception):
    """Raised when sampling fails."""

    def __init__(self, msg: str):
        super().__init__(msg)


def _float_to_ordinal(x: ieee754.Float):
    pos_ord = ieee754.digital_to_bits(x.fabs(), ctx=x.ctx)
    return (-1 if x.negative else 1) * pos_ord

def _ordinal_to_float(x: int, ctx: ieee754.IEEECtx):
    negative = x < 0
    x = ieee754.bits_to_digital(abs(x))
    return ieee754.Float(negative=negative, x=x, ctx=ctx)

def _sample_between(
    lo: ieee754.Float,
    hi: ieee754.Float,
    ctx: ieee754.IEEECtx
):
    lo_ord = _float_to_ordinal(lo)
    hi_ord = _float_to_ordinal(hi)
    x_ord = random.randint(lo_ord, hi_ord)
    return _ordinal_to_float(x_ord, ctx)

def _sample_real(
    fun: Function,
    table: RangeTable,
    ctx: ieee754.IEEECtx,
    only_real: bool
) -> list[ieee754.Float]:
    if len(fun.args) == 0:
        return []
    else:
        pt: list[ieee754.Float] = []
        for arg in fun.args:
            # TODO: open/closed intervals
            range = table[arg.name]
            lo = ieee754.Float(x=range.lo.val, ctx=ctx)
            hi = ieee754.Float(x=range.hi.val, ctx=ctx)

            x = _sample_between(lo, hi, ctx)
            while only_real and x.is_nar():
                x = _sample_between(lo, hi, ctx)
            pt.append(x)

        return pt

def _sample_ranges(
    fun: Function,
    table: RangeTable,
    num_samples: int,
    ctx: ieee754.IEEECtx,
    only_real: bool
):
    return [_sample_real(fun, table, ctx, only_real) for _ in range(num_samples)]

def _sample_rejection_one(
    fun: Function,
    ctx: ieee754.IEEECtx,
    only_real: bool,
    fuel: int
):
    if len(fun.args) == 0:
        return []
    else:
        lo = ieee754.Float(negative=True, isinf=True, ctx=ctx)
        hi = ieee754.Float(negative=False, isinf=True, ctx=ctx)
        rt = DefaultInterpreter()

        assert 'pre' in fun.ast.metadata, 'missing precondition'
        pre = Function(fun.ast.metadata['pre'], ForeignEnv.empty())

        start_fuel = fuel
        while fuel > 0:
            pt = [_sample_between(lo, hi, ctx) for _ in range(len(fun.args))]
            if rt.eval(pre, pt):
                return pt
            fuel -= 1

        # TODO: unsound, we should not return point
        print(f'FAIL: cannot sample after {start_fuel} attempts for {fun.name}, returning point anyways')
        return pt

def _sample_rejection(
    fun: Function,
    num_samples: int,
    ctx: ieee754.IEEECtx,
    only_real: bool,
    fuel: int,
    logging: bool
):
    pts: list[list[ieee754.Float]] = []
    for _ in range(num_samples):
        pt = _sample_rejection_one(fun, ctx, only_real, fuel)
        if logging:
            print('.', end='', flush=True)
        pts.append(pt)

    return pts

def sample_function(
    fun: Function,
    num_samples: int,
    *,
    seed: Optional[int] = None,
    only_real: bool = False,
    ignore_pre: bool = False,
    fuel: int = _DEFAULT_FUEL,
    logging: bool = False
):
    """
    Samples `num_samples` points for the function `fun`.

    Specify `only_real=true` to only sample real values
    (excludes infinity and NaN). Specify `ignore_pre=true`
    to ignore the preconditions of the function.
    """

    # set seed
    if seed is not None:
        random.seed(seed)

    # compute the context
    default_ctx = ieee754.ieee_ctx(11, 64)
    ctx = determine_ctx(default_ctx, fun.ast.metadata)

    # TODO: other sampling methods
    if not isinstance(ctx, ieee754.IEEECtx):
        raise ValueError(f"expected IEEE context, got {ctx}")

    # TODO: extend to other types
    for arg in fun.args:
        match arg.type:
            case AnyTypeAnn():
                pass
            case RealTypeAnn():
                pass
            case _:
                raise ValueError(f"expected Real, got {arg.type}")

    # process precondition
    ast = fun.ast
    if 'pre' not in ast.metadata or ignore_pre:
        table = RangeTable()
    else:
        pre: FuncDef = ast.metadata['pre']
        table = RangeTable.from_condition(pre)

    # add unmentioned variables
    for arg in fun.args:
        if arg.name not in table.table:
            table[arg.name] = RangeTable.default_interval()

    # branch on whether we have a sound table
    if table.sound:
        return _sample_ranges(fun, table, num_samples, ctx, only_real)
    else:
        return _sample_rejection(fun, num_samples, ctx, only_real, fuel, logging)
