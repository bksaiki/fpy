"""
This module defines sampling methods.
"""

import random

from titanfp.arithmetic.evalctx import EvalCtx, determine_ctx
from titanfp.arithmetic import ieee754

from .table import RangeTable
from ..function import Function
from ...ir import *

def _float_to_ordinal(x: ieee754.Float):
    pos_ord = ieee754.digital_to_bits(x.fabs())
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

def _sample_any(fun: Function, ctx: EvalCtx):
    if len(fun.args) == 0:
        return []

    pt: list[ieee754.Float] = []
    for _ in fun.args:
        bits = random.randint(0, 2 ** ctx.nbits - 1)
        x = ieee754.bits_to_digital(bits, ctx=ctx)
        pt.append(x)

    return pt

def _sample_real(fun: Function, ctx: EvalCtx):
    if len(fun.args) == 0:
        return []

    lo = ieee754.Float(negative=True, isinf=True, ctx=ctx)
    hi = ieee754.Float(isnan=True, ctx=ctx)
    return [_sample_between(lo, hi, ctx=ctx) for _ in fun.args]


def _sample_infallable(
    fun: Function,
    num_samples: int,
    ctx: ieee754.IEEECtx,
    only_real: bool,
):
    if only_real:
        return [_sample_real(fun, ctx) for _ in range(num_samples)]
    else:
        return [_sample_any(fun, ctx) for _ in range(num_samples)]

def _sample_hyperrect(
    fun: Function,
    num_samples: int,
    ctx: ieee754.IEEECtx,
    only_real: bool
):
    raise NotImplementedError

def _sample_default(
    fun: Function,
    num_samples: int,
    ctx: ieee754.IEEECtx,
    only_real: bool,
    ignore_pre: bool,
):
    return _sample_infallable(fun, num_samples, ctx, only_real)

def sample_function(
    fun: Function,
    num_samples: int,
    *,
    only_real: bool = False,
    ignore_pre: bool = False  
):
    """
    Samples `num_samples` points for the function `fun`.

    Specify `only_real=true` to only sample real values
    (excludes infinity and NaN). Specify `ignore_pre=true`
    to ignore the preconditions of the function.
    """

    # compute the context
    default_ctx = ieee754.ieee_ctx(11, 64)
    ctx = determine_ctx(default_ctx, fun.ir.ctx)

    # TODO: other sampling methods
    if not isinstance(ctx, ieee754.IEEECtx):
        raise ValueError(f"expected IEEE context, got {ctx}")

    # TODO: extend to other types
    for arg in fun.args:
        if not isinstance(arg.ty, AnyType | RealType):
            raise ValueError(f"expected Real, got {arg.ty}")

    # check which sampling method we should try
    if 'pre' not in fun.ir.ctx:
        # no precondition, so we should never reject a sample
        return _sample_infallable(fun, num_samples, ctx, only_real)

    # process precondition
    pre: FunctionDef = fun.ir.ctx['pre']
    print(pre)
    table = RangeTable.from_precondition(pre)

    # fallback to the default method
    return _sample_default(fun, num_samples, ctx, only_real, ignore_pre)
