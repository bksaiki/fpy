"""
This module defines sampling methods.
"""

import random

from titanfp.arithmetic.evalctx import EvalCtx, determine_ctx
from titanfp.arithmetic import ieee754

from .function import Function
from ..ir import *

def _sample_value(ctx: EvalCtx):
    match ctx:
        case ieee754.IEEECtx():
            x = random.randint(0, 2 ** ctx.nbits - 1)
            return ieee754.bits_to_digital(x, ctx)
        case _:
            raise NotImplementedError(ctx)

def _sample_one(args: list[Argument], ctx: EvalCtx) -> list[ieee754.Float]:
    if len(args) == 0:
        return []

    # check that args are all Real values
    for arg in args:
        if not isinstance(arg.ty, AnyType | RealType):
            raise ValueError(f"expected Real, got {arg.ty}")

    # sample point for each argument
    pt = [_sample_value(ctx) for _ in args]

    # TODO: reject points that are not in the domain of the function
    return pt



def sample_function(
    fun: Function,
    num_samples: int,
    *,
    only_real: bool = False,
    ignore_pre: bool = False  
):
    default_ctx = ieee754.ieee_ctx(11, 64)
    ctx = determine_ctx(default_ctx, fun.ir.ctx)

    pts: list[list[ieee754.Float]] = []
    while len(pts) < num_samples:
        pt = _sample_one(fun.args, ctx)
        if only_real and any(map(lambda x: x.is_nar(), pt)):
            continue
        pts.append(pt)

    return pts
