import random
import unittest

from fpy2 import IEEEContext, RM
from fpy2.math import *

_unary_ops = [
    acos,
    acosh,
    asin,
    atan,
    atanh,
    cbrt,
    cos,
    cosh,
    erf,
    erfc,
    exp,
    exp2,
    exp10,
    expm1,
    fabs,
    lgamma,
    log,
    log10,
    log1p,
    log2,
    neg,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    tgamma,
    ceil,
    floor,
    trunc,
    nearbyint,
    round
]

_binary_ops = [
    add,
    atan2,
    copysign,
    div,
    fdim,
    fmax,
    fmin,
    fmod,
    hypot,
    mul,
    pow,
    remainder
]

_ternary_ops = [
    fma
]

_rms = [
    RM.RNE,
    RM.RNA,
    RM.RTP,
    RM.RTN,
    RM.RTZ,
    RM.RAZ
]

_ctxs = [
    IEEEContext(11, 64, RM.RNE),
    IEEEContext(8, 32, RM.RNE),
    IEEEContext(5, 16, RM.RNE),
    IEEEContext(5, 8, RM.RNE)
]

class MathNoExceptTestCase(unittest.TestCase):
    """
    Fuzz testing for each operation under `fpy2.math`.

    Ensures that the operations in `fpy2.math` don't
    throw an exception for randomly sampled inputs.
    """

    def test_fuzz_unary(self, num_inputs: int = 256):
        for op in _unary_ops:
            for ctx_base in _ctxs:
                for rm in _rms:
                    ctx = ctx_base.with_rm(rm)
                    for _ in range(num_inputs):
                        # sample point
                        i = random.randint(0, 1 << ctx.nbits - 1)
                        x = ctx.decode(i)
                        # evaluate
                        op(x, ctx)


    def test_fuzz_binary(self, num_inputs: int = 256):
        for op in _binary_ops:
            for ctx_base in _ctxs:
                for rm in _rms:
                    ctx = ctx_base.with_rm(rm)
                    for _ in range(num_inputs):
                        # sample point
                        i = random.randint(0, 1 << ctx.nbits - 1)
                        j = random.randint(0, 1 << ctx.nbits - 1)
                        x = ctx.decode(i)
                        y = ctx.decode(j)
                        # evaluate
                        op(x, y, ctx)

    def test_fuzz_ternary(self, num_inputs: int = 256):
        for op in _ternary_ops:
            for ctx_base in _ctxs:
                for rm in _rms:
                    ctx = ctx_base.with_rm(rm)
                    for _ in range(num_inputs):
                        # sample point
                        i = random.randint(0, 1 << ctx.nbits - 1)
                        j = random.randint(0, 1 << ctx.nbits - 1)
                        k = random.randint(0, 1 << ctx.nbits - 1)
                        x = ctx.decode(i)
                        y = ctx.decode(j)
                        z = ctx.decode(k)
                        # evaluate
                        op(x, y, z, ctx)
