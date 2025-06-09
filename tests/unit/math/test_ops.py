import random
import unittest

from fpy2 import IEEEContext, MPFixedContext, FixedContext, RM, OF, FP64, FP32, FP16
from fpy2.math import *

_unary_ops = [
    acos,
    acosh,
    asin,
    asinh,
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
    FP64,
    FP32,
    FP16,
    IEEEContext(5, 8, RM.RNE)
]

class MathIEEENoExceptTestCase(unittest.TestCase):
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

class MathIntegerNoExceptTestCase(unittest.TestCase):
    """
    Fuzz testing for integer operations under `fpy2.math`.

    Ensures that the integer operations in `fpy2.math`
    don't throw an exception for randomly sampled inputs.
    """

    _default_max_integer = 1 << 16
    _max_integer = {
        exp: 1 << 8,
        exp2: 1 << 8,
        exp10: 1 << 8,
        expm1: 1 << 8,
        pow: 1 << 4,
        cosh: 1 << 8,
        sinh: 1 << 8,
        tgamma: 1 << 4,
    }

    def test_fuzz_unary(self, num_inputs: int = 256):
        for op in _unary_ops:
            max_value = self._max_integer.get(op, self._default_max_integer)
            for rm in _rms:
                ctx = MPFixedContext(-1, rm, enable_nan=True, enable_inf=True)
                for _ in range(num_inputs):
                    # sample point
                    i = random.randint(0, max_value) * random.choice([-1, 1])
                    i *= random.choice([-1, 1])
                    x = Float.from_int(i, ctx)
                    # evaluate
                    op(x, ctx)

    def test_fuzz_binary(self, num_inputs: int = 256):
        for op in _binary_ops:
            max_value = self._max_integer.get(op, self._default_max_integer)
            for rm in _rms:
                ctx = MPFixedContext(-1, rm, enable_nan=True, enable_inf=True)
                for _ in range(num_inputs):
                    # sample point
                    i = random.randint(0, max_value) * random.choice([-1, 1])
                    j = random.randint(0, max_value) * random.choice([-1, 1])
                    x = Float.from_int(i, ctx)
                    y = Float.from_int(j, ctx)
                    # evaluate
                    op(x, y, ctx)

    def test_fuzz_ternary(self, num_inputs: int = 256):
        for op in _ternary_ops:
            max_value = self._max_integer.get(op, self._default_max_integer)
            for rm in _rms:
                ctx = MPFixedContext(-1, rm, enable_nan=True, enable_inf=True)
                for _ in range(num_inputs):
                    # sample point
                    i = random.randint(0, max_value) * random.choice([-1, 1])
                    j = random.randint(0, max_value) * random.choice([-1, 1])
                    k = random.randint(0,max_value) * random.choice([-1, 1])
                    x = Float.from_int(i, ctx)
                    y = Float.from_int(j, ctx)
                    z = Float.from_int(k, ctx)
                    # evaluate
                    op(x, y, z, ctx)

class MathInt64NoExceptTestCase(unittest.TestCase):
    """
    Fuzz testing for integer operations under `fpy2.math`.

    Ensures that the integer operations in `fpy2.math`
    don't throw an exception for randomly sampled inputs.
    """

    _default_max_integer = 1 << 62
    _max_integer = {
        exp: 1 << 8,
        exp2: 1 << 8,
        exp10: 1 << 8,
        expm1: 1 << 8,
        pow: 1 << 4,
        cosh: 1 << 8,
        sinh: 1 << 8,
        tgamma: 1 << 4,
    }

    def test_fuzz_unary(self, num_inputs: int = 256):
        INT64 = FixedContext(True, 0, 64, RM.RTZ, OF.WRAP)
        INT64 = FixedContext(True, 0, 64, RM.RTZ, OF.WRAP, nan_value=INT64.maxval(s=True), inf_value=INT64.maxval(s=True))
        for op in _unary_ops:
            max_value = self._max_integer.get(op, self._default_max_integer)
            for rm in _rms:
                for _ in range(num_inputs):
                    # sample point
                    i = random.randint(0, max_value) * random.choice([-1, 1])
                    i *= random.choice([-1, 1])
                    x = Float.from_int(i, INT64)
                    # evaluate
                    op(x, INT64)

    def test_fuzz_binary(self, num_inputs: int = 256):
        INT64 = FixedContext(True, 0, 64, RM.RTZ, OF.WRAP)
        INT64 = FixedContext(True, 0, 64, RM.RTZ, OF.WRAP, nan_value=INT64.maxval(s=True), inf_value=INT64.maxval(s=True))
        for op in _binary_ops:
            max_value = self._max_integer.get(op, self._default_max_integer)
            for rm in _rms:
                ctx = MPFixedContext(-1, rm, enable_nan=True, enable_inf=True)
                for _ in range(num_inputs):
                    # sample point
                    i = random.randint(0, max_value) * random.choice([-1, 1])
                    j = random.randint(0, max_value) * random.choice([-1, 1])
                    x = Float.from_int(i, ctx)
                    y = Float.from_int(j, ctx)
                    # evaluate
                    op(x, y, ctx)

    def test_fuzz_ternary(self, num_inputs: int = 256):
        INT64 = FixedContext(True, 0, 64, RM.RTZ, OF.WRAP)
        INT64 = FixedContext(True, 0, 64, RM.RTZ, OF.WRAP, nan_value=INT64.maxval(s=True), inf_value=INT64.maxval(s=True))
        for op in _ternary_ops:
            max_value = self._max_integer.get(op, self._default_max_integer)
            for rm in _rms:
                ctx = MPFixedContext(-1, rm, enable_nan=True, enable_inf=True)
                for _ in range(num_inputs):
                    # sample point
                    i = random.randint(0, max_value) * random.choice([-1, 1])
                    j = random.randint(0, max_value) * random.choice([-1, 1])
                    k = random.randint(0,max_value) * random.choice([-1, 1])
                    x = Float.from_int(i, ctx)
                    y = Float.from_int(j, ctx)
                    z = Float.from_int(k, ctx)
                    # evaluate
                    op(x, y, z, ctx)
