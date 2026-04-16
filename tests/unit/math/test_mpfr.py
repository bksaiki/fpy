import gmpy2 as gmp
import random

from dataclasses import dataclass
from typing import Callable

from fpy2 import IEEEContext, Float, RM
from fpy2.ops import *
from fpy2.number.gmputils import float_to_mpfr, mpfr_to_float

def _gmp_neg(x):
    return -x

def _gmp_abs(x):
    return abs(x)

def _gmp_pow(x, y):
    return x ** y

def _gmp_lgamma(x):
    y, _ = gmp.lgamma(x)
    return y

def _gmp_fdim(x, y):
    if gmp.is_nan(x) or gmp.is_nan(y):
        # C reference: if either argument is NaN, NaN is returned
        return gmp.nan()
    elif x > y:
        # if `x > y`, returns `x - y`
        return gmp.sub(x, y)
    else:
        # otherwise, returns +0
        return gmp.zero()

_unary_ops = {
    acos : gmp.acos,
    acosh: gmp.acosh,
    asin: gmp.asin,
    asinh: gmp.asinh,
    atan: gmp.atan,
    atanh : gmp.atanh,
    cbrt : gmp.cbrt,
    cos : gmp.cos,
    cosh : gmp.cosh,
    erf : gmp.erf,
    erfc : gmp.erfc,
    exp : gmp.exp,
    exp2 : gmp.exp2,
    exp10 : gmp.exp10,
    expm1 : gmp.expm1,
    fabs : _gmp_abs,
    lgamma : _gmp_lgamma,
    log : gmp.log,
    log10 : gmp.log10,
    log1p : gmp.log1p,
    log2 : gmp.log2,
    neg : _gmp_neg,
    sin : gmp.sin,
    sinh : gmp.sinh,
    sqrt : gmp.sqrt,
    tan : gmp.tan,
    tanh : gmp.tanh,
    tgamma : gmp.gamma,
    ceil : gmp.ceil,
    floor : gmp.floor,
    trunc : gmp.trunc,
    nearbyint : gmp.rint,
    roundint : gmp.round_away
}

_binary_ops = {
    add : gmp.add,
    atan2 : gmp.atan2,
    copysign : gmp.copy_sign,
    div : gmp.div,
    fdim : _gmp_fdim,
    fmax: gmp.maxnum,
    fmin: gmp.minnum,
    fmod: gmp.fmod,
    hypot: gmp.hypot,
    mul : gmp.mul,
    pow : _gmp_pow,
    remainder : gmp.remainder
}

_ternary_ops = {
    fma : gmp.fma
}

_rms = [
    RM.RNE,
    # RM.RNA,
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

@dataclass(frozen=True)
class MPFRFlags:
    """The IEEE 754 flags as returned by MPFR."""
    invalid: bool
    divzero: bool
    overflow: bool
    underflow: bool
    inexact: bool


def _mpfr_rm(rm: RM):
    match rm:
        case RM.RNE:
            return gmp.RoundToNearest
        case RM.RTP:
            return gmp.RoundUp
        case RM.RTN:
            return gmp.RoundDown
        case RM.RTZ:
            return gmp.RoundToZero
        case RM.RAZ:
            return gmp.RoundAwayZero
        case _:
            raise ValueError(f'Unsupported rounding mode: {rm}')

def _mpfr_apply(
    op: Callable,
    xs: tuple[Float, ...],
    ctx: IEEEContext
) -> tuple[Float, MPFRFlags]:
    with gmp.context(
        precision=ctx.pmax,
        emin=ctx.expmin + 1,
        emax=ctx.emax + 1,
        subnormalize=True,
        round=_mpfr_rm(ctx.rm),
        trap_underflow=False,
        trap_overflow=False,
        trap_inexact=False,
        trap_divzero=False,
    ) as gmp_ctx:
        # compute the numerical result
        y = op(*map(float_to_mpfr, xs))

        # fetch the flags
        invalid: bool = gmp_ctx.invalid
        divzero: bool = gmp_ctx.divzero
        overflow: bool = gmp_ctx.overflow
        underflow: bool = gmp_ctx.underflow
        inexact: bool = gmp_ctx.inexact

        # fixup flags to match IEEE 754 semantics
        if underflow and not inexact:
            # MPFR may set underflow without inexact if the result is subnormal,
            # but IEEE 754 requires that underflow implies inexact.
            underflow = False

        # store the flags
        flags = MPFRFlags(invalid, divzero, overflow, underflow, inexact)

        # return the result with the flags
        return mpfr_to_float(y), flags


class MPFREquivTestCase():
    """
    Fuzz testing under `fpy2.math`.

    Ensures that the operations in `fpy2.math` match those from MPFR.
    """

    def test_fuzz_unary(self, num_inputs: int = 256):
        for op, mpfr in _unary_ops.items():
            for ctx_base in _ctxs:
                for rm in _rms:
                    ctx = ctx_base.with_params(rm=rm)
                    for _ in range(num_inputs):
                        # sample point
                        i = random.randint(0, 1 << ctx.nbits - 1)
                        x = ctx.decode(i)
                        # evaluate operation
                        fl = op(x, ctx=ctx)
                        # evaluate equivalent operation
                        r, flags = _mpfr_apply(mpfr, (x,), ctx)
                        ref = ctx.round(r)
                        # sanity checks
                        assert not ref.inexact, f'r={r}, ref={ref}'
                        assert isinstance(fl, Float)
                        # check that they are the numerical values are the same
                        if fl.isnan:
                            assert fl.isnan, f'op={op}, rm={rm}, x={x}, fl={fl}, ref={ref}'
                        else:
                            assert fl == ref, f'op={op}, rm={rm}, x={x}, fl={fl}, ref={ref}'
                        # check that the flags are the same
                        assert fl.overflow == flags.overflow, f'op={op}, rm={rm}, x={x}, fl={fl}, ref={ref}, flags={flags}'
                        assert fl.inexact == flags.inexact, f'op={op}, rm={rm}, x={x}, fl={fl}, ref={ref}, flags={flags}'

    def test_fuzz_binary(self, num_inputs: int = 256):
        for op, mpfr in _binary_ops.items():
            for ctx_base in _ctxs:
                for rm in _rms:
                    ctx = ctx_base.with_params(rm=rm)
                    for _ in range(num_inputs):
                        # sample point
                        i = random.randint(0, 1 << ctx.nbits - 1)
                        j = random.randint(0, 1 << ctx.nbits - 1)
                        x = ctx.decode(i)
                        y = ctx.decode(j)
                        # evaluate operation
                        fl = op(x, y, ctx=ctx)
                        # evaluate equivalent operation
                        r, flags = _mpfr_apply(mpfr, (x, y), ctx)
                        ref = ctx.round(r)
                        # sanity check: ref should be exact
                        assert not ref.inexact
                        assert isinstance(fl, Float)
                        # check that they are the same
                        if fl.isnan:
                            assert fl.isnan, f'op={op}, rm={rm}, x={x}, y={y}, fl={fl}, ref={ref}'
                        else:
                            assert fl == ref, f'op={op}, rm={rm}, x={x}, y={y}, fl={fl}, ref={ref}'
                        # check that the flags are the same
                        assert fl.overflow == flags.overflow, f'op={op}, rm={rm}, x={x}, y={y}, fl={fl}, ref={ref}, flags={flags}'
                        assert fl.inexact == flags.inexact, f'op={op}, rm={rm}, x={x}, y={y}, fl={fl}, ref={ref}, flags={flags}'

    def test_fuzz_ternary(self, num_inputs: int = 256):
        for op, mpfr in _ternary_ops.items():
            for ctx_base in _ctxs:
                for rm in _rms:
                    ctx = ctx_base.with_params(rm=rm)
                    for _ in range(num_inputs):
                        # sample point
                        i = random.randint(0, 1 << ctx.nbits - 1)
                        j = random.randint(0, 1 << ctx.nbits - 1)
                        k = random.randint(0, 1 << ctx.nbits - 1)
                        x = ctx.decode(i)
                        y = ctx.decode(j)
                        z = ctx.decode(k)
                        # evaluate operation
                        fl = op(x, y, z, ctx=ctx)
                        # evaluate equivalent operation
                        r, flags = _mpfr_apply(mpfr, (x, y, z), ctx)
                        ref = ctx.round(r)
                        # sanity check: ref should be exact
                        assert not ref.inexact
                        assert isinstance(fl, Float)
                        # check that they are the same
                        if fl.isnan:
                            assert fl.isnan, f'op={op}, rm={rm}, x={x}, y={y}, z={z}, fl={fl}, ref={ref}'
                        else:
                            assert fl == ref, f'op={op}, rm={rm}, x={x}, y={y}, z={z}, fl={fl}, ref={ref}'
                        # check that the flags are the same
                        assert fl.overflow == flags.overflow, f'op={op}, rm={rm}, x={x}, y={y}, z={z}, fl={fl}, ref={ref}, flags={flags}'
                        assert fl.inexact == flags.inexact, f'op={op}, rm={rm}, x={x}, y={y}, z={z}, fl={fl}, ref={ref}, flags={flags}'
