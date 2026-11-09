import types

from fpy2 import Float, IEEEContext, RoundingMode, fmax, fmin, nearbyint, pow as fpy_pow
from fractions import Fraction

from titanfp.arithmetic.evalctx import EvalCtx, RM
from titanfp.arithmetic.ieee754 import IEEECtx
from titanfp.arithmetic.mpmf import MPMF, Interpreter
from titanfp.fpbench.fpcast import Fmax, Fmin, Nearbyint, Pow
from titanfp.titanic import ndarray, utils

def fpy_to_mpmf(x: bool | Float | list):
    match x:
        case bool():
            return x
        case Float():
            return MPMF(negative=x.s, exp=x.exp, c=x.c, isinf=x.isinf, isnan=x.isnan)
        case list():
            return [fpy_to_mpmf(v) for v in x]
        case _:
            raise TypeError(f'Expected Float or ndarray.NDArray, got {x}')

def mpmf_to_fpy(x: bool | MPMF | ndarray.NDArray):
    match x:
        case bool():
            return x
        case MPMF():
            return Float(s=x.negative, exp=x.exp, c=x.c, isinf=x.isinf, isnan=x.isnan)
        case ndarray.NDArray():
            return [mpmf_to_fpy(v) for v in x]
        case _:
            raise TypeError(f'Expected MPMF or ndarray.NDArray, got {x}')

def compare(
    expect: bool | Float | list,
    actual: bool | Float | list,
    **kwargs
):
    match expect, actual:
        case bool(), bool():
            if expect != actual:
                kwarg_str = '\n'.join(f'{k}={v}' for k, v in kwargs.items())
                raise ValueError(f'Outputs do not match: {expect} != {actual}\n kwargs={kwarg_str}')
        case Float() | Fraction(), Float() | Fraction():
            if not actual.isnan if expect.isnan else expect != actual:
                kwarg_str = '\n'.join(f'{k}={v}' for k, v in kwargs.items())
                raise ValueError(f'Outputs do not match: {expect} != {actual}\n kwargs={kwarg_str}')
        case list() | tuple(), list() | tuple():
            if len(expect) != len(actual):
                kwarg_str = '\n'.join(f'{k}={v}' for k, v in kwargs.items())
                raise ValueError(f'Outputs do not match: {expect} != {actual}\n kwargs={kwarg_str}')
            for expect_elt, actual_elt in zip(expect, actual):
                compare(expect_elt, actual_elt, **kwargs)
        case _:
            print(type(expect), type(actual))
            kwarg_str = '\n'.join(f'{k}={v}' for k, v in kwargs.items())
            raise ValueError(f'Outputs do not match: {expect} != {actual}\n kwargs={kwarg_str}')

def to_fpy_rm(rm: RM):
    match rm:
        case RM.RNE:
            return RoundingMode.RNE
        case RM.RNA:
            return RoundingMode.RNA
        case RM.RTP:
            return RoundingMode.RTP
        case RM.RTN:
            return RoundingMode.RTN
        case RM.RTZ:
            return RoundingMode.RTZ
        case RM.RAZ:
            return RoundingMode.RAZ
        case _:
            raise RuntimeError(f'unsupported rounding mode: {rm}')

def to_fpy_context(ctx: EvalCtx):
    match ctx:
        case IEEECtx():
            return IEEEContext(ctx.es, ctx.nbits, to_fpy_rm(ctx.rm))
        case _:
            raise RuntimeError(f'unsupported context type: {type(ctx)} for {ctx}')

def _eval_nearbyint(rt: Interpreter, e: Nearbyint, ctx: EvalCtx):
    arg = rt.evaluate(e.children[0], ctx)
    v = nearbyint(mpmf_to_fpy(arg), ctx=to_fpy_context(ctx))
    result = fpy_to_mpmf(v)
    return [v], result

def _eval_fmax(rt: Interpreter, e: Fmax, ctx: EvalCtx):
    in0, in1 = (rt.evaluate(child, ctx) for child in e.children)
    v = fmax(mpmf_to_fpy(in0), mpmf_to_fpy(in1), ctx=to_fpy_context(ctx))
    return [in0, in1], fpy_to_mpmf(v)

def _eval_fmin(rt: Interpreter, e: Fmin, ctx: EvalCtx):
    in0, in1 = (rt.evaluate(child, ctx) for child in e.children)
    v = fmin(mpmf_to_fpy(in0), mpmf_to_fpy(in1), ctx=to_fpy_context(ctx))
    return [in0, in1], fpy_to_mpmf(v)

def _eval_pow(rt: Interpreter, e: Pow, ctx: EvalCtx):
    in0, in1 = (rt.evaluate(child, ctx) for child in e.children)
    if in1.isinf or in1.isnan:
        v = fpy_to_mpmf(fpy_pow(mpmf_to_fpy(in0), mpmf_to_fpy(in1), ctx=to_fpy_context(ctx)))
    else:
        v = in0.pow(in1, ctx=ctx)
    return [in0, in1], v

def _is_even_for_rounding(c: int, exp: int):
    return c & 1 == 0

def make_check_offset(check_offset):
    def _check_offset(data, shape, start, strides):
        # bail for empty tensors
        for dim in shape:
            if dim == 0:
                return
        return check_offset(data, shape, start, strides)
    return _check_offset

def mpmf_interpreter():
    """
    Creates a hacked version of the MPMF interpreter and
    manually patches Titanic's runtime

    - fixes `nearbyint` to use FPy's implementation;
    Titanic's `nearbyint` is hopeless broken.
    - fixes `fmax` and `fmin` to use FPy's implementation;
    Titanic's are Python's `max` and `min`, which ignore NaN
    only as the second argument.
    - fixes `pow` with an infinite or NaN exponent to use FPy's implementation;
    Titanic's treats them as zero, e.g., `pow(2, inf) = 1`.
    - fixes ties to even when the significand has one bit, as at the
    bottom of an IEEE format; Titanic decides by the exponent,
    e.g., `3 * 2**-1074 / 2 = 2**-1074`.
    - fixes ndarray `check_offset` for empty tensors
    """
    rt = Interpreter()
    rt._eval_nearbyint = types.MethodType(_eval_nearbyint, rt)
    rt._eval_fmax = types.MethodType(_eval_fmax, rt)
    rt._eval_fmin = types.MethodType(_eval_fmin, rt)
    rt._eval_pow = types.MethodType(_eval_pow, rt)
    utils.is_even_for_rounding = _is_even_for_rounding
    ndarray.check_offset = make_check_offset(ndarray.check_offset)
    return rt
