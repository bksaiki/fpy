import fpy2.math
import types

from fpy2 import Float, IEEEContext, RoundingMode
from titanfp.arithmetic.evalctx import EvalCtx, RM
from titanfp.arithmetic.ieee754 import IEEECtx
from titanfp.arithmetic.mpmf import MPMF, Interpreter
from titanfp.fpbench.fpcast import Nearbyint
from titanfp.titanic.ndarray import NDArray

def fpy_to_mpmf(x: bool | Float | NDArray):
    match x:
        case bool():
            return x
        case Float():
            return MPMF(negative=x.s, exp=x.exp, c=x.c, isinf=x.isinf, isnan=x.isnan)
        case NDArray():
            return NDArray([fpy_to_mpmf(v) for v in x])
        case _:
            raise TypeError(f'Expected Float or NDArray, got {x}')

def mpmf_to_fpy(x: bool | MPMF | NDArray):
    match x:
        case bool():
            return x
        case MPMF():
            return Float(s=x.negative, exp=x.exp, c=x.c, isinf=x.isinf, isnan=x.isnan)
        case NDArray():
            return NDArray([mpmf_to_fpy(v) for v in x])
        case _:
            raise TypeError(f'Expected MPMF or NDArray, got {x}')

def compare(
    expect: bool | Float | NDArray,
    actual: bool | Float | NDArray,
    **kwargs
):
    match expect, actual:
        case bool(), bool():
            if expect != actual:
                kwarg_str = '\n'.join(f'{k}={v}' for k, v in kwargs.items())
                raise ValueError(f'Outputs do not match: {expect} != {actual}\n kwargs={kwarg_str}')
        case Float(), Float():
            if not actual.isnan if expect.isnan else expect != actual:
                kwarg_str = '\n'.join(f'{k}={v}' for k, v in kwargs.items())
                raise ValueError(f'Outputs do not match: {expect} != {actual}\n kwargs={kwarg_str}')
        case NDArray(), NDArray():
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
    v = fpy2.math.nearbyint(mpmf_to_fpy(arg), ctx=to_fpy_context(ctx))
    result = fpy_to_mpmf(v)
    return [v], result

def mpmf_interpreter():
    """
    Creates a hacked version of the MPMF interpreter.

    - fixes `nearbyint` to use FPy's implementation;
    Titanic's `nearbyint` is hopeless broken.
    """
    rt = Interpreter()
    rt._eval_nearbyint = types.MethodType(_eval_nearbyint, rt)
    return rt
