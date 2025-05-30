import fpy2.math
import types

from fpy2 import Function, ForeignEnv, Float, IEEEContext, RoundingMode
from titanfp.fpbench.fpcast import FPCore, Nearbyint
from titanfp.arithmetic.evalctx import EvalCtx, RM
from titanfp.arithmetic.ieee754 import IEEECtx
from titanfp.arithmetic.mpmf import MPMF, Interpreter

from .fetch import fetch_cores

_skip_cores = [
    # infinite loops
    'Euler Oscillator',
    'Filter',
    'Circle',
]

def _mpmf_to_float(x: MPMF) -> Float:
    return Float(s=x.negative, exp=x.exp, c=x.c, isinf=x.isinf, isnan=x.isnan)

def _to_fpy_rm(rm: RM):
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

def _to_fpy_context(ctx: EvalCtx):
    match ctx:
        case IEEECtx():
            return IEEEContext(ctx.es, ctx.nbits, _to_fpy_rm(ctx.rm))
        case _:
            raise RuntimeError(f'unsupported context type: {type(ctx)} for {ctx}')

def _eval_nearbyint(rt: Interpreter, e: Nearbyint, ctx: EvalCtx):
    arg = rt.evaluate(e.children[0], ctx)
    v = fpy2.math.nearbyint(_mpmf_to_float(arg), ctx=_to_fpy_context(ctx))
    result = MPMF(negative=v.s, exp=v.exp, c=v.c, isinf=v.isinf, isnan=v.isnan, inexact=v.inexact)
    return [v], result

def _mpmf_interpreter():
    """
    Creates a hacked version of the MPMF interpreter.

    - fixes `nearbyint` to use FPy's implementation;
    Titanic's `nearbyint` is hopeless broken.
    """
    rt = Interpreter()
    rt._eval_nearbyint = types.MethodType(_eval_nearbyint, rt)
    return rt


def eval(
    rt: Interpreter,
    env: ForeignEnv,
    core: FPCore,
    *,
    num_inputs: int = 10
):
    # apply filter
    if core.name in _skip_cores:
        print('skipping', core.ident, 'due to filter')
        return

    # convert to FPy
    fun = Function.from_fpcore(core, ignore_unknown=True)
    fun.env = env

    # register the function
    if core.ident is not None:
        rt.register_function(core)
        print(fun.name)
        env.globals[fun.name] = fun

    # sample inputs
    if core.inputs == []:
        inputs: list[list] = [[] for _ in range(num_inputs)]
    else:
        print('skipping sampling for', core.ident)
        return

    print('evaluating', core.name, 'with', num_inputs, 'inputs')

    # evaluate both FPy and FPCore functions
    for input in inputs:
        # evaluate functions on point
        fpcore = _mpmf_to_float(rt.interpret(core, input))
        fpy = fun(*input)

        # compare the outputs
        if (fpcore.isnan and not fpy.isnan) or (fpcore != fpy):
            raise ValueError(f'Outputs do not match for {core.name} {input}: {fpcore} != {fpy}\n{core}\n{fun.format()}')

        # TODO: check outputs are equal


def test_eval():
    rt = _mpmf_interpreter()
    env = ForeignEnv.empty()
    fpbench = fetch_cores()
    for core in fpbench.all_cores():
        eval(rt, env, core)



if __name__ == "__main__":
    test_eval()
