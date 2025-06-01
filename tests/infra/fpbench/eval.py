import fpy2.math
import types

from fpy2 import Function, ForeignEnv, Float, IEEEContext, RoundingMode, NoSuchContextError
from titanfp.arithmetic.evalctx import EvalCtx, RM
from titanfp.arithmetic.ieee754 import IEEECtx
from titanfp.arithmetic.mpmf import MPMF, Interpreter
from titanfp.fpbench.fpcast import FPCore, Nearbyint
from titanfp.titanic.ndarray import NDArray

from .fetch import fetch_cores

_skip_cores = [
    # infinite loops
    'Euler Oscillator',
    'Filter',
    'Circle',
    # loops too long
    'Rocket Trajectory',
    'Flower',
    # unsupported context
    'arclength of a wiggly function',
    'arclength of a wiggly function (old version)',
    # mutual recursion
    'even-int'
]

def _fpy_to_mpmf(x: bool | Float | NDArray):
    match x:
        case bool():
            return x
        case Float():
            return MPMF(negative=x.s, exp=x.exp, c=x.c, isinf=x.isinf, isnan=x.isnan)
        case NDArray():
            return NDArray([_fpy_to_mpmf(v) for v in x])
        case _:
            raise TypeError(f'Expected Float or NDArray, got {x}')

def _mpmf_to_fpy(x: bool | MPMF | NDArray):
    match x:
        case bool():
            return x
        case MPMF():
            return Float(s=x.negative, exp=x.exp, c=x.c, isinf=x.isinf, isnan=x.isnan)
        case NDArray():
            return NDArray([_mpmf_to_fpy(v) for v in x])
        case _:
            raise TypeError(f'Expected MPMF or NDArray, got {x}')

def _compare(
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
                _compare(expect_elt, actual_elt, **kwargs)
        case _:
            print(type(expect), type(actual))
            kwarg_str = '\n'.join(f'{k}={v}' for k, v in kwargs.items())
            raise ValueError(f'Outputs do not match: {expect} != {actual}\n kwargs={kwarg_str}')

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
    v = fpy2.math.nearbyint(_mpmf_to_fpy(arg), ctx=_to_fpy_context(ctx))
    result = _fpy_to_mpmf(v)
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
    # convert to FPy
    fun = Function.from_fpcore(core, ignore_unknown=True)
    fun.env = env

    # register the function
    if core.ident is not None:
        rt.register_function(core)
        env.globals[fun.name] = fun

    # apply filter
    if core.name in _skip_cores or core.ident in _skip_cores:
        print('skipping', core.ident, 'due to filter')
        return

    # sample inputs
    if core.inputs == []:
        inputs: list[list] = [[]]
    else:
        inputs = []
        for _ in range(num_inputs):
            input: list[Float] = []
            for name, _, shape in core.inputs:
                if shape is None:
                    # scalar argument
                    # TODO: sampling
                    v = Float.from_float(1.0, None)
                    input.append(v)
                else:
                    # tensor argument
                    # Replace symbolic dimensions with N=3
                    dims = []
                    for dim in shape:
                        if isinstance(dim, int):
                            dims.append(dim)
                        else:
                            dims.append(3)
                    def make_tensor(dims):
                        if not dims:
                            return Float.from_float(1.0, None)
                        return NDArray([make_tensor(dims[1:]) for _ in range(dims[0])])
                    v = make_tensor(dims)
                    input.append(v)
                    print(v)
            inputs.append(input)

    print('evaluating', core.name, 'with', len(inputs), 'inputs', end='')

    # evaluate both FPy and FPCore functions
    for input in inputs:
        # evaluate functions on point
        try:
            fpcore = _mpmf_to_fpy(rt.interpret(core, list(map(_fpy_to_mpmf, input))))
            fpy = fun(*input)
        except NoSuchContextError:
            # TODO: implement integer contexts
            print('skipping', core.ident, 'due to NoSuchContextError')
            return 
        _compare(fpcore, fpy, core=core, func=fun.format(), input=input)
        print('.', end='', flush=True)

    print('', flush=True)


def test_eval():
    rt = _mpmf_interpreter()
    env = ForeignEnv.empty()
    fpbench = fetch_cores()
    for core in fpbench.tensor_cores:
        eval(rt, env, core)



if __name__ == "__main__":
    test_eval()
