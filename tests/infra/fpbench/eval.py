from fpy2 import Function, ForeignEnv, Float, FP64, NoSuchContextError
from titanfp.arithmetic.mpmf import Interpreter
from titanfp.fpbench.fpcast import FPCore
from titanfp.titanic.ndarray import NDArray

from .fetch import fetch_cores
from .shim import fpy_to_mpmf, mpmf_to_fpy, mpmf_interpreter, compare

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
    'even-int',
    'forward-euler-3d',
    'midpoint-3d',
    'ralston-3d',
    'rk4-step-3d'
]

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
            for _, _, shape in core.inputs:
                if shape is None:
                    # scalar argument
                    # TODO: sampling
                    input.append(Float.from_float(1.0, FP64))
                else:
                    dims = []
                    for dim in shape:
                        if isinstance(dim, int):
                            dims.append(dim)
                        else:
                            dims.append(3)
                    def make_tensor(dims):
                        if not dims:
                            return Float.from_float(1.0, FP64)
                        return [make_tensor(dims[1:]) for _ in range(dims[0])]
                    input.append(make_tensor(dims))
            inputs.append(input)

    print('evaluating', core.name, 'with', len(inputs), 'inputs', end='')

    # evaluate both FPy and FPCore functions
    for input in inputs:
        # evaluate functions on point
        try:
            fpcore = mpmf_to_fpy(rt.interpret(core, list(map(fpy_to_mpmf, input))))
            fpy = fun(*input)
        except NoSuchContextError:
            # TODO: implement integer contexts
            print('skipping', core.ident, 'due to NoSuchContextError')
            return

        # compare output
        compare(fpcore, fpy, core=core, func=fun.format(), input=input)
        print('.', end='', flush=True)

    print('', flush=True)


def test_eval():
    rt = mpmf_interpreter()
    env = ForeignEnv.empty()
    fpbench = fetch_cores()
    for core in fpbench.all_cores():
        eval(rt, env, core)

if __name__ == "__main__":
    test_eval()
