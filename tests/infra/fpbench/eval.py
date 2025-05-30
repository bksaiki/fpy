from fpy2 import Function, ForeignEnv
from titanfp.fpbench.fpcast import FPCore
from titanfp.arithmetic.mpmf import MPMF, Interpreter

from .fetch import fetch_cores

_skip_cores = [
    # infinite loops
    'Euler Oscillator',
    'Filter',
    'Circle',
]



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
        fpcore = rt.interpret(core, input)
        fpy = fun(*input)

        # TODO: check outputs are equal


def test_eval():
    rt = Interpreter()
    env = ForeignEnv.empty()
    fpbench = fetch_cores()
    for core in fpbench.all_cores():
        eval(rt, env, core)



if __name__ == "__main__":
    test_eval()
