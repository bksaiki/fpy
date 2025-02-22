from .config import Config
from .load import load_funs
from .sample import sample

from fpy2 import *

def _run_one(fun: Function, rt: Interpreter, num_samples: int):
    # sample N points
    pts = sample(fun, num_samples)
    # evaluate over each point
    print(f'evaluating {fun.name} ', end='', flush=True)
    for pt in pts:
        rt.eval(fun, pt)
        print('.', end='', flush=True)
    print('', flush=True)


def run_eval_real(config: Config):
    rt = TitanicInterpreter()
    funs = load_funs(config.input_paths)

    print(f'testing over {len(funs)} functions')
    for fun in funs:
        if fun.name == 'Rocket_Trajectory' or fun.name == 'Eigenvalue_Computation':
            print(f'skipping {fun.name}')
        else:
            _run_one(fun, rt, config.num_samples)

