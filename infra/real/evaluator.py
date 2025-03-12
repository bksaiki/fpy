from typing import Optional

from fpy2 import *
from fpy2.runtime.sampling import sample_function
from fpy2.runtime.real.rival_manager import PrecisionLimitExceeded, ConvergenceFailed

from .common import _disabled
from .config import Config
from .load import load_funs

def _run_one(
    fun: Function,
    rt: Interpreter,
    num_samples: int,
    *,
    seed: Optional[int] = None,
    print_result: bool = False
):
    # sample N points
    pts = sample_function(fun, num_samples, seed=seed, only_real=True)

    # evaluate over each point
    print(f'evaluating {fun.name} ', end='', flush=True)
    for pt in pts:
        try:
            result = rt.eval(fun, pt)
            if print_result:
                print(' ', result, flush=True)
            else:
                print('.', end='', flush=True)
        except ConvergenceFailed:
            if print_result:
                print(' X', flush=True)
            else:
                print('X', end='', flush=True)
        except PrecisionLimitExceeded:
            if print_result:
                print(' ?', flush=True)
            else:
                print('?', end='', flush=True)
    print('', flush=True)


def run_eval_real(config: Config):
    rt = RealInterpreter()
    funs = load_funs(config.input_paths)
    print(len(funs))

    print(f'testing over {len(funs)} functions')
    for fun in funs:
        if fun.name in _disabled:
            print(f'skipping {fun.name}')
        else:
            _run_one(fun, rt, config.num_samples, seed=config.seed)

