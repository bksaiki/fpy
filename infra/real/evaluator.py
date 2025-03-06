from .config import Config
from .load import load_funs

from fpy2 import *
from fpy2.runtime import sample_function
from fpy2.runtime.real.rival_manager import PrecisionLimitExceeded, ConvergenceFailed

_disabled = [
    # Too hard
    'Rocket_Trajectory',
    'Eigenvalue_Computation',
    'Pendulum',
    'Flower',
    'arclength_of_a_wiggly_function',
    'arclength_of_a_wiggly_function__u40_old_version_u41_',
    'Jacobi_u39_s_Method',
    # Infinite loops
    'Euler_Oscillator',
    'Filter',
    'Circle',
    # tensor
    'Symplectic_Oscillator',
    'Flower',
    'Arrow_Hurwicz'
]

def _run_one(
    fun: Function,
    rt: Interpreter,
    num_samples: int,
    print_result: bool = False
):
    # sample N points
    pts = sample_function(fun, num_samples, only_real=True)
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
            _run_one(fun, rt, config.num_samples)

