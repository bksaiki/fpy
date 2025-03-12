from typing import Optional

from fpy2 import *
from fpy2.runtime.sampling import sample_function
from fpy2.runtime.real import ExpressionProfiler

from .common import disabled_tests
from .config import Config
from .load import load_funs


_disabled = disabled_tests() + ['PID']

def _run_one(
    fun: Function,
    profiler: ExpressionProfiler,
    num_samples: int,
    *,
    seed: Optional[int] = None,
):
    # sample N points
    pts = sample_function(fun, num_samples, seed=seed, only_real=True)

    # evaluate over each point
    print(f'profiling {fun.name} ', end='', flush=True)
    report = profiler.profile(fun, pts)
    print(report)


def run_expr_profiler(config: Config):
    profiler = ExpressionProfiler(logging=True)
    funs = load_funs(config.input_paths)
    print(len(funs))

    print(f'testing over {len(funs)} functions')
    for fun in funs:
        if fun.name in _disabled:
            print(f'skipping {fun.name}')
        else:
            _run_one(fun, profiler, config.num_samples, seed=config.seed)
