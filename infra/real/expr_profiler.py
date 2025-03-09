from .config import Config
from .load import load_funs

from fpy2 import *
from fpy2.runtime.sampling import sample_function
from fpy2.runtime.real import ExpressionProfiler

from .common import _disabled


def _run_one(fun: Function, profiler: ExpressionProfiler, num_samples: int):
    # sample N points
    pts = sample_function(fun, num_samples, only_real=True)
    # evaluate over each point
    print(f'profiling {fun.name} ', end='', flush=True)
    report = profiler.profile(fun, pts)
    print(report)


def run_expr_profiler(config: Config):
    profiler = ExpressionProfiler()
    funs = load_funs(config.input_paths)
    print(len(funs))

    print(f'testing over {len(funs)} functions')
    for fun in funs:
        if fun.name in _disabled:
            print(f'skipping {fun.name}')
        else:
            _run_one(fun, profiler, config.num_samples)
