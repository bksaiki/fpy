from typing import Optional

from fpy2 import *
from fpy2.runtime.sampling import sample_function
from fpy2.runtime.real import FunctionProfiler

from .common import disabled_tests, select_interpreter
from .config import Config
from .load import load_funs

def _run_one(
    fun: Function,
    profiler: FunctionProfiler,
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

def run_func_profiler(config: Config):
    reference = select_interpreter(config.ref_mode)
    profiler = FunctionProfiler(logging=True)
    funs = load_funs(config.input_paths)
    print(len(funs))

    print(f'testing over {len(funs)} functions')
    for fun in funs:
        if fun.name in disabled_tests():
            print(f'skipping {fun.name}')
        else:
            _run_one(fun, profiler, config.num_samples, seed=config.seed)
