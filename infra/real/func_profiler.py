from typing import Optional

from fpy2 import *
from fpy2.sample import sample_function

from .common import disabled_tests, select_interpreter
from .config import Config, ReferenceMode
from .load import load_funs

# the profilers
profiler_ref = FunctionProfiler(reference=select_interpreter(ReferenceMode.REAL), logging=True)
profiler_1k = FunctionProfiler(reference=select_interpreter(ReferenceMode.FLOAT_1K), logging=True)
profiler_2k = FunctionProfiler(reference=select_interpreter(ReferenceMode.FLOAT_2K), logging=True)
profiler_4k = FunctionProfiler(reference=select_interpreter(ReferenceMode.FLOAT_4K), logging=True)

limit = 3

def _print_result(ref: Optional[bool], actual: bool):
    if ref is None:
        if actual:
            print('UP ', end='')
        else:
            print('UN ', end='')
    elif ref == actual:
        if actual:
            print('TP ', end='')
        else:
            print('TN ', end='')
    else:
        if actual:
            print('FP ', end='')
        else:
            print('FN ', end='')


def _run_one(
    fun: Function,
    num_samples: int,
    *,
    seed: Optional[int] = None,
):
    # sample N points
    pts = sample_function(fun, num_samples, seed=seed, only_real=True)

    # evaluate over each point
    print(f'profiling {fun.name} ', end='', flush=True)
    errors_ref, _ = profiler_ref.profile(fun, pts)
    print('|', end='', flush=True)
    errors_1k, _ = profiler_1k.profile(fun, pts)
    print('|', end='', flush=True)
    errors_2k, _ = profiler_2k.profile(fun, pts)
    print('|', end='', flush=True)
    errors_4k, _ = profiler_4k.profile(fun, pts)
    print('')

    avg_1k = sum(errors_1k) / len(errors_1k)
    avg_2k = sum(errors_2k) / len(errors_2k)
    avg_4k = sum(errors_4k) / len(errors_4k)

    err_1k = avg_1k > limit
    err_2k = avg_2k > limit
    err_4k = avg_4k > limit

    if errors_ref is None:
        _print_result(None, err_1k)
        _print_result(None, err_2k)
        _print_result(None, err_4k)
    else:
        avg_ref = sum(errors_ref) / len(errors_ref)
        err_ref = avg_ref > limit

        print(err_ref - err_1k, err_ref - err_2k, err_ref - err_4k)

        _print_result(err_ref, err_1k)
        _print_result(err_ref, err_2k)
        _print_result(err_ref, err_4k)
    print('', flush=True)


def run_func_profiler(config: Config):
    funs = load_funs(config.input_paths)
    disabled = disabled_tests()

    print(f'testing over {len(funs)} functions')
    for fun in funs:
        if fun.name in disabled:
            print(f'skipping {fun.name}')
        else:
            _run_one(fun, config.num_samples, seed=config.seed)
