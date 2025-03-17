from typing import Optional
import json

from fpy2 import *
from fpy2.runtime.sampling import sample_function

from .common import disabled_tests, select_interpreter
from .config import Config, ReferenceMode
from .load import load_funs


def _run_one(
    fun: Function,
    profiler: ExprProfiler,
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
    return report.result_dict()


def run_expr_profiler(config: Config):
    modes = [ReferenceMode.REAL, ReferenceMode.FLOAT_1K, ReferenceMode.FLOAT_2K, ReferenceMode.FLOAT_4K]
    profilers = [(mode, ExprProfiler(reference=select_interpreter(mode), logging=True)) for mode in modes]
    funs = load_funs(config.input_paths)

    disabled = disabled_tests()

    print(f'testing over {len(funs)} functions')

    result: dict[str, dict[ReferenceMode, dict]] = {}
    for fun in funs:
        if fun.name in disabled:
            print(f'skipping {fun.name}')
        else:
            result[fun.name] = {}
            for mode, profiler in profilers:
                result[fun.name][mode] = _run_one(fun, profiler, config.num_samples, seed=config.seed)

    with open(f"experiment_result.json", "w") as json_file:
        json.dump(result, json_file, indent=4)
