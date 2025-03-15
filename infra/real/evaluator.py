import time

from dataclasses import dataclass
from statistics import geometric_mean
from typing import Optional

from fpy2 import *
from fpy2.runtime.sampling import sample_function
from fpy2.runtime.real.rival_manager import PrecisionLimitExceeded

from .common import disabled_tests
from .config import Config
from .load import load_funs

@dataclass
class Report:
    name: str
    seed: Optional[int]
    num_samples: int
    num_failed: int
    slowdown: Optional[float]

    @property
    def num_passed(self):
        return self.num_samples - self.num_failed


def _run_one(
    fun: Function,
    rt: Interpreter,
    num_samples: int,
    *,
    seed: Optional[int] = None,
    print_result: bool = False,
    base_rt: Optional[Interpreter] = None,
):
    # sample N points
    pts = sample_function(fun, num_samples, seed=seed, only_real=True)

    print(f'evaluating {fun.name} ', end='', flush=True)

    # evaluate over each point
    num_failed = 0
    slowdowns = []
    for pt in pts:
        try:
            start = time.time()
            result = rt.eval(fun, pt)
            end = time.time()

            if print_result:
                print(' ', result, flush=True)
            else:
                print('.', end='', flush=True)

            if base_rt is not None:
                base_start = time.time()
                base_rt.eval(fun, pt)
                base_end = time.time()
                slowdown = (end - start) / (base_end - base_start)
                slowdowns.append(slowdown)

        except PrecisionLimitExceeded:
            num_failed += 1
            if print_result:
                print(' ?', flush=True)
            else:
                print('?', end='', flush=True)

    print('', flush=True)

    return Report(
        name=fun.name,
        seed=seed,
        num_samples=num_samples,
        num_failed=num_failed,
        slowdown=None if slowdowns == [] else sum(slowdowns) / len(slowdowns),
    )


def _summarize(reports: list[Report]):
    num_passed = sum(r.num_passed for r in reports)
    num_samples = sum(r.num_samples for r in reports)
    pct = 100 * (num_passed / num_samples)
    print(f'Successful: {pct:.2f}%')

    slowdowns: list[float] = []
    for report in reports:
        if report.slowdown is not None:
            slowdowns.append(report.slowdown)

    avg_slowdown = geometric_mean(slowdowns)
    print(f'Average slowdown: {avg_slowdown:.2f}')


def run_eval_real(config: Config):
    rt = RealInterpreter()
    base_rt = TitanicInterpreter()
    funs = load_funs(config.input_paths)

    disabled = disabled_tests()

    print(f'testing over {len(funs)} functions')
    reports: list[Report] = []
    for fun in funs:
        if fun.name in disabled:
            print(f'skipping {fun.name}')
        else:
            report = _run_one(fun, rt, config.num_samples, seed=config.seed, base_rt=base_rt)
            reports.append(report)

    _summarize(reports)
