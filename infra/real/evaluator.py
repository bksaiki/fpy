import time

from dataclasses import dataclass
from statistics import geometric_mean
from typing import Optional, Sequence

from fpy2 import *
from fpy2.sample import sample_function
from fpy2.runtime.real.rival_manager import PrecisionLimitExceeded
from titanfp.arithmetic.ieee754 import ieee_ctx

from .common import disabled_tests
from .config import Config
from .load import load_funs

@dataclass
class Report:
    name: str
    seed: Optional[int]
    num_samples: int
    num_failed: int
    slowdowns: list[list[float]]

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
    base_rts: Optional[Sequence[Interpreter]] = None,
):
    # sample N points
    pts = sample_function(fun, num_samples, seed=seed, only_real=True)

    # prepare base interpreters
    if base_rts is None:
        base_rts = []

    print(f'evaluating {fun.name} ', end='', flush=True)

    # evaluate over each point
    num_failed = 0
    slowdownss: list[list[float]] = [[] for _ in base_rts]
    for pt in pts:
        try:
            start = time.monotonic_ns()
            result = rt.eval(fun, pt)
            end = time.monotonic_ns()
            time_elapsed = end - start
            assert time_elapsed >= 0, f'{time_elapsed}'

            if print_result:
                print(' ', result, flush=True)
            else:
                print('.', end='', flush=True)

            for base_rt, slowdowns in zip(base_rts, slowdownss):
                base_start = time.monotonic_ns()
                base_rt.eval(fun, pt)
                base_end = time.monotonic_ns()
                base_time_elapsed = base_end - base_start
                assert base_time_elapsed >= 0, f'{base_time_elapsed}'

                slowdown = time_elapsed / base_time_elapsed
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
        slowdowns=slowdownss
    )

def _summarize(reports: list[Report]):
    assert reports != [], 'no reports to summarize'

    # report success rate
    num_passed = sum(r.num_passed for r in reports)
    num_samples = sum(r.num_samples for r in reports)
    pct = 100 * (num_passed / num_samples)
    print(f'Successful: {pct:.2f}%')

    # report overhead
    num_base_rts = len(reports[0].slowdowns)
    for i in range(num_base_rts):
        slowdowns: list[float] = []
        for r in reports:
            slowdowns.extend(r.slowdowns[i])
        avg = geometric_mean(slowdowns)
        print(f'Baseline {i} avg slowdown: {avg:.2f}')


def run_eval_real(config: Config):
    # reference interpreter
    rt = RealInterpreter()

    # baseline interpreter
    rt_1k = TitanicInterpreter(ctx=ieee_ctx(19, 1024))
    rt_2k = TitanicInterpreter(ctx=ieee_ctx(19, 2048))
    rt_4k = TitanicInterpreter(ctx=ieee_ctx(19, 4096))
    base_rts = [rt_1k, rt_2k, rt_4k]

    # load benchmarks
    funs = load_funs(config.input_paths)
    disabled = disabled_tests()

    print(f'testing over {len(funs)} functions')

    num_skipped = 0
    reports: list[Report] = []
    for fun in funs:
        if fun.name in disabled:
            print(f'skipping {fun.name}')
            num_skipped += 1
        else:
            report = _run_one(fun, rt, config.num_samples, seed=config.seed, base_rts=base_rts)
            reports.append(report)

    _summarize(reports)
    print(f'Skipped: {num_skipped}')
