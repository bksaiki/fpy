import fpy2 as fp
from pathlib import Path
from argparse import ArgumentParser
from typing import TypeAlias

from .examples import *
from .options import CompileConfig, EvalConfig
from .time import time_benchmark

ExampleType: TypeAlias = tuple[fp.Function, tuple[fp.Context | None, ...]]


DEFAULT_NUM_INPUTS = 1_000_000

EXAMPLES: list[ExampleType] = [
    (pre_round_mul, (fp.FP32, fp.FP32)),
    (dot_prod_1, (fp.FP32, fp.FP32, fp.FP32)),
    (dot_prod_blocked, (fp.FP32, fp.FP32, fp.FP32)),
    (dot_prod_arm, (fp.FP32, fp.FP32, fp.FP32)),
    (mx_block_round, (fp.FP32,)),
    (mx_dot_prod, (fp.MX_E8M0, fp.MX_E8M0, fp.MX_E5M2, fp.MX_E5M2))
]

def run_eval(config: EvalConfig, examples: list[ExampleType]) -> None:
    # build list of benchmarks to run
    benchmarks: list[tuple[ExampleType, CompileConfig]] = []
    for func, ctxs in examples:
        for elim_round in (False, True):
            for allow_exact in (False, True):
                compile_config = CompileConfig(elim_round=elim_round, allow_exact=allow_exact)
                benchmarks.append(((func, ctxs), compile_config))

    for i, benchmark in enumerate(benchmarks):
        (func, ctxs), compile_config = benchmark
        time_benchmark(func, ctxs, i, config, compile_config)

if __name__ == '__main__':
    parser = ArgumentParser(description='FPy/MPFX evaluation')
    parser.add_argument('--num-inputs', type=int, default=DEFAULT_NUM_INPUTS, help='Number of inputs to generate per benchmark (default: 100000)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for input generation (default: 1)')
    parser.add_argument('output_dir', type=Path, help='Output directory for results')
    args = parser.parse_args()

    num_inputs: int = args.num_inputs
    seed: int = args.seed
    output_dir: Path = args.output_dir.resolve()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Evaluation output directory: `{output_dir}`')

    # Eval configuration
    config = EvalConfig(
        output_dir=output_dir,
        num_inputs=num_inputs,
        seed=seed
    )

    # Run evaluation harness
    run_eval(config, EXAMPLES)
    