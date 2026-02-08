import fpy2 as fp
from pathlib import Path
from argparse import ArgumentParser
from typing import NamedTuple, TypeAlias

from .examples import *
from .options import CompileConfig, EvalConfig
from .time import time_benchmark
from .utils import Benchmark

EXAMPLES: list[Benchmark] = [
    Benchmark(vec_mul, (fp.FP32, fp.FP32), 1000, (1 << 16)),
    Benchmark(dot_prod_1, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    Benchmark(dot_prod_blocked, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 12)),
    Benchmark(dot_prod_arm, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 12)),
    Benchmark(mx_block_round, (fp.FP32,), 1000, (1 << 12)),
    Benchmark(mx_dot_prod, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 12)),
    Benchmark(mx_matmul, (fp.FP32, fp.FP32), 10, 256)
]

def run_eval(config: EvalConfig, examples: list[Benchmark]) -> None:
    # build list of benchmarks to run
    benchmarks: list[tuple[Benchmark, CompileConfig]] = []
    for example in examples:
        for elim_round in (False, True):
            for allow_exact in (False, True):
                compile_config = CompileConfig(elim_round=elim_round, allow_exact=allow_exact)
                benchmarks.append((example, compile_config))

    for i, (benchmark, compile_config) in enumerate(benchmarks):
        time_benchmark(benchmark, i, config, compile_config)

if __name__ == '__main__':
    parser = ArgumentParser(description='FPy/MPFX evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for input generation (default: 1)')
    parser.add_argument('output_dir', type=Path, help='Output directory for results')
    args = parser.parse_args()

    seed: int = args.seed
    output_dir: Path = args.output_dir.resolve()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Evaluation output directory: `{output_dir}`')

    # Eval configuration
    config = EvalConfig(
        output_dir=output_dir,
        seed=seed
    )

    # Run evaluation harness
    run_eval(config, EXAMPLES)
    