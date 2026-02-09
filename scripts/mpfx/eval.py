import fpy2 as fp
import pickle, gzip

from pathlib import Path
from argparse import ArgumentParser

from .examples import *
from .options import CompileConfig, EvalConfig, OptOptions
from .plot import plot_times
from .time import time_benchmark
from .utils import Benchmark

EXAMPLES: list[Benchmark] = [
    Benchmark(vec_mul, (fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(dot_prod_1, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(dot_prod_blocked, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(dot_prod_arm, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(mx_block_round, (fp.FP32,), 1000, (1 << 16)),
    # Benchmark(mx_dot_prod, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(mx_matmul, (fp.FP32, fp.FP32), 10, 256)
]

# Global option combinations: (elim_round, allow_exact)
OPTIONS: list[OptOptions] = [
    OptOptions(elim_round=False, allow_exact=False),  # baseline
    OptOptions(elim_round=True, allow_exact=False),   # elim_round only
    # OptOptions(elim_round=False, allow_exact=True),   # allow_exact only
    OptOptions(elim_round=True, allow_exact=True)     # both
]

def run_eval(config: EvalConfig, examples: list[Benchmark]) -> None:
    # if not replotting, run benchmarks and save times to file for potential replotting
    if not config.replot:
        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)
        print(f'Evaluation output directory: `{config.output_dir}`')

        # build list of benchmarks to run
        bench_configs: list[CompileConfig] = []
        for example in examples:
            for option in OPTIONS:
                bench_configs.append(CompileConfig(
                    benchmark=example,
                    eval_config=config,
                    opt_options=option
                ))

        results: dict[tuple[str, OptOptions], list[float]] = {}
        for i, bench_config in enumerate(bench_configs):
            times = time_benchmark(bench_config, i)
            key = (bench_config.benchmark.func.name, bench_config.opt_options)
            results[key] = times

        # Save times to file for potential replotting
        times_file = config.output_dir / 'times.pkl.gz'
        with gzip.open(times_file, 'wb') as f:
            pickle.dump(results, f)

        print(f'Saved benchmark times to `{times_file}` for potential replotting.')

    # if replotting, load times from file and reconstruct
    else:
        times_file = config.output_dir / 'times.pkl.gz'
        if not times_file.exists():
            print(f'Error: Times file `{times_file}` not found for replotting.')
            return

        with gzip.open(times_file, 'rb') as f:
            results: dict[tuple[str, OptOptions], list[float]] = pickle.load(f)

    # Generate plot
    plot_times(results, OPTIONS, config.output_dir)

if __name__ == '__main__':
    parser = ArgumentParser(description='FPy/MPFX evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for input generation (default: 1)')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations for each benchmark (default: 1)')
    parser.add_argument('--replot', action='store_true', help='Whether to regenerate plots from existing benchmark data')
    parser.add_argument('output_dir', type=Path, help='Output directory for results')
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    seed: int = args.seed
    num_iterations: int = args.iterations
    replot: bool = args.replot

    # Eval configuration
    config = EvalConfig(
        output_dir=output_dir,
        num_iterations=num_iterations,
        seed=seed,
        replot=replot
    )

    # Run evaluation harness
    run_eval(config, EXAMPLES)
    