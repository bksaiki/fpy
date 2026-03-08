import fpy2 as fp
import pickle, gzip

from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed

from .examples import *
from .options import CompileConfig, EvalConfig, OptOptions, WorkerTask
from .plot import plot_times, plot_speedup
from .time import time_benchmark
from .utils import Benchmark

EXAMPLES: list[Benchmark] = [
    Benchmark(talk_example, (fp.FP32, fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(vec_add_fp8, (fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(vec_mul_fp8, (fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(dot_prod_mp, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(dot_prod_blocked, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(dot_prod_arm, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(mx_quantize_blocks, (fp.FP32,), 1000, (1 << 16)),
    # Benchmark(mx_quantize_dot_prod, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(mx_matmul, (fp.FP32, fp.FP32), 10, 256),
    # Benchmark(run_rk4_lorenz_3d, (fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(fastblur_example, (fp.FP16, fp.FP16, fp.FP16, fp.FP16), 30, 256)
]

# Global option combinations: (elim_round, allow_exact)
OPTIONS: list[OptOptions] = [
    OptOptions(elim_round=False, allow_exact=False),  # baseline
    OptOptions(elim_round=True, allow_exact=False),   # elim_round only
    # OptOptions(elim_round=False, allow_exact=True),   # allow_exact only
    OptOptions(elim_round=True, allow_exact=True)     # both
]

def _run_benchmark_worker(task: WorkerTask) -> tuple[str, OptOptions, list[float]]:
    """Worker function for parallel benchmark execution. Takes picklable task data."""
    # Reconstruct the config from picklable data
    example = EXAMPLES[task.benchmark_idx]
    bench_config = CompileConfig(
        benchmark=example,
        eval_config=task.eval_config,
        opt_options=task.opt_options
    )
    
    # Run the benchmark
    times = time_benchmark(bench_config, task.job_idx)
    return (example.func.name, task.opt_options, times)

def run_eval(config: EvalConfig, examples: list[Benchmark]) -> None:
    # file used to store benchmark times for potential replotting
    times_file = config.output_dir / 'times.pkl.gz'

    # if not replotting, run benchmarks and save times to file for potential replotting
    if config.replot:
        if not times_file.exists():
            print(f'Error: Times file `{times_file}` not found for replotting.')
            return

        with gzip.open(times_file, 'rb') as f:
            results: dict[tuple[str, OptOptions], list[float]] = pickle.load(f)
    else:
        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)
        print(f'Evaluation output directory: `{config.output_dir}`')

        # Build list of benchmark tasks
        tasks = []
        job_idx = 0
        for bench_idx, _ in enumerate(examples):
            for option in OPTIONS:
                tasks.append(WorkerTask(
                    job_idx=job_idx,
                    benchmark_idx=bench_idx,
                    opt_options=option,
                    eval_config=config
                ))
                job_idx += 1

        # Run benchmarks in parallel using multiprocessing
        results = {}
        if config.num_threads > 1:
            print(f'Running benchmarks in parallel with {config.num_threads} threads...')
            with ProcessPoolExecutor(max_workers=config.num_threads) as executor:
                # Submit all benchmark tasks
                futures = {
                    executor.submit(_run_benchmark_worker, task): task
                    for task in tasks
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        bench_name, opt_options, times = future.result()
                        key = (bench_name, opt_options)
                        results[key] = times
                    except Exception as exc:
                        print(f'Benchmark job {task.job_idx} generated an exception: {exc}')
                        raise exc
        else:
            print('Running benchmarks sequentially...')
            for task in tasks:
                bench_name, opt_options, times = _run_benchmark_worker(task)
                key = (bench_name, opt_options)
                results[key] = times

        # Save times to file for potential replotting
        with gzip.open(times_file, 'wb') as f:
            pickle.dump(results, f)
        print(f'Saved benchmark times to `{times_file}` for potential replotting.')

    # Generate plots
    plot_times(results, OPTIONS, config.output_dir)
    plot_speedup(results, OPTIONS, config.output_dir)


if __name__ == '__main__':
    parser = ArgumentParser(description='FPy/MPFX evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for input generation (default: 1)')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations for each benchmark (default: 1)')
    parser.add_argument('--threads', type=int, default=1, help='Number of parallel threads for benchmarking (default: 1)')
    parser.add_argument('--replot', action='store_true', help='Whether to regenerate plots from existing benchmark data')
    parser.add_argument('output_dir', type=Path, help='Output directory for results')
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    seed: int = args.seed
    num_iterations: int = args.iterations
    num_threads: int = args.threads
    replot: bool = args.replot

    # Eval configuration
    config = EvalConfig(
        output_dir=output_dir,
        num_iterations=num_iterations,
        num_threads=num_threads,
        seed=seed,
        replot=replot
    )

    # Run evaluation harness
    run_eval(config, EXAMPLES)
    