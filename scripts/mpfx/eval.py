import fpy2 as fp
import pickle, gzip

from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed

from .examples import *
from .options import CompileConfig, EvalConfig, OptOptions, CompileTask, ExecutionTask
from .plot import plot_speedup, plot_speedup_bar
from .time import compile_benchmark_task, run_compiled_benchmark
from .utils import Benchmark

EXAMPLES: list[Benchmark] = [
    Benchmark(talk_example, (fp.FP32, fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(vec_add_fp8, (fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(vec_mul_fp8, (fp.FP32, fp.FP32), 1000, (1 << 16)),
    Benchmark(dot_prod_mp, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    Benchmark(dot_prod_blocked, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    Benchmark(dot_prod_arm, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    # Benchmark(mx_quantize_blocks, (fp.FP32,), 1000, (1 << 16)),
    Benchmark(mx_quantize_dot_prod, (fp.FP32, fp.FP32, fp.FP32), 1000, (1 << 16)),
    Benchmark(mx_matmul, (fp.FP32, fp.FP32), 10, 256),
    Benchmark(run_rk4_lorenz_3d, (fp.FP32, fp.FP32), 1000, (1 << 16)),
    Benchmark(fastblur_example, (fp.FP16, fp.FP16, fp.FP16, fp.FP16), 30, 256)
]

# Global option combinations: (elim_round, allow_exact)
OPTIONS: list[OptOptions] = [
    OptOptions(elim_round=False, allow_exact=False),  # baseline
    OptOptions(elim_round=True, allow_exact=False),   # elim_round only
    # OptOptions(elim_round=False, allow_exact=True),   # allow_exact only
    OptOptions(elim_round=True, allow_exact=True)     # both
]

def _compile_worker(task: CompileTask) -> tuple[str, OptOptions, Path, int, int]:
    """Worker function for parallel compilation. Returns compiled binary info."""
    # Reconstruct the config from picklable data
    example = EXAMPLES[task.benchmark_idx]
    bench_config = CompileConfig(
        benchmark=example,
        eval_config=task.eval_config,
        opt_options=task.opt_options
    )

    # Compile the benchmark
    binary_path = compile_benchmark_task(bench_config, task.job_idx)
    vector_size = 1 if example.vector_size is None else example.vector_size

    return (example.func.name, task.opt_options, binary_path, example.num_inputs, vector_size)


def _run_worker(task: ExecutionTask) -> tuple[str, OptOptions, float]:
    """Worker function for parallel execution. Returns single timing result."""
    # Run the compiled benchmark once
    time = run_compiled_benchmark(
        binary_path=task.binary_path,
        num_inputs=task.num_inputs,
        vector_size=task.vector_size,
        seed=task.seed,
        benchmark_name=task.benchmark_name,
        iteration_num=task.iteration_num
    )

    return (task.benchmark_name, task.opt_options, time)

def _load_cached_results(times_file: Path) -> dict[tuple[str, OptOptions], list[float]] | None:
    """Load cached benchmark results from file. Returns None if file doesn't exist."""
    if not times_file.exists():
        print(f'Error: Times file `{times_file}` not found for replotting.')
        return None

    with gzip.open(times_file, 'rb') as f:
        return pickle.load(f)


def _build_compile_tasks(config: EvalConfig, examples: list[Benchmark]) -> list[CompileTask]:
    """Build list of compilation tasks from examples and options."""
    tasks = []
    job_idx = 0
    for bench_idx, _ in enumerate(examples):
        for option in OPTIONS:
            tasks.append(CompileTask(
                job_idx=job_idx,
                benchmark_idx=bench_idx,
                opt_options=option,
                eval_config=config
            ))
            job_idx += 1
    return tasks


def _run_compilation_phase(
    compile_tasks: list[CompileTask],
    num_threads: int
) -> list[tuple[str, OptOptions, Path, int, int]]:
    """Execute compilation phase and return compiled benchmark info."""
    print('\n=== Phase 1: Compilation ===')
    compiled_benchmarks: list[tuple[str, OptOptions, Path, int, int]] = []

    if num_threads > 1:
        print(f'Compiling benchmarks in parallel with {num_threads} threads...')
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(_compile_worker, task): task
                for task in compile_tasks
            }

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    compiled_benchmarks.append(result)
                    print(f'  Compiled benchmark {task.job_idx + 1}/{len(compile_tasks)}')
                except Exception as exc:
                    print(f'Compilation job {task.job_idx} generated an exception: {exc}')
                    raise exc
    else:
        print('Compiling benchmarks sequentially...')
        for task in compile_tasks:
            result = _compile_worker(task)
            compiled_benchmarks.append(result)

    print(f'\nSuccessfully compiled {len(compiled_benchmarks)} benchmarks.')
    return compiled_benchmarks


def _build_execution_tasks(
    compiled_benchmarks: list[tuple[str, OptOptions, Path, int, int]],
    config: EvalConfig
) -> list[ExecutionTask]:
    """Build list of execution tasks from compiled benchmarks."""
    execution_tasks = []
    for bench_name, opt_options, binary_path, num_inputs, vector_size in compiled_benchmarks:
        for iteration in range(config.num_iterations):
            execution_tasks.append(ExecutionTask(
                job_idx=len(execution_tasks),
                benchmark_name=bench_name,
                opt_options=opt_options,
                binary_path=binary_path,
                num_inputs=num_inputs,
                vector_size=vector_size,
                seed=config.seed,
                iteration_num=iteration + 1
            ))
    return execution_tasks


def _run_execution_phase(
    execution_tasks: list[ExecutionTask],
    num_threads: int,
    num_benchmarks: int,
    num_iterations: int
) -> dict[tuple[str, OptOptions], list[float]]:
    """Execute benchmarks and aggregate results."""
    print('\n=== Phase 2: Execution ===')
    results: dict[tuple[str, OptOptions], list[float]] = {}

    if num_threads > 1:
        print(f'Running benchmarks in parallel with {num_threads} threads...')
        print(f'Total execution tasks: {len(execution_tasks)} ({num_benchmarks} benchmarks × {num_iterations} iterations)')
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(_run_worker, task): task
                for task in execution_tasks
            }

            for future in as_completed(futures):
                task = futures[future]
                try:
                    bench_name, opt_options, time = future.result()
                    key = (bench_name, opt_options)
                    if key not in results:
                        results[key] = []
                    results[key].append(time)
                    print(f'  Completed execution task {task.job_idx + 1}/{len(execution_tasks)}')
                except Exception as exc:
                    print(f'Execution job {task.job_idx} generated an exception: {exc}')
                    raise exc
    else:
        print('Running benchmarks sequentially...')
        for task in execution_tasks:
            bench_name, opt_options, time = _run_worker(task)
            key = (bench_name, opt_options)
            if key not in results:
                results[key] = []
            results[key].append(time)

    print(f'\nSuccessfully executed {len(results)} benchmarks.')
    return results


def _save_results(results: dict[tuple[str, OptOptions], list[float]], times_file: Path) -> None:
    """Save benchmark results to file for potential replotting."""
    with gzip.open(times_file, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved benchmark times to `{times_file}` for potential replotting.')


def _generate_plots(results: dict[tuple[str, OptOptions], list[float]], output_dir: Path) -> None:
    """Generate all plots from benchmark results."""
    print('\n=== Generating Plots ===')
    plot_speedup(results, EXAMPLES, OPTIONS, output_dir)
    plot_speedup_bar(results, EXAMPLES, OPTIONS, output_dir)


def run_eval(config: EvalConfig, examples: list[Benchmark]) -> None:
    """Run complete evaluation pipeline: compile, execute, and plot results."""
    times_file = config.output_dir / 'times.pkl.gz'

    if config.replot:
        # Load cached results and regenerate plots
        results = _load_cached_results(times_file)
        if results is None:
            return
    else:
        # Run full benchmark pipeline
        config.output_dir.mkdir(parents=True, exist_ok=True)
        print(f'Evaluation output directory: `{config.output_dir}`')

        # Phase 1: Compilation
        compile_tasks = _build_compile_tasks(config, examples)
        compiled_benchmarks = _run_compilation_phase(compile_tasks, config.num_threads)

        # Phase 2: Execution
        execution_tasks = _build_execution_tasks(compiled_benchmarks, config)
        results = _run_execution_phase(
            execution_tasks,
            config.num_threads,
            len(compiled_benchmarks),
            config.num_iterations
        )

        # Save results
        _save_results(results, times_file)

    # Generate plots
    _generate_plots(results, config.output_dir)


if __name__ == '__main__':
    parser = ArgumentParser(description='FPy/MPFX evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for input generation (default: 1)')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations for each benchmark (default: 1)')
    parser.add_argument('--threads', type=int, default=1, help='Number of parallel threads for benchmarking (default: 1)')
    parser.add_argument('--replot', action='store_true', help='Whether to regenerate plots from existing benchmark data')
    parser.add_argument('--mpfx-root', type=Path, help='Root directory of MPFX library')
    parser.add_argument('output_dir', type=Path, help='Output directory for results')
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    seed: int = args.seed
    num_iterations: int = args.iterations
    num_threads: int = args.threads
    replot: bool = args.replot
    mpfx_root: Path | None = args.mpfx_root.resolve() if args.mpfx_root else None

    # Eval configuration
    config = EvalConfig(
        output_dir=output_dir,
        num_iterations=num_iterations,
        num_threads=num_threads,
        seed=seed,
        replot=replot,
        mpfx_root=mpfx_root
    )

    # Run evaluation harness
    run_eval(config, EXAMPLES)
