"""
fused-dp/sweep.py: Performs parameter sweep of FPy model.

This model is based on "Optimized Fused Floating-Point Many-Term Dot-Product
Hardware for Machine Learning Accelerators" (Kaul et al., 2019).
"""

import concurrent.futures
import fpy2 as fp
import gzip
import matplotlib.pyplot as plt
import math
import pickle
import random

from argparse import ArgumentParser
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TypeAlias

from .model import dot_prod_impl

Sample: TypeAlias = list[tuple[list[fp.Float], list[fp.Float]]]


@dataclass(frozen=True)
class CachedSample:
    inputs: Sample
    ref_vals: list[fp.Float]

@dataclass(frozen=True)
class WorkerKey:
    n: int
    prec: int

@dataclass(frozen=True)
class WorkerConfig:
    idx: int
    n: int
    prec: int
    octx: fp.EFloatContext
    cache_dir: Path
    sample_file: Path

    def key(self):
        return WorkerKey(self.n, self.prec)

@dataclass(frozen=True)
class WorkerResult:
    log_rel_errs: list[float]


def _log(where: str, msg: str):
    print(f'[{where}] {msg}')

def _force_float(x: fp.Real):
    match x:
        case fp.Float():
            return x
        case Fraction():
            return fp.Float.from_rational(x)
        case int():
            return fp.Float.from_int(x)
        case float():
            return fp.Float.from_float(x)
        case _:
            raise TypeError(f'Unexpected: x={x}')


def _sample_repr(ctx: fp.EFloatContext, expmin: int = -(2 ** 5), expmax: int = 2 ** 5):
    """
    Samples floats from the representable numbers in `ctx`
    with exponent in [expmin, expmax].
    """
    while True:
        i = random.randint(0, 2 ** ctx.nbits - 1)
        x = ctx.decode(i)
        if x.is_finite() and (x == 0 or (expmin <= x.exp <= expmax)):
            return x

def _sample_uniform(ctx: fp.EFloatContext, a: float = -1.0, b: float = 1.0):
    """
    Samples double-precision floats uniformly from [a, b]
    and rounds under `ctx`.
    """
    x = random.uniform(a, b)
    return ctx.round(x)

def _sample(k: int, n: int, ctx: fp.EFloatContext, *, seed: int | None = None, method: str = 'uniform'):
    """
    Samples dot products inputs rounded under `ctx`
    where `k` is the number of samples and `n` is the vector length.
    """
    if seed is not None:
        random.seed(seed)

    if method == 'uniform':
        gen_fn = lambda ctx: _sample_uniform(ctx)
    elif method == 'repr':
        gen_fn = lambda ctx: _sample_repr(ctx)
    else:
        raise ValueError(f'Unexpected method: {method}')

    inputs: Sample = []
    for _ in range(k):
        xs: list[fp.Float] = []
        ys: list[fp.Float] = []
        for _ in range(n):
            xs.append(gen_fn(ctx))
            ys.append(gen_fn(ctx))
        inputs.append((xs, ys))

    return inputs


def _run_ref(sample: Sample):
    ref_vals: list[fp.Float] = []
    for xs, ys in sample:
        ref_val = fp.libraries.vector.dot(xs, ys, ctx=fp.REAL)
        ref_vals.append(_force_float(ref_val))
    return ref_vals

def _generate_sample(n: int, num_inputs: int, ictx: fp.EFloatContext, cache_dir: Path, seed: int, method: str):
    _log('_generate_sample', f'sampling for N={n}')

    # generate the sample
    sample = _sample(num_inputs, n, ictx, seed=seed, method=method)

    # compute reference values / condition numbers
    ref_vals = _run_ref(sample)

    # create cached sample
    cache = CachedSample(sample, ref_vals)

    # Create temporary file for this sample
    sample_file = cache_dir / f'sample_n{n}.pkl.gz'
    with gzip.open(sample_file, 'wb') as f:
        pickle.dump(cache, f)

    _log('_generate_samples', f'saved N={n} sample to `{sample_file}`')
    return sample_file


def _run_one(config: WorkerConfig):
    _log('_run_one', f'running {config.idx} (N={config.n}, P={config.prec})')

    # load cached sample
    _log('_run_one', f'loading sample from `{config.sample_file}`')
    with gzip.open(config.sample_file, 'rb') as f:
        cache: CachedSample = pickle.load(f)

    # fractional BNA precision
    p = config.prec - 1

    # compute relative error
    log_rel_errs: list[float] = []
    for (xs, ys), ref_val in zip(cache.inputs, cache.ref_vals):
        impl_val = _force_float(dot_prod_impl(xs, ys, p, ctx=config.octx))
        rel_err = fp.libraries.metrics.relative_error(impl_val, ref_val, ctx=fp.FP64)
        log_rel_err = math.log2(float(rel_err)) if rel_err > 0 else float('-inf')
        log_rel_errs.append(log_rel_err)

    # result
    _log('_run_one', f'completed {config.idx} (N={config.n}, P={config.prec})')
    result = WorkerResult(log_rel_errs)

    # cache result
    result_file = config.cache_dir / f'result_n{config.n}_p{config.prec}.pkl.gz'
    with gzip.open(result_file, 'wb') as f:
        pickle.dump(result, f)

    del cache
    _log('_run_one', f'saved result to `{result_file}`')
    return config.key(), result_file


def _setup_plots(precs: list[int], ns: list[int]) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    num_rows = len(ns)
    num_cols = len(precs)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    if num_rows == 1:
        axes = [axes]
    if num_cols == 1:
        axes = [[ax] for ax in axes]
    return fig, axes

def _plot(
    output_dir: Path,
    precs: list[int],
    ns: list[int],
    results: dict[WorkerKey, WorkerResult]
):
    print('Creating plots...')
    LOG_ERR_MIN = -32
    LOG_ERR_MAX = 0

    # histogram of relative error
    # 1. calculate maximum frequency for consistent x-axis scaling
    max_freq = 0
    for n in ns:
        for prec in precs:
            result = results[WorkerKey(n, prec)]
            counts, _, _ = plt.hist(result.log_rel_errs, bins=30, range=(LOG_ERR_MIN, LOG_ERR_MAX))
            max_freq = max(max_freq, max(counts))
            plt.close()

    # 2. create plot
    fig, axes = _setup_plots(precs, ns)
    for row, n in enumerate(ns):
        for col, prec in enumerate(precs):
            ax = axes[row][col]
            result = results[WorkerKey(n, prec)]

            ax.set_title(f'N={n}, P={prec}')
            ax.hist(result.log_rel_errs, bins=30, range=(LOG_ERR_MIN, LOG_ERR_MAX), alpha=0.7, orientation='horizontal')
            ax.set_ylabel('log2(relative error)')
            ax.set_xlabel('Frequency')
            ax.set_xlim(0, max_freq)  # Set consistent x-axis limits for frequency
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'error_histogram.png', dpi=120)


def _run(
    precs: list[int],
    ns: list[int],
    num_inputs: int,
    seed: int,
    ictx: fp.EFloatContext,
    octx: fp.EFloatContext,
    output_dir: Path,
    *,
    method: str = 'repr',
    threads: int = 1
):
    # create cache directory
    cache_dir = output_dir / 'cache'
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)

    # build worker configs
    configs: list[WorkerConfig] = []
    for n in ns:
        sample_path = _generate_sample(n, num_inputs, ictx, cache_dir, seed, method)
        for prec in precs:
            idx = len(configs)
            config = WorkerConfig(idx, n, prec, octx, cache_dir, sample_path)
            configs.append(config)

    # run evaluation
    result_paths: dict[WorkerKey, Path] = {}
    if threads > 1 and len(configs) > 1:
        # use concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            futures = { executor.submit(_run_one, config): config for config in configs }
            for future in concurrent.futures.as_completed(futures):
                try:
                    key, path = future.result()
                    result_paths[key] = path
                except Exception as e:
                    config = futures[future]
                    _log('_run', f'Error in config idx={config.idx} (N={config.n}, P={config.prec}): {e}')
    else:
        # single-threaded
        _log('_run', f'Running {len(configs)} configs in single-threaded mode')
        for config in configs:
            key, path = _run_one(config)
            result_paths[key] = path

    # load results from cache
    results: dict[WorkerKey, WorkerResult] = {}
    for key, path in result_paths.items():
        with gzip.open(path, 'rb') as f:
            result: WorkerResult = pickle.load(f)
            results[key] = result

    # plot results
    _plot(output_dir, precs, ns, results)


if __name__ == "__main__":
    PRECS = [16, 20, 24, 32, 36, 40, 44]
    NS = [8, 16, 32, 64]
    ICTX = fp.FP16
    OCTX = fp.FP32
    METHOD = 'repr'

    parser = ArgumentParser()
    parser.add_argument('-o', '--output', type=Path, default='out', help='output directory for results')
    parser.add_argument('--threads', type=int, default=1, help='number of threads to use')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('num_inputs', type=int, help='number of input values to use')
    args = parser.parse_args()

    output_dir: Path = args.output.resolve()
    threads: int = args.threads
    seed: int = args.seed
    num_inputs: int = args.num_inputs

    _run(PRECS, NS, num_inputs, seed, ICTX, OCTX, output_dir, method=METHOD, threads=threads)
