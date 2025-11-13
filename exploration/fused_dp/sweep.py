"""
fused-dp/sweep.py: Performs parameter sweep of FPy model.

This model is based on "Optimized Fused Floating-Point Many-Term Dot-Product
Hardware for Machine Learning Accelerators" (Kaul et al., 2019).
"""

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

Input: TypeAlias = tuple[list[fp.Float], list[fp.Float]]


@dataclass(frozen=True)
class Sample:
    inputs: list[Input]
    ref_vals: list[fp.Float]

@dataclass(frozen=True)
class SampleKey:
    seed: int
    n: int

@dataclass(frozen=True)
class Config:
    n: int
    prec: int

@dataclass(frozen=True)
class Result:
    log_rel_errs: list[float]


class Explorer(fp.Runner[Config, SampleKey, Result]):
    """
    Design-space explorer for fused dot-product implementation.
    """

    precs: list[int]
    ns: list[int]
    num_inputs: int
    ictx: fp.EFloatContext
    octx: fp.EFloatContext
    method: str

    def __init__(
        self,
        precs: list[int],
        ns: list[int],
        num_inputs: int,
        ictx: fp.EFloatContext,
        octx: fp.EFloatContext,
        *,
        method: str = 'repr',
        logging: bool = False
    ):
        super().__init__(logging=logging)
        self.precs = precs
        self.ns = ns
        self.num_inputs = num_inputs
        self.ictx = ictx
        self.octx = octx
        self.method = method

    def configs(self):
        configs: list[Config] = []
        for n in self.ns:
            for prec in self.precs:
                configs.append(Config(n, prec))
        return configs

    def sample_key(self, config, seed):
        return SampleKey(seed, config.n)

    def _sample_repr(self, ctx: fp.EFloatContext, expmin: int = -(2 ** 5), expmax: int = 2 ** 5):
        """
        Samples floats from the representable numbers in `ctx`
        with exponent in [expmin, expmax].
        """
        while True:
            i = random.randint(0, 2 ** ctx.nbits - 1)
            x = ctx.decode(i)
            if x.is_finite() and (x == 0 or (expmin <= x.exp <= expmax)):
                return x

    def _sample_uniform(self, ctx: fp.EFloatContext, a: float = -1.0, b: float = 1.0):
        """
        Samples double-precision floats uniformly from [a, b]
        and rounds under `ctx`.
        """
        x = random.uniform(a, b)
        return ctx.round(x)

    def _sample(self, k: int, n: int, ctx: fp.EFloatContext, *, seed: int | None = None, method: str = 'uniform'):
        """
        Samples dot products inputs rounded under `ctx`
        where `k` is the number of samples and `n` is the vector length.
        """
        if seed is not None:
            random.seed(seed)

        if method == 'uniform':
            gen_fn = lambda ctx: self._sample_uniform(ctx)
        elif method == 'repr':
            gen_fn = lambda ctx: self._sample_repr(ctx)
        else:
            raise ValueError(f'Unexpected method: {method}')

        inputs: list[Input] = []
        for _ in range(k):
            xs: list[fp.Float] = []
            ys: list[fp.Float] = []
            for _ in range(n):
                xs.append(gen_fn(ctx))
                ys.append(gen_fn(ctx))
            inputs.append((xs, ys))

        return inputs


    def _run_ref(self, sample: list[Input]):
        ref_vals: list[fp.Float] = []
        for xs, ys in sample:
            ref_val = fp.libraries.vector.dot(xs, ys, ctx=fp.REAL)
            ref_vals.append(self._force_float(ref_val))
        return ref_vals

    def sample(self, key: SampleKey, output_dir: Path, seed: int, no_cache: bool):
        self.log('_generate_sample', f'sampling for `{key.n}`')

        # create cache directory
        cache_dir = self._open_cache(output_dir)

        # look for existing sample
        sample_file = cache_dir / f'sample_n{key.n}.pkl.gz'
        if not no_cache and sample_file.exists():
            # load from cache
            self.log('sample', f'found cached N={key.n} sample at `{sample_file}`')
            cache: Sample = self._read_cache(sample_file)
            # check sample size
            if len(cache.inputs) >= self.num_inputs:
                return sample_file

        # generate the sample
        sample = self._sample(num_inputs, key.n, self.ictx, seed=seed, method=self.method)

        # compute reference values / condition numbers
        ref_vals = self._run_ref(sample)

        # create cached sample
        cache = Sample(sample, ref_vals)

        # Create temporary file for this sample
        sample_file = cache_dir / f'sample_n{key.n}.pkl.gz'
        self._write_cache(sample_file, cache)

        self.log('sample', f'saved N={key.n} sample to `{sample_file}`')
        return sample_file

    def _force_float(self, x: fp.Real):
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

    def run_one(self, task):
        # load cached sample
        self.log('run_one', f'loading sample from `{task.sample}`')
        sample: Sample = self._read_cache(task.sample)
        if len(sample.inputs) < self.num_inputs:
            raise ValueError(f'Insufficient sample size: got {len(sample.inputs)}, expected at least {self.num_inputs}')
        elif len(sample.inputs) > self.num_inputs:
            inputs = sample.inputs[:self.num_inputs]
            ref_vals = sample.ref_vals[:self.num_inputs]
        else:
            inputs = sample.inputs
            ref_vals = sample.ref_vals

        # fractional BNA precision
        config = task.config
        p = config.prec - 1

        # compute relative error
        log_rel_errs: list[float] = []
        for (xs, ys), ref_val in zip(inputs, ref_vals):
            impl_val = self._force_float(dot_prod_impl(xs, ys, p, ctx=self.octx))
            rel_err = fp.libraries.metrics.relative_error(impl_val, ref_val, ctx=fp.FP64)
            log_rel_err = math.log2(float(rel_err)) if rel_err > 0 else float('-inf')
            log_rel_errs.append(log_rel_err)

        # result
        return Result(log_rel_errs)


    def _setup_plots(self, precs: list[int], ns: list[int]) -> tuple[plt.Figure, list[list[plt.Axes]]]:
        num_rows = len(ns)
        num_cols = len(precs)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
        if num_rows == 1:
            axes = [axes]
        if num_cols == 1:
            axes = [[ax] for ax in axes]
        return fig, axes

    def plot(self, configs, results, output_dir, seed):
        LOG_ERR_MIN = -32
        LOG_ERR_MAX = 0

        # recover unique ns and precs
        ns = sorted(set(config.n for config in configs))
        precs = sorted(set(config.prec for config in configs))

        # histogram of relative error
        # 1. calculate maximum frequency for consistent x-axis scaling
        max_freq = 0
        for config in configs:
            result = results[config]
            counts, _, _ = plt.hist(result.log_rel_errs, bins=30, range=(LOG_ERR_MIN, LOG_ERR_MAX))
            max_freq = max(max_freq, max(counts))
            plt.close()

        # 2. create plot
        fig, axes = self._setup_plots(precs, ns)
        for row, n in enumerate(ns):
            for col, prec in enumerate(precs):
                config = Config(n, prec)
                ax = axes[row][col]
                result = results[config]

                if col == 0:
                    ax.set_ylabel(f'N={n}', fontsize=20)
                if row == 0:
                    ax.set_title(f'P={prec}', fontsize=20)

                # ax.set_title(f'N={n}, P={prec}', fontsize=14)
                ax.hist(result.log_rel_errs, bins=30, range=(LOG_ERR_MIN, LOG_ERR_MAX), alpha=0.7, orientation='horizontal')
                ax.set_xlim(0, max_freq)  # Set consistent x-axis limits for frequency
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=14)

        # fig.supxlabel('Frequency', fontsize=16)
        # fig.supylabel('log2(relative error)', fontsize=16)
        fig.tight_layout()
        fig.savefig(output_dir / 'error_histogram.png', dpi=120)


if __name__ == "__main__":
    PRECS = [16, 20, 24, 28, 32, 36, 40, 44]
    NS = [8, 16, 32, 64]
    ICTX = fp.FP16
    OCTX = fp.FP32
    METHOD = 'repr'

    parser = ArgumentParser()
    parser.add_argument('-o', '--output', type=Path, default='out', help='output directory for results')
    parser.add_argument('--threads', type=int, default=1, help='number of threads to use')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('num_inputs', type=int, help='number of input values to use')
    parser.add_argument('--replot', action='store_true', help='replot from existing results without rerunning experiments')
    args = parser.parse_args()

    output_dir: Path = args.output.resolve()
    threads: int = args.threads
    seed: int = args.seed
    num_inputs: int = args.num_inputs
    replot: bool = args.replot

    explorer = Explorer(PRECS, NS, num_inputs, ICTX, OCTX, method=METHOD, logging=True)
    explorer.run(output_dir, seed=seed, num_threads=threads, replot=replot)
