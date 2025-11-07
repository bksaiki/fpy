"""
This module defines a running for design-space exploration tasks.
"""

import concurrent.futures
import gzip
import hashlib
import pickle
import random

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

__all__ = [
    'Runner',
    'RunnerWorkerTask',
]


C = TypeVar("C") # Config type

@dataclass(frozen=True)
class RunnerWorkerTask(Generic[C]):
    """
    Data class representing a worker configuration for a `Runner`.
    """
    config: C
    """the runner configuration"""
    sample: Path
    """the path to the sample for this worker"""
    output_dir: Path
    """the output directory for this worker"""
    seed: int
    """the random seed for this worker"""
    idx: int
    """the index of this worker"""


K = TypeVar("K") # Sample key type
R = TypeVar("R") # Result type


class Runner(ABC, Generic[C, K, R]):
    """
    Abstract base class defining a design-space explorer.

    Type Parameters:
    - C: The configuration type.
    - K: The sample key type.
    - R: The result type.
    """

    def __init__(self, logging: bool = False):
        self.logging = logging

    @abstractmethod
    def prefix(self) -> str:
        """
        Returns a prefix string for output files.
        """
        ...

    @abstractmethod
    def configs(self) -> list[C]:
        """
        Returns a list of configurations to explore.
        """
        ...

    @abstractmethod
    def sample_key(self, config: C) -> K:
        """
        Extracts the sample key from a given configuration.

        Parameters:
        - config: The configuration to extract the sample key from.

        Returns:
        - The sample key.
        """
        ...

    @abstractmethod
    def sample(self, key: K, output_dir: Path, seed: int, no_cache: bool) -> Path:
        """
        Generates a `RunnerSample` from a given key.

        Parameters:
        - key: The key for the sample.
        - output_dir: The output directory for the sample.
        - seed: A random seed for reproducibility.
        - no_cache: If True, disables caching of the sample.

        Returns:
        - The path to the generated sample.
        """
        ...

    @abstractmethod
    def run_one(self, task: RunnerWorkerTask[C]) -> R:
        """
        Runs a single configuration.

        Parameters:
        - task: The `RunnerWorkerTask` instance to run.
        """
        ...

    @abstractmethod
    def plot(
        self,
        configs: list[C],
        results: dict[C, R],
        output_dir: Path,
        seed: int
    ):
        """
        Plots the results of the design-space exploration.

        Parameters:
        - configs: The list of `RunnerConfig` instances.
        - results: A dictionary mapping sample keys to results.
        - output_dir: The directory to store output plots.
        - seed: A random seed for reproducibility.
        """
        ...

    def log(self, where: str, *args):
        """
        Logs a message if logging is enabled.
        """
        if self.logging:
            print(f'[Runner.{where}]', *args)

    def run(
        self,
        output_dir: Path, *,
        seed: int = 1,
        num_threads: int = 1,
        no_cache: bool = False,
        replot: bool = False
    ):
        """
        Runs the design-space exploration.

        Parameters:
        - output_dir: The directory to store output results.
        - seed: A random seed for reproducibility.
        - num_threads: The number of threads to use for parallel execution.
        - no_cache: If True, disables caching of samples
        - replot: If True, only replots existing results from cache.
        """

        # resolve output directory and create if not exists
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log('run', f'output directory at `{output_dir}`')

        # cache file
        cache_file = output_dir / f'{self.prefix()}results.pkl.gz'

        if replot:
            # reload configurations and results from cache
            self.log('run', f'reloading results from cache `{cache_file}`')
            cached = self._read_cache(cache_file)
            configs: list[C] = cached[0]
            results: dict[C, R] = cached[1]
        else:
            # generate configurations
            configs = self.configs()
            self.log('run', f'generated {len(configs)} configurations')

            # generate sample keys
            keys: dict[C, K] = { config: self.sample_key(config) for config in configs }

            # generate samples
            samples: dict[K, Path] = {}
            for key in { keys[config] for config in configs }:
                random.seed(seed + hash(key))
                samples[key] = self.sample(key, output_dir, seed, no_cache)
            self.log('run', f'generated {len(samples)} unique samples')

            # create worker configurations
            tasks: list[RunnerWorkerTask[C]] = [
                RunnerWorkerTask(config, samples[keys[config]], output_dir, seed, idx)
                for idx, config in enumerate(configs)
            ]

            # run workers
            results = {}
            if num_threads > 1 and len(tasks) > 1:
                # run with multiple processes
                self.log('run', f'running {len(tasks)} configs with {num_threads} threads')
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
                    futures = { executor.submit(self._run_one, task): task for task in tasks }
                    for future in concurrent.futures.as_completed(futures):
                        task = futures[future]
                        try:
                            r = future.result()
                            results[task.config] = r
                        except Exception as e:
                            self.log('run', f'config {task.config} generated an exception: {e}')
            else:
                # single-threaded mode
                self.log('run', f'running {len(tasks)} configs in single-threaded mode')
                for task in tasks:
                    r = self._run_one(task)
                    results[task.config] = r

            # save results to cache
            self.log('run', 'saving results to cache')
            self._write_cache(cache_file, (configs, results))

        # plot results
        self.log('run', 'plotting results')
        self.plot(configs, results, output_dir, seed)


    def _run_one(self, task: RunnerWorkerTask[C]) -> R:
        """
        Internal method to run a single configuration.
        """
        self.log('_run_one', f'Running config {task.config} (idx={task.idx})')
        result = self.run_one(task)
        self.log('_run_one', f'Completed config {task.config} (idx={task.idx})')
        return result

    def _gen_cache_name(self, key) -> str:
        skey = '_'.join(str(x) for x in key)
        return hashlib.md5(skey.encode()).hexdigest()

    def _write_cache(self, path: Path, data):
        """Writes data to a gzipped cache file."""
        self.log('write_cache', f'writing cache to `{path}`')
        with gzip.open(path, 'wb') as f:
            pickle.dump(data, f)

    def _read_cache(self, path: Path):
        """Reads data from a gzipped cache file."""
        self.log('read_cache', f'reading cache from `{path}`')
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
