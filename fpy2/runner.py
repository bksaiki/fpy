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

P = TypeVar("P") # Params type
S = TypeVar("S") # Sample type
K = TypeVar("K") # Sample key type
R = TypeVar("R") # Result type


@dataclass(frozen=True)
class RunnerConfig(Generic[P, K]):
    """Data class representing a `Runner` configuration."""
    params: P
    """user-defined parameters for this configuration"""
    key: K
    """sample key for this configuration"""


@dataclass(frozen=True)
class RunnerWorkerTask(Generic[P, K]):
    """
    Data class representing a worker configuration for a `Runner`.
    """
    config: RunnerConfig[P, K]
    """the runner configuration"""
    output_dir: Path
    """the output directory for this worker"""
    seed: int
    """the random seed for this worker"""
    idx: int
    """the index of this worker"""


@dataclass(frozen=True)
class RunnerSample(Generic[S, K]):
    """
    Data class representing a sample for a `Runner`.
    """
    sample: S
    key: K


class Runner(ABC, Generic[P, K, S, R]):
    """
    Abstract base class defining a design-space explorer.

    Generics:
    - P: Params type.
    - K: Sample key type.
    - S: Sample type.
    - R: Result type.

    Subclasses must implement the following methods:
    - configs: to generate a list of `RunnerConfig` instances.
    - run_config: to run a single configuration.
    - plot: to visualize the results.
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
    def configs(self) -> list[RunnerConfig[P, K]]:
        """
        Returns a list of `RunnerConfig` instances for
        design-space exploration.
        """
        ...

    @abstractmethod
    def sample(self, key: K, output_dir: Path, no_cache: bool) -> S:
        """
        Generates a `RunnerSample` from a given key.

        Parameters:
        - key: The key for the sample.
        - output_dir: The output directory for the sample.
        - no_cache: If True, disables caching of the sample.

        Returns:
        - A `RunnerSample` instance.
        """
        ...

    @abstractmethod
    def run_one(self, task: RunnerWorkerTask[P, K]) -> R:
        """
        Runs a single configuration.

        Parameters:
        - task: The `RunnerWorkerTask` instance to run.
        """
        ...

    @abstractmethod
    def plot(
        self,
        configs: list[RunnerConfig[P, K]],
        results: dict[RunnerConfig[P, K], R],
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
            configs: list[RunnerConfig[P, K]] = cached[0]
            results: dict[RunnerConfig[P, K], R] = cached[1]
        else:
            # generate configurations
            configs = self.configs()
            self.log('run', f'generated {len(configs)} configurations')

            # generate samples
            samples: dict[K, S] = {}
            for config in configs:
                if config.key not in samples:
                    samples[config.key] = self.sample(config.key, output_dir, no_cache)
            self.log('run', f'generated {len(samples)} unique samples')

            # create worker configurations
            tasks: list[RunnerWorkerTask[P, K]] = [
                RunnerWorkerTask(config, output_dir, seed, idx)
                for idx, config in enumerate(configs)
            ]

            # run workers
            results = {}
            if num_threads > 1 and len(tasks) > 1:
                # run with multiple processes
                self.log('run', f'running {len(tasks)} configs with {num_threads} threads')
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
                    futures = { executor.submit(self.run_one, task): task for task in tasks }
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
                    r = self.run_one(task)
                    results[task.config] = r

            # save results to cache
            self.log('run', f'saving results to cache')
            self._write_cache(cache_file, (configs, results))

        # plot results
        self.log('run', 'plotting results')
        self.plot(configs, results, output_dir, seed)

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
