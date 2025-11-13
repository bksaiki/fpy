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
K = TypeVar("K") # Sample key type
R = TypeVar("R") # Result type

@dataclass(frozen=True)
class SampleWorkerTask(Generic[K]):
    """
    Data class representing a worker configuration for sampling.
    """
    key: K
    """the sample key"""
    output_dir: Path
    """the output directory for this worker"""
    seed: int
    """the random seed for this worker"""
    no_cache: bool
    """whether to disable caching for this worker"""
    idx: int
    """the index of this worker"""


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
    def configs(self) -> list[C]:
        """
        Returns a list of configurations to explore.
        """
        ...

    @abstractmethod
    def sample_key(self, config: C, seed: int) -> K:
        """
        Extracts the sample key from a given configuration.

        Parameters:
        - config: The configuration to extract the sample key from.
        - seed: A random seed for reproducibility.

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
        cache_file = output_dir / self._format_cache_name('results.pkl.gz')

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

            # generate sampling tasks
            uniq_keys: set[K] = set()
            key_by_config: dict[C, K] = {}
            sampling_tasks: list[SampleWorkerTask[K]] = []
            for config in configs:
                key = self.sample_key(config, seed)
                key_by_config[config] = key
                if key not in uniq_keys:
                    task = SampleWorkerTask(key, output_dir, seed, no_cache, len(uniq_keys))
                    sampling_tasks.append(task)
                    uniq_keys.add(key)

            # run sampling
            samples = self._run_sampling(sampling_tasks, num_threads, no_cache)

            # generate sweep tasks
            sweep_tasks: list[RunnerWorkerTask[C]] = [
                RunnerWorkerTask(config, samples[key_by_config[config]], output_dir, seed, idx)
                for idx, config in enumerate(configs)
            ]

            # run sweep
            results = self._run_sweep(sweep_tasks, cache_file, num_threads)

        # plot results
        self.log('run', 'plotting results')
        self.plot(configs, results, output_dir, seed)

    def _run_sampling(self, tasks: list[SampleWorkerTask[K]], num_threads: int, no_cache: bool) -> dict[K, Path]:
        """
        Internal method to run a sweep of sampling tasks.
        """
        # run workers
        samples = {}
        if num_threads > 1 and len(tasks) > 1:
            # run with multiple processes
            self.log('run_sampling', f'running {len(tasks)} samples with {num_threads} threads')
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
                futures = { executor.submit(self._sample_one, task): task for task in tasks }
                for future in concurrent.futures.as_completed(futures):
                    task = futures[future]
                    try:
                        path = future.result()
                        samples[task.key] = path
                    except Exception as e:
                        self.log('run_sampling', f'sample {task.key} generated an exception')
                        raise e
        else:
            # single-threaded mode
            self.log('run_sampling', f'running {len(tasks)} samples in single-threaded mode')
            for task in tasks:
                path = self._sample_one(task)
                samples[task.key] = path

        return samples

    def _sample_one(self, task: SampleWorkerTask[K]) -> Path:
        """
        Internal method to run a single sampling task.
        """
        self.log('_sample_one', f'generating sample for key {task.key} (idx={task.idx})')
        sample_path = self.sample(task.key, task.output_dir, task.seed, task.no_cache)
        self.log('_sample_one', f'completed sample for key {task.key} (idx={task.idx})')
        return sample_path

    def _run_sweep(self, tasks: list[RunnerWorkerTask[C]], cache_path: Path, num_threads: int) -> dict[C, R]:
        """
        Internal method to run a sweep of configurations.
        """
        # run workers
        results = {}
        if num_threads > 1 and len(tasks) > 1:
            # run with multiple processes
            self.log('run_sweep', f'running {len(tasks)} configs with {num_threads} threads')
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
                futures = { executor.submit(self._run_one, task): task for task in tasks }
                for future in concurrent.futures.as_completed(futures):
                    task = futures[future]
                    try:
                        r = future.result()
                        results[task.config] = r
                    except Exception as e:
                        self.log('run', f'config {task.config} generated an exception')
                        raise e
        else:
            # single-threaded mode
            self.log('run_sweep', f'running {len(tasks)} configs in single-threaded mode')
            for task in tasks:
                r = self._run_one(task)
                results[task.config] = r

        # save results to cache
        self.log('run_sweep', 'saving results to cache')
        configs = [task.config for task in tasks]
        self._write_cache(cache_path, (configs, results))
        return results

    def _run_one(self, task: RunnerWorkerTask[C]) -> R:
        """
        Internal method to run a single configuration.
        """
        self.log('_run_one', f'running config {task.config} (idx={task.idx})')
        result = self.run_one(task)
        self.log('_run_one', f'completed config {task.config} (idx={task.idx})')
        return result

    def _format_cache_name(self, name: str) -> str:
        """
        Formats a cache file name.

        Override this method to customize cache file naming.
        """
        return name

    def _gen_cache_name(self, key) -> str:
        """
        Generates a cache file name from a key.
        """
        if isinstance(key, (tuple, list)):
            skey = '_'.join(str(x) for x in key)
        else:
            skey = str(key)
        return hashlib.md5(skey.encode()).hexdigest()

    def _open_cache(self, output_dir: Path, cache_name: str = '__cache__'):
        """
        Opens (and creates) a cache directory inside the output directory.
        """
        cache_dir = output_dir / self._format_cache_name(cache_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.log('open_cache', f'opened cache directory `{cache_dir}`')
        return cache_dir

    def _write_cache(self, path: Path, data):
        """Writes data to a gzipped cache file."""
        self.log('write_cache', f'writing cache to `{path}`')
        with gzip.open(path, 'wb') as f:
            pickle.dump(data, f)

    def _read_cache(self, path: Path):
        """Reads data from a gzipped cache file."""
        self.log('read_cache', f'reading cache from `{path}`')
        try:
            with gzip.open(path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, gzip.BadGzipFile, EOFError, FileNotFoundError):
            self.log('read_cache', f'failed to read cache: `{path}`')
            return None
