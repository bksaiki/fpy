from dataclasses import dataclass
from pathlib import Path

from .utils import Benchmark

@dataclass(frozen=True)
class EvalConfig:
    output_dir: Path
    num_iterations: int
    num_threads: int
    seed: int
    replot: bool

@dataclass(frozen=True)
class OptOptions:
    elim_round: bool
    allow_exact: bool

@dataclass(frozen=True)
class CompileConfig:
    benchmark: Benchmark
    eval_config: EvalConfig
    opt_options: OptOptions

@dataclass(frozen=True)
class WorkerTask:
    """Picklable task data for multiprocessing workers."""
    job_idx: int
    benchmark_idx: int
    opt_options: OptOptions
    eval_config: EvalConfig
