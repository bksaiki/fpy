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
    mpfx_root: Path | None

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
class CompileTask:
    """Picklable task data for compilation workers."""
    job_idx: int
    benchmark_idx: int
    opt_options: OptOptions
    eval_config: EvalConfig

@dataclass(frozen=True)
class ExecutionTask:
    """Picklable task data for execution workers."""
    job_idx: int
    benchmark_name: str
    opt_options: OptOptions
    binary_path: Path
    num_inputs: int
    vector_size: int
    seed: int
    iteration_num: int
