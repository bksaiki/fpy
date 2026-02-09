from dataclasses import dataclass
from pathlib import Path

from .utils import Benchmark

@dataclass(frozen=True)
class EvalConfig:
    output_dir: Path
    num_iterations: int
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
