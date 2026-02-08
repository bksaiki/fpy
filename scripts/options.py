from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class EvalConfig:
    output_dir: Path
    seed: int

@dataclass(frozen=True)
class CompileConfig:
    elim_round: bool
    allow_exact: bool
