from enum import IntEnum
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

class EvalMode(IntEnum):
    REAL = 0
    FUN_PROFILE = 1
    EXPR_PROFILE= 2

    @staticmethod
    def from_str(s: str):
        match s:
            case 'real':
                return EvalMode.REAL
            case 'function':
                return EvalMode.FUN_PROFILE
            case 'expr':
                return EvalMode.EXPR_PROFILE
            case _:
                raise ValueError(f"invalid mode: {s}")


@dataclass
class Config:
    mode: EvalMode
    input_paths: list[Path]
    num_samples: int
    seed: Optional[int]
