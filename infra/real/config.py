from enum import IntEnum
from dataclasses import dataclass
from pathlib import Path

class EvalMode(IntEnum):
    REAL = 0
    FUN_PROFILE = 1

    @staticmethod
    def from_str(s: str):
        match s:
            case 'real':
                return EvalMode.REAL
            case 'function':
                return EvalMode.FUN_PROFILE
            case _:
                raise ValueError(f"invalid mode: {s}")


@dataclass
class Config:
    mode: EvalMode
    input_paths: list[Path]
    num_samples: int
