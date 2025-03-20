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

    @staticmethod
    def options():
        return ['real', 'function', 'expr']


class ReferenceMode(IntEnum):
    REAL = 0,
    FLOAT_1K = 1,
    FLOAT_2K = 2,
    FLOAT_4K = 3,

    @staticmethod
    def from_str(s: str):
        match s:
            case 'real':
                return ReferenceMode.REAL
            case '1k':
                return ReferenceMode.FLOAT_1K
            case '2k':
                return ReferenceMode.FLOAT_2K
            case '4k':
                return ReferenceMode.FLOAT_4K
            case _:
                raise ValueError(f"invalid reference mode: {s}")

    @staticmethod
    def options():
        return ['real', '1k', '2k', '4k']

@dataclass
class Config:
    mode: EvalMode
    input_paths: list[Path]
    num_samples: int
    ref_mode: ReferenceMode
    seed: Optional[int]
