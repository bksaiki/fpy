from enum import IntEnum
from dataclasses import dataclass
from pathlib import Path

class EvalMode(IntEnum):
    REAL = 0

    @staticmethod
    def from_str(s: str):
        match s:
            case 'real':
                return EvalMode.REAL
            case _:
                raise ValueError(f"invalid mode: {s}")


@dataclass
class Config:
    mode: EvalMode
    input_paths: list[Path]
