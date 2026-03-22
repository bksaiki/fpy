import fpy2 as fp

from dataclasses import dataclass
from typing import TypeAlias

Input: TypeAlias = tuple[list[fp.Float], list[fp.Float]]

@dataclass(frozen=True)
class Sample:
    inputs: list[Input]
    ref_vals: list[fp.Float]

@dataclass(frozen=True)
class SampleKey:
    seed: int
    n: int

@dataclass(frozen=True)
class Config:
    n: int
    prec: int

@dataclass(frozen=True)
class Result:
    log_rel_errs: list[float]
