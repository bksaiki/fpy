"""
Input sources: where a design's arguments come from.

A source is chosen by name (`SOURCES`), so a workload -- operands taken from a
real GEMM -- is one more class here, not a change to the driver.
"""

import random
from collections.abc import Iterator
from typing import Any, Protocol

from compile import _HARD, _HARD_EVERY, _row

from fpy2.types import Type


class InputSource(Protocol):
    name: str

    def inputs(self, arg_types: list[Type], n: int, seed: int) -> Iterator[list[Any]]:
        """*n* argument lists for a design taking *arg_types*."""
        ...


class RandomSource:
    """`compile.py`'s draws: over each format's whole range or near one, and
    every `_HARD_EVERY`-th draw heavy in hard cases (zeros, extremes,
    subnormals, specials).  The same seed gives `compile.py -r`'s inputs."""

    name = 'random'

    def inputs(self, arg_types: list[Type], n: int, seed: int) -> Iterator[list[Any]]:
        rng = random.Random(seed)
        for i in range(n):
            hard = _HARD if (i + 1) % _HARD_EVERY == 0 else 0.0
            yield [_row(t, rng, hard) for t in arg_types]


SOURCES: dict[str, InputSource] = {s.name: s for s in [RandomSource()]}
