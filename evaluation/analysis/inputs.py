"""
Input sources: where a design's arguments come from, chosen by name (`SOURCES`).
"""

import enum
import random
from collections.abc import Iterator
from functools import cache
from typing import Any, Protocol

from compile import _HARD, _HARD_EVERY, _fmt, _row

from fpy2.number import Float
from fpy2.number.context import SizedFormat
from fpy2.types import ListType, Type


class InputSource(Protocol):
    name: str

    def inputs(self, arg_types: list[Type], n: int, seed: int) -> Iterator[list[Any]]:
        """*n* argument lists for a design taking *arg_types*."""
        ...

    def is_random(self, i: int, n: int) -> bool:
        """Whether the *i*-th of *n* inputs is a random draw."""
        ...


class RandomSource:
    """`compile.py`'s draws: over each format's whole range or near one, and
    every `_HARD_EVERY`-th draw heavy in hard cases (zeros, extremes,
    subnormals, specials).  The same seed gives `compile.py -r`'s inputs."""

    name: str = 'random'

    def is_random(self, i: int, n: int) -> bool:
        return True

    def inputs(self, arg_types: list[Type], n: int, seed: int) -> Iterator[list[Any]]:
        rng = random.Random(seed)
        for i in range(n):
            hard = _HARD if (i + 1) % _HARD_EVERY == 0 else 0.0
            yield [_row(t, rng, hard) for t in arg_types]


class Extreme(enum.Enum):
    """Where each operand of an adversarial draw sits in its format."""

    TINY = enum.auto()
    HUGE = enum.auto()
    MIXED = enum.auto()
    """Each operand tiny or huge, independently."""
    EDGE = enum.auto()
    """Each list all zero or all tiny, each scalar among its format's few
    smallest values: a sum of products that are all zero plus the smallest
    accumulator reaches the accumulator format's lowest digit."""
    SUBNORMAL = enum.auto()
    """Every operand subnormal (or, without subnormals, among the smallest)."""
    MINVAL = enum.auto()
    """Every operand the smallest positive value, either sign."""


class AdversarialSource:
    """Draws with every operand at one extreme of its format at once, which
    random draws almost never give: a digit position's extreme often needs
    all of them there, e.g. a truncated term's lowest digit, `e_max - k`,
    needs every product tiny.  Draws cycle through `Extreme`; tiny and huge
    are the `1 / WINDOW` of a format's positive values nearest each end."""

    name: str = 'adversarial'
    WINDOW: int = 16

    def is_random(self, i: int, n: int) -> bool:
        return False

    def inputs(self, arg_types: list[Type], n: int, seed: int) -> Iterator[list[Any]]:
        rng = random.Random(seed)
        modes = list(Extreme)
        for i in range(n):
            mode = modes[i % len(modes)]
            yield [self._arg(t, rng, mode) for t in arg_types]

    def _arg(self, t: Type, rng: random.Random, mode: Extreme) -> float | list[float]:
        fmt = _fmt(t)
        if not isinstance(t, ListType):
            return self._value(fmt, rng, mode, scalar=True)
        if mode is Extreme.EDGE and rng.random() < 0.5 and _has_zero(fmt):
            return [0.0] * t.length
        return [self._value(fmt, rng, mode) for _ in range(t.length)]

    def _value(self, fmt: SizedFormat, rng: random.Random, mode: Extreme, scalar: bool = False) -> float:
        hi = fmt.to_ordinal(fmt.maxval())
        w = max(1, (hi + 1) // self.WINDOW)
        if mode is Extreme.MINVAL:
            top = 1
        elif mode is Extreme.SUBNORMAL:
            top = _subnormals(fmt) or min(_EDGE, w)
        elif mode is Extreme.EDGE and scalar:
            top = min(_EDGE, w)
        elif mode is Extreme.HUGE or (mode is Extreme.MIXED and rng.random() < 0.5):
            top = None
        else:
            top = w
        o = rng.randint(hi - w + 1, hi) if top is None else rng.randint(1, top)
        negate = rng.random() < 0.5 and _signed(fmt)
        return float(fmt.from_ordinal(-o if negate else o))


_EDGE = 4
"""How many of its smallest values a scalar takes in `Extreme.EDGE`."""


@cache
def _subnormals(fmt: SizedFormat) -> int:
    """How many positive subnormals *fmt* has."""
    normal = getattr(fmt, 'min_normal', None)
    return 0 if normal is None else fmt.to_ordinal(normal()) - 1


@cache
def _has_zero(fmt: SizedFormat) -> bool:
    return fmt.representable_in(Float.from_float(0.0))


@cache
def _signed(fmt: SizedFormat) -> bool:
    try:
        fmt.from_ordinal(-1)
    except Exception:  # noqa: BLE001 -- an unsigned format
        return False
    return True


class MixedSource:
    """`RandomSource`'s draws with every `EVERY`-th one `AdversarialSource`'s
    instead: one input set holding both."""

    name: str = 'mixed'
    EVERY: int = 5

    def inputs(self, arg_types: list[Type], n: int, seed: int) -> Iterator[list[Any]]:
        adversarial = AdversarialSource().inputs(arg_types, n // self.EVERY, seed)
        rest = RandomSource().inputs(arg_types, n - n // self.EVERY, seed)
        for i in range(n):
            yield next(rest if self.is_random(i, n) else adversarial)

    def is_random(self, i: int, n: int) -> bool:
        return (i + 1) % self.EVERY != 0 or i // self.EVERY >= n // self.EVERY


SOURCES: dict[str, InputSource] = {
    s.name: s for s in [RandomSource(), AdversarialSource(), MixedSource()]
}
