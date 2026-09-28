"""
Statistics over units (sequences, items, prompts): a paired mean and its
standard error, its p-value, Holm's correction, and a bootstrap interval.
"""

import math
import random
import statistics
from collections.abc import Callable, Sequence
from typing import TypeVar

_T = TypeVar('_T')


def paired(a: Sequence[float], b: Sequence[float] | None = None) -> tuple[float, float]:
    """The mean over units of `a - b` (or of *a*), and its standard error
    (NaN for one unit)."""
    d = a if b is None else [x - y for x, y in zip(a, b, strict=True)]
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else math.nan
    return statistics.fmean(d), se


def p_value(mean: float, se: float) -> float:
    """Two-sided, for a mean of *mean* with standard error *se* against 0, by
    the normal approximation (too small for a few units, where Student's t
    would be wider)."""
    if se == 0:
        return 1.0 if mean == 0 else 0.0
    return math.erfc(abs(mean / se) / math.sqrt(2))


def holm(ps: Sequence[float]) -> list[float]:
    """*ps* Holm-adjusted, for the family they make; a NaN stays NaN and
    counts for nothing."""
    order = sorted((i for i, p in enumerate(ps) if not math.isnan(p)), key=lambda i: ps[i])
    out, top = list(ps), 0.0
    for rank, i in enumerate(order):
        top = max(top, min(1.0, (len(order) - rank) * ps[i]))
        out[i] = top
    return out


def bootstrap(units: Sequence[_T], stat: Callable[[Sequence[_T]], float],
                 iters: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """A 95% percentile interval of *stat* over *units* resampled with
    replacement; a resample where it is NaN counts for nothing."""
    rng = random.Random(seed)
    vals = sorted(v for _ in range(iters)
                  if not math.isnan(v := stat(rng.choices(units, k=len(units)))))
    if not vals:
        return math.nan, math.nan
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]
