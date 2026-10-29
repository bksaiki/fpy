"""
Runs scored against the scheme's exact run on the same token sequences,
teacher-forced: :func:`frozen`.  Log-probabilities go through a block of
positions at a time; a reference's are kept in host memory.
"""

import enum
import math
import statistics
import sys
import time
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import torch

from . import swap

_ROWS = 512
"""Tokens per block of log-softmax and comparison."""


@torch.no_grad()
def _log_probs(model: torch.nn.Module, seg: torch.Tensor,
               dtype: torch.dtype = torch.float32) -> Iterator[torch.Tensor]:
    """Next-token log-probabilities of *seg*'s tokens but the first, in
    *dtype*, in blocks of :data:`_ROWS` positions (a quarter in FP64),
    `lm_head` applied a block at a time so the whole `[t, vocab]` is never
    held."""
    h = model.model(seg).last_hidden_state[0, :-1]
    for block in h.split(_ROWS // 4 if dtype == torch.float64 else _ROWS):
        yield torch.log_softmax(model.lm_head(block).to(dtype), -1)


@dataclass
class Divergence:
    """One run's distance from the exact run, per sequence: mean KL(R1 ||
    run) and top-1 disagreement, and, frozen, the share of its activation
    elements (and scales) that would round differently; its seconds over
    them all."""

    kl: list[float] = field(default_factory=list)
    disagree: list[float] = field(default_factory=list)
    rerounded: list[float] = field(default_factory=list)
    rescaled: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def add(self, ref: list[torch.Tensor], lps: Iterable[torch.Tensor]) -> None:
        """One sequence: *lps* against *ref*'s blocks (pinned, on the host),
        summed on the GPU and read once."""
        kl = differ = n = 0
        for r, q in zip(ref, lps, strict=True):
            r = r.to(q.device, non_blocking=True)
            kl = kl + (r.exp() * (r - q)).sum()
            differ = differ + (r.argmax(-1) != q.argmax(-1)).sum()
            n += q.shape[0]
        self.kl.append(float(kl) / n)
        self.disagree.append(float(differ) / n)

    def report(self) -> dict[str, float]:
        """Means over sequences, each with its standard error, and the
        total seconds."""
        out: dict[str, float] = {}
        for k in ('kl', 'disagree', 'rerounded', 'rescaled'):
            if v := getattr(self, k):
                out[k], out[f'{k}_se'] = _mean_se(v)
        out['seconds'] = self.seconds
        return out


class Kind(enum.StrEnum):
    """How :func:`frozen` runs a mode: frozen to the exact run, or as it runs."""

    FROZEN = 'frozen'
    FREE = 'free'


def frozen(
    model: torch.nn.Module, run: swap.Run, segs: Iterable[torch.Tensor], modes: Sequence[str],
    free: Collection[str] = (), *, seed: int = 0, progress: bool = False,
) -> dict[Kind, dict[str, Divergence]]:
    """Each of *modes* against *run*'s scheme's exact run over *segs*,
    frozen to it (`swap.Frozen`), and each of *free* as it runs:
    by :class:`Kind`, then mode; per segment.  Log-probabilities in FP64: in
    FP32, their rounding alone is a KL near 1e-9, above the smallest frozen
    runs' (~1e-12).  A random-sign run's signs are *seed*'s and the segment's
    (`Run.reseed`), frozen or free alike."""
    exact = f'{run.scheme.name}-exact'
    out: dict[Kind, dict[str, Divergence]] = {Kind.FROZEN: {}, Kind.FREE: {}}
    pinned: list[torch.Tensor] = []
    try:
        for i, seg in enumerate(segs):
            run.mode, run.record, run.frozen = exact, swap.Frozen(), None
            ref = [_pin(pinned, j, b) for j, b in enumerate(_log_probs(model, seg, dtype=torch.float64))]
            rec, run.record = run.record, None
            for mode in dict.fromkeys([*modes, *free]):
                run.mode = mode
                for kind in [k for k, of in ((Kind.FROZEN, modes), (Kind.FREE, free)) if mode in of]:
                    start = time.perf_counter()
                    run.frozen = rec.replay() if kind is Kind.FROZEN else None
                    run.reseed(seed, i)
                    d = out[kind].setdefault(mode, Divergence())
                    d.add(ref, _log_probs(model, seg, dtype=torch.float64))
                    torch.cuda.synchronize()
                    d.seconds += time.perf_counter() - start
                    if (f := run.frozen) is not None:
                        d.rerounded.append(float(f.rerounded) / f.elements)
                        if f.scales:
                            d.rescaled.append(float(f.rescaled) / f.scales)
            del rec
            torch.cuda.empty_cache()
            if progress:
                print(f'segment {i + 1}', file=sys.stderr, flush=True)
    finally:
        run.mode, run.record, run.frozen = 'fp32', None, None
    return out


def _pin(pinned: list[torch.Tensor], i: int, b: torch.Tensor) -> torch.Tensor:
    """*b* copied to the *i*-th of *pinned*'s host buffers, made or remade
    to its shape: the copies back to the GPU then run without blocking."""
    if i >= len(pinned) or pinned[i].shape != b.shape:
        buf = torch.empty(b.shape, dtype=b.dtype, pin_memory=True)
        pinned[i:i + 1] = [buf]
    return pinned[i].copy_(b)


def _mean_se(v: Sequence[float]) -> tuple[float, float]:
    """The mean over sequences of *v*, and its standard error (NaN for one)."""
    se = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else math.nan
    return statistics.fmean(v), se
